#include <climits>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <vector>

#include "base_tensor.hpp"
#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "exception.hpp"
#include "serial_tensor.hpp"
#include "storage.hpp"
using namespace std;

namespace ts {

TensorImpl TensorImpl::slice(int index, pair<int, int> range) const {
    TensorImpl new_data = slice(index);
    new_data.shape[0] = range.second - range.first;
    new_data.data.dp += range.first * new_data.stride[0];
    return new_data;
}

TensorImpl TensorImpl::slice(int idx, int dim) const {
    CHECK_IN_RANGE(dim, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim);
    CHECK_IN_RANGE(idx, 0, size(dim),
                   "Index %d is out of bound for dimension %d with size %zu",
                   idx, dim, size(dim));

    Storage new_data(data, stride[dim] * idx);
    if (ndim == 1) {
        return TensorImpl(new_data, Size({1}), vector<int>({1}), dtype, device);
    }
    Size new_shape(shape, dim);
    vector<int> new_stride = vector<int>(shape.ndim - 1);

    int i = 0;
    for (; i < dim; ++i) {
        new_stride[i] = stride[i];
    }
    for (; i < new_stride.size(); ++i) {
        new_stride[i] = stride[i + 1];
    }
    return TensorImpl(new_data, Size(new_shape), new_stride, dtype, device);
}
TensorImpl TensorImpl::permute(vector<int> dims) const {
    CHECK_EQUAL(ndim, dims.size(), "Tensor dimension mismatch: %d vs %zu", ndim,
                dims.size());
    vector<int> new_shape = vector<int>(ndim);
    vector<int> new_stride = vector<int>(ndim);
    for (int i = 0; i < ndim; i++) {
        new_shape[i] = shape[dims[i]];
        new_stride[i] = stride[dims[i]];
    }
    Storage new_data = Storage(data, 0);
    return TensorImpl(new_data, Size(new_shape), new_stride, dtype, device);
}

TensorImpl TensorImpl::transpose(int dim1, int dim2) const {
    CHECK_IN_RANGE(dim1, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim1);
    CHECK_IN_RANGE(dim2, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim2);
    vector<int> new_shape(shape.shape);
    vector<int> new_stride(stride);
    swap(new_shape[dim1], new_shape[dim2]);
    swap(new_stride[dim1], new_stride[dim2]);

    Storage new_data = Storage(data, 0);
    return TensorImpl(new_data, Size(new_shape), new_stride, dtype, device);
}

TensorImpl TensorImpl::view(vector<int> shape) const {
    CHECK_CONTIGUOUS(*this);
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    CHECK_EQUAL(size, this->shape.data_len(), "Tensor size mismatch: %d vs %zu",
                size, this->shape.data_len());
    Storage new_data = Storage(data, 0);
    vector<int> new_stride = init_stride(shape);
    return TensorImpl(new_data, Size(shape), new_stride, dtype, device);
}

ostream &operator<<(ostream &os, TensorImpl t) {
    if (t.device == dev::cuda) {
        t = t.to(dev::cpu);
    }
    std::ios_base::fmtflags flags = os.flags();
    os.setf(std::ios::fixed);
    os.precision(4);

    os << "[";

    if (t.ndim == 1) {
        for (size_t i = 0; i < t.shape[0]; ++i) {
            os << t.get(i);
            if (i != t.shape[0] - 1) os << ", ";
        }
    } else if (t.ndim == 2) {
        for (int i = 0; i < t.size(0); i++) {
            os << t.slice(i);
            if (i != t.shape[0] - 1) os << ", " << endl;
        }
    } else {
        for (int i = 0; i < t.size(0); i++) {
            os << t.slice(i);
            if (i != t.shape[0] - 1) os << ", " << endl << endl;
        }
    }
    os << "]";

    return os;
}
TensorImpl TensorImpl::operator()(int index) const { return slice(index); }
TensorImpl TensorImpl::operator()(int index, pair<int, int> range) const {
    TensorImpl new_data = slice(index);
    new_data.shape[0] = range.second - range.first;
    new_data.data.dp += range.first * new_data.stride[0];
    return new_data;
}
data_t &TensorImpl::locate(vector<size_t> inds) {
    if (inds.size() == 0) {
        return data[0];
    }
    CHECK_EQUAL(ndim, inds.size(), "Invalid %zuD indices for %dD tensor",
                inds.size(), ndim);
    size_t offset = 0, i = 0;
    for (auto idx : inds) {
        CHECK_IN_RANGE(
            idx, 0, shape[i],
            "Index %zu is out of bound for dimension %zu with size %zu", idx, i,
            size(i));
        offset += idx * stride[i++];
    }
    return data[offset];
}
data_t TensorImpl::locate(vector<size_t> inds) const {
    if (inds.size() == 0) {
        return data[0];
    }
    CHECK_EQUAL(ndim, inds.size(), "Invalid %zuD indices for %dD tensor",
                inds.size(), ndim);
    size_t offset = 0, i = 0;
    for (auto idx : inds) {
        CHECK_IN_RANGE(
            idx, 0, shape[i],
            "Index %zu is out of bound for dimension %zu with size %zu", idx, i,
            size(i));
        offset += idx * stride[i++];
    }
    return data[offset];
}

TensorImpl TensorImpl::operator[](size_t index) const { return slice(index); }

TensorImpl &TensorImpl::operator=(BaseTensor<> bt) {
    vector<data_t> nt(bt.shape.data_len());
    int i = 0;
    for (auto a : bt.get_data()) {
        nt[i++] = a;
    }
    TensorImpl t = TensorImpl(nt, bt.shape.shape, this->dtype);
    CHECK_SAME_SHAPE(*this, t);
    if (this->device == dev::cpu) {
        for (int i = 0; i < shape.data_len(); i++) {
            this->get(i) = t.data[i];
        }
    } else {
        for (int i = 0; i < shape.data_len(); i++) {
            size_t idx = get_data_idx(i, *this);

            c_cudaMemcpy(this->data.dp + idx, &t.data[i], sizeof(data_t),
                         c_cudaMemcpyHostToDevice);
        }
    }

    return *this;
}

TensorImpl &TensorImpl::operator=(double val) {
    for (int i = 0; i < this->size(); i++) {
        size_t idx = get_data_idx(i, *this);
        if (this->device == dev::cpu) {
            this->data.dp[idx] = val;
        } else {
            data_t tmp;
            tmp.set_dtype(this->dtype);
            tmp = val;
            c_cudaMemcpy(this->data.dp + idx, &tmp, sizeof(data_t),
                         c_cudaMemcpyHostToDevice);
        }
    }
    return *this;
}

TensorImpl &TensorImpl::operator=(const TensorImpl &other) {
    this->data = other.data;
    this->ndim = other.ndim;
    this->shape = Size(other.shape);
    this->stride = vector<int>(other.stride);
    this->origin_stride = vector<int>(other.origin_stride);
    this->offset = other.offset;
    this->dtype = other.dtype;
    this->device = other.device;
    return *this;
}

size_t TensorImpl::get_dim() const { return this->ndim; }
size_t TensorImpl::size(int i) const { return this->shape.size(i); }
size_t TensorImpl::size() const { return this->shape.data_len(); }

void *TensorImpl::data_ptr() const { return (void *)data.bp.get(); }
string TensorImpl::type() const {
    switch (this->dtype) {
        case dt::int8:
            return "int8";
        case dt::float32:
            return "float32";
        case dt::int32:
            return "int32";
        case dt::float64:
            return "float64";
        case dt::bool8:
            return "bool8";
    }
    return "unknown";
}

int TensorImpl::get_size(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

vector<data_t> TensorImpl::get_serial_data() const {
    vector<data_t> new_data(this->shape.data_len());
    if (this->device == dev::cuda) {
        void *tmp;
        c_cudaMalloc(&tmp, this->shape.data_len() * sizeof(data_t));
        get_serial_tensor_kernel(tmp, *this);
        c_cudaMemcpy(new_data.data(), tmp,
                     this->shape.data_len() * sizeof(data_t),
                     c_cudaMemcpyDeviceToHost);
        c_cudaFree(tmp);

    } else {
        for (int i = 0; i < this->shape.data_len(); i++) {
            size_t offset = get_data_idx(i, *this);
            new_data[i] = this->data[offset];
            offset += this->stride[i];
        }
    }
    return new_data;
}

vector<int> init_stride(vector<int> shape) {
    vector<int> stride(shape.size());
    int st = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = st;
        st *= shape[i];
    }
    return stride;
}

TensorImpl tensor(BaseTensor<> bt, dt dtype) {
    vector<data_t> nt(bt.shape.data_len());
    int i = 0;
    for (auto a : bt.get_data()) {
        nt[i] = a;
        nt[i++].set_dtype(dtype);
    }
    cerr << dtype << endl;
    return TensorImpl(nt, bt.shape.shape, dtype);
}

TensorImpl rand(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        switch (dtype) {
            case dt::float32:
                data[i] = (float)random() / (float)RAND_MAX;
                break;
            case dt::int32:
                data[i] = (int)random() % INT_MAX;
                break;
            case dt::int8:
                data[i] = ((int)random()) % 256;
                break;
            case dt::float64:
                data[i] = (double)random() / RAND_MAX;
                break;

            case dt::bool8:
                data[i] = (double)random() / RAND_MAX > 0.5;
                break;
            default:
                throw std::invalid_argument("Invalid dtype");
        }
        data[i].set_dtype(dtype);
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return TensorImpl(data, sz.shape, dtype);
}
TensorImpl zeros(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i] = 0;
        data[i].set_dtype(dtype);
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return TensorImpl(data, sz.shape, dtype);
}
TensorImpl ones(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i] = 1;
        data[i].set_dtype(dtype);
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return TensorImpl(data, sz.shape, dtype);
}
TensorImpl full(Size sz, data_t val, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i] = val;
        data[i].set_dtype(dtype);
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return TensorImpl(data, sz.shape, dtype);
}
TensorImpl eye(Size sz, dt dtype) {
    CHECK_IN_RANGE(sz.ndim, 0, 3,
                   "Eye dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   3, sz.ndim);

    if (sz.ndim == 0) {
        sz = Size({1, 1});
    } else if (sz.ndim == 1) {
        sz = Size({sz.shape[0], sz.shape[0]});
    }
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i] = 0;
        data[i].set_dtype(dtype);
    }
    for (int i = 0; i < sz.data_len(); i += sz.shape[1] + 1) {
        data[i] = 1;
        data[i].set_dtype(dtype);
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return TensorImpl(data, sz.shape, dtype);
}
size_t compute_offset(const vector<size_t> &indices, const Size &shape) {
    // Validate input: indices size and shape dimensions should match
    if (indices.size() != shape.ndim) {
        throw std::invalid_argument(
            "Indices size does not match tensor dimensions");
    }

    size_t offset = 0;
    size_t stride = 1;

    // Calculate the linear offset
    for (int i = indices.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range for tensor dimension");
        }
        offset += indices[i] * stride;
        stride *= shape[i];
    }

    return offset;
}
TensorImpl cat(vector<TensorImpl> tensors, int dim) {
    if (tensors.size() == 0) {
        return TensorImpl();
    }
    TensorImpl first = tensors[0];
    size_t total_size = 0;
    vector<int> step_sizes(tensors.size());
    int sum_step_sizes = 0;
    for (int i = 0; i < tensors.size(); i++) {
        step_sizes[i] = tensors[i].stride[dim] * tensors[i].size(dim);
        sum_step_sizes += step_sizes[i];
        CHECK_EQUAL(first.ndim, tensors[i].ndim,
                    "Tensor dimension mismatch: %d vs %d", first.ndim,
                    tensors[i].ndim);
        for (int j = 0; j < first.ndim; j++) {
            if (j == dim) continue;
            CHECK_EQUAL(first.shape[j], tensors[i].shape[j],
                        "Tensor shape mismatch: %d vs %d", first.shape[j],
                        tensors[i].shape[j]);
        }
        total_size += tensors[i].shape.data_len();
    }

    vector<data_t> data(total_size);
    size_t t_offset = 0;
    for (int i = 0; i < tensors.size(); i++) {
        size_t target_offset = 0;
        size_t self_offset = 0;
        vector<data_t> tensor_data = tensors[i].get_serial_data();
        while (target_offset < total_size) {
            copy(tensor_data.data() + self_offset,
                 tensor_data.data() + step_sizes[i] + self_offset,
                 data.begin() + target_offset + t_offset);
            self_offset += step_sizes[i];
            target_offset += sum_step_sizes;
        }
        t_offset += step_sizes[i];
    }
    vector<int> new_shape = first.shape.shape;
    new_shape[dim] = sum_step_sizes / first.stride[dim];
    return TensorImpl(data, new_shape);
}
// TODO: below
vector<int> vec_mul(vector<int> v1, vector<int> v2) {
    CHECK_EQUAL(v1.size(), v2.size(), "Vector size mismatch: %zu vs %zu",
                v1.size(), v2.size());
    vector<int> ret = vector<int>(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        ret[i] = v1[i] * v2[i];
    }
    return ret;
}
TensorImpl tile(const TensorImpl &t, vector<int> reps) {
    CHECK_EQUAL(t.ndim, reps.size(), "Tensor dimension mismatch: %d vs %zu",
                t.ndim, reps.size());
    TensorImpl new_t = TensorImpl(t);
    for (int i = t.ndim - 1; i >= 0; i--) {
        TensorImpl tmp = TensorImpl(new_t);
        for (int j = 0; j < reps[i] - 1; j++) {
            tmp = cat({tmp, new_t}, i);
        }
        new_t = tmp;
    }
    return new_t;
}
TensorImpl transpose(const TensorImpl &t, int dim1, int dim2) {
    CHECK_IN_RANGE(dim1, 0, t.ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   t.ndim, dim1);
    CHECK_IN_RANGE(dim2, 0, t.ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   t.ndim, dim2);
    vector<int> new_shape = t.shape.shape;
    vector<int> new_stride = t.stride;
    swap(new_shape[dim1], new_shape[dim2]);
    swap(new_stride[dim1], new_stride[dim2]);
    Storage new_data = Storage(t.data, 0);
    return TensorImpl(new_data, Size(new_shape), new_stride, t.dtype, t.device);
}
TensorImpl permute(const TensorImpl &t, vector<int> dims) {
    CHECK_EQUAL(t.ndim, dims.size(), "Tensor dimension mismatch: %d vs %zu",
                t.ndim, dims.size());
    vector<int> new_shape = vector<int>(t.ndim);
    vector<int> new_stride = vector<int>(t.ndim);
    for (int i = 0; i < t.ndim; i++) {
        new_shape[i] = t.shape[dims[i]];
        new_stride[i] = t.stride[dims[i]];
    }
    Storage new_data = Storage(t.data, 0);
    return TensorImpl(new_data, Size(new_shape), new_stride, t.dtype, t.device);
}

TensorImpl view(const TensorImpl &t, vector<int> shape) {
    CHECK_CONTIGUOUS(t);
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    CHECK_EQUAL(size, t.shape.data_len(), "Tensor size mismatch: %d vs %zu",
                size, t.shape.data_len());
    Storage new_data = Storage(t.data, 0);
    vector<int> new_stride = init_stride(shape);
    return TensorImpl(new_data, shape, new_stride, t.dtype, t.device);
}

// save and load
void save(const TensorImpl &t, string filename) {
    vector<data_t> tmp = t.get_serial_data();
    ofstream file(save_path + filename, ios::binary);
    if (file.is_open()) {
        char *offset = 0;
        // Dtype
        file.write((char *)&t.dtype, sizeof(t.dtype));
        // Device
        file.write((char *)&t.device, sizeof(t.device));
        // Ndim
        file.write((char *)&t.ndim, sizeof(int));
        // Then shape data
        file.write((char *)t.shape.shape.data(), t.ndim * sizeof(int));
        // Then data
        file.write((char *)tmp.data(), tmp.size() * sizeof(data_t));
        file.close();
    } else {
        throw runtime_error("Unable to open file");
    }
    cout << "Saved Successfully! [" << filename << "]" << endl;
}

TensorImpl load(string filename) {
    ifstream file(save_path + filename, ios::binary);
    if (file.is_open()) {
        int ndim;
        int data_size = 1;
        dt dtype;
        dev device;

        file.read((char *)&dtype, sizeof(dtype));
        file.read((char *)&device, sizeof(device));
        file.read((char *)&ndim, sizeof(ndim));

        vector<int> shape(ndim);
        file.read((char *)shape.data(), sizeof(int) * ndim);
        for (int i = 0; i < ndim; i++) {
            data_size *= shape[i];
        }
        vector<data_t> data(data_size);
        file.read((char *)data.data(), data_size * sizeof(data_t));

        file.close();
        cout << "Load Successfully! [" << filename << "]" << endl;
        return TensorImpl(data, shape, dtype, device);
    } else {
        throw runtime_error("Unable to open file");
    }
}
vector<int> get_dim_idx(size_t index, vector<int> shape,
                        vector<int> origin_stride) {
    vector<int> indices(shape.size());
    for (int i = 0; i < shape.size(); i++) {
        size_t tmp = index / origin_stride[i];
        indices[i] = tmp;
        index -= tmp * origin_stride[i];
    }
    return indices;
}

TensorImpl TensorImpl::squeeze() const {
    vector<int> new_shape;
    for (int i = 0; i < this->ndim; i++) {
        if (this->shape[i] != 1) {
            new_shape.push_back(this->shape[i]);
        }
    }
    vector<data_t> new_data = this->get_serial_data();
    return TensorImpl(new_data, new_shape, this->dtype, this->device);
}

TensorImpl TensorImpl::unsqueeze(int dim) const {
    CHECK_IN_RANGE(dim, 0, this->ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   this->ndim, dim);
    vector<int> new_shape = this->shape.shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    vector<data_t> new_data = this->get_serial_data();
    return TensorImpl(new_data, new_shape, this->dtype, this->device);
}
size_t get_data_idx(size_t index, TensorImpl t) {
    size_t offset = 0;
    size_t tmp = 0;
    for (int i = 0; i < t.shape.shape.size(); i++) {
        tmp = index / t.origin_stride[i];
        offset += tmp * t.stride[i];
        index -= tmp * t.origin_stride[i];
    }

    return offset;
}

void TensorImpl::info(string name) const {
    int width = 18;
    cerr << "--------------------------" << endl;
    cerr << "Tensor: " << setw(width) << left << name << "|" << endl;
    cerr << "Dim:    " << setw(width) << left << ndim << "|" << endl;
    cerr << "Dtype:  " << setw(width) << left << dtype << "|" << endl;
    cerr << "Shape:  " << setw(width) << left << shape << "|" << endl;
    cerr << "Device: " << setw(width) << left << device << "|" << endl;
    cerr << "PTR:    " << setw(width) << left << data_ptr() << "|" << endl;
    cerr << "--------------------------" << endl;
}

bool TensorImpl::is_contiguous() const {
    if (this->ndim == 1) {
        return true;
    }
    for (int i = 0; i < this->ndim - 1; i++) {
        if (this->stride[i] != this->shape[i + 1] * this->stride[i + 1]) {
            return false;
        }
    }
    return true;
}

data_t &TensorImpl::get(size_t index) {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset = get_data_idx(index, *this);
    return data[offset];
}

data_t TensorImpl::get(size_t index) const {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset = get_data_idx(index, *this);
    return data[offset];
}

data_t TensorImpl::at(size_t index) const {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset = get_data_idx(index, *this);
    if (this->device == dev::cuda) {
        data_t tmp;
        c_cudaMemcpy(&tmp, this->data.dp + offset, sizeof(data_t),
                     c_cudaMemcpyDeviceToHost);
        return tmp;
    } else {
        return this->data[offset];
    }
    return data[offset];
}
void TensorImpl::set_at(size_t index, data_t val) {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset = get_data_idx(index, *this);
    if (this->device == dev::cuda) {
        val.set_dtype(this->dtype);
        c_cudaMemcpy(this->data.dp + offset, &val, sizeof(data_t),
                     c_cudaMemcpyHostToDevice);
    } else {
        this->data[offset] = val;
    }
}
}  // namespace ts
