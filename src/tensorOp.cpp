#include <climits>
#include <cstddef>
#include <fstream>
#include <vector>

#include "base_tensor.hpp"
#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "exception"
#include "exception.hpp"
#include "serial_tensor.hpp"

using namespace std;

namespace ts {
Tensor Tensor::slice(int idx, int dim) {
    CHECK_IN_RANGE(dim, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim);
    CHECK_IN_RANGE(idx, 0, size(dim),
                   "Index %d is out of bound for dimension %d with size %zu",
                   idx, dim, size(dim));

    Storage new_data(data, stride[dim] * idx);

    Size new_shape(shape, dim);
    vector<int> new_stride = vector<int>(shape.ndim - 1);

    int i = 0;
    for (; i < dim; ++i) {
        new_stride[i] = stride[i];
    }
    for (; i < new_stride.size(); ++i) {
        new_stride[i] = stride[i + 1];
    }
    Tensor nt = Tensor(new_data, Size(new_shape), new_stride, dtype, device);
    return nt;
}

Tensor Tensor::permute(vector<int> dims) {
    CHECK_EQUAL(ndim, dims.size(), "Tensor dimension mismatch: %d vs %zu", ndim,
                dims.size());
    vector<int> new_shape = vector<int>(ndim);
    vector<int> new_stride = vector<int>(ndim);
    for (int i = 0; i < ndim; i++) {
        new_shape[i] = shape[dims[i]];
        new_stride[i] = stride[dims[i]];
    }
    Storage new_data = Storage(data, 0);
    return Tensor(new_data, Size(new_shape), new_stride, dtype, device);
}

Tensor Tensor::transpose(int dim1, int dim2) {
    CHECK_IN_RANGE(dim1, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim1);
    CHECK_IN_RANGE(dim2, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim2);
    vector<int> new_shape = shape.shape;
    vector<int> new_stride = stride;
    swap(new_shape[dim1], new_shape[dim2]);
    swap(new_stride[dim1], new_stride[dim2]);
    Storage new_data = Storage(data, 0);
    return Tensor(new_data, Size(new_shape), new_stride, dtype, device);
}

Tensor Tensor::view(vector<int> shape) {
    CHECK_TRUE(is_contiguous(), "View only support contiguous Tensor");
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    CHECK_EQUAL(size, this->shape.data_len(), "Tensor size mismatch: %d vs %zu",
                size, this->shape.data_len());
    Storage new_data = Storage(data, 0);
    vector<int> new_stride = init_stride(shape);
    return Tensor(new_data, Size(shape), new_stride, dtype, device);
}

ostream &operator<<(ostream &os, Tensor t) {
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
Tensor Tensor::operator()(int index) { return slice(index); }
Tensor Tensor::operator()(int index, pair<int, int> range) {
    Tensor new_data = slice(index);
    new_data.shape[0] = range.second - range.first;
    new_data.data.dp += range.first * new_data.stride[0];
    return new_data;
}
data_t &Tensor::operator[](vector<size_t> inds) {
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
data_t Tensor::operator[](vector<size_t> inds) const {
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

// Tensor &Tensor::operator[](size_t idx) {
//     CHECK_IN_RANGE(idx, 0, this->size(), "Invalid index %zu for Size %zu", idx,
//                    this->size());
//     size_t offset =
//         get_data_idx(idx, this->shape.shape, this->stride, this->origin_stride);
//     return data[offset];
// }

// data_t Tensor::operator[](size_t idx) const {
//     CHECK_IN_RANGE(idx, 0, this->size(), "Invalid index %zu for Size %zu", idx,
//                    this->size());
//     size_t offset =
//         get_data_idx(idx, this->shape.shape, this->stride, this->origin_stride);
//     return data[offset];
// }

    Tensor Tensor::operator[](size_t index) {
        return slice(index);
    }

Tensor &Tensor::operator=(BaseTensor<> bt) {
    vector<data_t> nt(bt.shape.data_len());
    int i = 0;
    for (auto a : bt.get_data()) {
        nt[i++] = a;
    }
    Tensor t = Tensor(nt, bt.shape.shape);

    CHECK_EQUAL(ndim, t.ndim, "Tensor dimension mismatch: %d vs %d", ndim,
                t.ndim);
    for (int i = 0; i < ndim; i++) {
        CHECK_EQUAL(shape[i], t.shape[i], "Tensor shape mismatch: %d vs %d",
                    shape[i], t.shape[i]);
    }
    for (int i = 0; i < shape.data_len(); i++) {
        data[i] = t.data[i];
    }
    return *this;
}

Tensor &Tensor::operator=(int val) {
    for (int i = 0; i < this->size(); i++) {
        size_t idx = get_data_idx(i, this->shape.shape, this->stride,
                                  this->origin_stride);
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

Tensor &Tensor::operator=(double val) {
    for (int i = 0; i < this->size(); i++) {
        size_t idx = get_data_idx(i, this->shape.shape, this->stride,
                                  this->origin_stride);
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

Tensor &Tensor::operator=(bool val) {
    for (int i = 0; i < this->size(); i++) {
        size_t idx = get_data_idx(i, this->shape.shape, this->stride,
                                  this->origin_stride);
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

size_t Tensor::get_dim() const { return this->ndim; }
size_t Tensor::size(int i) const { return this->shape.size(i); }
size_t Tensor::size() const { return this->shape.data_len(); }

void *Tensor::data_ptr() { return (void *)data.bp.get(); }
string Tensor::type() const {
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

int Tensor::get_size(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}
// vector<data_t> Tensor::get_serial_data() const {
//     vector<data_t> data(this->shape.data_len());
//     for (int i = 0; i < this->shape.data_len(); i++) {
//         size_t offset = get_data_idx(
//             i, this->shape.shape, this->stride, this->origin_stride);
//         data[i] = this->data[offset];
//         offset += this->stride[i];
//     }
//     return data;
// }

vector<data_t> Tensor::get_serial_data() const {
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
            size_t offset = get_data_idx(i, this->shape.shape, this->stride,
                                         this->origin_stride);
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

Tensor tensor(BaseTensor<> bt, dt dtype) {
    vector<data_t> nt(bt.shape.data_len());
    int i = 0;
    for (auto a : bt.get_data()) {
        nt[i].set_dtype(dtype);
        nt[i++] = a;
    }
    return Tensor(nt, bt.shape.shape, dtype);
}
Tensor rand(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i].set_dtype(dtype);
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
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return Tensor(data, sz.shape, dtype);
}
Tensor zeros(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i].set_dtype(dtype);
        data[i] = 0;
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return Tensor(data, sz.shape, dtype);
}
Tensor ones(Size sz, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i].set_dtype(dtype);
        data[i] = 1;
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return Tensor(data, sz.shape, dtype);
}
Tensor full(Size sz, data_t val, dt dtype) {
    vector<data_t> data(sz.data_len());
    for (int i = 0; i < sz.data_len(); i++) {
        data[i].set_dtype(dtype);
        data[i] = val;
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return Tensor(data, sz.shape, dtype);
}
Tensor eye(Size sz, dt dtype) {
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
        data[i].set_dtype(dtype);
        data[i] = 0;
    }
    for (int i = 0; i < sz.data_len(); i += sz.shape[1] + 1) {
        data[i].set_dtype(dtype);
        data[i] = 1;
    }
    Storage st(data.data(), sz.data_len(), dtype);
    return Tensor(data, sz.shape, dtype);
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
Tensor cat(vector<Tensor> tensors, int dim) {
    if (tensors.size() == 0) {
        return Tensor();
    }
    Tensor first = tensors[0];
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
    return Tensor(data, new_shape);
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
Tensor tile(Tensor t, vector<int> reps) {
    CHECK_EQUAL(t.ndim, reps.size(), "Tensor dimension mismatch: %d vs %zu",
                t.ndim, reps.size());
    Tensor new_t = Tensor(t);
    for (int i = t.ndim; i >= 0; i--) {
        Tensor tmp = Tensor(new_t);
        for (int j = 0; j < reps[i] - 1; j++) {
            tmp = cat({tmp, new_t}, i);
        }
        new_t = tmp;
    }
    return new_t;
}
Tensor transpose(Tensor t, int dim1, int dim2) {
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
    return Tensor(new_data, Size(new_shape), new_stride, t.dtype, t.device);
}
Tensor permute(Tensor t, vector<int> dims) {
    CHECK_EQUAL(t.ndim, dims.size(), "Tensor dimension mismatch: %d vs %zu",
                t.ndim, dims.size());
    vector<int> new_shape = vector<int>(t.ndim);
    vector<int> new_stride = vector<int>(t.ndim);
    for (int i = 0; i < t.ndim; i++) {
        new_shape[i] = t.shape[dims[i]];
        new_stride[i] = t.stride[dims[i]];
    }
    Storage new_data = Storage(t.data, 0);
    return Tensor(new_data, Size(new_shape), new_stride, t.dtype, t.device);
}

Tensor view(Tensor t, vector<int> shape) {
    CHECK_TRUE(t.is_contiguous(), "View only support contiguous Tensor");
    int size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    CHECK_EQUAL(size, t.shape.data_len(), "Tensor size mismatch: %d vs %zu",
                size, t.shape.data_len());
    Storage new_data = Storage(t.data, 0);
    vector<int> new_stride = init_stride(shape);
    return Tensor(new_data, shape, new_stride, t.dtype, t.device);
}

// save and load
void save(Tensor t, string filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << t.shape.data_len() << endl;
        for (size_t i = 0; i < t.shape.data_len(); ++i) {
            file << t.shape[i] << endl;
            for (size_t j = 0; j < t.shape[i]; ++j) {
                file << t.data[i * t.shape[i] + j] << endl;
            }
        }
        file.close();
    } else {
        throw runtime_error("Unable to open file");
    }
}

Tensor load(string filename) {
    ifstream file(filename);
    if (file.is_open()) {
        int size;
        file >> size;
        vector<int> shape(size);
        for (int i = 0; i < size; i++) {
            file >> shape[i];
        }
        vector<data_t> data;
        int temp;
        while (file >> temp)  // read data
        {
            data.push_back(temp);
        }
        file.close();
        return Tensor(data, shape);
    } else {
        throw runtime_error("Unable to open file");
    }
}

size_t get_data_idx(size_t index, vector<int> shape_v, vector<int> stride,
                    vector<int> origin_stride) {
    size_t offset = 0;
    size_t tmp = 0;
    for (int i = 0; i < shape_v.size(); i++) {
        tmp = index / origin_stride[i];
        offset += tmp * stride[i];
        index -= tmp * origin_stride[i];
    }
    return offset;
}


bool Tensor::is_contiguous() {
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

data_t &Tensor::get(size_t index) {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset =
        get_data_idx(index, this->shape.shape, this->stride, this->origin_stride);
    return data[offset];
}

data_t Tensor::get(size_t index) const {
    CHECK_IN_RANGE(index, 0, this->size(), "Invalid index %zu for Size %zu",
                   index, this->size());
    size_t offset =
        get_data_idx(index, this->shape.shape, this->stride, this->origin_stride);
    return data[offset];
}
}  // namespace ts
