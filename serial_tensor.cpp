#include "serial_tensor.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "exception.hpp"
#include "storage.hpp"

using namespace std;

namespace ts {

Tensor::Tensor() : Tensor(0) {}
Tensor::Tensor(int i_dim) : data(i_dim) {
    this->ndim = i_dim;
    this->shape = Size(i_dim);
    this->stride.reserve(i_dim);
    this->offset = 0;
}

Tensor::Tensor(const vector<data_t> &i_data, const vector<int> &i_shape,
               dt dtype) {
    if (i_shape.size() == 0) {
        this->ndim = 1;
        this->shape = Size(i_data.size());
    } else {
        this->ndim = i_shape.size();
        this->shape = Size(i_shape);
    }

    this->data = Storage(i_data.data(), this->shape.size());
    this->dtype = dtype;
    this->offset = 0;
    init_stride();
}
void Tensor::init_stride() {
    stride.resize(ndim);

    int st = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        stride[i] = st;
        st *= shape[i];
    }
}

Tensor::Tensor(const Storage &i_data, const Size &i_shape,
               const vector<int> i_stride, dt dtype)
    : data(i_data), stride(i_stride), shape(i_shape) {
    this->ndim = i_shape.ndim;
    this->dtype = dtype;
}

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
    Tensor nt = Tensor(new_data, Size(new_shape), new_stride, dtype);

    return nt;
}

ostream &operator<<(ostream &os, Tensor t) {
    std::ios_base::fmtflags flags = os.flags();
    os.setf(std::ios::fixed);
    os.precision(4);

    os << "[";

    if (t.ndim == 1) {
        for (size_t i = 0; i < t.shape[0]; ++i) {
            os << t[{i}];
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

// data_t& Tensor::operator[](initializer_list<size_t> inds) {
//     CHECK_EQUAL(ndim, inds.size(),
//         "Invalid %dD indices for %dD tensor", inds.size(), ndim);
//     size_t offset = 0, i = 0;
//     for (auto idx : inds) {
//         CHECK_IN_RANGE(idx, 0, shape[i],
//             "Index %zu is out of bound for dimension %zu with size %zu",
//             idx, i, size(i));
//         offset += idx * stride[i++];
//     }
//     return data[offset];
// }
// data_t Tensor::operator[](initializer_list<size_t> inds) const {
//     CHECK_EQUAL(ndim, inds.size(),
//         "Invalid %dD indices for %dD tensor", inds.size(), ndim);
//     size_t offset = 0, i = 0;
//     for (auto idx : inds) {
//         CHECK_IN_RANGE(idx, 0, shape[i],
//             "Index %zu is out of bound for dimension %zu with size %zu",
//             idx, i, size(i));
//         offset += idx * stride[i++];
//     }
//     return data[offset];
// }
data_t &Tensor::operator[](vector<size_t> inds) {
    CHECK_EQUAL(ndim, inds.size(), "Invalid %dD indices for %dD tensor",
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
    CHECK_EQUAL(ndim, inds.size(), "Invalid %dD indices for %dD tensor",
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

size_t Tensor::get_dim() const { return this->ndim; }
size_t Tensor::size(int i) const { return this->shape.size(i); }

void *Tensor::data_ptr() { return (void *)data.bp.get(); }

int Tensor::get_size(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

vector<data_t> Tensor::get_data() {
    vector<data_t> data(this->shape.size());
    for (int i = 0; i < this->shape.size(); i++) {
        data[i] = this->data[i];
    }
    return data;
}

Tensor tensor(BaseTensor<> bt) { return Tensor(bt.get_data(), bt.shape.shape); }

Tensor rand(Size sz) {
    vector<data_t> data(sz.size());
    for (int i = 0; i < sz.size(); i++) {
        data[i] = (data_t)random() / RAND_MAX;
    }
    Storage st(data.data(), sz.size());
    return Tensor(data, sz.shape);
}

Tensor zeros(Size sz) {
    vector<data_t> data(sz.size());
    for (int i = 0; i < sz.size(); i++) {
        data[i] = 0;
    }
    Storage st(data.data(), sz.size());
    return Tensor(data, sz.shape);
}

Tensor ones(Size sz) {
    vector<data_t> data(sz.size());
    for (int i = 0; i < sz.size(); i++) {
        data[i] = 1;
    }
    Storage st(data.data(), sz.size());
    return Tensor(data, sz.shape);
}

Tensor full(Size sz, data_t val) {
    vector<data_t> data(sz.size());
    for (int i = 0; i < sz.size(); i++) {
        data[i] = val;
    }
    Storage st(data.data(), sz.size());
    return Tensor(data, sz.shape);
}

Tensor eye(Size sz) {
    CHECK_IN_RANGE(sz.ndim, 0, 3,
                   "Eye dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   3, sz.ndim);

    if (sz.ndim == 0) {
        sz = Size({1, 1});
    } else if (sz.ndim == 1) {
        sz = Size({sz.shape[0], sz.shape[0]});
    }
    vector<data_t> data(sz.size());
    for (int i = 0; i < sz.size(); i++) {
        data[i] = 0;
    }
    for (int i = 0; i < sz.size(); i += sz.shape[1] + 1) {
        data[i] = 1;
    }
    Storage st(data.data(), sz.size());
    return Tensor(data, sz.shape);
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


// TODO: below
Tensor cat(vector<Tensor> tensors, int dim) {
    if (tensors.size() == 0) {
        return Tensor();
    }
    Tensor first = tensors[0];
    size_t total_size = 0;
    for (int i = 0; i < tensors.size(); i++) {
        CHECK_EQUAL(first.ndim, tensors[i].ndim,
                    "Tensor dimension mismatch: %d vs %d", first.ndim,
                    tensors[i].ndim);
        for (int j = 0; j < first.ndim; j++) {
            if (j == dim) continue;
            CHECK_EQUAL(first.shape[j], tensors[i].shape[j],
                        "Tensor shape mismatch: %d vs %d", first.shape[j],
                        tensors[i].shape[j]);
        }
        total_size += tensors[i].shape.size();
    }
    vector<data_t> data(total_size);
    size_t offset = 0;
    for (int i = 0; i < tensors.size(); i++) {
        vector<data_t> t_data = tensors[i].get_data();
        size_t end = offset + tensors[i].shape.size();
        copy(t_data.begin(), t_data.end(), data.begin() + offset);
        offset = end;
    }
    vector<int> new_shape = first.shape.shape;
    new_shape[dim] *= tensors.size();
    return Tensor(data, new_shape);
}

void repeat(vector<data_t> &v, int repeats) {
    size_t total_size = v.size() * repeats;
    vector<data_t> new_v(total_size);
    for (int i = 0; i < repeats; i++) {
        copy(v.begin(), v.end(), new_v.begin() + i * v.size());
    }
    v = new_v;
    cout << "repeat" << endl;
}

vector<int> vec_mul(vector<int> v1, vector<int> v2) {
    CHECK_EQUAL(v1.size(), v2.size(), "Vector size mismatch: %d vs %d",
                v1.size(), v2.size());
    vector<int> ret = vector<int>(v1.size());
    for (int i = 0; i < v1.size(); i++) {
        ret[i] = v1[i] * v2[i];
    }
    return ret;
}

Tensor tile(Tensor t, vector<int> reps) {
    CHECK_EQUAL(t.ndim, reps.size(),
                "Tensor dimension mismatch: %d vs %d", t.ndim, reps.size());
    vector<data_t> data = t.get_data();
    Tensor new_t = Tensor(t);
    for (int i = 0; i < t.ndim; i++) {
        for (int j = 0; j < reps[i]; j++) {
            repeat(data, t.shape[i]);
        }
    }
    Tensor test= Tensor(data, vec_mul(t.shape.shape, reps));
    return test;
}

}  // namespace ts
