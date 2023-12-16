#include "serial_tensor.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
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
    this->ndim = i_shape.dim;
    this->dtype = dtype;
}

// data_t &Tensor::operator[](int index) { return data[offset + index]; }

ostream &operator<<(ostream &os, Tensor t) {
    std::ios_base::fmtflags flags = os.flags();
    os.setf(std::ios::fixed);
    os.precision(4);

    os << "[";
    if (t.ndim == 0) {
        os << t.data[0];
    } else if (t.ndim == 1) {
        for (int i = 0; i < t.shape[0]; ++i) {
            os << t.data[i];
            if (i != t.shape[0] - 1) os << ", ";
        }
    } else if (t.ndim == 2) {
        for (int i = 0; i < t.shape[0]; ++i) {
            os << "[";
            for (int j = 0; j < t.shape[1]; ++j) {
                os << t.data[i * t.shape[1] + j];
                if (j != t.shape[1] - 1) os << ", ";
            }
            os << "]";
            if (i != t.shape[0] - 1) {
                os << ", ";
                os << endl;
            } else {
            }
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

size_t Tensor::get_dim() const { return this->ndim; }
size_t Tensor::size(int i) const { return this->shape.size(i); }

Tensor Tensor::slice(int idx, int dim) {
    CHECK_IN_RANGE(dim, 0, ndim,
                   "Dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   ndim, dim);
    CHECK_IN_RANGE(idx, 0, size(dim),
                   "Index %d is out of bound for dimension %d with size %zu",
                   idx, dim, size(dim));
    Storage new_data(data, stride[dim] * idx);
    vector<int> new_shape(this->shape.shape);
    vector<int> new_stride(this->stride);

    new_shape.pop_back();
    new_stride.pop_back();

    // Tensor nt = Tensor(new_data, shape, dtype);

    int i = 0;
    for (; i < dim; ++i) {
        new_shape[i] = shape[i];
        new_stride[i] = stride[i];
    }
    for (; i < this->ndim - 1; ++i) {
        new_shape[i] = shape[i + 1];
        new_stride[i] = stride[i + 1];
    }
    Tensor nt = Tensor(new_data, Size(new_shape), new_stride, dtype);

    return nt;
}

Tensor Tensor::operator()(int index) { return slice(index); }
Tensor Tensor::operator()(int index, pair<int, int> range) {
    CHECK_IN_RANGE(range.first, 0, size(index),
                   "Index %d is out of bound for dimension %d with size %zu",
                   range.first, index, size(index));
    CHECK_IN_RANGE(range.second, 0, size(index),
                   "Index %d is out of bound for dimension %d with size %zu",
                   range.second, index, size(index));

    Storage new_data(data, stride[index] * range.first);
    vector<int> new_shape(this->shape.shape);
    vector<int> new_stride(this->stride);

    new_shape[index] = range.second - range.first;
    new_stride[index] = stride[index];
    
    return Tensor(new_data, Size(new_shape), new_stride, dtype);;
}

void *Tensor::data_ptr() { return (void *)data.bp.get(); }

int Tensor::get_size(vector<int> shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
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
    CHECK_IN_RANGE(sz.dim, 0, 3,
                   "Eye dimension out of range (expected to be in range of [0, "
                   "%d), but got %d)",
                   3, sz.dim);

    if (sz.dim == 0) {
        sz = Size({1, 1});
    } else if (sz.dim == 1) {
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

}  // namespace ts
