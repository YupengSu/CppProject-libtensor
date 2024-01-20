#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <regex>
#include <vector>

#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "exception.hpp"
#include "serial_tensor.hpp"
#include "storage.hpp"

namespace ts {

//////////////add operators

TensorImpl add(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) + t2.get(i);
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        addKernel(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride, t1.dtype,
                      t1.device);
}
TensorImpl add(const TensorImpl& t1, const data_t& t2) {
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    int size = t1.shape.data_len();
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) + t2;
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        addKernelNum(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape.shape, t1.origin_stride,
                      t1.dtype, t1.device);
}
TensorImpl TensorImpl::add(const TensorImpl& other) const {
    return ts::add(*this, other);
}
TensorImpl TensorImpl::add(const data_t& other) const {
    return ts::add(*this, other);
}
TensorImpl TensorImpl::operator+(const TensorImpl& other) const {
    return ts::add(*this, other);
}
TensorImpl TensorImpl::operator+(const data_t& other) const {
    return ts::add(*this, other);
}

//////////////////sub operators
TensorImpl sub(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) - t2.get(i);
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        subKernel(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride, t1.dtype,
                      t1.device);
}
TensorImpl sub(const TensorImpl& t1, const data_t& t2) {
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    int size = t1.shape.data_len();
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) - t2;
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        subKernelNum(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape.shape, t1.origin_stride,
                      t1.dtype, t1.device);
}
TensorImpl TensorImpl::sub(const TensorImpl& other) const {
    return ts::sub(*this, other);
}
TensorImpl TensorImpl::sub(const data_t& other) const {
    return ts::sub(*this, other);
}
TensorImpl TensorImpl::operator-(const TensorImpl& other) const {
    return ts::sub(*this, other);
}
TensorImpl TensorImpl::operator-(const data_t& other) const {
    return ts::sub(*this, other);
}

////////////////mul operators
TensorImpl mul(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) * t2.get(i);
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        mulKernel(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      target_dtype, t1.device);
}
TensorImpl mul(const TensorImpl& t1, const data_t& t2) {
    int size = t1.shape.data_len();
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) * t2;
            new_data[i].set_dtype(target_dtype);
        }

    } else {
        mulKernelNum(new_data.dp, t1, t2, size, target_dtype);
    }
    return TensorImpl(new_data, t1.shape.shape, t1.origin_stride,
                      target_dtype, t1.device);
}
TensorImpl TensorImpl::mul(const TensorImpl& other) const {
    return ts::mul(*this, other);
}
TensorImpl TensorImpl::mul(const data_t& other) const {
    return ts::mul(*this, other);
}
TensorImpl TensorImpl::operator*(const TensorImpl& other) const {
    return ts::mul(*this, other);
}
TensorImpl TensorImpl::operator*(const data_t& other) const {
    return ts::mul(*this, other);
}

////////////////////////div operators
TensorImpl div(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    int size = t1.size();
    dt target_dtype;
    if (t1.dtype == dt::float64 || t2.dtype == dt::float64) {
        target_dtype = dt::float64;
    } else {
        target_dtype = dt::float32;
    }
    Storage new_data = Storage(size, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) / t2.get(i);
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        divKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      target_dtype, t1.device);
}
TensorImpl div(const TensorImpl& t1, const data_t& t2) {
    int size = t1.shape.data_len();
    Storage new_data = Storage(size, t1.device);
    dt target_dtype;
    if (t1.dtype == dt::float64 || t2.dtype == dt::float64) {
        target_dtype = dt::float64;
    } else {
        target_dtype = dt::float32;
    }
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            new_data[i] = t1.get(i) / t2;
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        divKernelNum(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape.shape, t1.origin_stride,
                      target_dtype, t1.device);
}
TensorImpl TensorImpl::div(const TensorImpl& other) const {
    return ts::div(*this, other);
}
TensorImpl TensorImpl::div(const data_t& other) const {
    return ts::div(*this, other);
}
TensorImpl TensorImpl::operator/(const TensorImpl& other) const {
    return ts::div(*this, other);
}
TensorImpl TensorImpl::operator/(const data_t& other) const {
    return ts::div(*this, other);
}

// Such operators are not support CUDA acceleration
///////////log operators
TensorImpl log(const TensorImpl& t) {
    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::float32;
    }

    int size = t.shape.data_len();
    Storage new_data = Storage(size, t.device);
    if (t.device == dev::cpu) {
        int size = t.shape.data_len();
        for (int i = 0; i < size; i++) {
            new_data[i] = std::log((double)t.get(i));
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        logKernel(new_data.dp, t, size, target_dtype);
    }

    return TensorImpl(new_data, t.shape, t.origin_stride, target_dtype, t.device);
}

// sum
TensorImpl sum(const TensorImpl& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid sum dim. %d out of %zu-D Tensor", dim, t.get_dim());

    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::int32;
    }
    target_dtype = dt::float32;

    int size = t.shape.data_len();
    int outer_size = t.shape.outer_size(dim);
    int inner_size = t.shape.inner_size(dim);
    int new_size = outer_size * inner_size;
    vector<data_t> data(new_size);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;
                data[index_new] = 0.0;
                for (int k = 0; k < t.shape[dim]; k++) {
                    data[index_new] += t.get(index_old + k * inner_size);
                }
                data[index_new].set_dtype(target_dtype);
            }
        }
    }
    // else {
    //     sumKernel(data, t, dim, outer_size, inner_size);
    // }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    return TensorImpl(data, new_shape, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::sum(int dim) const { return ts::sum(*this, dim); }

// mean
TensorImpl mean(const TensorImpl& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid mean dim. %d out of %zu-D Tensor", dim,
                   t.get_dim());

    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::float32;
    }
    target_dtype = dt::float32;

    int size = t.shape.data_len();
    int outer_size = t.shape.outer_size(dim);
    int inner_size = t.shape.inner_size(dim);
    int new_size = outer_size * inner_size;
    vector<data_t> data(new_size);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;
                data[index_new] = 0.0;
                for (int k = 0; k < t.shape[dim]; k++) {
                    data[index_new] += t.get(index_old + k * inner_size);
                }
                data[index_new] /= t.shape[dim];
                data[index_new].set_dtype(target_dtype);
            }
        }
    }
    // else {
    //     sumKernel(data, t, dim, outer_size, inner_size);
    // }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    return TensorImpl(data, new_shape, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::mean(int dim) const { return ts::mean(*this, dim); }

// max
TensorImpl max(const TensorImpl& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid max dim. %d out of %zu-D Tensor", dim, t.get_dim());

    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::float32;
    }

    int size = t.shape.data_len();
    int outer_size = t.shape.outer_size(dim);
    int inner_size = t.shape.inner_size(dim);
    int new_size = outer_size * inner_size;
    vector<data_t> data(new_size);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;

                data[index_new] = t.get(index_old);
                for (int k = 0; k < t.shape[dim]; k++) {
                    data[index_new] =
                        data[index_new] >= t.get(index_old + k * inner_size)
                            ? data[index_new]
                            : t.get(index_old + k * inner_size);
                }
                data[index_new].set_dtype(target_dtype);
            }
        }
    }
    // else {
    //     sumKernel(data, t, dim, outer_size, inner_size);
    // }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    return TensorImpl(data, new_shape, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::max(int dim) const { return ts::max(*this, dim); }

// min
TensorImpl min(const TensorImpl& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid min dim. %d out of %zu-D Tensor", dim, t.get_dim());

    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::float32;
    }

    int size = t.shape.data_len();
    int outer_size = t.shape.outer_size(dim);
    int inner_size = t.shape.inner_size(dim);
    int new_size = outer_size * inner_size;
    vector<data_t> data(new_size);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;

                data[index_new] = t.get(index_old);
                for (int k = 0; k < t.shape[dim]; k++) {
                    data[index_new] =
                        data[index_new] <= t.get(index_old + k * inner_size)
                            ? data[i * inner_size + j]
                            : t.get(index_old + k * inner_size);
                }
                data[index_new].set_dtype(target_dtype);
            }
        }
    }
    // else {
    //     sumKernel(data, t, dim, outer_size, inner_size);
    // }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    return TensorImpl(data, new_shape, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::min(int dim) const { return ts::min(*this, dim); }

///////////////comparison

TensorImpl eq(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) == t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        eqKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::eq(const TensorImpl& other) const {
    return ts::eq(*this, other);
}

TensorImpl TensorImpl::operator==(const TensorImpl& other) const {
    return ts::eq(*this, other);
}

///////////////comparison

TensorImpl ne(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) != t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        neKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::ne(const TensorImpl& other) const {
    return ts::ne(*this, other);
}

TensorImpl TensorImpl::operator!=(const TensorImpl& other) const {
    return ts::ne(*this, other);
}

TensorImpl gt(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) > t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        gtKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::gt(const TensorImpl& other) const {
    return ts::gt(*this, other);
}

TensorImpl TensorImpl::operator>(const TensorImpl& other) const {
    return ts::gt(*this, other);
}

TensorImpl lt(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) < t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        ltKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::lt(const TensorImpl& other) const {
    return ts::lt(*this, other);
}

TensorImpl TensorImpl::operator<(const TensorImpl& other) const {
    return ts::lt(*this, other);
}

TensorImpl le(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) <= t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        leKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::le(const TensorImpl& other) const {
    return ts::le(*this, other);
}

TensorImpl TensorImpl::operator<=(const TensorImpl& other) const {
    return ts::le(*this, other);
}

TensorImpl ge(const TensorImpl& t1, const TensorImpl& t2) {
    CHECK_SAME_SHAPE(t1, t2);
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = dt::bool8;
    int size = t1.size();
    Storage new_data = Storage(size, t1.device);

    if (t1.device == dev::cpu) {
        for (int i = 0; i < size; i++) {
            if (t1.get(i) >= t2.get(i)) {
                new_data[i] = true;
            } else {
                new_data[i] = false;
            }
            new_data[i].set_dtype(target_dtype);
        }
    } else {
        geKernel(new_data.dp, t1, t2, size);
    }
    return TensorImpl(new_data, t1.shape, t1.origin_stride,
                      dt::bool8, t1.device);
}

TensorImpl TensorImpl::ge(const TensorImpl& other) const {
    return ts::ge(*this, other);
}

TensorImpl TensorImpl::operator>=(const TensorImpl& other) const {
    return ts::ge(*this, other);
}

//////////////other
TensorImpl einsum(string eq, vector<TensorImpl> tensors) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::regex dot("([a-zA-Z]),\\1->");
    std::regex outer("([a-zA-Z]),([a-zA-Z])->\\1\\2");
    std::regex batch(
        "([a-zA-Z])([a-zA-Z])([a-zA-Z]),\\1\\3([a-zA-Z])->\\1\\2\\4");
    std::regex diag("([a-zA-Z])\\1->\\1");
    std::regex elewise("([a-zA-Z]),\\1->\\1");

    if (regex_match(eq, dot)) {
        // dot production
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for dot product");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];

        CHECK_SAME_SHAPE(t1, t2);
        const Size& shape = t1.shape;
        vector<data_t> data(1);
        // data_t dot_product;
        for (size_t i = 0; i < shape.data_len(); ++i) {
            data[0] += t1.get(i) * t2.get(i);
            // cout << "data1 " << t1.get(i) << " data2 " << t2.get(i) << endl;
        }
        cout << data[0] << endl;
        return TensorImpl(data, {});  // scalar //todo test
    } else if (regex_match(eq, elewise)) {
        // element-wise production
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for outer product");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        CHECK_SAME_SHAPE(t1, t2);
        vector<data_t> data(t1.shape.data_len());
        for (size_t i = 0; i < t1.shape.data_len(); ++i) {
            data[i] = t1.get(i) * t2.get(i);
        }
        return TensorImpl(data, {t1.shape.shape});
    } else if (regex_match(eq, diag)) {
        // diagnoal of t1
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for diagnoal");
        }
        const TensorImpl& t1 = tensors[0];
        const Size& shape = t1.shape;
        if (shape[0] != shape[1] || t1.ndim != 2) {
            throw std::runtime_error("Tensor is not a square matrix");
        }
        vector<data_t> data(shape[0]);
        for (size_t i = 0; i < shape[0]; ++i) {
            data[i] = t1.get(i * shape[0] + i);
        }
        return TensorImpl(data, {shape[0]});
    } else if (regex_match(eq, outer)) {
        // outer product
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for matrix multiplication");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        size_t n = t1.shape.shape[0];
        size_t m = t2.shape.shape[1];
        vector<data_t> data(n * m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                data[i * m + j] = t1.get(i) * t2.get(j);
            }
        }
        return TensorImpl(data, {t1.shape.shape[0], t2.shape.shape[1]});
    } else if (regex_match(eq, batch)) {
        // batch matrix multiplication
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for batch matrix "
                "multiplication");
        }
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for batch matrix "
                "multiplication");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        if (t1.ndim != 3 || t2.ndim != 3) {
            throw std::runtime_error("Tensors are not matrices");
        }
        const Size& shape1 = t1.shape;
        const Size& shape2 = t2.shape;
        int batches = shape1[0];
        int rows1 = shape1[1];
        int cols1 = shape1[2];
        int rows2 = shape2[1];
        int cols2 = shape2[2];

        vector<data_t> data(batches * rows1 * cols2);
        for (size_t b = 0; b < batches; ++b) {
            for (size_t i = 0; i < rows1; ++i) {
                for (size_t j = 0; j < cols2; ++j) {
                    for (size_t k = 0; k < cols1; ++k) {
                        data[(b * rows1 + i) * cols2 + j] +=
                            t1.get((b * rows1 + i) * cols1 + k) *
                            t2.get((b * rows2 + k) * cols2 + j);
                    }
                }
            }
        }
        return TensorImpl(data, {batches, rows1, cols2});
    }
    throw std::runtime_error("Invalid equation for einsum");
}
}  // namespace ts