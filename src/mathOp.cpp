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
    Storage new_data = Storage(new_size, t.device);
    
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;
                new_data[index_new] = 0.0;
                for (int k = 0; k < t.shape[dim]; k++) {
                    new_data[index_new] += t.get(index_old + k * inner_size);
                }
                new_data[index_new].set_dtype(target_dtype);
            }
        }
    }
    else {
        sumKernel(new_data.dp, t, dim, outer_size, inner_size, target_dtype);
    }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    if (new_shape.size() == 0) {
        new_shape.push_back(1);
    }
    vector<int> new_stride(new_shape.size());
    int stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_stride[i] = stride;
        stride *= new_shape[i];
    }
    return TensorImpl(new_data, new_shape, new_stride, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::sum(int dim) const { return ts::sum(*this, dim); }

// mean
TensorImpl mean(const TensorImpl& t, int dim) {
    return ts::div(ts::sum(t, dim), t.shape[dim]);
}

TensorImpl TensorImpl::mean(int dim) const { return ts::mean(*this, dim); }

// max
TensorImpl max(const TensorImpl& t, int dim) {
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
    Storage new_data = Storage(new_size, t.device);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;
                for (int k = 0; k < t.shape[dim]; k++) {
                    if (k == 0) {
                        new_data[index_new] = t.get(index_old + k * inner_size);
                    }
                    else {
                        new_data[index_new] =
                            new_data[index_new] >= t.get(index_old + k * inner_size)
                                ? new_data[index_new]
                                : t.get(index_old + k * inner_size);
                    }
                }
                new_data[index_new].set_dtype(target_dtype);
            }
        }
    }
    else {
        maxKernal(new_data.dp, t, dim, outer_size, inner_size, target_dtype);
    }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    if (new_shape.size() == 0) {
        new_shape.push_back(1);
    }
    vector<int> new_stride(new_shape.size());
    int stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_stride[i] = stride;
        stride *= new_shape[i];
    }
    return TensorImpl(new_data, new_shape, new_stride, target_dtype, t.device);

    // Reduce dim
}

TensorImpl TensorImpl::max(int dim) const { return ts::max(*this, dim); }

// min
TensorImpl min(const TensorImpl& t, int dim) {
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
    Storage new_data = Storage(new_size, t.device);
    if (t.device == dev::cpu) {
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                size_t index_new = i * inner_size + j;
                size_t index_old = i * inner_size * t.shape[dim] + j;
                for (int k = 0; k < t.shape[dim]; k++) {
                    if (k == 0) {
                        new_data[index_new] = t.get(index_old + k * inner_size);
                    }
                    else {
                        new_data[index_new] =
                            new_data[index_new] <= t.get(index_old + k * inner_size)
                                ? new_data[index_new]
                                : t.get(index_old + k * inner_size);
                    }
                }
                new_data[index_new].set_dtype(target_dtype);
            }
        }
    }
    else {
        minKernal(new_data.dp, t, dim, outer_size, inner_size, target_dtype);
    }
    vector<int> new_shape = t.shape.shape;
    new_shape.erase(new_shape.begin() + dim);
    if (new_shape.size() == 0) {
        new_shape.push_back(1);
    }
    vector<int> new_stride(new_shape.size());
    int stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_stride[i] = stride;
        stride *= new_shape[i];
    }
    return TensorImpl(new_data, new_shape, new_stride, target_dtype, t.device);

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
TensorImpl matrix_multiply(const TensorImpl& t1, const TensorImpl t2) {
    CHECK_SAME_DEVICE(t1, t2);
    dt target_dtype = descision_dtype(t1.dtype, t2.dtype);
    size_t M = t1.shape[0];
    size_t N = t2.shape[1];
    size_t K = t1.shape[1];
    if (t1.shape[1] != t2.shape[0]) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    Storage new_data = Storage(M * N, t1.device);
    new_data.dtype = target_dtype;
    if (t1.device == dev::cpu) {
        for (int i = 0; i < t1.shape[0]; i++) {
            for (int j = 0; j < t2.shape[1]; j++) {
                new_data[i * t2.shape[1] + j] = 0;
                for (int k = 0; k < t1.shape[1]; k++) {
                    new_data[i * t2.shape[1] + j] +=
                        t1.get(i * t1.shape[1] + k) * t2.get(k * t2.shape[1] + j);
                }
                new_data[i * t2.shape[1] + j].set_dtype(target_dtype);
            }
        }

    } else {
        matrixMultiplyKernel(new_data.dp, t1, t2, M, N, K, target_dtype);
    }
    vector<int> new_shape = {(int)M, (int)N};
    vector<int> new_stride = {(int)N, 1};

    return TensorImpl(new_data, new_shape, new_stride, t1.dtype,
                      t1.device);
}



// 9) Pointwise mul and reduce sum, ‘ij,ij->’ , 
// 12) Tensor contraction, ‘pqrs,tuqvr->pstuv’ ,
// 13) Bilinear transformation, ‘ik,jkl->ij’

TensorImpl einsum(string eq, vector<TensorImpl> tensors) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::regex dot("([a-zA-Z]),\\1->");                                          // 8) Dot product, ‘i,i->’ 
    std::regex outer("([a-zA-Z]),([a-zA-Z])->\\1\\2");                           //10) Outer product, ‘i,j->ij’
    std::regex batch(
        "([a-zA-Z])([a-zA-Z])([a-zA-Z]),\\1\\3([a-zA-Z])->\\1\\2\\4");          //11) Batch matrix mul, ‘ijk,ikl->ijl’ ,
    std::regex diag("([a-zA-Z])\\1->\\1");                                       //1) Extracting elements along diagonal, ‘ii->i’ 
    std::regex elewise("([a-zA-Z]),\\1->\\1");

    std::regex transposed("([a-zA-Z]),([a-zA-Z])->\\2\\1");                      // 2) Transpose, ‘ij->ji’, 
    std::regex permute(".*([a-zA-Z])([a-zA-Z])->.*\\2\\1");                          //3) Permuate, ‘…ij->…ji’ , 
    std::regex sum_along_dimension("([a-zA-Z])([a-zA-Z])->\\2");                  // 5) Sum along dimension, ‘ij->j’ , 
    std::regex matrix_mul("([a-zA-Z])([a-zA-Z]),([a-zA-Z])([a-zA-Z])->\\1\\4");  //7) Matrix mul, ‘ik, kj->ij’ ,
    std::regex reduce_sum("([a-zA-Z])([a-zA-Z])->");                                         // 4) Reduce sum, ‘ij->’,
    std::regex matrix_vector_multiply("([a-zA-Z])([a-zA-Z]),\\2->\\1");           //6) Matrix and vector mul, ‘ik, k->i’, 7) 
    std::regex point_wise("([a-zA-Z])([a-zA-Z]),\\1\\2->");                           // 9) Pointwise mul and reduce sum, ‘ij,ij->’ ,
    std::regex bilinear("([a-zA-Z])([a-zA-Z]),([a-zA-Z])([a-zA-Z])->\\1\\3\\2\\4");  // 13) Bilinear transformation, ‘ik,jkl->ij’
    std::regex contraction("([a-zA-Z])([a-zA-Z])([a-zA-Z])([a-zA-Z]),([a-zA-Z])([a-zA-Z])\\2([a-zA-Z])\\3->\\1\\4\\5\\6\\7");  // 12) Tensor contraction, ‘pqrs,tuqvr->pstuv’ ,

    if (regex_match(eq, dot)) {
        // dot production
        cout << "dot product" << endl;
        if (tensors.size() != 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for dot product");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];

        if (t1.ndim != 1 || t2.ndim != 1) {
            throw std::runtime_error("Tensors are not vectors");
        }
        
        return ts::sum(ts::mul(t1, t2), 0);  // scalar 
    } 
    else if (regex_match(eq, elewise)) {
        // element-wise production
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for outer product");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        return ts::mul(t1, t2);
    } 
    else if (regex_match(eq, diag)) {
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
    } 
    else if (regex_match(eq, outer)) {
        // outer product
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for matrix multiplication");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        t1.unsqueeze(1);
        t2.unsqueeze(0);
        return ts::matrix_multiply(t1, t2);
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
        int cols2 = shape2[2];
        TensorImpl result = ts::zeros({batches, rows1, cols2});
        for (size_t b = 0; b < batches; ++b) {
            result[b] = ts::matrix_multiply(t1[b], t2[b]);
        }
        return result;
    } else if(regex_match(eq,transposed)){
        // transpose
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for transpose");
        }
        return tensors[0].transpose(tensors[0].ndim - 1, tensors[0].ndim - 2);
    } 
    else if(regex_match(eq,permute)){
        // permute: ‘…ij->…ji’ means
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for permute");
        }
        return tensors[0].transpose(tensors[0].ndim - 1, tensors[0].ndim - 2);
    } else if(regex_match(eq,sum_along_dimension)){
        // sum along dimension
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for sum along dimension");
        }
        const TensorImpl& t1 = tensors[0];
        return ts::sum(t1, 0);
    } 
    else if(regex_match(eq,matrix_mul)){
        // matrix multiply
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for matrix multiply");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        if (t1.ndim != 2 || t2.ndim != 2) {
            throw std::runtime_error("Tensors are not matrices");
        }
        return ts::matrix_multiply(t1, t2);
    }
    else if(regex_match(eq,reduce_sum)){
        // reduce sum
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for reduce sum");
        }
        const TensorImpl& t1 = tensors[0];
        return ts::sum(ts::sum(t1, 0), 0);
    } 
    else if(regex_match(eq,point_wise)){
        // point wise
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for point wise");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        return ts::sum(ts::sum(ts::mul(t1, t2),0), 0);
    }
    
    else if(regex_match(eq,matrix_vector_multiply)){
        // matrix vector multiply
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for matrix vector multiply");
        }
        const TensorImpl& t1 = tensors[0];
        const TensorImpl& t2 = tensors[1];
        if (t1.ndim != 2 || t2.ndim != 1) {
            throw std::runtime_error("Tensors are not matrices");
        }
        TensorImpl t3 = t2.unsqueeze(1);
        return ts::matrix_multiply(t1, t3).squeeze();
    } 
    throw std::runtime_error("Invalid equation for einsum");
}
}  // namespace ts