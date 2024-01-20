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

Tensor add(const Tensor& t1, const Tensor& t2)  {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), t1.dtype,
                  t1.device);
}
Tensor add(const Tensor& t1, const data_t& t2) {
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
    return Tensor(new_data, t1.shape.shape, init_stride(t1.shape.shape),
                  t1.dtype, t1.device);
}
Tensor Tensor::add(const Tensor& other) const  { return ts::add(*this, other); }
Tensor Tensor::add(const data_t& other) const { return ts::add(*this, other); }
Tensor Tensor::operator+(const Tensor& other) const { return ts::add(*this, other); }
Tensor Tensor::operator+(const data_t& other) const { return ts::add(*this, other); }

//////////////////sub operators
Tensor sub(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), t1.dtype,
                  t1.device);
}
Tensor sub(const Tensor& t1, const data_t& t2) {
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
    return Tensor(new_data, t1.shape.shape, init_stride(t1.shape.shape),
                  t1.dtype, t1.device);
}
Tensor Tensor::sub(const Tensor& other) const { return ts::sub(*this, other); }
Tensor Tensor::sub(const data_t& other) const { return ts::sub(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return ts::sub(*this, other); }
Tensor Tensor::operator-(const data_t& other) const { return ts::sub(*this, other); }

////////////////mul operators
Tensor mul(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), target_dtype,
                  t1.device);
}
Tensor mul(const Tensor& t1, const data_t& t2) {
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
    return Tensor(new_data, t1.shape.shape, init_stride(t1.shape.shape),
                  target_dtype, t1.device);
}
Tensor Tensor::mul(const Tensor& other) const { return ts::mul(*this, other); }
Tensor Tensor::mul(const data_t& other) const { return ts::mul(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return ts::mul(*this, other); }
Tensor Tensor::operator*(const data_t& other) const { return ts::mul(*this, other); }

////////////////////////div operators
Tensor div(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), target_dtype,
                  t1.device);
}
Tensor div(const Tensor& t1, const data_t& t2) {
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
    return Tensor(new_data, t1.shape.shape, init_stride(t1.shape.shape),
                  target_dtype, t1.device);
}
Tensor Tensor::div(const Tensor& other) const { return ts::div(*this, other); }
Tensor Tensor::div(const data_t& other) const { return ts::div(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return ts::div(*this, other); }
Tensor Tensor::operator/(const data_t& other) const { return ts::div(*this, other); }


// Such operators are not support CUDA acceleration
///////////log operators
Tensor log(const Tensor& t) {
    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::float32;
    }
    vector<data_t> data(t.shape.data_len());
    int size = t.shape.data_len();
    for (int i = 0; i < size; i++) {
        data[i] = std::log((double)t.get(i));
        data[i].set_dtype(target_dtype);
    }
    return Tensor(data, t.shape.shape, target_dtype, t.device);
}

// sum
Tensor sum(const Tensor& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid sum dim. %d out of %zu-D Tensor", dim, t.get_dim());

    dt target_dtype;
    if (is_floating(t.dtype)) {
        target_dtype = t.dtype;
    } else {
        target_dtype = dt::int32;
    }
    if (t.get_dim() == 1) {
        vector<data_t> newdata(1);
        vector<data_t> data(t.get_serial_data());
        double sum = 0.0;
        for (int i = 0; i < t.shape[dim]; i++) {
            sum += data[i].to_dt(dt::float64).data.tensor_float64;
        }
        newdata[0] = sum;
        newdata[0].set_dtype(target_dtype);
        return Tensor(newdata, {1}, target_dtype, t.device);
    } else {
        vector<data_t> newdata(t.shape[dim]);
        for (int i = 0; i < t.shape[dim]; i++) {
            vector<data_t> data(t.slice(i, dim).get_serial_data());
            double sum = 0.0;
            for (int j = 0; j < data.size(); j++) {
                sum += data[j].to_dt(dt::float64).data.tensor_float64;
            }

            newdata[i] = sum;
            newdata[i].set_dtype(target_dtype);
        }
        return Tensor(newdata, {t.shape[dim]}, target_dtype, t.device);
    }
}

Tensor Tensor::sum(int dim) const {
    return ts::sum(*this, dim);
}

// mean
Tensor mean(const Tensor& t, int dim) {
    dt target_dtype = t.dtype;
    if (t.get_dim() == 1) {
        vector<data_t> newdata(1);
        vector<data_t> data(t.get_serial_data());
        double sum = 0.0;
        for (int i = 0; i < t.shape[dim]; i++) {
            sum += data[i].to_dt(dt::float64).data.tensor_float64;
        }
        newdata[0] = sum / t.shape[dim];
        newdata[0].set_dtype(target_dtype);
        return Tensor(newdata, {1}, target_dtype, t.device);
    } else {
        vector<data_t> newdata(t.shape[dim]);
        for (int i = 0; i < t.shape[dim]; i++) {
            vector<data_t> data(t.slice(i, dim).get_serial_data());
            double sum = 0.0;
            for (int j = 0; j < data.size(); j++) {
                sum += data[j].to_dt(dt::float64).data.tensor_float64;
            }

            newdata[i] = sum / t.shape[dim];
            newdata[i].set_dtype(target_dtype);
        }
        return Tensor(newdata, {t.shape[dim]}, target_dtype, t.device);
    }
}

Tensor Tensor::mean(int dim) const {
    return ts::mean(*this, dim);
}

// max
Tensor max(const Tensor& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid max dim. %d out of %zu-D Tensor", dim, t.get_dim());
    dt target_dtype = t.dtype;
    if (t.get_dim() == 1) {
        vector<data_t> newdata(1);
        vector<data_t> data(t.get_serial_data());
        double max = data[0].to_dt(dt::float64).data.tensor_float64;
        for (int i = 1; i < t.shape[dim]; i++) {
            double tmp = data[i].to_dt(dt::float64).data.tensor_float64;
            max = tmp > max ? tmp : max;
        }
        newdata[0] = max;
        newdata[0].set_dtype(target_dtype);
        return Tensor(newdata, {1}, target_dtype, t.device);
    } else {
        vector<data_t> newdata(t.shape[dim]);
        for (int i = 0; i < t.shape[dim]; i++) {
            vector<data_t> data(t.slice(i, dim).get_serial_data());
            double max = data[0].to_dt(dt::float64).data.tensor_float64;
            for (int j = 1; j < data.size(); j++) {
                double tmp = data[i].to_dt(dt::float64).data.tensor_float64;
                max = tmp > max ? tmp : max;
            }

            newdata[i] = max;
            newdata[i].set_dtype(target_dtype);
        }
        return Tensor(newdata, {t.shape[dim]}, target_dtype, t.device);
    }
}

Tensor Tensor::max(int dim) const {
    return ts::max(*this, dim);
}

// min
Tensor min(const Tensor& t, int dim) {
    CHECK_IN_RANGE(dim, 0, t.get_dim(),
                   "Invalid min dim. %d out of %zu-D Tensor", dim, t.get_dim());
    dt target_dtype = t.dtype;
    if (t.get_dim() == 1) {
        vector<data_t> newdata(1);
        vector<data_t> data(t.get_serial_data());
        double min = data[0].to_dt(dt::float64).data.tensor_float64;
        for (int i = 1; i < t.shape[dim]; i++) {
            double tmp = data[i].to_dt(dt::float64).data.tensor_float64;
            min = tmp < min ? tmp : min;
        }
        newdata[0] = min;
        newdata[0].set_dtype(target_dtype);
        return Tensor(newdata, {1}, target_dtype, t.device);
    } else {
        vector<data_t> newdata(t.shape[dim]);
        for (int i = 0; i < t.shape[dim]; i++) {
            vector<data_t> data(t.slice(i, dim).get_serial_data());
            double min = data[0].to_dt(dt::float64).data.tensor_float64;
            for (int j = 1; j < data.size(); j++) {
                double tmp = data[i].to_dt(dt::float64).data.tensor_float64;
                min = tmp < min ? tmp : min;
            }

            newdata[i] = min;
            newdata[i].set_dtype(target_dtype);
        }
        return Tensor(newdata, {t.shape[dim]}, target_dtype, t.device);
    }
}

Tensor Tensor::min(int dim) const {
    return ts::min(*this, dim);
}


///////////////comparison

Tensor eq(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::eq(const Tensor& other) const {
    return ts::eq(*this, other);
}

Tensor Tensor::operator==(const Tensor& other) const {
    return ts::eq(*this, other);
}

///////////////comparison

Tensor ne(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::ne(const Tensor& other) const {
    return ts::ne(*this, other);
}

Tensor Tensor::operator!=(const Tensor& other) const {
    return ts::ne(*this, other);
}

Tensor gt(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::gt(const Tensor& other) const {
    return ts::gt(*this, other);
}

Tensor Tensor::operator>(const Tensor& other) const {
    return ts::gt(*this, other);
}

Tensor lt(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::lt(const Tensor& other) const {
    return ts::lt(*this, other);
}

Tensor Tensor::operator<(const Tensor& other) const {
    return ts::lt(*this, other);
}

Tensor le(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::le(const Tensor& other) const {
    return ts::le(*this, other);
}

Tensor Tensor::operator<=(const Tensor& other) const {
    return ts::le(*this, other);
}

Tensor ge(const Tensor& t1, const Tensor& t2) {
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
    return Tensor(new_data, t1.shape, init_stride(t1.shape.shape), dt::bool8, t1.device);
}

Tensor Tensor::ge(const Tensor& other) const {
    return ts::ge(*this, other);
}

Tensor Tensor::operator>=(const Tensor& other) const {
    return ts::ge(*this, other);
}



//////////////other
Tensor einsum(string eq, vector<Tensor> tensors) {
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
        const Tensor& t1 = tensors[0];
        const Tensor& t2 = tensors[1];

        CHECK_SAME_SHAPE(t1, t2);
        const Size& shape = t1.shape;
        vector<data_t> data(1);
        // data_t dot_product;
        for (size_t i = 0; i < shape.data_len(); ++i) {
            data[0] += t1.get(i) * t2.get(i);
            // cout << "data1 " << t1.get(i) << " data2 " << t2.get(i) << endl;
        }
        cout << data[0] << endl;
        return Tensor(data, {});  // scalar //todo test
    } else if (regex_match(eq, elewise)) {
        // element-wise production
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for outer product");
        }
        const Tensor& t1 = tensors[0];
        const Tensor& t2 = tensors[1];
        CHECK_SAME_SHAPE(t1, t2);
        vector<data_t> data(t1.shape.data_len());
        for (size_t i = 0; i < t1.shape.data_len(); ++i) {
            data[i] = t1.get(i) * t2.get(i);
        }
        return Tensor(data, {t1.shape.shape});
    } else if (regex_match(eq, diag)) {
        // diagnoal of t1
        if (tensors.size() < 1) {
            throw std::runtime_error(
                "Insufficient number of tensors for diagnoal");
        }
        const Tensor& t1 = tensors[0];
        const Size& shape = t1.shape;
        if (shape[0] != shape[1] || t1.ndim != 2) {
            throw std::runtime_error("Tensor is not a square matrix");
        }
        vector<data_t> data(shape[0]);
        for (size_t i = 0; i < shape[0]; ++i) {
            data[i] = t1.get(i * shape[0] + i);
        }
        return Tensor(data, {shape[0]});
    } else if (regex_match(eq, outer)) {
        // outer product
        if (tensors.size() < 2) {
            throw std::runtime_error(
                "Insufficient number of tensors for matrix multiplication");
        }
        const Tensor& t1 = tensors[0];
        const Tensor& t2 = tensors[1];
        size_t n = t1.shape.shape[0];
        size_t m = t2.shape.shape[1];
        vector<data_t> data(n * m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                data[i * m + j] = t1.get(i) * t2.get(j);
            }
        }
        return Tensor(data, {t1.shape.shape[0], t2.shape.shape[1]});
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
        const Tensor& t1 = tensors[0];
        const Tensor& t2 = tensors[1];
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
        return Tensor(data, {batches, rows1, cols2});
    }
    throw std::runtime_error("Invalid equation for einsum");
}
}  // namespace ts