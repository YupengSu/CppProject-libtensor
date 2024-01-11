#include <cstddef>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>

#include "serial_tensor.hpp"
#include "exception.hpp"
#include "storage.hpp"
#include "math.hpp"

namespace ts
{

    Tensor::Tensor(Size shape) : data(shape.size())
    {
        this->ndim = shape.ndim;          // number of dimensions
        this->shape = shape;              // shape of tensor
        this->stride.reserve(shape.ndim); // stride of tensor
        // reserve() is used to allocate memory for a vector
        this->offset = 0;
    }

    bool CHECK_SAME_SHAPE(Tensor t1, Tensor t2, string msg)
    {
        if (t1.shape != t2.shape)
        {
            throw runtime_error(msg);
        }
        return true;
    }

    //////////////add operators

    Tensor add(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] + t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor add(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] + t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::add(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::add(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other;
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator+(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] + other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    //////////////////sub operators
    Tensor sub(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] - t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor sub(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] - t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::sub(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator-(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::sub(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] - other;
        }
        return Tensor(data, this->shape.shape);
    }

    ////////////////mul operators

    Tensor mul(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] * t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor mul(const Tensor t1, data_t t2)
    {
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] * t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::operator*(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::mul(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::mul(data_t other)
    {
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] * other;
        }
        return Tensor(data, this->shape.shape);
    }

    ////////////////////////div operators

    Tensor div(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] / t2.data[i];
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor div(const Tensor t1, data_t t2)
    {
        if (t2 == 0)
            throw runtime_error("Division by zero");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = t1.data[i] / t2;
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::div(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (other.data[i] == 0)
            {
                throw runtime_error("Division by zero");
            }
            data[i] = this->data[i] / other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator/(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (other.data[i] == 0)
            {
                throw runtime_error("Division by zero");
            }
            data[i] = this->data[i] / other.data[i];
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::div(data_t other)
    {
        if (other == 0)
            throw runtime_error("Division by zero");
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = this->data[i] / other;
        }
        return Tensor(data, this->shape.shape);
    }

    ///////////log operators

    Tensor log(const Tensor t)
    {
        vector<data_t> data(t.data.size);
        int size = t.data.size;
        for (int i = 0; i < size; i++)
        {
            data[i] = std::log(t.data[i]);
        }
        return Tensor(data, t.shape.shape);
    }

    ///////////////find sum 0 means row wise and 1 means column wise

    // sum
    Tensor sum(const Tensor t, int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(t.shape[0]);
            for (int i = 0; i < t.shape[0]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < t.shape[1]; j++)
                {
                    data[i] += t.data[i * t.shape[1] + j];
                }
            }
            return Tensor(data, {t.shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(t.shape[1]);
            for (int i = 0; i < t.shape[1]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < t.shape[0]; j++)
                {
                    data[i] += t.data[j * t.shape[1] + i];
                }
            }
            return Tensor(data, {t.shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::sum(int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(this->shape[0]);
            for (int i = 0; i < this->shape[0]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < this->shape[1]; j++)
                {
                    data[i] += this->data[i * this->shape[1] + j];
                }
            }
            return Tensor(data, {this->shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(this->shape[1]);
            for (int i = 0; i < this->shape[1]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < this->shape[0]; j++)
                {
                    data[i] += this->data[j * this->shape[1] + i];
                }
            }
            return Tensor(data, {this->shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    // mean
    Tensor mean(const Tensor t, int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(t.shape[0]);
            for (int i = 0; i < t.shape[0]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < t.shape[1]; j++)
                {
                    data[i] += t.data[i * t.shape[1] + j];
                }
                data[i] /= t.shape[1];
            }
            return Tensor(data, {t.shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(t.shape[1]);
            for (int i = 0; i < t.shape[1]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < t.shape[0]; j++)
                {
                    data[i] += t.data[j * t.shape[1] + i];
                }
                data[i] /= t.shape[0];
            }
            return Tensor(data, {t.shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::mean(int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(this->shape[0]);
            for (int i = 0; i < this->shape[0]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < this->shape[1]; j++)
                {
                    data[i] += this->data[i * this->shape[1] + j];
                }
                data[i] /= this->shape[1];
            }
            return Tensor(data, {this->shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(this->shape[1]);
            for (int i = 0; i < this->shape[1]; i++)
            {
                data[i] = 0;
                for (int j = 0; j < this->shape[0]; j++)
                {
                    data[i] += this->data[j * this->shape[1] + i];
                }
                data[i] /= this->shape[0];
            }
            return Tensor(data, {this->shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    // max
    Tensor max(Tensor t, int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(t.shape[0]);
            for (int i = 0; i < t.shape[0]; i++)
            {
                data[i] = t.data[i * t.shape[1]];
                for (int j = 0; j < t.shape[1]; j++)
                {
                    if (data[i] < t.data[i * t.shape[1] + j])
                    {
                        data[i] = t.data[i * t.shape[1] + j];
                    }
                }
            }
            return Tensor(data, {t.shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(t.shape[1]);
            for (int i = 0; i < t.shape[1]; i++)
            {
                data[i] = t.data[i];
                for (int j = 0; j < t.shape[0]; j++)
                {
                    if (data[i] < t.data[j * t.shape[1] + i])
                    {
                        data[i] = t.data[j * t.shape[1] + i];
                    }
                }
            }
            return Tensor(data, {t.shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::max(int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(this->shape[0]);
            for (int i = 0; i < this->shape[0]; i++)
            {
                data[i] = this->data[i * this->shape[1]];
                for (int j = 0; j < this->shape[1]; j++)
                {
                    if (data[i] < this->data[i * this->shape[1] + j])
                    {
                        data[i] = this->data[i * this->shape[1] + j];
                    }
                }
            }
            return Tensor(data, {this->shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(this->shape[1]);
            for (int i = 0; i < this->shape[1]; i++)
            {
                data[i] = this->data[i];
                for (int j = 0; j < this->shape[0]; j++)
                {
                    if (data[i] < this->data[j * this->shape[1] + i])
                    {
                        data[i] = this->data[j * this->shape[1] + i];
                    }
                }
            }
            return Tensor(data, {this->shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    // min
    Tensor min(Tensor t, int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(t.shape[0]);
            for (int i = 0; i < t.shape[0]; i++)
            {
                data[i] = t.data[i * t.shape[1]];
                for (int j = 0; j < t.shape[1]; j++)
                {
                    if (data[i] > t.data[i * t.shape[1] + j])
                    {
                        data[i] = t.data[i * t.shape[1] + j];
                    }
                }
            }
            return Tensor(data, {t.shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(t.shape[1]);
            for (int i = 0; i < t.shape[1]; i++)
            {
                data[i] = t.data[i];
                for (int j = 0; j < t.shape[0]; j++)
                {
                    if (data[i] > t.data[j * t.shape[1] + i])
                    {
                        data[i] = t.data[j * t.shape[1] + i];
                    }
                }
            }
            return Tensor(data, {t.shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    Tensor Tensor::min(int dim)
    {
        if (dim == 0)
        {
            vector<data_t> data(this->shape[0]);
            for (int i = 0; i < this->shape[0]; i++)
            {
                data[i] = this->data[i * this->shape[1]];
                for (int j = 0; j < this->shape[1]; j++)
                {
                    if (data[i] > this->data[i * this->shape[1] + j])
                    {
                        data[i] = this->data[i * this->shape[1] + j];
                    }
                }
            }
            return Tensor(data, {this->shape[0]});
        }
        else if (dim == 1)
        {
            vector<data_t> data(this->shape[1]);
            for (int i = 0; i < this->shape[1]; i++)
            {
                data[i] = this->data[i];
                for (int j = 0; j < this->shape[0]; j++)
                {
                    if (data[i] > this->data[j * this->shape[1] + i])
                    {
                        data[i] = this->data[j * this->shape[1] + i];
                    }
                }
            }
            return Tensor(data, {this->shape[1]});
        }
        else
        {
            throw runtime_error("Dimension out of range");
        }
    }

    ///////////////comparison
    Tensor eq(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] == t2.data[i])
            {
                data[i] = 1;
            }
            else
            {
                data[i] = 0;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor eq(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] == t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::eq(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] == other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator==(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] == other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    // ne
    Tensor ne(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] != t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::ne(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] != other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator!=(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] != other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    // gt
    Tensor gt(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] > t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::gt(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] > other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator>(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] > other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    // ge
    Tensor ge(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] >= t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::ge(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] >= other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator>=(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] >= other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    // lt
    Tensor lt(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] < t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::lt(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] < other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator<(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] < other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    // le
    Tensor le(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(t1.data.size);
        int size = t1.data.size;
        for (int i = 0; i < size; i++)
        {
            if (t1.data[i] <= t2.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, t1.shape.shape);
    }

    Tensor Tensor::le(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] <= other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    Tensor Tensor::operator<=(const Tensor &other)
    {
        CHECK_SAME_SHAPE(*this, other, "Tensor shapes do not match");
        std::cout.setf(std::ios::boolalpha);
        vector<data_t> data(this->data.size);
        int size = this->data.size;
        for (int i = 0; i < size; i++)
        {
            if (this->data[i] <= other.data[i])
            {
                data[i] = true;
            }
            else
            {
                data[i] = false;
            }
        }
        return Tensor(data, this->shape.shape);
    }

    //////////////other
    Tensor einsum(string eq, vector<Tensor> tensors)
    {
        if (eq == "i,i->")
        {
            // dot production
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for dot product");
            }

            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match for dot product");

            const Size &shape = t1.shape();
            const data_t *data1 = t1.data();
            const data_t *data2 = t2.data();
            data_t dot_product = 0;

            for (size_t i = 0; i < shape.size(); ++i)
            {
                dot_product += data1[i] * data2[i];
            }

            result = dot_product;
            return Tensor(result, {});
        }
        else if (eq == "i,i->i")
        {
            // element-wise production
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for outer product");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match for outer product");
            const Size &shape = t1.shape();
            const data_t *data1 = t1.data();
            const data_t *data2 = t2.data();
            std::vector<data_t> result(shape.size());

            for (size_t i = 0; i < shape.size(); ++i)
            {
                result[i] = data1[i] * data2[i];
            }

            return Tensor(result, shape);
        }
        else if (eq == "ii->i")
        {
            // diagnoal of t1
            if (tensors.size() < 1)
            {
                throw std::runtime_error("Insufficient number of tensors for diagnoal");
            }
            const Tensor &t1 = tensors[0];
            const Size &shape = t1.shape();
            const data_t *data1 = t1.data();
            vector<data_t> result(shape.size());
            for (size_t i = 0; i < shape.size(); ++i)
            {
                result[i] = data1[i * shape[1] + i];
            }
            return Tensor(result, {shape[0]});
        }
        else if (eq == "i,j->ij")
        {
            // outer product
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for matrix multiplication");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            CHECK_MATRIX_SHAPE(t1, "First tensor is not a matrix");
            CHECK_MATRIX_SHAPE(t2, "Second tensor is not a matrix");
            const Size &shape1 = t1.shape();
            const Size &shape2 = t2.shape();
            size_t rows = shape1[0];
            size_t cols1 = shape1[1];
            size_t cols2 = shape2[1];
            const data_t *data1 = t1.data();
            const data_t *data2 = t2.data();
            std::vector<data_t> result(rows * cols2);

            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < cols2; ++j)
                {
                    data_t element = 0;
                    for (size_t k = 0; k < cols1; ++k)
                    {
                        element += data1[i * cols1 + k] * data2[k * cols2 + j];
                    }
                    result[i * cols2 + j] = element;
                }
            }

            return Tensor(result, {rows, cols2});
        }
        else if (eq == "bij,bjk->bik")
        {
            // batch matrix multiplication
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for batch matrix multiplication");
            }
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for batch matrix multiplication");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            CHECK_BATCH_MATRIX_SHAPE(t1, "First tensor is not a batch matrix");
            CHECK_BATCH_MATRIX_SHAPE(t2, "Second tensor is not a batch matrix");
            const Size &shape1 = t1.shape();
            const Size &shape2 = t2.shape();
            size_t batches = shape1[0];
            size_t rows1 = shape1[1];
            size_t cols1 = shape1[2];
            size_t rows2 = shape2[1];
            size_t cols2 = shape2[2];
            const data_t *data1 = t1.data();
            const data_t *data2 = t2.data();
            std::vector<data_t> result(batches * rows1 * cols2);

            for (size_t b = 0; b < batches; ++b)
            {
                for (size_t i = 0; i < rows1; ++i)
                {
                    for (size_t j = 0; j < cols2; ++j)
                    {
                        data_t element = 0;
                        for (size_t k = 0; k < cols1; ++k)
                        {
                            element += data1[(b * rows1 + i) * cols1 + k] * data2[(b * rows2 + k) * cols2 + j];
                        }
                        result[(b * rows1 + i) * cols2 + j] = element;
                    }
                }
            }

            return Tensor(result, {batches, rows1, cols2});
        }
        throw std::runtime_error("Invalid equation for einsum");
    }
}