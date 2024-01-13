#include <cstddef>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <cmath>
#include <regex>

#include "serial_tensor.hpp"

namespace ts
{
    bool CHECK_SAME_SHAPE(Tensor t1, Tensor t2, string msg)
    {
        if (t1.shape != t2.shape)
        {
            throw runtime_error(msg);
        }
        return true;
    }

    bool CHECK_SAME_DEVICE(Tensor t1, Tensor t2, string msg)
    {
        if (t1.device != t2.device)
        {
            throw runtime_error(msg);
        }
        return true;
    }

    //////////////add operators

    Tensor add(const Tensor t1, const Tensor t2)
    {
        CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match");
        CHECK_SAME_DEVICE(t1, t2, "Tensor devices do not match");
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
            data[i] = std::log((float)t.data[i]);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, t1.shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
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
        return Tensor(data, this->shape.shape, dt::bool8);
    }

    //////////////other
    Tensor einsum(string eq, vector<Tensor> tensors)
    {
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::regex dot("([a-zA-Z]),\\1->");
        std::regex outer("([a-zA-Z]),([a-zA-Z])->\\1\\2");
        std::regex batch("([a-zA-Z])([a-zA-Z])([a-zA-Z]),\\1\\3([a-zA-Z])->\\1\\2\\4");
        std::regex diag("([a-zA-Z])\\1->\\1");
        std::regex elewise("([a-zA-Z]),\\1->\\1");

        if (regex_match(eq, dot))
        {
            // dot production
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for dot product");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];

            CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match for dot product");
            const Size &shape = t1.shape;
            vector<data_t> data(1);
            // data_t dot_product;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                data[0] += t1.data[i] * t2.data[i];
                cout << "data1 " << t1.data[i] << " data2 " << t2.data[i] << endl;
            }
            cout << data[0] << endl;
            return Tensor(data, {}); // scalar //todo test
        }
        else if (regex_match(eq, elewise))
        {
            // element-wise production
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for outer product");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            CHECK_SAME_SHAPE(t1, t2, "Tensor shapes do not match for outer product");
            vector<data_t> data(t1.data.size);
            for (size_t i = 0; i < t1.data.size; ++i)
            {
                data[i] = t1.data[i] * t2.data[i];
            }
            return Tensor(data, {t1.shape.shape});
        }
        else if (regex_match(eq, diag))
        {
            // diagnoal of t1
            if (tensors.size() < 1)
            {
                throw std::runtime_error("Insufficient number of tensors for diagnoal");
            }
            const Tensor &t1 = tensors[0];
            const Size &shape = t1.shape;
            if (shape[0] != shape[1] || t1.ndim != 2)
            {
                throw std::runtime_error("Tensor is not a square matrix");
            }
            vector<data_t> data(shape[0]);
            for (size_t i = 0; i < shape[0]; ++i)
            {
                data[i] = t1.data[i * shape[0] + i];
            }
            return Tensor(data, {shape[0]});
        }
        else if (regex_match(eq, outer))
        {
            // outer product
            if (tensors.size() < 2)
            {
                throw std::runtime_error("Insufficient number of tensors for matrix multiplication");
            }
            const Tensor &t1 = tensors[0];
            const Tensor &t2 = tensors[1];
            size_t n = t1.shape.shape[0];
            size_t m = t2.shape.shape[1];
            vector<data_t> data(n * m);
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    data[i * m + j] = t1.data[i] * t2.data[j];
                }
            }
            return Tensor(data, {t1.shape.shape[0], t2.shape.shape[1]});
        }
        else if (regex_match(eq, batch))
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
            if (t1.ndim != 3 || t2.ndim!= 3)
            {
                throw std::runtime_error("Tensors are not matrices");
            }
            const Size &shape1 = t1.shape;
            const Size &shape2 = t2.shape;
            int batches = shape1[0];
            int rows1 = shape1[1];
            int cols1 = shape1[2];
            int rows2 = shape2[1];
            int cols2 = shape2[2];

            vector<data_t> data(batches * rows1 * cols2);
            for (size_t b = 0; b < batches; ++b)
            {
                for (size_t i = 0; i < rows1; ++i)
                {
                    for (size_t j = 0; j < cols2; ++j)
                    {
                        for (size_t k = 0; k < cols1; ++k)
                        {
                            data[(b * rows1 + i) * cols2 + j] += t1.data[(b * rows1 + i) * cols1 + k] * t2.data[(b * rows2 + k) * cols2 + j];
                        }
                    }
                }
            }
            return Tensor(data, {batches, rows1, cols2});
        }
        throw std::runtime_error("Invalid equation for einsum");
    }
}