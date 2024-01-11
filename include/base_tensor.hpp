#pragma once
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include "config.hpp"
#include "size.hpp"
#include "storage.hpp"

using namespace std;

namespace ts {

template <class value_type = float>
class BaseTensor {
   public:
    int dim;
    int my_dim;
    Size shape;  // Using shared_ptr with arrays
    dt dtype;

    vector<BaseTensor<value_type>> data;
    value_type scaler;

    ~BaseTensor() {}

    BaseTensor() {
        this->dim = 0;

        this->shape = Size();
        this->data = {};
    }
    BaseTensor &operator=(int data) {
        assert(this->dim == 0);
        this->scaler = data;
        return *this;
    }

    BaseTensor &operator[](int index) {
        assert(this->dim > 0);
        assert(index < this->shape.shape[0]);

        return this->data[index];
    }

    BaseTensor(value_type data, dt dtype = DEFAULT_DTYPE) {
        this->dim = 0;
        this->shape = Size();
        this->scaler = data;
        this->set_dtype(dtype);
    }

    BaseTensor(initializer_list<BaseTensor<value_type>> data,
               dt dtype = DEFAULT_DTYPE) {
        this->data = {};

        for (BaseTensor<value_type> i : data) {
            this->data.push_back(i);
        }
        BaseTensor<value_type> first = this->data[0];
        if (first.dim == 0) {
            this->shape = Size((int)this->data.size());
        } else {
            vector<int> shape(first.shape.shape);
            shape.insert(shape.begin(), (int)this->data.size());
            this->shape = Size(shape);
        }
        this->dim = first.dim + 1;

        for (BaseTensor ts : data) {
            assert(first.dim == ts.dim);
            for (int i = 0; i < this->dim; i++) {
                assert(first.shape.shape[i] == ts.shape.shape[i]);
            }
        }
        this->set_dtype(dtype);
    }

    friend ostream &operator<<(ostream &os, BaseTensor<value_type> &tsr) {
        if (tsr.dim == 0) {
            os << tsr.scaler;
            return os;
        } else {
            os << "[";
            for (int i = 0; i < tsr.data.size(); i++) {
                os << tsr.data[i];
                if (i != tsr.data.size() - 1) {
                    os << ", ";

                    if (tsr.dim >= 2) {
                        os << endl;
                    }
                }
            }
            os << ']';

            return os;
        }
    }

    void set_dtype(dt dtype) {
        if (this->dtype == dtype) {
            return;
        }
        this->dtype = dtype;

        if (this->dim == 0) {
            switch (dtype) {
                case dt::int8:
                    this->scaler = (uint8_t)this->scaler;
                    break;
                case dt::float32:
                    this->scaler = (float)this->scaler;
                    break;
                case dt::bool8:
                    this->scaler = (bool)this->scaler;
                    break;
                case dt::int32:
                    this->scaler = (int)this->scaler;
                    break;
                case dt::float64:
                    this->scaler = (double)this->scaler;
                    break;
                default:
                    break;
            }
            this->dtype = dtype;
        } else {
            for (BaseTensor<value_type> &ts : this->data) {
                ts.set_dtype(dtype);
            }
        }
        this->dtype = dtype;
    }

    vector<value_type> get_data() {
        vector<value_type> data;
        if (this->dim == 0) {
            data.push_back(this->scaler);
        } else {
            for (BaseTensor<value_type> &ts : this->data) {
                vector<value_type> ts_data = ts.get_data();
                for (value_type &i : ts_data) {
                    data.push_back(i);
                }
            }
        }
        return data;
    }
};

}  // namespace ts