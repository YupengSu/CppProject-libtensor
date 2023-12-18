#pragma once
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>
#include "size.hpp"
#include "storage.hpp"

using namespace std;

namespace ts {

template <class value_type = data_t>
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

    template <class iterT>
    BaseTensor(iterT begin, iterT end, dt dtype = float32) {
        this->dim = 1;
        // this->shape = Size((int)ts.size());
        this->shape = Size(begin - end);
        for (iterT i = begin; i != end; i++) {
            this->data.push_back(BaseTensor<value_type>(*i));
        }
        this->set_dtype(dtype);
    }

    BaseTensor(value_type data, dt dtype = float32) {
        this->dim = 0;
        this->shape = Size();
        this->scaler = data;
        this->set_dtype(dtype);
    }

    BaseTensor(initializer_list<BaseTensor<value_type>> data, dt dtype = float32) {
        this->data = {};

        for (BaseTensor<value_type> i : data) {
            this->data.push_back(i);
        }
        BaseTensor<value_type> first = this->data[0];
        if (first.dim == 0) {
            this->shape = Size((int)this->data.size());
        } else {
            this->shape = Size(first.shape, (int)this->data.size());
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

    ostream &operator<<(ostream &os) {
        os << "[";
        for (value_type i : data) {
            os << i << " ";
        }
        os << ']' << endl;
        return os;
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

    template <class T>
    bool operator==(BaseTensor<T> ts) {
        if (this->dim != ts.dim) {
            return false;
        }
        if (this->dim == 0) {
            return this->scaler == ts.scaler;
        }
        if (this->shape != ts.shape) {
            return false;
        }
        for (int i = 0; i < this->data.size(); i++) {
            if (this->data[i] != ts.data[i]) {
                return false;
            }
        }
        return true;
    }
    template <class T>
    bool operator!=(BaseTensor<T> ts) {
        if (this->dim != ts.dim) {
            return true;
        }
        if (this->dim == 0) {
            return this->scaler != ts.scaler;
        }
        if (this->shape != ts.shape) {
            return true;
        }
        for (int i = 0; i < this->data.size(); i++) {
            if (this->data[i] != ts.data[i]) {
                return true;
            }
        }
        return false;
    }

    BaseTensor<value_type> copy() {
        if (this->dim == 0) {
            return {BaseTensor<value_type>(this->scaler)};
        } else {
            BaseTensor<value_type> newT;
            for (BaseTensor<value_type> &ts : this->data) {
                newT.data.push_back(ts.copy());
            }
            newT.dim = this->dim;
            newT.shape = this->shape;
            return newT;
        }
    }

    void set_dtype(dt dtype) {
        if (this->dtype == dtype) {
            return;
        }
        this->dtype = dtype;

        if (this->dim == 0) {
            switch (dtype) {
                case int8:
                    this->scaler = (int8_t)this->scaler;
                    this->dtype = int8;
                    break;
                case float32:
                    this->scaler = (float)this->scaler;
                    this->dtype = float32;
                    break;
            }
        } else {
            for (BaseTensor<value_type> &ts : this->data) {
                ts.set_dtype(dtype);
            }
        }
        this->dtype = dtype;
    }

    string type() {
        switch (this->dtype) {
            case int8:
                return "int8";
            case float32:
                return "float32";
        }
    }

    void *data_ptr() {
        if (this->dim == 0) {
            return &this->scaler;
        } else {
            return &this->data;
        }
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