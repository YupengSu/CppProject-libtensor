#include <cassert>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

using namespace std;

namespace ts {
enum dt { int8, float32 };

class Size {
   public:
    int dim;
    vector<int> shape;
    ostream &operator<<(ostream &os) {
        os << "Tensor_Shape: (";
        for (int i = 0; i < shape.size(); i++) {
            os << shape[i];
            if (i != shape.size() - 1) {
                os << ", ";
            }
        }
        os << ')';
        return os;
    }
    friend ostream &operator<<(ostream &os, Size sz) {
        os << "Tensor_Shape: (";
        for (int i = 0; i < sz.shape.size(); i++) {
            os << sz.shape[i];
            if (i != sz.shape.size() - 1) {
                os << ", ";
            }
        }
        os << ')';
        return os;
    }

    bool operator==(Size sz) {
        if (dim != sz.dim) {
            return false;
        }
        for (int i = 0; i < dim; i++) {
            if (shape[i] != sz.shape[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(Size sz) {
        if (dim != sz.dim) {
            return true;
        }
        for (int i = 0; i < dim; i++) {
            if (shape[i] != sz.shape[i]) {
                return true;
            }
        }
        return false;
    }

    void operator=(Size sz) {
        dim = sz.dim;
        shape = sz.shape;
    }

    int operator[](int index) {
        assert(index < dim);
        return shape[index];
    }

    Size() : shape({0}), dim(0) {}

    Size(int len) : dim(1) { shape = {len}; }
    Size(vector<int> shape) : dim(shape.size()), shape(shape) {}
    Size(Size old, int new_dim) : dim(old.dim + 1) {
        shape = vector<int>(old.shape);
        shape.insert(shape.begin(), new_dim);
    }
};

template <class value_type = float>
class Tensor {
   public:
    int dim;
    int my_dim;
    Size shape;  // Using shared_ptr with arrays
    dt dtype;

    vector<Tensor<value_type>> data;
    value_type scaler;

    ~Tensor() {}

    Tensor() {
        this->dim = 0;

        this->shape = Size();
        this->data = {};
    }
    Tensor &operator=(int data) {
        assert(this->dim == 0);
        this->scaler = data;
        return *this;
    }

    Tensor &operator[](int index) {
        assert(this->dim > 0);
        assert(index < this->shape.shape[0]);
        // if (this->dim == 1) {
        //     return this->data[index];
        // } else {
        //     Tensor<value_type> newT;
        //     newT.dim = this->dim - 1;
        //     newT.shape =
        //         Size(this->shape.shape.begin() + 1, this->shape.shape.end());
        //     for (Tensor<value_type> ts : this->data) {
        //         newT.data.push_back(ts[index]);
        //     }
        //     return newT;
        // }
        return this->data[index];
    }

    template <class iterT>
    Tensor(iterT begin, iterT end, dt dtype = float32) {
        this->dim = 1;
        // this->shape = Size((int)ts.size());
        this->shape = Size(begin - end);
        for (iterT i = begin; i != end; i++) {
            this->data.push_back(Tensor<value_type>(*i));
        }
        this->update_dtype(dtype);
    }

    Tensor(value_type data, dt dtype = float32) {
        this->dim = 0;
        this->shape = Size();
        this->scaler = data;
        this->update_dtype(dtype);
    }

    Tensor(initializer_list<Tensor<value_type>> data, dt dtype = float32) {
        this->data = {};

        for (Tensor<value_type> i : data) {
            this->data.push_back(i);
        }
        Tensor<value_type> first = this->data[0];
        if (first.dim == 0) {
            this->shape = Size((int)this->data.size());
        } else {
            this->shape = Size(first.shape, (int)this->data.size());
        }
        this->dim = first.dim + 1;

        for (Tensor ts : data) {
            assert(first.dim == ts.dim);
            for (int i = 0; i < this->dim; i++) {
                assert(first.shape.shape[i] == ts.shape.shape[i]);
            }
        }
        this->update_dtype(dtype);
    }

    ostream &operator<<(ostream &os) {
        os << "[";
        for (value_type i : data) {
            os << i << " ";
        }
        os << ']' << endl;
        return os;
    }

    friend ostream &operator<<(ostream &os, Tensor<value_type> tsr) {
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
    bool operator==(Tensor<T> ts) {
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
    bool operator!=(Tensor<T> ts) {
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

    Tensor<value_type> copy() {
        if (this->dim == 0) {
            return {Tensor<value_type>(this->scaler)};
        } else {
            Tensor<value_type> newT;
            for (Tensor<value_type> ts : this->data) {
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
        if (this->dim == 0) {
            switch (dtype) {
                case int8:
                    this->scaler = (int8_t)this->scaler;
                    break;
                case float32:
                    this->scaler = (float)this->scaler;
                    break;
            }
        } else {
            for (Tensor<value_type> ts : this->data) {
                ts.set_dtype(dtype);
            }
        }
    }
    void update_dtype(dt dtype) {
        switch (dtype) {
            case int8:
                this->set_dtype(int8);
                break;
            case float32:
                this->set_dtype(float32);
                break;
        }
    }
};

}  // namespace ts