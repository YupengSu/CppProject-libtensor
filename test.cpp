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
    int ndim;
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
        if (ndim != sz.ndim) {
            return false;
        }
        for (int i = 0; i < ndim; i++) {
            if (shape[i] != sz.shape[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(Size sz) {
        if (ndim != sz.ndim) {
            return true;
        }
        for (int i = 0; i < ndim; i++) {
            if (shape[i] != sz.shape[i]) {
                return true;
            }
        }
        return false;
    }

    void operator=(Size sz) {
        ndim = sz.ndim;
        shape = sz.shape;
    }

    int operator[](int index) {
        assert(index < ndim);
        return shape[index];
    }

    Size() : shape({0}), ndim(0) {}

    Size(int len) : ndim(1) { shape = {len}; }
    Size(vector<int> shape) : ndim(shape.size()), shape(shape) {}
    Size(Size old, int new_dim) : ndim(old.ndim + 1) {
        shape = vector<int>(old.shape);
        shape.insert(shape.begin(), new_dim);
    }
};

class FloatTensor {
   public:
    int dim;
    int my_dim;
    Size shape;  // Using shared_ptr with arrays
    dt dtype;

    vector<FloatTensor> data;
    float scaler;

    ~FloatTensor() {}

    FloatTensor() {
        this->dim = 0;

        this->shape = Size();
        this->data = {};
    }
    FloatTensor &operator=(int data) {
        assert(this->dim == 0);
        this->scaler = data;
        return *this;
    }

    FloatTensor &operator[](int index) {
        assert(this->dim > 0);
        assert(index < this->shape.shape[0]);
        // if (this->dim == 1) {
        //     return this->data[index];
        // } else {
        //     Tensor newT;
        //     newT.dim = this->dim - 1;
        //     newT.shape =
        //         Size(this->shape.shape.begin() + 1, this->shape.shape.end());
        //     for (Tensor ts : this->data) {
        //         newT.data.push_back(ts[index]);
        //     }
        //     return newT;
        // }
        return this->data[index];
    }

    template <class iterT>
    FloatTensor(iterT begin, iterT end, dt dtype = float32) {
        this->dim = 1;
        // this->shape = Size((int)ts.size());
        this->shape = Size(begin - end);
        for (iterT i = begin; i != end; i++) {
            this->data.push_back(FloatTensor(*i));
        }
        this->update_dtype(dtype);
    }

    FloatTensor(float data, dt dtype = float32) {
        this->dim = 0;
        this->shape = Size();
        this->scaler = data;
        this->update_dtype(dtype);
    }

    FloatTensor(initializer_list<FloatTensor> data, dt dtype = float32) {
        this->data = {};

        for (FloatTensor i : data) {
            this->data.push_back(i);
        }
        FloatTensor first = this->data[0];
        if (first.dim == 0) {
            this->shape = Size((int)this->data.size());
        } else {
            this->shape = Size(first.shape, (int)this->data.size());
        }
        this->dim = first.dim + 1;

        for (FloatTensor ts : data) {
            assert(first.dim == ts.dim);
            for (int i = 0; i < this->dim; i++) {
                assert(first.shape.shape[i] == ts.shape.shape[i]);
            }
        }
        this->update_dtype(dtype);
    }

    ostream &operator<<(ostream &os) {
        os << "[";
        for (auto i : data) {
            os << i << " ";
        }
        os << ']' << endl;
        return os;
    }

    friend ostream &operator<<(ostream &os, FloatTensor tsr) {
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

    // template <class T>
    // bool operator==(Tensor ts) {
    //     if (this->dim != ts.dim) {
    //         return false;
    //     }
    //     if (this->dim == 0) {
    //         return this->scaler == ts.scaler;
    //     }
    //     if (this->shape != ts.shape) {
    //         return false;
    //     }
    //     for (int i = 0; i < this->data.size(); i++) {
    //         if (this->data[i] != ts.data[i]) {
    //             return false;
    //         }
    //     }
    //     return true;
    // }
    // template <class T>
    // bool operator!=(Tensor ts) {
    //     if (this->dim != ts.dim) {
    //         return true;
    //     }
    //     if (this->dim == 0) {
    //         return this->scaler != ts.scaler;
    //     }
    //     if (this->shape != ts.shape) {
    //         return true;
    //     }
    //     for (int i = 0; i < this->data.size(); i++) {
    //         if (this->data[i] != ts.data[i]) {
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    FloatTensor copy() {
        if (this->dim == 0) {
            return {FloatTensor(this->scaler)};
        } else {
            FloatTensor newT;
            for (FloatTensor ts : this->data) {
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
            for (FloatTensor ts : this->data) {
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

class IntTensor {
   public:
    int dim;
    int my_dim;
    Size shape;  // Using shared_ptr with arrays
    dt dtype;

    vector<IntTensor> data;
    int scaler;

    ~IntTensor() {}

    IntTensor() {
        this->dim = 0;

        this->shape = Size();
        this->data = {};
    }
    IntTensor &operator=(int data) {
        assert(this->dim == 0);
        this->scaler = data;
        return *this;
    }

    IntTensor &operator[](int index) {
        assert(this->dim > 0);
        assert(index < this->shape.shape[0]);
        // if (this->dim == 1) {
        //     return this->data[index];
        // } else {
        //     Tensor newT;
        //     newT.dim = this->dim - 1;
        //     newT.shape =
        //         Size(this->shape.shape.begin() + 1, this->shape.shape.end());
        //     for (Tensor ts : this->data) {
        //         newT.data.push_back(ts[index]);
        //     }
        //     return newT;
        // }
        return this->data[index];
    }

    template <class iterT>
    IntTensor(iterT begin, iterT end, dt dtype = float32) {
        this->dim = 1;
        // this->shape = Size((int)ts.size());
        this->shape = Size(begin - end);
        for (iterT i = begin; i != end; i++) {
            this->data.push_back(IntTensor(*i));
        }
        this->update_dtype(dtype);
    }

    IntTensor(int data, dt dtype = float32) {
        this->dim = 0;
        this->shape = Size();
        this->scaler = data;
        this->update_dtype(dtype);
    }

    IntTensor(initializer_list<IntTensor> data, dt dtype = float32) {
        this->data = {};

        for (IntTensor i : data) {
            this->data.push_back(i);
        }
        IntTensor first = this->data[0];
        if (first.dim == 0) {
            this->shape = Size((int)this->data.size());
        } else {
            this->shape = Size(first.shape, (int)this->data.size());
        }
        this->dim = first.dim + 1;

        for (IntTensor ts : data) {
            assert(first.dim == ts.dim);
            for (int i = 0; i < this->dim; i++) {
                assert(first.shape.shape[i] == ts.shape.shape[i]);
            }
        }
        this->update_dtype(dtype);
    }

    ostream &operator<<(ostream &os) {
        os << "[";
        for (auto i : data) {
            os << i << " ";
        }
        os << ']' << endl;
        return os;
    }

    friend ostream &operator<<(ostream &os, IntTensor tsr) {
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

    // template <class T>
    // bool operator==(Tensor ts) {
    //     if (this->dim != ts.dim) {
    //         return false;
    //     }
    //     if (this->dim == 0) {
    //         return this->scaler == ts.scaler;
    //     }
    //     if (this->shape != ts.shape) {
    //         return false;
    //     }
    //     for (int i = 0; i < this->data.size(); i++) {
    //         if (this->data[i] != ts.data[i]) {
    //             return false;
    //         }
    //     }
    //     return true;
    // }
    // template <class T>
    // bool operator!=(Tensor ts) {
    //     if (this->dim != ts.dim) {
    //         return true;
    //     }
    //     if (this->dim == 0) {
    //         return this->scaler != ts.scaler;
    //     }
    //     if (this->shape != ts.shape) {
    //         return true;
    //     }
    //     for (int i = 0; i < this->data.size(); i++) {
    //         if (this->data[i] != ts.data[i]) {
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    IntTensor copy() {
        if (this->dim == 0) {
            return {IntTensor(this->scaler)};
        } else {
            IntTensor newT;
            for (IntTensor ts : this->data) {
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
                    this->scaler = (int)this->scaler;
                    break;
            }
        } else {
            for (IntTensor ts : this->data) {
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

class Tensor {
   public:
    void *tensors;
};

Tensor tensor(initializer_list<initializer_list<float>> data, dt dtype = float32) {  // 3D Tensor
    Tensor ts;
    
    return ts;
}
Tensor tensor(initializer_list<initializer_list<initializer_list<float>>>
                  data) {  // 3D Tensor
    Tensor ts;
    return ts;
}
}  // namespace ts