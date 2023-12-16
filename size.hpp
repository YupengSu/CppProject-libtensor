#pragma once
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>
using namespace std;

namespace ts {
enum dt { int8, float32 };
class Size {
   public:
    int dim;
    vector<int> shape;
    ostream &operator<<(ostream &os) {
        os << "(";
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
        os << "(";
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

    int &operator[](int index) {
        assert(index < dim);
        return shape[index];
    }

    Size() : shape({0}), dim(0) {}

    Size(int len) : dim(1) { shape = {len}; }
    Size(vector<int> shape) : dim(shape.size()), shape(shape) {}
    Size(initializer_list<int> shape) : dim(shape.size()), shape(shape) {}

    Size(Size old, int new_dim) : dim(old.dim + 1) {
        shape = vector<int>(old.shape);
        shape.insert(shape.begin(), new_dim);
    }

    size_t size(int i) const {
        assert(i < dim);
        return shape[i];
    }

    size_t size() const {
        size_t sz = 1;
        for (int i = 0; i < dim; i++) {
            sz *= shape[i];
        }
        return sz;
    }
};
}  // namespace ts