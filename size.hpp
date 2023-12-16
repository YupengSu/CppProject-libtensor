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
    int ndim;
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

    int &operator[](int index) {
        assert(index < ndim);
        return shape[index];
    }
    int operator[](int index) const {
        assert(index < ndim);
        return shape[index];
    }

    Size() : shape({0}), ndim(0) {}

    Size(int len) : ndim(1) { shape = {len}; }
    Size(vector<int> shape) : ndim(shape.size()), shape(shape) {}
    Size(initializer_list<int> shape) : ndim(shape.size()), shape(shape) {}
    Size(const Size &other, size_t skip) : shape(other.ndim - 1) {
        int i = 0;
        for (; i < skip; ++i) shape[i] = other.shape[i];
        for (; i < shape.size(); ++i) shape[i] = other.shape[i + 1];
        ndim = shape.size();
    }

    Size(const Size &sz) : ndim(sz.ndim), shape(sz.shape) {}

    size_t size(int i) const {
        assert(i < ndim);
        return shape[i];
    }

    size_t size() const {
        size_t sz = 1;
        for (int i = 0; i < ndim; i++) {
            sz *= shape[i];
        }
        return sz;
    }
};
}  // namespace ts