#include "size.hpp"

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>
using namespace std;

using namespace ts;

ostream &Size::operator<<(ostream &os) {
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
ostream &operator<<(ostream &os, Size sz) {
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

bool Size::operator==(Size sz) {
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

bool Size::operator!=(Size sz) {
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

void Size::operator=(Size sz) {
    ndim = sz.ndim;
    shape = sz.shape;
}

int &Size::operator[](int index) {
    assert(index < ndim);
    return shape[index];
}
int Size::operator[](int index) const {
    assert(index < ndim);
    return shape[index];
}

Size::Size() : shape({0}), ndim(0) {}

Size::Size(int len) : ndim(1) { shape = {len}; }
Size::Size(vector<int> shape) : ndim(shape.size()), shape(shape) {}
Size::Size(initializer_list<int> shape) : ndim(shape.size()), shape(shape) {}
Size::Size(const Size &other, size_t skip) : shape(other.ndim - 1) {
    int i = 0;
    for (; i < skip; ++i) shape[i] = other.shape[i];
    for (; i < shape.size(); ++i) shape[i] = other.shape[i + 1];
    ndim = shape.size();
}

Size::Size(const Size &sz) : ndim(sz.ndim), shape(sz.shape) {}

size_t Size::size(int i) const {
    assert(i < ndim);
    return shape[i];
}

size_t Size::size() const {
    size_t sz = 1;
    for (int i = 0; i < ndim; i++) {
        sz *= shape[i];
    }
    return sz;
}
