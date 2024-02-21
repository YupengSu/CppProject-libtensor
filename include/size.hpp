#pragma once
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>
using namespace std;

namespace ts {
class Size {
   public:
    int ndim;
    vector<int> shape;
    ostream &operator<<(ostream &os);
    friend ostream &operator<<(ostream &os, Size sz);

    bool operator==(Size sz);

    bool operator!=(Size sz);

    // void operator=(Size sz);
    Size& operator=(const Size& sz);

    int &operator[](int index);
    int operator[](int index) const;

    Size();

    Size(int len);
    Size(vector<int> shape);
    Size(initializer_list<int> shape);
    Size(const Size &other, size_t skip);

    Size(const Size &sz);
    size_t size(int i) const;

    size_t data_len() const;
    size_t inner_size(int dim) const;
    size_t outer_size(int dim) const;
};
}  // namespace ts