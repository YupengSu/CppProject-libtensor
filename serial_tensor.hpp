#pragma once
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include "base_tensor.hpp"
#include "size.hpp"
#include "storage.hpp"

using namespace std;

namespace ts {
using data_t = float;
class Tensor {
   public:
    Storage data;
    int ndim;
    Size shape;
    vector<int> stride;
    size_t offset;
    dt dtype;
    Tensor();
    Tensor(int i_dim);

    ~Tensor(){};
    Tensor(const vector<data_t> &i_data, const vector<int> &i_shape = {},
           dt dtype = float32);

    Tensor(const Storage &i_data, const Size &i_shape,
           const vector<int> i_stride, dt dtype);

    void init_stride();

    // data_t &operator[](int index);
    // data_t &operator[](initializer_list<int> inds);

    // ostream &operator<<(ostream &os);
    friend ostream &operator<<(ostream &os, Tensor t);

    Tensor operator()(int index);
    Tensor operator()(int index, pair<int, int> range);
    // data_t &operator[](initializer_list<size_t> inds);
    // data_t operator[](initializer_list<size_t> inds) const;
    data_t &operator[](vector<size_t> inds);
    data_t operator[](vector<size_t> inds) const;
    size_t get_dim() const;
    size_t size(int i) const;

    vector<data_t> get_data();

    Tensor slice(int idx, int dim = 0);
    void *data_ptr();

   private:
    int get_size(vector<int> shape);
};

Tensor tensor(BaseTensor<> bt);
Tensor rand(Size sz);
Tensor zeros(Size sz);
Tensor ones(Size sz);
Tensor full(Size sz, data_t val);
Tensor eye(Size sz);
Tensor cat(vector<Tensor> tensors, int dim);
Tensor tile(Tensor t, vector<int> reps);
}  // namespace ts
