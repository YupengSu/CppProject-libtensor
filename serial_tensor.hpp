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
    // Tensor(int i_dim);

    ~Tensor(){};
    Tensor(const vector<data_t> &i_data, const vector<int> &i_shape = {},
           dt dtype = float32);

    Tensor(const Storage &i_data, const Size &i_shape,
           const vector<int> i_stride, dt dtype);

    friend ostream &operator<<(ostream &os, Tensor t);
    Tensor operator()(int index);
    Tensor operator()(int index, pair<int, int> range);
    data_t &operator()(vector<size_t> inds);
    data_t operator()(vector<size_t> inds) const;

    data_t &operator[](vector<size_t> inds);
    data_t operator[](vector<size_t> inds) const;

    Tensor &operator=(BaseTensor<> bt);
    Tensor &operator=(Tensor bt);

    size_t get_dim() const;
    size_t size(int i) const;
    vector<data_t> get_data();
    Tensor slice(int idx, int dim = 0);
    Tensor permute(vector<int> dims);
    Tensor transpose(int dim1, int dim2);
    Tensor view(vector<int> shape);
    void *data_ptr();

   private:
    int get_size(vector<int> shape);
};
vector<int>init_stride(vector<int> shape);

Tensor tensor(BaseTensor<> bt);
Tensor rand(Size sz);
Tensor zeros(Size sz);
Tensor ones(Size sz);
Tensor full(Size sz, data_t val);
Tensor eye(Size sz);
Tensor cat(vector<Tensor> tensors, int dim);
Tensor tile(Tensor t, vector<int> reps);
Tensor permute(Tensor t, vector<int> dims);
Tensor transpose(Tensor t, int dim1, int dim2);
Tensor view(Tensor t, vector<int> shape);
}  // namespace ts
