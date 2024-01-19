#pragma once
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <vector>

#include "base_tensor.hpp"
#include "config.hpp"
#include "data_type.cuh"
#include "size.hpp"
#include "storage.hpp"

using namespace std;
namespace ts {
class Tensor;
// size_t get_data_idx(size_t index, vector<int> shape, vector<int> stride,
// vector<int> origin_stride);
size_t get_data_idx(size_t index, Tensor t);
class Tensor {
   public:
    Storage data;
    int ndim = 0;
    Size shape;
    vector<int> stride;
    vector<int> origin_stride;
    size_t offset = 0;
    dt dtype;
    dev device;
    Tensor();
    Tensor(const Tensor& other) = default;
    // Tensor(int i_dim);
    ~Tensor(){};
    Tensor(const vector<data_t>& i_data, const vector<int>& i_shape = {},
           dt dtype = DEFAULT_DTYPE, dev device = DEFAULT_DEVICE);

    Tensor(const Storage& i_data, const Size& i_shape,
           const vector<int> i_stride, dt dtype = DEFAULT_DTYPE,
           dev device = DEFAULT_DEVICE);

    Tensor to(dev device);
    Tensor cuda();
    Tensor cpu();
    Tensor clone();
    data_t& get(size_t index);
    data_t get(size_t index) const;

    friend ostream& operator<<(ostream& os, Tensor t);
    Tensor operator()(int index);
    Tensor operator()(int index, pair<int, int> range);

    data_t& operator()(vector<size_t> inds);
    data_t operator()(vector<size_t> inds) const;

    data_t& operator[](vector<size_t> inds);
    data_t operator[](vector<size_t> inds) const;

    Tensor operator[](size_t index);
    Tensor& operator=(BaseTensor<> bt);
    // Tensor &operator=(int val);
    Tensor& operator=(double val);
    // Tensor &operator=(bool val);
    // Tensor &operator=(Tensor bt);

    size_t get_dim() const;
    size_t size(int i) const;
    vector<data_t> get_serial_data() const;
    Tensor slice(int idx, int dim = 0);
    Tensor slice(int idx, int dim = 0) const;
    Tensor permute(vector<int> dims);
    Tensor transpose(int dim1, int dim2);
    Tensor view(vector<int> shape);
    void* data_ptr();
    size_t size() const;
    string type() const;
    bool is_contiguous() const;
    void info(string name = "Tensor") const;

    Tensor operator+(const Tensor& other);
    Tensor operator+(const data_t& other);
    Tensor operator-(const Tensor& other);
    Tensor operator-(const data_t& other);
    Tensor operator*(const Tensor& other);
    Tensor operator*(const data_t& other);
    Tensor operator/(const Tensor& other);
    Tensor operator/(const data_t& other);

    Tensor add(const Tensor& other);
    Tensor sub(const Tensor& other);
    Tensor mul(const Tensor& other);
    Tensor div(const Tensor& other);

    Tensor add(const data_t& other);
    Tensor sub(const data_t& other);
    Tensor mul(const data_t& other);
    Tensor div(const data_t& other);

    Tensor sum(int dim);
    Tensor mean(int dim);
    Tensor max(int dim);
    Tensor min(int dim);

    Tensor eq(const Tensor& other);
    Tensor ne(const Tensor& other);
    Tensor gt(const Tensor& other);
    Tensor ge(const Tensor& other);
    Tensor lt(const Tensor& other);
    Tensor le(const Tensor& other);

    Tensor operator==(const Tensor& other);
    Tensor operator!=(const Tensor& other);
    Tensor operator>(const Tensor& other);
    Tensor operator>=(const Tensor& other);
    Tensor operator<(const Tensor& other);
    Tensor operator<=(const Tensor& other);

   private:
    int get_size(vector<int> shape);
};
vector<int> init_stride(vector<int> shape);
Tensor tensor(BaseTensor<> bt, dt dtype = DEFAULT_DTYPE);

Tensor rand(Size sz, dt dtype = DEFAULT_DTYPE);
Tensor zeros(Size sz, dt dtype = DEFAULT_DTYPE);
Tensor ones(Size sz, dt dtype = DEFAULT_DTYPE);
Tensor full(Size sz, data_t val, dt dtype = DEFAULT_DTYPE);
Tensor eye(Size sz, dt dtype = DEFAULT_DTYPE);

Tensor cat(vector<Tensor> tensors, int dim);
Tensor tile(Tensor t, vector<int> reps);
Tensor permute(Tensor t, vector<int> dims);
Tensor transpose(Tensor t, int dim1, int dim2);
Tensor view(Tensor t, vector<int> shape);

Tensor add(const Tensor& t1, const data_t& t2);
Tensor sub(const Tensor& t1, const data_t& t2);
Tensor mul(const Tensor& t1, const data_t& t2);
Tensor div(const Tensor& t1, const data_t& t2);

Tensor add(const Tensor& t1, const Tensor& t2);
Tensor sub(const Tensor& t1, const Tensor& t2);
Tensor mul(const Tensor& t1, const Tensor& t2);
Tensor div(const Tensor& t1, const Tensor& t2);

Tensor log(const Tensor& t);

Tensor sum(const Tensor& t, int dim = -1);
Tensor mean(const Tensor& t, int dim = -1);
Tensor max(const Tensor& t, int dim = -1);
Tensor min(const Tensor& t, int dim = -1);

// comparison
Tensor eq(const Tensor& t1, const Tensor& t2);
Tensor ne(const Tensor& t1, const Tensor& t2);
Tensor gt(const Tensor& t1, const Tensor& t2);
Tensor ge(const Tensor& t1, const Tensor& t2);
Tensor lt(const Tensor& t1, const Tensor& t2);
Tensor le(const Tensor& t1, const Tensor& t2);

// other
Tensor einsum(string eq, vector<Tensor> tensors);

// save and load
void save(const Tensor& t, string filename);
Tensor load(string filename);

}  // namespace ts
