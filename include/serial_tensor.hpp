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
class TensorImpl;
// size_t get_data_idx(size_t index, vector<int> shape, vector<int> stride,
// vector<int> origin_stride);
size_t get_data_idx(size_t index, TensorImpl t);
class TensorImpl {
   public:
    Storage data;
    int ndim = 0;
    Size shape;
    vector<int> stride;
    vector<int> origin_stride;

    // shared_ptr<int> cuda_stride;
    // shared_ptr<int> cuda_shape;
    // shared_ptr<int> cuda_origin_stride;

    size_t offset = 0;
    dt dtype;
    dev device;
    TensorImpl();
    TensorImpl(const TensorImpl& other) = default;
    TensorImpl(const vector<data_t>& i_data, const vector<int>& i_shape = {},
           dt dtype = DEFAULT_DTYPE, dev device = DEFAULT_DEVICE);

    TensorImpl(const Storage& i_data, const Size& i_shape,
           const vector<int> i_stride, dt dtype = DEFAULT_DTYPE,
           dev device = DEFAULT_DEVICE);

    TensorImpl to(dev device) const;
    TensorImpl cuda() const;
    TensorImpl cpu() const;
    TensorImpl clone() const;
    data_t& get(size_t index);
    data_t get(size_t index) const;

    friend ostream& operator<<(ostream& os, TensorImpl t);
    TensorImpl operator()(int index) const;
    TensorImpl operator()(int index, pair<int, int> range) const;

    data_t& operator()(vector<size_t> inds);
    data_t operator()(vector<size_t> inds) const;

    data_t& operator[](vector<size_t> inds);
    data_t operator[](vector<size_t> inds) const;

    TensorImpl operator[](size_t index) const;
    TensorImpl& operator=(BaseTensor<> bt);
    TensorImpl& operator=(double val);

    size_t get_dim() const;
    size_t size(int i) const;
    vector<data_t> get_serial_data() const;
    TensorImpl slice(int idx, int dim = 0) const;
    TensorImpl permute(vector<int> dims) const;
    TensorImpl transpose(int dim1, int dim2) const;
    TensorImpl view(vector<int> shape) const;

    void* data_ptr() const;
    size_t size() const;
    string type() const;
    bool is_contiguous() const;
    void info(string name = "Tensor") const;

    TensorImpl operator+(const TensorImpl& other) const;
    TensorImpl operator+(const data_t& other) const;
    TensorImpl operator-(const TensorImpl& other) const;
    TensorImpl operator-(const data_t& other) const;
    TensorImpl operator*(const TensorImpl& other) const;
    TensorImpl operator*(const data_t& other) const;
    TensorImpl operator/(const TensorImpl& other) const;
    TensorImpl operator/(const data_t& other) const;

    TensorImpl add(const TensorImpl& other) const;
    TensorImpl sub(const TensorImpl& other) const;
    TensorImpl mul(const TensorImpl& other) const;
    TensorImpl div(const TensorImpl& other) const;

    TensorImpl add(const data_t& other) const;
    TensorImpl sub(const data_t& other) const;
    TensorImpl mul(const data_t& other) const;
    TensorImpl div(const data_t& other) const;

    TensorImpl sum(int dim) const;
    TensorImpl mean(int dim) const;
    TensorImpl max(int dim) const;
    TensorImpl min(int dim) const;

    TensorImpl eq(const TensorImpl& other) const;
    TensorImpl ne(const TensorImpl& other) const;
    TensorImpl gt(const TensorImpl& other) const;
    TensorImpl ge(const TensorImpl& other) const;
    TensorImpl lt(const TensorImpl& other) const;
    TensorImpl le(const TensorImpl& other) const;

    TensorImpl operator==(const TensorImpl& other) const;
    TensorImpl operator!=(const TensorImpl& other) const;
    TensorImpl operator>(const TensorImpl& other) const;
    TensorImpl operator>=(const TensorImpl& other) const;
    TensorImpl operator<(const TensorImpl& other) const;
    TensorImpl operator<=(const TensorImpl& other) const;

   private:
    int get_size(vector<int> shape);
};
vector<int> init_stride(vector<int> shape);
TensorImpl tensor(BaseTensor<> bt, dt dtype = DEFAULT_DTYPE);

TensorImpl rand(Size sz, dt dtype = DEFAULT_DTYPE);
TensorImpl zeros(Size sz, dt dtype = DEFAULT_DTYPE);
TensorImpl ones(Size sz, dt dtype = DEFAULT_DTYPE);
TensorImpl full(Size sz, data_t val, dt dtype = DEFAULT_DTYPE);
TensorImpl eye(Size sz, dt dtype = DEFAULT_DTYPE);

TensorImpl cat(vector<TensorImpl> tensors, int dim);
TensorImpl tile(const TensorImpl & t, vector<int> reps);
TensorImpl permute(const TensorImpl & t, vector<int> dims);
TensorImpl transpose(const TensorImpl & t, int dim1, int dim2);
TensorImpl view( const TensorImpl & t, vector<int> shape);

TensorImpl add(const TensorImpl& t1, const data_t& t2);
TensorImpl sub(const TensorImpl& t1, const data_t& t2);
TensorImpl mul(const TensorImpl& t1, const data_t& t2);
TensorImpl div(const TensorImpl& t1, const data_t& t2);

TensorImpl add(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl sub(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl mul(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl div(const TensorImpl& t1, const TensorImpl& t2);

TensorImpl log(const TensorImpl& t);

TensorImpl sum(const TensorImpl& t, int dim = -1);
TensorImpl mean(const TensorImpl& t, int dim = -1);
TensorImpl max(const TensorImpl& t, int dim = -1);
TensorImpl min(const TensorImpl& t, int dim = -1);

// comparison
TensorImpl eq(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl ne(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl gt(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl ge(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl lt(const TensorImpl& t1, const TensorImpl& t2);
TensorImpl le(const TensorImpl& t1, const TensorImpl& t2);

// other
TensorImpl einsum(string eq, vector<TensorImpl> tensors);

// save and load
void save(const TensorImpl& t, string filename);
TensorImpl load(string filename);

}  // namespace ts
