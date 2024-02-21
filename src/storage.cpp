#include "storage.hpp"

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>

#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
using namespace std;

namespace ts {

Storage::Storage() = default;
Storage::Storage(size_t size, dev device) {
    this->size = size;
    if (device == dev::cpu)
        this->bp = shared_ptr<data_t>(new data_t[size]);
    else {
        void* tmp;
        c_cudaMalloc(&tmp, size * sizeof(data_t));
        // cudaMalloc(&tmp, size * sizeof(data_t));
        this->bp = shared_ptr<data_t>((data_t*)tmp, c_cudaFree);
    }
    this->dp = bp.get();
    this->device = device;
}

Storage::Storage(const Storage& other, size_t offset) {
    this->size = other.size;
    this->bp = other.bp;
    this->dp = other.dp + offset;
    this->device = other.device;
}

Storage::Storage(data_t val, size_t size, dt dtype, dev device)
    : Storage(size, device) {
    vector<data_t> tmp(size);
    for (int i = 0; i < size; i++) tmp[i] = val;
    for (int i = 0; i < size; i++) tmp[i].set_dtype(dtype);
    this->dtype = dtype;
    this->device = device;

    if (device == dev::cpu) {
        memcpy(this->dp, tmp.data(), size * sizeof(data_t));
    } else {
        c_cudaMemcpy(this->dp, tmp.data(), size * sizeof(data_t),
                     c_cudaMemcpyHostToDevice);
    }
}

Storage::Storage(const data_t* data, size_t size, dt dtype, dev device)
    : Storage(size, device) {
    vector<data_t> tmp(size);
    for (int i = 0; i < size; i++) tmp[i] = data[i];
    for (int i = 0; i < size; i++) tmp[i].set_dtype(dtype);
    this->dtype = dtype;
    this->device = device;

    if (device == dev::cpu) {
        memcpy(this->dp, tmp.data(), size * sizeof(data_t));
    } else {
        c_cudaMemcpy(this->dp, tmp.data(), size * sizeof(data_t),
                     c_cudaMemcpyHostToDevice);
    }
}
Storage::Storage(const Storage& other) = default;
//  Storage(Storage&& other) = default;
Storage::~Storage() = default;
// Storage& operator=(const Storage& other) = delete;

// data_t Storage::operator[](size_t idx)  {
//     if (this->device == dev::cpu) return dp[idx];
//     return item{*this, idx};
// }
data_t& Storage::operator[](size_t idx) {
    return dp[idx];
}
data_t Storage::operator[](size_t idx) const { return dp[idx]; }

size_t Storage::offset() { return dp - bp.get(); }
data_t rand_data_t(dt dtype) {
    switch (dtype) {
        case dt::float32:
            return (float)random() / (float)RAND_MAX;
            break;
        case dt::int32:
            return (int)random() % INT_MAX;
            break;
        case dt::int8:
            return ((int)random()) % 256;
            break;
        case dt::float64:
            return (double)random() / RAND_MAX;
            break;

        case dt::bool8:
            return (double)random() / RAND_MAX > 0.5;
            break;
        default:
            throw std::invalid_argument("Invalid dtype");
    }
}

void set_dtype(Storage st, dt dtype) {
    for (int i = 0; i < st.size; i++) st.dp[i].dtype = dtype;
}

};  // namespace ts