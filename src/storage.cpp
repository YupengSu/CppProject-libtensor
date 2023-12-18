#include "storage.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "config.hpp"
using namespace std;

using namespace ts;

Storage::Storage() = default;
Storage::Storage(size_t size) {
    this->size = size;
    this->bp = shared_ptr<Vdata>(new Vdata[size]);
    this->dp = bp->data_;
}

Storage::Storage(const Storage& other, size_t offset) {
    this->size = other.size;
    this->bp = other.bp;
    this->dp = other.dp + offset;
}

Storage::Storage(data_t val, size_t size, dt dtype) : Storage(size) {
    for (int i = 0; i < size; i++) dp[i] = val;
    for (int i = 0; i < size; i++) dp[i].set_dtype(dtype);
    this->dtype = dtype;
}

Storage::Storage(const data_t* data, size_t size, dt dtype) : Storage(size) {
    for (int i = 0; i < size; i++) dp[i] = data[i];
    for (int i = 0; i < size; i++) dp[i].set_dtype(dtype);
    this->dtype = dtype;
}
Storage::Storage(const Storage& other) = default;
//  Storage(Storage&& other) = default;
Storage::~Storage() = default;

// Storage& operator=(const Storage& other) = delete;

data_t Storage::operator[](size_t idx) const { return dp[idx]; }
data_t& Storage::operator[](size_t idx) { return dp[idx]; }

size_t Storage::offset() { return dp - bp->data_; }
data_t rand_data_t(dt dtype) {
    switch (dtype) {
        case dt::float32:
           return(float)random() / RAND_MAX;
            break;
        case dt::int32:
           return(int)random() % INT_MAX;
            break;
        case dt::int8:
           return((int)random()) % 256;
            break;
        case dt::float64:
           return(double)random() / RAND_MAX;
            break;

        case dt::bool8:
           return(double)random() / RAND_MAX > 0.5;
            break;
        default:
            throw std::invalid_argument("Invalid dtype");
    }
}