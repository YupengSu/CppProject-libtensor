#include "storage.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
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

Storage::Storage(size_t size, data_t val) : Storage(size) {
    for (int i = 0; i < size; i++) dp[i] = val;
}

Storage::Storage(const data_t* data, size_t size) : Storage(size) {
    for (int i = 0; i < size; i++) dp[i] = data[i];
}
Storage::Storage(const Storage& other) = default;
//  Storage(Storage&& other) = default;
Storage::~Storage() = default;

// Storage& operator=(const Storage& other) = delete;

data_t Storage::operator[](size_t idx) const { return dp[idx]; }
data_t& Storage::operator[](size_t idx) { return dp[idx]; }

size_t Storage::offset() { return dp - bp->data_; }
