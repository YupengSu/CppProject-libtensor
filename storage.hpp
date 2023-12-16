#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
using namespace std;

namespace ts {
using data_t = float;
class Storage {
   private:
    struct Vdata {
        // size_t version_;
        data_t data_[1];
    };

   public:
    data_t* dp;
    size_t size;
    shared_ptr<Vdata> bp;
    Storage() = default;
     Storage(size_t size) {
        this->size = size;
        this->bp = shared_ptr<Vdata>(new Vdata[size]);
        this->dp = bp->data_;
    }

    Storage(const Storage& other, size_t offset) {
        this->size = other.size;
        this->bp = other.bp;
        this->dp = other.dp + offset;
    }

    Storage(size_t size, data_t val) : Storage(size) {
        for (int i = 0; i < size; i++) dp[i] = val;
    }

    Storage(const data_t* data, size_t size) : Storage(size) {
        for (int i = 0; i < size; i++) dp[i] = data[i];
    }
    Storage(const Storage& other) = default;
    //  Storage(Storage&& other) = default;
    ~Storage() = default;
    
    // Storage& operator=(const Storage& other) = delete;

    data_t operator[](size_t idx) const { return dp[idx]; }
    data_t& operator[](size_t idx) { return dp[idx]; }

    size_t offset() { return dp - bp->data_; }
};

}  // namespace ts