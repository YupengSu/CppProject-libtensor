#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include "config.hpp"
using namespace std;

namespace ts {
    
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
    dt dtype;
    Storage();
    Storage(size_t size) ;
    Storage(const Storage& other, size_t offset);

    Storage(data_t val, size_t size, dt dtype=DEFAULT_DTYPE);

    Storage(const data_t* data, size_t size, dt dtype=DEFAULT_DTYPE);
    Storage(const Storage& other);
    ~Storage() ;

    // Storage& operator=(const Storage& other) = delete;

    data_t operator[](size_t idx) const ;
    data_t& operator[](size_t idx);

    size_t offset();
};

data_t rand_data_t(dt dtype);

}  // namespace ts