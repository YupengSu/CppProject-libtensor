#pragma once
#include <csignal>
#include <cstdint>
#include <ios>
#include <iostream>
#include <ostream>
#include <ratio>
#include <type_traits>

#include "config.hpp"

#define DEFAULT_DTYPE dt::float32
#define DEFAULT_DEVICE dev::cpu

using namespace std;
namespace ts {

typedef union {
    uint8_t tensor_int8;
    uint32_t tensor_int32;
    float tensor_float32;
    double tensor_float64;
    bool tensor_bool;
} data_tt;

class data_t {
   public:
    data_tt data;
    dt dtype = DEFAULT_DTYPE;

   public:
    data_t() = default;
    data_t(data_tt data) : data(data) {

    }
    data_t(const data_t& other) : data(other.data), dtype(other.dtype){

    }
    template <class T>
    data_t(T data, dt dtype) {
        this->dtype = dtype;
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 = (uint8_t)data;
                break;
            case dt::float32:
                this->data.tensor_float32 = (float)data;
                break;
            case dt::bool8:
                this->data.tensor_bool = (bool)data;
                break;
            case dt::int32:
                this->data.tensor_int32 = (int)data;
                break;
            case dt::float64:
                this->data.tensor_float64 = (double)data;
                break;
            default:
                break;
        }
    }
    data_t(int8_t data) {
        this->data.tensor_int8 = data;
        this->dtype = dt::int8;
    }
    data_t(float data) {
        this->data.tensor_float32 = data;
        this->dtype = dt::float32;
    }
    data_t(double data) {
        this->data.tensor_float64 = data;
        this->dtype = dt::float64;
    }
    data_t(int data) {
        this->data.tensor_int32 = data;
        this->dtype = dt::int32;
    }
    data_t(bool data) {
        this->data.tensor_bool = data;
        this->dtype = dt::bool8;
    }

    operator float() {
        switch (dtype) {
            case dt::int8:
                return float(data.tensor_int8);
                break;
            case dt::float32:
                return float(data.tensor_float32);
                break;
            case dt::bool8:
                return float(data.tensor_bool);
                break;
            case dt::int32:
                return float(data.tensor_int32);
                break;
            case dt::float64:
                return float(data.tensor_float64);
                break;
            default:
                break;
        }
        return float(data.tensor_float32);
    }

    operator double() {
        switch (dtype) {
            case dt::int8:
                return double(data.tensor_int8);
                break;
            case dt::float32:
                return double(data.tensor_float32);
                break;
            case dt::bool8:
                return double(data.tensor_bool);
                break;
            case dt::int32:
                return double(data.tensor_int32);
                break;
            case dt::float64:
                return double(data.tensor_float64);
                break;
            default:
                break;
        }
        return double(data.tensor_float64);
    }
    operator bool() {
        switch (dtype) {
            case dt::int8:
                return bool(data.tensor_int8);
                break;
            case dt::float32:
                return bool(data.tensor_float32);
                break;
            case dt::bool8:
                return bool(data.tensor_bool);
                break;
            case dt::int32:
                return bool(data.tensor_int32);
                break;
            case dt::float64:
                return bool(data.tensor_float64);
                break;
            default:
                break;
        }
        return bool(data.tensor_bool);
    }
    operator int() {
        switch (dtype) {
            case dt::int8:
                return int(data.tensor_int8);
                break;
            case dt::float32:
                return int(data.tensor_float32);
                break;
            case dt::bool8:
                return int(data.tensor_bool);
                break;
            case dt::int32:
                return int(data.tensor_int32);
                break;
            case dt::float64:
                return int(data.tensor_float64);
                break;
            default:
                break;
        }
        return int(data.tensor_int32);
    }
    operator int8_t() {
        switch (dtype) {
            case dt::int8:
                return int8_t(data.tensor_int8);
                break;
            case dt::float32:
                return int8_t(data.tensor_float32);
                break;
            case dt::bool8:
                return int8_t(data.tensor_bool);
                break;
            case dt::int32:
                return int8_t(data.tensor_int32);
                break;
            case dt::float64:
                return int8_t(data.tensor_float64);
                break;
            default:
                break;
        }
        return int8_t(data.tensor_int8);
    }

    data_t to_dt(dt target) {
        if (target == this->dtype) {
            return *this;
        }
        data_t res;
        switch (target) {
            case dt::int8:
                res.dtype = dt::int8;
                switch (dtype) {
                    case dt::int8:
                        res = (int8_t)this->data.tensor_int8;
                        break;
                    case dt::float32:
                        res = (int8_t)this->data.tensor_float32;
                        break;
                    case dt::bool8:
                        res = (int8_t)this->data.tensor_bool;
                        break;
                    case dt::int32:
                        res = (int8_t)this->data.tensor_int32;
                        break;
                    case dt::float64:
                        res = (int8_t)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case dt::float32:
                res.dtype = dt::float32;
                switch (dtype) {
                    case dt::int8:
                        res = (float)this->data.tensor_int8;
                        break;
                    case dt::float32:
                        res = (float)this->data.tensor_float32;
                        break;
                    case dt::bool8:
                        res = (float)this->data.tensor_bool;
                        break;
                    case dt::int32:
                        res = (float)this->data.tensor_int32;
                        break;
                    case dt::float64:
                        res = (float)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case dt::bool8:
                res.dtype = dt::bool8;
                switch (dtype) {
                    case dt::int8:
                        res = (bool)this->data.tensor_int8;
                        break;
                    case dt::float32:
                        res = (bool)this->data.tensor_float32;
                        break;
                    case dt::bool8:
                        res = (bool)this->data.tensor_bool;
                        break;
                    case dt::int32:
                        res = (bool)this->data.tensor_int32;
                        break;
                    case dt::float64:
                        res = (bool)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case dt::int32:
                res.dtype = dt::int32;
                switch (dtype) {
                    case dt::int8:
                        res = (int)this->data.tensor_int8;
                        break;
                    case dt::float32:
                        res = (int)this->data.tensor_float32;
                        break;
                    case dt::bool8:
                        res = (int)this->data.tensor_bool;
                        break;
                    case dt::int32:
                        res = (int)this->data.tensor_int32;
                        break;
                    case dt::float64:
                        res = (int)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case dt::float64:
                res.dtype = dt::float64;
                switch (dtype) {
                    case dt::int8:
                        res = (double)this->data.tensor_int8;
                        break;
                    case dt::float32:
                        res = (double)this->data.tensor_float32;
                        break;
                    case dt::bool8:
                        res = (double)this->data.tensor_bool;
                        break;
                    case dt::int32:
                        res = (double)this->data.tensor_int32;
                        break;
                    case dt::float64:
                        res = (double)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        return res;
    }

    void set_dtype(dt dtype) {
        *this = this->to_dt(dtype);
    }

    bool operator==(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 == data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 == data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool == data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 == data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 == data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    data_t operator+(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 + data.data.tensor_int8;
            case dt::float32:
                return this->data.tensor_float32 + data.data.tensor_float32;
            case dt::bool8:
                return this->data.tensor_bool + data.data.tensor_bool;
            case dt::int32:
                return data_t(this->data.tensor_int32 + data.data.tensor_int32,
                              dt::int32);
            case dt::float64:
                return this->data.tensor_float64 + data.data.tensor_float64;
        }
        return *this;
    }

    data_t operator-(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 - data.data.tensor_int8;
            case dt::float32:
                return this->data.tensor_float32 - data.data.tensor_float32;
            case dt::bool8:
                return this->data.tensor_bool - data.data.tensor_bool;
            case dt::int32:
                return data_t(this->data.tensor_int32 - data.data.tensor_int32,
                              dt::int32);
            case dt::float64:
                return this->data.tensor_float64 - data.data.tensor_float64;
        }
        return *this;
    }

    data_t operator*(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 * data.data.tensor_int8;
            case dt::float32:
                return this->data.tensor_float32 * data.data.tensor_float32;
            case dt::bool8:
                return this->data.tensor_bool * data.data.tensor_bool;
            case dt::int32:
                return data_t(this->data.tensor_int32 * data.data.tensor_int32,
                              dt::int32);
            case dt::float64:
                return this->data.tensor_float64 * data.data.tensor_float64;
        }
        return *this;
    }

    data_t operator/(data_t data) {
        if (data.dtype == dt::float64) {
            data = data.to_dt(dt::float64);
            data_t me = this->to_dt(dt::float64);
            return me.data.tensor_float64 / data.data.tensor_float64;
        } else {
            data = data.to_dt(dt::float32);
            data_t me = this->to_dt(dt::float32);
            return me.data.tensor_float32 / data.data.tensor_float32;
        }
    }

    bool operator==(int32_t data) {
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 == data;
                break;
            case dt::float32:
                return this->data.tensor_float32 == data;
                break;
            case dt::bool8:
                return this->data.tensor_bool == data;
                break;
            case dt::int32:
                return this->data.tensor_int32 == data;
                break;
            case dt::float64:
                return this->data.tensor_float64 == data;
                break;
            default:
                break;
        }
        return false;
    }

    data_t operator+=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 += data.data.tensor_int8;
                break;
            case dt::float32:
                this->data.tensor_float32 += data.data.tensor_float32;
                break;
            case dt::bool8:
                this->data.tensor_bool += data.data.tensor_bool;
                break;
            case dt::int32:
                this->data.tensor_int32 += data.data.tensor_int32;
                break;
            case dt::float64:
                this->data.tensor_float64 += data.data.tensor_float64;
                break;
            default:
                break;
        }
        return *this;
    }

    data_t operator-=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 -= data.data.tensor_int8;
                break;
            case dt::float32:
                this->data.tensor_float32 -= data.data.tensor_float32;
                break;
            case dt::bool8:
                this->data.tensor_bool -= data.data.tensor_bool;
                break;
            case dt::int32:
                this->data.tensor_int32 -= data.data.tensor_int32;
                break;
            case dt::float64:
                this->data.tensor_float64 -= data.data.tensor_float64;
                break;
            default:
                break;
        }
        return *this;
    }

    data_t operator*=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 *= data.data.tensor_int8;
                break;
            case dt::float32:
                this->data.tensor_float32 *= data.data.tensor_float32;
                break;
            case dt::bool8:
                this->data.tensor_bool *= data.data.tensor_bool;
                break;
            case dt::int32:
                this->data.tensor_int32 *= data.data.tensor_int32;
                break;
            case dt::float64:
                this->data.tensor_float64 *= data.data.tensor_float64;
                break;
            default:
                break;
        }
        return *this;
    }

    data_t operator/=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 /= data.data.tensor_int8;
                break;
            case dt::float32:
                this->data.tensor_float32 /= data.data.tensor_float32;
                break;
            case dt::bool8:
                this->data.tensor_bool /= data.data.tensor_bool;
                break;
            case dt::int32:
                this->data.tensor_int32 /= data.data.tensor_int32;
                break;
            case dt::float64:
                this->data.tensor_float64 /= data.data.tensor_float64;
                break;
            default:
                break;
        }
        return *this;
    }

    bool operator<(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 < data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 < data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool < data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 < data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 < data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    bool operator>(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 > data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 > data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool > data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 > data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 > data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    bool operator!=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 != data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 != data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool != data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 != data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 != data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    bool operator>=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 >= data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 >= data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool >= data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 >= data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 >= data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    bool operator<=(data_t data) {
        data = data.to_dt(dtype);
        switch (dtype) {
            case dt::int8:
                return this->data.tensor_int8 <= data.data.tensor_int8;
                break;
            case dt::float32:
                return this->data.tensor_float32 <= data.data.tensor_float32;
                break;
            case dt::bool8:
                return this->data.tensor_bool <= data.data.tensor_bool;
                break;
            case dt::int32:
                return this->data.tensor_int32 <= data.data.tensor_int32;
                break;
            case dt::float64:
                return this->data.tensor_float64 <= data.data.tensor_float64;
                break;
            default:
                break;
        }
        return false;
    }

    // template <typename T>
    friend ostream &operator<<(ostream &os, data_t data) {
        switch (data.dtype) {
            case dt::int8:
                os << (short)data.data.tensor_int8;
                break;
            case dt::float32:
                os << data.data.tensor_float32;
                break;
            case dt::bool8:
                os << boolalpha << data.data.tensor_bool;
                break;
            case dt::int32:
                os << data.data.tensor_int32;
                break;
            case dt::float64:
                os << data.data.tensor_float64;
                break;
            default:
                break;
        }
        return os;
    }
};

}  // namespace ts
