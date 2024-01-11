#pragma once
#include <cstdint>
#include <iostream>
#include <ostream>

#define DEFAULT_DTYPE dt::float32

using namespace std;
namespace ts {
enum class dt { 
    int8, float32, float64, int32, bool8

};

 ostream &operator<<(ostream &os, const dt dtype);

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
    data_t(data_tt data) : data(data){}
    data_t(int8_t data) { this->data.tensor_int8 = data; this->dtype= dt::int8;}
    data_t(float data) {
        this->data.tensor_float32 = data;
        this->dtype = dt::float32;
    }
    data_t(double data) { this->data.tensor_float64 = data; this->dtype = dt::float64;}
    data_t(int32_t data) { this->data.tensor_int32 = data; this->dtype = dt::int32; }
    data_t(bool data) { this->data.tensor_bool = data; this->dtype = dt::bool8; }
    template <typename T>
    T &get_data() {
        switch (dtype) {
            case dt::int8:
                return *(T *)(&data);
            case dt::float32:
                return *(T *)(&data);
            case dt::bool8:
                return *(T *)(&data);
            case dt::int32:
                return *(T *)(&data);
            case dt::float64:
                return *(T *)(&data);
            default:
                break;
        }
        return *(T *)(&data);
    }

    data_t to_dt(dt target) {
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

    template <typename T>
    operator T() {
        return *(T *)(&data);
    }

    void set_dtype(dt dtype) {
        data_t tmp = this->to_dt(dtype);
        this->dtype = dtype;
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 = tmp.data.tensor_int8;
                break;
            case dt::float32:
                this->data.tensor_float32 = tmp.data.tensor_float32;
                break;
            case dt::bool8:
                this->data.tensor_bool = tmp.data.tensor_bool;
                break;
            case dt::int32:
                this->data.tensor_int32 = tmp.data.tensor_int32;
                break;
            case dt::float64:
                this->data.tensor_float64 = tmp.data.tensor_float64;
                break;
            default:
                break;
        }
    }

    template <typename T>
    T &operator=(T i_data) {
        switch (dtype) {
            case dt::int8:
                this->data.tensor_int8 = (uint8_t)i_data;
                break;
            case dt::float32:
                this->data.tensor_float32 = (float)i_data;
                break;
            case dt::bool8:
                this->data.tensor_bool = (bool)i_data;
                break;
            case dt::int32:
                this->data.tensor_int32 = (int)i_data;
                break;
            case dt::float64:
                this->data.tensor_float64 = (double)i_data;
                break;
            default:
                break;
        }
        return *(T *)(&data);
    }

    bool operator==(data_t data) {
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

    data_t& operator+(data_t data) {
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

    data_t& operator-(data_t data) {
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

    data_t& operator*(data_t data) {
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

    data_t& operator/(data_t data) {
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

    bool operator>= (data_t data) {
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
                os << data.data.tensor_int8;
                break;
            case dt::float32:
                os << data.data.tensor_float32;
                break;
            case dt::bool8:
                os << data.data.tensor_bool;
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
