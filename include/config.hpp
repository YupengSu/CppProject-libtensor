#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>

#define DEFAULT_DTYPE int8
using namespace std;
namespace ts {
enum dt { int8, float32, float64, int32, bool8 };

typedef union {
    uint8_t tensor_int8;
    uint32_t tensor_int32;
    float tensor_float32;
    double tensor_float64;
    bool tensor_bool;
} data_tt;

class data_t {
    data_tt data;
    dt dtype = DEFAULT_DTYPE;

   public:
    data_t() = default;
    data_t(data_tt data) : data(data) {}
    data_t(int8_t data) { this->data.tensor_int8 = data; }
    data_t(float data) {
        this->data.tensor_float32 = data;
    }
    data_t(double data) { this->data.tensor_float64 = data; }
    data_t(uint8_t data) { this->data.tensor_int8 = data; }
    data_t(uint32_t data) { this->data.tensor_int32 = data; }
    data_t(bool data) { this->data.tensor_bool = data; }
    template <typename T>
    T &get_data() {
        switch (dtype) {
            case int8:
                return *(T *)(&data);
            case float32:
                return *(T *)(&data);
            case bool8:
                return *(T *)(&data);
            case int32:
                return *(T *)(&data);
            case float64:
                return *(T *)(&data);
            default:
                break;
        }
        return *(T *)(&data);
    }

    data_t to_dt(dt target) {
        data_t res;
        switch (target) {
            case int8:
                res.dtype = int8;
                switch (dtype) {
                    case int8:
                        res = (int8_t)this->data.tensor_int8;
                        break;
                    case float32:
                        res = (int8_t)this->data.tensor_float32;
                        break;
                    case bool8:
                        res = (int8_t)this->data.tensor_bool;
                        break;
                    case int32:
                        res = (int8_t)this->data.tensor_int32;
                        break;
                    case float64:
                        res = (int8_t)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case float32:
                res.dtype = float32;
                switch (dtype) {
                    case int8:
                        res = (float)this->data.tensor_int8;
                        break;
                    case float32:
                        res = (float)this->data.tensor_float32;
                        break;
                    case bool8:
                        res = (float)this->data.tensor_bool;
                        break;
                    case int32:
                        res = (float)this->data.tensor_int32;
                        break;
                    case float64:
                        res = (float)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case bool8:
                res.dtype = bool8;
                switch (dtype) {
                    case int8:
                        res = (bool)this->data.tensor_int8;
                        break;
                    case float32:
                        res = (bool)this->data.tensor_float32;
                        break;
                    case bool8:
                        res = (bool)this->data.tensor_bool;
                        break;
                    case int32:
                        res = (bool)this->data.tensor_int32;
                        break;
                    case float64:
                        res = (bool)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case int32:
                res.dtype = int32;
                switch (dtype) {
                    case int8:
                        res = (int)this->data.tensor_int8;
                        break;
                    case float32:
                        res = (int)this->data.tensor_float32;
                        break;
                    case bool8:
                        res = (int)this->data.tensor_bool;
                        break;
                    case int32:
                        res = (int)this->data.tensor_int32;
                        break;
                    case float64:
                        res = (int)this->data.tensor_float64;
                        break;
                    default:
                        break;
                }
                break;
            case float64:
                res.dtype = float64;
                switch (dtype) {
                    case int8:
                        res = (double)this->data.tensor_int8;
                        break;
                    case float32:
                        res = (double)this->data.tensor_float32;
                        break;
                    case bool8:
                        res = (double)this->data.tensor_bool;
                        break;
                    case int32:
                        res = (double)this->data.tensor_int32;
                        break;
                    case float64:
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
            case int8:
                this->data.tensor_int8 = tmp.data.tensor_int8;
                break;
            case float32:
                this->data.tensor_float32 = tmp.data.tensor_float32;
                break;
            case bool8:
                this->data.tensor_bool = tmp.data.tensor_bool;
                break;
            case int32:
                this->data.tensor_int32 = tmp.data.tensor_int32;
                break;
            case float64:
                this->data.tensor_float64 = tmp.data.tensor_float64;
                break;
            default:
                break;
        }
    }

    template <typename T>
    T &operator=(T i_data) {
        switch (dtype) {
            case int8:
                this->data.tensor_int8 = (uint8_t)i_data;
                break;
            case float32:
                this->data.tensor_float32 = (float)i_data;
                break;
            case bool8:
                this->data.tensor_bool = (bool)i_data;
                break;
            case int32:
                this->data.tensor_int32 = (int)i_data;
                break;
            case float64:
                this->data.tensor_float64 = (double)i_data;
                break;
            default:
                break;
        }
        return *(T *)(&data);
    }

    template <typename T>
    friend ostream &operator<<(ostream &os, T data) {
        switch (data.dtype) {
            case int8:
                os << (int)data.data.tensor_int8;
                break;
            case float32:
                os << data.data.tensor_float32;
                break;
            case bool8:
                os << data.data.tensor_bool;
                break;
            case int32:
                os << data.data.tensor_int32;
                break;
            case float64:
                os << data.data.tensor_float64;
                break;
            default:
                break;
        }
        return os;
    }
};
}  // namespace ts
