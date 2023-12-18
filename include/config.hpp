#pragma once
#include <cstddef>
#include <istream>
#include <cstdint>

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
    dt dtype=DEFAULT_DTYPE;
    public:
    data_t() = default;
    data_t(data_tt data) : data(data) {}
    data_t(int8_t data) {
        this->data.tensor_int8 = data;
    }
    data_t(float data) {
        this->data.tensor_float32 = data;
    }
    data_t(double data) {
        this->data.tensor_float64 = data;
    }
    data_t(uint8_t data) {
        this->data.tensor_int8 = data;
    }
    data_t(uint32_t data) {
        this->data.tensor_int32 = data;
    }
    data_t(bool data) {
        this->data.tensor_bool = data;
    }
    template<typename T>
    T &get_data() {
        switch (dtype) {
            case int8:
                return *(T*)(&data.tensor_int8);
            case float32:
                return *(T*)(&data.tensor_float32);
            case bool8:
                return *(T*)(&data.tensor_bool);
            case int32:
                return *(T*)(&data.tensor_int32);
            case float64:
                return *(T*)(&data.tensor_float64);
            default:
                break;
        }
        return *(T*)(&data);
    }


    template<typename  T>
    operator T() {
        switch (dtype) {
            case int8:
                return *(T*)(&data.tensor_int8);
            case float32:
                return *(T*)(&data.tensor_float32);
            case bool8:
                return *(T*)(&data.tensor_bool);
            case int32:
                return *(T*)(&data.tensor_int32);
            case float64:
                return *(T*)(&data.tensor_float64);
            default:
                break;
        }
        return *(T*)(&data);
    }

    void set_dtype(dt dtype) {
        switch (dtype) {
            case int8:
                this->data.tensor_int8 = (int8_t)this->data.tensor_int8;
                break;
            case float32:
                this->data.tensor_float32 = (float)this->data.tensor_float32;
                break;
            case bool8:
                this->data.tensor_bool = (bool)this->data.tensor_bool;
                break;
            case int32:
                this->data.tensor_int32 = (int)this->data.tensor_int32;
                break;
            case float64:
                this->data.tensor_float64 = (double)this->data.tensor_float64;
                break;
            default:
                break;
        }
        this->dtype = dtype;
    }

    template<typename T>
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
        return *(T*)(&data);
    }

    template<typename T>
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
