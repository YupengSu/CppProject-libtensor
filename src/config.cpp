#include "config.hpp"

namespace ts 
{
    ostream &operator<<(ostream &os, const dt dtype) {
        switch (dtype) {
            case dt::int8:
                os << "dtype::int8";
                break;
            case dt::float32:
                os << "dtype::float32";
                break;
            case dt::bool8:
                os << "dtype::bool8";
                break;
            case dt::int32:
                os << "dtype::int32";
                break;
            case dt::float64:
                os << "dtype::float64";
                break;
            default:
                os << "Error dtype" << (int)dtype;
                break;
        }
        return os;
    }

    ostream &operator<<(ostream &os, const dev device) {
        switch (device) {
            case dev::cpu:
                os << "dev::cpu";
                break;
            case dev::cuda:
                os << "dev::cuda";
                break;
            default:
                break;
        }
        return os;
    }

    
    bool is_floating(dt dtype) {
        return dtype == dt::float32 || dtype == dt::float64;
    }

    string dtype_name(dt dtype) {
        switch (dtype) {
            case dt::int8:
                return "dt::int8";
            case dt::float32:
                return "dt::float32";
            case dt::bool8:
                return "dt::bool8";
            case dt::int32:
                return "dt::int32";
            case dt::float64:
                return "dt::float64";
            default:
                return "Error dtype" + to_string((int)dtype);
        }
    }

    dt descision_dtype(dt dtype1, dt dtype2) {
        if (dtype1 == dtype2) {
            return dtype1;
        }
        else if (dtype1 == dt::float64 || dtype2 == dt::float64) {
            return dt::float64;
        }
        else if (dtype1 == dt::float32 || dtype2 == dt::float32) {
            return dt::float32;
        }
        else if (dtype1 == dt::int32 || dtype2 == dt::int32) {
            return dt::int32;
        } else if (dtype1 == dt::int8 || dtype2 == dt::int8) {
            return dt::int8;
        } else {
            return dt::bool8;
        }
    }
}