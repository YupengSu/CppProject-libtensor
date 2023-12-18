#include "config.hpp"

namespace ts {
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
            break;
    }
    return os;
}

};  // namespace ts
