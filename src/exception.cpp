#include "exception.hpp"
#include <cstring>
#include <sstream>
namespace ts {
namespace err {

char Error::msg_[300] = {0};

Error::Error(const char* file, const char* func, unsigned int line)
    : file_(file), func_(func), line_(line){};

const char* Error::what() const noexcept {
    std::stringstream s;
    s << std::endl;
    s << file_ << ":" << line_ << ": ";
    s << "In function " << func_ << "()." << std::endl;
    s << msg_;
    auto&& str = s.str();
    memcpy(msg_, str.c_str(), str.length());
    return msg_;
}

}  // namespace err

void CHECK_SAME_SHAPE(const Tensor& t1, const Tensor& t2) {
    CHECK_EQUAL(t1.ndim, t2.ndim,
                "Expect the same dimensions, but got %dD and %dD", t1.ndim,
                t2.ndim);
    for (size_t i = 0; i < t1.ndim; ++i)
        CHECK_EQUAL(t1.size(i), t2.size(i),
                    "Expect the same size on the %zu dimension, but got "
                    "%zu and %zu.",
                    i, t1.size(i), t2.size(i));
}  // namespace ts

void CHECK_SAME_DEVICE(const Tensor& t1, const Tensor& t2) {
    CHECK_EQUAL(t1.device, t2.device,
                "Expect the same device, but got %s and %s",
                t1.device == dev::cpu ? "CPU" : "GPU",
                t2.device == dev::cpu ? "CPU" : "GPU");
}

void CHECK_INDEX_VALID(int x, const Tensor& t) {
    CHECK_IN_RANGE(x, 0, t.size(), "Index %zu out of range [0, %zu)", x, t.size());
}
}  // namespace st

namespace ts {

};  // namespace ts