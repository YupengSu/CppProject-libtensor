#pragma once

#include <algorithm>
#include <cstdio>
#include <exception>
using namespace std;

namespace st {
namespace err {

struct Error : public std::exception {
    Error(const char* file, const char* func, unsigned int line);
    const char* what() const noexcept;

    static char msg_[300];
    const char* file_;
    const char* func_;
    const unsigned int line_;
};

}  // namespace err

#define ERROR_LOCATION __FILE__, __func__, __LINE__

#define THROW_ERROR(format, ...)                                       \
    do {                                                               \
        std::sprintf(::st::err::Error::msg_, (format), ##__VA_ARGS__); \
        throw ::st::err::Error(ERROR_LOCATION);                        \
    } while (0)

#define CHECK_TRUE(expr, format, ...) \
    if (!(expr)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_NOT_NULL(ptr, format, ...) \
    if (nullptr == (ptr)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_EQUAL(x, y, format, ...) \
    if ((x) != (y)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_IN_RANGE(x, lower, upper, format, ...) \
    if ((x) < (lower) || (x) >= (upper)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_FLOAT_EQUAL(x, y, format, ...) \
    if (std::abs((x) - (y)) > 1e-4) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_INDEX_VALID(x, format, ...) \
    if ((x) > INDEX_MAX) THROW_ERROR((format), ##__VA_ARGS__)

// assert macro only working for ExpImpl
#define CHECK_EXP_SAME_SHAPE(e1_, e2_)                                       \
    do {                                                                     \
        auto& e1 = (e1_);                                                    \
        auto& e2 = (e2_);                                                    \
        CHECK_EQUAL(e1.ndim(), e2.ndim(),                                    \
                    "Expect the same dimensions, but got %dD and %dD",       \
                    e1.ndim(), e2.ndim());                                   \
        for (index_t i = 0; i < e1.ndim(); ++i)                              \
            CHECK_EQUAL(e1.size(i), e2.size(i),                              \
                        "Expect the same size on the %d dimension, but got " \
                        "%d and %d.",                                        \
                        i, e1.size(i), e2.size(i));                          \
    } while (0)

}  // namespace st