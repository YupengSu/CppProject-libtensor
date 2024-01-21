#pragma once

#include <algorithm>
#include <cstdio>
#include <exception>

#include "serial_tensor.hpp"
using namespace std;

namespace ts {
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
        std::sprintf(ts::err::Error::msg_, (format), ##__VA_ARGS__); \
        throw ts::err::Error(ERROR_LOCATION);                        \
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


void CHECK_SAME_SHAPE(const TensorImpl& t1, const TensorImpl& t2);
void CHECK_SAME_DEVICE(const TensorImpl& t1, const TensorImpl& t2);
void CHECK_INDEX_VALID(size_t x, const TensorImpl& t);
void CHECK_CONTIGUOUS(const TensorImpl& t);
void CHECK_FLOATING(dt dtype);
}  // namespace ts
