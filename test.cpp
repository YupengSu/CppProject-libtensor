#include <iostream>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <type_traits>

enum dt { int8, float32 };

template <typename T>
struct is_vector : std::false_type {};

template <typename T>
struct is_vector<std::vector<T>> : std::true_type {};

template <class value_type>
class BaseTensor {
public:
    BaseTensor(const value_type& data) {
        std::cout << "BaseTensor constructor called with data: " << data << std::endl;
    }

    // 其他成员函数和定义
};

class Tensor {
public:
    void *base;
    dt dtype;

    Tensor() : base(nullptr), dtype(int8) {}

    template<typename T>
    void initializeBaseTensor(const T& data) {
        base = new BaseTensor<T>(data);
    }

    template <class T>
    Tensor(const std::vector<T>& data) {
        initializeBaseTensor(data);
    }

    // 重载Tensor构造函数以处理嵌套花括号输入
    template <class T>
    Tensor(const std::initializer_list<T>& data) {
        if constexpr (is_vector<T>::value) {
            initializeBaseTensor(data);
        } else {
            assert(false); // 这里可以根据情况处理其他类型的数据结构
        }
    }

    // 这里添加其他Tensor的构造函数重载用于处理其他类型的数据结构

    friend std::ostream &operator<<(std::ostream &os, const Tensor &ts) {
        if (ts.dtype == int8) {
            os << *static_cast<BaseTensor<int> *>(ts.base);
        } else if (ts.dtype == float32) {
            os << *static_cast<BaseTensor<float> *>(ts.base);
        } else {
            assert(false);
        }
        return os;
    }
};

Tensor tensor() {
    return Tensor();
}

template<typename T, typename... Args>
Tensor tensor(const T& arg, const Args&... args) {
    return Tensor{arg, args...};
}

int main() {
    // 嵌套花括号输入示例
    Tensor a = tensor(
        std::vector<std::vector<int>>{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        },
        std::vector<std::vector<int>>{
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1}
        }
    );

    std::cout << a << std::endl;

    return 0;
}
