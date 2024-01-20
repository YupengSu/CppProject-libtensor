#include <cstddef>
#include <typeinfo>
#include <utility>
#include <vector>

#include "api.hpp"

// include your implementation's header file here, e.g.
// #include "../my_tensor.hpp"
#include "../include/serial_tensor.hpp"

namespace bm {

template <typename T>
ts::Tensor<T> create_with_data(const std::vector<size_t> &shape,
                               const T *data) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;
    size_t size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }

    for (int i = 0; i < size; i++) {
        tmp.push_back(data[i]);
    }
    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }

    res.my_tensor = ts::TensorImpl(tmp, shape_);
    return res;
    // TODO
}

template <typename T>
ts::Tensor<T> rand(const std::vector<size_t> &shape) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;

    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }

    res.my_tensor = ts::rand(shape_, target_dtype);
    return res;
}

template <typename T>
ts::Tensor<T> zeros(const std::vector<size_t> &shape) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;

    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }
    res.my_tensor = ts::zeros(shape_, target_dtype);
    return res;
}

template <typename T>
ts::Tensor<T> ones(const std::vector<size_t> &shape) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;

    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }
    res.my_tensor = ts::ones(shape_, target_dtype);
    return res;
}

template <typename T>
ts::Tensor<T> full(const std::vector<size_t> &shape, const T &value) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;

    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }
    res.my_tensor = ts::full(shape_, value);
    return res;
}

template <typename T>
ts::Tensor<T> eye(size_t rows, size_t cols) {
    ts::Tensor<T> res;
    vector<ts::data_t> tmp;
    ts::dt target_dtype;
    vector<int> shape_;

    shape_.push_back(rows);
    shape_.push_back(cols);

    if (typeid(T) == typeid(bool)) {
        target_dtype = ts::dt::bool8;
    } else if (typeid(T) == typeid(int)) {
        target_dtype = ts::dt::int32;
    } else {
        target_dtype = ts::dt::float32;
    }
    res.my_tensor = ts::eye(shape_, target_dtype);
    return res;
}

template <typename T>
ts::Tensor<T> slice(const ts::Tensor<T> &tensor,
                    const std::vector<std::pair<size_t, size_t>> &slices) {
    ts::Tensor<T> res = tensor;
    for (pair<size_t, size_t> i : slices) {
        res.my_tensor = res.my_tensor.slice(i.first, i.second);
    }
    // TODO
    return res;
}

template <typename T>
ts::Tensor<T> concat(const std::vector<ts::Tensor<T>> &tensors, size_t axis) {
    // TODO
    ts::TensorImpl res;
    vector<ts::TensorImpl> tmp;
    for (int i = 0; i < tensors.size(); i++) {
        tmp.push_back(tensors[i].my_tensor);
    }
    res = ts::cat(tmp, axis);
    ts::Tensor<T> res_;
    res_.my_tensor = res;
    return res_;
}

template <typename T>
ts::Tensor<T> tile(const ts::Tensor<T> &tensor,
                   const std::vector<size_t> &shape) {
    // TODO
    ts::Tensor<T> res;
    vector<int> shape_;
    for (int i = 0; i < shape.size(); i++) {
        shape_.push_back(shape[i]);
    }
    res.my_tensor = ts::tile(tensor.my_tensor, shape_);
    return res;
}

template <typename T>
ts::Tensor<T> transpose(const ts::Tensor<T> &tensor, size_t dim1, size_t dim2) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = tensor.my_tensor.transpose(dim1, dim2);
    return res;
}

template <typename T>
ts::Tensor<T> permute(const ts::Tensor<T> &tensor,
                      const std::vector<size_t> &permutation) {
    ts::Tensor<T> res;
    vector<int> permutation_;
    for (int i = 0; i < permutation.size(); i++) {
        permutation_.push_back(permutation[i]);
    }
    res.my_tensor = tensor.my_tensor.permute(permutation_);
    return res;
}

template <typename T>
T at(const ts::Tensor<T> &tensor, const std::vector<size_t> &indices) {
    size_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
        offset += indices[i] * tensor.my_tensor.origin_stride[i];
    }

    if (tensor.my_tensor.device == ts::dev::cuda) {
        return tensor.my_tensor.at(offset);
    } else {
        return tensor.my_tensor.locate(indices);
    }
}

template <typename T>
void set_at(ts::Tensor<T> &tensor, const std::vector<size_t> &indices,
            const T &value) {
    size_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
        offset += indices[i] * tensor.my_tensor.origin_stride[i];
    }

    if (tensor.my_tensor.device == ts::dev::cuda) {
        tensor.my_tensor.set_at(offset, value);
    } else {
        tensor.my_tensor.get(offset) = value;
    }
}

template <typename T>
ts::Tensor<T> pointwise_add(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) + b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<T> pointwise_sub(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) - b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<T> pointwise_mul(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) * b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<T> pointwise_div(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) / b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<T> pointwise_log(const ts::Tensor<T> &tensor) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = ts::log(tensor.my_tensor).to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<T> reduce_sum(const ts::Tensor<T> &tensor, size_t axis) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = tensor.my_tensor.sum(axis);
    return res;
}

template <typename T>
ts::Tensor<T> reduce_mean(const ts::Tensor<T> &tensor, size_t axis) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = tensor.my_tensor.mean(axis);
    return res;
}

template <typename T>
ts::Tensor<T> reduce_max(const ts::Tensor<T> &tensor, size_t axis) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = tensor.my_tensor.max(axis);
    return res;
}

template <typename T>
ts::Tensor<T> reduce_min(const ts::Tensor<T> &tensor, size_t axis) {
    // TODO
    ts::Tensor<T> res;
    res.my_tensor = tensor.my_tensor.min(axis);
    return res;
}

// you may modify the following functions' implementation if necessary

template <typename T>
ts::Tensor<bool> eq(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) == b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<bool> ne(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) != b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<bool> gt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) > b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<bool> ge(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) >= b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<bool> lt(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) < b.my_tensor.to(TARGET_PLATFORM);
    return res;
}

template <typename T>
ts::Tensor<bool> le(const ts::Tensor<T> &a, const ts::Tensor<T> &b) {
    ts::Tensor<bool> res;
    res.my_tensor =
        a.my_tensor.to(TARGET_PLATFORM) <= b.my_tensor.to(TARGET_PLATFORM);
    return res;
}
}  // namespace bm
