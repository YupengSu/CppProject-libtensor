#include "serial_tensor.hpp"
#include <climits>
#include <vector>
#include "config.hpp"

using namespace std;

namespace ts
{
    Tensor::Tensor()
    {
        this->ndim = 0;
        this->shape = Size(0);
        this->stride.reserve(0);
        this->offset = 0;
    }
    Tensor::Tensor(const vector<data_t> &i_data, const vector<int> &i_shape,
                   dt dtype, dev device)
    {
        if (i_shape.size() == 0)
        {
            this->ndim = 1;
            this->shape = Size(i_data.size());
        }
        else
        {
            this->ndim = i_shape.size();
            this->shape = Size(i_shape);
        }

        this->data = Storage(i_data.data(), this->shape.size(), dtype, device);
        this->dtype = dtype;
        this->device = device;
        this->offset = 0;
        this->stride = init_stride(this->shape.shape);
    }

    Tensor::Tensor(const Storage &i_data, const Size &i_shape,
                   const vector<int> i_stride, dt dtype, dev device)
        : data(i_data), stride(i_stride), shape(i_shape)
    {
        this->ndim = i_shape.ndim;
        this->dtype = dtype;
        this->device = device;
    }

    Tensor Tensor::to(dev device)
    {
        this->device = device; 
        return *this;
    }
} // namespace ts