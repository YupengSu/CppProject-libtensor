#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "serial_tensor.hpp"
#include "size.hpp"

namespace ts {
void checkCudaErrorFunc(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file
                  << ":" << line << std::endl;
        exit(1);
    }
}
#define checkCudaError(err) checkCudaErrorFunc(err, __FILE__, __LINE__)

void c_cudaMalloc(void** ptr, size_t size) {
    checkCudaError(cudaMalloc(ptr, size));
}

void* c_cudaMallocHost(size_t size) {
    void* ptr;
    checkCudaError(cudaMallocHost(&ptr, size));
    return ptr;
}
void c_cudaFreeHost(void* ptr) { checkCudaError(cudaFreeHost(ptr)); }

void c_cudaMemcpy(void* dst, void* src, size_t size, c_cudaMemcpyKind kind) {
    checkCudaError(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
}

void c_cudaFree(void* src) { checkCudaError(cudaFree(src)); }


__device__ bool CUDA_EPS_EQUAL(double a, double b) {
    return (a - b) < EPS && (a - b) > -EPS;
}
__device__ size_t get_idx(size_t index, const int* shape_v, const int* stride,
                          const int* origin_stride, int dim) {
    size_t offset = 0;
    size_t tmp = 0;
    for (int i = 0; i < dim; i++) {
        tmp = index / origin_stride[i];
        offset += tmp * stride[i];
        index -= tmp * origin_stride[i];
    }
    return offset;
}

__global__ void get_serial_tensorMM(void* dst, void* src, size_t size,
                                    const int* shape, const int* stride,
                                    const int* origin_stride, int dim) {
    data_t* dev_dst = (data_t*)dst;
    data_t* dev_src = (data_t*)src;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        dev_dst[i] = dev_src[offset];
    }
}

void get_serial_tensor_kernel(void* dst, const TensorImpl& a) {
    data_t* dev_dst = (data_t*)dst;
    data_t* dev_src = (data_t*)a.data.dp;
    int* shape;
    int* stride;
    int* origin_stride;
    size_t size = a.shape.data_len();
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    get_serial_tensorMM<<<blocksPerGrid, threadsPerBlock>>>(
        dev_dst, dev_src, size, shape, stride, origin_stride, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

__global__ void get_data_t(data_t& dst, void* ptr) { dst = *(data_t*)ptr; }

__device__ void add_data_t(data_t& dst, const data_t& a_origin, const data_t& b_origin, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a_origin.dtype) {
        case dt::int8:
            tmp_a = (double)a_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a_origin.data.tensor_float64;
            break;
    }
    switch (b_origin.dtype) {
        case dt::int8:
            tmp_b = (double)b_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b_origin.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 = tmp_a + tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 = tmp_a + tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool = tmp_a + tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 = tmp_a + tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 = tmp_a + tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

__device__ void sub_data_t(data_t& dst, const data_t& a_origin, const data_t& b_origin, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a_origin.dtype) {
        case dt::int8:
            tmp_a = (double)a_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a_origin.data.tensor_float64;
            break;
    }
    switch (b_origin.dtype) {
        case dt::int8:
            tmp_b = (double)b_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b_origin.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 = tmp_a - tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 = tmp_a - tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool = tmp_a - tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 = tmp_a - tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 = tmp_a - tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

 // TODO
__device__ void mul_data_t(data_t& dst, const data_t& a_origin, const data_t& b_origin, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a_origin.dtype) {
        case dt::int8:
            tmp_a = (double)a_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a_origin.data.tensor_float64;
            break;
    }
    switch (b_origin.dtype) {
        case dt::int8:
            tmp_b = (double)b_origin.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b_origin.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b_origin.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b_origin.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b_origin.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 = tmp_a * tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 = tmp_a * tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool = tmp_a * tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 = tmp_a * tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 = tmp_a * tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

__device__ void div_data_t(data_t& dst, const data_t& a, const data_t& b, dt target_dtype) {
    double tmp;
    switch (b.dtype) {
        case dt::int8:
            tmp = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp = b.data.tensor_float64;
            break;
    }
    if (target_dtype == dt::float64) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_float64 = (double)a.data.tensor_int8 / tmp;
                break;
            case dt::float32:
                dst.data.tensor_float64 = (double)a.data.tensor_float32 / tmp;
                break;
            case dt::bool8:
                dst.data.tensor_float64 = (double)a.data.tensor_bool / tmp;
                break;
            case dt::int32:
                dst.data.tensor_float64 = (double)a.data.tensor_int32 / tmp;
                break;
            case dt::float64:
                dst.data.tensor_float64 = a.data.tensor_float64 / tmp;
                break;
        }
    } else {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_float32 = (float)a.data.tensor_int8 / tmp;
                break;
            case dt::float32:
                dst.data.tensor_float32 = (float)a.data.tensor_float32 / tmp;
                break;
            case dt::bool8:
                dst.data.tensor_float32 = (float)a.data.tensor_bool / tmp;
                break;
            case dt::int32:
                dst.data.tensor_float32 = (float)a.data.tensor_int32 / tmp;
                break;
            case dt::float64:
                dst.data.tensor_float64 = a.data.tensor_float64 / tmp;
                break;
        }
    }
    dst.dtype = target_dtype;
}

__device__ void eq_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = CUDA_EPS_EQUAL(tmp_a, tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void ne_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = !CUDA_EPS_EQUAL(tmp_a, tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void gt_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = (tmp_a > tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void ge_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = (tmp_a >= tmp_b) || CUDA_EPS_EQUAL(tmp_a, tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void lt_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = (tmp_a < tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void le_data_t(data_t& dst, const data_t& a, const data_t& b) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    dst.data.tensor_bool = (tmp_a <= tmp_b) || CUDA_EPS_EQUAL(tmp_a, tmp_b);
    dst.dtype = dt::bool8;
}

__device__ void log_data_t(data_t& dst, const data_t& a) {
    switch (a.dtype) {
        case dt::int8:
            dst.data.tensor_float64 = ::log((double)a.data.tensor_int8);
            break;
        case dt::float32:
            dst.data.tensor_float64 = ::log((double)a.data.tensor_float32);
            break;
        case dt::bool8:
            dst.data.tensor_float64 = ::log((double)a.data.tensor_bool);
            break;
        case dt::int32:
            dst.data.tensor_float64 = ::log((double)a.data.tensor_int32);
            break;
        case dt::float64:
            dst.data.tensor_float64 = ::log(a.data.tensor_float64);
            break;
    }
    dst.dtype = dt::float64;
}

__device__ void cmp_data_t(data_t& dst, const data_t& a, const data_t& b, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 = tmp_a > tmp_b? tmp_a: tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 = tmp_a > tmp_b? tmp_a: tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool = tmp_a > tmp_b? tmp_a: tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 = tmp_a > tmp_b? tmp_a: tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 = tmp_a > tmp_b? tmp_a: tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

__device__ void cmn_data_t(data_t& dst, const data_t& a, const data_t& b, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 = tmp_a < tmp_b? tmp_a: tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 = tmp_a < tmp_b? tmp_a: tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool = tmp_a < tmp_b? tmp_a: tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 = tmp_a < tmp_b? tmp_a: tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 = tmp_a < tmp_b? tmp_a: tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

__device__ void muladd_data_t(data_t& dst, const data_t& a, const data_t& b, dt target_dtype) {
    double tmp_a, tmp_b;
    switch (a.dtype) {
        case dt::int8:
            tmp_a = (double)a.data.tensor_int8;
            break;
        case dt::float32:
            tmp_a = (double)a.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_a = (double)a.data.tensor_bool;
            break;
        case dt::int32:
            tmp_a = (double)a.data.tensor_int32;
            break;
        case dt::float64:
            tmp_a = a.data.tensor_float64;
            break;
    }
    switch (b.dtype) {
        case dt::int8:
            tmp_b = (double)b.data.tensor_int8;
            break;
        case dt::float32:
            tmp_b = (double)b.data.tensor_float32;
            break;
        case dt::bool8:
            tmp_b = (double)b.data.tensor_bool;
            break;
        case dt::int32:
            tmp_b = (double)b.data.tensor_int32;
            break;
        case dt::float64:
            tmp_b = b.data.tensor_float64;
            break;
    }
    switch (target_dtype) {
        case dt::int8:
            dst.data.tensor_int8 += tmp_a * tmp_b;
            break;
        case dt::float32:
            dst.data.tensor_float32 += tmp_a * tmp_b;
            break;
        case dt::bool8:
            dst.data.tensor_bool += tmp_a * tmp_b;
            break;
        case dt::int32:
            dst.data.tensor_int32 += tmp_a * tmp_b;
            break;
        case dt::float64:
            dst.data.tensor_float64 += tmp_a * tmp_b;
            break;
    }
    dst.dtype = target_dtype;
}

__global__ void addTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                int* shape, int* stride_a, int* stride_b,
                                int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset_a], b[offset_b], target_dtype);
    }
}

__global__ void addTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                   int* shape, int* stride, int* origin_stride,
                                   int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset], b, target_dtype);
    }
}

__global__ void subTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                int* shape, int* stride_a, int* stride_b,
                                int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        sub_data_t(c[i], a[offset_a], b[offset_b], target_dtype);
    }
}

__global__ void subTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                   int* shape, int* stride, int* origin_stride,
                                   int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        // offset = i;
        sub_data_t(c[i], a[offset], b, target_dtype);
    }
}

__global__ void mulTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                int* shape, int* stride_a, int* stride_b,
                                int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        mul_data_t(c[i], a[offset_a], b[offset_b], target_dtype);
    }
}

__global__ void mulTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                   int* shape, int* stride, int* origin_stride,
                                   int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        // offset = i;
        mul_data_t(c[i], a[offset], b, target_dtype);
    }
}

__global__ void divTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                int* shape, int* stride_a, int* stride_b,
                                int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        div_data_t(c[i], a[offset_a], b[offset_b], target_dtype);
    }
}

__global__ void divTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                   int* shape, int* stride, int* origin_stride,
                                   int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        div_data_t(c[i], a[offset], b, target_dtype);
    }
}

__global__ void eqTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        eq_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void neTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        ne_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void gtTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        gt_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void geTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        ge_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void ltTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        lt_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void leTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                               int* shape, int* stride_a, int* stride_b,
                               int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        le_data_t(c[i], a[offset_a], b[offset_b]);
    }
}

__global__ void matrixMultiplyTensorKernel(data_t* c, data_t* a, data_t* b,
                                     size_t M, size_t N, size_t K, int* shape_a, int* shape_b,
                                     int* stride_a, int* stride_b,
                                     int* origin_stride_a, int* origin_stride_b,
                                     int dim_a, int dim_b, dt target_dtype) {
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < M && ty < N) {
        for (int i = 0; i < K; i++) {
            size_t offset_a = get_idx(tx * K + i, shape_a, stride_a, origin_stride_a, dim_a);
            size_t offset_b = get_idx(i * N + ty, shape_b, stride_b, origin_stride_b, dim_b);
            if (i == 0)
                mul_data_t(c[tx * N + ty], a[offset_a], b[offset_b], target_dtype);
            else
                muladd_data_t(c[tx * N + ty], a[offset_a], b[offset_b], target_dtype);
        }
    }
}

__global__ void logTensorKernel(data_t* c, data_t* a, size_t size, int* shape,
                                int* stride, int* origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        log_data_t(c[i], a[offset]);
    }
}
__global__ void sumTensorKernel(data_t* c, data_t* a, size_t dim_size, size_t outer_size, size_t inner_size, int* shape,
                                int* stride, int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < outer_size && j < inner_size) {
        size_t index_new = i * inner_size + j;
        size_t index_old = i * inner_size * dim_size + j;
        // offset = i;
        for (int k = 0; k < dim_size; k++) {
            size_t offset = get_idx(index_old + k * inner_size, shape, stride, origin_stride, dim);
            if (k == 0)
                c[index_new] = a[offset];
            else
                add_data_t(c[index_new], c[index_new], a[offset], target_dtype);
        }
    }
}

__global__ void maxTensorKernel(data_t* c, data_t* a, size_t dim_size, size_t outer_size, size_t inner_size, int* shape,
                                int* stride, int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < outer_size && j < inner_size) {
        size_t index_new = i * inner_size + j;
        size_t index_old = i * inner_size * dim_size + j;
        // offset = i;
        for (int k = 0; k < dim_size; k++) {
            size_t offset = get_idx(index_old + k * inner_size, shape, stride, origin_stride, dim);
            if (k == 0)
                c[index_new] = a[offset];
            else
                cmp_data_t(c[index_new], c[index_new], a[offset], target_dtype);
        }
    }
}

__global__ void minTensorKernel(data_t* c, data_t* a, size_t dim_size, size_t outer_size, size_t inner_size, int* shape,
                                int* stride, int* origin_stride, int dim, dt target_dtype) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < outer_size && j < inner_size) {
        size_t index_new = i * inner_size + j;
        size_t index_old = i * inner_size * dim_size + j;
        // offset = i;
        for (int k = 0; k < dim_size; k++) {
            size_t offset = get_idx(index_old + k * inner_size, shape, stride, origin_stride, dim);
            if (k == 0)
                c[index_new] = a[offset];
            else
                cmn_data_t(c[index_new], c[index_new], a[offset], target_dtype);
        }
    }
}

void addKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    addTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void addKernelNum(void* dst, TensorImpl a, data_t b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    b = b.to_dt(a.dtype);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    addTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void subKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    subTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void subKernelNum(void* dst, TensorImpl a, data_t b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    b = b.to_dt(a.dtype);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    subTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void mulKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    mulTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void mulKernelNum(void* dst, TensorImpl a, data_t b, size_t size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    b = b.to_dt(a.dtype);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    mulTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void divKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    dt target_dtype;
    if (a.dtype == dt::float64 || b.dtype == dt::float64) {
        target_dtype = dt::float64;
    } else {
        target_dtype = dt::float32;
    }

    divTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void divKernelNum(void* dst, TensorImpl a, data_t b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int* shape;
    int* stride;
    int* origin_stride;
    dt target_dtype;
    if (a.dtype == dt::float64) {
        target_dtype = dt::float64;
    } else {
        target_dtype = dt::float32;
    }
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    divTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim(),
        target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void matrixMultiplyKernel(void* dst, TensorImpl a, TensorImpl b, size_t M, size_t N, size_t K, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    
    dim3 blocksPerGrid(16, 16);
    dim3 threadsPerBlock((M + blocksPerGrid.x - 1) / blocksPerGrid.x, 
                        (N + blocksPerGrid.y - 1) / blocksPerGrid.y);
    int* shape_a;
    int* shape_b;
    int* stride_a;
    int* stride_b;
    int* origin_stride_a;
    int* origin_stride_b;
    
    checkCudaError(cudaMalloc(&shape_a, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&shape_b, b.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride_a, a.shape.shape.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride_b, b.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape_a, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(shape_b, b.shape.shape.data(),
                              b.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride_a, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride_b, b.origin_stride.data(),
                              b.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    matrixMultiplyTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, M, N, K, shape_a, shape_b, stride_a, stride_b,
        origin_stride_a, origin_stride_b, a.get_dim(), b.get_dim(), target_dtype);
    
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape_a));
    checkCudaError(cudaFree(shape_b));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride_a));
    checkCudaError(cudaFree(origin_stride_b));
}

// NOT DONE
void eqKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    eqTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void neKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    neTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void gtKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    gtTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void geKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    geTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void ltKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    ltTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void leKernel(void* dst, const TensorImpl& a, const TensorImpl& b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* shape;
    int* stride_a;
    int* stride_b;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_a, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride_b, b.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_a, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride_b, b.stride.data(),
                              b.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    leTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
        a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

void logKernel(void *dst, TensorImpl a, size_t size, dt target_dtype) {
    data_t *dev_a = (data_t *)a.data.dp;
    data_t *dev_c = (data_t *)dst;
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int *shape;
    int *stride;
    int *origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(), a.shape.shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(), a.stride.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(), a.origin_stride.size() * sizeof(int), cudaMemcpyHostToDevice));

    logTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, size, shape, stride, origin_stride, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void sumKernel(void* dst, TensorImpl a, size_t dim, size_t outer_size,size_t inner_size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    dim3 threadsPerBlock (16, 16); // 256 threads per block
    dim3 blocksPerGrid ((outer_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (inner_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    sumTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, a.shape[dim], outer_size, inner_size,
        shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

void maxKernal(void* dst, TensorImpl a, size_t dim, size_t outer_size,size_t inner_size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    dim3 threadsPerBlock (16, 16); // 256 threads per block
    dim3 blocksPerGrid ((outer_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (inner_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    maxTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, a.shape[dim], outer_size, inner_size,
        shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));

} 

void minKernal(void* dst, TensorImpl a, size_t dim, size_t outer_size,size_t inner_size, dt target_dtype) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    dim3 threadsPerBlock (16, 16); // 256 threads per block
    dim3 blocksPerGrid ((outer_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (inner_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int* shape;
    int* stride;
    int* origin_stride;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
    checkCudaError(
        cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(),
                              a.shape.shape.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stride, a.stride.data(),
                              a.stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(origin_stride, a.origin_stride.data(),
                              a.origin_stride.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    minTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        dev_c, dev_a, a.shape[dim], outer_size, inner_size,
        shape, stride, origin_stride, a.get_dim(), target_dtype);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

}// namespace ts
