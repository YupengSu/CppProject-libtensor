#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "size.hpp"
#include "serial_tensor.hpp"

namespace ts {
void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file
                  << ":" << line << std::endl;
        exit(1);
    }
}

#define checkCudaError(err) checkCudaError(err, __FILE__, __LINE__)

__device__ void add_data_t(data_t& dst, data_t& a, data_t& b) {
    switch (a.dtype) {
        case dt::int8:
            dst.data.tensor_int8 = a.data.tensor_int8 + b.data.tensor_int8;
        break;
        case dt::float32:
            dst.data.tensor_float32 =
                a.data.tensor_float32 + b.data.tensor_float32;
        break;
        case dt::bool8:
            dst.data.tensor_bool = a.data.tensor_bool + b.data.tensor_bool;
        break;
        case dt::int32:
            dst.data.tensor_int32 = a.data.tensor_int32 + b.data.tensor_int32;
        break;
        case dt::float64:
            dst.data.tensor_float64 =
                a.data.tensor_float64 + b.data.tensor_float64;
        break;
    }
    dst.dtype=a.dtype;
}
__global__ void addMMKernel(data_t* c, data_t* a, data_t* b, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        add_data_t(c[i], a[i], b[i]);
    }

}

__device__ void get_idx(size_t &dst, size_t index, int * shape, int * stride, int dim) {
    dst = 0;
    for (int i = 0; i < dim; i++) {
        dst += index / stride[i] % shape[i] * stride[i];
    }
}

__global__ void addTensorKernel(data_t* c, data_t* a, data_t* b, size_t size, int * shape, int * stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset;
        get_idx(offset, i, shape, stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset], b[offset]);
    }
}
__global__ void addTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size, int * shape, int * stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset;
        get_idx(offset, i, shape, stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset], b);
    }
}


extern void c_cudaMalloc(void** ptr, size_t size) {
    checkCudaError(cudaMalloc(ptr, size));
}

extern void c_cudaMemcpy(void* dst, void* src, size_t size,
                         c_cudaMemcpyKind kind) {
    checkCudaError(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
}

extern void c_cudaFree(void* src) { checkCudaError(cudaFree(src)); }

extern void addMM(void* c, void* a, void* b, size_t size) {
    data_t* dev_c = (data_t*)c;
    data_t* dev_a = (data_t*)a;
    data_t* dev_b = (data_t*)b;
    size_t bytes = size * sizeof(data_t);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addMMKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
}


extern void addKernel(void* dst, Tensor a, Tensor b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_b = (data_t*)b.data.dp;
    data_t* dev_c = (data_t*)dst;
    size_t bytes = size * sizeof(data_t);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int * shape;
    int * stide;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stide, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(), a.shape.shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stide, a.stride.data(), a.stride.size() * sizeof(int), cudaMemcpyHostToDevice));

    addTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b,  size, shape, stide, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
}

extern void addKernelNum(void *dst, Tensor a, data_t b, size_t size) {
    data_t* dev_a = (data_t*)a.data.dp;
    data_t* dev_c = (data_t*)dst;
    b = b.to_dt(a.dtype);
    size_t bytes = size * sizeof(data_t);
    size_t threadsPerBlock = THREAD_PER_BLOCK;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int * shape;
    int * stide;
    checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
    checkCudaError(cudaMalloc(&stide, a.stride.size() * sizeof(int)));
    checkCudaError(cudaMemcpy(shape, a.shape.shape.data(), a.shape.shape.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(stide, a.stride.data(), a.stride.size() * sizeof(int), cudaMemcpyHostToDevice));

    addTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, b,  size, shape, stide, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
}


}  // namespace ts
