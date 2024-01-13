#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <iostream>

#include "config.hpp"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "storage.hpp"

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
__global__ void addMMKernel(data_t* c, data_t* a, data_t* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        add_data_t(c[i], a[i], b[i]);
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

extern void addMM(void* c, void* a, void* b, int size) {
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
}  // namespace ts
