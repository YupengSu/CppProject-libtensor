#include <cuda_runtime.h>

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
void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file
                  << ":" << line << std::endl;
        exit(1);
    }
}

#define checkCudaError(err) checkCudaError(err, __FILE__, __LINE__)

__global__ void get_data_t(data_t& dst, void* ptr) { dst = *(data_t*)ptr; }

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
    dst.dtype = a.dtype;
}

__device__ size_t get_idx(size_t index, const  int* shape_v,const  int* stride,
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
__global__ void addMMKernel(data_t* c, data_t* a, data_t* b, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        add_data_t(c[i], a[i], b[i]);
    }
}


__global__ void addTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                int* shape, int* stride_a, int* stride_b, int* origin_stride,
                                int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        
       size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
        size_t offset_b =  get_idx(i, shape, stride_b, origin_stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset_a], b[offset_b]);
    }
}
__global__ void addTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                   int* shape, int* stride, int*origin_stride, int dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
       size_t offset= get_idx(i, shape, stride, origin_stride, dim);
        // offset = i;
        add_data_t(c[i], a[offset], b);
    }
}

extern void c_cudaMalloc(void** ptr, size_t size) {
    checkCudaError(cudaMalloc(ptr, size));
}

extern void c_cudaMemcpy(void* dst, void* src, size_t size,
                         c_cudaMemcpyKind kind) {
                            // cerr<<size/sizeof(data_t)<<endl;
    checkCudaError(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
}

extern void c_cudaFree(void* src) { checkCudaError(cudaFree(src)); }

extern void addMM(void* c, void* a, void* b, size_t size) {
    data_t* dev_c = (data_t*)c;
    data_t* dev_a = (data_t*)a;
    data_t* dev_b = (data_t*)b;
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
        dev_c, dev_a, dev_b, size, shape, stride_a, stride_b,origin_stride, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride_a));
    checkCudaError(cudaFree(stride_b));
    checkCudaError(cudaFree(origin_stride));
}

extern void addKernelNum(void* dst, Tensor a, data_t b, size_t size) {
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
        dev_c, dev_a, b, size, shape, stride,origin_stride, a.get_dim());

    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaFree(shape));
    checkCudaError(cudaFree(stride));
    checkCudaError(cudaFree(origin_stride));
}

__global__ void get_serial_tensorMM(void* dst, void* src, size_t size, const  int *shape, const int *stride, const int *origin_stride, int dim) {
    data_t *dev_dst = (data_t*)dst;
    data_t *dev_src = (data_t*)src;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        size_t offset = get_idx(i, shape, stride, origin_stride, dim);
        dev_dst[i] = dev_src[offset];
    }

}

extern void get_serial_tensor_kernel(void* dst, Tensor a) {
    data_t *dev_dst = (data_t*)dst;
    data_t *dev_src = (data_t*)a.data.dp;
    int* shape;
    int* stride;
    int* origin_stride;
    size_t size = a.data.size;
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
    get_serial_tensorMM<<<blocksPerGrid, threadsPerBlock>>>(dev_dst, dev_src, size, shape, stride, origin_stride, a.get_dim());
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
}

extern void c_get_data_t(data_t& dst, void* ptr) {
    data_t* tmp;
    cudaMalloc(&tmp, sizeof(data_t));
    get_data_t<<<1, 1>>>(*tmp, ptr);
    cudaMemcpy(&dst, tmp, sizeof(data_t), cudaMemcpyDeviceToHost);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());
    cudaFree(tmp);
}
}  // namespace ts
