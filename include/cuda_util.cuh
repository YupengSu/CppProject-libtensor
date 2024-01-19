#pragma once
#include <cstddef>

#include "serial_tensor.hpp"

#define THREAD_PER_BLOCK 256

namespace ts {
enum c_cudaMemcpyKind {
    c_cudaMemcpyHostToHost = 0,     /**< Host   -> Host */
    c_cudaMemcpyHostToDevice = 1,   /**< Host   -> Device */
    c_cudaMemcpyDeviceToHost = 2,   /**< Device -> Host */
    c_cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
    c_cudaMemcpyDefault =
        4 /**< Direction of the transfer is inferred from the pointer values.
             Requires unified virtual addressing */
};
void c_cudaMalloc(void** ptr, size_t size);
void c_cudaFree(void* src);
void c_cudaMemcpy(void* dst, void* src, size_t count,
                         c_cudaMemcpyKind kind);
void c_cudaFreeHost(void* ptr);
void* c_cudaMallocHost(size_t size);

void addKernel(void* dst, Tensor a, Tensor b, size_t size, dt target_dtype);
void addKernelNum(void* dst, Tensor a, data_t b, size_t size, dt target_dtype);

void subKernel(void* dst, Tensor a, Tensor b, size_t size, dt target_dtype);
void subKernelNum(void* dst, Tensor a, data_t b, size_t size, dt target_dtype);

void mulKernel(void* dst, Tensor a, Tensor b, size_t size, dt target_dtype);
void mulKernelNum(void* dst, Tensor a, data_t b, size_t size, dt target_dtype);

void divKernel(void* dst, Tensor a, Tensor b, size_t size);
void divKernelNum(void* dst, Tensor a, data_t b, size_t size);


void eqKernel(void* dst, Tensor a, Tensor b, size_t size);

void ltKernel(void* dst, Tensor a, Tensor b, size_t size);

void gtKernel(void* dst, Tensor a, Tensor b, size_t size);

void leKernel(void* dst, Tensor a, Tensor b, size_t size);

void geKernel(void* dst, Tensor a, Tensor b, size_t size);

void neKernel(void* dst, Tensor a, Tensor b, size_t size);



void get_serial_tensor_kernel(void* dst, const Tensor a);


}  // namespace ts
