#pragma once
#include <cstddef>
#include "serial_tensor.hpp"

#define THREAD_PER_BLOCK 256

namespace ts {
    enum  c_cudaMemcpyKind
    {
        c_cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
        c_cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
        c_cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
        c_cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
        c_cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
    };
    extern void c_cudaMalloc(void** ptr, size_t size);
    extern void c_cudaFree(void* src);
    extern void c_cudaMemcpy(void* dst, void* src, size_t count,c_cudaMemcpyKind kind);
    extern void addMM(void* c, void* a, void* b, int size);
    extern void addKernel(void* dst, Tensor a, Tensor b, size_t size);
    extern void addKernelNum(void* dst, Tensor a, data_t b, size_t size);
    extern void c_get_data_t(data_t& dst, void * ptr);
}
