#include "cuda_util.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_PER_BLOCK 256

namespace ts {
    void checkCudaError(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }
    
    #define checkCudaError(err) checkCudaError(err, __FILE__, __LINE__)
    
    __global__ void addMMKernel(double* c, double* a, double* b, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            c[i] = a[i] + b[i];
        }
    }
    
    extern void addMM(void* c, void* a, void* b, int size) {
        double* dev_c;
        double* dev_a;
        double* dev_b;
        size_t bytes = size * sizeof(double);
    
        checkCudaError(cudaMalloc(&dev_c, bytes));
        checkCudaError(cudaMalloc(&dev_a, bytes));
        checkCudaError(cudaMalloc(&dev_b, bytes));
        
        checkCudaError(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

        size_t threadsPerBlock = THREAD_PER_BLOCK;
        size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        addMMKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost));

        checkCudaError(cudaFree(dev_c));
        checkCudaError(cudaFree(dev_a));
        checkCudaError(cudaFree(dev_b));
    }
}

