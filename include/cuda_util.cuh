#pragma once
#define THREAD_PER_BLOCK 256

namespace ts{
    void checkCudaError(cudaError_t err, const char* file, int line);

    #define checkCudaError(err) checkCudaError(err, __FILE__, __LINE__)
    
    __global__ void addMMKernel(double* c, double* a, double* b, int size);

    extern void addMM(void* c, void* a, void* b, int size);

}
