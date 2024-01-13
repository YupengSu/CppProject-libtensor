#pragma once
#include "config.hpp"

namespace ts{

    __global__ void addMMKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    extern "C" void addMM(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void addMNKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void subMMKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void subMNKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void mulMMKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void mulMNKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void divMMKernel(data_t* c, const data_t* a, const data_t* b, const int size);

    __global__ void divMNKernel(data_t* c, const data_t* a, const data_t* b, const int size);

}
