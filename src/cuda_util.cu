
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
    extern void checkCudaErrorFunc(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file
                    << ":" << line << std::endl;
            exit(1);
        }
    }

    #define checkCudaError(err) checkCudaErrorFunc(err, __FILE__, __LINE__)

    extern void c_cudaMalloc(void** ptr, size_t size) {
        checkCudaError(cudaMalloc(ptr, size));
    }

    extern void* c_cudaMallocHost(size_t size) {
        void *ptr;
        checkCudaError(cudaMallocHost(&ptr, size));
        return ptr;
    }
    extern void c_cudaFreeHost(void * ptr) {
        checkCudaError(cudaFreeHost(ptr));
    }

    extern void c_cudaMemcpy(void* dst, void* src, size_t size,
                            c_cudaMemcpyKind kind) {
        checkCudaError(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
    }

    extern void c_cudaFree(void* src) { checkCudaError(cudaFree(src)); }

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

    extern void get_serial_tensor_kernel(void* dst, const Tensor a) {

        data_t* dev_dst = (data_t*)dst;
        data_t* dev_src = (data_t*)a.data.dp;
        int* shape;
        int* stride;
        int* origin_stride;
        size_t size = a.shape.data_len();
        checkCudaError(cudaMalloc(&shape, a.shape.shape.size() * sizeof(int)));
        checkCudaError(cudaMalloc(&stride, a.stride.size() * sizeof(int)));
        checkCudaError(cudaMalloc(&origin_stride, a.shape.shape.size() * sizeof(int)));
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

    __device__ void sub_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_int8 = a.data.tensor_int8 - b.data.tensor_int8;
                break;
            case dt::float32:
                dst.data.tensor_float32 =
                    a.data.tensor_float32 - b.data.tensor_float32;
                break;
            case dt::bool8:
                dst.data.tensor_bool = a.data.tensor_bool - b.data.tensor_bool;
                break;
            case dt::int32:
                dst.data.tensor_int32 = a.data.tensor_int32 - b.data.tensor_int32;
                break;
            case dt::float64:
                dst.data.tensor_float64 =
                    a.data.tensor_float64 - b.data.tensor_float64;
                break;
        }
        dst.dtype = a.dtype;
    }

    __device__ void mul_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_int8 = a.data.tensor_int8 * b.data.tensor_int8;
                break;
            case dt::float32:
                dst.data.tensor_float32 =
                    a.data.tensor_float32 * b.data.tensor_float32;
                break;
            case dt::bool8:
                dst.data.tensor_bool = a.data.tensor_bool * b.data.tensor_bool;
                break;
            case dt::int32:
                dst.data.tensor_int32 = a.data.tensor_int32 * b.data.tensor_int32;
                break;
            case dt::float64:
                dst.data.tensor_float64 =
                    a.data.tensor_float64 * b.data.tensor_float64;
                break;
        }
        dst.dtype = a.dtype;
    }

    __device__ void div_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_int8 = a.data.tensor_int8 / b.data.tensor_int8;
                break;
            case dt::float32:
                dst.data.tensor_float32 =
                    a.data.tensor_float32 / b.data.tensor_float32;
                break;
            case dt::bool8:
                dst.data.tensor_bool = a.data.tensor_bool / b.data.tensor_bool;
                break;
            case dt::int32:
                dst.data.tensor_int32 = a.data.tensor_int32 / b.data.tensor_int32;
                break;
            case dt::float64:
                dst.data.tensor_float64 =
                    a.data.tensor_float64 / b.data.tensor_float64;
                break;
        }
        dst.dtype = a.dtype;
    }

    __device__ void eq_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 == b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 == b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool == b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 == b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 == b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __device__ void ne_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 != b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 != b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool != b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 != b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 != b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __device__ void gt_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 > b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 > b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool > b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 > b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 > b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __device__ void ge_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 >= b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 >= b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool >= b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 >= b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 >= b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __device__ void lt_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 < b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 < b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool < b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 < b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 < b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __device__ void le_data_t(data_t& dst, data_t& a, data_t& b) {
        switch (a.dtype) {
            case dt::int8:
                dst.data.tensor_bool = (a.data.tensor_int8 <= b.data.tensor_int8);
                break;
            case dt::float32:
                dst.data.tensor_bool =
                    (a.data.tensor_float32 <= b.data.tensor_float32);
                break;
            case dt::bool8:
                dst.data.tensor_bool = (a.data.tensor_bool <= b.data.tensor_bool);
                break;
            case dt::int32:
                dst.data.tensor_bool = (a.data.tensor_int32 <= b.data.tensor_int32);
                break;
            case dt::float64:
                dst.data.tensor_bool =
                    (a.data.tensor_float64 <= b.data.tensor_float64);
                break;
        }
        dst.dtype = dt::bool8;
    }

    __global__ void addTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                    int* shape, int* stride_a, int* stride_b,
                                    int* origin_stride, int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
            size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
            // offset = i;
            add_data_t(c[i], a[offset_a], b[offset_b]);
        }
    }

    __global__ void addTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                    int* shape, int* stride, int* origin_stride,
                                    int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset = get_idx(i, shape, stride, origin_stride, dim);
            // offset = i;
            add_data_t(c[i], a[offset], b);
        }
    }

    __global__ void subTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                    int* shape, int* stride_a, int* stride_b,
                                    int* origin_stride, int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
            size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
            // offset = i;
            sub_data_t(c[i], a[offset_a], b[offset_b]);
        }
    }

    __global__ void subTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                    int* shape, int* stride, int* origin_stride,
                                    int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset = get_idx(i, shape, stride, origin_stride, dim);
            // offset = i;
            sub_data_t(c[i], a[offset], b);
        }
    }

    __global__ void mulTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                    int* shape, int* stride_a, int* stride_b,
                                    int* origin_stride, int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
            size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
            // offset = i;
            mul_data_t(c[i], a[offset_a], b[offset_b]);
        }
    }

    __global__ void mulTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                    int* shape, int* stride, int* origin_stride,
                                    int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset = get_idx(i, shape, stride, origin_stride, dim);
            // offset = i;
            mul_data_t(c[i], a[offset], b);
        }
    }

    __global__ void divTensorKernel(data_t* c, data_t* a, data_t* b, size_t size,
                                    int* shape, int* stride_a, int* stride_b,
                                    int* origin_stride, int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset_a = get_idx(i, shape, stride_a, origin_stride, dim);
            size_t offset_b = get_idx(i, shape, stride_b, origin_stride, dim);
            // offset = i;
            div_data_t(c[i], a[offset_a], b[offset_b]);
        }
    }

    __global__ void divTensorKernelNum(data_t* c, data_t* a, data_t b, size_t size,
                                    int* shape, int* stride, int* origin_stride,
                                    int dim) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset = get_idx(i, shape, stride, origin_stride, dim);
            // offset = i;
            div_data_t(c[i], a[offset], b);
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

    __global__ void matrixMultiplyKernel(data_t* c, data_t* a, data_t* b,
                                        size_t size, int* shape_a, int* shape_b,
                                        int* stride_a, int* stride_b,
                                        int* origin_stride_a,
                                        int* origin_stride_b, int dim_a,
                                        int dim_b) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            size_t offset_a = get_idx(i, shape_a, stride_a, origin_stride_a, dim_a);
            size_t offset_b = get_idx(i, shape_b, stride_b, origin_stride_b, dim_b);
            // offset = i;
            mul_data_t(c[i], a[offset_a], b[offset_b]);
        }
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
            dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
            a.get_dim());

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
            dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void subKernel(void* dst, Tensor a, Tensor b, size_t size) {
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
            a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride_a));
        checkCudaError(cudaFree(stride_b));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void subKernelNum(void* dst, Tensor a, data_t b, size_t size) {
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
            dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void mulKernel(void* dst, Tensor a, Tensor b, size_t size) {
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
            a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride_a));
        checkCudaError(cudaFree(stride_b));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void mulKernelNum(void* dst, Tensor a, data_t b, size_t size) {
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
            dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void divKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

        divTensorKernel<<<blocksPerGrid, threadsPerBlock>>>(
            dev_c, dev_a, dev_b, size, shape, stride_a, stride_b, origin_stride,
            a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride_a));
        checkCudaError(cudaFree(stride_b));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void divKernelNum(void* dst, Tensor a, data_t b, size_t size) {
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

        divTensorKernelNum<<<blocksPerGrid, threadsPerBlock>>>(
            dev_c, dev_a, b, size, shape, stride, origin_stride, a.get_dim());

        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
        checkCudaError(cudaFree(shape));
        checkCudaError(cudaFree(stride));
        checkCudaError(cudaFree(origin_stride));
    }

    extern void eqKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

    extern void neKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

    extern void gtKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

    extern void geKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

    extern void ltKernel(void* dst, Tensor a, Tensor b, size_t size) {
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

    extern void leKernel(void* dst, Tensor a, Tensor b, size_t size) {
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




}  // namespace ts
