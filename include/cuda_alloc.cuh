#pragma once
#include <cstddef>

namespace ts {
extern void c_cudaMalloc(void** ptr, size_t size);
extern void c_cudaFree(void* src);
extern void c_cudaFreeHost(void* ptr);
extern void* c_cudaMallocHost(size_t size);

template <class T>
class cudaAllocator {
   public:
    typedef T value_type;
    cudaAllocator() = default;
    template <class U>
    cudaAllocator(const cudaAllocator<U>&) {}
    T* allocate(std::size_t n) {
        void* ptr;
        c_cudaMalloc(&ptr, n * sizeof(T));
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) { c_cudaFree(p); }
};

template <class T>
class cudaHostAllocator {
   public:
    typedef T value_type;
    cudaHostAllocator() = default;
    template <class U>
    cudaHostAllocator(const cudaHostAllocator<U>&) {}
    T* allocate(std::size_t n) {
        void* ptr = c_cudaMallocHost(n * sizeof(T));
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) { c_cudaFreeHost(p); }
};

}