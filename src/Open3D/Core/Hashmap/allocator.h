#include "config.h"
#include "helper_cuda.h"

#pragma once

class Allocator {
public:
    Allocator(int device_id = 0) : device_id_(device_id) {}
    void* allocate(size_t size) { return nullptr; }

    void deallocate(void* ptr) {}

protected:
    int device_id_;
};

class CudaAllocator : public Allocator {
public:
    CudaAllocator(int device_id = 0) : Allocator(device_id) {}
    void* allocate(size_t size) {
        void* ptr;
        CHECK_CUDA(cudaMalloc(&ptr, size));
        return ptr;
    }

    void deallocate(void* ptr) { CHECK_CUDA(cudaFree(ptr)); }
};
