//
// Created by awesyr on 26.02.2021.
//

#include "Managed.h"

void* Managed::operator new(size_t len) {
    void* ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
}

void Managed::operator delete(void* ptr) {
    cudaFree(ptr);
}
