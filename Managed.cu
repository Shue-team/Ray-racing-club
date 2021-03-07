//
// Created by awesyr on 26.02.2021.
//

#include "Managed.h"
#include "Common/ErrorHandling.h"

void* Managed::operator new(size_t len) {
    void* ptr;
    catchError(cudaMallocManaged(&ptr, len));
    return ptr;
}

void Managed::operator delete(void* ptr) {
    catchError(cudaFree(ptr));
}
