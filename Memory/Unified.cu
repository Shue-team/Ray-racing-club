//
// Created by awesyr on 26.02.2021.
//

#include "Unified.h"
#include "../ErrorProcessing/ErrorHandling.h"

void* Unified::operator new(size_t len) {
    void* ptr;
    catchError(cudaMallocManaged(&ptr, len));
    return ptr;
}

void Unified::operator delete(void* ptr) {
    catchError(cudaFree(ptr));
}
