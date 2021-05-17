#ifndef HITTABLEDEF_H
#define HITTABLEDEF_H

#include "Hittable.h"
#include "../ErrorProcessing/ErrorHandling.h"
#include <type_traits>

enum class HittableType {
    Sphere,
};

struct HittableDef {
    virtual ~HittableDef() = default;

    __host__ __device__ virtual HittableType type() const = 0;

    template<typename T>
    static inline HittableDef* transferToGPU(const T* ptr) {
        static_assert(std::is_base_of<HittableDef, T>::value);
        T* gpuPtr;
        catchError(cudaMalloc(&gpuPtr, sizeof(T)));
        catchError(cudaMemcpy(gpuPtr, ptr, sizeof(T), cudaMemcpyHostToDevice));
        return gpuPtr;
    }
};

__host__ __device__ Hittable* createHittable(const HittableDef* def, Material* material);

#endif // HITTABLEDEF_H
