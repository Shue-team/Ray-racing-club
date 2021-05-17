#ifndef HITTABLEDEF_H
#define HITTABLEDEF_H

#include "Hittable.h"
#include "../ErrorProcessing/ErrorHandling.h"
#include <type_traits>

enum class HittableType {
    Sphere,
};

struct HittableDef {
    HittableDef(HittableType type) : mType(type) {}

    __host__ __device__ HittableType type() const { return mType; }

    template<typename T>
    static inline HittableDef* transferToGPU(const T* ptr) {
        static_assert(std::is_base_of<HittableDef, T>::value, "T must inherits HittableDef");
        T* gpuPtr;
        catchError(cudaMalloc(&gpuPtr, sizeof(T)));
        catchError(cudaMemcpy(gpuPtr, ptr, sizeof(T), cudaMemcpyHostToDevice));
        return gpuPtr;
    }

private:
    HittableType mType;
};

__host__ __device__ Hittable* createHittable(const HittableDef* def, Material* material);

#endif // HITTABLEDEF_H
