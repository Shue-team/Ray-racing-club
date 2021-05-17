#ifndef MATERIALDEF_H
#define MATERIALDEF_H

#include <curand_kernel.h>
#include <type_traits>

#include "../ErrorProcessing/ErrorHandling.h"
#include "Material.h"

enum class MaterialType {
    Dielectric,
    Metal,
    Lambertian,
};

struct MaterialDef {
    MaterialDef(MaterialType type) : mType(type) {}

    __host__ __device__ MaterialType type() const { return mType; }
    template<typename T>
    static inline MaterialDef* transferToGPU(const T* ptr) {
        static_assert(std::is_base_of<MaterialDef, T>::value, "T must inherits MaterialDef");
        T* gpuPtr;
        catchError(cudaMalloc(&gpuPtr, sizeof(T)));
        catchError(cudaMemcpy(gpuPtr, ptr, sizeof(T), cudaMemcpyHostToDevice));
        return gpuPtr;
    }

private:
    MaterialType mType;
};

__host__ __device__ Material* createMaterial(const MaterialDef* def);

#endif // MATERIALDEF_H
