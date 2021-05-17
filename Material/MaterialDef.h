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
    virtual ~MaterialDef() = default;

    __host__ __device__ virtual MaterialType type() const = 0;

    template<typename T>
    static inline MaterialDef* transferToGPU(const T* ptr) {
        static_assert(std::is_base_of<MaterialDef, T>::value);
        T* gpuPtr;
        catchError(cudaMalloc(&gpuPtr, sizeof(T)));
        catchError(cudaMemcpy(gpuPtr, ptr, sizeof(T), cudaMemcpyHostToDevice));
        return gpuPtr;
    }
};

__host__ __device__ Material* createMaterial(const MaterialDef* def);

#endif // MATERIALDEF_H
