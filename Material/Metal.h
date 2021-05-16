//
// Created by awesyr on 24.03.2021.
//

#ifndef RAY_RACING_CLUB_METAL_H
#define RAY_RACING_CLUB_METAL_H

#include "Material.h"

class Metal : public Material {
public:
    __host__ __device__ Metal(const Color& albedo, float fuzz)
        : mAlbedo(albedo), mFuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

    __device__ bool scatter(const Ray& rIn, const HitRecord& rec,
                            Color& attenuation, Ray& scattered,
                            curandState* randState) const override;

private:
    Color mAlbedo;
    float mFuzz;
};

#endif //RAY_RACING_CLUB_METAL_H
