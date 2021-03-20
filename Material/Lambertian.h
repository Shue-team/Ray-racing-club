//
// Created by awesyr on 20.03.2021.
//

#ifndef RAY_RACING_CLUB_LAMBERTIAN_H
#define RAY_RACING_CLUB_LAMBERTIAN_H

#include "Material.h"

class Lambertian : public Material {
public:
    __host__ __device__ Lambertian(const Color& albedo) : mAlbedo(albedo) {}

    __host__ __device__ bool scatter(const Ray& rIn, const HitRecord& rec,
                                     Color& attenuation, Ray& scattered,
                                     curandState* randState) const override;

private:
    Color mAlbedo;
};


#endif //RAY_RACING_CLUB_LAMBERTIAN_H
