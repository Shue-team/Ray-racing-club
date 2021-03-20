//
// Created by awesyr on 20.03.2021.
//

#ifndef RAY_RACING_CLUB_MATERIAL_H
#define RAY_RACING_CLUB_MATERIAL_H

#include <curand_kernel.h>
#include "../Hittable/Hittable.h"

class Material {
public:
    __host__ __device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec,
                                             Color& attenuation, Ray& scattered,
                                             curandState* randState) const = 0;
};

#endif //RAY_RACING_CLUB_MATERIAL_H
