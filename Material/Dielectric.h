//
// Created by awesyr on 25.03.2021.
//

#ifndef RAY_RACING_CLUB_DIELECTRIC_H
#define RAY_RACING_CLUB_DIELECTRIC_H

#include "Material.h"

struct DielectricDef : MaterialDef {
    DielectricDef(float refractIdx)
    : MaterialDef(MaterialType::Dielectric),
    refractIdx(refractIdx) {}


    float refractIdx;
};

class Dielectric : public Material {
public:
    __host__ __device__ Dielectric(const DielectricDef* def)
        : mRefractIdx(def->refractIdx) {}

    __device__ bool scatter(const Ray& rIn, const HitRecord& rec,
                            Color& attenuation, Ray& scattered,
                            curandState* randState) const override;

private:
    __host__ __device__ static float reflectance(float cos, float refractRatio);

    float mRefractIdx;
};

#endif //RAY_RACING_CLUB_DIELECTRIC_H
