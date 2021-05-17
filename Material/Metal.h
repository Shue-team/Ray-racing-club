#ifndef RAY_RACING_CLUB_METAL_H
#define RAY_RACING_CLUB_METAL_H

#include "Material.h"

struct MetalDef : MaterialDef {
    MetalDef(const Color& albedo, float fuzz)
        : MaterialDef(MaterialType::Metal),
        albedo(albedo), fuzz(fuzz) {}

    Color albedo;
    float fuzz;
};

class Metal : public Material {
public:
    __host__ __device__ Metal(const MetalDef* def)
        : mAlbedo(def->albedo),
          mFuzz(def->fuzz < 1.0f ? def->fuzz : 1.0f) {}

    __device__ bool scatter(const Ray& rIn, const HitRecord& rec,
                            Color& attenuation, Ray& scattered,
                            curandState* randState) const override;

private:
    Color mAlbedo;
    float mFuzz;
};

#endif //RAY_RACING_CLUB_METAL_H
