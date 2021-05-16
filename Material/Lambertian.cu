//
// Created by awesyr on 20.03.2021.
//

#include "Lambertian.h"

__device__ bool Lambertian::scatter(const Ray& rIn, const HitRecord& rec,
                                    Color& attenuation, Ray& scattered,
                                    curandState* randState) const {
    Vector3D scatterDirection = rec.normal + Vector3D::randomUnit(randState);

    if (scatterDirection.fuzzyIsNull())
        scatterDirection = rec.normal;

    scattered = Ray(rec.intersection, scatterDirection);
    attenuation = mAlbedo;
    return true;
}
