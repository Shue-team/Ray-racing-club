//
// Created by awesyr on 25.03.2021.
//

#include "Dielectric.h"
#include "../Math/Rand.h"
#include <cmath>

__device__ bool Dielectric::scatter(const Ray& rIn, const HitRecord& rec,
                                    Color& attenuation, Ray& scattered,
                                    curandState* randState) const {
    attenuation = Color(1.0f, 1.0f, 1.0f);
    float refractRatio = rec.frontFace ? (1.0f / mRefractIdx) : mRefractIdx;

    Vector3D unitDirection = rIn.direction().normalized();
    float cosTheta = fmin(Vector3D::dotProduct(-unitDirection, rec.normal), 1.0f);
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    bool cannotRefract = refractRatio * sinTheta > 1.0f;
    Vector3D direction;

    if (cannotRefract || reflectance(cosTheta, refractRatio) > randomFloat(randState))
        direction = Vector3D::reflect(unitDirection, rec.normal);
    else
        direction = Vector3D::refract(unitDirection, rec.normal, refractRatio);

    scattered = Ray(rec.intersection, direction);
    return true;
}

float Dielectric::reflectance(float cos, float refractRatio) {
    float r0 = (1.0f - refractRatio) / (1.0f + refractRatio);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cos), 5.0f);
}

