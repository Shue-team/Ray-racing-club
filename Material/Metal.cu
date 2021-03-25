//
// Created by awesyr on 24.03.2021.
//

#include "Metal.h"

__device__ bool Metal::scatter(const Ray& rIn, const HitRecord& rec,
                               Color& attenuation, Ray& scattered,
                               curandState* randState) const {
    Vector3D reflected = Vector3D::reflect(rIn.direction().normalized(), rec.normal);
    scattered = Ray(rec.intersection, reflected + mFuzz * Vector3D::randomInUnitSphere(randState));
    attenuation = mAlbedo;
    return Vector3D::dotProduct(scattered.direction(), rec.normal) > 0.0f;
}