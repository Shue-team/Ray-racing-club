//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_SPHERE_H
#define RAY_RACING_CLUB_SPHERE_H

#include "Hittable.h"
#include "HittableDef.h"

struct SphereDef : HittableDef {
    SphereDef(const Point3D& center, float radius)
        : center(center), radius(radius) {}

    __host__ __device__ HittableType type() const override;

    Point3D center;
    float radius;
};

class Sphere : public Hittable {
public:
    __host__ __device__ Sphere(const SphereDef* def, Material* material);

    __host__  __device__ bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;

private:
    Point3D mCenter;
    float mRadius;
    Material* mMaterial;
};

#endif //RAY_RACING_CLUB_SPHERE_H
