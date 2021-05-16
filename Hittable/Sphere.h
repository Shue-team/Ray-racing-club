//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_SPHERE_H
#define RAY_RACING_CLUB_SPHERE_H

#include "Hittable.h"

class Sphere : public Hittable {
public:
    __host__ __device__ Sphere(const Point3D& center, float radius, Material* material)
        : mCenter(center), mRadius(radius), mMaterial(material) {}

    __host__ __device__ bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;

private:
    Point3D mCenter;
    float mRadius;
    Material* mMaterial;
};

#endif //RAY_RACING_CLUB_SPHERE_H
