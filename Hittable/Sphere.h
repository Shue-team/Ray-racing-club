//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_SPHERE_H
#define RAY_RACING_CLUB_SPHERE_H

#include "Hittable.h"

class Sphere : public Hittable {
public:
    Sphere(const Point3D& center, float radius)
        : mCenter(center), mRadius(radius) {}

    bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;

private:
    Point3D mCenter;
    float mRadius;
};

#endif //RAY_RACING_CLUB_SPHERE_H
