//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_SPHERE_H
#define RAY_RACING_CLUB_SPHERE_H

#include "Hittable.h"
#include "../Materials/Material.h"

class Sphere : public Hittable {
public:
    Sphere(const Point3D& center, float radius, Material* m)
        : mCenter(center), mRadius(radius), material(m) {}

    bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;
    ~Sphere(){delete material;}

private:
    Point3D mCenter;
    float mRadius;
    Material* material;
};

#endif //RAY_RACING_CLUB_SPHERE_H
