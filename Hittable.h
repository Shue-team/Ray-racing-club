//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_HITTABLE_H
#define RAY_RACING_CLUB_HITTABLE_H

#include "Ray.h"

struct HitRecord {
    QPoint3D intersection;
    QVector3D normal;
    float t;
};

class Hittable {
    virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const = 0;
};

#endif //RAY_RACING_CLUB_HITTABLE_H
