//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_HITTABLE_H
#define RAY_RACING_CLUB_HITTABLE_H

#include "../Ray.h"
class Material;

struct HitRecord {
    Point3D intersection;
    Vector3D normal;
    Material* matPtr;
    float t;
    bool frontFace;

    inline void setFaceNormal(const Ray& ray, const Vector3D& outwardNormal) {
        frontFace = Vector3D::dotProduct(outwardNormal, ray.direction()) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const = 0;
    virtual ~Hittable() {};
};

#endif //RAY_RACING_CLUB_HITTABLE_H
