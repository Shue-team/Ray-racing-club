//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_HITTABLE_H
#define RAY_RACING_CLUB_HITTABLE_H

#include "../Math/Ray.h"

class Material;

struct HitRecord {
    Point3D intersection;
    Vector3D normal;
    Material* material;
    float t;
    bool frontFace;

    __host__ __device__ inline void setFaceNormal(const Ray &ray, const Vector3D &outwardNormal) {
        frontFace = Vector3D::dotProduct(outwardNormal, ray.direction()) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    __host__ __device__ virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const = 0;

    __host__ __device__ virtual ~Hittable() {}
};

#endif //RAY_RACING_CLUB_HITTABLE_H
