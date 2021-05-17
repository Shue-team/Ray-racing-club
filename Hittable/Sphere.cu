//
// Created by arseny on 07.02.2021.
//

#include <cmath>
#include "Sphere.h"
#include "../Material/Material.h"

Sphere::Sphere(const SphereDef* def, Material *material)
    : mCenter(def->center),
      mRadius(def->radius),
      mMaterial(material) {}

bool Sphere::hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const {
    Point3D oc = ray.origin() - mCenter;
    float a = ray.direction().lengthSquared();
    float b = Vector3D::dotProduct(oc, ray.direction());
    float c = oc.lengthSquared() - mRadius * mRadius;

    float discriminant = b * b - a * c;
    if (discriminant < 0) { return false; }
    float sqrtd = sqrt(discriminant);

    float root = (-b - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-b + sqrtd) / a;

        if (root < tMin || root > tMax) {
            return false;
        }
    }

    record.t = root;
    record.intersection = ray.at(root);
    record.material = mMaterial;

    Vector3D outwardNormal = (record.intersection - mCenter) / mRadius;
    record.setFaceNormal(ray, outwardNormal);

    return true;
}

HittableType SphereDef::type() const {
    return HittableType::Sphere;
}
