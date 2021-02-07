//
// Created by arseny on 07.02.2021.
//

#include <cmath>
#include "Sphere.h"

bool Sphere::hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const {
    Point3D oc = ray.origin() - mCenter;
    double a = ray.direction().lengthSquared();
    double b = QVector3D::dotProduct(oc, ray.direction());
    double c = oc.lengthSquared() - mRadius * mRadius;

    double discriminant = b * b - a * c;
    if (discriminant < 0) { return false; }
    double sqrtd = sqrt(discriminant);

    double root = (-b - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-b + sqrtd) / a;

        if (root < tMin || root > tMax) {
            return false;
        }
    }

    record.t = root;
    record.intersection = ray.at(root);
    record.normal = (record.intersection - mCenter) / mRadius;

    return true;
}
