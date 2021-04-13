//
// Created by Comp on 13.04.2021.
//
#include "Triangle.h"
#define CULLING // When culling is active, rays intersecting triangles from behind will be discarded.

bool Triangle::hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const {
    const float epsilon = 1e-6;

    Vector3D p = Vector3D::crossProduct(ray.direction(), mEdge2);
    float det = Vector3D::dotProduct(mEdge1, p);

#ifdef CULLING
    if (det < epsilon)
        return false;
    Vector3D t = ray.origin() - mV0;
    double u = Vector3D::dotProduct(t, p);
    if(u < 0 || u > det)
        return false;
    Vector3D q = Vector3D::crossProduct(t, mEdge1);
    double v = Vector3D::dotProduct(ray.direction(), q);
    if (v < 0 || u + v > det)
        return false;
    record.t = Vector3D::dotProduct(mEdge2, q) / det;
#else
    if (det > -epsilon && det < epsilon)
        return false;
    Vector3D t = ray.origin() - mV0;
    double u = Vector3D::dotProduct(t, p) / det;
    if(u < 0 || u > 1)
        return false;
    Vector3D q = Vector3D::crossProduct(t, mEdge1);
    double v = Vector3D::dotProduct(ray.direction(), q) / det;
    if (v < 0 || u + v > 1)
        return false;
    record.t = Vector3D::dotProduct(mEdge2, q) / det;
#endif
    if (record.t < tMin || record.t > tMax) {
        return false;
    }
    record.intersection = ray.at(record.t);
    record.setFaceNormal(ray, mNormal);
    record.matPtr = mMaterial;
    return true;
}