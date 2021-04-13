//
// Created by Comp on 12.04.2021.
//

#ifndef RAY_RACING_CLUB_TRIANGLE_H
#define RAY_RACING_CLUB_TRIANGLE_H

#include "Hittable.h"
#include "../Materials/Material.h"

class Triangle : public Hittable {
public:
    Triangle(const Point3D& v0, const Point3D& v1, const Point3D& v2, Material* material)
            : mV0(v0), mEdge1(v1 - v0), mEdge2(v2 - v0),
            mNormal(Vector3D::crossProduct(mEdge1, mEdge2)), mMaterial(material) {}

    bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;
    ~Triangle(){delete mMaterial;}

private:
    Point3D mV0;
    Vector3D mEdge1;
    Vector3D mEdge2;
    Point3D mNormal;
    Material* mMaterial;
};


#endif //RAY_RACING_CLUB_TRIANGLE_H
