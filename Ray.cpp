//
// Created by awesyr on 06.03.2021.
//

#include "Ray.h"

Ray::Ray(const Point3D& origin, const Vector3D& direction)
    : mOrigin(origin), mDirection(direction) {}

Point3D Ray::origin() const { return mOrigin; }

Vector3D Ray::direction() const { return mDirection;}

Point3D Ray::at(float t) const { return mOrigin + t * mDirection; }









