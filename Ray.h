//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RAY_H
#define RAY_RACING_CLUB_RAY_H

#include "CommonMath.h"
#include "Vector3D.h"

class Ray {
public:
    Ray() {}
    Ray(const Point3D& origin, const Vector3D& direction)
        : mOrigin(origin), mDirection(direction) {}

    Point3D origin() const { return mOrigin; }
    Vector3D direction() const { return mDirection;}

    Point3D at(float t) const { return mOrigin + t * mDirection; }

private:
    Point3D mOrigin;
    Vector3D mDirection;
};

#endif //RAY_RACING_CLUB_RAY_H
