//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RAY_H
#define RAY_RACING_CLUB_RAY_H

#include "Common/Math.h"
#include "Common/Vector3D.h"

class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3D& origin, const Vector3D& direction);

    __host__ __device__ Point3D origin() const;
    __host__ __device__ Vector3D direction() const;

    __host__ __device__ Point3D at(float t) const;

private:
    Point3D mOrigin;
    Vector3D mDirection;
};

#endif //RAY_RACING_CLUB_RAY_H
