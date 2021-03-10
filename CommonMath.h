//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_COMMONMATH_H
#define RAY_RACING_CLUB_COMMONMATH_H

#include <limits>
#include "Vector3D.h"

constexpr float infinity = std::numeric_limits<float>::infinity();

inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline float randomFloat();

inline float randomFloat(float min, float max);

inline Vector3D random(float min, float max);

Vector3D randomInUnitSphere();

Vector3D reflect(const Vector3D& v, const Vector3D& n);

Vector3D randomUnitVector();

Vector3D refract(const Vector3D& uv, const Vector3D& n, float etaiOverEtat);

#endif //RAY_RACING_CLUB_COMMONMATH_H
