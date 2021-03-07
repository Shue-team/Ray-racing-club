//
// Created by Comp on 01.03.2021.
//

#ifndef RAY_RACING_CLUB_UTILITY_H
#define RAY_RACING_CLUB_UTILITY_H

#include "../Vector3D.h"

inline float randomFloat();

inline float randomFloat(float min, float max);

inline Vector3D random(float min, float max);

Vector3D randomInUnitSphere();

Vector3D reflect(const Vector3D& v, const Vector3D& n);

Vector3D randomUnitVector();

Vector3D refract(const Vector3D& uv, const Vector3D& n, float etaiOverEtat);

#endif //RAY_RACING_CLUB_UTILITY_H
