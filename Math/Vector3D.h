//
// Created by awesyr on 20.02.2021.
//

#ifndef RAY_RACING_CLUB_VECTOR3D_H
#define RAY_RACING_CLUB_VECTOR3D_H

#include "cuda_runtime.h"
#include <curand_kernel.h>

class Vector3D {
public:
    __host__ __device__ Vector3D();
    __host__ __device__ Vector3D(float x, float y, float z);

    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;

    __host__ __device__ Vector3D operator-() const;

    __host__ __device__ Vector3D& operator+=(const Vector3D& other);
    __host__ __device__ Vector3D& operator*=(const Vector3D& other);
    __host__ __device__ Vector3D& operator*=(float value);
    __host__ __device__ Vector3D& operator/=(float value);

    __host__ __device__ float operator[] (int i) const;
    __host__ __device__ float& operator[] (int i);

    __host__ __device__ float lengthSquared() const;
    __host__ __device__ float length() const;

    __host__ __device__ Vector3D normalized() const;

    __host__ __device__ static float dotProduct(const Vector3D& a, const Vector3D& b);
    __host__ __device__ static Vector3D crossProduct(const Vector3D& a, const Vector3D& b);

    __device__ static Vector3D getRandomInUnitDisk(curandState* randState);

    __host__ __device__ bool fuzzyIsNull() const;

    __device__ void atomicAddVec(const Vector3D& other);

    __device__ static Vector3D random(float min, float max, curandState* randState);
    __device__ static Vector3D randomInUnitSphere(curandState* randState);
    __device__ static Vector3D randomUnit(curandState* randState);

    __host__ __device__ static Vector3D reflect(const Vector3D& v, const Vector3D& n);
    __host__ __device__ static Vector3D refract(const Vector3D& uv, const Vector3D& n, float etaiOverEtat);

private:
    float mCoords[3];
};

using Point3D = Vector3D;
using Color = Vector3D;

__host__ __device__ Vector3D operator+(const Vector3D& a, const Vector3D& b);
__host__ __device__ Vector3D operator-(const Vector3D& a, const Vector3D& b);
__host__ __device__ Vector3D operator*(const Vector3D& a, const Vector3D& b);

__host__ __device__ Vector3D operator*(const Vector3D& a, float t);
__host__ __device__ Vector3D operator*(float t, const Vector3D& a);
__host__ __device__ Vector3D operator/(const Vector3D& a, float t);

#endif //RAY_RACING_CLUB_VECTOR3D_H
