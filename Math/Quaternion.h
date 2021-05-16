#pragma once
#include "../Math/Vector3D.h"

class Quaternion {
public:
    __host__ __device__ Quaternion(float scalar, const Vector3D& vector);
    __host__ __device__ Quaternion(float scalar, float xPos, float yPos, float zPos);
    
    __host__ __device__ static Quaternion fromAxisAndAngle(const Vector3D& axis, float angle);

    __host__ __device__ Vector3D rotate(const Vector3D& vector);
    
    __host__ __device__ Quaternion conjugated() const;
    
    __host__ __device__ Vector3D vector() const;
    __host__ __device__ float scalar() const;
private:
    float mReal;
    Vector3D mImag;
};

__host__ __device__ Quaternion operator*(const Quaternion& a, const Quaternion& b);