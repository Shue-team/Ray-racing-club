#pragma once
#include "Common/Vector3D.h"

class Quaternion {
public:
    Quaternion(float scalar, const Vector3D& vector);
    Quaternion(float scalar, float xPos, float yPos, float zPos);
    
    static Quaternion fromAxisAndAngle(const Vector3D& axis, float angle);

    Vector3D rotate(const Vector3D& vector);
    
    Quaternion conjugated() const;
    
    Vector3D vector() const;
    float scalar() const;
private:
    float mReal;
    Vector3D mImag;
};

Quaternion operator*(const Quaternion& a, const Quaternion& b);