#include "Quaternion.h"
#include <cmath>

Quaternion Quaternion::fromAxisAndAngle(const Vector3D& axis, float angle) {
    return Quaternion(cos(0.5f * angle), sin(0.5f * angle) * axis.normalized());
}

Quaternion::Quaternion(float scalar, const Vector3D& vector):
    mReal(scalar), mImag(vector) {}

Quaternion::Quaternion(float scalar, float xPos, float yPos, float zPos):
    mReal(scalar), mImag(xPos, yPos, zPos) {}

Vector3D Quaternion::rotate(const Vector3D& vector) {
    Quaternion result = *this * Quaternion(0, vector) * conjugated();
    return result.mImag;
}

Quaternion operator*(const Quaternion& a, const Quaternion& b) {
    float realPart = a.scalar() * b.scalar() - Vector3D::dotProduct(a.vector(), b.vector());
    Vector3D imagPart = a.scalar() * b.vector() + b.scalar() * a.vector()+ Vector3D::crossProduct(a.vector(), b.vector());
    return Quaternion(realPart, imagPart);
}

Quaternion Quaternion::conjugated() const {
    return Quaternion(mReal, -mImag);
}

Vector3D Quaternion::vector() const {
    return mImag;
}

float Quaternion::scalar() const {
    return mReal;
}
