#include "Quaternion.h"
#include <cmath>
#include <iostream>

Quaternion::Quaternion(float angle, const Vector3D& axis) {
    mReal = cos(0.5f * angle);
    mImag = sin(0.5f * angle) * axis.normalized();
}

Quaternion::Quaternion(float s, float x, float y, float z):mImag(x, y, z) {
    mReal = s;
}

Vector3D Quaternion::rotate(const Vector3D& vector) {
    Quaternion reversed = getConjugate();
    Quaternion result = *this * vector;
    result = result * reversed;
    return result.mImag;
}

Quaternion Quaternion::operator*(const Vector3D& a) {
    Vector3D imagPart = mReal * a + Vector3D::crossProduct(mImag, a);
    float realPart = -1 * Vector3D::dotProduct(mImag, a);
    return Quaternion(realPart, imagPart[0], imagPart[1], imagPart[2]);
}

Quaternion Quaternion::operator*(const Quaternion& a) {
    float realPart = mReal * a.mReal - Vector3D::dotProduct(mImag, a.mImag);
    Vector3D imagPart = mReal * a.mImag + a.mReal * mImag + Vector3D::crossProduct(mImag, a.mImag);
    return Quaternion(realPart, imagPart[0], imagPart[1], imagPart[2]);
}

Quaternion Quaternion::getConjugate() {
    return Quaternion(mReal, -mImag[0], -mImag[1], -mImag[2]);
}
