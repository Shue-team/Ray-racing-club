//
// Created by awesyr on 20.02.2021.
//

#include <cmath>

#include "Vector3D.h"

Vector3D::Vector3D() {
    mCoords[0] = 0.0f;
    mCoords[1] = 0.0f;
    mCoords[2] = 0.0f;
}

Vector3D::Vector3D(float x, float y, float z) {
    mCoords[0] = x;
    mCoords[1] = y;
    mCoords[2] = z;
}

float Vector3D::x() const {
    return mCoords[0];
}

float Vector3D::y() const {
    return mCoords[1];
}

float Vector3D::z() const {
    return mCoords[2];
}

Vector3D Vector3D::operator-() const {
    return Vector3D(-mCoords[0], -mCoords[1], -mCoords[2]);
}

Vector3D& Vector3D::operator+=(const Vector3D& other) {
    mCoords[0] += other.mCoords[0];
    mCoords[1] += other.mCoords[1];
    mCoords[2] += other.mCoords[2];
    return *this;
}

Vector3D& Vector3D::operator*=(float value) {
    mCoords[0] *= value;
    mCoords[1] *= value;
    mCoords[2] *= value;
    return *this;
}

Vector3D& Vector3D::operator/=(float value) {
    return *this *= 1.0f / value;
}

float Vector3D::operator[](int i) const {
    return mCoords[i];
}

float& Vector3D::operator[](int i) {
    return mCoords[i];
}

float Vector3D::lengthSquared() const {
    return mCoords[0] * mCoords[0] +
           mCoords[1] * mCoords[1] +
           mCoords[2] * mCoords[2];
}

float Vector3D::length() const {
    return sqrt(lengthSquared());
}

Vector3D Vector3D::normalized() const {
    return *this / length();
}

float Vector3D::dotProduct(const Vector3D& a, const Vector3D& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector3D Vector3D::crossProduct(const Vector3D& a, const Vector3D& b) {
    return Vector3D(a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]);
}

bool Vector3D::nearZero() const {
    const float s = 1e-8f;
    return (fabs(mCoords[0]) < s) && (fabs(mCoords[1]) < s) && (fabs(mCoords[2]) < s);
}

Vector3D operator+(const Vector3D& a, const Vector3D& b) {
    return Vector3D(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

Vector3D operator-(const Vector3D& a, const Vector3D& b) {
    return Vector3D(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

Vector3D operator*(const Vector3D& a, float t) {
    return Vector3D(a[0] * t, a[1] * t, a[2] * t);
}

Vector3D operator*(const Vector3D &u, const Vector3D &v) {
    return Vector3D(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

Vector3D operator*(float t, const Vector3D& a) {
    return a * t;
}

Vector3D operator/(const Vector3D& a, float t) {
    return 1.0f / t * a;
}

