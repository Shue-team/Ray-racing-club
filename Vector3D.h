//
// Created by awesyr on 20.02.2021.
//

#ifndef CMAKE_AND_CUDA_VECTOR3D_H
#define CMAKE_AND_CUDA_VECTOR3D_H

class Vector3D {
public:
    Vector3D();
    Vector3D(float x, float y, float z);

    float x() const;
    float y() const;
    float z() const;

    Vector3D operator-() const;

    Vector3D& operator+=(const Vector3D& other);
    Vector3D& operator*=(float value);
    Vector3D& operator/=(float value);

    float operator[] (int i) const;
    float& operator[] (int i);

    float lengthSquared() const;
    float length() const;

    Vector3D normalized() const;

    static float dotProduct(const Vector3D& a, const Vector3D& b);
    static Vector3D crossProduct(const Vector3D& a, const Vector3D& b);


    bool nearZero() const;

private:
    float mCoords[3];
};

using Point3D = Vector3D;
using Color = Vector3D;

Vector3D operator+(const Vector3D& a, const Vector3D& b);
Vector3D operator-(const Vector3D& a, const Vector3D& b);

Vector3D operator*(const Vector3D& a, float t);
Vector3D operator*(float t, const Vector3D& a);
Vector3D operator*(const Vector3D &u, const Vector3D &v);
Vector3D operator/(const Vector3D& a, float t);

#endif //CMAKE_AND_CUDA_VECTOR3D_H
