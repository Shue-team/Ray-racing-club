//
// Created by arseny on 07.02.2021.
//

#include "Camera.h"
#include <cmath>
#include "Quaternion.h"
#include <iostream>

Camera::Camera(float aspectRatio) {
    float viewportHeight = 2.0f;
    float viewportWidth = viewportHeight * aspectRatio;
    float focalLength = 1.0f;

    mOrigin = Point3D(0.0f, 0.0f, 0.0f);
    mHorizontal = Vector3D(viewportWidth, 0.0f, 0.0f);
    mVertical = Vector3D(0.0f, viewportHeight, 0.0f);

    Vector3D vUp(0.0f, 0.0f, focalLength);
    mBottomLeftCorner = mOrigin - mHorizontal / 2.0f - mVertical / 2.0f - vUp;
}

Ray Camera::getRay(float u, float v) const {
    return Ray(mOrigin, mBottomLeftCorner + u * mHorizontal + v * mVertical - mOrigin);
}

// move along camera's horizontal axi
void Camera::moveHorz(float dx) {
    Vector3D add(dx * mHorizontal.normalized());
    mBottomLeftCorner += add;
    mOrigin += add;
}

// move along camera's vertical axi
void Camera::moveVert(float dy) {
    Vector3D add(dy * mVertical.normalized());
    mBottomLeftCorner += add;
    mOrigin += add;
}

// moving screen to zoom
void Camera::moveDepth(float dz) {
    Vector3D add(Vector3D::crossProduct(mVertical, mHorizontal).normalized());
    mBottomLeftCorner += -dz * add;
    //mOrigin += z * add;
}

// rotation around camera's horizontal axi
void Camera::rotateOx(float alpha) {
    rotate(mHorizontal, alpha);
}

// rotation around camera's vertical axi
void Camera::rotateOy(float alpha) {
    rotate(mVertical, alpha);
}

/*
 * rotating hor vector, vert vector and screen using quaternion
 * screen need to be rotated using lookAt vector (from origin to leftBottom point)
*/
void Camera::rotate(const Vector3D &axis, float alpha) {
    Vector3D lookAt = mBottomLeftCorner - mOrigin;
    Quaternion rotQuat(alpha, axis);
    mVertical = rotQuat.rotate(mVertical);
    mHorizontal = rotQuat.rotate(mHorizontal);
    mBottomLeftCorner = mOrigin + rotQuat.rotate(lookAt);
} 
