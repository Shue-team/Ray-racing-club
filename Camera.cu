//
// Created by arseny on 07.02.2021.
//

#include "Camera.h"
#include <cmath>

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
void Camera::moveHorz(float x) {
    Vector3D add(x * mHorizontal.normalized());
    mBottomLeftCorner += add;
    mOrigin += add;
}

// move along camera's vertical axi
void Camera::moveVert(float y) {
    Vector3D add(y * mVertical.normalized());
    mBottomLeftCorner += add;
    mOrigin += add;
}

// moving screen to zoom
void Camera::moveDepth(float z) {
    Vector3D add(Vector3D::crossProduct(mVertical, mHorizontal).normalized());
    mBottomLeftCorner += -z * add;
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
 * rotating hor vector, vert vector and screen using rotation matrix from wikipedia (rotation around arbitrary axi)
 * screen need to be rotated using lookAt vector (from origin to leftBottom point)
*/
void Camera::rotate(const Vector3D &axi, float alpha) {
    Vector3D lookAt = mBottomLeftCorner - mOrigin;
    Vector3D vertRes, horRes, lookAtRes;
    float sinBuf = sin(alpha);
    float cosBuf = cos(alpha);
    Vector3D ax = axi.normalized();
    Vector3D matrix[3] = {
            {cosBuf + (1.f - cosBuf) * ax[0] * ax[0], (1.f - cosBuf) * ax[0] * ax[1] - sinBuf * ax[2], (1.f - cosBuf) * ax[0] * ax[2] + sinBuf * ax[1]},
            {(1.f - cosBuf) * ax[0] * ax[1] + sinBuf * ax[2], cosBuf + (1.f - cosBuf) * ax[1] * ax[1], (1.f - cosBuf) * ax[1] * ax[2] - sinBuf * ax[0]},
            {(1.f - cosBuf) * ax[0] * ax[2] - sinBuf * ax[1], (1.f - cosBuf) * ax[1] * ax[2] + sinBuf * ax[0], cosBuf + (1.f - cosBuf) * ax[2] * ax[2]}
    };
    // rotation matrix and vectors multiplication
    for (int i = 0; i < 3; i++) {
        vertRes[i] = Vector3D::dotProduct(matrix[i], mVertical);
        horRes[i] = Vector3D::dotProduct(matrix[i], mHorizontal);
        lookAtRes[i] = Vector3D::dotProduct(matrix[i], lookAt);
    }
    // calculating final results
    mVertical = vertRes;
    mHorizontal = horRes;
    mBottomLeftCorner = mOrigin + lookAtRes;
}

