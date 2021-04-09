#include "Camera.h"
//
// Created by arseny on 07.02.2021.
//

#include <cmath>
#include <iostream>
#include "Quaternion.h"
#include "Camera.h"

constexpr float pi = 3.14159265;


Camera::Camera(const Camera::CamParams& params) {
    w = params.lookFrom - params.lookAt;
    w = w.normalized();
    u = Vector3D::crossProduct(params.vUp, w);
    u = u.normalized();
    v = Vector3D::crossProduct(w, u);
    
    mOrigin = params.lookFrom;

    mLensRadius = 0.5f * params.aperture;
    mVFOV = params.vfov;
    mAspectRatio = params.aspectRatio;
    mFocusDist = params.focusDist;
    applyFOV();
}

void Camera::applyFOV() {
    float theta = mVFOV / 180.f * pi;
    float h = tanf(theta / 2);
    float viewportHeight = 2.0f * h;
    float viewportWidth = viewportHeight * mAspectRatio;
    mHorizontal = mFocusDist * viewportWidth * u;
    mVertical = mFocusDist * viewportHeight * v;
    mBottomLeftCorner = mOrigin - 0.5f * mHorizontal - 0.5f * mVertical - mFocusDist * w;
}

__device__ Vector3D Camera::getRandomInUnitDisk(curandState* randState) const {
    while (true) {
        Vector3D vec((curand_uniform(randState) - 0.5f) * 2, (curand_uniform(randState) - 0.5f) * 2, 0);
        if (vec.lengthSquared() >= 1) {
            continue;
        }
        return vec;
    }
}

__device__ Ray Camera::getRay(float s, float t, curandState* randState) const {
    Vector3D rd = mLensRadius * getRandomInUnitDisk(randState);
    Vector3D offset = u * rd.x() + v * rd.y();
    return Ray(mOrigin + offset, mBottomLeftCorner + s * mHorizontal + t * mVertical - mOrigin - offset);
}

// move along camera's horizontal axi
void Camera::moveHorz(float dx) {
    Vector3D add(dx * u);
    mBottomLeftCorner += add;
    mOrigin += add;
}

// move along camera's vertical axi
void Camera::moveVert(float dy) {
    Vector3D add(dy * v);
    mBottomLeftCorner += add;
    mOrigin += add;
}

// moving screen to zoom
void Camera::moveDepth(float dz) {
    mBottomLeftCorner += dz * w;
}

// rotation around camera's horizontal axi
void Camera::rotateOx(float alpha) {
    rotate(u, alpha);
}

// rotation around camera's vertical axi
void Camera::rotateOy(float alpha) {
    rotate(v, alpha);
}

/*
 * rotating hor vector, vert vector and screen using quaternion
 * screen need to be rotated using lookAt vector (from origin to leftBottom point)
*/
void Camera::rotate(const Vector3D &axis, float alpha) {
    Quaternion rotQuat = Quaternion::fromAxisAndAngle(axis, alpha);
    v = rotQuat.rotate(v);
    u = rotQuat.rotate(u);
    w = rotQuat.rotate(w);
    applyFOV();
}


