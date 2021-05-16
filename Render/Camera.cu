//
// Created by arseny on 07.02.2021.
//

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include "../Math/Quaternion.h"
#include "Camera.h"
#include "math_constants.h"


Camera::Camera(const CamParams& params) {
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
    float theta = mVFOV / 180.f * (float)M_PI;
    float h = tanf(theta / 2);
    float viewportHeight = 2.0f * h;
    float viewportWidth = viewportHeight * mAspectRatio;
    mHorizontal = mFocusDist * viewportWidth * u;
    mVertical = mFocusDist * viewportHeight * v;
    mBottomLeftCorner = mOrigin - 0.5f * mHorizontal - 0.5f * mVertical - mFocusDist * w;
}

__device__ Ray Camera::getRay(float s, float t, curandState* randState) const {
    Vector3D rd = mLensRadius * Vector3D::getRandomInUnitDisk(randState);
    Vector3D offset = u * rd.x() + v * rd.y();
    return Ray(mOrigin + offset, mBottomLeftCorner + s * mHorizontal + t * mVertical - mOrigin - offset);
}

// move along camera's horizontal axis
void Camera::moveHorz(float dx) {
    Vector3D add(dx * u);
    mBottomLeftCorner += add;
    mOrigin += add;
}

// move along camera's vertical axis
void Camera::moveVert(float dy) {
    Vector3D add(dy * v);
    mBottomLeftCorner += add;
    mOrigin += add;
}

// moving screen to zoom
void Camera::moveDepth(float dz) {
    mBottomLeftCorner += dz * w;
}

// rotation around camera's horizontal axis
void Camera::rotateOx(float alpha) {
    rotate(u, alpha);
}

// rotation around camera's vertical axis
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


