//
// Created by arseny on 07.02.2021.
//

#include "Camera.h"

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
