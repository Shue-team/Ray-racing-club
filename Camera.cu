//
// Created by arseny on 07.02.2021.
//

#include "Camera.h"

Camera::Camera(float aspectRatio) {
    float viewportHeight = 2.0f;
    float viewportWidth = viewportHeight * aspectRatio;
    float focalLength = 1.0f;

    mOrigin = Point3D(0, 0, 0);
    mHorizontal = Vector3D(viewportWidth, 0, 0);
    mVertical = Vector3D(0, viewportHeight, 0);
    Vector3D vUp(0, 0, focalLength);
    mBottomLeftCorner = mOrigin - mHorizontal / 2 - mVertical / 2 - vUp;
}

Ray Camera::getRay(float u, float v) const {
    return Ray(mOrigin, mBottomLeftCorner + u * mHorizontal + v * mVertical - mOrigin);
}
