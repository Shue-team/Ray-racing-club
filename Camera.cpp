//
// Created by arseny on 07.02.2021.
//

#include "Camera.h"

Camera::Camera(double aspectRatio) {
    double viewportHeight = 2.0;
    double viewportWidth = viewportHeight * aspectRatio;
    double focalLength = 1.0;

    mOrigin = Point3D(0, 0, 0);
    mHorizontal = QVector3D(viewportWidth, 0, 0);
    mVertical = QVector3D(0, viewportHeight, 0);
    QVector3D vUp(0, 0, focalLength);
    mBottomLeftCorner = mOrigin - mHorizontal / 2 - mVertical / 2 - vUp;
}

Ray Camera::getRay(double u, double v) const {
    return Ray(mOrigin, mBottomLeftCorner + u * mHorizontal + v * mVertical - mOrigin);
}
