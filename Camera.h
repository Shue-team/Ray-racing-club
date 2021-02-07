//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_CAMERA_H
#define RAY_RACING_CLUB_CAMERA_H

#include "CommonMath.h"
#include "Ray.h"

class Camera {
public:
    Camera(double aspectRatio);

    Ray getRay(double u, double v) const;

private:
    Point3D mOrigin;
    Point3D mBottomLeftCorner;
    QVector3D mHorizontal;
    QVector3D mVertical;
};


#endif //RAY_RACING_CLUB_CAMERA_H
