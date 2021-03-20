//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_CAMERA_H
#define RAY_RACING_CLUB_CAMERA_H

#include "Ray.h"
#include "Managed.h"

class Camera : public Managed {
public:
    __host__ __device__ Camera(float aspectRatio);

    __host__ __device__ Ray getRay(float u, float v) const;

private:
    Point3D mOrigin;
    Point3D mBottomLeftCorner;
    Vector3D mHorizontal;
    Vector3D mVertical;
};


#endif //RAY_RACING_CLUB_CAMERA_H