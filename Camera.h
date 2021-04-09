//
// Created by arseny on 07.02.2021.
//

#ifndef RAY_RACING_CLUB_CAMERA_H
#define RAY_RACING_CLUB_CAMERA_H

#include "Ray.h"
#include "Managed.h"
#include <curand_kernel.h>

class Camera : public Managed {
public:
    struct CamParams {
        Point3D lookFrom;
        Point3D lookAt;
        Vector3D vUp;
        float aspectRatio;
        float vfov;
        float aperture;
        float focusDist;
    };

    __host__ __device__ Camera(const CamParams& params);

    __device__ Ray getRay(float s, float t, curandState* randState) const;

    __host__ __device__ void moveHorz(float x);

    __host__ __device__ void moveVert(float y);

    __host__ __device__ void moveDepth(float z);

    __host__ __device__ void rotateOx(float alpha);

    __host__ __device__ void rotateOy(float alpha);

private:
    __host__ __device__ void rotate(const Vector3D& axis, float alpha);
    __host__ __device__ void applyFOV();
    __device__ Vector3D getRandomInUnitDisk(curandState* randState) const;
    Point3D mOrigin;
    Point3D mBottomLeftCorner;
    Vector3D u, v, w;
    Vector3D mHorizontal;
    Vector3D mVertical;
    float mLensRadius;
    float mFocusDist;
    float mVFOV;
    float mAspectRatio;
};


#endif //RAY_RACING_CLUB_CAMERA_H
