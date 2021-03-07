//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include "Camera.h"
#include "Common/Math.h"
#include "Hittable/Hittable.h"
#include "Common/ErrorHandling.h"

#include <curand_kernel.h>

class QImage;

struct RenderInfo {
    int imgWidth;
    int imgHeight;
    int samplesPerPixel;
    int threadBlockWidth;
    int threadBlockHeight;

    RenderInfo() {
        threadBlockWidth = 33;
        threadBlockHeight = 40;
    }
};

class Renderer : public Invalidatable {
public:
    explicit Renderer(const RenderInfo& renderInfo);

    QImage render(const Camera* camera);

    ~Renderer();

private:
    uchar8* renderRaw(const Camera* camera);

    int mImgWidth;
    int mImgHeight;
    int mSamplesPerPixel;

    int mThreadBlockWidth;
    int mThreadBlockHeight;

    int mColorDataSize;
    uchar8* mColorData_d;
    uchar8* mColorData_h;

    curandState* mRandStateArr;

    Hittable** mWorld_d;
};

#endif //RAY_RACING_CLUB_RENDERER_H
