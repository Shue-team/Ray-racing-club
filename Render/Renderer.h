//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include "Camera.h"
#include "../Math/Math.h"
#include "../Hittable/Hittable.h"
#include "../ErrorProcessing/ErrorHandling.h"

#include <curand_kernel.h>

class QImage;

constexpr int threadBlockWidth = 16;
constexpr int threadBlockHeight = 16;

struct RenderInfo {
    int imgWidth;
    int imgHeight;
    int samplesPerPixel;
    int maxDepth;
};

class Renderer : public Invalidatable {
public:
    explicit Renderer(const RenderInfo& renderInfo);

    QImage render(const Camera* camera);

    ~Renderer();

private:
    uchar8* renderRaw(const Camera* camera);

    RenderInfo mRi;
    dim3 mGridDim;
    dim3 mBlockDim;

    int mColorDataSize;
    uchar8* mColorData_d;
    uchar8* mColorData_h;

    curandState* mRandStateArr;

    Hittable** mWorld_d;
};

#endif //RAY_RACING_CLUB_RENDERER_H
