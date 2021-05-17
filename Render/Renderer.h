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
#include "../Material/MaterialDef.h"
#include "../Hittable/HittableDef.h"

using std::vector;

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

    uchar8* render(const Camera* camera);

    void setWorld(Hittable** world);

    ~Renderer();

private:

    RenderInfo mRi;
    dim3 mGridDim;
    dim3 mBlockDim;

    int mColorDataSize;
    uchar8* mColorData_d;
    uchar8* mColorData_h;

    curandState* mRandStateArr;

    Hittable** mWorld_d = nullptr;
};

#endif //RAY_RACING_CLUB_RENDERER_H
