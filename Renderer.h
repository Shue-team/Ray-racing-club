//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include "Camera.h"
#include "Common/Math.h"
#include "Hittable/Hittable.h"

struct RenderInfo {
    int imgWidth;
    int imgHeight;
    int samplesPerPixel;
    int threadBlockWidth;
    int threadBlockHeight;

    RenderInfo() {
        threadBlockWidth = 8;
        threadBlockHeight = 8;
    }
};

class Renderer {
public:
    Renderer(const RenderInfo& renderInfo);

    uchar8* render(const Camera* camera);

    ~Renderer();

private:
    int mImgWidth;
    int mImgHeight;
    int mSamplesPerPixel;

    int mThreadBlockWidth;
    int mThreadBlockHeight;

    int mColorDataSize;
    uchar8* mColorData_d;
    uchar8* mColorData_h;

    Hittable** mWorld_d;
};


#endif //RAY_RACING_CLUB_RENDERER_H
