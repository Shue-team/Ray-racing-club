//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include "Camera.h"
#include "CommonMath.h"
#include "Hittable/Hittable.h"

class Renderer {
public:
    Renderer(int imgWidth, int imgHeight, int samplesPerPixel);
    uchar8* render(const Camera* camera);

    ~Renderer();

private:
    int mImgWidth;
    int mImgHeight;
    int mSamplesPerPixel;

    uchar8* mColorBuff_d;
    uchar8* mColorBuff_h;

    Hittable* mWorld;
};


#endif //RAY_RACING_CLUB_RENDERER_H
