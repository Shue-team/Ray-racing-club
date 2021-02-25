//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#pragma warning(disable: 4251)

#include "Camera.h"
#include "Hittable/World.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <QImage>
#include <curand_kernel.h>


struct Image {
    int height;
    int width;
    unsigned char* d_pixels;
    unsigned char* pixels;
};

class Renderer {
public:
    enum class RenderMode {
        FAST,
        ANTIALIASING,
    };
    Renderer(const int height, const int width);
    unsigned char* render();
    QImage::Format getImageFormat() {
        return format;
    }
    int getHeight() {
        return image.height;
    }
    int getWidth() {
        return image.width;
    }
    void changeMode();
    ~Renderer();
private:
    static const int threadsFast = 128;
    static const int samplesPerPix = 100;
    static const QImage::Format format = QImage::Format_RGB888;
    Camera cam;
    Image image;
    Hittable** d_world; // thrust::device_ptr ?? nah
    RenderMode mode;
    curandState* randState;
};


#endif //RAY_RACING_CLUB_RENDERER_H
