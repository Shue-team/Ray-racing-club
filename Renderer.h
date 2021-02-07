//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include <QImage>

#include "Ray.h"
#include "Hittable/Hittable.h"
#include "Camera.h"

class Renderer {
public:
    Renderer(int samplesPerPixel);

    QImage render(int imgWidth, int imgHeight);

private:
    Color rayColor(const Ray& ray) const;
    void renderPixel(QImage& img, const QPoint& pixelCoord) const;

    QColor toQColor(const Color& color) const;

    Hittable* mWorld;

    Camera* mCamera;
    int mSamplesPerPixel;
};

#endif //RAY_RACING_CLUB_RENDERER_H
