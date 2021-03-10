//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_RENDERER_H
#define RAY_RACING_CLUB_RENDERER_H

#include <QImage>
#include "Hittable/Hittable.h"
#include "Ray.h"

class Renderer {
public:
    Renderer();
    QImage render(int imgWidth, int imgHeight) const;

private:
    Color rayColor(const Ray& ray, Hittable* world, int depth) const;
    QColor toQColor(const Color& color) const;
};

#endif //RAY_RACING_CLUB_RENDERER_H
