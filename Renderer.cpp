//
// Created by arseny on 06.02.2021.
//

#include "Renderer.h"

Renderer::Renderer() {

}

QImage Renderer::render(int imgWidth, int imgHeight) const {
    QImage img(imgWidth, imgHeight, QImage::Format_RGB32);

    float aspectRatio = imgWidth / imgHeight;

    float focalLength = 1.0;
    float viewportHeight = 2.0;
    float viewportWidth =  aspectRatio * viewportHeight;

    Point3D origin(0, 0, 0);
    QVector3D horizontal(viewportWidth, 0, 0);
    QVector3D vertical(0, viewportHeight, 0);
    Point3D bottomLeft = origin - horizontal / 2 - vertical / 2 - QVector3D(0, 0, focalLength);

    for (int ix = 0; ix < imgWidth; ix++) {
        for (int iy = imgHeight - 1; iy >= 0; iy--) {
            //Temporary simple render
            float u = ix / float(imgWidth - 1);
            float v = (imgHeight - iy - 1) / float(imgHeight - 1);

            Ray ray(origin, bottomLeft + u * horizontal + v * vertical - origin);
            Color pixelColor = rayColor(ray);
            img.setPixelColor(ix, iy, toQColor(pixelColor));
        }
    }
    return img;
}

Color Renderer::rayColor(const Ray& ray) const {
    QVector3D unitDirection = ray.direction().normalized();
    float t = 0.5  * (unitDirection.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

QColor Renderer::toQColor(const Color& color) const {
    Color scaled = color * 255;
    return QColor(scaled[0], scaled[1], scaled[2]);
}


