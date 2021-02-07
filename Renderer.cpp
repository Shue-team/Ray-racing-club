//
// Created by arseny on 06.02.2021.
//

#include <QtConcurrent>
#include <QFuture>

#include "Renderer.h"
#include "Hittable/Sphere.h"

Renderer::Renderer(int samplesPerPixel) {
    mWorld = new Sphere(Point3D(0, 0, -1), 0.5);
    mSamplesPerPixel = samplesPerPixel;
}

QImage Renderer::render(int imgWidth, int imgHeight) {
    QImage img(imgWidth, imgHeight, QImage::Format_RGB32);

    mCamera = new Camera(imgWidth / imgHeight);

    QVector<QPoint> pixelCoords(imgWidth * imgHeight);
    for (int ix = 0; ix < imgWidth; ix++) {
        for (int iy = 0; iy < imgHeight; iy++) {
            pixelCoords[iy * imgWidth + ix] = QPoint(ix, iy);
        }
    }

    auto wrapper = [&img, this] (const QPoint& pixelCoord) {
        renderPixel(img, pixelCoord);
    };
    QtConcurrent::blockingMap(pixelCoords,  wrapper);
    return img;
}

Color Renderer::rayColor(const Ray& ray) const {
    HitRecord record;

    if (mWorld->hit(ray, 0, infinity, record)) {
        return 0.5 * (record.normal + Color(1, 1, 1));
    }

    QVector3D unitDirection = ray.direction().normalized();
    double t = 0.5  * (unitDirection.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

void Renderer::renderPixel(QImage& img, const QPoint& pixelCoord) const {
    int x = pixelCoord.x();
    int y = pixelCoord.y();
    double width = img.width();
    double height = img.height();

    Color pixelColor;
    for (int is = 0; is < mSamplesPerPixel; is++) {
        double u = (x + randomDouble()) / (width - 1);
        double v = (img.height() - y + randomDouble() - 1) / (height - 1);

        Ray ray = mCamera->getRay(u, v);
        pixelColor += rayColor(ray);
    }

    QColor qPixelColor = toQColor(pixelColor);
    img.setPixelColor(pixelCoord, qPixelColor);
}

QColor Renderer::toQColor(const Color& color) const {
    Color scaled = color / mSamplesPerPixel;

    double red = std::clamp(0.0, scaled[0] * 255.0, 255.0);
    double green = std::clamp(0.0, scaled[1] * 255.0, 255.0);
    double blue = std::clamp(0.0, scaled[2] * 255.0, 255.0);

    return QColor(red, green, blue);
}