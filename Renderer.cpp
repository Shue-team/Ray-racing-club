//
// Created by arseny on 06.02.2021.
//

#include "Renderer.h"
#include "Hittable/Sphere.h"
#include "Hittable/Triangle.h"
#include "Hittable/HittableList.h"

Renderer::Renderer() {

}

HittableList* createWorld() {
    HittableList* world = new HittableList;

    Material* matGround = new Lambertian(Color(0.8f, 0.8f, 0.0f));
    Hittable* ground = new Sphere(Vector3D(0.0f, 100.5f, -1.0f), 100.0f, matGround);

    Material* mat1 = new Metal(Color(0.7f, 0.3f, 0.3f), 0.0f);
    Material* mat2 = new Metal(Color(0.7f, 0.3f, 0.3f), 0.0f);
    Material* mat3 = new Metal(Color(0.8f, 0.8f, 0.8f), 0.0f);
    Material* mat4 = new Metal(Color(0.8f, 0.8f, 0.8f), 0.0f);

    Hittable* tri1 = new Triangle(Vector3D(0.0f, 0.5f, -1.0f),  Vector3D(0.0f, -0.5f, -1.0f), Vector3D(1.0f, 0.5f, -2.0f), mat1);
    Hittable* tri2 = new Triangle(Vector3D(0.0f, -0.5f, -1.0f), Vector3D(1.0f, -0.5f, -2.0f), Vector3D(1.0f, 0.5f, -2.0f), mat2);
    Hittable* tri3 = new Triangle(Vector3D(0.0f, 0.5f, -1.0f), Vector3D(-1.0f, 0.5f, -2.0f), Vector3D(0.0f, -0.5f, -1.0f),  mat3);
    Hittable* tri4 = new Triangle(Vector3D(0.0f, -0.5f, -1.0f),  Vector3D(-1.0f, 0.5f, -2.0f), Vector3D(-1.0f, -0.5f, -2.0f), mat4);

    world->add(tri1);
    world->add(tri2);
    world->add(tri3);
    world->add(tri4);
    world->add(ground);

    return world;
}

QImage Renderer::render(int imgWidth, int imgHeight) const {
    QImage img(imgWidth, imgHeight, QImage::Format_RGB888);

    double aspectRatio = (double) imgWidth / imgHeight;
    double focalLength = 1.0;
    double viewportHeight = 2.0;
    double viewportWidth =  aspectRatio * viewportHeight;
    const int samplesPerPixel = 20;
    const int maxDepth = 15;

    Point3D origin(0, 0, 0);
    Vector3D horizontal(viewportWidth, 0, 0);
    Vector3D vertical(0, viewportHeight, 0);
    Point3D bottomLeft = origin - horizontal / 2 - vertical / 2 - Vector3D(0, 0, focalLength);

    HittableList* world = createWorld();

    for (int iy = imgHeight - 1; iy >= 0; --iy) {
        for (int ix = 0; ix < imgWidth; ++ix) {
            Color pixelColor(0, 0, 0);

            for (int s = 0; s < samplesPerPixel; ++s) {
                float u = (ix + randomFloat()) / (imgWidth-1);
                float v = (iy + randomFloat()) / (imgHeight-1);
                Ray ray(origin, bottomLeft + u * horizontal + v * vertical - origin);
                pixelColor += rayColor(ray, world, maxDepth);
            }

            pixelColor[0] = clamp(pixelColor[0]/samplesPerPixel, 0.0f, 0.999f);
            pixelColor[1] = clamp(pixelColor[1]/samplesPerPixel, 0.0f, 0.999f);
            pixelColor[2] = clamp(pixelColor[2]/samplesPerPixel, 0.0f, 0.999f);

            img.setPixelColor(ix, iy, toQColor(pixelColor));
        }
    }

    delete world;
    return img;
}

Color Renderer::rayColor(const Ray& ray, Hittable* world, int depth) const {
    HitRecord rec;

    if (depth <= 0)
        return Color(0,0,0);

    if (world->hit(ray, 0.001f, infinity, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.matPtr->scatter(ray, rec, attenuation, scattered))
            return attenuation * rayColor(scattered, world, depth-1);
        return Color(0,0,0);
    }

    Vector3D unitDirection = ray.direction().normalized();
    float t = 0.5f  * (unitDirection.y() + 1.0f);
    return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

QColor Renderer::toQColor(const Color& color) const {
    Color scaled = color * 255;
    return QColor(scaled[0], scaled[1], scaled[2]);
}


