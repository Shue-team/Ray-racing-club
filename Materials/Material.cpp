//
// Created by Comp on 01.03.2021.
//
#include "Material.h"

#include <cmath>
bool Lambertian::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    Vector3D scatterDirection = rec.normal + randomUnitVector();

    if (scatterDirection.nearZero())
        scatterDirection = rec.normal;

    scattered = Ray(rec.intersection, scatterDirection);
    attenuation = mAlbedo;
    return true;
}

bool Metal::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    Vector3D reflected = reflect(rIn.direction().normalized(), rec.normal);
    scattered = Ray(rec.intersection, reflected + mFuzz*randomInUnitSphere());
    attenuation = mAlbedo;
    return (Vector3D::dotProduct(scattered.direction(), rec.normal) > 0.0f);
}

bool Dielectric::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    attenuation = Color(1.0f, 1.0f, 1.0f);
    float refractionRatio = rec.frontFace ? (1.0f/mRefractionIndex) : mRefractionIndex;

    Vector3D unitDirection = rIn.direction().normalized();
    float cosTheta = fminf(Vector3D::dotProduct(-unitDirection, rec.normal), 1.0f);
    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);

    bool cannotRefract = refractionRatio * sinTheta > 1.0f;
    Vector3D direction;

    if (cannotRefract || reflectance(cosTheta, refractionRatio) > randomFloat())
        direction = reflect(unitDirection, rec.normal);
    else
        direction = refract(unitDirection, rec.normal, refractionRatio);

    scattered = Ray(rec.intersection, direction);
    return true;
}

float Dielectric::reflectance(float cosine, float refractionRatio) {
    float r0 = (1.0f - refractionRatio) / (1.0f + refractionRatio);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine),5);
}

