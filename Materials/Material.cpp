//
// Created by Comp on 01.03.2021.
//
#include "Material.h"
bool Lambertian::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    Vector3D scatterDirection = rec.normal + randomUnitVector();

    if (scatterDirection.near_zero())
        scatterDirection = rec.normal;

    scattered = Ray(rec.intersection, scatterDirection);
    attenuation = albedo;
    return true;
}

bool Metal::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    Vector3D reflected = reflect(rIn.direction().normalized(), rec.normal);
    scattered = Ray(rec.intersection, reflected + fuzz*randomInUnitSphere());
    attenuation = albedo;
    return (Vector3D::dotProduct(scattered.direction(), rec.normal) > 0);
}

bool Dielectric::scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
    attenuation = Color(1.0, 1.0, 1.0);
    float refractionRatio = rec.frontFace ? (1.0f/RefractionIndex) : RefractionIndex;

    Vector3D unitDirection = rIn.direction().normalized();
    float cosTheta = fmin(Vector3D::dotProduct(-unitDirection, rec.normal), 1.0f);
    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);

    bool cannotRefract = refractionRatio * sinTheta > 1.0;
    Vector3D direction;

    if (cannotRefract || reflectance(cosTheta, refractionRatio) > randomFloat())
        direction = reflect(unitDirection, rec.normal);
    else
        direction = refract(unitDirection, rec.normal, refractionRatio);

    scattered = Ray(rec.intersection, direction);
    return true;
}

float Dielectric::reflectance(float cosine, float refractionRatio) {
    float r0 = (1-refractionRatio) / (1+refractionRatio);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

