//
// Created by Comp on 01.03.2021.
//

#ifndef RAY_RACING_CLUB_MATERIAL_H
#define RAY_RACING_CLUB_MATERIAL_H

#include "Utility.h"
#include "../Hittable/Hittable.h"

class Material {
public:
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material {
public:
    Lambertian(const Color& a) : albedo(a) {}
    bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;
public:
    Color albedo;
};

class Metal : public Material {
public:
    Metal(const Color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;
public:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material {
public:
    Dielectric(float ir) : RefractionIndex(ir) {}
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;
public:
    float RefractionIndex;
private:
    static float reflectance(float cosine, float refractionRatio);
};

#endif //RAY_RACING_CLUB_MATERIAL_H
