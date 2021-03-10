//
// Created by Comp on 01.03.2021.
//

#ifndef RAY_RACING_CLUB_MATERIAL_H
#define RAY_RACING_CLUB_MATERIAL_H

#include "../CommonMath.h"
#include "../Hittable/Hittable.h"

class Material {
public:
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material {
public:
    Lambertian(const Color& albedo) : mAlbedo(albedo) {}
    bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;

private:
    Color mAlbedo;
};

class Metal : public Material {
public:
    Metal(const Color& albedo, float fuzz) : mAlbedo(albedo), mFuzz(fuzz < 1 ? fuzz : 1) {}
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;

private:
    Color mAlbedo;
    float mFuzz;
};

class Dielectric : public Material {
public:
    Dielectric(float refractionIndex) : mRefractionIndex(refractionIndex) {}
    virtual bool scatter(const Ray& rIn, const HitRecord& rec, Color& attenuation, Ray& scattered) const override;

private:
    float mRefractionIndex;
    static float reflectance(float cosine, float refractionRatio);
};

#endif //RAY_RACING_CLUB_MATERIAL_H
