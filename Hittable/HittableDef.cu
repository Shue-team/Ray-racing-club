#include "Sphere.h"
#include "HittableDef.h"

Hittable* createHittable(const HittableDef* def, Material* material) {
    switch(def->type()) {
    case HittableType::Sphere:
        return new Sphere(static_cast<const SphereDef*>(def), material);

    default:
        return nullptr;
    }
}