#include "Dielectric.h"
#include "Lambertian.h"
#include "Metal.h"
#include "MaterialDef.h"

Material* createMaterial(const MaterialDef* def) {
    switch(def->type()) {
    case MaterialType::Dielectric:
        return new Dielectric((const DielectricDef*)def);

    case MaterialType::Lambertian:
        return new Lambertian((const LambertianDef*)def);

    case MaterialType::Metal:
        return new Metal((const MetalDef*)def);

    default:
        return nullptr;
    }
}
