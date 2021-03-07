//
// Created by Comp on 01.03.2021.
#include "Utility.h"
#include <random>

#include <cstdlib>

float randomFloat() {
    return rand() / (RAND_MAX + 1.0f);
}

float randomFloat(float min, float max) {
    return min + (max-min)*randomFloat();
}

Vector3D randomInUnitSphere() {
    while (true) {
        auto p = random(-1,1);
        if (p.lengthSquared() >= 1) continue;
        return p;
    }
}

Vector3D random(float min, float max) {
    return Vector3D(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max));
}

Vector3D randomUnitVector() {
    return randomInUnitSphere().normalized();
}

Vector3D reflect(const Vector3D& v, const Vector3D& n) {
    return v - 2*Vector3D::dotProduct(v,n)*n;
}

Vector3D refract(const Vector3D& uv, const Vector3D& n, float etaiOverEtat) {
    float cosTheta = fmin(Vector3D::dotProduct(-uv, n), 1.0f);
    Vector3D rOutPerp =  etaiOverEtat * (uv + cosTheta*n);
    Vector3D rOutParallel = -sqrt(fabs(1.0f - rOutPerp.lengthSquared())) * n;
    return rOutPerp + rOutParallel;
}