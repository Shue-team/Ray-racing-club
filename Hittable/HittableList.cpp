//
// Created by Stasy on 24.02.2021.
//

#include "HittableList.h"

HittableList::HittableList() {

}

HittableList::HittableList(Hittable* object) {
    add(object);
}

void HittableList::add(Hittable* object) {
    mObjects.push_back(object);
}

void HittableList::clear() {
    for(int i = 0; i < mObjects.size(); i++)
        delete mObjects[i];
    mObjects.clear();
}

bool HittableList::hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const {
    HitRecord bufRec;
    bool hitAnything = false;
    float curClosest = tMax;

    for (const Hittable* object : mObjects) {
        if (object->hit(ray, tMin, curClosest, bufRec)) {
            hitAnything = true;
            curClosest = bufRec.t;
            record = bufRec;
        }
    }

    return hitAnything;
}

HittableList::~HittableList() {
    clear();
}