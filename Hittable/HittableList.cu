#include "HittableList.h"

bool HittableList::hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const {
    HitRecord tmpRecord;
    bool isHit = false;
    float closest = tMax;

    for (int i = 0; i < mSize; i++) {
        if (mList[i]->hit(ray, tMin, closest, tmpRecord)) {
            isHit = true;
            closest = tmpRecord.t;
            record = tmpRecord;
        }
    }
    return isHit;
}

HittableList::~HittableList() {
    for (int i = 0; i < mSize; i++) {
        delete mList[i];
    }
    delete[] mList;
}
