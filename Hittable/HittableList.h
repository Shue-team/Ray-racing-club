#ifndef RAY_RACING_CLUB_HITTABLELIST_H
#define RAY_RACING_CLUB_HITTABLELIST_H

#include "Hittable.h"
#include "HittableDef.h"
#include "../Material/MaterialDef.h"

class HittableList : public Hittable {
public:
    __host__ __device__ HittableList(Hittable** list, int size)
        : mList(list), mSize(size) {}

    __host__ __device__ bool hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const override;

    __host__ __device__ ~HittableList();

private:
    Hittable** mList;
    int mSize;
};



#endif //RAY_RACING_CLUB_HITTABLELIST_H
