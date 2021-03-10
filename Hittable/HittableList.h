//
// Created by Stasy on 24.02.2021.
//

#ifndef RAY_RACING_CLUB_WORLD_CUH
#define RAY_RACING_CLUB_WORLD_CUH

#include <vector>
#include "Hittable.h"

class HittableList: public Hittable {
public:
    HittableList();
    HittableList(Hittable* object);
    void clear();
    void add(Hittable* object);
    virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const override;
    ~HittableList();

private:
    std::vector<Hittable*> mObjects;
};


#endif //RAY_RACING_CLUB_WORLD_CUH
