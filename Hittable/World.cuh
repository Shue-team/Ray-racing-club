//
// Created by Stasy on 24.02.2021.
//

#ifndef RAY_RACING_CLUB_WORLD_CUH
#define RAY_RACING_CLUB_WORLD_CUH

#include <thrust/device_vector.h>
#include "Hittable.h"

// I don't want to use shared_ptr (at least not yet)
// World will take care of the bare pointers in it (delete hittable)

class Node {
public:
    __host__ __device__ Node(Hittable* object);
    __host__ __device__ void setNext(Node* node);
    __host__ __device__ Hittable* getObject();
    __host__ __device__ Node* getNext();
private:
    Hittable* object;
    Node* next;
};

class World: public Hittable {
public:
    __host__ __device__ World();
    __host__ __device__ void add(Hittable* object);
    __host__ __device__ int getSize();
    __host__ __device__ void clear();
    __host__ __device__ virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const override;
    __host__ __device__ ~World();
private:
    __host__ __device__ void deleteList(Node* curNode);
    Node* head;
    int curSize;
};


#endif //RAY_RACING_CLUB_WORLD_CUH
