//
// Created by Stasy on 24.02.2021.
//

#ifndef RAY_RACING_CLUB_WORLD_CUH
#define RAY_RACING_CLUB_WORLD_CUH

#include "Hittable.h"

// I don't want to use shared_ptr (at least not yet)
// World will take care of the bare pointers in it (delete hittable)

class Node {
public:
    Node(Hittable* object);
    void setNext(Node* node);
    Hittable* getObject();
    Node* getNext();
private:
    Hittable* object;
    Node* next;
};

class World: public Hittable {
public:
    World();
    void add(Hittable* object);
    int getSize();
    void clear();
    virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& record) const override;
    ~World();
private:
    void deleteList(Node* curNode);
    Node* head;
    int curSize;
};


#endif //RAY_RACING_CLUB_WORLD_CUH
