//
// Created by Stasy on 24.02.2021.
//

#include "World.cuh"

World::World() {
    head = nullptr;
    curSize = 0;
}

void World::add(Hittable *object) {
    Node* newNode = new Node(object);
    newNode->setNext(head);
    head = newNode;
    curSize++;
}

void World::clear() {
    deleteList(head);
    head = nullptr;
}

bool World::hit(const Ray &ray, float tMin, float tMax, HitRecord &record) const {
    HitRecord bufRec;
    bool hitAnything = false;
    float curClosest = tMax;
    Node* curNode = head;
    while (curNode) {
        Hittable* object = curNode->getObject();
        if (object->hit(ray, tMin, curClosest, bufRec)) {
            hitAnything = true;
            curClosest = bufRec.t;
            record = bufRec;
        }
        curNode = curNode->getNext();
    }
    return hitAnything;
}

World::~World() {
    deleteList(head);
}

// todo: recursion is evil
void World::deleteList(Node *curNode) {
    if (!curNode) {
        return;
    }
    deleteList(curNode->getNext());
    delete curNode->getObject();
    delete curNode;
}

int World::getSize() {
    return curSize;
}

Node::Node(Hittable *object) {
    this->object = object;
    next = nullptr;
}

void Node::setNext(Node *node) {
    next = node;
}

Hittable *Node::getObject() {
    return object;
}

Node *Node::getNext() {
    return next;
}
