#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "Hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class HitList : public Hittable {           //������ ������� ��������.
public:
    HitList() {}
    HitList(shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }       //�������� ������
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }  //�������� ������

    virtual bool Hit(  //�������, ������� �������� ��������� ��������� ����� ��������������� ���� � ������� �������� �������
        const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

public:
    std::vector<shared_ptr<Hittable>> objects;
};

bool HitList::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    HitRecord tempRec;
    bool hitAnything = false;
    auto closestSoFar = tMax;

    for (const auto& object : objects) {                    //��������� ��� �������. 
        if (object->hit(r, tMin, closestSoFar, tempRec)) {  //���� ������� ����� ��������������� �����, ��� ����� ������, �� ��������� �����.
            hitAnything = true;
            closestSoFar = tempRec.t;
            rec = tempRec;
        }
    }

    return hitAnything;                                     //���������� ����, ����� �� �� ���� ���� �����
}

#endif