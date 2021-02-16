#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "Hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class HitList : public Hittable {           //Список видимых объектов.
public:
    HitList() {}
    HitList(shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }       //Очистить список
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }  //добавить объект

    virtual bool Hit(  //Функция, которая отмечает некоторые параметры точки соприкосновения луча и первого видимого объекта
        const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

public:
    std::vector<shared_ptr<Hittable>> objects;
};

bool HitList::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    HitRecord tempRec;
    bool hitAnything = false;
    auto closestSoFar = tMax;

    for (const auto& object : objects) {                    //Проверяем все объекты. 
        if (object->hit(r, tMin, closestSoFar, tempRec)) {  //Если находим точку соприкосновения ближе, чем нашли раньше, то вписываем новую.
            hitAnything = true;
            closestSoFar = tempRec.t;
            rec = tempRec;
        }
    }

    return hitAnything;                                     //возвращаем флаг, нашли ли мы хоть одну точку
}

#endif