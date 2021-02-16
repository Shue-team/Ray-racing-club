#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include <QVector3D>

class sphere : public Hittable {      //Видимый объект сфера.
public:
    sphere() {}
    sphere(Point3D cen, double r, shared_ptr<material> m)        //По центру cen, радиусу r и материалу m строится сфера
        : center(cen), radius(r), MatPtr(m) {};

    virtual bool Hit(                                            //Функция, которая отмечает некоторые параметры точки соприкосновения луча и объекта
        const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

public:
    Point3D center;                     //Центр сферы.
    double radius;                      //Радиус сферы.
    shared_ptr<material> MatPtr;        //Материал сферы
};

bool sphere::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    QVector3D oc = r.origin() - center;
    auto a = r.direction().lengthSquared();
    auto Halfb = dotProduct(oc, r.direction());
    auto c = oc.lengthSquared() - radius * radius;                                  //Решаем квадратное уравнение t^2*b^2+2*tb(A-C)^2-r^2=0 относительно t
                                                                                    //где b - вектор направления луча, A - начало пути луча, C - центр сферы, r - радиус сферы
    auto discriminant = Halfb * Halfb - a * c;
    if (discriminant < 0) return false;                                             //Если нет корней - значит луч не пересекается со сферой.
    auto sqrtd = sqrt(discriminant);

    //Ищем ближайшую к нам точку пересечения, находящуюся в нашем поле зрения, т.е. между tMin и tMax
    auto root = (-Halfb - sqrtd) / a;
    if (root < tMin || tMax < root) {
        root = (-Halfb + sqrtd) / a;
        if (root < tMin || tMax < root)
            return false;
    }

    //Записываем параметры точки пересечения
    rec.t = root;
    rec.intersection = r.at(rec.t);
    QVector3D outwardNormal = (rec.intersection - center) / radius;
    rec.SetFaceNormal(r, outwardNormal);
    rec.MatPtr = MatPtr;

    return true;
}

#endif