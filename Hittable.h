#ifndef RAY_RACING_CLUB_HITTABLE_H
#define RAY_RACING_CLUB_HITTABLE_H

#include "Ray.h"

#include "rtweekend.h"

class material;

struct HitRecord {                   //Структура, отслеживающая некоторые аспекты при попадании луча в объект
    Point3D intersection;            //Точка пересечения луча и объекта.
    QVector3D normal;                //Направление нормали в точке касания.
    shared_ptr<material> MatPtr;     //Материал точки касания.
    double t;                        //Растояние от камеры до точки касания. 
    bool FrontFace;                  //Переменная, определяющая, куда смотрит нормаль. True - наружу объекта, False - внутрь объекта.

    inline void SetFaceNormal(const Ray& r, const QVector3D& outward_normal) {      //Функция, разворачивающая нормаль в сторону напротив r
                                                                                    //на вход принимает r - направление нужного нам луча (обычно от камеры)
                                                                                    //и outward_normal - неориентированную нормаль
        FrontFace = dotProduct(r.direction(), outwardNormal) < 0;                   //Считаем сколярное произведение между r и outward_normal
        normal = FrontFace ? outwardNormal : -outwardNormal;                        //Если сколярное произведение < 0, значит нормаль смотрит напротив луча
                                                                                    //и ничего не меняем. Иначе - умножаем на -1.
    };

    class Hittable {  //Класс объектов, которые мы можем увидеть
        virtual bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const = 0;  //Виртуальная функция "попадания" луча ray в объект.
                                                                                                  //На вход принимает ray - направление нужного нам луча (обычно от камеры)
                                                                                                  //tMin и tMax - соответственно минимальное и максимальное расстояние, в пределах
                                                                                                  //которых мы можем увидеть объект
                                                                                                  //и record - переменная, куда мы запишем аспекты первой увиденной точки объекта.
    };

#endif //RAY_RACING_CLUB_HITTABLE_H