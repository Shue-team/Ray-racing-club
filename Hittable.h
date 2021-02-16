#ifndef RAY_RACING_CLUB_HITTABLE_H
#define RAY_RACING_CLUB_HITTABLE_H

#include "Ray.h"

#include "rtweekend.h"

class material;

struct HitRecord {                   //���������, ������������� ��������� ������� ��� ��������� ���� � ������
    Point3D intersection;            //����� ����������� ���� � �������.
    QVector3D normal;                //����������� ������� � ����� �������.
    shared_ptr<material> MatPtr;     //�������� ����� �������.
    double t;                        //��������� �� ������ �� ����� �������. 
    bool FrontFace;                  //����������, ������������, ���� ������� �������. True - ������ �������, False - ������ �������.

    inline void SetFaceNormal(const Ray& r, const QVector3D& outward_normal) {      //�������, ��������������� ������� � ������� �������� r
                                                                                    //�� ���� ��������� r - ����������� ������� ��� ���� (������ �� ������)
                                                                                    //� outward_normal - ����������������� �������
        FrontFace = dotProduct(r.direction(), outwardNormal) < 0;                   //������� ��������� ������������ ����� r � outward_normal
        normal = FrontFace ? outwardNormal : -outwardNormal;                        //���� ��������� ������������ < 0, ������ ������� ������� �������� ����
                                                                                    //� ������ �� ������. ����� - �������� �� -1.
    };

    class Hittable {  //����� ��������, ������� �� ����� �������
        virtual bool Hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const = 0;  //����������� ������� "���������" ���� ray � ������.
                                                                                                  //�� ���� ��������� ray - ����������� ������� ��� ���� (������ �� ������)
                                                                                                  //tMin � tMax - �������������� ����������� � ������������ ����������, � ��������
                                                                                                  //������� �� ����� ������� ������
                                                                                                  //� record - ����������, ���� �� ������� ������� ������ ��������� ����� �������.
    };

#endif //RAY_RACING_CLUB_HITTABLE_H