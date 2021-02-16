#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"
#include <QVector3D>

class sphere : public Hittable {      //������� ������ �����.
public:
    sphere() {}
    sphere(Point3D cen, double r, shared_ptr<material> m)        //�� ������ cen, ������� r � ��������� m �������� �����
        : center(cen), radius(r), MatPtr(m) {};

    virtual bool Hit(                                            //�������, ������� �������� ��������� ��������� ����� ��������������� ���� � �������
        const Ray& r, double tMin, double tMax, HitRecord& rec) const override;

public:
    Point3D center;                     //����� �����.
    double radius;                      //������ �����.
    shared_ptr<material> MatPtr;        //�������� �����
};

bool sphere::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
    QVector3D oc = r.origin() - center;
    auto a = r.direction().lengthSquared();
    auto Halfb = dotProduct(oc, r.direction());
    auto c = oc.lengthSquared() - radius * radius;                                  //������ ���������� ��������� t^2*b^2+2*tb(A-C)^2-r^2=0 ������������ t
                                                                                    //��� b - ������ ����������� ����, A - ������ ���� ����, C - ����� �����, r - ������ �����
    auto discriminant = Halfb * Halfb - a * c;
    if (discriminant < 0) return false;                                             //���� ��� ������ - ������ ��� �� ������������ �� ������.
    auto sqrtd = sqrt(discriminant);

    //���� ��������� � ��� ����� �����������, ����������� � ����� ���� ������, �.�. ����� tMin � tMax
    auto root = (-Halfb - sqrtd) / a;
    if (root < tMin || tMax < root) {
        root = (-Halfb + sqrtd) / a;
        if (root < tMin || tMax < root)
            return false;
    }

    //���������� ��������� ����� �����������
    rec.t = root;
    rec.intersection = r.at(rec.t);
    QVector3D outwardNormal = (rec.intersection - center) / radius;
    rec.SetFaceNormal(r, outwardNormal);
    rec.MatPtr = MatPtr;

    return true;
}

#endif