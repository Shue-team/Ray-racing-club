#include "Common/Vector3D.h"

class Quaternion {
public:
    Quaternion(float angle, const Vector3D& axis);
    Quaternion(float s, float x, float y, float z);
    Vector3D rotate(const Vector3D& vector);
    Quaternion operator*(const Vector3D& a);
    Quaternion operator*(const Quaternion& a);
private:
    Quaternion getConjugate();
    float mReal;
    Vector3D mImag;
};