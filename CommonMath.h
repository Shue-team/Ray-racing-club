//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_COMMONMATH_H
#define RAY_RACING_CLUB_COMMONMATH_H

#include <QVector3D>

#include <limits>
#include <random>
#include <algorithm>

using Point3D = QVector3D;
using Color = QVector3D;

constexpr double infinity = std::numeric_limits<double>::infinity();

inline double randomDouble() {
    static std::uniform_real_distribution<> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

#endif //RAY_RACING_CLUB_COMMONMATH_H
