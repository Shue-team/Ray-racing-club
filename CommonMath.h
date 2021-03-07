//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_COMMONMATH_H
#define RAY_RACING_CLUB_COMMONMATH_H

#include <limits>
#include <random>


constexpr float infinity = std::numeric_limits<float>::infinity();
//using uchar8 = unsigned char;

inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#endif //RAY_RACING_CLUB_COMMONMATH_H
