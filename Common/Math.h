//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_MATH_H
#define RAY_RACING_CLUB_MATH_H

#include <limits>
#include <random>

#include "cuda_runtime.h"

constexpr float infinity = std::numeric_limits<float>::infinity();

using uchar8 = unsigned char;
using uint32 = unsigned int;

template<class T>
__host__ __device__ const T& clamp( const T& value, const T& low, const T& high) {
    return value < low ? low : high < value ? high : value;
}

#endif //RAY_RACING_CLUB_MATH_H
