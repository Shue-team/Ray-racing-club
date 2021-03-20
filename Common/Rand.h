//
// Created by awesyr on 20.03.2021.
//

#ifndef RAY_RACING_CLUB_RAND_H
#define RAY_RACING_CLUB_RAND_H

#include <curand_kernel.h>

__device__ inline float randomFloat(curandState* randState) {
    return 1.0f - curand_uniform(randState);
}

__device__ inline float randomFloat(float min, float max, curandState* randState) {
    return min + (max - min) * randomFloat(randState);
}

#endif //RAY_RACING_CLUB_RAND_H
