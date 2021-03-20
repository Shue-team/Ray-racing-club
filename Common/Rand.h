//
// Created by awesyr on 20.03.2021.
//

#ifndef RAY_RACING_CLUB_RAND_H
#define RAY_RACING_CLUB_RAND_H

#include <curand_kernel.h>

#ifdef __CUDA_ARCH__
__device__ inline float randomFloat(curandState* randState) {
    return 1 - curand_uniform(randState);
}
#else
__host__ inline float randomFloat(curandState* randState) {
    return rand() / (RAND_MAX + 1.0f);
}
#endif

__host__ __device__ inline float randomFloat(float min, float max, curandState* randState) {
    return min + (max - min) * randomFloat(randState);
}

#endif //RAY_RACING_CLUB_RAND_H
