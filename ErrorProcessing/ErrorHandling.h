//
// Created by Vladimir on 07.03.2021.
//

#ifndef RAY_RACING_CLUB_ERRORHANDLING_H
#define RAY_RACING_CLUB_ERRORHANDLING_H

#include <cuda_runtime.h>

#include "Invalidatable.h"

void handleError(cudaError err, const char* funcName, const char* fileName,
                 int line, Invalidatable* sourceObject = nullptr);

#define catchError(val) { \
    handleError((val), #val, __FILE__, __LINE__); \
}

#define checkError(funcName) { \
    handleError(cudaGetLastError(), funcName, __FILE__, __LINE__); \
}

#define catchErrorInClass(val) { \
    handleError((val), #val, __FILE__, __LINE__, this); \
}

#define checkErrorInClass(funcName) { \
    handleError(cudaGetLastError(), funcName, __FILE__, __LINE__, this); \
}

#endif //RAY_RACING_CLUB_ERRORHANDLING_H
