//
// Created by Vladimir on 07.03.2021.
//
#include <iostream>

#include "ErrorHandling.h"

void handleError(cudaError err, const char* funcName, const char* fileName,
                 int line, Invalidatable* sourceObject) {
    if (err == cudaSuccess) { return; }

    std::cerr << "An error " << cudaGetErrorString(err)
              << " has occurred in " << funcName << std::endl
              << "File: " << fileName << std::endl
              << "Line: " << line << std::endl;


    if (sourceObject) {
        sourceObject->setValid(false);
    }
}
