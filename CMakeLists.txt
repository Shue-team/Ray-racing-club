cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(Ray-racing-club LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(Ray-racing-club Main.cpp
    Hittable/Scene.h
    Hittable/Scene.cu
    Math/Math.h
    Hittable/Hittable.h
    Math/Ray.h
    Material/MaterialDef.h
    Hittable/HittableDef.h
    Hittable/HittableDef.cu
    Hittable/Sphere.cu
    Hittable/Sphere.h
    Hittable/HittableList.h
    Hittable/HittableList.cu
    Math/Vector3D.cu
    Math/Vector3D.h
    Memory/Unified.cu
    Memory/Unified.h
    Math/Ray.cu
    ErrorProcessing/Invalidatable.h
    ErrorProcessing/Invalidatable.cpp
    ErrorProcessing/ErrorHandling.h
    ErrorProcessing/ErrorHandling.cpp
    Material/Lambertian.cu
    Material/Lambertian.h
    Math/Rand.h
    Material/Material.h
    Material/MaterialDef.cu
    Material/Metal.h
    Material/Metal.cu
    Material/Dielectric.cu
    Material/Dielectric.h
    Render/Renderer.cu
    Render/Renderer.h
    Render/Camera.cu
    Render/Camera.h)

add_subdirectory(UI)

target_link_libraries(Ray-racing-club UI_lib)
