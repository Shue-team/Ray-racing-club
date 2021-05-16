cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(Ray-racing-club LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_PREFIX_PATH "A:\\Qt\\5.15.2\\msvc2019_64\\lib\\cmake") #personal
find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

cuda_add_executable(Ray-racing-club
        Main.cpp
        UI/MainWindow.cpp
        UI/MainWindow.h
        UI/MainWindow.ui
        UI/Controller.cpp
        UI/Controller.h
        Math/Math.h
        Math/Ray.h
        Math/Vector3D.cu
        Math/Vector3D.h
        Math/Ray.cu
        Math/Rand.h
        Math/Quaternion.cu
        Math/Quaternion.h
        Hittable/Hittable.h
        Render/Renderer.cu
        Render/Renderer.h
        Hittable/Sphere.cu
        Hittable/Sphere.h
        Hittable/HittableList.h
        Hittable/HittableList.cu
        Render/Camera.cu
        Render/Camera.h
        Memory/Unified.cu
        Memory/Unified.h
        Render/Renderer.cpp
        ErrorProcessing/Invalidatable.h
        ErrorProcessing/Invalidatable.cpp
        ErrorProcessing/ErrorHandling.h
        ErrorProcessing/ErrorHandling.cpp
        Material/Lambertian.cu
        Material/Lambertian.h
        Material/Metal.h
        Material/Metal.cu
        Material/Dielectric.cu
        Material/Dielectric.h

        )

target_link_libraries(Ray-racing-club Qt5::Widgets)
