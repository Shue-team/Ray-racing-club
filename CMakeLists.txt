cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Ray-racing-club LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "D:\\Qt\\5.15.2\\msvc2019_64\\lib\\cmake") #personal
find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

cuda_add_executable(Ray-racing-club
        Main.cpp
        MainWindow.cpp
        MainWindow.h
        MainWindow.ui
        Common/Math.h
        Hittable/Hittable.h
        Ray.h
        Renderer.cu
        Renderer.h
        Hittable/Sphere.cu
        Hittable/Sphere.h
        Camera.cu
        Camera.h
        Common/Vector3D.cu
        Common/Vector3D.h
        Managed.cu
        Managed.h
        Renderer.cpp
        Ray.cu
        Common/Invalidatable.h
        Common/Invalidatable.cpp)

target_link_libraries(Ray-racing-club Qt5::Widgets)
