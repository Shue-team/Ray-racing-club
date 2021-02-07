cmake_minimum_required(VERSION 3.10)
project(Ray-racing-club)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Widgets Concurrent REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_executable(Ray-racing-club
        Main.cpp
        MainWindow.cpp
        MainWindow.h
        MainWindow.ui
        CommonMath.h
        Hittable/Hittable.h
        Ray.h
        Renderer.cpp
        Renderer.h
        Hittable/Sphere.cpp
        Hittable/Sphere.h Camera.cpp Camera.h)
target_link_libraries(Ray-racing-club Qt5::Widgets Qt5::Concurrent)