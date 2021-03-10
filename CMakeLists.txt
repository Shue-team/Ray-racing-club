cmake_minimum_required(VERSION 3.10)
project(Ray-racing-club)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.14.2\\5.14.2\\msvc2017_64\\lib\\cmake")
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_executable(Ray-racing-club
        Main.cpp
        MainWindow.cpp
        MainWindow.h
        MainWindow.ui
        Hittable/Hittable.h
        Ray.h
        Renderer.cpp
        Renderer.h
        Hittable/Sphere.cpp
        Hittable/Sphere.h
        Vector3D.cpp
        Vector3D.h
        Hittable/HittableList.cpp
        Hittable/HittableList.h
        Materials/Material.h
        Materials/Material.cpp
        CommonMath.h
        CommonMath.cpp)
target_link_libraries(Ray-racing-club Qt5::Widgets)