cmake_minimum_required(VERSION 3.10)
project(Ray-racing-club)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "C:\\Users\\admin\\Desktop\\1\\QT\\5.15.2\\msvc2019\\lib\\cmake")


find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_executable(Ray-racing-club Main.cpp MainWindow.cpp MainWindow.h MainWindow.ui CommonMath.h Hittable.h Ray.h Renderer.cpp Renderer.h "HitList.h" "Sphere.h")
target_link_libraries(Ray-racing-club Qt5::Widgets)