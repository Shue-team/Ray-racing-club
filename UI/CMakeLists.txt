project(UI)

set(CMAKE_PREFIX_PATH "D:\\Qt\\5.15.2\\msvc2019_64\\lib\\cmake")

find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(HEADER_FILES MainWindow.h
    WorldEdit.h SphereEdit.cpp
    TriangleEdit.h MaterialEdit.h
    HittableEdit.h ../Render/Renderer.h
    MetalEdit.h DielectricEdit.h
    LambertianEdit.h)
set(SOURCE_FILES MainWindow.cpp
    WorldEdit.cpp TriangleEdit.cpp
    SphereEdit.h MainWindow.ui
    MetalEdit.cpp DielectricEdit.cpp
    LambertianEdit.cpp)

add_library(UI_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(UI_lib Qt5::Widgets)
