add_library(data_loader data_loader.cpp)

set(OpenCV_DIR D:/opencv/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(D:dlib-19.13/dlib dlib_build)
target_link_libraries (data_loader dlib::dlib)
