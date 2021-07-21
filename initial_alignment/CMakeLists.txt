cmake_minimum_required(VERSION 3.5)

project(initial_alignment)

find_package(Ceres REQUIRED COMPONENTS EigenSparse)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

include_directories("${JSONCPP_INCLUDE_DIRS}")

link_libraries(${JSONCPP_LIBRARIES})
add_executable(homography Homography.cc)

target_link_libraries(homography Ceres::ceres)
target_link_libraries(homography ${JSONCPP_LIBRARIES})