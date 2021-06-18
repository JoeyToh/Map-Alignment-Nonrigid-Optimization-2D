cmake_minimum_required(VERSION 3.5)

project(proj_ceres)

find_package(Ceres REQUIRED COMPONENTS EigenSparse)

add_executable(homography Homography.cc)
target_link_libraries(homography Ceres::ceres)
