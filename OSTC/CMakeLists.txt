cmake_minimum_required(VERSION 3.30)
project(OSTC)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
add_executable(OSTC src/main.cpp)
add_library(trajectory STATIC src/trajectory.cpp)
add_library(distance STATIC src/distance.cpp)
target_link_libraries(OSTC trajectory distance)
