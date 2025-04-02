cmake_minimum_required(VERSION 3.12)
project(OSTC)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)  # Ensure this path is correct
include_directories(include)

# Add your source files
add_library(ostc_lib STATIC
    src/trajectory.cpp
    src/distance.cpp
)

# Create the Python extension module
pybind11_add_module(ostc 
    src/main.cpp  # Contains the pybind11 module definition
)

target_link_libraries(ostc PRIVATE ostc_lib pybind11::pybind11)
