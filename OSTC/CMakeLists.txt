cmake_minimum_required(VERSION 3.12)
project(OSTC)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)
include_directories(include)

# Add source files
add_library(ostc_lib STATIC
    src/trajectory.cpp
    src/distance.cpp
)

# Check for Debug flag
option(Debug "Enable executable build instead of Python module" OFF)

if(Debug)
    # Debug build (Executable)
    add_executable(OSTC
        src/main.cpp  # Change this if needed
    )
    target_link_libraries(OSTC PRIVATE ostc_lib)
    target_compile_definitions(OSTC PRIVATE Debug)  # Define Debug for preprocessor
else()
    # Default build (Python extension)
    pybind11_add_module(ostc
        src/main.cpp
    )
    target_link_libraries(ostc PRIVATE ostc_lib pybind11::pybind11)
endif()
