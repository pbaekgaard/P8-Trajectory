cmake_minimum_required(VERSION 3.12)

# Define the project name
set(This OSTC_Tests)

# Collect all .cpp files in the current directory and subdirectories
file(GLOB_RECURSE Sources
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Add an executable for the tests
add_executable(${This} ${Sources})

find_package(Python COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(pybind
        compress/compress_tests.cpp  # Contains the pybind11 module definition
)

# Link to the required libraries
target_link_libraries(${This} PUBLIC
        gtest_main
        ostc_lib
        pybind11::pybind11
)

# Add a CTest to run the tests
add_test(
        NAME ${This}
        COMMAND ${This}
)
