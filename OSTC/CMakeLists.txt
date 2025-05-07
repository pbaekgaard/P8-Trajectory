cmake_minimum_required(VERSION 3.12)
set(This OSTC)
project(${This})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(Debug "Enable Executing in Debug Mode" OFF)

if(Debug)
    # Find Python and Pybind11
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(extern/pybind11)
    include_directories(include)


    # Testing
    include(CTest)
    enable_testing()
    add_subdirectory(extern/googletest)
    add_subdirectory(tests)

    # Add source files
    add_library(ostc_lib STATIC
        src/trajectory.cpp
        src/distance.cpp
    )
    add_executable(${This}
        src/main.cpp
            # Change this if needed
    )
    target_link_libraries(${This} PRIVATE ostc_lib)
    target_compile_definitions(${This} PRIVATE Debug)  # Define Debug for preprocessor
else()
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
endif()
