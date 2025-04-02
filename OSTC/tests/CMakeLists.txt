cmake_minimum_required(VERSION 3.12)
set(This OSTC_Tests)
set(Sources
    test1.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} PUBLIC
    gtest_main
    ostc_lib
)

add_test(
    NAME ${This}
    COMMAND ${This}
)
