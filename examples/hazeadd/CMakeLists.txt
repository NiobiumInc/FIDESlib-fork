cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CXX_STANDARD 20)
project(hazeadd LANGUAGES CXX)

find_package(fideslib REQUIRED CONFIG)

set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

add_executable(hazeadd src/hazeadd.cpp)
target_link_libraries(hazeadd PRIVATE fideslib::fideslib)
