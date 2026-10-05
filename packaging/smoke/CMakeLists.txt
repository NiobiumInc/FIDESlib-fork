# Standalone consumer of an installed FIDESlib package, for consumers who use CMake.
# packaging/test-package.sh does not build this -- it links src/smoke.cpp with plain
# -I/-L/-l flags instead, since find_package(CONFIG) is CMake-specific and not every
# consumer has CMake at all. This file is kept as a working example for those who do:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<unpacked tarball>
#   cmake --build build

cmake_minimum_required(VERSION 3.16)
project(fideslib-smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fideslib REQUIRED CONFIG)

# The package records the toolchain it was built with rather than imposing it. A
# mismatched C++ ABI is the likeliest cause of a link failure here, so say so early.
if(FIDESLIB_CXX_COMPILER_ID AND NOT CMAKE_CXX_COMPILER_ID STREQUAL FIDESLIB_CXX_COMPILER_ID)
  message(WARNING "fideslib was built with ${FIDESLIB_CXX_COMPILER_ID} "
                  "${FIDESLIB_CXX_COMPILER_VERSION}; this build uses "
                  "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
endif()

add_executable(fideslib-smoke src/smoke.cpp)
target_link_libraries(fideslib-smoke PRIVATE fideslib::fideslib)
