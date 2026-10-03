cmake_minimum_required(VERSION 3.25.2)
cmake_policy(VERSION 3.21.3...3.30)

############### Project configuration ###################

# Set target architectures.
if (NOT DEFINED FIDESLIB_ARCH)
    set(FIDESLIB_ARCH "80-real;86-real;89-real;90-real;90-virtual;100-real;120-real" CACHE STRING "Set the backend architecture target.")
endif ()

# Set build type.
set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Set installation paths.
if (NOT DEFINED FIDESLIB_INSTALL_PREFIX)
    set(FIDESLIB_INSTALL_PREFIX "/usr/local" CACHE STRING "FIDESlib installation path.")
else ()
    get_filename_component(TMP_PATH_FIDES "${FIDESLIB_INSTALL_PREFIX}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "FIDESlib install path at: ${TMP_PATH_FIDES}")
    set(FIDESLIB_INSTALL_PREFIX "${TMP_PATH_FIDES}" CACHE STRING "FIDESlib installation path." FORCE)
endif ()
if (NOT DEFINED OPENFHE_INSTALL_PREFIX)
    set(OPENFHE_INSTALL_PREFIX "/usr/local" CACHE STRING "OpenFHE installation path.")
else ()
    get_filename_component(TMP_PATH_OPENFHE "${OPENFHE_INSTALL_PREFIX}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "OpenFHE install path at: ${TMP_PATH_OPENFHE}")
    set(OPENFHE_INSTALL_PREFIX ${TMP_PATH_OPENFHE} CACHE STRING "OpenFHE installation path." FORCE)
endif ()

# Option to auto-install patched OpenFHE.
option(FIDESLIB_INSTALL_OPENFHE "Set ON to install a patched version of OpenFHE." OFF)

# Option to compile the benchmark suite.
option(FIDESLIB_COMPILE_BENCHMARKS "Set ON to compile the FIDESlib benchmark suite." ON)

# Option to compile the test suite.
option(FIDESLIB_COMPILE_TESTS "Set ON to compile the FIDESlib test suite." ON)

# CUDA backend. ON (default) requires the CUDA toolchain and builds the CUDA engine; OFF builds
# CPU-only (CPU/OpenFHE is always available, no CUDA toolchain needed). Future accelerator backends
# are added as their own FIDESLIB_ENABLE_<backend> flag.
option(FIDESLIB_ENABLE_CUDA "Build the CUDA backend (CPU/OpenFHE is always available)." ON)

# Export compile commands for debugging.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


############### CPP ###################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

############### CUDA Configuration ###################

if (FIDESLIB_ENABLE_CUDA)
    # Set CUDA_PATH if not defined previously.
    if (NOT DEFINED CUDA_PATH)
        set(CUDA_PATH "/usr/local/cuda")
    endif ()

    set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc")

    # Check if CUDA is installed.
    if (NOT EXISTS "${CUDA_PATH}")
        message(FATAL_ERROR "CUDA not found at ${CUDA_PATH}. Please install CUDA.")
    endif ()

    # Global language config.
    set(CMAKE_CUDA_ARCHITECTURES "${FIDESLIB_ARCH}")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS ON)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif ()

################ Project Definition ###################

if (NOT FIDESLIB_ENABLE_CUDA)
    project(fideslib VERSION 1.0 DESCRIPTION "Homomorphic Encryption Library" LANGUAGES CXX)
else ()
    project(fideslib VERSION 1.0 DESCRIPTION "Homomorphic Encryption Library" LANGUAGES CXX CUDA)
endif ()

################ Extra Flags ###################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if (FIDESLIB_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -Wno-restrict -Wall -rdc=true --diag-suppress=3133,611")
endif ()

############### Utils ###################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchContent)
include(CMakePrintHelpers)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(ClangFormat)


################ CUDA Libraries ###################

if (FIDESLIB_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if (NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "CUDA not found!")
    else ()
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA include path: ${CUDAToolkit_INCLUDE_DIRS}")
    endif ()

    # Check for NCCL installation. Do not do anything more. It is managed by CUDA. This is just to
    # enable conditional NCCL-dependent code.
    find_library(nccl_FOUND NAMES nccl HINTS /usr/local/cuda/lib64 /usr/lib /usr/local/lib)
    if (nccl_FOUND)
        message(STATUS "Found NCCL: ${nccl_FOUND}")
        add_compile_definitions(MULTI_GPU NCCL RCCL)
    else ()
        message(STATUS "NCCL not found: building without NCCL support")
    endif ()
endif ()

############### OpenFHE Configuration ###################

# Default runs will not update nor install OpenFHE. Specify manually to install.
if (FIDESLIB_INSTALL_OPENFHE)
    execute_process(COMMAND ./build.sh ${OPENFHE_INSTALL_PREFIX}
            WORKING_DIRECTORY "../deps"
            COMMAND_ERROR_IS_FATAL ANY)
endif ()

find_package(OpenFHE 1.5.1 CONFIG EXACT REQUIRED PATHS "${OPENFHE_INSTALL_PREFIX}")

option(BUILD_STATIC "Set to ON to include static versions of the library" ON)

if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Use Clang's OpenMP implementation.

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use Clang's OpenMP implementation.
    set(FIX_OMP_FLAGS "-fopenmp=libomp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(FIX_OMP_FLAGS "-fopenmp")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIX_OMP_FLAGS}")
if (FIDESLIB_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${FIX_OMP_FLAGS}")
endif ()

############### Optimization Flags ###################

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    if (FIDESLIB_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -Xcompiler -O0 -Xcompiler -g -G")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT supported OUTPUT error)

    if (supported)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(STATUS "IPO / LTO not supported: <${error}>")
    endif ()
    if (FIDESLIB_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if (FIDESLIB_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O2 -Xcompiler -g")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    if (FIDESLIB_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Os")
    endif ()
endif ()

############### Configuration export ###################

set(CMAKE_INSTALL_PREFIX "${FIDESLIB_INSTALL_PREFIX}")

set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
set(INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
set(INSTALL_BINARY_DIR "${CMAKE_INSTALL_BINDIR}")
set(INSTALL_DATA_DIR "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake")


############### NVTX ###################
if (FIDESLIB_ENABLE_CUDA)
    include(nvtx/c/nvtxImportedTargets.cmake)
endif ()

############### FIDESlib Definition ###################

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api")

if (FIDESLIB_ENABLE_CUDA)
    file(GLOB_RECURSE SOURCE_FILES_CUDA ${SOURCE_DIR}/*.cu)
    file(GLOB_RECURSE SOURCE_FILES_CXX ${SOURCE_DIR}/*.cpp)
    file(GLOB_RECURSE HEADER_FILES_CUDA ${SOURCE_DIR}/*.cuh)
    file(GLOB_RECURSE HEADER_FILES_CXX ${SOURCE_DIR}/*.hpp ${SOURCE_DIR}/*.h ${SOURCE_DIR}/*.inc)
    set_source_files_properties(${SOURCE_FILES_CUDA} ${HEADER_FILES_CUDA} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${SOURCE_FILES_CXX} ${HEADER_FILES_CXX} PROPERTIES LANGUAGE CXX)
endif ()

file(GLOB_RECURSE SOURCE_FILES_API ${API_DIR}/*.cpp)
# engine/cuda/CudaEngine.cpp is the CUDA backend; exclude it from CPU-only builds.
if (NOT FIDESLIB_ENABLE_CUDA)
    list(REMOVE_ITEM SOURCE_FILES_API ${API_DIR}/engine/cuda/CudaEngine.cpp)
endif ()
file(GLOB_RECURSE HEADER_FILES_PUBLIC ${API_DIR}/*.hpp)
set_source_files_properties(${HEADER_FILES_PUBLIC} ${SOURCE_FILES_API} PROPERTIES LANGUAGE CXX)

add_library(fideslib STATIC)

if (NOT FIDESLIB_ENABLE_CUDA)
    target_sources(fideslib PRIVATE ${SOURCE_FILES_API})
else ()
    target_compile_definitions(fideslib PUBLIC FIDESLIB_ENABLE_CUDA)
    target_sources(fideslib PRIVATE ${SOURCE_FILES_CXX} ${SOURCE_FILES_CUDA} ${SOURCE_FILES_API})
endif ()

if (FIDESLIB_ENABLE_CUDA)
    target_sources(fideslib PRIVATE
            FILE_SET private_headers
            TYPE HEADERS
            BASE_DIRS "${SOURCE_DIR}"
            DESTINATION "${INSTALL_INCLUDE_DIR}"
            FILES ${HEADER_FILES_CXX} ${HEADER_FILES_CUDA}
    )
endif ()
target_sources(fideslib PUBLIC
        FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS "${API_DIR}"
        DESTINATION "${INSTALL_INCLUDE_DIR}"
        FILES ${HEADER_FILES_PUBLIC}
)

target_include_directories(fideslib
        # api/ is the include root so headers use api-relative paths (e.g. "engine/cuda/CudaEngine.hpp").
        PUBLIC $<BUILD_INTERFACE:${API_DIR}>
)
if (FIDESLIB_ENABLE_CUDA)
    target_include_directories(fideslib PRIVATE nvtx/c/include)
endif ()

target_link_libraries(fideslib
        PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEpke_static.a"
        PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEbinfhe_static.a"
        PRIVATE "${OpenFHE_LIBDIR}/libOPENFHEcore_static.a"
)

if (FIDESLIB_ENABLE_CUDA)
    target_link_libraries(fideslib
            PRIVATE CUDA::cudart
            PRIVATE CUDA::cuda_driver
    )

    if (nccl_FOUND)
        message(STATUS "FIDESlib linked against NCCL")
        target_link_libraries(fideslib PRIVATE nccl)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNCCL")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DNCCL")
    endif ()
endif ()

target_link_options(fideslib PRIVATE -static-libstdc++ -static-libgcc)

set_target_properties(fideslib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PREFIX ""
)
if (FIDESLIB_ENABLE_CUDA)
    set_target_properties(fideslib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

############### GPU Checker ###################

if (FIDESLIB_ENABLE_CUDA)
    add_executable(gpu-test)

    file(GLOB_RECURSE GPUTEST_SOURCES other/GPUTest.cpp)
    set_source_files_properties(${GPUTEST_SOURCES} PROPERTIES LANGUAGE CUDA)

    target_sources(gpu-test PRIVATE ${GPUTEST_SOURCES})
    set_target_properties(gpu-test PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(gpu-test PRIVATE CUDA::cudart)

    set_target_properties(gpu-test PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
    )
endif ()

############### Dummy executable ###################

add_executable(dummy)

file(GLOB_RECURSE DUMMY_SOURCES other/Dummy.cpp)
set_source_files_properties(${DUMMY_SOURCES} PROPERTIES LANGUAGE CXX)

target_sources(dummy PRIVATE ${DUMMY_SOURCES})
target_link_libraries(dummy PRIVATE fideslib)
set_target_properties(dummy PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (FIDESLIB_ENABLE_CUDA)
    target_sources(dummy PUBLIC
            FILE_SET dependency_headers
            TYPE HEADERS
            BASE_DIRS ${SOURCE_DIR}
            FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
    )
    set_target_properties(dummy PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

############### Library Installation ###################

install(
        TARGETS fideslib
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION "${INSTALL_LIBRARY_DIR}"
        ARCHIVE DESTINATION "${INSTALL_LIBRARY_DIR}"
        RUNTIME DESTINATION "${INSTALL_BINARY_DIR}"
        INCLUDES DESTINATION "${INSTALL_INCLUDE_DIR}"
        FILE_SET public_headers DESTINATION "${INSTALL_INCLUDE_DIR}"
)

# The private_headers FILE_SET (src/ implementation headers) only exists in GPU builds.
if (FIDESLIB_ENABLE_CUDA)
    install(
            TARGETS fideslib
            FILE_SET private_headers DESTINATION "${INSTALL_INCLUDE_DIR}"
    )
endif ()

install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION "${INSTALL_DATA_DIR}"
)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${INSTALL_DATA_DIR}"
)

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        @ONLY
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${INSTALL_DATA_DIR}"
)

# Also install private headers so downstream targets can include implementation details when needed.
if (FIDESLIB_ENABLE_CUDA)
    install(
            DIRECTORY "${SOURCE_DIR}/"
            DESTINATION "${INSTALL_INCLUDE_DIR}"
            FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h"
            PATTERN "*.cuh"
            PATTERN "*.inc"
    )
endif ()


############### Test ###################

if (FIDESLIB_COMPILE_TESTS)

    # Use system-installed GTest when available (e.g. inside a nix derivation);
    # fall back to FetchContent for dev-shell use where there is network access.
    find_package(GTest QUIET)
    if (GTest_FOUND)
        add_library(gtest      ALIAS GTest::gtest)
        add_library(gtest_main ALIAS GTest::gtest_main)
    else ()
        set(INSTALL_GTEST OFF)
        FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG main)
        FetchContent_MakeAvailable(googletest)
    endif ()
    include(GoogleTest)

    set(TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

    if (FIDESLIB_ENABLE_CUDA)
        file(GLOB_RECURSE TEST_SOURCE ${TEST_SOURCE_DIR}/*.cpp ${TEST_SOURCE_DIR}/*.cu)
        set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)

        add_executable(fideslib-test)

        target_sources(fideslib-test PRIVATE ${TEST_SOURCE})

        target_sources(fideslib-test PRIVATE
                FILE_SET dependency_headers
                TYPE HEADERS
                BASE_DIRS ${SOURCE_DIR}
                FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
        )

        target_link_libraries(fideslib-test
                PRIVATE gtest
                PRIVATE gtest_main
                PRIVATE fideslib
        )

        set_target_properties(fideslib-test PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                CUDA_SEPARABLE_COMPILATION ON
                POSITION_INDEPENDENT_CODE ON
        )
    else ()
        # CPU-only build: the .cu suite (GPU-vs-OpenFHE parity) needs CUDA and is excluded.
        # The api-level suites compile as plain CXX and run against the CPU backend.
        add_executable(fideslib-cpu-test)
        target_sources(fideslib-cpu-test PRIVATE
                ${TEST_SOURCE_DIR}/ApiTests.cpp
                ${TEST_SOURCE_DIR}/ApiParityTest.cpp)
        target_link_libraries(fideslib-cpu-test PRIVATE gtest gtest_main fideslib)
        set_target_properties(fideslib-cpu-test PROPERTIES POSITION_INDEPENDENT_CODE ON)
        gtest_discover_tests(fideslib-cpu-test DISCOVERY_MODE PRE_TEST)
    endif ()

endif ()


############### Benchmark ###################

if (FIDESLIB_COMPILE_BENCHMARKS AND FIDESLIB_ENABLE_CUDA)

    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_INSTALL_DOCS OFF)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_EXCEPTIONS ON CACHE BOOL "" FORCE)
    # Fuerza regex para evitar autodetección (elige uno que funcione en tu sistema):
    set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)

    FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG main)
    FetchContent_MakeAvailable(benchmark)

    set(BENCH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bench)

    file(GLOB_RECURSE BENCH_SOURCE_FILES ${BENCH_SOURCE_DIR}/*.cu ${BENCH_SOURCE_DIR}/*.cpp)
    set_source_files_properties(${BENCH_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

    add_executable(fideslib-bench)

    target_sources(fideslib-bench PRIVATE ${BENCH_SOURCE_FILES})
    target_sources(fideslib-bench PUBLIC
            FILE_SET dependency_headers
            TYPE HEADERS
            BASE_DIRS ${SOURCE_DIR}
            FILES ${HEADER_FILES_CUDA} ${HEADER_FILES_CXX}
    )

    target_link_libraries(fideslib-bench
            PRIVATE benchmark::benchmark
            PRIVATE fideslib
    )

    set_target_properties(fideslib-bench PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
    )

endif ()
