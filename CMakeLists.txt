cmake_minimum_required(VERSION 3.25.2)
cmake_policy(VERSION 3.21.3...3.30)

############### Project configuration ###################

# Set target architectures.
if (NOT DEFINED FIDESLIB_ARCH)
    set(FIDESLIB_ARCH "75-real;80-real;86-real;89-real;90-real;90-virtual;100-real;120-real" CACHE STRING "Set the backend architecture target.")
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

# Build a shared, redistributable libfideslib instead of the static fideslib.a.
# The shared build absorbs OpenFHE's static archives whole and exports its own
# symbols plus lbcrypto's, so a consumer needs no separate OpenFHE install. OFF
# by default: existing consumers link $FIDESLIB_DIR/lib64/fideslib.a directly.
option(FIDESLIB_BUILD_SHARED "Build libfideslib as a redistributable shared library." OFF)

# CUDA backend. ON (default) requires the CUDA toolchain and builds the CUDA engine; OFF builds
# CPU-only (CPU/OpenFHE is always available, no CUDA toolchain needed). Future accelerator backends
# are added as their own FIDESLIB_ENABLE_<backend> flag.
option(FIDESLIB_ENABLE_CUDA "Build the CUDA backend (CPU/OpenFHE is always available)." ON)

# haze (FHETCH) backend. OFF by default. When ON, the vendored deps/niobium-haze
# submodule is built automatically (see the FIDESLIB_ENABLE_HAZE block below).
# HAZE is orthogonal to CUDA — both may be ON simultaneously.
option(FIDESLIB_ENABLE_HAZE "Enable the haze (FHETCH) backend (Niobium accelerator)." OFF)
# Optional override: point at an externally-built haze tree (containing build/libhaze.{so,dylib},
# include/, and replay_bridge/include/) to skip the vendored auto-build. Empty (default) uses the
# vendored deps/niobium-haze submodule. Set via -DFIDESLIB_HAZE_DIR=<path>.
set(FIDESLIB_HAZE_DIR "" CACHE PATH "Override: externally-built haze tree (empty uses the vendored submodule).")
# Override the library path itself. FIDESLIB_HAZE_DIR assumes haze's in-tree layout
# (<dir>/build/libhaze.so); a release build configures haze into its own build
# directory, so the headers and the artifact no longer share a root.
set(FIDESLIB_HAZE_LIB "" CACHE FILEPATH "Override: path to libhaze.{so,dylib} (empty derives it from FIDESLIB_HAZE_DIR).")

# Export compile commands for debugging.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")


############### CPP ###################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# Bare names, resolved through PATH. Only applied when the caller has not chosen a
# compiler: an unconditional set() here silently overrode CC/CXX and any
# -DCMAKE_CXX_COMPILER on the command line.
if (NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(CMAKE_C_COMPILER "gcc")
endif ()
if (NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    set(CMAKE_CXX_COMPILER "g++")
endif ()

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

if (APPLE)
    # Homebrew's libomp is keg-only (off the default search path), unlike Linux's
    # system-wide libgomp. Every target's compile+link line gets FIX_OMP_FLAGS below,
    # so this is computed once here and reused by the find_package(OpenMP) hints
    # further down for the FIDESLIB_BUILD_SHARED target.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(_LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    else ()
        set(_LIBOMP_PREFIX "/usr/local/opt/libomp")
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use Clang's OpenMP implementation.
    set(FIX_OMP_FLAGS "-fopenmp=libomp")
    if (APPLE)
        set(FIX_OMP_FLAGS "${FIX_OMP_FLAGS} -L${_LIBOMP_PREFIX}/lib -I${_LIBOMP_PREFIX}/include")
    endif ()
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

    # Overridable: gcc-13 ICEs linking large host binaries against LTO
    # bytecode, so consumers must be able to keep Release -O3 with IPO off.
    option(FIDESLIB_ENABLE_IPO "Enable IPO/LTO in Release builds" ON)
    if (supported AND FIDESLIB_ENABLE_IPO)
        message(STATUS "IPO / LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    elseif (NOT FIDESLIB_ENABLE_IPO)
        message(STATUS "IPO / LTO disabled (FIDESLIB_ENABLE_IPO=OFF)")
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
# lib/cmake/<pkg> is where find_package() looks with only CMAKE_PREFIX_PATH set,
# so a relocated package needs no -Dfideslib_DIR.
set(INSTALL_DATA_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")


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
# engine/haze/*.cpp are the haze backend; exclude them when haze is not enabled.
if (NOT FIDESLIB_ENABLE_HAZE)
    file(GLOB _HAZE_SOURCES ${API_DIR}/engine/haze/*.cpp)
    list(REMOVE_ITEM SOURCE_FILES_API ${_HAZE_SOURCES})
endif ()
file(GLOB_RECURSE HEADER_FILES_PUBLIC ${API_DIR}/*.hpp)
set_source_files_properties(${HEADER_FILES_PUBLIC} ${SOURCE_FILES_API} PROPERTIES LANGUAGE CXX)

if (FIDESLIB_BUILD_SHARED)
    add_library(fideslib SHARED)
else ()
    add_library(fideslib STATIC)
endif ()

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
# The installed public headers include OpenFHE by paths relative to core/ and pke/
# ("lattice/hal/lat-backend.h", "key/evalkey-fwd.h"), so all five roots have to be
# on a consumer's include path, not just include/openfhe. That holds for the static
# build as much as the shared one: api/Ciphertext.hpp includes OpenFHE either way.
# <prefix>/include/fideslib for the API headers, and <prefix>/include itself so the
# haze C ABI resolves as <haze/haze.h> -- the spelling consumers already use.
target_include_directories(fideslib PUBLIC
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Mirror only the OpenFHE roots this install actually has. CMake errors at generate
# time on a non-existent INTERFACE_INCLUDE_DIRECTORIES entry, and third-party/include
# is absent from installs built without NTL/tcmalloc.
foreach (_ofhe_sub "" "/core" "/pke" "/binfhe" "/third-party/include")
    if (EXISTS "${OpenFHE_INCLUDE}${_ofhe_sub}")
        target_include_directories(fideslib PUBLIC
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/openfhe${_ofhe_sub}>)
    endif ()
endforeach ()
if (FIDESLIB_ENABLE_CUDA)
    target_include_directories(fideslib PRIVATE nvtx/c/include)
endif ()

set(_FIDESLIB_OPENFHE_ARCHIVES
        "${OpenFHE_LIBDIR}/libOPENFHEpke_static.a"
        "${OpenFHE_LIBDIR}/libOPENFHEbinfhe_static.a"
        "${OpenFHE_LIBDIR}/libOPENFHEcore_static.a"
)
if (FIDESLIB_BUILD_SHARED)
    # Absorb OpenFHE WHOLE. Two reasons: the public headers expose lbcrypto types
    # (api/Ciphertext.hpp:57 returns std::vector<lbcrypto::DCRTPoly>&), so consumers
    # must be able to resolve lbcrypto symbols from this .so; and linking the archives
    # as ordinary PRIVATE items records their absolute build-host paths in the
    # exported target, which is what makes the current package non-relocatable.
    # $<LINK_LIBRARY:WHOLE_ARCHIVE,...> rather than raw linker flags: CMake picks
    # --whole-archive or -force_load per platform, and -- unlike target_link_options --
    # it does not also feed the CUDA device-link step, where each archive would arrive
    # as -Xcompiler=<path>.a and reach gcc as a source file.
    target_link_libraries(fideslib PRIVATE
            "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_FIDESLIB_OPENFHE_ARCHIVES}>")
    # OPENFHEcore_static calls dlopen/dlsym for its pluggable PRNG engine.
    # OpenFHE's own CMakeLists.txt adds -ldl for that within its own configure run,
    # but that does not carry over here: this is a separate top-level CMake project
    # linking the raw .a directly. Only glibc >= 2.34 merges libdl into libc, so this
    # is silently fine there and a hard link failure on an older glibc.
    target_link_libraries(fideslib PRIVATE ${CMAKE_DL_LIBS})
    # OpenFHE is built WITH_OPENMP=ON, so libgomp is a genuine runtime dependency.
    # COMPONENTS CXX: with CUDA enabled as a language, a bare find_package(OpenMP)
    # also demands OpenMP_CUDA, which no toolkit provides here.
    if (APPLE)
        # Homebrew's libomp is keg-only (not on the default include/library search
        # path), so a bare find_package(OpenMP) fails to detect it. deps/openfhe-src's
        # own CMakeLists.txt (see its "OpenMP logic" section) works around this the
        # same way, and does so successfully against the same toolchain. _LIBOMP_PREFIX
        # is set above, next to FIX_OMP_FLAGS, which already puts -L/-I on every
        # target's compile+link line -- this block only needs to satisfy find_package's
        # own REQUIRED check.
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp")
        set(OpenMP_libomp_LIBRARY "${_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif ()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(fideslib PRIVATE OpenMP::OpenMP_CXX)
else ()
    foreach (_a IN LISTS _FIDESLIB_OPENFHE_ARCHIVES)
        target_link_libraries(fideslib PRIVATE "${_a}")
    endforeach ()
endif ()

if (FIDESLIB_ENABLE_CUDA)
    if (FIDESLIB_BUILD_SHARED)
        # Absorb the CUDA runtime so the package does not require a CUDA toolkit
        # installation on the target. The driver is the one piece that cannot travel:
        # libcuda.so.1 ships with the NVIDIA kernel driver and is version-locked to it,
        # so it is linked against the toolkit stub here and resolved from the system at
        # run time -- which is exactly where it exists on any machine with a CUDA card.
        target_link_libraries(fideslib
                PRIVATE CUDA::cudart_static
                PRIVATE CUDA::cuda_driver
        )
    else ()
        target_link_libraries(fideslib
                PRIVATE CUDA::cudart
                PRIVATE CUDA::cuda_driver
        )
    endif ()

    if (nccl_FOUND)
        message(STATUS "FIDESlib linked against NCCL")
        target_link_libraries(fideslib PRIVATE nccl)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNCCL")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -DNCCL")
    endif ()
endif ()

if (FIDESLIB_ENABLE_HAZE)
    # haze (FHETCH) backend. Vendored as a submodule at deps/niobium-haze and built
    # automatically below. Set -DFIDESLIB_HAZE_DIR=<path> to instead consume an
    # externally-built haze tree (skips the auto-build).
    if (FIDESLIB_HAZE_DIR)
        set(_HAZE_DIR "${FIDESLIB_HAZE_DIR}")
        set(_HAZE_VENDORED OFF)
    else ()
        set(_HAZE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/niobium-haze")
        set(_HAZE_VENDORED ON)
    endif ()

    # haze artifact layout (produced by `make build MODE=release` in the haze tree):
    #   <dir>/build/libhaze.{so,dylib} — the single shipped shared library; absorbs
    #                                    libnbfhetch + replay_bridge + instrumented OpenFHE.
    #   <dir>/include/haze/            — haze.h, haze_types.h (core API)
    #   <dir>/replay_bridge/include/   — haze/replay_bridge.h (hazeReplayBridge* ABI)
    if (FIDESLIB_HAZE_LIB)
        set(_HAZE_SO "${FIDESLIB_HAZE_LIB}")
    else ()
        set(_HAZE_SO "${_HAZE_DIR}/build/libhaze${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif ()

    if (_HAZE_VENDORED)
        if (NOT EXISTS "${_HAZE_DIR}/Makefile")
            message(FATAL_ERROR "vendored haze missing at ${_HAZE_DIR}. Run: "
                                "git submodule update --init --recursive deps/niobium-haze")
        endif ()
        # Auto-build the vendored haze via its Makefile. HAZE_BUILD_E2E_TESTS=0 skips the
        # stock-OpenFHE + e2e build (not needed to produce libhaze). Builds at build time,
        # so configure stays fast; requires cmake/ninja/make on PATH.
        include(ExternalProject)
        ExternalProject_Add(niobium_haze
            SOURCE_DIR        "${_HAZE_DIR}"
            BUILD_IN_SOURCE   1
            CONFIGURE_COMMAND ""
            BUILD_COMMAND     make build MODE=release HAZE_BUILD_E2E_TESTS=0
            INSTALL_COMMAND   ""
            # Always invoke haze's make: it is timestamp-incremental, so this is a
            # cheap no-op when nothing changed and recompiles only what a submodule
            # bump touched (a haze update thus triggers a haze rebuild automatically).
            BUILD_ALWAYS      1
            BUILD_BYPRODUCTS  "${_HAZE_SO}"
        )
    else ()
        if (NOT EXISTS "${_HAZE_SO}")
            message(FATAL_ERROR "haze library not found at ${_HAZE_SO}. Build haze first "
                                "(make build MODE=release in the haze tree) or adjust FIDESLIB_HAZE_DIR.")
        endif ()
    endif ()

    # Single imported target: libhaze absorbs both the core haze API and the
    # replay_bridge symbols (hazeReplayBridge* are OBJECT-library-absorbed at link time).
    add_library(haze::haze SHARED IMPORTED GLOBAL)
    set_target_properties(haze::haze PROPERTIES
        IMPORTED_LOCATION "${_HAZE_SO}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${_HAZE_DIR}/include;${_HAZE_DIR}/replay_bridge/include"
    )
    if (_HAZE_VENDORED)
        # Auto-built: libhaze does not exist at configure time, so CMake does not add
        # its directory to consumers' rpath implicitly. Bake it in explicitly so
        # build-tree executables (fideslib-test, dummy, ...) find libhaze (loaded via
        # @rpath) in the vendored build dir -- true whether fideslib itself is shared
        # or static. This never reaches a packaged/installed consumer:
        # build-release.sh always passes FIDESLIB_HAZE_DIR explicitly, so
        # _HAZE_VENDORED is OFF for that build, and $ORIGIN/patchelf handle the
        # shipped tarball instead. (An external FIDESLIB_HAZE_DIR tree already exists
        # at configure time, so CMake's implicit imported-location rpath covers that
        # non-vendored case on its own.)
        set_property(TARGET haze::haze APPEND PROPERTY
            INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_HAZE_DIR}/build")
        add_dependencies(fideslib niobium_haze)
    endif ()

    target_compile_definitions(fideslib PUBLIC FIDESLIB_ENABLE_HAZE)
    target_link_libraries(fideslib PUBLIC haze::haze)

    message(STATUS "haze (FHETCH) backend enabled: ${_HAZE_SO} (vendored: ${_HAZE_VENDORED})")
endif ()

# -static-libstdc++ in a SHARED library whose public API is C++ would put two copies
# of the C++ runtime on either side of the ABI boundary: mismatched allocation
# ownership, dynamic_cast/catch failing on RTTI identity, two unwinder state machines.
# The shared build relies on gcc-toolset's libstdc++_nonshared.a for its ABI floor
# instead. Keep the flags for the static build, where the consumer links it all.
if (NOT FIDESLIB_BUILD_SHARED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(fideslib PRIVATE -static-libstdc++ -static-libgcc)
endif ()

if (FIDESLIB_BUILD_SHARED)
    set_target_properties(fideslib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            VERSION "${PROJECT_VERSION}"
            SOVERSION "${PROJECT_VERSION_MAJOR}"
    )
    # libhaze ships beside libfideslib in the package, so resolve it relative to the
    # loading library rather than by absolute path.
    if (APPLE)
        set_target_properties(fideslib PROPERTIES INSTALL_RPATH "@loader_path")
    else ()
        set_target_properties(fideslib PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif ()
else ()
    # PREFIX "" yields "fideslib.a"; downstream trees hardcode that name.
    set_target_properties(fideslib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            PREFIX ""
    )
endif ()
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

# OpenFHE headers: api/Ciphertext.hpp and api/CryptoContext.hpp include them, so
# they are part of this package's compile interface whether or not we like it, in
# the static build as much as the shared one.
install(
        DIRECTORY "${OpenFHE_INCLUDE}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/openfhe"
)

# A redistributable package has to carry the pieces the link line depends on, or it
# is only usable on the build host.
if (FIDESLIB_BUILD_SHARED)
    if (FIDESLIB_ENABLE_HAZE)
        # libhaze.so is a sibling runtime dependency; nothing installed it before, so
        # every consumer was pinned to the haze build tree by rpath.
        install(FILES "${_HAZE_SO}" DESTINATION "${INSTALL_LIBRARY_DIR}")
        install(
                DIRECTORY "${_HAZE_DIR}/include/haze"
                DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
        install(
                DIRECTORY "${_HAZE_DIR}/replay_bridge/include/haze"
                DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
    endif ()
endif ()

install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION "${INSTALL_DATA_DIR}"
)

# A redistributable package resolves libhaze from inside itself ($ORIGIN), so the
# build-host haze paths must not be written into the shipped config at all -- they
# would be dead absolute strings naming this machine.
if (FIDESLIB_BUILD_SHARED)
    set(_FIDESLIB_CFG_HAZE_LIB "")
    set(_FIDESLIB_CFG_HAZE_DIR "")
else ()
    # _HAZE_SO/_HAZE_DIR (set above), not the raw FIDESLIB_HAZE_LIB/DIR cache
    # variables: the raw variables are empty in the default vendored-submodule case
    # (no override given), which left a static build with the default vendored haze
    # unable to find it at find_package() time even though it built and installed
    # correctly. _HAZE_SO/_HAZE_DIR already resolve that default.
    set(_FIDESLIB_CFG_HAZE_LIB "${_HAZE_SO}")
    set(_FIDESLIB_CFG_HAZE_DIR "${_HAZE_DIR}")
endif ()

# PATH_VARS turns each of these into a PACKAGE_<name> variable resolved relative to
# the package at load time, which is what makes the installed config relocatable.
configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${INSTALL_DATA_DIR}"
        PATH_VARS INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_BINARY_DIR INSTALL_DATA_DIR
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
        # The multi-GPU tests include nccl.h unconditionally, so on a machine
        # without NCCL the whole suite failed to build (single-GPU CI runners).
        # Leave them out when NCCL is absent; the library itself already builds
        # without NCCL support in that case (see the nccl_FOUND check above).
        if (NOT nccl_FOUND)
            list(FILTER TEST_SOURCE EXCLUDE REGEX "/(Microbench|PeerUtilsTests)\\.(cu|cpp)$")
        endif ()
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
                ${TEST_SOURCE_DIR}/ApiKeyAccessTests.cpp
                ${TEST_SOURCE_DIR}/ApiBootstrapToLevelTests.cpp
                ${TEST_SOURCE_DIR}/ApiPreparedLinearTransformTests.cpp
                ${TEST_SOURCE_DIR}/ApiParityTest.cpp
                ${TEST_SOURCE_DIR}/ReducedNoiseTests.cpp)
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

############### Atomic-op timing harness ###################
# In-tree so it links the `fideslib` target directly. The installed package
# exports a dependency on the imported `haze::haze` target, which a consuming
# project outside this tree cannot resolve without haze's own package config.

option(FIDESLIB_COMPILE_ATOMIC_OPS_ENGINE
       "Build the single-source atomic-op timing harness (examples/atomic-ops-engine)." OFF)

if (FIDESLIB_COMPILE_ATOMIC_OPS_ENGINE)
    add_executable(atomic_ops_engine examples/atomic-ops-engine/src/atomic_ops_engine.cpp)
    set_source_files_properties(examples/atomic-ops-engine/src/atomic_ops_engine.cpp
            PROPERTIES LANGUAGE CXX)
    target_link_libraries(atomic_ops_engine PRIVATE fideslib)
    set_target_properties(atomic_ops_engine PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
    )
endif ()
