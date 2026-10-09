#!/usr/bin/env bash
#
# Build a redistributable FIDESlib package: shared library, headers, CMake package
# and the smoke consumer, in a layout that works wherever it is unpacked.
#
# One toolchain builds all four components (FIDESlib's OpenFHE, haze's instrumented
# OpenFHE, haze, FIDESlib). Mixing toolchains is what produced the ABI workarounds
# this replaces. On Linux that toolchain is Red Hat's gcc-toolset: it pairs a C++23
# compiler with libstdc++_nonshared.a, so the output needs only the base system's
# libstdc++, and its glibc headers do not rewrite strtol into __isoc23_strtol the way
# glibc >= 2.38 does in C++23 mode.
#
# Usage: build-release.sh [version]
#
# Environment:
#   GT_ROOT       gcc-toolset root      (default /opt/rh/gcc-toolset-14/root)
#   BUILD_ROOT    scratch build trees   (default /opt/$USER/fideslib-release/build)
#   OUT_DIR       where the tarball goes (default <repo>/packaging/dist)
#   JOBS          parallelism           (default nproc)
#   WITH_CUDA=1   also build the CUDA backend (Linux x86_64 only; see below)
#   CUDA_HOME     CUDA toolkit for WITH_CUDA (default /usr/local/cuda)
#   CUDA_ARCH     CUDA architectures         (default 120-real)
#   MAX_GLIBC     verify-artifact.sh's glibc floor    (default 2.28, the manylinux_2_28
#                 release floor; override for a local build on a newer host that isn't
#                 meant to match the release floor)
#   MAX_GLIBCXX   verify-artifact.sh's libstdc++ floor (default 3.4.29)
#   TEST_BUILD=1  build fideslib-cpu-test (CPU and haze backends, static) against the same
#                 three dependency builds and stop: no package. The haze CI job uses it,
#                 so the tests run on what the release ships.
#   DEPS_CACHED=1 skip steps 1-3 and reuse $BUILD_ROOT/{openfhe-install,haze-openfhe-install,haze}
#                 from an earlier run (CI restores them from its cache)

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:-$(date -u +%Y.%m.%d)}
JOBS=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
BUILD_ROOT=${BUILD_ROOT:-/opt/${USER}/fideslib-release/build}
OUT_DIR=${OUT_DIR:-$REPO/packaging/dist}
MAX_GLIBC=${MAX_GLIBC:-2.28}
MAX_GLIBCXX=${MAX_GLIBCXX:-3.4.29}

# ---- platform / arch, spelled as the published artifacts name them ---------------
case "$(uname -s)" in
    Linux)  PLATFORM=linux ;;
    Darwin) PLATFORM=macos ;;
    *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64|amd64)  ARCH=x86-64 ;;
    aarch64|arm64) ARCH=aarch64 ;;
    *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac
TRIPLE="$PLATFORM-$ARCH"

# The CUDA backend is a separate artifact, not a superset: it needs a CUDA toolkit to
# build and the NVIDIA driver at run time, so shipping it as the default would make
# every consumer carry that requirement. libcudart is absorbed statically; only
# libcuda.so.1 -- which ships with the kernel driver and is version-locked to it -- is
# resolved from the system, where any machine with a CUDA card already has it.
WITH_CUDA=${WITH_CUDA:-0}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUDA_ARCH=${CUDA_ARCH:-120-real}
if [ "$WITH_CUDA" = 1 ]; then
    [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc at CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
    PKGNAME="fideslib-cuda-$TRIPLE-${VERSION//./-}"
else
    PKGNAME="fideslib-$TRIPLE-${VERSION//./-}"
fi

# ---- toolchain ------------------------------------------------------------------
if [ "$PLATFORM" = linux ]; then
    GT_ROOT=${GT_ROOT:-/opt/rh/gcc-toolset-14/root}
    [ -x "$GT_ROOT/usr/bin/g++" ] || { echo "no gcc-toolset at $GT_ROOT" >&2; exit 1; }
    export CC="$GT_ROOT/usr/bin/gcc" CXX="$GT_ROOT/usr/bin/g++"
    export PATH="$GT_ROOT/usr/bin:$PATH"
fi

STAGE="$BUILD_ROOT/stage"
OFHE_SRC="$REPO/deps/openfhe-src"
OFHE_INS="$BUILD_ROOT/openfhe-install"
HAZE_SRC="$REPO/deps/niobium-haze"
HAZE_OFHE_SRC="$HAZE_SRC/vendor/niobium-fhetch/vendor/openfhe"
HAZE_OFHE_INS="$BUILD_ROOT/haze-openfhe-install"
HAZE_BUILD="$BUILD_ROOT/haze"

# Rewrite absolute source paths baked into __FILE__/assert strings, so the shipped
# binaries do not describe the machine that built them. Also makes builds from
# different checkouts byte-comparable.
PREFIX_MAP="-ffile-prefix-map=$REPO=/fideslib"
PREFIX_MAP="$PREFIX_MAP -ffile-prefix-map=$OFHE_INS=/openfhe"
PREFIX_MAP="$PREFIX_MAP -ffile-prefix-map=$HAZE_OFHE_INS=/openfhe-haze"
PREFIX_MAP="$PREFIX_MAP -ffile-prefix-map=$BUILD_ROOT=/build"

# haze is C++23 with -Werror and is CI-tested against clang. gcc rejects clang's
# [[clang::suppress]] under -Werror=attributes; whitelisting that attribute namespace
# is narrower than -Wno-error and changes no haze source (it must stay at its pinned rev).
# Clang itself does not recognise this scoped -Wno-attributes= syntax at all, so it
# only applies on the gcc-toolset leg, not the clang leg (macOS).
# Same reasoning applies to -ldl: OPENFHEcore_static calls dlopen/dlsym for its
# pluggable PRNG engine, and glibc < 2.34 (manylinux_2_28) has not merged libdl into
# libc, so libhaze.so fails to link without it. haze's CMakeLists.txt is a pinned
# vendored file and does not add this itself.
if [ "$PLATFORM" = linux ]; then
    HAZE_FLAGS="-Wno-attributes=clang:: $PREFIX_MAP"
    HAZE_LINKER_FLAGS="-ldl"
else
    HAZE_FLAGS="$PREFIX_MAP"
    # haze's std::println needs libc++'s <print> support, which only Homebrew llvm@19's
    # own libc++.dylib implements (__is_posix_terminal); the system libc++ the linker
    # finds by default does not, and CMake links libhaze.dylib against that one unless
    # told otherwise. Point it at the same libc++ that $CXX (llvm@19 clang++) compiled
    # against instead.
    LLVM_ROOT=$(cd "$(dirname "$CXX")/.." && pwd)
    HAZE_LINKER_FLAGS="-L$LLVM_ROOT/lib/c++ -Wl,-rpath,$LLVM_ROOT/lib/c++"
fi

# -DWITH_NATIVEOPT=OFF is mandatory: OpenFHE defaults it ON, which bakes -march=native
# in and pins the artifact to this machine's CPU.
OFHE_COMMON="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED=OFF -DBUILD_STATIC=ON
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_NATIVEOPT=OFF
             -DWITH_REDUCED_NOISE=ON -DBUILD_EXAMPLES=OFF -DBUILD_UNITTESTS=OFF
             -DBUILD_BENCHMARKS=OFF -DBUILD_EXTRAS=OFF"

echo "==> FIDESlib release $VERSION ($TRIPLE), $(${CXX:-c++} --version | head -1)"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

if [ "${DEPS_CACHED:-0}" = 1 ]; then
    echo "==> [1-3/5] dependencies from cache"
    [ -d "$OFHE_INS" ] && [ -d "$HAZE_OFHE_INS" ] && [ -d "$HAZE_BUILD" ] \
        || { echo "DEPS_CACHED=1 but a cached build tree is missing under $BUILD_ROOT" >&2; exit 1; }
else
    # ---- 1. OpenFHE oracle (patched; WITH_REDUCED_NOISE is required for haze parity) --
    # deps/build.sh (the normal dev-build path) applies deps/fideslib-ref-1.5.1.6.patch
    # before building deps/openfhe-src. FIDESlib's api/*.cpp and CUDA sources reach into
    # OpenFHE internals (m_FHE, m_bootPrecomMap, K_SPARSE, ...) that are protected/private
    # upstream and call an EvalBootstrapSetup overload the patch adds; without it, this
    # step compiles but step 4 fails.
    echo "==> [1/5] OpenFHE"
    git -C "$OFHE_SRC" checkout -- .
    git -C "$OFHE_SRC" apply "$REPO/deps/fideslib-ref-1.5.1.6.patch"
    cmake -S "$OFHE_SRC" -B "$BUILD_ROOT/openfhe" $OFHE_COMMON \
          -DCMAKE_INSTALL_PREFIX="$OFHE_INS" -DWITH_OPENMP=ON \
          -DCMAKE_CXX_FLAGS="$PREFIX_MAP" >/dev/null
    cmake --build "$BUILD_ROOT/openfhe" -j"$JOBS" --target install >/dev/null

    # ---- 2. haze's instrumented OpenFHE ---------------------------------------------
    # A different OpenFHE from step 1: CPROBES-instrumented, absorbed whole into libhaze
    # and hidden behind its version script. OpenMP off, matching haze's own build.
    echo "==> [2/5] instrumented OpenFHE (haze)"
    cmake -S "$HAZE_OFHE_SRC" -B "$BUILD_ROOT/haze-openfhe" $OFHE_COMMON \
          -DCMAKE_INSTALL_PREFIX="$HAZE_OFHE_INS" -DWITH_CPROBES=ON -DWITH_OPENMP=OFF \
          -DCMAKE_CXX_FLAGS="$PREFIX_MAP" >/dev/null
    cmake --build "$BUILD_ROOT/haze-openfhe" -j"$JOBS" --target install >/dev/null

    # ---- 3. libhaze ------------------------------------------------------------------
    echo "==> [3/5] libhaze"
    cmake -S "$HAZE_SRC" -B "$HAZE_BUILD" -DCMAKE_BUILD_TYPE=Release \
          -DOPENFHE_INSTALL_DIR="$HAZE_OFHE_INS" \
          -DHAZE_BUILD_TESTS=OFF -DHAZE_BUILD_E2E_TESTS=OFF \
          -DCMAKE_CXX_FLAGS="$HAZE_FLAGS" \
          -DCMAKE_SHARED_LINKER_FLAGS="$HAZE_LINKER_FLAGS" >/dev/null
    cmake --build "$HAZE_BUILD" -j"$JOBS" --target haze >/dev/null
fi

HAZE_LIB=$(find "$HAZE_BUILD" -maxdepth 1 -name 'libhaze.*' -type f | head -1)
[ -n "$HAZE_LIB" ] || { echo "libhaze not produced" >&2; exit 1; }

if [ "${TEST_BUILD:-0}" = 1 ]; then
    echo "==> test build"
    cmake -S "$REPO" -B "$BUILD_ROOT/fideslib-test" -DCMAKE_BUILD_TYPE=Release \
          -DFIDESLIB_ENABLE_CUDA=OFF -DFIDESLIB_ENABLE_HAZE=ON \
          -DFIDESLIB_COMPILE_TESTS=ON -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
          -DFIDESLIB_HAZE_DIR="$HAZE_SRC" -DFIDESLIB_HAZE_LIB="$HAZE_LIB" \
          -DFIDESLIB_INSTALL_OPENFHE=OFF -DOPENFHE_INSTALL_PREFIX="$OFHE_INS" \
          -DCMAKE_CXX_FLAGS="$PREFIX_MAP" >/dev/null
    cmake --build "$BUILD_ROOT/fideslib-test" -j"$JOBS" --target fideslib-cpu-test
    echo "==> test binary: $BUILD_ROOT/fideslib-test/fideslib-cpu-test"
    exit 0
fi

# ---- 4. FIDESlib -----------------------------------------------------------------
echo "==> [4/5] libfideslib"
rm -rf "$STAGE"
# -DCMAKE_INSTALL_LIBDIR=lib forces GNUInstallDirs to skip its default lib64 for a
# 64-bit build. manylinux_2_28 is RHEL-family, where CMake also sets
# FIND_LIBRARY_USE_LIB64_PATHS -- but Debian/Ubuntu consumers do not set that
# property, so their own find_package(fideslib CONFIG) skips the lib64/cmake/
# glob entirely and never finds the exported package, no matter what the tarball
# actually contains. A fixed "lib" is found on every consumer regardless of host.
if [ "$WITH_CUDA" = 1 ]; then
    # The prefix map has to reach nvcc too, or the .cu translation units keep the build
    # host's absolute paths in their __FILE__ strings. FIDESlib's CUDA flags already carry
    # -forward-unknown-to-host-compiler, so the host flags pass through as written.
    CUDA_ARGS=(-DFIDESLIB_ENABLE_CUDA=ON -DCUDA_PATH="$CUDA_HOME"
               -DFIDESLIB_ARCH="$CUDA_ARCH" -Dnccl_FOUND=OFF
               -DCMAKE_CUDA_FLAGS="$PREFIX_MAP")
else
    CUDA_ARGS=(-DFIDESLIB_ENABLE_CUDA=OFF)
fi
cmake -S "$REPO" -B "$BUILD_ROOT/fideslib" -DCMAKE_BUILD_TYPE=Release \
      -DFIDESLIB_BUILD_SHARED=ON "${CUDA_ARGS[@]}" -DFIDESLIB_ENABLE_HAZE=ON \
      -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
      -DFIDESLIB_HAZE_DIR="$HAZE_SRC" -DFIDESLIB_HAZE_LIB="$HAZE_LIB" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DOPENFHE_INSTALL_PREFIX="$OFHE_INS" -DFIDESLIB_INSTALL_PREFIX="$STAGE" \
      -DCMAKE_CXX_FLAGS="$PREFIX_MAP" >/dev/null
cmake --build "$BUILD_ROOT/fideslib" -j"$JOBS" >/dev/null
cmake --install "$BUILD_ROOT/fideslib" >/dev/null

# ---- 5. bundle, normalise, package ------------------------------------------------
echo "==> [5/5] package"
LIBDIR=$(find "$STAGE" -maxdepth 1 -type d -name 'lib*' | head -1)
[ -n "$LIBDIR" ] || { echo "no lib directory found under $STAGE" >&2; exit 1; }

# libgomp is a genuine runtime dependency (OpenFHE is built WITH_OPENMP=ON) and is not
# present on a machine without a GCC install, so it travels with the package. It is a
# leaf C library, so a second copy in the process is harmless -- unlike libstdc++,
# which must stay shared because this package's public API is C++.
if [ "$PLATFORM" = linux ]; then
    GOMP=$("${CXX}" -print-file-name=libgomp.so.1)
    if [ -f "$GOMP" ]; then cp -L "$GOMP" "$LIBDIR/"; else echo "warn: libgomp.so.1 not found" >&2; fi
else
    OMP="$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"
    if [ -f "$OMP" ]; then cp -L "$OMP" "$LIBDIR/"; else echo "warn: libomp.dylib not found" >&2; fi
    # haze's std::println needs Homebrew llvm@19's own libc++ (see step 3), which in
    # turn needs libc++abi and libunwind (checked via otool -L against a local
    # install); none of the three are on a machine without this exact keg.
    for f in c++/libc++.1.dylib c++/libc++abi.1.dylib unwind/libunwind.1.dylib; do
        src="$LLVM_ROOT/lib/$f"
        if [ -f "$src" ]; then cp -L "$src" "$LIBDIR/"; else echo "warn: $f not found" >&2; fi
    done
fi

# Resolve siblings relative to the library itself, never by absolute path. $STAGE was
# rm -rf'd and freshly reinstalled just above, so every file here is a first-time copy
# within this run -- there is no legitimate "already applied" case for any of the
# following, and a failure here means the shipped library will not load. All of it
# fails the build rather than warning, on purpose.
if [ "$PLATFORM" = linux ]; then
    command -v patchelf >/dev/null || { echo "patchelf not found; cannot set \$ORIGIN rpath" >&2; exit 1; }
    for so in "$LIBDIR"/*.so*; do
        [ -f "$so" ] && [ ! -L "$so" ] || continue
        patchelf --set-rpath '$ORIGIN' "$so" || { echo "patchelf failed on $so" >&2; exit 1; }
    done
else
    for dy in "$LIBDIR"/*.dylib; do
        [ -f "$dy" ] || continue
        install_name_tool -id "@rpath/$(basename "$dy")" "$dy" \
            || { echo "install_name_tool -id failed on $dy" >&2; exit 1; }
        # The bundled libc++*/libunwind already carry @loader_path from their own
        # Homebrew build (checked with otool -l), so -add_rpath on those specific
        # files fails with "already exists" every time -- a real, expected outcome,
        # not a hedge to swallow. Check first instead of blindly retrying or ignoring
        # the result either way.
        existing_rpaths=$(otool -l "$dy" | awk '/cmd LC_RPATH/{getline; getline; print $2}')
        if ! printf '%s\n' "$existing_rpaths" | grep -qx '@loader_path'; then
            install_name_tool -add_rpath "@loader_path" "$dy" \
                || { echo "install_name_tool -add_rpath failed on $dy" >&2; exit 1; }
        fi
        # Mach-O load commands are resolved at link time, unlike ELF's soname+rpath
        # indirection, so linking against Homebrew's libomp/libc++/libc++abi/libunwind
        # bakes their absolute Cellar/opt paths into libfideslib/libhaze -- paths that
        # only exist on this build host. Repoint any dependency this package already
        # bundles (as a sibling *.dylib in the same dir) at that sibling instead. The
        # bundled libc++*/libunwind already reference each other via @rpath (checked
        # locally with otool -l: they carry an @loader_path LC_RPATH of their own), so
        # this only ever rewrites something for libfideslib/libhaze themselves.
        for sib in "$LIBDIR"/*.dylib; do
            [ -f "$sib" ] || continue
            sib_name=$(basename "$sib")
            [ "$sib_name" = "$(basename "$dy")" ] && continue
            # A literal suffix match, not a regex match: library names like
            # libc++.1.dylib contain '+' and '.', which are regex metacharacters with
            # no literal fallback for '+' -- an ERE match on the unescaped name silently
            # never matches, which is exactly how this missed libhaze's libc++ dep.
            old=$(otool -L "$dy" | awk -v suf="/$sib_name" '
                { path = $1 } { n = length(path); s = length(suf) }
                n >= s && substr(path, n - s + 1) == suf { print path; exit }')
            if [ -n "$old" ] && [ "$old" != "@rpath/$sib_name" ]; then
                install_name_tool -change "$old" "@rpath/$sib_name" "$dy" \
                    || { echo "install_name_tool -change failed on $dy" >&2; exit 1; }
            fi
        done
        # install_name_tool invalidates the signature, and arm64 refuses to load an
        # unsigned Mach-O. Re-sign ad hoc; there is no benign reason for this to fail.
        codesign -f -s - "$dy" || { echo "codesign failed on $dy" >&2; exit 1; }
    done
fi

cp -a "$REPO/packaging/smoke" "$STAGE/share/fideslib/examples/smoke" 2>/dev/null || {
    mkdir -p "$STAGE/share/fideslib/examples"
    cp -a "$REPO/packaging/smoke" "$STAGE/share/fideslib/examples/smoke"
}
cp "$REPO/LICENSE.txt" "$STAGE/" 2>/dev/null || true
cp "$REPO/packaging/README.md" "$STAGE/" 2>/dev/null || true
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -a "$STAGE" "$WORK/$PKGNAME"
tar -czf "$OUT_DIR/$PKGNAME.tar.gz" -C "$WORK" "$PKGNAME"

echo "==> wrote $OUT_DIR/$PKGNAME.tar.gz"
echo "==> verifying"
BUILD_HOST_PATHS="$REPO $BUILD_ROOT" \
    "$REPO/packaging/verify-artifact.sh" "$OUT_DIR/$PKGNAME.tar.gz" "$MAX_GLIBC" "$MAX_GLIBCXX"
