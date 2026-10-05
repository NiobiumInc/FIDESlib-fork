# Top-level build orchestration for FIDESlib.
#
# Builds the OpenFHE dependency (and, when haze is selected, the vendored
# niobium-haze submodule via FIDESlib's own CMake logic) and then configures
# and builds FIDESlib itself. CPU/OpenFHE support is always built; CUDA and
# haze are opt-in accelerator backends selected via BACKEND.
#
# Run inside the flake's dev shell (`nix develop`) so cmake/gcc/etc. are on
# PATH: `nix develop --command make BACKEND=cpu`, or `nix develop` then
# plain `make`.
#
# Steps are keyed off stamp files, so a re-run only redoes what's stale:
# submodule init, the OpenFHE build, cmake's own configure step, and the
# FIDESlib build itself (which cmake --build/make already does incrementally
# per translation unit) are all skipped when their inputs haven't changed.
#
# Usage:
#   make BACKEND=cpu            # CPU only (default)
#   make BACKEND=cuda           # CPU + CUDA
#   make BACKEND=haze           # CPU + haze
#   make BACKEND=all            # CPU + CUDA + haze
#   make deps                   # build/install OpenFHE only
#   make clean                  # remove the FIDESlib build directory
#   make distclean               # also remove the OpenFHE build/install trees
#
# Common overrides:
#   BUILD_TYPE=Debug|Release|MinSizeRel|RelWithDebInfo (default: Release)
#   JOBS=<n>                                            (default: nproc)
#   BUILD_DIR=<path>            FIDESlib build directory (default: build)
#   OPENFHE_INSTALL_PREFIX=<path>                        (default: deps-install/openfhe)

BACKEND ?= cpu
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

BUILD_DIR ?= build
OPENFHE_BUILD_DIR ?= build-openfhe
OPENFHE_INSTALL_PREFIX ?= deps-install/openfhe

ifeq ($(BACKEND),cpu)
  CMAKE_ENABLE_FLAGS := -DFIDESLIB_ENABLE_CUDA=OFF -DFIDESLIB_ENABLE_HAZE=OFF
else ifeq ($(BACKEND),cuda)
  CMAKE_ENABLE_FLAGS := -DFIDESLIB_ENABLE_CUDA=ON -DFIDESLIB_ENABLE_HAZE=OFF
else ifeq ($(BACKEND),haze)
  CMAKE_ENABLE_FLAGS := -DFIDESLIB_ENABLE_CUDA=OFF -DFIDESLIB_ENABLE_HAZE=ON
else ifeq ($(BACKEND),all)
  CMAKE_ENABLE_FLAGS := -DFIDESLIB_ENABLE_CUDA=ON -DFIDESLIB_ENABLE_HAZE=ON
else
  $(error unknown BACKEND '$(BACKEND)' (expected: cpu, cuda, haze, all))
endif

# Stamp files record completed steps so re-running `make` only redoes what's
# stale, instead of unconditionally re-running every step on every invocation.
SUBMODULE_STAMP := deps/.submodules-$(BACKEND).stamp
OPENFHE_STAMP := $(OPENFHE_INSTALL_PREFIX)/lib/cmake/OpenFHE/OpenFHEConfig.cmake
CONFIGURE_STAMP := $(BUILD_DIR)/.configured-$(BACKEND)-$(BUILD_TYPE).stamp

.PHONY: all deps submodules configure build install clean distclean help

all: build

help:
	@echo "usage: make BACKEND=cpu|cuda|haze|all [BUILD_TYPE=Release] [JOBS=N]"

# Fetch the git submodules needed for the selected backend. openfhe-src is
# always required; niobium-haze only when the haze backend is selected.
submodules: $(SUBMODULE_STAMP)

$(SUBMODULE_STAMP): .gitmodules
	git submodule update --init deps/openfhe-src
ifneq (,$(filter $(BACKEND),haze all))
	git submodule update --init --recursive deps/niobium-haze
endif
	@mkdir -p "$(dir $@)"
	@touch "$@"

# Build and install the patched OpenFHE dependency; skipped once installed.
deps: $(OPENFHE_STAMP)

# build.sh resolves its install-prefix argument from several directories
# deep (deps/openfhe-src/build/), so it must be passed as an absolute path.
$(OPENFHE_STAMP): $(SUBMODULE_STAMP)
	cd deps && ./build.sh "$(abspath $(OPENFHE_INSTALL_PREFIX))"

configure: $(CONFIGURE_STAMP)

$(CONFIGURE_STAMP): $(OPENFHE_STAMP) CMakeLists.txt
	cmake -S . -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DOPENFHE_INSTALL_PREFIX="$(OPENFHE_INSTALL_PREFIX)" \
		$(CMAKE_ENABLE_FLAGS)
	@touch "$@"

# cmake --build / the underlying make or ninja invocation is already
# incremental per translation unit, so this always runs but only recompiles
# what changed.
build: $(CONFIGURE_STAMP)
	cmake --build "$(BUILD_DIR)" -j$(JOBS)

install: build
	cmake --build "$(BUILD_DIR)" --target install -j$(JOBS)

clean:
	rm -rf "$(BUILD_DIR)"

distclean: clean
	rm -rf "$(OPENFHE_BUILD_DIR)" "$(OPENFHE_INSTALL_PREFIX)" deps/.submodules-*.stamp
