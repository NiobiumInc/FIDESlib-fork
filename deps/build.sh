#!/usr/bin/env bash

set -e
set -x

#Remove previous installation.
rm -rf openfhe-install
rm -rf openfhe-src

# Target installation directory.
mkdir -p $1
git submodule update --init --recursive --remote

#Source submodule.
cd openfhe-src
git checkout v1.5.1
#git config user.email "FIDESlib"
#git config user.name "FIDESlib"
git apply ../openfhe-1.5.1.patch

# Compilation and installation.
mkdir build
cd build
echo "Installing into $1"
# OPENFHE_EXTRA_CMAKE_FLAGS: optional extra -D flags. The haze (FHETCH) backend's
# bit-exact parity requires the oracle built with -DWITH_REDUCED_NOISE=ON (haze's
# recorder implements that keyswitch/ModDown rounding; see haze's own Makefile, which
# builds its reference OpenFHE the same way). The default (flag unset) is unchanged
# and matches what the CUDA backend was verified against.
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$1" ${OPENFHE_EXTRA_CMAKE_FLAGS} ..
make -j12
make install -j12