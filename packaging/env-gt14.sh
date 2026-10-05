# shellcheck shell=bash
# Toolchain for relocatable FIDESlib builds.
#
# Red Hat's gcc-toolset ships libstdc++_nonshared.a, which statically absorbs the
# post-GLIBCXX_3.4.29 ABI. Combined with RHEL 9's glibc 2.34 headers (which, unlike
# glibc >= 2.38, do not redirect strtol to __isoc23_strtol in C++23 mode) this yields
# a C++23-capable compiler whose output floors at GLIBC_2.26 / GLIBCXX_3.4.29.
# Measured, not assumed: see packaging/verify-artifact.sh.
GT_ROOT=${GT_ROOT:-/opt/rh/gcc-toolset-14/root}
export CC="$GT_ROOT/usr/bin/gcc"
export CXX="$GT_ROOT/usr/bin/g++"
export PATH="$GT_ROOT/usr/bin:$PATH"
