#!/usr/bin/env bash
#
# Prove a built package is self-contained: unpack it somewhere unrelated to the source
# tree, build the bundled smoke example against it with nothing but a compiler --
# plain -I/-L/-l flags, no cmake, no find_package -- and run it on each backend the
# machine can support. find_package(CONFIG) is CMake-specific; a consumer that does
# not use CMake at all still has to be able to link this package.
#
# This is the same script CI runs on a fresh runner and that you can run locally
# against a tarball, so a local pass and a CI pass mean the same thing.
#
# Usage: test-package.sh <tarball-or-unpacked-dir> [workdir]
#
# Environment:
#   NBCC_FHETCH_REPLAY   enables the FUNC_SIM leg when set to a replay binary
#   FIDESLIB_FPGA_TARGET e.g. <device-target>; enables the FPGA leg (needs hardware)
#   SCRUB=0              keep the ambient environment (default scrubs it with env -i)

set -euo pipefail

TARGET=${1:?usage: test-package.sh <tarball-or-unpacked-dir> [workdir]}
WORK=${2:-$(mktemp -d)}
SCRUB=${SCRUB:-1}

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

if [ -f "$TARGET" ]; then
    tar -xzf "$TARGET" -C "$WORK"
    PKG=$(find "$WORK" -maxdepth 1 -mindepth 1 -type d -name 'fideslib-*' | head -1)
else
    PKG=$(cd "$TARGET" && pwd)
fi
[ -d "$PKG" ] || { echo "test-package: no package at $TARGET" >&2; exit 2; }

SMOKE="$PKG/share/fideslib/examples/smoke"
[ -d "$SMOKE" ] || { echo "test-package: package has no smoke example" >&2; exit 2; }

RUNS="$WORK/haze-runs"
rm -rf "$RUNS"; mkdir -p "$RUNS"

# The package must be usable with nothing from the build machine on the environment.
# env -i is how that gets enforced rather than assumed.
if [ "$SCRUB" = 1 ]; then
    RUN=(env -i "PATH=/usr/bin:/bin" "HOME=$WORK")
else
    RUN=(env)
fi

echo "== package: $PKG"
echo "== building the smoke consumer against it (plain -I/-L/-l, no cmake)"
# This mirrors fideslib's own target_include_directories(fideslib PUBLIC ...)
# (CMakeLists.txt) exactly: include/fideslib for the API headers, include itself so
# <haze/haze.h> resolves, and every OpenFHE root its headers cross-reference by a
# path relative to core/ or pke/ rather than by a full openfhe/-prefixed path.
# third-party/include only exists when the package was built with NTL/tcmalloc.
INCLUDES=(-I"$PKG/include/fideslib" -I"$PKG/include")
for sub in "" /core /pke /binfhe /third-party/include; do
    [ -d "$PKG/include/openfhe$sub" ] && INCLUDES+=(-I"$PKG/include/openfhe$sub")
done
mkdir -p "$WORK/build"
BIN="$WORK/build/fideslib-smoke"
"${RUN[@]}" "${CXX:-c++}" -std=c++20 -O2 "${INCLUDES[@]}" "$SMOKE/src/smoke.cpp" \
    -o "$BIN" -L"$PKG/lib" -lfideslib -Wl,-rpath,"$PKG/lib" \
    >"$WORK/build.log" 2>&1 || {
        echo "build failed:" >&2; tail -30 "$WORK/build.log" >&2; exit 1; }
fail=0

leg() {
    local name=$1; shift
    local out rc
    set +e
    out=$("${RUN[@]}" "FIDESLIB_HAZE_RUNS_DIR=$RUNS" "$@" "$BIN" 2>&1)
    rc=$?
    set -e
    printf '%s\n' "$out" > "$WORK/$name.log"
    if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q '^PASS'; then
        printf '  %-10s PASS  %s\n' "$name" "$(printf '%s' "$out" | grep '^backend=' || true)"
    else
        printf '  %-10s FAIL (rc=%s, see %s)\n' "$name" "$rc" "$WORK/$name.log"
        fail=1
    fi
}

echo "== running"
leg cpu       FIDESLIB_BACKEND=cpu
# The local haze target replays in-process, so it needs no external replay binary --
# which is what makes the accelerator backend testable on a stock CI runner.
leg haze      FIDESLIB_BACKEND=haze

if [ -n "${NBCC_FHETCH_REPLAY:-}" ]; then
    leg func-sim FIDESLIB_BACKEND=haze FIDESLIB_HAZE_TARGET=FUNC_SIM \
                 "NBCC_FHETCH_REPLAY=$NBCC_FHETCH_REPLAY"
else
    echo "  func-sim   skip (set NBCC_FHETCH_REPLAY to enable)"
fi

if [ -n "${FIDESLIB_FPGA_TARGET:-}" ] && [ -n "${NBCC_FHETCH_REPLAY:-}" ]; then
    leg fpga FIDESLIB_BACKEND=haze "FIDESLIB_HAZE_TARGET=$FIDESLIB_FPGA_TARGET" \
             "NBCC_FHETCH_REPLAY=$NBCC_FHETCH_REPLAY"
else
    echo "  fpga       skip (set FIDESLIB_FPGA_TARGET + NBCC_FHETCH_REPLAY, needs hardware)"
fi

echo
if [ "$fail" -eq 0 ]; then echo "PACKAGE TEST PASSED"; else echo "PACKAGE TEST FAILED" >&2; fi
exit "$fail"
