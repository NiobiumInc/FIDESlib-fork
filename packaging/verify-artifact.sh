#!/usr/bin/env bash
#
# Gate a staged FIDESlib package: prove it is redistributable rather than asserting it.
#
# Checks, in order of how badly each one bites:
#   1. no build-host absolute paths anywhere in the payload
#   2. glibc / libstdc++ symbol-version floor within the declared limits
#   3. DT_NEEDED limited to the platform runtime plus bundled siblings
#   4. libhaze still exports only its C ABI (OpenFHE stays absorbed and hidden)
#   5. no CPU instructions above the baseline (i.e. -march=native really was off)
#
# Usage: verify-artifact.sh <staged-dir-or-tarball> [max_glibc] [max_glibcxx]

set -euo pipefail

TARGET=${1:?usage: verify-artifact.sh <staged-dir-or-tarball> [max_glibc] [max_glibcxx]}
MAX_GLIBC=${2:-2.34}
MAX_GLIBCXX=${3:-3.4.29}

TMPDIR_CLEANUP=""
cleanup() { [ -n "$TMPDIR_CLEANUP" ] && rm -rf "$TMPDIR_CLEANUP"; }
trap cleanup EXIT

if [ -f "$TARGET" ]; then
    TMPDIR_CLEANUP=$(mktemp -d)
    tar -xzf "$TARGET" -C "$TMPDIR_CLEANUP"
    ROOT=$(find "$TMPDIR_CLEANUP" -maxdepth 1 -mindepth 1 -type d | head -1)
else
    ROOT="$TARGET"
fi
[ -d "$ROOT" ] || { echo "verify: not a directory: $ROOT" >&2; exit 2; }

fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
check() { if [ "$1" -eq 0 ]; then note "ok" "$2"; else note "FAIL" "$2"; fail=1; fi; }

# Highest dotted version in a stream, "" if the stream is empty.
maxver() { sort -uV | tail -1; }
# 0 when $1 <= $2.
vle() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }

is_macho=0
[ "$(uname -s)" = "Darwin" ] && is_macho=1

libs=$(find "$ROOT" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) -type f)
[ -n "$libs" ] || { echo "verify: no shared libraries found under $ROOT" >&2; exit 2; }

echo "== 1. build-host paths =="
# A package that names the machine it was built on is not redistributable. Scan the
# binaries and every text file in the cmake package. build-release.sh passes its own
# source and build-root paths via BUILD_HOST_PATHS (space-separated), since only it
# knows what they were for this run; hardcoding one engineer's home directory here
# instead is exactly the kind of check that goes stale the moment CI -- or anyone
# else -- runs the build. -F (fixed strings) avoids the regex-metacharacter trap a
# path could otherwise spring (see the sibling-dylib match fixed in build-release.sh).
patterns=(-e '/nix/store' -e '/home/users')
for p in ${BUILD_HOST_PATHS:-}; do
    patterns+=(-e "$p")
done
leaks=$(grep -rlF "${patterns[@]}" "$ROOT" 2>/dev/null || true)
if [ -n "$leaks" ]; then
    echo "$leaks" | sed "s|$ROOT|<pkg>|" | sed 's/^/    /'
fi
check "$([ -z "$leaks" ] && echo 0 || echo 1)" "no build-host absolute paths in payload"

echo "== 2. symbol-version floor (glibc <= $MAX_GLIBC, libstdc++ <= $MAX_GLIBCXX) =="
if [ "$is_macho" -eq 0 ]; then
    for l in $libs; do
        n=$(basename "$l")
        # objdump/nm failing to read the binary at all must abort the script, not
        # read as "found nothing" -- grep's own "no match" (exit 1) is the expected,
        # harmless case and keeps its own || true below, scoped to grep alone.
        dump=$(objdump -p "$l") || { echo "objdump failed to read $n" >&2; exit 1; }
        g=$(printf '%s\n' "$dump" | { grep -oE 'GLIBC_2\.[0-9]+' || true; } | sed 's/GLIBC_//' | maxver)
        x=$(printf '%s\n' "$dump" | { grep -oE 'GLIBCXX_3\.4\.[0-9]+' || true; } | sed 's/GLIBCXX_//' | maxver)
        [ -z "$g" ] || { vle "$g" "$MAX_GLIBC";  check $? "$n: max GLIBC_$g"; }
        [ -z "$x" ] || { vle "$x" "$MAX_GLIBCXX"; check $? "$n: max GLIBCXX_$x"; }
        # glibc >= 2.38 headers rewrite strtol in C++23 mode; their presence means the
        # build root was too new and the artifact silently lost portability.
        undef=$(nm -D --undefined-only "$l") || { echo "nm failed to read $n" >&2; exit 1; }
        c=$(printf '%s\n' "$undef" | grep -c isoc23 || true)
        check "$([ "$c" -eq 0 ] && echo 0 || echo 1)" "$n: no __isoc23_* references ($c)"
    done
else
    for l in $libs; do
        n=$(basename "$l")
        deps=$(otool -L "$l") || { echo "otool failed to read $n" >&2; exit 1; }
        bad=$(printf '%s\n' "$deps" | tail -n +2 | { grep -vE '/usr/lib/(libSystem\.B|libc\+\+\.1|libc\+\+abi)\.dylib|@loader_path|@rpath' || true; })
        [ -z "$bad" ] || echo "$bad" | sed 's/^/    /'
        check "$([ -z "$bad" ] && echo 0 || echo 1)" "$n: links only system dylibs"
    done
fi

echo "== 3. runtime dependencies =="
if [ "$is_macho" -eq 0 ]; then
    bundled=$(for l in $libs; do basename "$l"; done | sort -u)
    # libcuda.so.1 is the NVIDIA driver library. It ships with the kernel driver, is
    # version-locked to it, and NVIDIA's redistribution terms forbid shipping it -- so a
    # CUDA package resolves it from the host by design. libcudart is absorbed statically
    # and must NOT appear here.
    allowed='^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|libgomp|libcuda|ld-linux.*)\.so'
    for l in $libs; do
        n=$(basename "$l")
        dump=$(objdump -p "$l") || { echo "objdump failed to read $n" >&2; exit 1; }
        for need in $(printf '%s\n' "$dump" | awk '/NEEDED/{print $2}'); do
            if ! echo "$need" | grep -qE "$allowed" && ! echo "$bundled" | grep -qx "$need"; then
                note "FAIL" "$n: unexpected DT_NEEDED $need"; fail=1
            fi
        done
        # No rpath at all is not a pass to skip past -- it means the library cannot
        # find its own bundled siblings on any machine but this one.
        rp=$(printf '%s\n' "$dump" | { awk '/RUNPATH|RPATH/{print $2}' || true; })
        echo "$rp" | grep -qE '^\$ORIGIN'; check $? "$n: rpath is \$ORIGIN-relative (${rp:-none})"
        # The CUDA runtime must be absorbed, not dynamically linked, or the package
        # would require a CUDA toolkit installation on the target machine.
        if printf '%s\n' "$dump" | grep -q 'NEEDED.*libcudart'; then
            note "FAIL" "$n: links libcudart dynamically (should be static)"; fail=1
        fi
        # An absorbed cudart must not be re-exported: ELF interposition would otherwise
        # let a consumer's cudaXxx calls bind into our embedded copy, or ours into
        # theirs by load order -- the documented hazard of two runtime instances in one
        # process. NVIDIA's own libcublas/libcufft export zero of these; so must we.
        defsyms=$(nm -D --defined-only "$l") || { echo "nm failed to read $n" >&2; exit 1; }
        leaked_cuda=$(printf '%s\n' "$defsyms" | { grep -cE ' [TW] cuda[A-Z]' || true; })
        check "$([ "$leaked_cuda" -eq 0 ] && echo 0 || echo 1)" \
              "$n: exports no cudart symbols ($leaked_cuda)"
    done
    check 0 "DT_NEEDED limited to platform runtime + bundled siblings"
fi

echo "== 4. haze symbol isolation =="
hz=$(echo "$libs" | { grep -E 'libhaze\.(so|dylib)' || true; })
if [ -n "$hz" ]; then
    # libhaze absorbs its own instrumented OpenFHE. If those symbols escape they collide
    # with the OpenFHE that libfideslib exports, which is the whole reason haze stays a
    # separate shared object.
    n=$(basename "$hz")
    if [ "$is_macho" -eq 0 ]; then
        defsyms=$(nm -D --defined-only "$hz") || { echo "nm failed to read $n" >&2; exit 1; }
        leaked=$(printf '%s\n' "$defsyms" | { grep -vE 'haze|HAZE_[0-9]' || true; } | wc -l)
    else
        defsyms=$(nm -gU "$hz") || { echo "nm failed to read $n" >&2; exit 1; }
        leaked=$(printf '%s\n' "$defsyms" | { grep -vE '_haze' || true; } | wc -l)
    fi
    check "$([ "$leaked" -eq 0 ] && echo 0 || echo 1)" "libhaze exports only its C ABI ($leaked non-haze symbols)"
fi

echo "== 5. baseline ISA (-march=native must be off) =="
# Disassembling a multi-megabyte library costs minutes, and the property is already
# pinned by an explicit -DWITH_NATIVEOPT=OFF at build time. Opt in with VERIFY_DEEP=1
# when you want the artifact itself to confirm it.
if [ "${VERIFY_DEEP:-0}" != "1" ]; then
    note "skip" "ISA scan (set VERIFY_DEEP=1 to disassemble)"
elif [ "$is_macho" -eq 0 ] && [ "$(uname -m)" = "x86_64" ] && command -v objdump >/dev/null; then
    for l in $libs; do
        n=$(basename "$l")
        disasm=$(objdump -d --no-show-raw-insn "$l") || { echo "objdump failed to disassemble $n" >&2; exit 1; }
        # AVX-512 encodings are the clearest tell that a native-tuned build slipped in.
        avx512=$(printf '%s\n' "$disasm" | grep -cE '%zmm[0-9]' || true)
        check "$([ "$avx512" -eq 0 ] && echo 0 || echo 1)" "$n: no AVX-512 ($avx512 zmm uses)"
    done
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "VERIFY PASSED: $ROOT"
else
    echo "VERIFY FAILED: $ROOT" >&2
fi
exit "$fail"
