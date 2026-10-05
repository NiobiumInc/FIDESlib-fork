# Redistributable FIDESlib packages

A per-platform tarball containing `libfideslib`, `libhaze`, the headers the public API
exposes, and a CMake package. It builds and runs a consumer on a machine with no
package manager, no nix, and no OpenFHE install.

Backends: CPU (OpenFHE) and HAZE (FHETCH record/replay). CUDA is not included — it
would require the consumer to carry a matching CUDA toolkit, and the raw
`FIDESlib::CKKS` GPU headers expose OpenFHE types that cannot be hidden.

## Layout

```
fideslib-<platform>-<arch>-<version>/
  include/fideslib/     public API headers
  include/openfhe/      core, pke, binfhe, cereal
  include/haze/         haze C ABI + replay bridge
  lib/                  libfideslib.so.1.0, libhaze.so, libgomp.so.1
  lib/cmake/fideslib/   the CMake package
  share/fideslib/examples/smoke/
  VERSION  LICENSE.txt  README.md
```

## Usage

No build system is required. Link with plain compiler flags:

```sh
tar -xzf fideslib-linux-x86-64-<version>.tar.gz
PKG=$PWD/fideslib-linux-x86-64-<version>
c++ -std=c++20 -I"$PKG/include/fideslib" -I"$PKG/include" \
    -I"$PKG/include/openfhe" -I"$PKG/include/openfhe/core" \
    -I"$PKG/include/openfhe/pke" -I"$PKG/include/openfhe/binfhe" \
    your-project.cpp -o your-project \
    -L"$PKG/lib" -lfideslib -Wl,-rpath,"$PKG/lib"
```

This is exactly what `packaging/test-package.sh` does to the bundled smoke example in
CI — no `find_package`, no CMake required at all, since not every consumer uses it.
CMake users can instead do:

```sh
cmake -S your-project -B build -DCMAKE_PREFIX_PATH=$PWD/fideslib-linux-x86-64-<version>
```

and `find_package(fideslib REQUIRED CONFIG)` + link `fideslib::fideslib` — see
`share/fideslib/examples/smoke/CMakeLists.txt` for a worked example, kept as a
convenience but not what CI verifies.

Select the backend at run time with `FIDESLIB_BACKEND=cpu|haze`. On haze,
`FIDESLIB_HAZE_TARGET` chooses the replay target: unset runs the in-process
simulator, `FUNC_SIM` and hardware targets dispatch to `nbcc_fhetch_replay`, which must
be on `PATH` or named by `NBCC_FHETCH_REPLAY`. Set `FIDESLIB_HAZE_RUNS_DIR` to a
writable path — the default is under `/tmp` and is commonly not writable.

## What "self-contained" means here

The package carries everything specific to FIDESlib. It still uses the platform's own
C library and C++ runtime, because it must: `libfideslib`'s public API exposes
`std::vector`, `std::shared_ptr` and `lbcrypto::` types (`api/Ciphertext.hpp:57`
returns `const std::vector<lbcrypto::DCRTPoly>&`), so the consumer and the library
have to agree on one C++ runtime. A privately statically-linked `libstdc++` would put
two runtimes on either side of that boundary — mismatched allocation ownership,
`dynamic_cast` and `catch` failing on RTTI identity, two unwinder state machines.

`libhaze` is the opposite case and is treated accordingly: its ABI is pure C, 79
exported symbols all named `haze*`, with its own instrumented OpenFHE absorbed and
hidden behind a linker version script. It stays a *separate* shared object for that
reason — `libfideslib` exports OpenFHE's symbols at default visibility, and merging
the two would put two builds of OpenFHE 1.5.1 in one namespace.

`libgomp.so.1` is bundled because OpenFHE is built `WITH_OPENMP=ON` and a machine
without GCC installed has no libgomp. It is a leaf C library, so a second copy in the
process is harmless.

### ABI contract

| Property | Value |
|---|---|
| C++ standard library | libstdc++, `_GLIBCXX_USE_CXX11_ABI=1` |
| glibc floor | 2.28 (release builds) / the build host's, for local builds |
| libstdc++ floor | `GLIBCXX_3.4.29` |
| CPU baseline | generic; `-march=native` is explicitly disabled |
| OpenFHE | patched `fideslib-ref-v1.5.1.6`, `WITH_REDUCED_NOISE=ON` |

The OpenFHE inside this package is **not** stock: it carries
`deps/fideslib-ref-1.5.1.6.patch` and is built `WITH_REDUCED_NOISE=ON`, which the haze
backend needs for bit-exact parity. A distro OpenFHE cannot be substituted.

`packaging/verify-artifact.sh` checks each of these against the built artifact rather
than trusting the build flags.

## Toolchain, and why

One compiler builds all four components (FIDESlib's OpenFHE, haze's instrumented
OpenFHE, haze, FIDESlib). Earlier builds used three — nix gcc-15, nix clang-21 and a
system gcc 11.5 — which forced several ABI workarounds.

On Linux that compiler is Red Hat's **gcc-toolset-14**, which is what `manylinux_2_28`
ships. It is the specific pairing this needs:

- C++23, which haze requires (`std::println` in `src/common/errors.cpp`).
- `libstdc++_nonshared.a`, which statically absorbs the post-`GLIBCXX_3.4.29` ABI, so
  the output runs against an old system `libstdc++.so.6`.
- glibc 2.28/2.34 headers, which — unlike glibc ≥ 2.38 — do not rewrite `strtol` into
  `__isoc23_strtol` in C++23 mode. That rewrite is what pinned earlier nix-built
  artifacts to `GLIBC_2.38`, above the glibc 2.34 of a RHEL 9-class host.

clang is not used on Linux for a mundane reason: OpenFHE's `CMakeLists.txt:28` does
`set(CMAKE_C_COMPILER "gcc")`, ignoring `CC`/`CXX` and `-DCMAKE_CXX_COMPILER`.
Overriding it would mean patching OpenFHE in two places, one of which is inside haze's
pinned submodule, which must be consumed unmodified. On macOS that same hardcoded name
resolves to Apple clang, so the platform gets the right compiler anyway.

### Why nix does not build the Linux artifacts

nix owns the dev shell, dependency pinning, and consuming the published tarball
(`emit-nix-expr.sh` produces a `fetchurl` + `autoPatchelfHook` expression). It does not
produce them, for a measured reason: nixpkgs pins one glibc per revision, and no
revision pairs an old glibc with a C++23 compiler.

| nixpkgs | glibc | default gcc |
|---|---|---|
| 20.09 | 2.31 | 9.3.0 |
| 22.05 | **2.34** | 11.3.0 |
| 23.05 | 2.37 | 12.2.0 |
| 24.05 | 2.39 | 13.2.0 (gcc14 available) |

glibc crossed 2.34 two years before gcc crossed 14, and the two are coupled through a
single bootstrap stdenv. `pkgsStatic` does not help: it is musl, and under it
`hostPlatform.hasSharedLibraries` is `false` — it cannot emit a `.so` at all.
`zig cc -target x86_64-linux-gnu.2.28` does hit the floor, but `zig c++` uses libc++,
which is ABI-incompatible with the libstdc++ our consumers compile against.

## Building

```sh
# Locally (floors at the host's glibc):
JOBS=32 packaging/build-release.sh

# As CI does, at the 2.28 floor:
docker run --rm -v "$PWD:/src" -w /src quay.io/pypa/manylinux_2_28_x86_64 \
  packaging/build-release.sh
```

## The CUDA variant

`WITH_CUDA=1 packaging/build-release.sh` produces a second, separate artifact,
`fideslib-cuda-<platform>-<arch>-<version>.tar.gz`. It is not a superset: it requires
the NVIDIA driver at run time, so the plain package stays usable on machines with no
GPU.

It absorbs `libcudart` statically and resolves only `libcuda.so.1` from the host. That
is NVIDIA's own pattern rather than an invention: `nvcc` links the static runtime by
default, the CUDA EULA's Attachment A lists `libcudart_static.a` as redistributable and
omits the driver, and NVIDIA's own `libcublas`/`libcufft`/`libcurand` each export zero
`cudaXxx` symbols and carry no `DT_NEEDED libcudart`. `verify-artifact.sh` asserts the
same two properties of this package.

| Property | Value |
|---|---|
| Driver floor | r580 (the CUDA 13.x minimum) |
| CUDA toolkit on the target | not required |
| Device code | real cubins for `FIDESLIB_ARCH`, not PTX-only |

Two consequences worth knowing:

- A consumer that calls the CUDA runtime itself must link `CUDA::cudart` — this package
  exports none of those symbols, exactly as `libcublas` does not. That puts two runtime
  instances in the process, which is safe as long as no CUDA runtime *symbol* (a
  `__global__` function, a `__device__`/`__constant__` variable) crosses the library
  boundary. FIDESlib's public API exposes none. If that ever changes, this has to move
  to `--cudart=shared` on both sides.
- `DT_RUNPATH` is not transitive, so `libcuda.so.1` must be findable by the system
  loader. On any machine with the NVIDIA driver installed it is, via ldconfig; inside a
  hermetic environment (a nix shell, say) point `LD_LIBRARY_PATH` at a directory holding
  just that one library.

## Testing a package

`packaging/test-package.sh` unpacks a tarball somewhere unrelated to the source tree,
builds the bundled smoke example against it with nothing but a compiler (plain
`-I`/`-L`/`-l` flags, no cmake), and runs each backend. CI runs this exact script, so a
local pass and a CI pass mean the same thing.

```sh
# What a GitHub runner does: CPU and the in-process haze simulator.
packaging/test-package.sh packaging/dist/fideslib-linux-x86-64-<version>.tar.gz

# On a machine with the compiler and an FPGA, the replay legs switch on too.
NBCC_FHETCH_REPLAY=/path/to/nbcc_fhetch_replay FIDESLIB_FPGA_TARGET=<device-target> packaging/test-package.sh packaging/dist/fideslib-linux-x86-64-<version>.tar.gz
```

The replay legs are opt-in because they need a `nbcc_fhetch_replay` binary from the
proprietary compiler, and the FPGA leg needs the accelerator. They skip cleanly when absent, so
the script is green on a stock runner without pretending it tested hardware. By
default the run is scrubbed with `env -i`; pass `SCRUB=0` to keep your environment.

## CI

`.github/workflows/release.yml` builds `linux-x86-64`, `linux-aarch64` and
`macos-aarch64`, then runs `test-package.sh` against each tarball on runners that
never checked out the source — the actual proof of self-containment. The consume
matrix deliberately includes AlmaLinux 8 (glibc 2.28) and Ubuntu 22.04 to confirm the
floor holds *below* the image that built the artifact.

GitHub-hosted runners have no accelerator, so the FPGA leg is not covered there; run
`test-package.sh` with `FIDESLIB_FPGA_TARGET` on an FPGA host for that.

`NiobiumInc/niobium-haze` and its nested `NiobiumInc/niobium-fhetch` submodule are
public repositories, so the default `GITHUB_TOKEN` clones them without any extra
setup.
