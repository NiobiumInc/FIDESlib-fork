{
  description = "FIDESlib: dev shell with the toolchain needed to build OpenFHE, haze and FIDESlib itself";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      # CUDA (via cudaPackages) is Linux-only in nixpkgs; the CUDA backend is
      # simply unavailable on darwin, same as upstream FIDESlib.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems f;
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          # cudaPackages needs the unfree CUDA redistributables.
          config.allowUnfree = true;
        };
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = pkgsFor system;
          isLinux = pkgs.stdenv.isLinux;

          # Required unconditionally: builds/installs OpenFHE and configures
          # FIDESlib's CPU/OpenFHE backend (always built regardless of which
          # accelerator backends are enabled).
          coreTools = with pkgs; [
            cmake
            gnumake
            gcc13 # GCC >= 11, matches CMakeLists.txt's CMAKE_C/CXX_COMPILER=gcc/g++
            git
            pkg-config
          ];

          # OpenMP development library (README requirement); libgomp ships with
          # gcc, but nixpkgs' standalone `openmp` package is what provides the
          # dev headers when a non-gcc toolchain picks it up.
          openmpLib = [ pkgs.llvmPackages.openmp ];

          # Only needed for -DFIDESLIB_ENABLE_CUDA=ON. cudaPackages is not
          # available on darwin, so the CUDA backend can't be built there —
          # matches upstream FIDESlib's own requirement (NVIDIA CUDA 12/13).
          cudaTools = pkgs.lib.optionals isLinux (
            with pkgs.cudaPackages;
            [
              cudatoolkit
              cudnn
            ]
          );

          # Needed to build the vendored deps/niobium-haze submodule for
          # -DFIDESLIB_ENABLE_HAZE=ON (see deps/niobium-haze):
          # a C++23 clang toolchain, Catch2 v3 for its test suite, and
          # clang-tools for clangd/clang-tidy/clang-format.
          hazeTools = with pkgs; [
            clang-tools
            catch2_3
          ];
        in
        {
          default = pkgs.mkShell {
            name = "fideslib-dev";

            packages = coreTools ++ openmpLib ++ cudaTools ++ hazeTools;

            shellHook = ''
              echo "FIDESlib dev shell ready (cmake, gcc, openmp${
                pkgs.lib.optionalString isLinux ", cudatoolkit"
              }, clang-tools, catch2)."
              echo "Build with: make BACKEND=cpu|cuda|haze|all"
              ${pkgs.lib.optionalString isLinux ''
                export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
              ''}
            '';
          };
        }
      );

      formatter = forEachSystem (system: (pkgsFor system).nixfmt);
    };
}
