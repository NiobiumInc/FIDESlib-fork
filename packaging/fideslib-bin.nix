# FIDESlib, consumed as a published binary release.
#
# Usage:
#   pkgs.callPackage ./fideslib-bin.nix { }
#
# The pinned block below names one release's tarballs. packaging/emit-nix-expr.sh fills
# it in for each release, and that copy is the fideslib-bin.nix the release publishes;
# the template in the repository leaves it empty.
#
# autoPatchelfHook rewrites the ELF interpreter and RPATHs to point into the nix
# store, which is exactly what it is for: importing a foreign binary into nix. The
# tarball itself is built against an old glibc in a manylinux container so that it
# also works on non-nix machines.
{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  writeText,
}:

let
  # begin pinned release
  version = "2026.11.19-4d2e88a";
  sources = {
    "x86_64-linux" = {
      url = "https://github.com/NiobiumInc/FIDESlib-fork/releases/download/v2026.11.19-4d2e88a/fideslib-linux-x86-64-2026-11-19.tar.gz";
      hash = "sha256-UsPH/IzI6AGj5TLxcOWv/MK4K6l261xIrGFqZCsXY7Q=";
    };
    "aarch64-linux" = {
      url = "https://github.com/NiobiumInc/FIDESlib-fork/releases/download/v2026.11.19-4d2e88a/fideslib-linux-aarch64-2026-11-19.tar.gz";
      hash = "sha256-bvtQnsHhSCxUgQbV1n1WrlhIR/5QpzHQRH84HHd3MgQ=";
    };
    "aarch64-darwin" = {
      url = "https://github.com/NiobiumInc/FIDESlib-fork/releases/download/v2026.11.19-4d2e88a/fideslib-macos-aarch64-2026-11-19.tar.gz";
      hash = "sha256-Pa5g2fmkgMJUwtHM25SZvu0LHJBoaX8YxdWsgUvo8Gg=";
    };
  };
  # end pinned release
  src =
    sources.${stdenv.hostPlatform.system}
      or (throw "fideslib-bin: no released artifact for ${stdenv.hostPlatform.system}");
in
stdenv.mkDerivation {
  pname = "fideslib-bin";
  inherit version;

  src = fetchurl src;

  nativeBuildInputs = lib.optionals stdenv.hostPlatform.isLinux [ autoPatchelfHook ];
  # Supplies libstdc++.so.6 / libgcc_s.so.1 for the rewrite. The library links the
  # C++ runtime dynamically on purpose: its public API exposes std::vector and
  # lbcrypto:: types, so a private static libstdc++ would put two C++ runtimes on
  # either side of the ABI boundary.
  buildInputs = lib.optionals stdenv.hostPlatform.isLinux [ stdenv.cc.cc.lib ];

  # The libraries ship as the release built and tested them; the only edit is
  # autoPatchelfHook's on Linux. On darwin nothing needs rewriting: nix links its own
  # programs against the system libc++ (its libcxx is an SDK stub naming
  # /usr/lib/libc++.1.dylib), which is the one the libraries use.
  dontStrip = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r . $out/
    runHook postInstall
  '';

  # Every shell that lists this package (mkShell, nix develop) gets FIDESLIB, the
  # package root, which plain -I/-L/-l builds use as their prefix.
  setupHook = writeText "fideslib-bin-setup-hook" ''
    export FIDESLIB=@out@
  '';

  meta = {
    description = "FIDESlib CKKS library with CPU and HAZE backends (prebuilt)";
    platforms = builtins.attrNames sources;
  };
}
