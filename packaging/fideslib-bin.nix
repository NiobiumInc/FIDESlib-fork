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
  version = "unpinned";
  sources = { };
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
