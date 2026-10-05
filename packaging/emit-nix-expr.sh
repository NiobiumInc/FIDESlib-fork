#!/usr/bin/env bash
#
# Emit a nix expression that consumes the published tarballs.
#
# nix packages the release rather than producing it. nixpkgs pins a single glibc per
# revision, and no revision pairs a glibc old enough to be portable with a compiler new
# enough for C++23 (22.05 has glibc 2.34 but gcc 11.3; gcc 14 first appears at 24.05,
# where glibc is already 2.39). See packaging/README.md. This is the same division
# sunscreen-llvm uses: build in a container, consume from nix by hash.
#
# Usage: emit-nix-expr.sh <dist-dir> <tag> <repo-url>

set -euo pipefail

DIST=${1:?usage: emit-nix-expr.sh <dist-dir> <tag> <repo-url>}
TAG=${2:?missing tag}
REPO=${3:?missing repo url}

# nix wants SRI hashes; sha256sum gives hex.
sri() {
    if command -v nix-hash >/dev/null 2>&1; then
        echo "sha256-$(nix-hash --type sha256 --flat --base64 "$1")"
    else
        echo "sha256-$(sha256sum "$1" | cut -d' ' -f1 | xxd -r -p | base64)"
    fi
}

emit_system() {
    local nix_system=$1 pattern=$2
    local f
    f=$(find "$DIST" -name "$pattern" | head -1)
    [ -n "$f" ] || return 0
    cat <<EOF
    $nix_system = {
      url = "$REPO/releases/download/$TAG/$(basename "$f")";
      hash = "$(sri "$f")";
    };
EOF
}

cat <<'HEADER'
# FIDESlib, consumed as a published binary release.
#
# Usage:
#   pkgs.callPackage ./fideslib-bin.nix { }
#
# autoPatchelfHook rewrites the ELF interpreter and RPATHs to point into the nix
# store, which is exactly what it is for: importing a foreign binary into nix. The
# tarball itself is built against an old glibc in a manylinux container so that it
# also works on non-nix machines.
{ lib, stdenv, fetchurl, autoPatchelfHook }:

let
  sources = {
HEADER
emit_system '"x86_64-linux"'  'fideslib-linux-x86-64-*.tar.gz'
emit_system '"aarch64-linux"' 'fideslib-linux-aarch64-*.tar.gz'
emit_system '"aarch64-darwin"' 'fideslib-macos-aarch64-*.tar.gz'
cat <<HEADER2
  };
  src = sources.\${stdenv.hostPlatform.system} or (throw
    "fideslib-bin: no released artifact for \${stdenv.hostPlatform.system}");
in
stdenv.mkDerivation {
  pname = "fideslib-bin";
  version = "${TAG#v}";

  src = fetchurl src;

  nativeBuildInputs = lib.optionals stdenv.isLinux [ autoPatchelfHook ];
  # Supplies libstdc++.so.6 / libgcc_s.so.1 for the rewrite. The library links the
  # C++ runtime dynamically on purpose: its public API exposes std::vector and
  # lbcrypto:: types, so a private static libstdc++ would put two C++ runtimes on
  # either side of the ABI boundary.
  buildInputs = [ stdenv.cc.cc.lib ];

  installPhase = ''
    runHook preInstall
    mkdir -p \$out
    cp -r . \$out/
    runHook postInstall
  '';

  meta = {
    description = "FIDESlib CKKS library with CPU and HAZE backends (prebuilt)";
    platforms = builtins.attrNames sources;
  };
}
HEADER2
