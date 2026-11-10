#!/usr/bin/env bash
#
# Emit the nix expression that consumes one release's published tarballs: the template
# packaging/fideslib-bin.nix with its pinned block (version and sources) filled in.
# Everything else in the expression comes from the template unchanged.
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
TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fideslib-bin.nix"

# nix wants SRI hashes; sha256sum gives hex.
sri() {
    if command -v nix-hash >/dev/null 2>&1; then
        echo "sha256-$(nix-hash --type sha256 --flat --base64 "$1")"
    else
        echo "sha256-$(sha256sum "$1" | cut -d' ' -f1 | xxd -r -p | base64)"
    fi
}

# Exactly one tarball per system. The release job runs only when every build leg
# succeeded, so a missing or doubled tarball is a bug, not a platform to leave out:
# meta.platforms, and with it the flake's packages, come from this list.
emit_system() {
    local nix_system=$1 pattern=$2
    local f count
    f=$(find "$DIST" -name "$pattern")
    count=$(printf '%s' "$f" | grep -c . || true)
    if [ "$count" -ne 1 ]; then
        echo "expected one $pattern in $DIST, found $count" >&2
        exit 1
    fi
    cat <<EOF
    $nix_system = {
      url = "$REPO/releases/download/$TAG/$(basename "$f")";
      hash = "$(sri "$f")";
    };
EOF
}

pinned_block() {
    echo "  version = \"${TAG#v}\";"
    echo "  sources = {"
    emit_system '"x86_64-linux"'  'fideslib-linux-x86-64-*.tar.gz'
    emit_system '"aarch64-linux"' 'fideslib-linux-aarch64-*.tar.gz'
    emit_system '"aarch64-darwin"' 'fideslib-macos-aarch64-*.tar.gz'
    echo "  };"
}

BEGIN_MARK='# begin pinned release'
END_MARK='# end pinned release'

# A plain assignment, so set -e stops the script when pinned_block fails.
block=$(pinned_block)

# Replace what lies between the markers and keep the markers, so the output can be
# re-pinned the same way. One pass: state 0 before the block, 1 inside it, 2 after; any
# marker out of that order, or a block that never closes, is an error. The values go
# through the environment rather than awk -v, which some awks reject when they span lines.
PINNED="$block" BEGIN_MARK="$BEGIN_MARK" END_MARK="$END_MARK" awk '
    index($0, ENVIRON["BEGIN_MARK"]) {
        if (state != 0) exit 2
        print; print ENVIRON["PINNED"]; state = 1; next
    }
    index($0, ENVIRON["END_MARK"]) { if (state != 1) exit 2; state = 2 }
    state != 1 { print }
    END { if (state != 2) exit 2 }
' "$TEMPLATE" || {
    echo "$TEMPLATE needs exactly one pinned block: a '$BEGIN_MARK' line, then a '$END_MARK' line" >&2
    exit 1
}
