#!/usr/bin/env bash
#
# Exercise packaging/pin-nix-release.sh against a throwaway bare remote. It is the one
# release step that rewrites public refs, and in CI it otherwise runs only after a real
# publish, so this runs on every release-workflow run instead. It needs git and nothing
# else.
#
# History used, with r1..r3 standing for release commits:
#
#   r1 -- r2        (r2 descends from r1)
#     \
#      r3           (r3 does not descend from r2)
#
# Usage: test-pin-nix-release.sh [workdir]

set -euo pipefail

PIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pin-nix-release.sh"
WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

# Keep the caller's git configuration (signing, hooks, identity) out of the test.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid

src="$WORK/src"
git init -q -b main "$src"
mkdir -p "$src/packaging"
release_commit() {
    echo "template $1" > "$src/packaging/fideslib-bin.nix"
    git -C "$src" add packaging/fideslib-bin.nix
    git -C "$src" commit -q -m "release $1"
    git -C "$src" rev-parse HEAD
}
r1=$(release_commit r1)
r2=$(release_commit r2)
git -C "$src" -c advice.detachedHead=false checkout -q "$r1"
r3=$(release_commit r3)
git clone -q --bare "$src" "$WORK/remote.git"

# Run the pin for one release commit from a fresh clone, as the release job does. Each
# run (numbered by the caller, since this runs in a $(...) subshell) sees a different
# clock, so a pin commit that depended on the time of the run rather than on the
# release would come out different on a re-run. Bash clears -e in a $(...) subshell;
# set it again so a failing step fails the run.
pin() {
    set -e
    local release=$1 tag=$2 run=$3
    local clone="$WORK/clone-$run" clock="@$((2000000000 + run * 3600))"
    git clone -q --no-checkout "$WORK/remote.git" "$clone"
    git -C "$clone" -c advice.detachedHead=false checkout -q "$release"
    echo "published $tag" > "$WORK/$tag.nix"
    (cd "$clone" && GIT_AUTHOR_DATE="$clock" GIT_COMMITTER_DATE="$clock" \
        bash "$PIN" "$tag" "$WORK/$tag.nix" "$WORK/remote.git")
}
remote_ref() { git -C "$WORK/remote.git" rev-parse --verify -q "$1" || echo none; }

fail=0
check() {
    if [ "$1" = "$2" ]; then
        printf '  %-38s PASS\n' "$3"
    else
        printf '  %-38s FAIL (got %s, want %s)\n' "$3" "$1" "$2"
        fail=1
    fi
}

echo "== pinning"
out=$(pin "$r1" v1 1) || true; p1=$(sed -n 's/^commit=//p' <<< "$out")
check "$(remote_ref refs/tags/v1-nix)" "$p1" "first pin: tag"
check "$(remote_ref refs/heads/nix-release)" "$p1" "first pin: branch"
check "$(git -C "$WORK/remote.git" rev-parse "$p1^")" "$r1" "pin commit sits on the release commit"
check "$(git -C "$WORK/remote.git" show "$p1:packaging/fideslib-bin.nix")" "published v1" \
      "pin commit holds the published file"

out=$(pin "$r1" v1 2) || true
check "$(sed -n 's/^commit=//p' <<< "$out")" "$p1" "re-run: same commit"
check "$(remote_ref refs/heads/nix-release)" "$p1" "re-run: branch unchanged"

out=$(pin "$r2" v2 3) || true; p2=$(sed -n 's/^commit=//p' <<< "$out")
check "$(remote_ref refs/tags/v2-nix)" "$p2" "newer release: tag"
check "$(remote_ref refs/heads/nix-release)" "$p2" "newer release: branch moves"

out=$(pin "$r1" v1 4) || true
check "$(sed -n 's/^moved=//p' <<< "$out" | cut -c1-2)" "no" "older release: reports no move"
check "$(remote_ref refs/heads/nix-release)" "$p2" "older release: branch stays"
check "$(remote_ref refs/tags/v1-nix)" "$p1" "older release: tag unchanged"

out=$(pin "$r3" v3 5) || true; p3=$(sed -n 's/^commit=//p' <<< "$out")
check "$(remote_ref refs/tags/v3-nix)" "$p3" "unrelated release: tag"
check "$(remote_ref refs/heads/nix-release)" "$p2" "unrelated release: branch stays"

echo
if [ "$fail" -eq 0 ]; then echo "PIN TEST PASSED"; else echo "PIN TEST FAILED" >&2; fi
exit "$fail"
