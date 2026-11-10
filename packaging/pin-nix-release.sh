#!/usr/bin/env bash
#
# Pin one published release for nix: commit the release's own fideslib-bin.nix on top of
# the release commit, tag that commit <tag>-nix, and move the nix-release branch to it.
#
# A commit cannot carry the hashes of the release built from it, so the flake's
# fideslib-bin can only be pinned after publishing; this is that step. The tag keeps every
# pin commit reachable after the branch moves on, so a flake.lock that recorded one keeps
# resolving. The branch moves only forward: when it already pins a release, it moves only
# if this release's commit descends from that one, so re-running an older run, or pinning
# a release built elsewhere, still tags but leaves nix-release alone.
#
# The pin commit reuses the release commit's dates, so it is a function of the release
# alone: running this again for the same release pushes the same commit and tag, and
# changes nothing.
#
# Run from a checkout of the release commit with its history (merge-base needs it).
# packaging/test-pin-nix-release.sh exercises this against a throwaway remote.
#
# Usage: pin-nix-release.sh <tag> <published fideslib-bin.nix> <remote>

set -euo pipefail

TAG=${1:?usage: pin-nix-release.sh <tag> <published fideslib-bin.nix> <remote>}
EXPR=${2:?missing the published fideslib-bin.nix}
REMOTE=${3:?missing remote}
PIN_FILE=packaging/fideslib-bin.nix
BRANCH=nix-release
BOT_NAME='github-actions[bot]'
BOT_EMAIL='41898282+github-actions[bot]@users.noreply.github.com'

release=$(git rev-parse HEAD)
date=$(git log -1 --format=%cI "$release")

cp "$EXPR" "$PIN_FILE"
GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git -c user.name="$BOT_NAME" -c user.email="$BOT_EMAIL" \
        commit -q -m "chore(packaging): pin fideslib-bin to $TAG" -- "$PIN_FILE"
pin=$(git rev-parse HEAD)

refs=("$pin:refs/tags/$TAG-nix")
if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null; then
    git fetch -q "$REMOTE" "refs/heads/$BRANCH"
    pinned_release=$(git rev-parse FETCH_HEAD^)
    if git merge-base --is-ancestor "$pinned_release" "$release"; then
        refs+=("+$pin:refs/heads/$BRANCH")
        moved=yes
    else
        moved="no: $BRANCH pins ${pinned_release:0:7}, which $TAG does not descend from"
    fi
else
    refs+=("$pin:refs/heads/$BRANCH")
    moved=yes
fi

# One atomic push: the tag and the branch move together or not at all.
git push -q --atomic "$REMOTE" "${refs[@]}"

echo "tag=$TAG-nix"
echo "commit=$pin"
echo "moved=$moved"
