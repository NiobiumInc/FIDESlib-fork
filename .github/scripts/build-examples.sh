#!/usr/bin/env bash
# Builds every example under examples/ as a standalone CMake project against an
# installed FIDESlib, the way a user consumes the library, and reports which
# ones built. Each example is tried even after another fails. Examples listed
# in .github/examples-skip.txt are reported as skipped, with their reason.
# Exits 1 when any other example fails to configure or build.
#
# Usage: build-examples.sh <FIDESlib install prefix> <build dir> [cmake args...]
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <FIDESlib install prefix> <build dir> [cmake args...]" >&2
  exit 2
fi
prefix=$1 out=$2
shift 2
mkdir -p "$out"

trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<< "$1"; }

# Example name -> reason, from the skip list.
declare -A skip_reason=()
while IFS= read -r line; do
  name=$(trim "${line%%#*}")
  [ -n "$name" ] || continue
  reason=""
  [[ "$line" == *"#"* ]] && reason=$(trim "${line#*#}")
  skip_reason[$name]=$reason
done < .github/examples-skip.txt

built=() failed=() skipped=()
for dir in examples/*/; do
  name=$(basename "$dir")
  [ -f "$dir/CMakeLists.txt" ] || continue
  if [ -n "${skip_reason[$name]+set}" ]; then
    skipped+=("$name")
    continue
  fi
  echo "::group::Example $name"
  # The prefix goes through the environment: an example that changes its
  # compiler after project() makes CMake drop its cache and configure again,
  # which loses a -DCMAKE_PREFIX_PATH but not the environment variable.
  if CMAKE_PREFIX_PATH="$prefix" cmake -S "$dir" -B "$out/$name" \
       -DCMAKE_BUILD_TYPE=Release "$@" &&
     cmake --build "$out/$name" -j "$(nproc)"; then
    built+=("$name")
  else
    failed+=("$name")
  fi
  echo "::endgroup::"
done

{
  echo "## Examples against the installed package"
  echo
  echo "| Example | Result |"
  echo "|---|---|"
  for name in "${built[@]}"; do echo "| \`$name\` | built |"; done
  for name in "${failed[@]}"; do echo "| \`$name\` | **failed** |"; done
  for name in "${skipped[@]}"; do echo "| \`$name\` | skipped: ${skip_reason[$name]} |"; done
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"

if [ ${#failed[@]} -gt 0 ]; then
  echo "::error::Examples that did not build: ${failed[*]}"
  exit 1
fi
echo "All ${#built[@]} examples built."
