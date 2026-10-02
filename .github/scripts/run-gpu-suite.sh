#!/usr/bin/env bash
# Runs the GPU test suite without the tests listed in the skips file, then
# retries the failing tests once: OpenFHE seeds its random generator on every
# run, so a few tests flip with no source change. Exits 0 when no test outside
# the skips file fails twice.
#
# Usage: run-gpu-suite.sh <fideslib-test> <skips file> <results dir>
set -uo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 <fideslib-test> <skips file> <results dir>" >&2
  exit 2
fi
bin=$1 skips=$2 out=$3
mkdir -p "$out"

# The skips file as one gtest negative filter, comments and blank lines dropped.
skip=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$skips" | grep -v '^$' | paste -sd: -)

# Prints the failing test names in a gtest XML report, joined with ':'.
failed_tests() {
  python3 - "$1" <<'EOF'
import sys
import xml.etree.ElementTree as ET

names = []
for suite in ET.parse(sys.argv[1]).getroot().iter("testsuite"):
    for case in suite.iter("testcase"):
        if case.find("failure") is not None:
            names.append(f"{suite.get('name')}.{case.get('name')}")
print(":".join(names))
EOF
}

"$bin" --gtest_filter="-$skip" --gtest_output="xml:$out/run.xml"
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "GPU suite: no failures outside the skips file."
  exit 0
fi
if [ ! -s "$out/run.xml" ]; then
  echo "::error::fideslib-test exited $rc without writing its report; it probably crashed."
  exit "$rc"
fi
failed=$(failed_tests "$out/run.xml")
if [ -z "$failed" ]; then
  echo "::error::fideslib-test exited $rc but its report lists no failing test."
  exit "$rc"
fi

echo "Retrying the failing tests once: ${failed//:/ }"
if "$bin" --gtest_filter="$failed" --gtest_output="xml:$out/retry.xml"; then
  echo "::warning::Passed on retry, possibly flaky: ${failed//:/ }"
  exit 0
fi
still=$(failed_tests "$out/retry.xml" 2>/dev/null || true)
echo "::error::Failed twice: ${still:-$failed}"
exit 1
