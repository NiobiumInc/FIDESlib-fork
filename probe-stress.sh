#!/bin/bash
# Runs the stress filter with a time limit and prints, right away, what ran, what failed, what was slowest
# and where it stopped.
name=$1; repeat=$2; pooled=$3; limit=$4
filter='OpenFHEInterfaceTests/OpenFHEInterfaceTest.*:*ApiParity*:*CKKSRotateMany*:*AccumulateBroadcast*:*ChebyshevFixedManualCoworker*:*ContextLifetime*'
log=stress-$name.log
start=$(date +%s)
FIDESLIB_PROBE_POOLED_PT=$pooled timeout "$limit" ./fideslib-test --gtest_filter="$filter" --gtest_repeat="$repeat" > "$log" 2>&1
rc=$?
end=$(date +%s)
{
  echo "### $name (repeat=$repeat pooled=$pooled limit=${limit}s): exit=$rc, $((end-start)) s"
  echo "ran:    $(grep -c '^\[ RUN      \]' "$log")  ok: $(grep -c '^\[       OK \]' "$log")  failed lines: $(grep -c '^\[  FAILED  \] [A-Za-z0-9_/]*\.' "$log")"
  grep -E 'Repeating all tests' "$log" | tail -1
  echo "failed tests:"; grep -E '^\[  FAILED  \] [A-Za-z0-9_/]+\.[A-Za-z0-9_/]+' "$log" | sed -E 's/ \([0-9]+ ms\)//; s/ , where.*//' | sort | uniq -c | head -10
  echo "slowest:"; grep -E '^\[       OK \]' "$log" | sed -E 's/.*\] (.*) \(([0-9]+) ms\)/\2 \1/' | sort -rn | head -6
  echo "last RUN line: $(grep '^\[ RUN      \]' "$log" | tail -1)"
  echo "last 3 lines:"; tail -3 "$log" | cut -c1-200
} | tee -a probe-stress.summary >> "$GITHUB_STEP_SUMMARY"
