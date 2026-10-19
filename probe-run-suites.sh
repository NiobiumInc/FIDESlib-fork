#!/bin/bash
# Runs every gtest suite of fideslib-test in its own process and records which tests failed.
tag=$1
out=probe-$tag
: > $out.summary
./fideslib-test --gtest_list_tests 2>/dev/null | grep -E '^[A-Za-z0-9_/]+\.$' | sed 's/\.$//' > $out.suites
while read -r s; do
  ./fideslib-test --gtest_filter="$s.*" > "$out.$(echo "$s" | tr '/' '_').log" 2>&1
  rc=$?
  tot=$(grep -E '^\[==========\] [0-9]+ tests' "$out.$(echo "$s" | tr '/' '_').log" | sed -E 's/.*\] ([0-9]+) tests.*/\1/' | tail -1)
  bad=$(grep -cE '^\[  FAILED  \] .*\.[A-Za-z0-9_/]+' "$out.$(echo "$s" | tr '/' '_').log")
  echo "$tag suite=$s tests=${tot:-0} exit=$rc failed_lines=$bad" >> $out.summary
  grep -E '^\[  FAILED  \] [A-Za-z0-9_/]+\.[A-Za-z0-9_/]+' "$out.$(echo "$s" | tr '/' '_').log" | sed -E 's/ , where.*//' | sort -u >> $out.failed
done < $out.suites
{ echo "### $tag"; cat $out.summary; echo '--- failed tests'; sort -u $out.failed 2>/dev/null | head -80; } >> "$GITHUB_STEP_SUMMARY"
