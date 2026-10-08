#!/usr/bin/env bash
# Measurement for the concurrent flaky list (temporary branches only): the lane tests, with
# StagedLanesMatchSerial, together in one process, three times over, as many times as fit before
# DEADLINE minutes (the gcp-ephemeral VM stops about 99 minutes into the job). No retries.
# Run from the build directory; writes gpu-results/measure/ and never fails the job.
set -u
out=gpu-results/measure
mkdir -p "$out"
export FIDESLIB_TEST_BACKEND=cuda FIDESLIB_CONCURRENT_OPS=1
T=ConcurrencyTests/ConcurrentOpsTest
UNIT="$T.StagedLanesMatchSerial/*:$T.LanesMatchSerial/*:$T.SharedOperandsMatchSerial/*"
DEADLINE=${DEADLINE:-70}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >"$out/gpu.txt"
echo "commit $(git -C .. rev-parse --short HEAD)" | tee "$out/summary.txt"
i=0
while [ $SECONDS -lt $((DEADLINE * 60)) ]; do
	i=$((i + 1))
	start=$SECONDS
	timeout 1500 ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 >"$out/unit$i.log" 2>&1
	rc=$?
	failed=$(grep -cE '^\[  FAILED  \] .*\([0-9]+ ms\)$' "$out/unit$i.log")
	ran=$(grep -cE '^\[ +(OK|FAILED) +\] .*\([0-9]+ ms\)$' "$out/unit$i.log")
	printf 'unit %2d rc=%-4s failed %3s of %3s %5ss %s\n' "$i" "$rc" "$failed" "$ran" $((SECONDS - start)) \
		"$(grep -m1 -oE 'Cuda failure.*|CudaError.*' "$out/unit$i.log")" | tee -a "$out/summary.txt"
	grep -E '^\[  FAILED  \] .*\([0-9]+ ms\)$' "$out/unit$i.log" | sed 's/, where GetParam.*//' | sed 's/^/    /' | tee -a "$out/summary.txt"
done
{ echo '```'; cat "$out/summary.txt"; echo '```'; } >>"$GITHUB_STEP_SUMMARY" 2>/dev/null || true
exit 0
