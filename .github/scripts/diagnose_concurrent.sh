#!/usr/bin/env bash
# Diagnostics for the race between threads in the concurrent lanes (runs in one process).
# Run from the build directory. Writes to diag/. Variants come from .github/diag-variants.txt:
#   base.N                 the unit as is
#   lock:<kinds>.N         FIDESLIB_CONCURRENT_OPS_DIAG=<kinds> (those op kinds behind one host lock)
#   asan.N                 the unit once, for a binary built with AddressSanitizer
set -u
out=diag
mkdir -p "$out"
export FIDESLIB_TEST_BACKEND=cuda FIDESLIB_CONCURRENT_OPS=1
T=ConcurrencyTests/ConcurrentOpsTest
UNIT="$T.LanesMatchSerial/*:$T.SharedOperandsMatchSerial/*"
SAN="${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer"

run() {
	# The file name drops ':' and ',', which artifact uploads refuse; the summary keeps the variant.
	local name=$1 limit=$2 file
	file=$(printf '%s' "$1" | tr ':,' '-_')
	shift 2
	local start=$SECONDS
	timeout "$limit" "$@" >"$out/$file.log" 2>&1
	local rc=$?
	local failed ran
	failed=$(grep -cE '^\[  FAILED  \] .*\(([0-9]+) ms\)$' "$out/$file.log")
	ran=$(grep -cE '^\[ +(OK|FAILED) +\] .*\(([0-9]+) ms\)$' "$out/$file.log")
	printf '%-60s rc=%-4s failed %3s of %3s %5ss %s %s\n' "$name" "$rc" "$failed" "$ran" $((SECONDS - start)) \
		"$(grep -m1 -o 'Cuda failure.*' "$out/$file.log")" \
		"$(grep -m1 -o 'corrupted [a-z -]*\|double free[a-z -]*\|free(): [a-z -]*' "$out/$file.log")" | tee -a "$out/summary.txt"
	grep -E '^\[  FAILED  \] .*\(([0-9]+) ms\)$' "$out/$file.log" | sed 's/, where GetParam.*//' | sed 's/^/    /' | tee -a "$out/summary.txt"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu.txt"
for v in $(grep -v '^#' ../.github/diag-variants.txt); do
	x=${v%.*}
	free -g | awk 'NR==2 { printf "    host memory: %s GB used of %s\n", $3, $2 }'
	case "$x" in
	base) run "$v" 1500 ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	lock:*) run "$v" 1500 env FIDESLIB_CONCURRENT_OPS_DIAG="${x#lock:}" ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	asan)
		rm -f "$out"/asanreport.*
		run "$v" 2400 env OMP_NUM_THREADS=8 OMP_WAIT_POLICY=passive \
			ASAN_OPTIONS="protect_shadow_gap=0:detect_leaks=0:quarantine_size_mb=64:malloc_context_size=20:log_path=$out/asanreport" \
			./fideslib-test --gtest_filter="$UNIT"
		for f in "$out"/asanreport.*; do
			[ -f "$f" ] || continue
			echo "    ===== $f" | tee -a "$out/summary.txt"
			grep -v '^\s*$' "$f" | head -80 | sed 's/^/    /' | tee -a "$out/summary.txt"
			mv "$f" "$out/$(printf '%s' "$v" | tr ':,' '-_').$(basename "$f").txt"
		done
		;;
	esac
done
{ echo '```'; cat "$out/summary.txt"; echo '```'; } >>"$GITHUB_STEP_SUMMARY" 2>/dev/null || true
exit 0
