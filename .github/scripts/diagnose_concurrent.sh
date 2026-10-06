#!/usr/bin/env bash
# Diagnostics for the race between threads in the concurrent lanes (runs in one process).
# Run from the build directory. Writes to diag/. Variants come from .github/diag-variants.txt:
#   base.N                 the unit as is
#   lock:<kinds>.N         FIDESLIB_CONCURRENT_OPS_DIAG=<kinds> (those op kinds behind one host lock)
#   asan.N                 the unit once, for a binary built with AddressSanitizer
#   env:VAR=1,VAR2=1.N     the unit with those environment variables set
# The gcp-ephemeral VM is stopped about 99 minutes into the job, so no variant starts after
# DEADLINE minutes of this script, and the artifacts always get uploaded.
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
	grep -E '^\[(tabletrace\] (HOT|allocs=.*\(final\))|noxsreuse\])' "$out/$file.log" | head -20 | sed 's/^/    /' | tee -a "$out/summary.txt"
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu.txt"
DEADLINE=${DEADLINE:-78}
for v in $(grep -v '^#' ../.github/diag-variants.txt); do
	if [ $SECONDS -gt $((DEADLINE * 60)) ]; then
		echo "deadline reached after $((SECONDS / 60)) min: not starting $v" | tee -a "$out/summary.txt"
		break
	fi
	x=${v%.*}
	free -g | awk 'NR==2 { printf "    host memory: %s GB used of %s\n", $3, $2 }'
	case "$x" in
	base) run "$v" 1500 ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	lock:*) run "$v" 1500 env FIDESLIB_CONCURRENT_OPS_DIAG="${x#lock:}" ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	env:*) run "$v" 1500 env $(printf '%s' "${x#env:}" | tr ',' ' ') ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	memcheck-sor)
		# Stream-ordered race tracking: a use of a cudaMallocAsync allocation after its
		# cudaFreeAsync (or before its allocation) in stream order, whatever the timing.
		f=$(printf '%s' "$v" | tr ':,' '-_')
		run "$v" 2700 "$SAN" --tool memcheck --track-stream-ordered-races all --show-backtrace device \
			--print-limit 30 --log-file "$out/$f.sanitizer.txt" ./fideslib-test --gtest_filter="$UNIT"
		{ echo "    ===== $f.sanitizer.txt"; grep -v '^\s*$' "$out/$f.sanitizer.txt" | head -120 | sed 's/^/    /'; } | tee -a "$out/summary.txt"
		;;
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
for f in "$out"/*.sanitizer.txt; do
	[ -f "$f" ] && echo "$(basename "$f" .sanitizer.txt): $(grep -h 'ERROR SUMMARY' "$f" | tail -1)" | tee -a "$out/summary.txt"
done
{ echo '```'; cat "$out/summary.txt"; echo '```'; } >>"$GITHUB_STEP_SUMMARY" 2>/dev/null || true
exit 0
