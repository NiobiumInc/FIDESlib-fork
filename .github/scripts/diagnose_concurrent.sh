#!/usr/bin/env bash
# Diagnostics for the race between threads in the concurrent lanes (runs in one process).
# Run from the build directory. Writes to diag/. Variants come from .github/diag-variants.txt:
#   base.N                 the unit as is
#   lock:<kinds>.N         FIDESLIB_CONCURRENT_OPS_DIAG=<kinds> (those op kinds behind one host lock)
#   asan.N                 the unit once, for a binary built with AddressSanitizer
#   env:VAR=1,VAR2=1.N     the unit with those environment variables set
#   repro:<pool mode>.N    the allocator alone (.github/diag/pool_reuse_repro.cu)
#   probe.N                deterministic stream-ordering checks (.github/diag/ordering_probe.cu)
#   markrepro:<pool mode>.N  the allocator's hand-offs checked with free marks (.github/diag/pool_mark_repro.cu)
# The gcp-ephemeral VM is stopped about 99 minutes into the job, so no variant starts after
# DEADLINE minutes of this script, and the artifacts always get uploaded.
#
# Every variant runs with lightweight GPU core dumps on (no device memory in them): a kernel that
# faults leaves a dump that names the kernel and the source line, read below with cuda-gdb when the
# runner has it.
set -u
out=diag
mkdir -p "$out"
export FIDESLIB_TEST_BACKEND=cuda FIDESLIB_CONCURRENT_OPS=1
# Newer drivers no longer read CUDA_ENABLE_LIGHTWEIGHT_COREDUMP; without it a dump holds all of the
# device memory. These flags are the set cuda-gdb lists as lightweight. After the dump the driver
# aborts the process (no skip_abort here), so a whole dump ends the run with SIGABRT.
# This branch reads the table-owner report instead: a dump would abort the process before the host
# prints it.
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory
export CUDA_COREDUMP_FILE="$PWD/$out/gpucore_%p.nvcudmp"
# The library exits through exit(0) on a CUDA failure, which cut the dumps short; hold it instead.
export FIDESLIB_FAILURE_HOLD_S=120
GDB="${CUDA_PATH:-/usr/local/cuda}/bin/cuda-gdb"
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
	grep -E '^\[(tabletrace\] (HOT|allocs=.*\(final\))|noxsreuse\]|repro\]|pinnedstaging\]|freemark\]|quarantine\] (LIVE|WRITE|tracked=))' "$out/$file.log" | head -30 | sed 's/^/    /' | tee -a "$out/summary.txt"
	# Table-owner report (TableOwner.cuh): every site=<object>+0x<return address> decoded in place.
	grep -E '^\[tableowner\] ' "$out/$file.log" | grep -v 'table checks on' | head -160 | while IFS= read -r line; do
		echo "    $line"
		site=$(printf '%s' "$line" | grep -o 'site=[^ ]*+0x[0-9a-f]*')
		if [ -n "$site" ]; then
			obj=${site#site=}
			off=${obj##*+}
			obj=${obj%+*}
			echo "        -> $(addr2line -f -C -i -e "$obj" "$(printf '0x%x' $((off - 1)))" 2>&1 | paste -sd' ' | cut -c1-300)"
		fi
	done | tee -a "$out/summary.txt"
	# Quarantine sites are <object>+0x<return address>; one byte back is the call itself.
	grep -o 'site=[^ ]*+0x[0-9a-f]*' "$out/$file.log" | sort | uniq -c | sort -rn | head -8 | while read -r n site; do
		obj=${site#site=}
		off=${obj##*+}
		obj=${obj%+*}
		printf '    %5s x %s\n' "$n" "$(addr2line -f -C -i -e "$obj" "$(printf '0x%x' $((off - 1)))" 2>&1 | paste -sd' ')"
	done | tee -a "$out/summary.txt"
	local core i=0
	for core in "$out"/gpucore_*.nvcudmp; do
		[ -f "$core" ] || continue
		i=$((i + 1))
		mv "$core" "$out/$file.gpucore$i.nvcudmp"
		core="$out/$file.gpucore$i.nvcudmp"
		# A dump cut short ends before its ELF section table does.
		local size need
		size=$(stat -c %s "$core")
		need=$(LC_ALL=C readelf -h "$core" 2>/dev/null | awk '/Start of section headers/ { o = $5 } /Size of section headers/ { e = $5 }
			/Number of section headers/ { n = $5 } END { if (o != "") print o + n * e }')
		echo "    dump $(basename "$core"): $size bytes, section table ends at ${need:-?}: $(
			[ -n "$need" ] && { [ "$size" -ge "$need" ] && echo whole || echo TRUNCATED; } || echo 'not an ELF file')" |
			tee -a "$out/summary.txt"
		if [ -x "$GDB" ]; then
			timeout 300 "$GDB" -nx -batch -ex 'set pagination off' -ex "target cudacore $core" \
				-ex 'info cuda kernels' -ex 'bt' -ex 'print $errorpc' -ex 'info line *$errorpc' -ex 'x/6i $errorpc' "${GDB_PROG:-./fideslib-test}" >"$core.txt" 2>&1
			{ echo "    ===== $(basename "$core")"; grep -v '^\s*$' "$core.txt" | head -60 | sed 's/^/    /'; } | tee -a "$out/summary.txt"
		else
			echo "    ===== $(basename "$core") written; no cuda-gdb at $GDB" | tee -a "$out/summary.txt"
		fi
	done
}

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu.txt"
echo "cuda-gdb: $([ -x "$GDB" ] && "$GDB" --version | head -1 || echo none)" | tee -a "$out/summary.txt"
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
	# The persistent table ring's fence, and the same test with the fence taken out (it must fail).
	fence) run "$v" 300 ./fideslib-test --gtest_filter='OpTableRingTest.*' ;;
	nofence) run "$v" 300 env FIDESLIB_TABLE_RING_NOFENCE=1 ./fideslib-test --gtest_filter='OpTableRingTest.*' ;;
	lock:*) run "$v" 1500 env FIDESLIB_CONCURRENT_OPS_DIAG="${x#lock:}" ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	env:*) run "$v" 1500 env $(printf '%s' "${x#env:}" | tr ',' ' ') ./fideslib-test --gtest_filter="$UNIT" --gtest_repeat=3 ;;
	# The allocator alone, without the library (.github/diag/pool_reuse_repro.cu): repro:<pool mode>.N
	repro:*)
		if [ ! -x ./pool-reuse-repro ]; then
			"${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" -O2 -std=c++17 -arch=native -o pool-reuse-repro ../.github/diag/pool_reuse_repro.cu >"$out/repro-build.log" 2>&1 ||
				{ echo "repro build failed:" | tee -a "$out/summary.txt"; head -20 "$out/repro-build.log" | sed 's/^/    /' | tee -a "$out/summary.txt"; }
		fi
		run "$v" 600 ./pool-reuse-repro --mode "${x#repro:}" --seconds 150
		;;
	# The allocator's hand-offs checked with free marks, without the library (.github/diag/pool_mark_repro.cu):
	# markrepro:<pool mode>.N
	markrepro:*)
		if [ ! -x ./pool-mark-repro ]; then
			"${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" -O2 -std=c++17 -arch=native -o pool-mark-repro ../.github/diag/pool_mark_repro.cu >"$out/markrepro-build.log" 2>&1 ||
				{ echo "markrepro build failed:" | tee -a "$out/summary.txt"; head -20 "$out/markrepro-build.log" | sed 's/^/    /' | tee -a "$out/summary.txt"; }
		fi
		run "$v" 900 ./pool-mark-repro --mode "${x#markrepro:}" --iters 2000 --seconds 90
		grep '^\[markrepro\]' "$out/$(printf '%s' "$v" | tr ':,' '-_').log" | sed 's/^/    /' | tee -a "$out/summary.txt"
		;;
	# Deterministic orderings a table relies on, without the library (.github/diag/ordering_probe.cu).
	probe)
		if [ ! -x ./ordering-probe ]; then
			"${CUDA_PATH:-/usr/local/cuda}/bin/nvcc" -O2 -std=c++17 -arch=native -o ordering-probe ../.github/diag/ordering_probe.cu >"$out/probe-build.log" 2>&1 ||
				{ echo "probe build failed:" | tee -a "$out/summary.txt"; head -20 "$out/probe-build.log" | sed 's/^/    /' | tee -a "$out/summary.txt"; }
		fi
		run "$v" 600 ./ordering-probe --iters 100
		grep '^\[probe\]' "$out/$(printf '%s' "$v" | tr ':,' '-_').log" | sed 's/^/    /' | tee -a "$out/summary.txt"
		;;
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
