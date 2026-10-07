// DEBUG BRANCH ONLY. The stream-ordered allocator's reuse across streams, with several host threads
// issuing on a small set of shared stream handles, as the library's concurrent mode does.
//
// Every op allocates a small table on one stream, uploads a tag into it from pageable memory, checks
// the tag in a kernel on the same stream and frees the table on that stream. Between ops the threads
// record events on one shared handle and make another wait on them, as Stream::wait does. Whatever
// the pool does, a check must see its own tag: it can only see another if the block was handed to
// its stream before the upload of an earlier owner had run.
//
//   --mode 0   the pool's defaults (all three reuse policies on)
//   --mode 1   event-dependency reuse only (internal dependencies and opportunistic reuse off)
//   --mode 2   no reuse across streams
//   --mode 3   event-dependency reuse off, the other two on
//
// Prints one "[repro]" line; exits 1 when a check saw a tag that was not its own.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Report {
	unsigned long long mismatches; ///< ops whose check saw a word that was not its tag
	unsigned long long expected, got, address;
};

__device__ Report* g_report;

/// Waits `spin` cycles, so the device stays behind the host, then compares every word with `tag`.
__global__ void Check(const unsigned long long* table, const int n, const unsigned long long tag, const long long spin) {
	const long long start = clock64();
	while (clock64() - start < spin) {
	}
	for (int i = 0; i < n; ++i) {
		const unsigned long long v = table[i];
		if (v != tag) {
			if (atomicAdd(&g_report->mismatches, 1ull) == 0) {
				g_report->expected = tag;
				g_report->got	   = v;
				g_report->address  = reinterpret_cast<unsigned long long>(table + i);
			}
			return;
		}
	}
}

void Fail(const char* what, const cudaError_t e) {
	std::printf("[repro] %s: %s\n", what, cudaGetErrorString(e));
	std::fflush(stdout);
	std::exit(2);
}

void Worker(const int id, const std::vector<cudaStream_t>& handles, const double seconds, const long long max_spin,
  std::atomic<unsigned long long>& ops) {
	std::mt19937_64 rng(static_cast<unsigned long long>(id) * 7919u + 1u);
	cudaEvent_t ev[2];
	for (cudaEvent_t& e : ev)
		if (const cudaError_t r = cudaEventCreateWithFlags(&e, cudaEventDisableTiming); r != cudaSuccess)
			Fail("cudaEventCreateWithFlags", r);
	const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
	unsigned long long i = 0;
	for (; (i & 255) != 0 || std::chrono::steady_clock::now() < end; ++i) {
		const cudaStream_t s			  = handles[rng() % handles.size()];
		const int n						  = 2 + static_cast<int>(rng() % 31); // 16 to 256 bytes, like the per-op tables
		const unsigned long long tag	  = (static_cast<unsigned long long>(id + 1) << 48) | (i & 0xffffffffffffull);
		const std::vector<unsigned long long> host(static_cast<size_t>(n), tag);
		unsigned long long* table		  = nullptr;
		if (const cudaError_t r = cudaMallocAsync(reinterpret_cast<void**>(&table), static_cast<size_t>(n) * 8, s); r != cudaSuccess)
			Fail("cudaMallocAsync", r);
		cudaMemcpyAsync(table, host.data(), static_cast<size_t>(n) * 8, cudaMemcpyHostToDevice, s);
		Check<<<1, 1, 0, s>>>(table, n, tag, static_cast<long long>(rng() % static_cast<unsigned long long>(max_spin)));
		cudaFreeAsync(table, s);
		// A fence between two shared handles, as Stream::wait issues it: record, then wait.
		if (rng() & 1) {
			const cudaStream_t x = handles[rng() % handles.size()];
			const cudaStream_t y = handles[rng() % handles.size()];
			if (x != y) {
				cudaEventRecord(ev[i & 1], x);
				cudaStreamWaitEvent(y, ev[i & 1], 0);
			}
		}
		if ((i & 1023) == 0)
			if (const cudaError_t r = cudaGetLastError(); r != cudaSuccess)
				Fail("op", r);
	}
	ops.fetch_add(i);
}

} // namespace

int main(int argc, char** argv) {
	int mode = 1, threads = 4, nhandles = 8;
	double seconds		= 60;
	long long max_spin = 200000;
	for (int a = 1; a + 1 < argc; a += 2) {
		const std::string k = argv[a];
		if (k == "--mode")
			mode = std::atoi(argv[a + 1]);
		else if (k == "--threads")
			threads = std::atoi(argv[a + 1]);
		else if (k == "--handles")
			nhandles = std::atoi(argv[a + 1]);
		else if (k == "--seconds")
			seconds = std::atof(argv[a + 1]);
		else if (k == "--spin")
			max_spin = std::atoll(argv[a + 1]);
	}
	if (threads < 1 || nhandles < 2 || max_spin < 1 || mode < 0 || mode > 3) {
		std::printf("[repro] bad arguments\n");
		return 2;
	}

	cudaMemPool_t pool = nullptr;
	if (const cudaError_t r = cudaDeviceGetDefaultMemPool(&pool, 0); r != cudaSuccess)
		Fail("cudaDeviceGetDefaultMemPool", r);
	uint64_t threshold = UINT64_MAX;
	cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
	int internal = 1, opportunistic = 1, follow = 1;
	if (mode == 1)
		internal = opportunistic = 0;
	else if (mode == 2)
		internal = opportunistic = follow = 0;
	else if (mode == 3)
		follow = 0;
	cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies, &internal);
	cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic, &opportunistic);
	cudaMemPoolSetAttribute(pool, cudaMemPoolReuseFollowEventDependencies, &follow);
	cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies, &internal);
	cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowOpportunistic, &opportunistic);
	cudaMemPoolGetAttribute(pool, cudaMemPoolReuseFollowEventDependencies, &follow);

	Report* report = nullptr;
	if (const cudaError_t r = cudaHostAlloc(&report, sizeof(Report), cudaHostAllocMapped); r != cudaSuccess)
		Fail("cudaHostAlloc", r);
	std::memset(report, 0, sizeof(Report));
	Report* device_report = nullptr;
	cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_report), report, 0);
	cudaMemcpyToSymbol(g_report, &device_report, sizeof(device_report));

	// Blocking streams, like the library's pool of shared handles.
	std::vector<cudaStream_t> handles(static_cast<size_t>(nhandles));
	for (cudaStream_t& h : handles)
		if (const cudaError_t r = cudaStreamCreateWithFlags(&h, cudaStreamDefault); r != cudaSuccess)
			Fail("cudaStreamCreateWithFlags", r);

	std::atomic<unsigned long long> ops{ 0 };
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; ++t)
		workers.emplace_back(Worker, t, std::cref(handles), seconds, max_spin, std::ref(ops));
	for (std::thread& w : workers)
		w.join();
	if (const cudaError_t r = cudaDeviceSynchronize(); r != cudaSuccess)
		Fail("cudaDeviceSynchronize", r);

	std::printf("[repro] mode=%d (internal=%d opportunistic=%d follow-events=%d) threads=%d handles=%d seconds=%.0f ops=%llu mismatches=%llu", mode,
	  internal, opportunistic, follow, threads, nhandles, seconds, ops.load(), report->mismatches);
	if (report->mismatches != 0)
		std::printf(" first: expected 0x%016llx (thread %llu, op %llu) got 0x%016llx (thread %llu, op %llu) at 0x%llx", report->expected,
		  (report->expected >> 48) - 1, report->expected & 0xffffffffffffull, report->got, (report->got >> 48) - 1, report->got & 0xffffffffffffull,
		  report->address);
	std::printf("\n");
	std::fflush(stdout);
	return report->mismatches != 0 ? 1 : 0;
}
