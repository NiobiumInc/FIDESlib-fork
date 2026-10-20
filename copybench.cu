// PROBE (not for merge): what a host-to-device copy costs the calling thread, by host memory kind,
// stream kind and the work pending on other streams. Standalone CUDA, no library code.
//
// One "item" is 24 copies of 0.5 MiB (one plaintext's limbs). Items cycle over 8 host and 8 device
// buffer sets. Optionally one spin kernel per item runs on a separate blocking stream, standing in for
// the multiply that consumes the previous plaintext.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#define CK(x)                                                                                              \
	do {                                                                                                   \
		cudaError_t e_ = (x);                                                                              \
		if (e_ != cudaSuccess) {                                                                           \
			std::printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_));           \
			std::exit(1);                                                                                  \
		}                                                                                                  \
	} while (0)

__global__ void spin(const long long cycles) {
	const long long t0 = clock64();
	while (clock64() - t0 < cycles) {
	}
}

constexpr size_t kLimbBytes = 512 * 1024;
constexpr int kLimbs		= 24;
constexpr int kItems		= 192;
constexpr int kSets			= 8;

enum class Mem { Pageable, Registered, PinnedAlloc, RegisteredHuge };
enum class Strm { Legacy, Blocking, NonBlocking };

static const char* Name(Mem m) { return m == Mem::Pageable ? "pageable" : m == Mem::Registered ? "registered" : m == Mem::PinnedAlloc ? "pinned_alloc" : "registered_huge"; }
static const char* Name(Strm s) { return s == Strm::Legacy ? "legacy" : s == Strm::Blocking ? "blocking" : "nonblocking"; }

struct Result {
	double p50, p90, p99, max, copy_ms, wall_ms;
};

static Result Run(Mem mem, Strm sk, double kernel_us, long long cycles_per_us, int nstreams = 0) {
	std::vector<std::vector<void*>> host(kSets, std::vector<void*>(kLimbs)), dev(kSets, std::vector<void*>(kLimbs));
	std::vector<void*> slabs(kSets, nullptr);
	constexpr size_t kHuge = 2u << 20;
	if (mem == Mem::RegisteredHuge)
		for (int s = 0; s < kSets; ++s) {
			const size_t bytes = ((kLimbs * kLimbBytes + kHuge - 1) / kHuge) * kHuge;
			slabs[s]		   = std::aligned_alloc(kHuge, bytes);
			madvise(slabs[s], bytes, MADV_HUGEPAGE);
			std::memset(slabs[s], s, bytes); // touch, so the pages exist before they are registered
			CK(cudaHostRegister(slabs[s], bytes, cudaHostRegisterDefault));
		}
	for (int s = 0; s < kSets; ++s)
		for (int l = 0; l < kLimbs; ++l) {
			void* h = nullptr;
			if (mem == Mem::RegisteredHuge) {
				h = static_cast<char*>(slabs[s]) + l * kLimbBytes;
				host[s][l] = h;
				CK(cudaMalloc(&dev[s][l], kLimbBytes));
				continue;
			}
			if (mem == Mem::PinnedAlloc) {
				CK(cudaHostAlloc(&h, kLimbBytes, cudaHostAllocDefault));
			} else {
				h = std::malloc(kLimbBytes);
				if (mem == Mem::Registered)
					CK(cudaHostRegister(h, kLimbBytes, cudaHostRegisterDefault));
			}
			std::memset(h, (s * 31 + l) & 0xff, kLimbBytes);
			host[s][l] = h;
			CK(cudaMalloc(&dev[s][l], kLimbBytes));
		}
	std::vector<cudaStream_t> many(nstreams > 0 ? nstreams : 0);
	for (auto& m : many)
		CK(cudaStreamCreateWithFlags(&m, cudaStreamDefault)); // like the library's pooled streams
	cudaStream_t copy_stream = nullptr;
	if (nstreams == 0 && sk == Strm::Blocking)
		CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamDefault));
	if (nstreams == 0 && sk == Strm::NonBlocking)
		CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
	cudaStream_t consume_stream;
	CK(cudaStreamCreateWithFlags(&consume_stream, cudaStreamDefault)); // like the library's pooled streams

	std::vector<double> us;
	us.reserve(kItems * kLimbs);
	CK(cudaDeviceSynchronize());
	const auto w0 = std::chrono::steady_clock::now();
	for (int i = 0; i < kItems; ++i) {
		const int s = i % kSets;
		for (int l = 0; l < kLimbs; ++l) {
			const auto a = std::chrono::steady_clock::now();
			CK(cudaMemcpyAsync(dev[s][l], host[s][l], kLimbBytes, cudaMemcpyHostToDevice, nstreams > 0 ? many[i % nstreams] : copy_stream));
			us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
		}
		if (kernel_us > 0)
			spin<<<1, 1, 0, consume_stream>>>(static_cast<long long>(kernel_us * cycles_per_us));
	}
	CK(cudaDeviceSynchronize());
	const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();

	std::vector<double> v = us;
	std::sort(v.begin(), v.end());
	double sum = 0;
	for (double x : v)
		sum += x;
	Result r{ v[v.size() / 2], v[static_cast<size_t>(v.size() * 0.9)], v[static_cast<size_t>(v.size() * 0.99)], v.back(), sum / 1000.0, wall };

	for (int s = 0; s < kSets; ++s)
		for (int l = 0; l < kLimbs; ++l) {
			CK(cudaFree(dev[s][l]));
			if (mem == Mem::RegisteredHuge) {
				// the slab is released below
			} else if (mem == Mem::PinnedAlloc)
				CK(cudaFreeHost(host[s][l]));
			else {
				if (mem == Mem::Registered)
					CK(cudaHostUnregister(host[s][l]));
				std::free(host[s][l]);
			}
		}
	if (mem == Mem::RegisteredHuge)
		for (int s = 0; s < kSets; ++s) {
			CK(cudaHostUnregister(slabs[s]));
			std::free(slabs[s]);
		}
	for (auto& m : many)
		CK(cudaStreamDestroy(m));
	if (copy_stream)
		CK(cudaStreamDestroy(copy_stream));
	CK(cudaStreamDestroy(consume_stream));
	return r;
}

// What registering host memory costs: allocate and touch `total` bytes, register them (one call per limb
// or one call for the whole block), unregister, free. Times in milliseconds.
static void RegisterCost(const char* kind, bool huge, bool per_limb, size_t total) {
	using clk = std::chrono::steady_clock;
	const auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
	const size_t align = huge ? (size_t(2) << 20) : size_t(4096);
	const auto t0	   = clk::now();
	void* base		   = std::aligned_alloc(align, total);
	if (base == nullptr) {
		std::printf("[copy-reg] kind=%s total_MiB=%zu allocation failed\n", kind, total >> 20);
		return;
	}
	if (huge)
		madvise(base, total, MADV_HUGEPAGE);
	std::memset(base, 1, total);
	const auto t1 = clk::now();
	size_t calls  = 0;
	cudaError_t err = cudaSuccess;
	if (per_limb) {
		for (size_t off = 0; off < total && err == cudaSuccess; off += kLimbBytes, ++calls)
			err = cudaHostRegister(static_cast<char*>(base) + off, kLimbBytes, cudaHostRegisterDefault);
	} else {
		err = cudaHostRegister(base, total, cudaHostRegisterDefault);
		calls = 1;
	}
	const auto t2 = clk::now();
	if (err != cudaSuccess) {
		std::printf("[copy-reg] kind=%s total_MiB=%zu register failed after %zu calls: %s\n", kind, total >> 20, calls, cudaGetErrorString(err));
		cudaGetLastError();
	}
	if (per_limb) {
		for (size_t off = 0, n = 0; n < calls && off < total; off += kLimbBytes, ++n)
			cudaHostUnregister(static_cast<char*>(base) + off);
	} else if (err == cudaSuccess) {
		cudaHostUnregister(base);
	}
	const auto t3 = clk::now();
	std::free(base);
	const auto t4 = clk::now();
	const double gib = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
	std::printf("[copy-reg] kind=%-22s total_MiB=%-5zu calls=%-6zu alloc_touch_ms=%8.1f register_ms=%8.1f unregister_ms=%8.1f free_ms=%7.1f register_ms_per_GiB=%7.1f register_us_per_call=%8.1f\n", kind, total >> 20, calls, ms(t0, t1), ms(t1, t2),
	  ms(t2, t3), ms(t3, t4), ms(t1, t2) / gib, ms(t1, t2) * 1000.0 / static_cast<double>(calls));
	std::fflush(stdout);
}

static void PinnedAllocCost(size_t total) {
	using clk = std::chrono::steady_clock;
	const auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
	void* p		  = nullptr;
	const auto t0 = clk::now();
	cudaError_t err = cudaHostAlloc(&p, total, cudaHostAllocDefault);
	if (err == cudaSuccess)
		std::memset(p, 1, total);
	const auto t1 = clk::now();
	if (err == cudaSuccess)
		cudaFreeHost(p);
	const auto t2 = clk::now();
	if (err != cudaSuccess)
		cudaGetLastError();
	std::printf("[copy-reg] kind=%-22s total_MiB=%-5zu alloc_touch_ms=%8.1f free_ms=%7.1f %s\n", "cudaHostAlloc", total >> 20, ms(t0, t1), ms(t1, t2), err == cudaSuccess ? "" : cudaGetErrorString(err));
	std::fflush(stdout);
}

int main() {
	int dev = 0;
	CK(cudaSetDevice(dev));
	// Calibrate the spin kernel against wall time instead of asking the driver for a clock rate.
	cudaEvent_t e0, e1;
	CK(cudaEventCreate(&e0));
	CK(cudaEventCreate(&e1));
	spin<<<1, 1>>>(1000); // load the module before timing
	CK(cudaDeviceSynchronize());
	constexpr long long kCal = 200'000'000;
	CK(cudaEventRecord(e0));
	spin<<<1, 1>>>(kCal);
	CK(cudaEventRecord(e1));
	CK(cudaEventSynchronize(e1));
	float cal_ms = 0;
	CK(cudaEventElapsedTime(&cal_ms, e0, e1));
	const long long cycles_per_us = static_cast<long long>(kCal / (cal_ms * 1000.0));
	cudaDeviceProp prop{};
	CK(cudaGetDeviceProperties(&prop, dev));
	std::printf("[copy] device=%s spin_cycles_per_us=%lld items=%d limbs_per_item=%d limb_KiB=%zu\n", prop.name, cycles_per_us, kItems, kLimbs, kLimbBytes / 1024);
	{ // warm-up
		Run(Mem::Registered, Strm::NonBlocking, 0, cycles_per_us);
	}
	for (const Mem mem : { Mem::Pageable, Mem::Registered, Mem::PinnedAlloc })
		for (const double kernel_us : { 0.0, 50.0, 500.0 })
			for (const Strm sk : { Strm::Legacy, Strm::Blocking, Strm::NonBlocking }) {
				Result best{};
				best.wall_ms = 1e300;
				for (int rep = 0; rep < 3; ++rep) {
					const Result r = Run(mem, sk, kernel_us, cycles_per_us);
					if (r.wall_ms < best.wall_ms)
						best = r;
				}
				std::printf("[copy] mem=%-12s stream=%-11s pending_kernel_us=%-5g p50_us=%7.2f p90_us=%7.2f p99_us=%8.2f max_us=%8.2f copy_calls_ms=%7.2f wall_ms=%7.2f\n", Name(mem), Name(sk),
				  kernel_us, best.p50, best.p90, best.p99, best.max, best.copy_ms, best.wall_ms);
				std::fflush(stdout);
			}
	// More than one stream: item i goes to stream i % N, as a plaintext's limbs go to the plaintext's stream.
	for (const Mem mem : { Mem::Registered, Mem::PinnedAlloc })
		for (const double kernel_us : { 0.0, 50.0 })
			for (const int n : { 1, 4, 16, 37 }) {
				Result best{};
				best.wall_ms = 1e300;
				for (int rep = 0; rep < 3; ++rep) {
					const Result r = Run(mem, Strm::Blocking, kernel_us, cycles_per_us, n);
					if (r.wall_ms < best.wall_ms)
						best = r;
				}
				std::printf("[copy-n] mem=%-12s streams=%-3d pending_kernel_us=%-5g p50_us=%7.2f p90_us=%7.2f p99_us=%8.2f max_us=%8.2f copy_calls_ms=%7.2f wall_ms=%7.2f\n", Name(mem), n,
				  kernel_us, best.p50, best.p90, best.p99, best.max, best.copy_ms, best.wall_ms);
				std::fflush(stdout);
			}
	// Registered with huge pages, one stream.
	for (const double kernel_us : { 0.0, 50.0 }) {
		Result best{};
		best.wall_ms = 1e300;
		for (int rep = 0; rep < 3; ++rep) {
			const Result r = Run(Mem::RegisteredHuge, Strm::Blocking, kernel_us, cycles_per_us, 0);
			if (r.wall_ms < best.wall_ms)
				best = r;
		}
		std::printf("[copy] mem=%-12s stream=%-11s pending_kernel_us=%-5g p50_us=%7.2f p90_us=%7.2f p99_us=%8.2f max_us=%8.2f copy_calls_ms=%7.2f wall_ms=%7.2f\n", Name(Mem::RegisteredHuge), "blocking",
		  kernel_us, best.p50, best.p90, best.p99, best.max, best.copy_ms, best.wall_ms);
		std::fflush(stdout);
	}
	// The price of registering the host buffers, which the runs above did not include.
	for (const size_t mib : { size_t(256), size_t(1024), size_t(4096) }) {
		const size_t total = mib << 20;
		RegisterCost("normal, per limb", false, true, total);
		RegisterCost("normal, one block", false, false, total);
		RegisterCost("huge pages, per limb", true, true, total);
		RegisterCost("huge pages, one block", true, false, total);
		PinnedAllocCost(total);
	}
	return 0;
}
