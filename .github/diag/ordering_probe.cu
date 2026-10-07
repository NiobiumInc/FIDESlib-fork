// DEBUG BRANCH ONLY. Deterministic checks of the stream orderings a per-op table relies on, one
// question per case, without the library.
//
// Every iteration starts with an idle device. A producer stream X spins for a few milliseconds and
// then writes tag X into a buffer; an event recorded on X after that write is what a consumer
// stream Y waits on before it writes tag Y into the same memory. When the device is idle again the
// host reads the buffer back: tag Y means Y's write landed last, as the wait requires; tag X means X's
// write landed after Y's, i.e. Y did not wait.
//
//   nowait           Y does not wait. With writer=kernel it MUST show late X writes: that proves the
//                    probe can see one. With writer=pageable it only says whether the copy held the host.
//   wait             a plain cudaStreamWaitEvent(Y, E), the buffer from cudaMalloc.
//   pool             X allocates the buffer from the device default pool and frees it right after its
//                    write; Y waits, then allocates (the pool may hand it the same block), writes, and
//                    a kernel on Y reads the block back. Pool modes as in pool_reuse_repro.cu.
//   pool-2thr        the same, with the record of E on X and everything on Y issued by a second
//                    host thread, strictly after the first thread's free returned.
//
// The producer writes with a kernel or with a pageable cudaMemcpyAsync (writer=), the consumer with a
// pageable cudaMemcpyAsync, a pinned one or a kernel (consumer=).
//
// One "[probe]" line per case. Exit 1 when a case other than nowait saw a late X write, or when
// nowait saw none (the probe would then prove nothing); 0 otherwise.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

__device__ unsigned long long Now() {
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

/// Keeps the stream busy for `ns` nanoseconds, so whatever is queued behind it runs late.
__global__ void Spin(const unsigned long long ns) {
	const unsigned long long start = Now();
	while (Now() - start < ns) {
	}
}

__global__ void Fill(unsigned long long* p, const int n, const unsigned long long tag) {
	for (int i = 0; i < n; ++i)
		p[i] = tag;
}

/// Copies the block's first word to mapped host memory: what a consumer kernel on Y would read.
__global__ void Peek(const unsigned long long* p, unsigned long long* out) {
	*out = p[0];
}

void Check(const cudaError_t e, const char* what) {
	if (e != cudaSuccess) {
		std::printf("[probe] %s: %s\n", what, cudaGetErrorString(e));
		std::fflush(stdout);
		std::exit(2);
	}
}

enum class Writer { kKernel, kPageable, kPinned };

const char* Name(const Writer w) {
	return w == Writer::kKernel ? "kernel" : w == Writer::kPageable ? "pageable" : "pinned";
}

struct Probe {
	cudaStream_t x = nullptr, y = nullptr;
	cudaEvent_t e  = nullptr;
	unsigned long long* pinned = nullptr; ///< pinned source for consumer=pinned
	unsigned long long* peek   = nullptr; ///< mapped: what Peek saw
	unsigned long long* peek_d = nullptr;
	unsigned long long spin_ns = 5'000'000;
	int words				   = 17; ///< 136 bytes, the size of the table in the failing report
};

void Write(const Probe& pr, const Writer w, unsigned long long* dst, const unsigned long long tag, const cudaStream_t s) {
	const size_t bytes = static_cast<size_t>(pr.words) * 8;
	if (w == Writer::kKernel) {
		Fill<<<1, 1, 0, s>>>(dst, pr.words, tag);
	} else if (w == Writer::kPinned) {
		for (int i = 0; i < pr.words; ++i)
			pr.pinned[i] = tag;
		Check(cudaMemcpyAsync(dst, pr.pinned, bytes, cudaMemcpyHostToDevice, s), "pinned copy");
	} else {
		const std::vector<unsigned long long> host(static_cast<size_t>(pr.words), tag);
		Check(cudaMemcpyAsync(dst, host.data(), bytes, cudaMemcpyHostToDevice, s), "pageable copy");
	}
}

struct Tally {
	int ok = 0, late = 0, other = 0, reused = 0;
	void Add(const unsigned long long seen, const unsigned long long tag_x, const unsigned long long tag_y) {
		if (seen == tag_y)
			++ok;
		else if (seen == tag_x)
			++late;
		else
			++other;
	}
};

void SetPoolMode(const int mode) {
	cudaMemPool_t pool = nullptr;
	Check(cudaDeviceGetDefaultMemPool(&pool, 0), "default pool");
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
}

/// nowait / wait: a plain buffer, no allocator.
Tally Plain(const Probe& pr, const bool wait, const Writer writer, const Writer consumer, const int iters) {
	Tally t;
	unsigned long long* buf = nullptr;
	Check(cudaMalloc(&buf, static_cast<size_t>(pr.words) * 8), "cudaMalloc");
	for (int i = 0; i < iters; ++i) {
		const unsigned long long tag_x = (1ull << 56) | static_cast<unsigned>(i), tag_y = (2ull << 56) | static_cast<unsigned>(i);
		Check(cudaDeviceSynchronize(), "sync");
		Spin<<<1, 1, 0, pr.x>>>(pr.spin_ns);
		Write(pr, writer, buf, tag_x, pr.x);
		Check(cudaEventRecord(pr.e, pr.x), "record");
		if (wait)
			Check(cudaStreamWaitEvent(pr.y, pr.e, 0), "wait");
		Write(pr, consumer, buf, tag_y, pr.y);
		Check(cudaDeviceSynchronize(), "sync");
		unsigned long long seen = 0;
		Check(cudaMemcpy(&seen, buf, 8, cudaMemcpyDeviceToHost), "read back");
		t.Add(seen, tag_x, tag_y);
	}
	cudaFree(buf);
	return t;
}

/// pool / pool-2thr: X allocates, writes late and frees; Y waits, allocates, writes and reads back.
Tally Pool(const Probe& pr, const bool two_threads, const Writer writer, const Writer consumer, const int iters) {
	Tally t;
	const size_t bytes = static_cast<size_t>(pr.words) * 8;
	for (int i = 0; i < iters; ++i) {
		const unsigned long long tag_x = (1ull << 56) | static_cast<unsigned>(i), tag_y = (2ull << 56) | static_cast<unsigned>(i);
		Check(cudaDeviceSynchronize(), "sync");
		unsigned long long* p = nullptr;
		const auto producer = [&] {
			Check(cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, pr.x), "cudaMallocAsync X");
			Spin<<<1, 1, 0, pr.x>>>(pr.spin_ns);
			Write(pr, writer, p, tag_x, pr.x);
			Check(cudaFreeAsync(p, pr.x), "cudaFreeAsync X");
		};
		unsigned long long* q = nullptr;
		const auto consumer_side = [&] {
			Check(cudaEventRecord(pr.e, pr.x), "record");
			Check(cudaStreamWaitEvent(pr.y, pr.e, 0), "wait");
			Check(cudaMallocAsync(reinterpret_cast<void**>(&q), bytes, pr.y), "cudaMallocAsync Y");
			Write(pr, consumer, q, tag_y, pr.y);
			Peek<<<1, 1, 0, pr.y>>>(q, pr.peek_d);
			Check(cudaFreeAsync(q, pr.y), "cudaFreeAsync Y");
		};
		if (two_threads) {
			std::atomic<bool> freed{ false };
			std::thread a([&] {
				producer();
				freed.store(true, std::memory_order_release);
			});
			std::thread b([&] {
				while (!freed.load(std::memory_order_acquire))
					std::this_thread::yield();
				consumer_side();
			});
			a.join();
			b.join();
		} else {
			producer();
			consumer_side();
		}
		Check(cudaDeviceSynchronize(), "sync");
		if (q == p)
			++t.reused;
		t.Add(*pr.peek, tag_x, tag_y);
	}
	return t;
}

enum class Expect { kLate, kAny, kInOrder };

int Report(const char* name, const int mode, const Writer writer, const Writer consumer, const int iters, const Tally& t, const Expect expect) {
	std::printf("[probe] case=%-9s pool-mode=%s writer=%-8s consumer=%-8s iters=%d same-block=%d ok=%d late=%d other=%d", name,
	  mode < 0 ? "-" : std::to_string(mode).c_str(), Name(writer), Name(consumer), iters, t.reused, t.ok, t.late, t.other);
	int bad = 0;
	if (expect == Expect::kLate && t.late == 0) {
		std::printf("  INSTRUMENT FAILED: no late write seen without a wait");
		bad = 1;
	} else if (expect == Expect::kInOrder && (t.late != 0 || t.other != 0)) {
		std::printf("  ORDER VIOLATED");
		bad = 1;
	}
	std::printf("\n");
	std::fflush(stdout);
	return bad;
}

} // namespace

int main(int argc, char** argv) {
	int iters = 100;
	Probe pr;
	for (int a = 1; a + 1 < argc; a += 2) {
		const std::string k = argv[a];
		if (k == "--iters")
			iters = std::atoi(argv[a + 1]);
		else if (k == "--spin-ms")
			pr.spin_ns = static_cast<unsigned long long>(std::atof(argv[a + 1]) * 1e6);
		else if (k == "--words")
			pr.words = std::atoi(argv[a + 1]);
	}
	if (iters < 1 || pr.words < 1 || pr.words > 4096) {
		std::printf("[probe] bad arguments\n");
		return 2;
	}
	Check(cudaSetDevice(0), "cudaSetDevice");
	// Blocking streams, like the library's pool of shared handles.
	Check(cudaStreamCreateWithFlags(&pr.x, cudaStreamDefault), "stream X");
	Check(cudaStreamCreateWithFlags(&pr.y, cudaStreamDefault), "stream Y");
	Check(cudaEventCreateWithFlags(&pr.e, cudaEventDisableTiming), "event");
	Check(cudaHostAlloc(&pr.pinned, static_cast<size_t>(pr.words) * 8, cudaHostAllocDefault), "pinned source");
	Check(cudaHostAlloc(&pr.peek, 8, cudaHostAllocMapped), "peek");
	Check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&pr.peek_d), pr.peek, 0), "peek device pointer");

	int bad = 0;
	for (const Writer w : { Writer::kKernel, Writer::kPageable })
		bad += Report("nowait", -1, w, Writer::kPageable, iters, Plain(pr, false, w, Writer::kPageable, iters),
		  w == Writer::kKernel ? Expect::kLate : Expect::kAny);
	for (const Writer w : { Writer::kKernel, Writer::kPageable })
		for (const Writer c : { Writer::kPageable, Writer::kPinned, Writer::kKernel })
			bad += Report("wait", -1, w, c, iters, Plain(pr, true, w, c, iters), Expect::kInOrder);
	for (const int mode : { 1, 0, 3, 2 }) {
		SetPoolMode(mode);
		for (const bool two : { false, true })
			for (const Writer w : { Writer::kKernel, Writer::kPageable })
				for (const Writer c : { Writer::kPageable, Writer::kKernel })
					bad += Report(two ? "pool-2thr" : "pool", mode, w, c, iters, Pool(pr, two, w, c, iters), Expect::kInOrder);
	}
	std::printf("[probe] done: %s\n", bad == 0 ? "every ordering held and the instrument saw a late write" : "see the lines above");
	std::fflush(stdout);
	return bad == 0 ? 0 : 1;
}
