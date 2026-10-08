// DEBUG BRANCH ONLY. Whether the stream-ordered allocator hands a block freed on one stream to an
// allocation on another stream before the device has reached the free, without the library.
//
// Every free is preceded, on its stream, by a kernel that writes the block's generation into a mark
// slot. Every allocation that takes over a freed block is followed, on its own stream, by a kernel that
// reads the previous owner's mark. A missing mark means the new owner's stream ran before the previous
// owner's free: the allocator handed the block over with no device ordering behind it. The check does
// not depend on any value being late or overwritten.
//
//   --case stale     one thread. The allocating stream waits on an event, and only then is the event
//                    re-recorded on a stream that is ordered after the free. The allocating stream
//                    never waited on that record. The control (stale-control) waits after the re-record.
//   --case overlap   two threads and one shared handle H. Thread A makes H wait on its free while
//                    thread B, at the same moment, records an event on H, makes its own stream wait on
//                    that event, and allocates. Whichever order the driver gives the two calls on H,
//                    the allocator must agree with the device. Run twice: reusing one event per thread
//                    (overlap) and with a fresh event every iteration (overlap-fresh).
//   --case chain     one thread, the overlap calls in fixed orders (no race possible): chain (H waited
//                    only on an older record of the event), late-wait (H waits on the new record after
//                    eH was recorded) and chain-control (before it; the only one where reuse is safe).
//   --case stress    several threads on a small set of shared blocking handles, with random fences
//                    through per-thread events, the way the library's concurrent lanes issue them.
//   --mode 0..3      the pool's reuse policies, as in pool_reuse_repro.cu (1: event dependencies only)
//
// A self-test runs first: a check that nothing orders after its mark must report it, and one behind
// an event wait must not. Prints "[markrepro]" lines; exits 1 when a case other than a control saw a
// violation, 2 when the self-test failed.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr unsigned long long kSlots = 1ull << 22;
constexpr int kOld					= 8;

struct Violation {
	unsigned long long gen, old_gen, seen;
};
struct Report {
	unsigned int count;
	unsigned int pad;
	Violation v[4];
};
struct CheckArgs {
	unsigned long long old[kOld];
	unsigned long long gen;
	int n;
};

__global__ void Mark(unsigned long long* marks, const unsigned long long gen) {
	__stcg(&marks[gen % kSlots], gen);
}

__global__ void Check(const unsigned long long* marks, const CheckArgs a, Report* r) {
	for (int i = 0; i < a.n; ++i) {
		const unsigned long long m = __ldcg(&marks[a.old[i] % kSlots]);
		if (m >= a.old[i])
			continue;
		const unsigned int k = atomicAdd(&r->count, 1u);
		if (k < 4) {
			atomicExch(&r->v[k].old_gen, a.old[i]);
			atomicExch(&r->v[k].seen, m);
			atomicExch(&r->v[k].gen, a.gen);
		}
	}
}

/// Spins until *flag is non-zero or `ns` have passed (flag may be null).
__global__ void Spin(const volatile unsigned int* flag, const unsigned long long ns) {
	unsigned long long start, now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do {
		if (flag != nullptr && *flag != 0)
			return;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	} while (now - start < ns);
}

__global__ void SetFlag(unsigned int* flag) {
	atomicExch(flag, 1u);
	__threadfence_system();
}

void Fail(const char* what, const cudaError_t e) {
	std::printf("[markrepro] %s: %s\n", what, cudaGetErrorString(e));
	std::fflush(stdout);
	std::exit(3);
}

void Ok(const char* what, const cudaError_t e) {
	if (e != cudaSuccess)
		Fail(what, e);
}

/// Marks, the report, and which freed blocks no allocation has taken yet.
struct Tracker {
	unsigned long long* marks = nullptr;
	Report* host			  = nullptr;
	Report* dev				  = nullptr;
	std::atomic<unsigned long long> next_gen{ 16 };
	std::mutex mu;
	struct Freed {
		unsigned long long gen;
		size_t bytes;
		cudaStream_t stream;
	};
	std::map<unsigned long long, Freed> freed;
	size_t max_bytes = 0;
	std::atomic<unsigned long long> checks{ 0 }, cross{ 0 };

	void Reset() {
		std::lock_guard<std::mutex> lk(mu);
		freed.clear();
		checks = 0;
		cross  = 0;
		std::memset(host, 0, sizeof(Report));
	}

	/// After cudaMallocAsync returned p on s: check, on s, the marks of every freed block it overlaps.
	/// Returns how many of them were freed on another stream.
	int Took(void* p, const size_t bytes, const cudaStream_t s) {
		CheckArgs a{};
		a.gen	   = next_gen.fetch_add(1);
		int other  = 0;
		{
			std::lock_guard<std::mutex> lk(mu);
			const auto lo = reinterpret_cast<unsigned long long>(p), hi = lo + bytes;
			auto it		  = freed.lower_bound(hi);
			while (it != freed.begin()) {
				--it;
				if (it->first + it->second.bytes <= lo) {
					if (it->first + max_bytes <= lo)
						break;
					continue;
				}
				if (a.n < kOld)
					a.old[a.n++] = it->second.gen;
				other += it->second.stream != s;
				it	  = freed.erase(it);
			}
		}
		if (a.n > 0) {
			Check<<<1, 1, 0, s>>>(marks, a, dev);
			// A check that failed to launch would hide a violation; a mark that failed would invent one.
			Ok("Check launch", cudaGetLastError());
			checks.fetch_add(1);
			if (other > 0)
				cross.fetch_add(1);
		}
		return other;
	}

	/// Marks p on s and frees it there. The block is recorded as freed before the free, so no
	/// allocation can take it untracked.
	void Free(void* p, const size_t bytes, const cudaStream_t s) {
		const unsigned long long gen = next_gen.fetch_add(1);
		Mark<<<1, 1, 0, s>>>(marks, gen);
		Ok("Mark launch", cudaGetLastError());
		{
			std::lock_guard<std::mutex> lk(mu);
			freed[reinterpret_cast<unsigned long long>(p)] = { gen, bytes, s };
			max_bytes									   = std::max(max_bytes, bytes);
		}
		Ok("cudaFreeAsync", cudaFreeAsync(p, s));
	}
};

Tracker T;

/// `reused`: stale, the allocation got the block just freed on the other stream; overlap, it took a
/// block freed on another stream.
void PrintResult(const char* name, const int mode, const int threads, const unsigned long long iters, const unsigned long long reused) {
	Ok("cudaDeviceSynchronize", cudaDeviceSynchronize());
	const unsigned int n = *static_cast<volatile unsigned int*>(&T.host->count);
	std::printf("[markrepro] case=%s mode=%d threads=%d iters=%llu checks=%llu cross-stream=%llu reused=%llu violations=%u", name, mode, threads, iters,
	  T.checks.load(), T.cross.load(), reused, n);
	if (n != 0)
		std::printf(" first: allocation gen %llu ran before the free of gen %llu (slot held %llu)", T.host->v[0].gen, T.host->v[0].old_gen,
		  T.host->v[0].seen);
	std::printf("\n");
	std::fflush(stdout);
}

bool SelfTest() {
	cudaStream_t a = nullptr, b = nullptr;
	cudaEvent_t e  = nullptr;
	unsigned int* flag = nullptr;
	Ok("stream", cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
	Ok("stream", cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
	Ok("event", cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
	Ok("cudaMalloc", cudaMalloc(reinterpret_cast<void**>(&flag), sizeof(unsigned int)));
	Ok("cudaMemset", cudaMemset(flag, 0, sizeof(unsigned int)));
	T.Reset();
	CheckArgs c{};
	c.n		 = 1;
	c.old[0] = 1;
	c.gen	 = 2;
	Spin<<<1, 1, 0, a>>>(flag, 2'000'000'000);
	Mark<<<1, 1, 0, a>>>(T.marks, 1);
	Check<<<1, 1, 0, b>>>(T.marks, c, T.dev);
	SetFlag<<<1, 1, 0, b>>>(flag);
	Spin<<<1, 1, 0, a>>>(nullptr, 20'000'000);
	Mark<<<1, 1, 0, a>>>(T.marks, 3);
	Ok("record", cudaEventRecord(e, a));
	Ok("wait", cudaStreamWaitEvent(b, e, 0));
	c.old[0] = 3;
	c.gen	 = 4;
	Check<<<1, 1, 0, b>>>(T.marks, c, T.dev);
	Ok("cudaDeviceSynchronize", cudaDeviceSynchronize());
	const bool ok = T.host->count == 1 && T.host->v[0].gen == 2 && T.host->v[0].old_gen == 1;
	std::printf("[markrepro] self-test %s: unordered check reported %u violation(s), first gen %llu old %llu (want 1, 2, 1)\n", ok ? "PASSED" : "FAILED",
	  T.host->count, T.host->v[0].gen, T.host->v[0].old_gen);
	std::fflush(stdout);
	cudaFree(flag);
	cudaEventDestroy(e);
	cudaStreamDestroy(a);
	cudaStreamDestroy(b);
	return ok;
}

/// One thread. N waits on E; E is then re-recorded on S after S waited on O's free; then N allocates.
/// With event dependencies as the only cross-stream reuse, N may not get O's block: it never waited on
/// the record that follows the free. The control makes N wait after the re-record, so N may get it.
void StaleCase(const int mode, const unsigned long long iters, const bool control) {
	cudaStream_t O, N, S;
	cudaEvent_t E, eO;
	for (cudaStream_t* s : { &O, &N, &S })
		Ok("stream", cudaStreamCreateWithFlags(s, cudaStreamDefault));
	Ok("event", cudaEventCreateWithFlags(&E, cudaEventDisableTiming));
	Ok("event", cudaEventCreateWithFlags(&eO, cudaEventDisableTiming));
	T.Reset();
	unsigned long long same = 0;
	// N keeps its blocks until the end: a block it freed itself would always be reusable on N, and
	// the only block left to take must be the one O just freed.
	std::vector<void*> kept;
	for (unsigned long long i = 0; i < iters; ++i) {
		Ok("record", cudaEventRecord(E, S));
		if (!control)
			Ok("wait", cudaStreamWaitEvent(N, E, 0));
		void* p = nullptr;
		Ok("cudaMallocAsync", cudaMallocAsync(&p, 64, O));
		T.Took(p, 64, O);
		Spin<<<1, 1, 0, O>>>(nullptr, 2'000'000);
		Ok("Spin launch", cudaGetLastError());
		T.Free(p, 64, O);
		Ok("record", cudaEventRecord(eO, O));
		Ok("wait", cudaStreamWaitEvent(S, eO, 0));
		Ok("record", cudaEventRecord(E, S));
		if (control)
			Ok("wait", cudaStreamWaitEvent(N, E, 0));
		void* q = nullptr;
		Ok("cudaMallocAsync", cudaMallocAsync(&q, 64, N));
		same += q == p;
		T.Took(q, 64, N);
		kept.push_back(q);
		Ok("sync", cudaDeviceSynchronize());
	}
	PrintResult(control ? "stale-control" : "stale", mode, 1, iters, same);
	for (void* q : kept)
		cudaFreeAsync(q, N);
	Ok("sync", cudaDeviceSynchronize());
	for (cudaStream_t s : { O, N, S })
		cudaStreamDestroy(s);
	cudaEventDestroy(E);
	cudaEventDestroy(eO);
}

/// A spin barrier for two threads: the two calls on H must land within a microsecond or so of each
/// other, which a sleeping wait cannot do. Bounded, so a stuck partner ends the case instead of hanging.
struct Barrier2 {
	std::atomic<unsigned long long> arrived{ 0 };
	bool Wait(const unsigned long long round) {
		arrived.fetch_add(1);
		const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (arrived.load() < 2 * round) {
			if (std::chrono::steady_clock::now() > end)
				return false;
		}
		return true;
	}
};

void BusyNs(const long long ns) {
	const auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
	while (std::chrono::steady_clock::now() < end) {
	}
}

/// Two threads, one shared handle H. A: allocate on O, spin, mark, free, record eO on O; then, at
/// the barrier, H waits on eO. B: at the barrier, records eH on H, N waits on eH, N allocates.
///
/// With `fresh`, every iteration records into an event that was not recorded in the last kRing
/// iterations. With one event re-recorded every iteration, a pool that wrongly followed an event's
/// newest record (instead of the record a wait captured) would fail the same way as one that loses a
/// race between the two threads' calls; with fresh events only the race remains.
void OverlapCase(const int mode, const unsigned long long iters, const bool fresh) {
	constexpr size_t kRing = 8192;
	cudaStream_t O, H, N;
	for (cudaStream_t* s : { &O, &H, &N })
		Ok("stream", cudaStreamCreateWithFlags(s, cudaStreamDefault));
	std::vector<cudaEvent_t> eOs(fresh ? kRing : 1), eHs(fresh ? kRing : 1);
	for (std::vector<cudaEvent_t>* v : { &eOs, &eHs })
		for (cudaEvent_t& e : *v)
			Ok("event", cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
	T.Reset();
	Barrier2 bar;
	std::atomic<unsigned long long> same{ 0 };
	std::atomic<bool> stop{ false };
	std::vector<void*> kept; // N keeps its blocks until the end, as in StaleCase
	std::thread a([&] {
		std::mt19937_64 rng(1);
		for (unsigned long long i = 0; i < iters && !stop; ++i) {
			void* p = nullptr;
			Ok("cudaMallocAsync", cudaMallocAsync(&p, 64, O));
			T.Took(p, 64, O);
			Spin<<<1, 1, 0, O>>>(nullptr, 20'000 + rng() % 200'000);
			Ok("Spin launch", cudaGetLastError());
			T.Free(p, 64, O);
			const cudaEvent_t eO = eOs[i % eOs.size()];
			Ok("record", cudaEventRecord(eO, O));
			if (!bar.Wait(2 * i + 1)) {
				stop = true;
				break;
			}
			BusyNs(static_cast<long long>(rng() % 3000));
			Ok("wait", cudaStreamWaitEvent(H, eO, 0));
			if (!bar.Wait(2 * i + 2)) {
				stop = true;
				break;
			}
		}
	});
	std::thread b([&] {
		std::mt19937_64 rng(2);
		for (unsigned long long i = 0; i < iters && !stop; ++i) {
			if (!bar.Wait(2 * i + 1)) {
				stop = true;
				break;
			}
			BusyNs(static_cast<long long>(rng() % 3000));
			const cudaEvent_t eH = eHs[i % eHs.size()];
			Ok("record", cudaEventRecord(eH, H));
			Ok("wait", cudaStreamWaitEvent(N, eH, 0));
			void* q = nullptr;
			Ok("cudaMallocAsync", cudaMallocAsync(&q, 64, N));
			if (T.Took(q, 64, N) > 0)
				same.fetch_add(1);
			kept.push_back(q);
			if (!bar.Wait(2 * i + 2)) {
				stop = true;
				break;
			}
			if ((i & 63) == 63)
				Ok("sync", cudaStreamSynchronize(N));
		}
	});
	a.join();
	b.join();
	if (stop)
		std::printf("[markrepro] overlap: a thread timed out at the barrier\n");
	PrintResult(fresh ? "overlap-fresh" : "overlap", mode, 2, iters, same.load());
	for (void* q : kept)
		cudaFreeAsync(q, N);
	Ok("sync", cudaDeviceSynchronize());
	for (cudaStream_t s : { O, H, N })
		cudaStreamDestroy(s);
	for (std::vector<cudaEvent_t>* v : { &eOs, &eHs })
		for (cudaEvent_t e : *v)
			cudaEventDestroy(e);
}

/// One thread, the overlap case's calls in a fixed order, so no race between threads is possible.
/// H waits on an earlier record of eO; O allocates, spins, marks and frees p and records eO again;
/// then H's event eH is recorded, and N waits on eH and allocates.
///   chain:     nothing else. eH captures only H's wait on the OLD record of eO, so N is not ordered
///              after p's free and may not get p. A pool that followed eO's NEWEST record would.
///   late-wait: H also waits on the new record of eO, but after eH was recorded and before N waits
///              and allocates. N is still not ordered after the free. A pool that read H's state at
///              the allocation instead of the state eH captured would hand N the block.
///   control:   H waits on the new record BEFORE eH is recorded: N is ordered after the free, the pool
///              may hand it p, and the check must pass.
void ChainCase(const int mode, const unsigned long long iters, const char* variant) {
	const std::string v = variant;
	cudaStream_t O, H, N;
	cudaEvent_t eO, eH;
	for (cudaStream_t* s : { &O, &H, &N })
		Ok("stream", cudaStreamCreateWithFlags(s, cudaStreamDefault));
	Ok("event", cudaEventCreateWithFlags(&eO, cudaEventDisableTiming));
	Ok("event", cudaEventCreateWithFlags(&eH, cudaEventDisableTiming));
	T.Reset();
	unsigned long long same = 0;
	std::vector<void*> kept; // N keeps its blocks, as in StaleCase
	for (unsigned long long i = 0; i < iters; ++i) {
		Ok("record", cudaEventRecord(eO, O));
		Ok("wait", cudaStreamWaitEvent(H, eO, 0)); // H depends on this (old) record of eO
		void* p = nullptr;
		Ok("cudaMallocAsync", cudaMallocAsync(&p, 64, O));
		T.Took(p, 64, O);
		Spin<<<1, 1, 0, O>>>(nullptr, 2'000'000);
		Ok("Spin launch", cudaGetLastError());
		T.Free(p, 64, O);
		Ok("record", cudaEventRecord(eO, O)); // eO now stands after the free
		if (v == "control")
			Ok("wait", cudaStreamWaitEvent(H, eO, 0));
		Ok("record", cudaEventRecord(eH, H));
		if (v == "late-wait")
			Ok("wait", cudaStreamWaitEvent(H, eO, 0));
		Ok("wait", cudaStreamWaitEvent(N, eH, 0));
		void* q = nullptr;
		Ok("cudaMallocAsync", cudaMallocAsync(&q, 64, N));
		same += q == p;
		T.Took(q, 64, N);
		kept.push_back(q);
		Ok("sync", cudaDeviceSynchronize());
	}
	PrintResult(("chain-" + v).c_str(), mode, 1, iters, same);
	for (void* q : kept)
		cudaFreeAsync(q, N);
	Ok("sync", cudaDeviceSynchronize());
	for (cudaStream_t s : { O, H, N })
		cudaStreamDestroy(s);
	cudaEventDestroy(eO);
	cudaEventDestroy(eH);
}

/// Several threads on shared handles: allocate on a random handle, hold the device for a random while,
/// free there, and now and then fence two handles through a per-thread event, re-recorded every time.
void StressCase(const int mode, const int threads, const int nhandles, const double seconds) {
	std::vector<cudaStream_t> handles(static_cast<size_t>(nhandles));
	for (cudaStream_t& h : handles)
		Ok("stream", cudaStreamCreateWithFlags(&h, cudaStreamDefault));
	T.Reset();
	std::atomic<unsigned long long> ops{ 0 };
	std::vector<std::thread> workers;
	for (int t = 0; t < threads; ++t)
		workers.emplace_back([&, t] {
			std::mt19937_64 rng(static_cast<unsigned long long>(t) * 7919u + 3u);
			// One event per handle per thread, like the library's fence event per stream and lane.
			std::vector<cudaEvent_t> ev(handles.size());
			for (cudaEvent_t& e : ev)
				Ok("event", cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
			const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
			unsigned long long i = 0;
			for (; (i & 255) != 0 || std::chrono::steady_clock::now() < end; ++i) {
				const size_t si		= rng() % handles.size();
				const cudaStream_t s = handles[si];
				const size_t bytes	= 16 + 8 * (rng() % 31);
				void* p				= nullptr;
				Ok("cudaMallocAsync", cudaMallocAsync(&p, bytes, s));
				T.Took(p, bytes, s);
				Spin<<<1, 1, 0, s>>>(nullptr, rng() % 50'000);
				Ok("Spin launch", cudaGetLastError());
				T.Free(p, bytes, s);
				for (int f = static_cast<int>(rng() % 3); f > 0; --f) {
					const size_t x = rng() % handles.size(), y = rng() % handles.size();
					if (x == y)
						continue;
					Ok("record", cudaEventRecord(ev[x], handles[x]));
					Ok("wait", cudaStreamWaitEvent(handles[y], ev[x], 0));
				}
			}
			Ok("sync", cudaDeviceSynchronize());
			for (cudaEvent_t e : ev)
				cudaEventDestroy(e);
			ops.fetch_add(i);
		});
	for (std::thread& w : workers)
		w.join();
	PrintResult("stress", mode, threads, ops.load(), 0);
	for (cudaStream_t h : handles)
		cudaStreamDestroy(h);
}

} // namespace

int main(int argc, char** argv) {
	// Load every kernel at start: a lazy first load may synchronize the context and hide an ordering.
	setenv("CUDA_MODULE_LOADING", "EAGER", 1);
	std::string which = "all";
	int mode = 1, threads = 6, nhandles = 8;
	unsigned long long iters = 2000;
	double seconds			 = 60;
	for (int a = 1; a + 1 < argc; a += 2) {
		const std::string k = argv[a];
		if (k == "--case")
			which = argv[a + 1];
		else if (k == "--mode")
			mode = std::atoi(argv[a + 1]);
		else if (k == "--threads")
			threads = std::atoi(argv[a + 1]);
		else if (k == "--handles")
			nhandles = std::atoi(argv[a + 1]);
		else if (k == "--iters")
			iters = std::strtoull(argv[a + 1], nullptr, 10);
		else if (k == "--seconds")
			seconds = std::atof(argv[a + 1]);
	}
	if (mode < 0 || mode > 3 || threads < 1 || nhandles < 2) {
		std::printf("[markrepro] bad arguments\n");
		return 3;
	}

	cudaMemPool_t pool = nullptr;
	Ok("cudaDeviceGetDefaultMemPool", cudaDeviceGetDefaultMemPool(&pool, 0));
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
	int driver = 0, runtime = 0;
	cudaDriverGetVersion(&driver);
	cudaRuntimeGetVersion(&runtime);
	std::printf("[markrepro] driver API %d, runtime %d, pool mode %d (internal=%d opportunistic=%d follow-events=%d)\n", driver, runtime, mode, internal,
	  opportunistic, follow);

	Ok("cudaMalloc", cudaMalloc(reinterpret_cast<void**>(&T.marks), kSlots * 8));
	Ok("cudaMemset", cudaMemset(T.marks, 0, kSlots * 8));
	void* h = nullptr;
	void* d = nullptr;
	Ok("cudaHostAlloc", cudaHostAlloc(&h, sizeof(Report), cudaHostAllocMapped | cudaHostAllocPortable));
	Ok("cudaHostGetDevicePointer", cudaHostGetDevicePointer(&d, h, 0));
	T.host = static_cast<Report*>(h);
	T.dev  = static_cast<Report*>(d);
	std::memset(h, 0, sizeof(Report));
	if (!SelfTest())
		return 2;

	bool bad = false;
	const auto ran = [&] { bad = bad || T.host->count != 0; };
	if (which == "all" || which == "stale") {
		StaleCase(mode, iters, false);
		ran();
		StaleCase(mode, iters, true);
	}
	if (which == "all" || which == "chain") {
		ChainCase(mode, iters, "chain");
		ran();
		ChainCase(mode, iters, "late-wait");
		ran();
		ChainCase(mode, iters, "control");
	}
	if (which == "all" || which == "overlap") {
		OverlapCase(mode, iters * 100, false);
		ran();
		OverlapCase(mode, iters * 100, true);
		ran();
	}
	if (which == "all" || which == "stress") {
		StressCase(mode, threads, nhandles, seconds);
		ran();
	}
	return bad ? 1 : 0;
}
