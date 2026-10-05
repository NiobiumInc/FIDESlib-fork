// Device memory pool: concurrency and cross-stream correctness.
//
// The per-stream pool replaced the pool's ONE free list + ONE shared event per device with per-stream free lists and
// a per-block handoff edge (see the block comment above PoolFor in src/CudaUtils.cu). That trades a
// sledgehammer ordering guarantee -- every malloc behind every free on the device -- for a precise
// one, and a precise guarantee is the kind that fails silently: a block handed to stream B while
// stream A's kernels are still reading it produces WRONG NUMBERS, not a crash, and only under
// contention that a single-threaded test never creates.
//
// So this suite does not test the allocator's bookkeeping. It tests the ONE property the pool
// actually owes its callers:
//
//     a block handed out must not be writable by its new owner before its previous owner's
//     work on that block has completed.
//
// The shape of every case below is the same, and is chosen so a violation cannot hide:
//
//   1. take a block;
//   2. FILL it with a tag unique to this (thread, iteration), on our stream;
//   3. SPIN on our stream, long enough that a racing stream would be inside our window;
//   4. CHECK every word still carries our tag, on our stream, counting mismatches into device
//      memory (a failure is a count, not a fault, so the run continues and reports the total);
//   5. give it back.
//
// If the pool ever hands a live block to another stream, step 4 of one thread and step 2 of another
// overlap on the same bytes and the counter moves. The spin in step 3 is what makes the window wide
// enough for that to be likely rather than theoretical.
//
// The suite deliberately covers both pooled size classes the library uses: a SMALL one that
// GPUmalloc rounds up and force-caches (< 64 KB), and a LARGE power-of-two one (a limb at
// logN = 16). It also forces the CROSS-STREAM HANDOFF path specifically -- the only path that
// records an event at all -- by freeing everything on one stream and then allocating it all from
// others, which is a pattern the random mix reaches only by luck.
//
// CUDA-ONLY BY CONSTRUCTION: a .cu file, and the root CMakeLists globs .cu sources only into
// fideslib-test, which is built only when FIDESLIB_ENABLE_CUDA is ON. Every case skips at runtime
// when no device is present.
//
// NOT RUNNABLE ON A LAPTOP: it needs a GPU. Run it on the runner with the rest of fideslib-test,
// and run it BOTH ways so a failure can be attributed:
//
//   * FIDESLIB_POOL_PER_STREAM=1 -- the arm, the per-stream lists these cases are written for;
//   * unset (or 0) -- the legacy single-event pool, which is this branch's DEFAULT and the A/B
//     control. Every case below must pass in this mode too; passing here proves only that the
//     harness is sound, not that the per-stream pool is.
//
// The env var is read ONCE into a function-local static inside the library, so it has to be set in
// the environment of the whole test binary -- a setenv from inside a TEST body is too late and
// would silently measure whichever mode the first allocation happened to latch.
//
// FIDESLIB_CONCURRENT_OPS MUST BE UNSET for either mode to mean anything. GPUmalloc/GPUfree test
// the concurrent per-lane branch FIRST and return from it, so with CONCURRENT_OPS set the
// per-stream pool is never reached and both modes above exercise the same (lane) code; the library
// says so on stderr. The lane pool has its own coverage in ConcurrentOpsTests.cu.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "CudaUtils.cuh"

namespace FIDESlib::Testing {

namespace {

constexpr int kDevice = 0;

/// Words per block for the large class: one limb at logN = 16 (512 KB), the size the device-encode
/// and limb paths actually allocate.
constexpr size_t kLargeWords = 1u << 16;
/// And a small class, which GPUmalloc rounds up to a power of two and force-caches.
constexpr size_t kSmallWords = 512; // 4 KB

__global__ void pool_fill_(uint64_t* __restrict__ p, const size_t n, const uint64_t tag) {
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		p[i] = tag ^ (uint64_t)i;
}

/// Occupy the stream long enough that a racing stream would land inside our window. Deliberately a
/// real kernel rather than a host sleep: the hazard is DEVICE-side overlap, so the delay has to be
/// on the device timeline.
__global__ void pool_spin_(const uint64_t iters, uint64_t* __restrict__ sink) {
	uint64_t x = 1;
	for (uint64_t i = 0; i < iters; ++i)
		x = x * 6364136223846793005ull + 1442695040888963407ull;
	if (threadIdx.x == 0 && blockIdx.x == 0 && x == 0)
		*sink = x; // never taken; keeps the loop from being optimised away
}

__global__ void pool_check_(const uint64_t* __restrict__ p, const size_t n, const uint64_t tag, int* __restrict__ errors) {
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && p[i] != (tag ^ (uint64_t)i))
		atomicAdd(errors, 1);
}

bool HaveDevice() {
	int n = 0;
	return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

/// One thread's work: `iters` rounds of take / fill / spin / check / give back on ONE stream.
///
/// A null return is counted into `nulls` rather than asserted: gtest's assertion macros are not
/// safe to fire from a non-main thread, and the whole point of this suite is to run on many.
void Churn(cudaStream_t stream, size_t words, int iters, uint64_t seed, int* d_errors, uint64_t* d_sink, std::atomic<int>* nulls) {
	cudaSetDevice(kDevice);
	const int bytes	 = (int)(words * sizeof(uint64_t));
	const dim3 block{ 256 };
	const dim3 grid{ (uint32_t)((words + 255) / 256) };

	for (int i = 0; i < iters; ++i) {
		auto* p = (uint64_t*)GPUmalloc(kDevice, bytes, stream, true);
		if (p == nullptr) {
			nulls->fetch_add(1);
			continue;
		}

		const uint64_t tag = (seed << 20) ^ (uint64_t)i ^ 0x9e3779b97f4a7c15ull;
		pool_fill_<<<grid, block, 0, stream>>>(p, words, tag);
		pool_spin_<<<1, 32, 0, stream>>>(20000, d_sink);
		pool_check_<<<grid, block, 0, stream>>>(p, words, tag, d_errors);

		GPUfree(p, kDevice, bytes, stream, true);
	}
}

/// Read the error counter after a full device sync, and reset it.
int DrainErrors(int* d_errors) {
	cudaDeviceSynchronize();
	int h = -1;
	cudaMemcpy(&h, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(d_errors, 0, sizeof(int));
	return h;
}

class PoolStressTest : public ::testing::Test {
  protected:
	static constexpr size_t kStreams = 8;

	std::vector<cudaStream_t> streams;
	int* d_errors	= nullptr;
	uint64_t* sink	= nullptr;
	std::atomic<int> nulls{ 0 };   ///< Pool returned null (never expected); counted off-thread.

	void SetUp() override {
		if (!HaveDevice())
			GTEST_SKIP() << "no CUDA device";
		cudaSetDevice(kDevice);
		initGPUprop();
		streams.resize(kStreams);
		for (size_t i = 0; i < kStreams; ++i)
			ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess);
		ASSERT_EQ(cudaMalloc(&d_errors, sizeof(int)), cudaSuccess);
		ASSERT_EQ(cudaMemset(d_errors, 0, sizeof(int)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(&sink, sizeof(uint64_t)), cudaSuccess);
		ASSERT_EQ(cudaMemset(sink, 0, sizeof(uint64_t)), cudaSuccess);
	}

	void TearDown() override {
		if (streams.empty())
			return;
		cudaDeviceSynchronize();
		for (cudaStream_t s : streams)
			cudaStreamDestroy(s);
		// The pool never returns memory, so the blocks themselves are deliberately left alone;
		// only this fixture's own scratch is released.
		cudaFree(d_errors);
		cudaFree(sink);
	}
};

} // namespace

// ---------------------------------------------------------------------------------------------
// One thread per stream, every thread hammering the SAME size class. This is the shape the
// application produces: 24 prefetch workers plus a consumer, all allocating limbs of one size.
// ---------------------------------------------------------------------------------------------
TEST_F(PoolStressTest, ManyThreadsOneSizeClass) {
	constexpr int kThreads = 16; // more threads than streams, so streams are shared too
	constexpr int kIters   = 60;

	std::vector<std::thread> workers;
	workers.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t)
		workers.emplace_back([&, t] { Churn(streams[t % kStreams], kLargeWords, kIters, (uint64_t)t + 1, d_errors, sink, &nulls); });
	for (auto& w : workers)
		w.join();

	EXPECT_EQ(nulls.load(), 0) << "the pool failed to serve an allocation";
	EXPECT_EQ(DrainErrors(d_errors), 0) << "a pooled block was written by its new owner before its previous owner's kernels finished";
}

// ---------------------------------------------------------------------------------------------
// Mixed size classes, so the per-size lists interleave and the small class (which GPUmalloc rounds
// up and force-caches) is exercised alongside the large one.
// ---------------------------------------------------------------------------------------------
TEST_F(PoolStressTest, ManyThreadsMixedSizeClasses) {
	constexpr int kThreads = 16;
	constexpr int kIters   = 60;

	std::vector<std::thread> workers;
	workers.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		const size_t words = (t % 2 == 0) ? kLargeWords : kSmallWords;
		workers.emplace_back([&, t, words] { Churn(streams[t % kStreams], words, kIters, (uint64_t)t + 101, d_errors, sink, &nulls); });
	}
	for (auto& w : workers)
		w.join();

	EXPECT_EQ(nulls.load(), 0) << "the pool failed to serve an allocation";
	EXPECT_EQ(DrainErrors(d_errors), 0) << "mixed-size churn handed out a live block";
}

// ---------------------------------------------------------------------------------------------
// The HANDOFF path, forced rather than hoped for.
//
// Stream 0 fills its own free list with a large number of blocks and frees them all, having first
// queued a long spin behind them so its stream is demonstrably still busy. Every other stream then
// allocates that same size class, which can only be served by taking stream 0's blocks -- the one
// path that records an event. Each taker writes and verifies its block while stream 0 is still
// working, so an event that is missing, recorded too early, or recorded on the wrong stream shows
// up as a mismatch count.
//
// This is the case the legacy pool's global event made trivially safe and that the per-stream pool has to earn.
// ---------------------------------------------------------------------------------------------
TEST_F(PoolStressTest, CrossStreamHandoffOrdersTheTaker) {
	constexpr size_t kBlocks = 64;
	const int bytes			 = (int)(kLargeWords * sizeof(uint64_t));
	const dim3 block{ 256 };
	const dim3 grid{ (uint32_t)((kLargeWords + 255) / 256) };

	// --- donor: take kBlocks on stream 0, write them, queue a LONG spin, then free them all ---
	cudaStream_t donor = streams[0];
	std::vector<uint64_t*> blocks;
	blocks.reserve(kBlocks);
	for (size_t i = 0; i < kBlocks; ++i) {
		auto* p = (uint64_t*)GPUmalloc(kDevice, bytes, donor, true);
		ASSERT_NE(p, nullptr);
		blocks.push_back(p);
		pool_fill_<<<grid, block, 0, donor>>>(p, kLargeWords, 0xD0D0D0D0ull ^ i);
	}
	// Queued AFTER the writes and BEFORE the frees: when the frees are issued the donor stream
	// still has all of this work outstanding, which is exactly the state a taker must be ordered
	// behind.
	pool_spin_<<<1, 32, 0, donor>>>(4000000, sink);
	for (size_t i = 0; i < kBlocks; ++i) {
		// A read of the donor's own bytes, issued behind the spin, so the donor is genuinely still
		// USING each block at the moment it is freed.
		pool_check_<<<grid, block, 0, donor>>>(blocks[i], kLargeWords, 0xD0D0D0D0ull ^ i, d_errors);
		GPUfree(blocks[i], kDevice, bytes, donor, true);
	}

	// --- takers: every other stream, immediately, on its own thread ---
	std::vector<std::thread> workers;
	for (size_t t = 1; t < kStreams; ++t)
		workers.emplace_back([&, t] { Churn(streams[t], kLargeWords, 8, (uint64_t)t + 1001, d_errors, sink, &nulls); });
	for (auto& w : workers)
		w.join();

	// A non-zero count here means either a taker overwrote the donor's bytes while the donor was
	// still reading them (the donor's pool_check_ fails), or a taker's own verify was clobbered.
	EXPECT_EQ(nulls.load(), 0) << "the pool failed to serve an allocation";
	EXPECT_EQ(DrainErrors(d_errors), 0) << "a cross-stream handoff was not ordered behind the donor stream";
}

// ---------------------------------------------------------------------------------------------
// Free on a DIFFERENT stream than the one that allocated. Legal, and it is how a block migrates
// between per-stream lists: the freeing stream becomes the owner, so the ordering edge a later
// taker records has to be the FREEING stream's, not the allocating one's.
// ---------------------------------------------------------------------------------------------
TEST_F(PoolStressTest, AllocateOnOneStreamFreeOnAnother) {
	constexpr int kThreads = 8;
	constexpr int kIters   = 40;
	const int bytes		   = (int)(kLargeWords * sizeof(uint64_t));
	const dim3 block{ 256 };
	const dim3 grid{ (uint32_t)((kLargeWords + 255) / 256) };

	std::vector<std::thread> workers;
	workers.reserve(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		workers.emplace_back([&, t] {
			cudaSetDevice(kDevice);
			cudaStream_t a = streams[t % kStreams];
			cudaStream_t b = streams[(t + 3) % kStreams];
			for (int i = 0; i < kIters; ++i) {
				auto* p = (uint64_t*)GPUmalloc(kDevice, bytes, a, true);
				if (p == nullptr) {
					nulls.fetch_add(1);
					continue;
				}
				const uint64_t tag = ((uint64_t)t << 24) ^ (uint64_t)i ^ 0xABCDEFull;
				pool_fill_<<<grid, block, 0, a>>>(p, kLargeWords, tag);
				pool_spin_<<<1, 32, 0, a>>>(20000, sink);
				pool_check_<<<grid, block, 0, a>>>(p, kLargeWords, tag, d_errors);
				// Hand the block to stream b's list WITHOUT first fencing b against a. The pool is
				// what has to make that safe for whoever takes it next.
				GPUfree(p, kDevice, bytes, b, true);
			}
		});
	}
	for (auto& w : workers)
		w.join();

	EXPECT_EQ(nulls.load(), 0) << "the pool failed to serve an allocation";
	EXPECT_EQ(DrainErrors(d_errors), 0) << "a block freed on a different stream than it was used on was handed out too early";
}

} // namespace FIDESlib::Testing
