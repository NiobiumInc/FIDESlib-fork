// The persistent per-op table ring (OpTableBuffer / OpTableRelease, used under
// FIDESLIB_PERSIST_TABLES): a buffer handed out again must not be written before the kernels of its
// previous user are done with it. A thread's ops run on many streams, so only the ring's own event
// orders the two.

#include <gtest/gtest.h>

#include <cstdint>

#include <cuda_runtime.h>

#include "CudaUtils.cuh"

namespace {

/// Waits about `cycles` clock cycles, then copies the table's first word to `out`.
__global__ void ReadAfterDelay(const std::uint64_t* table, std::uint64_t* out, const long long cycles) {
	const long long start = clock64();
	while (clock64() - start < cycles) {
	}
	*out = *table;
}

/// A class index the library does not use (it uses 0 to 4), so this test has a ring of its own.
constexpr int kTestClass = 7;

} // namespace

TEST(OpTableRingTest, ReuseWaitsForThePreviousUser) {
	int device = 0;
	ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
	cudaStream_t first_stream = nullptr, second_stream = nullptr;
	ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
	ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);
	std::uint64_t* out = nullptr;
	ASSERT_EQ(cudaMalloc(&out, sizeof(std::uint64_t)), cudaSuccess);

	const std::uint64_t first_value = 0x1111111111111111ull, second_value = 0x2222222222222222ull;

	// One lap of the ring first, so every buffer exists: a cudaMalloc during the walk below could
	// wait for the device and hide a missing fence.
	void* const lap_start = FIDESlib::OpTableBuffer(device, sizeof(std::uint64_t), kTestClass, second_stream);
	FIDESlib::OpTableRelease(device, lap_start, kTestClass, second_stream);
	for (int i = 0; i < 64; ++i) {
		void* const p = FIDESlib::OpTableBuffer(device, sizeof(std::uint64_t), kTestClass, second_stream);
		FIDESlib::OpTableRelease(device, p, kTestClass, second_stream);
		if (p == lap_start)
			break;
	}
	ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);

	// The first user uploads its table and a kernel reads it about a tenth of a second later.
	void* const first = FIDESlib::OpTableBuffer(device, sizeof(std::uint64_t), kTestClass, first_stream);
	ASSERT_NE(first, nullptr);
	ASSERT_EQ(cudaMemcpyAsync(first, &first_value, sizeof(std::uint64_t), cudaMemcpyHostToDevice, first_stream), cudaSuccess);
	ReadAfterDelay<<<1, 1, 0, first_stream>>>(static_cast<const std::uint64_t*>(first), out, 200'000'000);
	ASSERT_EQ(cudaGetLastError(), cudaSuccess);
	FIDESlib::OpTableRelease(device, first, kTestClass, first_stream);

	// Walk the ring on another stream until it hands out the same buffer again.
	void* again = nullptr;
	for (int i = 0; i < 64; ++i) {
		void* const p = FIDESlib::OpTableBuffer(device, sizeof(std::uint64_t), kTestClass, second_stream);
		if (p == first) {
			again = p;
			break;
		}
		FIDESlib::OpTableRelease(device, p, kTestClass, second_stream);
	}
	ASSERT_EQ(again, first) << "the ring never handed the first buffer out again";

	// The second user's upload is queued while the kernel above is still waiting. It must land after
	// the kernel's read, so the kernel sees the first user's value.
	ASSERT_EQ(cudaMemcpyAsync(again, &second_value, sizeof(std::uint64_t), cudaMemcpyHostToDevice, second_stream), cudaSuccess);
	FIDESlib::OpTableRelease(device, again, kTestClass, second_stream);

	ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
	ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);
	std::uint64_t seen = 0;
	ASSERT_EQ(cudaMemcpy(&seen, out, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), cudaSuccess);
	EXPECT_EQ(seen, first_value) << "the second user's upload overwrote the table while the first user's kernel still needed it";

	EXPECT_EQ(cudaFree(out), cudaSuccess);
	EXPECT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
	EXPECT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
}
