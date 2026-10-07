// The CUDA error checks (CudaCheckErrorMod, CudaCheckErrorModMGPU, CudaCheckErrorModNoSync) report a
// failure by throwing FIDESlib::CudaError. They used to call exit(0), which ended the caller's
// process with a success status. Each test sets a non-sticky error (a failed allocation), so the
// context stays usable and the next test is unaffected.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "CudaUtils.cuh"

namespace {

/// Leaves cudaErrorMemoryAllocation as the thread's last error: no device has 2^62 bytes.
void FailAnAllocation() {
	void* p = nullptr;
	ASSERT_EQ(cudaMalloc(&p, std::size_t{ 1 } << 62), cudaErrorMemoryAllocation);
}

void ExpectAllocationFailure(const FIDESlib::CudaError& e) {
	EXPECT_EQ(e.error(), cudaErrorMemoryAllocation);
	EXPECT_NE(std::string(e.file()).find("CudaErrorTests.cu"), std::string::npos);
	EXPECT_GT(e.line(), 0);
	EXPECT_EQ(std::string(e.what()).rfind("Cuda failure ", 0), 0u) << e.what();
	// The check consumed the error and the context still works.
	EXPECT_EQ(cudaGetLastError(), cudaSuccess);
	EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

} // namespace

TEST(CudaErrorTest, NoSyncCheckThrows) {
	FailAnAllocation();
	try {
		CudaCheckErrorModNoSync;
		FAIL() << "CudaCheckErrorModNoSync returned after a failed allocation";
	} catch (const FIDESlib::CudaError& e) {
		ExpectAllocationFailure(e);
	}
}

TEST(CudaErrorTest, SyncCheckThrows) {
	FailAnAllocation();
	try {
		CudaCheckErrorMod;
		FAIL() << "CudaCheckErrorMod returned after a failed allocation";
	} catch (const FIDESlib::CudaError& e) {
		ExpectAllocationFailure(e);
	}
}

TEST(CudaErrorTest, MgpuCheckThrows) {
	FailAnAllocation();
	try {
		CudaCheckErrorModMGPU;
		FAIL() << "CudaCheckErrorModMGPU returned after a failed allocation";
	} catch (const FIDESlib::CudaError& e) {
		ExpectAllocationFailure(e);
	}
}

TEST(CudaErrorTest, ChecksPassWithoutAnError) {
	ASSERT_EQ(cudaGetLastError(), cudaSuccess);
	EXPECT_NO_THROW(CudaCheckErrorModNoSync);
	EXPECT_NO_THROW(CudaCheckErrorMod);
	EXPECT_NO_THROW(CudaCheckErrorModMGPU);
}

TEST(CudaErrorTest, IsAStdRuntimeError) {
	// Callers of the public API catch it without including any CUDA header.
	FailAnAllocation();
	EXPECT_THROW(CudaCheckErrorModNoSync, std::runtime_error);
	EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
