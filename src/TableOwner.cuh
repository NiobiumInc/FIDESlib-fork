// DEBUG BRANCH ONLY. Names the previous owner of a per-op pointer table's memory when a kernel finds
// something that is not a pointer in the table.
//
// Device side (CKKS/ElemenwiseBatchKernels.cu): eval_linear_w_sum_, fusedDotKSK_2_ and
// dotProductLtBatchedPt2___ check the table entries their block reads. On a bad one the first
// block to see it writes a TableReport into mapped pinned memory and traps.
//
// Host side (CudaUtils.cu): every stream-ordered device allocation and free goes into a ring.
// CudaFailure prints the report and every ring event whose range covers the bad address, with the
// call site, so the object whose memory the table took over can be named.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace FIDESlib {

struct TableReport {
	unsigned int hit;			///< 0 until the first bad entry
	int kernel;					///< 1 eval_linear_w_sum_, 2 fusedDotKSK_2_, 3 dotProductLtBatchedPt2___
	int level;					///< 1: the per-op table entry, 2: the array that entry points to
	int index;					///< entry index in the table at that level
	unsigned long long table;	///< device address of the per-op table
	unsigned long long value;	///< the bad value
	unsigned long long array;	///< level 2: the array read through the table entry
	int bx, by, bz;
	int nwords;					///< words copied below
	unsigned long long words[32]; ///< the start of the table (level 1) or of the array (level 2)
};

/// Maps the report and points the kernels at it. Idempotent; the first call is on the op path.
void TableCheckInit();
/// The report as the host sees it (mapped pinned memory), or nullptr before TableCheckInit().
const TableReport* TableReportHost();

void MemLogAlloc(void* p, size_t bytes, cudaStream_t s, void* site);
void MemLogFree(void* p, cudaStream_t s, void* site);

/// Prints the report and the ring events around its address, once per process.
void TableOwnerReport();

} // namespace FIDESlib
