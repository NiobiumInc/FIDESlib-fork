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

/// The log position before a call: pass it as `enter` to the entry logged after the call returns. An
/// entry whose seq is below another's `enter` provably returned before that other call started, which is
/// what the report needs to call a chain of records and waits proven (0: not stamped).
unsigned long long MemLogEnter();
/// Logged after the allocation returns. With the free marks on, also launches on `s` a check that the
/// previous owners of this memory had reached their free (see MemLogFreeMark).
void MemLogAlloc(void* p, size_t bytes, cudaStream_t s, void* site, unsigned long long enter = 0);
/// FREE MARKS. Call right BEFORE cudaFreeAsync(p, s): launches on `s` a one-thread kernel that marks the
/// free of this allocation as reached. The check launched on the next owner's stream after its
/// allocation must then find the mark, whatever the timing, if the allocator ordered that stream after
/// the free; a missing mark is a hand-off the device did not order. Off with FIDESLIB_DEBUG_FREEMARK=0.
void MemLogFreeMark(void* p, cudaStream_t s);
/// Logged after cudaFreeAsync returns.
void MemLogFree(void* p, cudaStream_t s, void* site, unsigned long long enter = 0);
/// A table upload (destination, the first bytes of the source), logged after the copy is issued, and a
/// launch of one of the three checked kernels with its table, so the report shows the order of upload,
/// launch and free.
void MemLogUpload(void* dst, const void* src, size_t bytes, cudaStream_t s, void* site, unsigned long long enter = 0);
void MemLogLaunch(void* table, cudaStream_t s, void* site);
/// An event record ('R') or a stream wait on an event ('W'), from the wrappers of TableOwnerFences.cuh.
void MemLogFence(char op, cudaEvent_t e, cudaStream_t s, void* site, unsigned long long enter = 0);

/// The free-mark totals and any violation not printed yet. Also runs at exit and from TableOwnerReport.
void FreeMarkSummary();

/// Prints the report and the ring events around its address, once per process. When an upload in the
/// log carried the bad value, it also follows the records and waits between that owner's free and the
/// next allocation of the memory, and says whether any of them ordered the new owner after the free.
void TableOwnerReport();

} // namespace FIDESlib
