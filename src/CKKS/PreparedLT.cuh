//
// A RESIDENT device pointer table for the batched linear-transform MAC.
//

#ifndef FIDESLIB_CKKS_PREPAREDLT_CUH
#define FIDESLIB_CKKS_PREPAREDLT_CUH

#include "forwardDefs.cuh"
#include <vector>

namespace FIDESlib::CKKS {

/// @brief The DIAGONAL third of LTdotProductPtBatch's pointer table, allocated on the device once
/// and reused for the life of the diagonals.
///
/// THE PROBLEM. `LimbPartition::LTdotProductPtBatch` hands the MAC kernel its operands as a device
/// array of pointers: one entry per output polynomial, one per baby-rotation polynomial, and one
/// per diagonal — and, under `ext`, the whole thing again for the P basis. It builds that array
/// into a function-local `std::vector` and uploads it with a `cudaMemcpyAsync` out of PAGEABLE host
/// memory on EVERY call. The fork records what that costs in `src/CKKS/Limb.cu:91-94`: pageable
/// `cudaMemcpyAsync` is not asynchronous on the host side, because the driver stages the bytes
/// through its own pinned buffer and does not return until the staging copy is done. At the
/// transformer shapes this library is used at, the DIAGONAL third is the majority of those bytes
/// (num_LT * gStep * bStep entries against 2 * num_LT * (gStep + bStep) for the rest).
///
/// THE OBSERVATION. The output and baby-rotation entries are genuinely per-call — those
/// ciphertexts are allocated by the transform itself. The DIAGONAL entries are not: they are the
/// device limb-pointer-array addresses of plaintexts that stay resident across calls. So they can
/// be uploaded once.
///
/// WHAT THIS OWNS. One device allocation PER GPU PARTITION, holding the diagonal table of every
/// giant-step chunk of the group back to back, Q basis then (when `ext`) P basis:
///
///     buffer[p] = [ chunk 0 Q | chunk 0 P | chunk 1 Q | chunk 1 P | ... ]
///
/// `ptQ(p, c)` / `ptP(p, c)` hand out the base the kernel wants. Nothing here is mutated after
/// construction, which is what makes a single table safe to use from concurrent calls: the
/// per-call thirds of the table are still per-call allocations, exactly as before.
///
/// WHAT IT DOES NOT CHANGE. The same pointers reach the same kernel in the same order. There is no
/// numerical difference between a prepared call and an unprepared one, by construction.
///
/// STALENESS. This object holds RAW device pointers. It does not and cannot know that a diagonal
/// was re-encoded or freed — that is the api layer's job (`fideslib::PreparedLinearTransformImpl`
/// holds the plaintexts alive and re-checks their identity before every call, and throws rather
/// than let a stale entry reach a kernel). `hostImage()` is exposed so that check has something to
/// compare against, and so debug builds can assert the table the callee would have built matches
/// the one it is being handed.
class PreparedLTTable {
  public:
	/// The giant-step chunking `RNSPoly::LTdotProductPtBatch` splits a wide transform into. One
	/// constant, used both there and here, because a drift between them would mean this table's
	/// chunk `c` and that loop's chunk `c` describe different giant steps.
	static constexpr int kGiantChunk = 8;

	/// @param cc        the context the diagonals live in (for the GPU partition list).
	/// @param pts       [transform][diagonal], exactly as LinearTransformMany takes them. Entries
	///                  at index >= rowSize are not read; the null padding of the last giant step
	///                  is built here, as the transform itself builds it.
	/// @param rowSize   diagonals per transform.
	/// @param bStep     baby steps per giant step.
	/// @param stride    the pt_reuse_stride LTdotProductPtBatch takes (1 from LinearTransform*).
	/// @param ext       the diagonals are in the extended Q||P basis, so the P table is built too.
	PreparedLTTable(ContextData& cc, const std::vector<std::vector<Plaintext*>>& pts, int rowSize, int bStep, int stride, bool ext);
	~PreparedLTTable();

	PreparedLTTable(const PreparedLTTable&)			   = delete;
	PreparedLTTable& operator=(const PreparedLTTable&) = delete;
	PreparedLTTable(PreparedLTTable&&)				   = delete;
	PreparedLTTable& operator=(PreparedLTTable&&)	   = delete;

	int numLT() const {
		return num_LT_;
	}
	int rowSize() const {
		return rowSize_;
	}
	int bStep() const {
		return bStep_;
	}
	int gStep() const {
		return gStep_;
	}
	int stride() const {
		return stride_;
	}
	bool ext() const {
		return ext_;
	}
	int chunks() const {
		return chunks_;
	}
	int partitions() const {
		return static_cast<int>(buffer_.size());
	}
	/// Giant steps in chunk @p c — kGiantChunk except possibly the last.
	int chunkGiantSteps(int c) const;
	/// Diagonal entries in chunk @p c: num_LT * chunkGiantSteps(c) * bStep.
	int ptEntries(int c) const;

	/// Q-basis diagonal table for (GPU partition, giant-step chunk).
	void*** ptQ(int partition, int chunk) const;
	/// P-basis diagonal table for (GPU partition, giant-step chunk); nullptr when !ext.
	void*** ptP(int partition, int chunk) const;

	/// The host image of a partition's whole buffer, in the same layout. For the identity check and
	/// for the debug-build assertion in LTdotProductPtBatch.
	const std::vector<void**>& hostImage(int partition) const {
		return host_[static_cast<size_t>(partition)];
	}
	/// Offset of (chunk, basis) within a partition's buffer, in entries.
	int chunkOffset(int chunk, bool special) const;

  private:
	int num_LT_	 = 0;
	int rowSize_ = 0;
	int bStep_	 = 0;
	int gStep_	 = 0;
	int stride_	 = 1;
	bool ext_	 = false;
	int chunks_	 = 0;
	int entries_ = 0; // per partition
	std::vector<void***> buffer_;
	std::vector<int> device_;
	std::vector<std::vector<void**>> host_;
};

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_PREPAREDLT_CUH
