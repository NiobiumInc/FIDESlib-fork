#ifndef API_PREPAREDLINEARTRANSFORM_HPP
#define API_PREPAREDLINEARTRANSFORM_HPP

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief The SHAPE half of a linear transform: everything LinearTransformInPlace /
/// LinearTransformMany take except the diagonal VALUES.
struct LinearTransformShape {
	int rowSize = 0;
	int bStep	= 0;
	int stride	= 1;
	int offset	= 0;
	bool ext	= false;

	bool operator==(const LinearTransformShape& o) const {
		return rowSize == o.rowSize && bStep == o.bStep && stride == o.stride && offset == o.offset && ext == o.ext;
	}
	bool operator!=(const LinearTransformShape& o) const {
		return !(*this == o);
	}
	/// Human-readable, for the messages the staleness and mismatch throws carry.
	std::string str() const;
};

/// @brief A linear transform whose DIAGONALS ARE ALREADY RESIDENT, prepared once and reused.
///
/// WHAT IT IS FOR. The batched MAC kernel is handed its operands as a
/// device table of pointers: one entry per result polynomial, one per baby-rotation polynomial, and
/// one per diagonal — and, in the extended basis, the whole thing again. Today that table is
/// rebuilt into a function-local std::vector and uploaded with a PAGEABLE cudaMemcpyAsync on EVERY
/// giant step, which the fork's own comment (src/CKKS/Limb.cu:91-94) records as a host STALL: the
/// driver stages pageable bytes through its own pinned buffer and does not return until it has.
///
/// The result and baby-rotation entries are genuinely per-call. The DIAGONAL entries are not — they
/// are the device addresses of plaintexts that stay resident across calls, and at transformer
/// shapes they are the MAJORITY of the table. This handle owns them: one device buffer per GPU
/// partition, covering every giant step of the group, written and uploaded ONCE.
///
/// ONE HANDLE, ONE BATCH. A handle carries `sets()` diagonal sets, so the batched call it feeds
/// (LinearTransformMany) keeps everything batching buys it — one hoisted key-switch precompute and
/// one launch for the whole group. `PrepareLinearTransform` builds a one-set handle;
/// `PrepareLinearTransformMany` builds an N-set one. A one-set handle is what
/// LinearTransformInPlace takes.
///
/// WHAT IT OWNS AND WHY. It holds the diagonals BY SHARED HANDLE, so nothing it points at can be
/// freed underneath it, and it records a per-diagonal IDENTITY at prepare time (see
/// CryptoContextImpl::PlaintextIdentity: on a device backend the diagonal's device limb-pointer
/// addresses and RNS basis, on a host backend a fold of the encoding). Every prepared call
/// re-checks those identities first and THROWS on any mismatch. That is deliberate and not
/// negotiable: a stale pointer in this table is a WRONG ANSWER, not a slowdown, so the failure mode
/// is a throw and never a fallback to the vector path.
///
/// LIFETIME. The device payload is a shared_ptr in `device` and is released with the handle, so a
/// caching owner (the application's diagonal pool) drops the table by dropping the handle.
///
/// THREADING. Nothing in the handle or its device table is mutated after preparation — the per-call
/// thirds of the pointer table are still per-call allocations — so USING one is as thread-safe as
/// the underlying transform. Preparing is not thread-safe against use of the same handle.
class PreparedLinearTransformImpl {
  public:
	PreparedLinearTransformImpl(const LinearTransformShape& shape, std::vector<std::vector<Plaintext>> sets, std::vector<std::vector<uint64_t>> identity);
	~PreparedLinearTransformImpl() = default; // `device` releases its backend payload via RAII

	PreparedLinearTransformImpl(const PreparedLinearTransformImpl&)			   = delete;
	PreparedLinearTransformImpl& operator=(const PreparedLinearTransformImpl&) = delete;
	PreparedLinearTransformImpl(PreparedLinearTransformImpl&&)				   = delete;
	PreparedLinearTransformImpl& operator=(PreparedLinearTransformImpl&&)	   = delete;

	const LinearTransformShape& Shape() const {
		return shape_;
	}
	/// How many independent transforms this handle carries. 1 for PrepareLinearTransform.
	size_t Sets() const {
		return sets_.size();
	}
	/// All diagonal sets, in index order, held alive by this handle.
	const std::vector<std::vector<Plaintext>>& DiagonalSets() const {
		return sets_;
	}
	/// The single set of a one-set handle. Throws on a batch handle.
	const std::vector<Plaintext>& Diagonals() const;
	/// The per-diagonal identity recorded at prepare time, parallel to DiagonalSets().
	const std::vector<std::vector<uint64_t>>& Identity() const {
		return identity_;
	}

	/// @brief Re-check every recorded identity; throw naming the first diagonal that moved.
	///
	/// Called by every prepared entry point BEFORE anything is launched. Cheap: one host-side read
	/// per diagonal, no device traffic. @p what names the entry point in the message.
	void RequireFresh(CryptoContextImpl<DCRTPoly>& ctx, const char* what) const;

	/// @brief Backend payload — the CUDA engine's resident device pointer table. Empty on backends
	/// that have nothing to prepare (CPU, haze), which is what makes the handle a thin wrapper
	/// there rather than a second code path.
	std::any device;

  private:
	LinearTransformShape shape_;
	std::vector<std::vector<Plaintext>> sets_;
	std::vector<std::vector<uint64_t>> identity_;
};

/// @brief Shared pointer alias for PreparedLinearTransformImpl.
using PreparedLinearTransform = std::shared_ptr<PreparedLinearTransformImpl>;

} // namespace fideslib

#endif // API_PREPAREDLINEARTRANSFORM_HPP
