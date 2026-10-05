#ifndef API_CONCURRENTOPS_HPP
#define API_CONCURRENTOPS_HPP

#include <cstddef>
#include <cstdlib>
#include <iosfwd>
#include <string>
#include <vector>

#ifdef FIDESLIB_ENABLE_CUDA
namespace FIDESlib {
/// Defined in src/CudaUtils.cu, which is compiled only in a CUDA build. Declared here rather than
/// included so that this header stays usable by an application that has no access to the private
/// (src/) headers, and so that the CPU branch below needs no CUDA types at all.
std::string MemPoolStatsLine(int id);

/// The provenance tracer, declared for the same reason and on the same terms as the line above:
/// its definitions live in src/CudaUtils.cu, which only a CUDA build compiles, and the public
/// wrappers below must also exist in a CPU-only one. See the wrappers for what each does.
bool PoolTraceEnabled();
void PoolTraceDumpFor(const void* device_ptr, std::ostream& os);
void PoolTraceDumpRecent(std::ostream& os, size_t n);
std::string PoolTraceOwnersLine(const void* device_ptr);

} // namespace FIDESlib
#endif

namespace FIDESlib::CKKS {
/// Forward declaration only, and UNCONDITIONAL: CiphertextLimbPointers below names this type in
/// its signature, and this header must parse in a CPU-only build too -- where the type is never
/// defined and the function is never linked, because nothing in such a build can hold a GPU
/// ciphertext to ask about. Declaring it costs a name; including the private header would cost
/// the whole point of a public header.
class Ciphertext;
} // namespace FIDESlib::CKKS

namespace fideslib {

/// @brief Whether this run lets SEVERAL host threads issue ops on ONE CryptoContext
/// (`FIDESLIB_CONCURRENT_OPS=1`). Read once.
///
/// Default: one issuing thread per CryptoContext (the library's supported mode). Set to 1 to let
/// several host threads issue ops on one context concurrently; each issuing thread then gets its
/// own op scratch (FIDESLIB_SCRATCH_SLOTS).
///
/// This is the PUBLIC query for the option documented at its definition in `src/CudaUtils.cuh`
/// (`FIDESlib::ConcurrentOps`), which is where the list of scratch the concurrent mode splits per
/// thread lives. It is repeated here as a header-only read of the same variable so that an
/// application can ASK BEFORE it spawns issuing threads, and so that the query exists in CPU-only
/// builds, where the CUDA source that owns the other definition is not compiled. The two must name
/// the same variable and parse it the same way; there is no other coupling.
///
/// An application that issues ops from more than one thread on one context must set this and
/// should refuse to run if it is unset -- in the default mode two threads' key switches write into
/// the same auxiliary polynomial.
inline bool ConcurrentOpsEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_CONCURRENT_OPS");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled;
}

/// @brief ONE line of device-memory-pool facts for device `id`: chunks cut, drains performed,
/// blocks reclaimed, and how many blocks are sitting in the fresh tier and on the lanes' freed
/// lists. Empty in a CPU-only build, which has no device pool.
///
/// This is the PUBLIC read of the counters documented at their definition in `src/CudaUtils.cuh`
/// (`FIDESlib::MemPoolStats`). It exists so an application can print the pool's state at the end
/// of a pass without reaching into the private headers, and so that the call compiles in a CPU
/// build where the CUDA source that owns the counters is not built.
inline std::string MemPoolStatsLine(int id = 0) {
#ifdef FIDESLIB_ENABLE_CUDA
	return FIDESlib::MemPoolStatsLine(id);
#else
	(void)id;
	return {};
#endif
}

/// @brief What CryptoContext::ReleaseParkedScratch() gave back at a pass boundary.
///
/// All zero with `ran == false` whenever the release was a no-op: the default (non-concurrent)
/// mode, a CPU-only build, or a context whose backend has no device pool. `ran == false` therefore
/// means "nothing was attempted", NOT "nothing needed doing" -- an application that logs these
/// numbers should say so rather than report a successful release of zero.
///
/// It mirrors FIDESlib::CKKS::ContextData::ScratchRelease field for field. The duplication is on
/// purpose and is the same one MemPoolStatsLine above already pays: this header must compile in a
/// CPU-only build and in an application that has no access to the private (src/) CUDA headers, so
/// the public type cannot BE the private one. The conversion is one aggregate copy in
/// CudaEngine::releaseParkedScratch and is the only coupling between them.
struct ScratchReleaseStats {
	size_t polys_released = 0; ///< Parked auxiliary polynomials handed back to the device pool.
	size_t blocks_moved	  = 0; ///< Pooled blocks moved from the lanes' freed lists to the shared fresh tier.
	size_t slots_drained  = 0; ///< Scratch slots that were holding at least one parked polynomial.
	bool ran			  = false; ///< False = the release is not applicable to this run; see above.
};

/// @brief Whether the device-memory pool's BLOCK PROVENANCE TRACER is on.
///
/// The tracer answers, for a pooled block, "who held this, on which stream, and which fence did
/// its next owner take". It is the instrument for a concurrent-mode result that comes back wrong
/// with its shape intact -- a whole polynomial of garbage says the block backing it was written
/// by one lane while another still owned or was still reading it, and a lock bisection can only
/// say which scope hides that, never which interleaving produced it.
///
/// ON requires BOTH `FIDESLIB_POOL_TRACE=1` and `FIDESLIB_CONCURRENT_OPS=1`: the thing it traces
/// is the lane pool, which the default mode does not have. Read once. OFF -- the default, and
/// always in a CPU-only build -- every hook in the library is one cached-bool read and a return,
/// and nothing below allocates, locks or prints.
inline bool PoolTraceEnabled() {
#ifdef FIDESLIB_ENABLE_CUDA
	return FIDESlib::PoolTraceEnabled();
#else
	return false;
#endif
}

/// @brief Print the provenance of the pooled block behind `device_ptr`: its last few cut / take /
/// free / drain-move / boundary-move / output events, each with a monotonic sequence number, a
/// host thread, a pool lane, the streams involved and the fence event used.
///
/// `device_ptr` may be INTERIOR to the block (a limb's data pointer usually is); the block that
/// contains it is found by address. A pointer no pooled block covers prints one `no provenance`
/// line rather than nothing. No-op line if the tracer is off.
inline void PoolTraceDumpFor(const void* device_ptr, std::ostream& os) {
#ifdef FIDESLIB_ENABLE_CUDA
	FIDESlib::PoolTraceDumpFor(device_ptr, os);
#else
	(void)device_ptr;
	(void)os;
#endif
}

/// @brief Print the last `n` events of the GLOBAL FENCE RING -- every record / wait /
/// record_and_wait the library issued in its fence paths, including the calls that took an early
/// out and issued nothing.
///
/// Sequence numbers are shared with the per-block rings above, so a fence and the take or free it
/// belongs to are adjacent numbers and the two dumps can be read as one order.
inline void PoolTraceDumpRecent(std::ostream& os, size_t n = 2000) {
#ifdef FIDESLIB_ENABLE_CUDA
	FIDESlib::PoolTraceDumpRecent(os, n);
#else
	(void)os;
	(void)n;
#endif
}

/// @brief One line naming which lane last TOOK and which lane last FREED the block behind
/// `device_ptr` (and which lane last declared it part of an output it had finished writing).
/// Empty string if the tracer is off or the pointer is not a pooled block.
inline std::string PoolTraceOwnersLine(const void* device_ptr) {
#ifdef FIDESLIB_ENABLE_CUDA
	return FIDESlib::PoolTraceOwnersLine(device_ptr);
#else
	(void)device_ptr;
	return {};
#endif
}

/// @brief One limb of a ciphertext, as the tracer needs to name it: the device pointer its
/// coefficients live at and the stream its launches are issued on.
struct LimbBlockRef {
	const void* data   = nullptr; ///< Device pointer backing the limb's coefficient vector.
	const void* stream = nullptr; ///< The limb's launch stream.
	int poly		   = 0;		  ///< 0 = c0, 1 = c1.
	int partition	   = 0;		  ///< Index into the polynomial's per-device partitions.
	int limb		   = 0;		  ///< Index of the limb within that partition's Q limbs.
	int primeid		   = -1;	  ///< RNS prime this limb is a residue modulo.
};

/// @brief Every Q limb of `ct`'s c0 and c1, in (poly, partition, limb) order.
///
/// This is the MINIMUM reach into the private types a provenance dump needs: a test that has a
/// wrong ciphertext in hand has to turn it into the device pointers whose rings it wants printed,
/// and those live several levels down (Ciphertext -> RNSPoly -> LimbPartition -> Limb -> VectorGPU).
/// Returns the pointers only; it copies no device data and touches no stream.
///
/// NOT LINKED in a CPU-only build -- nothing there can hold a GPU ciphertext to ask about.
/// SPECIAL (P-basis) limbs are deliberately left out: a stored ciphertext has none, and the wrong
/// result the tracer is aimed at is a Q-basis polynomial.
std::vector<LimbBlockRef> CiphertextLimbPointers(const FIDESlib::CKKS::Ciphertext& ct);

} // namespace fideslib

#endif // API_CONCURRENTOPS_HPP
