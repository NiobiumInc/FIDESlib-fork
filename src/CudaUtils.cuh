//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cuda.h>
#include <driver_types.h>
#include <execinfo.h>
#include <atomic>
#include <functional>
#include <map>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace FIDESlib {

/// @brief Whether several host threads may issue ops on ONE CryptoContext
/// (`FIDESLIB_CONCURRENT_OPS=1`). Read once.
///
/// Default: one issuing thread per CryptoContext (the library's supported mode). Set to 1 to let
/// several host threads issue ops on one context concurrently; each issuing thread then gets its
/// own op scratch (FIDESLIB_SCRATCH_SLOTS).
///
/// WHY THERE IS AN OPTION AT ALL. An op writes into scratch that, in the default mode, is ONE
/// object per context per kind: the key-switch
/// auxiliary polynomials (every key switch, relinearization, rotation, conjugation and hoisted
/// rotation), the mod-down auxiliaries (every mod-down and every rescale), the top-limb rescale
/// staging buffer, the per-digit mod-up scheduling streams, the monomial cache (every bootstrap)
/// and the auxiliary-polynomial free list (every Ciphertext constructor and destructor). One
/// issuing thread is what that is built for, and it is what almost every application wants, so it
/// stays the default and stays exactly as it was: in the default mode every path below takes the
/// same branch, the same object and the same number of instructions it took before this option
/// existed -- no extra allocation, no lock, no atomic.
///
/// In the CONCURRENT mode the library additionally supports an application that issues INDEPENDENT
/// ops from several threads on one context: each issuing thread gets its own scratch set
/// (ScratchSlot / CKKS::ContextData::OpScratch), the auxiliary-polynomial free list and the
/// first-touch device load are locked, and a stream's fence takes that stream's lock. Ops still
/// compute exactly what they computed before -- the arithmetic is unchanged; what changes is only
/// that two threads no longer share one buffer.
///
/// The cost of the concurrent mode when the application is nonetheless single-threaded is one
/// uncontended lock per fence, per auxiliary-polynomial checkout and per first-touch device load,
/// plus one thread_local read per scratch accessor.
bool ConcurrentOps();

/// @brief DIAGNOSTIC (FIDESLIB_CONCURRENT_OPS_DIAG; concurrent mode only): a comma list of
/// primitives to serialize PROCESS-WIDE behind one recursive lock -- `fence` (Stream::wait /
/// Stream::record), `modup`, `ksk` (the dot-KSK family), `moddown`, `rescale`, `mult`
/// (Ciphertext::mult / square as a whole), or `all`. A bisection instrument for a concurrent-mode
/// mismatch: the primitive whose serialization makes concurrently issued lanes match the serial
/// run again is the one whose concurrent ISSUE is wrong. Unset -- the default, and always in the
/// default mode -- every scope is one cached-bool read and no lock. Unknown names throw.
enum class ConcurrentOpsDiag : unsigned {
	kFence	 = 1u,
	kModup	 = 2u,
	kKsk	 = 4u,
	kModdown = 8u,
	kRescale = 16u,
	kMult	 = 32u,
	kMultPt	 = 64u,	 ///< Ciphertext::multPt (fused ct x pt + rescale)
	kAuxPoly = 128u, ///< the auxiliary-polynomial free list hand-off (Ciphertext ctor/dtor)
	kNtt	 = 256u, ///< Limb / LimbPartition NTT and INTT launches
	kLoad	 = 512u,  ///< Limb::load(host vector): every host -> device upload of limb data
	kStore	 = 1024u, ///< Limb::store: every device -> host download of limb data
	kCtLoad	 = 2048u, ///< Ciphertext::load / store as a whole (grow/drop + the limb marshals)
	// Bootstrap sub-steps (the residual after the rescale-scratch fix is bootstrap-only).
	kBoot	   = 4096u,	  ///< Bootstrap() as a whole
	kLt		   = 8192u,	  ///< LinearTransform / LinearTransformMany (CtS, StC, hoisted rotations)
	kModRaise  = 16384u,  ///< ModRaise (incl. the encapsulated sparse-key switch)
	kEvalMod   = 32768u,  ///< approxModReduction / approxModReductionSparse
	kAccum	   = 65536u,  ///< Accumulate
	kKeySwitch = 131072u, ///< Ciphertext::keySwitch
	kMonomial  = 262144u, ///< Ciphertext::multMonomial
	kConj	   = 524288u, ///< Ciphertext::conjugate
	// The two below are NOT locks. Each one makes the lane pool's hand-off quiescent with a
	// device-wide synchronize, so that a block can never change hands while a kernel that touches
	// it is still in flight -- whatever stream that kernel was launched on, and however the free was
	// fenced. They split the concurrent-bootstrap race into two classes with one run: if either one
	// zeroes the StagedLanesMatchSerial mismatch, the defect is a memory hand-off whose fence names
	// the wrong stream (an under-fenced free); if neither does, the pool is exonerated and the
	// defect is an object the lanes share and mutate. Diagnostic only; never on by default.
	kFreeSync = 1048576u, ///< GPUfree (concurrent pool): cudaDeviceSynchronize before the block is listed
	kTakeSync = 2097152u, ///< GPUmalloc (concurrent pool): cudaDeviceSynchronize before the block is handed out
	// Not a lock either: freed blocks are QUARANTINED instead of listed, in a process-wide FIFO capped
	// at kPoolQuarantineBytes; only when the cap is exceeded is the oldest block synchronized and
	// listed as fence-free. With it on, no lane can reach memory another lane released in the recent
	// past, whatever fence the release carried and whenever the releasing lane's kernels run. If the
	// staged case fails under it, memory reuse -- pooled hand-off AND host-order use-after-free -- is
	// out, and the defect is host state the lanes share.
	kNoReuse = 4194304u, ///< GPUfree (concurrent pool): quarantine the block instead of listing it
	// Not a lock: every Stream::init creates its OWN cudaStream_t instead of taking one of the
	// POOL_SIZE shared handles round-robin. Under the concurrent mode distinct Stream objects in
	// different lanes routinely hold the same handle; correct ordering never depends on handles
	// being distinct, but any place that silently assumes it (an identity check, a handle-keyed
	// table, a fence that early-outs on an equal handle) becomes a cross-lane coupling. If the
	// staged mismatch disappears under this config, handle sharing is in the mechanism.
	kPrivStreams = 8388608u, ///< Stream::init: a private stream per Stream object (never pooled)
	// Not a lock: the same 37-handle round-robin pool as the default, but ONE POOL PER LANE
	// (MemPoolLane of the constructing thread). Intra-lane handle sharing and the handle count per
	// lane are unchanged; only cross-lane sharing is removed. With privstreams (8/8 clean)
	// this splits "lanes share handles" from "many distinct handles change the interleaving".
	kLanePool = 16777216u, ///< Stream::init: per-lane 37-handle pools instead of one shared pool
	// Not a lock: the SHARED pool exactly as the default, plus one cudaStreamCreate + cudaStreamDestroy
	// of a throwaway stream on every Stream::init and every ~Stream. Reproduces the host-side API churn
	// of privstreams (a 6x slower run) WITHOUT changing which handle any object gets. Clean
	// under churn = privstreams was a timing mask; failing under churn while bigpool is clean = handle
	// distinctness is the mechanism.
	kChurn = 33554432u, ///< Stream::init / ~Stream: create and destroy a throwaway stream, handles unchanged
	// Sub-kinds of kModup, for FIDESLIB_STALL_IN only (stall_modup alone hid the staged
	// mismatch 3/3). Each one brackets ONE fence site inside RNSPoly::modup / modupInto /
	// LimbPartition::modupMGPU, so that stalling the waiter of that site alone names the consumer
	// that runs ahead. Not locks; never part of `all`; no effect without FIDESLIB_STALL_US.
	kModupEntry	= 67108864u,	///< modupMGPU: `s.wait(c0.s)` (the aux2 scratch joins the poly stream)
	kModupDigits = 134217728u,	///< modupMGPU: the digit streams join `s` before the INTTs
	kModupDs2	= 268435456u,	///< modupMGPU: `getDigitStream2(d_).wait(stream1)` after the conv kernel
	kModupExit	= 536870912u,	///< modupMGPU: `s` joins the digit streams, then `c0.s.wait(s)`
	kModupCopy	= 1073741824u, ///< RNSPoly::modupInto: the `poly.copy(*this)` into the scratch
	kModupAlloc	= 2147483648u, ///< RNSPoly::modup: the two generateDecompAndDigit calls
};
/// `all` names every kind EXCEPT fence: a fence lock is taken inside the memory pool (GPUmalloc /
/// GPUfree hold mempool_lock and fence the pool stream), so combining it with a kind that holds
/// the diagnostic lock across an allocation inverts the two and deadlocks.
bool ConcurrentOpsDiagOn(ConcurrentOpsDiag which);
class ConcurrentOpsDiagScope {
	std::unique_lock<std::recursive_mutex> lk_;
	bool pushed_ = false; ///< this scope pushed its kind on the thread's stack (see DiagStall)

  public:
	explicit ConcurrentOpsDiagScope(ConcurrentOpsDiag which);
	~ConcurrentOpsDiagScope();
	ConcurrentOpsDiagScope(const ConcurrentOpsDiagScope&)			 = delete;
	ConcurrentOpsDiagScope& operator=(const ConcurrentOpsDiagScope&) = delete;
};

/// @brief The calling thread's op-scratch slot, assigned on first use and stable for the thread.
///
/// Always 0 in the default mode, and slot 0 IS the context's own scratch members -- so by default
/// this is a thread_local read that returns 0 and every accessor behaves exactly as it did before
/// slots existed. In the concurrent mode a second issuing thread gets slot 1 and its own scratch
/// polynomials, a third slot 2, and so on up to ScratchSlotCap(); a slot is RETURNED when its
/// thread exits and handed to the next thread that asks. See CKKS::ContextData::OpScratch.
int ScratchSlot();

/// @brief Maximum number of threads that may issue ops on one context AT ONCE
/// (FIDESLIB_SCRATCH_SLOTS, default 32).
///
/// Only consulted in the concurrent mode. This bounds CONCURRENT issuers, not threads ever
/// created: a slot is returned when its thread exits, so an application that fans out with fresh
/// threads per batch of work reuses the same few sets for the whole run. A thread that asks once
/// the cap is reached gets a `std::runtime_error` naming this variable -- NOT a shared slot: the
/// concurrent mode promises every issuing thread its own scratch, so a cap set below the real
/// concurrency fails loudly on that thread's first op rather than quietly sharing slot 0.
///
/// The default is generous because an unused slot costs one null pointer: a slot's polynomials,
/// streams and buffers are built on first use, so only the threads that actually issue ops pay
/// for a set (at N = 2^16, L = 25, dnum = 6 that is ~150 MiB of device memory per issuing thread
/// beyond the first, which uses the context's own scratch). Lower it to put a hard ceiling on that
/// footprint; raise it if the application issues from more threads than this.
int ScratchSlotCap();

/// @brief DIAGNOSTIC (mask test): persistent per-op tables AND a dummy
/// cudaMallocAsync/cudaFreeAsync pair at every table site (`FIDESLIB_PERSIST_CHURN=1`).
///
/// An earlier trace showed hot cross-lane reuse of table addresses at ~13,000 per run in
/// EVERY rep, passing and failing alike -- the stream-ordered allocator's normal, guaranteed-safe
/// behaviour, uncorrelated with the mismatch. So persisttables' 24/24 is suspected to be a TIMING
/// MASK: taking ~136k allocator calls off the hot path shifts the interleaving. This config keeps
/// the persistent buffers (the candidate "fix") and restores the allocator traffic (the candidate
/// "mask"). Clean => the buffer lifetime is what mattered. Failing at the control rate => it was
/// the traffic, persisttables is a mask, and the hunt continues.
bool PersistChurn();

/// @brief DIAGNOSTIC: cudaStreamSynchronize every SOURCE stream at
/// LimbPartition::evalLinearWSum entry (`FIDESLIB_WSUM_SRCSYNC=1`). An earlier bisection put the first
/// wrong value right after the fused weighted sum; if one of its sources is under-fenced (a
/// producer on a stream the entry waits never name), a full source sync closes it and nothing
/// else does. Clean => under-fenced wsum source. Failing => the wsum's inputs are sound and the
/// defect is in the kernel's own launch/stream or in the closing multiply.
bool WsumSrcSync();

/// @brief DIAGNOSTIC: take the PER-OP DEVICE TABLES out of the driver's
/// stream-ordered allocator (`FIDESLIB_PERSIST_TABLES=1`, and only when `FIDESLIB_CONCURRENT_OPS=1`).
///
/// WHAT IT IS FOR. An earlier bisection put the first wrong value right after a fused
/// weighted sum or the recursion's closing multiply, with copy, add and addScalar clean. What those
/// two ops have and the clean ones lack is a per-call device table -- a constants array or a digit
/// pointer table -- built with `cudaMallocAsync` on the op's stream, filled by an asynchronous
/// upload, read by the kernel, and released with `cudaFreeAsync`. That puts the table in the
/// DRIVER's memory pool, which is a different pool from the library's own: the `takesync` /
/// `noreuse` / `freesync` exclusions covered the library's pool and say nothing about this one.
///
/// With this set, those tables come from a per-(device, slot) ring of plain `cudaMalloc` buffers
/// that are never freed, so no table block ever changes hands. Clean => the table's lifetime in the
/// driver pool is the mechanism, and the fix is to give the tables an owner. Still failing =>
/// the class is excluded and the hunt moves to the kernels' own stream dependencies.
bool PersistOpTables();

/// @brief Whether a reduced-level bootstrap set is installed as SHARED VIEWS of the full-height
/// set (AddBootstrapPrecomputationShared, THE DEFAULT since 2026-10-03) instead of a re-encoded
/// copy (AddBootstrapPrecomputationReduced; `FIDESLIB_BOOT_SHARED_PRECOMP=0` is the kill switch).
/// Read once; independent of the concurrent mode. The views take no extra device memory and give the same outputs.
bool BootSharedPrecomp();

/// @brief Whether the two-iteration (Meta-BTS) bootstrap composes its glue ON THE DEVICE
/// (`FIDESLIB_METABTS_DEVICE=1`, opt-in) instead of reading each refresh back to the host and
/// running the scale-up / error subtraction / recombination in lbcrypto. The host glue costs three
/// device-to-host readbacks (two cudaDeviceSynchronize each) and one upload per refresh; an nsys
/// profile of a bootstrap-heavy run charged thousands of device-wide syncs to the refreshes.
/// THE DEFAULT since 2026-10-04 (faster end to end, error within tolerance);
/// `FIDESLIB_METABTS_DEVICE=0` is the kill switch and takes the previous host glue. Read once.
bool MetaBtsDevice();

/// @brief Whether a batched device encode (MakeCKKSPackedPlaintextDeviceMany) also runs the CKKS
/// inverse special FFT ON THE DEVICE (`FIDESLIB_DEVICE_IFFT=1`, opt-in): the host hands over the
/// slot values (N/2 doubles) instead of the N biased coefficients it computes today with
/// DiscreteFourierTransform::FFTSpecialInv + scale + llround per diagonal. The device transform is
/// the same Gentleman-Sande ladder over the same rotGroup/ksiPows tables, in round-to-nearest double
/// arithmetic without FMA contraction; the integers it rounds to are numerically, not necessarily
/// bit-for-bit, the host's. THE DEFAULT since 2026-10-04; `FIDESLIB_DEVICE_IFFT=0` is the
/// kill switch and takes the previous host transform. Read once.
bool DeviceIfft();

/// @brief Whether the device transform checks its coefficient magnitudes against OpenFHE's
/// MAX_BITS_IN_WORD (61) and throws where CKKSPackedEncoding::Encode would take the approxFactor
/// path (one stream sync per batch). Default on; `FIDESLIB_DEVICE_IFFT_CHECK=0` skips it.
bool DeviceIfftCheck();

/// @brief Whether a batched device encode grows its plaintexts' limbs CONSTANT (no per-limb NTT
/// scratch) and runs the encode NTT through one pooled scratch buffer per chunk
/// (`FIDESLIB_ENCODE_SCRATCH=1`, opt-in). Per plaintext that removes L scratch allocations, L
/// scratch frees, the auxptr memset and one pointer-table upload -- and with them ~2/3 of the pool
/// mutex acquisitions and event fences the prefetch workers pay (in an nsys profile of a
/// device-encode-heavy run, LimbPartition::generate took about half of the library's range time). The limbs'
/// contents are bit-identical. THE DEFAULT since 2026-10-04; `FIDESLIB_ENCODE_SCRATCH=0` is
/// the kill switch and takes the previous per-limb scratch. Read once.
bool EncodeScratch();

/// @brief Whether a batched device encode backs ALL of a plaintext's value limbs with ONE pooled
/// buffer (`FIDESLIB_PT_ARENA=1`, opt-in; implies the encode scratch) instead of one pool block per
/// limb: 1 pool operation per plaintext instead of L (each is the device's pool mutex plus an event
/// pair), at the cost of rounding L*N*8 bytes up to the pool's power-of-two class (5.5 -> 8 MB at
/// eleven 64-bit towers). The limbs are views into the buffer; the partition frees it whole. Limb
/// contents are bit-identical. Unset or 0 keeps one block per limb. Read once.
bool PlaintextArena();

/// @brief Whether the per-object CUDA driver queries that back VectorGPU's and Limb's constructor
/// asserts run in release builds too (`cudaPointerGetAttributes` on every view, `cudaGetDeviceCount`
/// + `cudaGetDevice` on every managed vector, `cudaGetDevice` on every limb -- ~60 driver-lock
/// acquisitions per device-encoded plaintext, contended by 24 workers; an nsys profile
/// put LimbPartition::generate at about 1 ms and the Plaintext constructor at 0.5 ms per
/// plaintext with nothing else left in them). Default ON (the shipped behaviour);
/// `FIDESLIB_DEVICE_CHECKS=0` skips them. Debug builds (no NDEBUG) always run them. Read once.
bool DeviceChecks();

/// @brief One persistent per-op table buffer of at least `bytes`. Only call when PersistOpTables().
///
/// `which` separates tables that are live AT THE SAME TIME inside one op (evalLinearWSum holds its
/// weights and its pointer table at once), so they can never be handed the same buffer.
void* OpTableBuffer(int device, size_t bytes, int which);

/// @brief DIAGNOSTIC / CANDIDATE FIX (R4, candidate C1): give each ISSUING SLOT its own
/// `cudaMemPool_t` for the per-op device tables (`FIDESLIB_SLOT_MEMPOOL=1`, concurrent mode only).
///
/// WHAT IT DISCRIMINATES. `cudaMallocAsync` / `cudaFreeAsync` draw from the DEVICE DEFAULT memory
/// pool -- one object shared by every stream and every host thread, whose three reuse policies
/// (same-stream fast path, `ReuseAllowOpportunistic`, `ReuseFollowEventDependencies`) this fork
/// never changes, so all three are on. The same-stream fast path is sound because the driver takes
/// program order on one stream to BE dependency order; under FIDESLIB_CONCURRENT_OPS that premise
/// is false at exactly one point -- two lane threads issue allocs, frees and copies to the SAME
/// pooled `cudaStream_t` handle (Stream::init hands out stream_pool[dev][idx++ % POOL_SIZE]). A
/// block one lane frees can then be handed to another lane while the first lane's kernel is still
/// reading it: shape-intact wrong values, nothing for a host sanitizer to see.
///
/// With this set, every per-op table on the op path comes from a pool PRIVATE to the issuing slot,
/// so a device block can never cross lanes. Nothing else moves: same stream handles, same number of
/// driver calls, same block lifetime, same upload. That is what no earlier config did --
/// `privstreams` varied handles AND timing, `persisttables` varied lifetime AND allocator,
/// `persistchurn` varied lifetime only. Clean => C1, and this IS the fix. Control rate => the
/// driver pool's cross-lane reuse is excluded and the hunt moves to the pageable staging (C2).
///
/// Slot 0 keeps the DEVICE DEFAULT pool, so the default (single-thread) mode is byte-identical.
bool SlotMemPool();

/// @brief This slot's private memory pool on `device`, created on first ask. nullptr when
/// FIDESLIB_SLOT_MEMPOOL is off, when `slot` is 0 (slot 0 is the device default pool) or when
/// either index is out of range. Called by the slot's OWN thread only -- from
/// CKKS::ContextData::opScratch when the slot's scratch is built, and from OpMallocAsync.
cudaMemPool_t SlotMemPoolFor(int device, int slot);

/// @brief Destroy every per-slot pool created so far. TEARDOWN ONLY, and under the same
/// precondition as CKKS::ContextData::destroyOpScratch (which calls it): no thread may be inside an
/// op. Idempotent.
void DestroySlotMemPools();

/// @brief Allocate one per-op device table of `bytes` on `s`.
///
/// `cudaMallocFromPoolAsync` out of SlotMemPoolFor(current device, ScratchSlot()) when
/// FIDESLIB_SLOT_MEMPOOL=1 and the caller is a concurrent lane with a slot of its own; plain
/// `cudaMallocAsync(&p, bytes, s)` -- i.e. the device default pool, exactly as before -- in every
/// other case, which includes the whole default mode.
void* OpMallocAsync(size_t bytes, cudaStream_t s);

/// @brief Release a table from OpMallocAsync on `s`.
///
/// ALWAYS a plain `cudaFreeAsync`, in both modes and on purpose: a stream-ordered block is returned
/// to the pool it was allocated from, so the free needs no pool argument, and keeping it
/// unconditional keeps the driver call count identical between `none` and `slotmempool`. It exists
/// so the pair reads as a pair at the call sites.
void OpFreeAsync(void* p, cudaStream_t s);

/// @brief DIAGNOSTIC: OBSERVE the driver pool's reuse of the per-op device
/// tables instead of perturbing it (`FIDESLIB_TABLE_TRACE=1`, concurrent mode only).
///
/// `persisttables` cleared the staged mismatch 8/8 against a 5/8 control, but so did
/// `privstreams` once and that turned out to be a TIMING MASK: taking allocator
/// calls off the hot path changes the interleaving whether or not the allocator is the defect.
/// This instrument answers the question directly. Every per-op table free records an event on
/// the freeing stream; every alloc that gets a previously-seen address checks whether that
/// event has been REACHED. A pending event means the driver recycled a block whose releasing
/// stream has not yet run the release -- so a kernel of the previous owner may still be reading
/// the table the new owner is about to overwrite. HOT_CROSS_LANE > 0 is the mechanism, observed.
/// Zero across the faulting reps means persisttables is a mask and the hunt continues.
bool TableTrace();
void TableTraceAlloc(void* p, const char* site);
void TableTraceFree(void* p, cudaStream_t s);
void TableTraceReport();

/// @brief DIAGNOSTIC / CANDIDATE FIX: give each ISSUING SLOT its own entry
/// in the fused key switch's graph-exec cache (`FIDESLIB_SLOT_GRAPH_CACHE=1`, concurrent mode only).
///
/// `LimbPartition::modup_ksk_moddown_mgpu` caches a captured graph per (device, parameters, level,
/// moddown) -- and NOT per thread. One cache entry owns one `digits` device buffer and one graph
/// captured against ONE lane's scratch buffers. Every lane that finds the entry writes its own
/// digit pointer table into that shared buffer and replays that shared graph, so a lane's kernels
/// can read another lane's pointers. They are valid pointers to the wrong lane's limbs, which is
/// why the output is shape-intact but wrong and why no sanitizer fires. Stage 5b's lock made the
/// map's insert thread-safe; it did not make an entry's CONTENTS per-lane.
///
/// With this set the key carries ScratchSlot(), so each issuing thread gets its own graph and its
/// own digits buffer. Clean => that sharing is the defect and this is the fix, modulo the memory
/// it costs (one captured graph per slot per level). Still failing => the sharing is not the whole
/// story and the remaining suspect is the per-op table class below.
bool SlotGraphCache();

/// @brief DIAGNOSTIC: drain the device at every op boundary while the concurrent mode is on
/// (`FIDESLIB_CONCURRENT_OPS_SYNC=1`, and only when `FIDESLIB_CONCURRENT_OPS=1`). Read once.
///
/// WHAT IT IS FOR. When ops issued from several host threads produce a wrong result, the cause is
/// one of two kinds and they need opposite fixes:
///
///   * a MISSING DEVICE DEPENDENCY -- some kernel is allowed to start before the kernel that
///     writes what it reads, because a fence between two streams is absent or bound to a stale
///     event. Host state is fine; the stream graph is wrong.
///   * SHARED HOST STATE -- two threads mutating one object (a cache, a counter, a buffer the
///     library thought only one op could be inside at a time). The stream graph is irrelevant.
///
/// Draining the device between ops makes every possible stream order legal, so it removes the
/// first kind entirely and leaves the second untouched. If a reproducer stops failing with this
/// set, the defect is a missing dependency; if it keeps failing, it is host state. That is the
/// whole purpose -- this is an instrument, not a mode to ship behind: it destroys every bit of
/// overlap the library has and can make a pass many times slower.
bool ConcurrentOpsSync();

/// @brief One such drain. A no-op unless ConcurrentOpsSync(); see it for what this is for.
void ConcurrentOpsSyncPoint();

/// @brief The calling thread's lane in the device-memory pool's release handshake. Always 0 in the
/// default mode.
///
/// The pool recycles device memory, so a block must not be handed to its next owner while the
/// previous owner's kernels still read it. The handshake is one stream per lane: GPUfree fences
/// the freeing thread's LANE stream on the releasing stream, and GPUmalloc fences the new owner's
/// stream on the lane that released the block it just popped. With one lane (the default) that is
/// the single pool stream this library has always used. With several issuing threads a single
/// lane would make every allocation wait on EVERY thread's most recent free -- a false dependency
/// between independent chains, paid per limb -- so each thread takes its own.
///
/// Sized off FIDESLIB_SCRATCH_SLOTS, and separate from ScratchSlot() on purpose: a thread that
/// only allocates (an encode/upload worker) must not consume an op-scratch slot. Overflow is NOT
/// an error here -- a thread past the cap uses lane 0 and gets the conservative shared fence,
/// which is slower and never wrong.
int MemPoolLane();

/// @brief One size class of the device-memory pool, as MemPoolStatsSnapshot reports it.
struct MemPoolSizeClassStats {
	int bytes						  = 0;
	unsigned long long chunks_cut	  = 0; ///< 1 GiB (or MBs-sized) chunks cudaMallocAsync'd for this class.
	unsigned long long drains		  = 0; ///< Exhaustion points at which the lanes' freed lists were reclaimed.
	unsigned long long blocks_reclaimed = 0; ///< Blocks moved lane -> fresh across all of those drains.
	/// Exhaustion points at which the drain was SKIPPED by its device guard and the pool cut a
	/// chunk instead (the pre-drain behaviour). See the guard in drainFreedLanes: a drain is only
	/// sound when the calling thread is on device `id` and that device is the only one in play.
	unsigned long long drain_skips	  = 0;
	size_t fresh_blocks				  = 0; ///< Blocks sitting in the shared fresh tier right now.
	/// Of `fresh_blocks`, how many carry the no-fence sentinel lane -- i.e. were reclaimed by a
	/// drain and made quiescent by its device synchronize, so their next owner waits on nothing.
	/// A SUBSET of `fresh_blocks`, not an addition to it.
	size_t fresh_no_fence_blocks	  = 0;
	/// Blocks of THIS class sitting on the lanes' freed lists, summed over lanes. With
	/// `fresh_blocks` this is every block of the class the pool is holding, so at a quiescent
	/// point `fresh_blocks + lane_freed_blocks == chunks_cut * (blocks per chunk)`.
	size_t lane_freed_blocks		  = 0;
	/// Blocks of this class moved lane -> fresh by MemPoolReleaseAllLanes (the PASS BOUNDARY
	/// release), as opposed to `blocks_reclaimed`, which counts the same movement performed by the
	/// on-demand drain at an exhaustion point. Kept apart on purpose: the drain firing says the
	/// pool ran dry mid-pass, the boundary release firing says nothing at all about pressure, so
	/// summing them would destroy the one reading `drains` exists to give.
	unsigned long long boundary_blocks_moved = 0;
};

/// @brief What the pool has done, per device. Read under `mempool_lock`, so it is a consistent
/// snapshot and not a set of independently sampled counters.
///
/// WHAT IT IS FOR. In the concurrent mode a block is freed onto the FREEING lane's list and only
/// that lane reuses it, so a producer/consumer application (the prefetch workers device-encode
/// plaintexts on their lanes, the consumer thread frees them on its) moves blocks one way and the
/// producers' fresh tier empties forever. Before the drain below, the only answer to an empty
/// fresh tier was another chunk, and the pool grew until the card was full. These counters are how
/// a run says which of the two happened: `chunks_cut` is the footprint the pool claimed,
/// `drains` / `blocks_reclaimed` how much of it was recycled instead of cut, and
/// `lane_freed_blocks` where the un-reclaimed blocks are sitting, and `drain_skips` how often the
/// drain's device guard declined to run and left the pool cutting.
struct MemPoolStats {
	std::vector<MemPoolSizeClassStats> classes; ///< Ascending by `bytes`.
	/// Indexed by pool lane; summed over size classes. Only real lanes appear here: the no-fence
	/// sentinel is a tag on a block in the FRESH tier and is never a freed-list index.
	std::vector<size_t> lane_freed_blocks;
	unsigned long long chunks_cut		= 0;
	unsigned long long drains			= 0;
	unsigned long long blocks_reclaimed = 0;
	unsigned long long drain_skips		= 0;
	/// Calls to MemPoolReleaseAllLanes that actually ran (its device guard declining is counted in
	/// `boundary_skips`, not here), and the blocks they moved lane -> fresh, summed over classes.
	unsigned long long boundary_releases	 = 0;
	unsigned long long boundary_blocks_moved = 0;
	unsigned long long boundary_skips		 = 0;
};

/// @brief Move EVERY lane's freed blocks, of EVERY size class, into the shared fresh tier.
/// Returns how many blocks moved (0 if the device guard declined). No-op returning 0 outside the
/// concurrent mode, which has no lane pool.
///
/// WHAT IT IS FOR. `drainFreedLanes` answers the question "this size class is exhausted, is there
/// anything to take back?", and it answers it for ONE class at the instant a chunk would be cut.
/// That leaves the cross-CLASS strand untouched: blocks of class A sitting on 30 lanes cannot
/// become a chunk for class B, and no chunk can be released or re-sliced because the pool does not
/// record chunk bases. An application that has a QUIESCENT POINT -- a server between requests, a
/// harness between passes -- can do better than wait for exhaustion: at that point nothing is in
/// flight, so every freed block on every lane can be made fresh at once, and the next pass draws
/// from one shared tier instead of from ~30 private ones.
///
/// WHY THE SYNCHRONIZE IS THE SAME ARGUMENT AS THE DRAIN'S. A block moved into the fresh tier owes
/// its next owner nothing only if nothing anywhere on the device can still be reading it. GPUfree
/// fences only the FREEING stream, and a block's last reader may be a kernel on another stream
/// entirely, so the only fence that makes the claim true is `cudaDeviceSynchronize` -- exactly as
/// in drainFreedLanes, and the blocks are tagged `kNoFenceLane` for exactly the same reason. The
/// difference is only WHEN it is paid: the drain pays it under memory pressure, this pays it where
/// the caller has said the device is idle anyway.
///
/// THE SAME DEVICE GUARD APPLIES, for the same reason: `cudaDeviceSynchronize` covers the calling
/// thread's current device only, and a P2P reader on another card is outside it. The guard is
/// therefore `current device == id && cudaGetDeviceCount() == 1`; declining is safe (it returns 0
/// and the pool behaves exactly as it did before this existed) and is counted as `boundary_skips`
/// so that a run which is not being helped says so.
///
/// CALLER'S PRECONDITION, which this function cannot check: no thread is issuing device work on
/// `id` that it will need after this returns. The synchronize makes the blocks quiescent, but a
/// thread that allocates a block one instant later is not made wrong by it -- what WOULD be wrong
/// is calling this from inside a pass and stalling the very workers whose work the sync is waiting
/// on. It is a quiescent-point call.
size_t MemPoolReleaseAllLanes(int id);

/// @brief A consistent snapshot of device `id`'s pool counters.
MemPoolStats MemPoolStatsSnapshot(int id);

/// @brief The snapshot as ONE line of facts, for an application to print at the end of a pass.
std::string MemPoolStatsLine(int id);

/// @brief The snapshot in full: the one line above plus a row per size class and the lane lengths.
/// Takes ONE snapshot and formats every part of the report from it, so the summary line and the
/// per-class rows below it describe the same instant and cannot disagree.
std::string MemPoolStatsReport(int id);

/// @brief Whether the library prints MemPoolStatsReport itself when a Context is torn down
/// (`FIDESLIB_POOL_STATS=1`). Read once; off by default, so a library that is not being
/// investigated prints nothing. The counters are collected either way.
bool MemPoolStatsEnabled();

/// @brief How many times a NULL stream has reached the lane pool's concurrent branches.
///
/// A pooled allocate or free is the point at which a block changes hands between two host
/// threads, and the ONLY thing that orders the new owner behind the old one is the stream the
/// caller hands in: `GPUfree` fences the freeing lane's handshake stream on it, `GPUmalloc` makes
/// the taker wait on that handshake stream. A null stream defeats both --
/// `Stream::wait(cudaStream_t)` returns at its `s == 0` early-out without issuing anything, so
/// the block goes back on the free list owing nothing and the next lane's fence is against an
/// empty stream. That is the concurrent-bootstrap race, and its source is an owner whose
/// `Stream::ptr_` is null: a partition built with `def_stream` (`Stream::initDefault`), or a
/// moved-from `Stream` still referenced by a `Limb`.
///
/// Both sources are fixed (see `FIDESlib::CKKS::initStream`), so in the concurrent mode this
/// counter must stay at 0; it exists so that a REGRESSION is visible in a release log -- the
/// counting branch also prints one line, once per process -- instead of silently unfenced. The
/// counter is only ever touched inside the concurrent branches, so the default path neither reads
/// nor writes it and gains no atomic.
///
/// @return the running total (allocations and frees together).
unsigned long long PoolNullStreamEvents();

// ---- BLOCK PROVENANCE TRACER (FIDESLIB_POOL_TRACE=1) -------------------------------------------
//
// WHAT IT IS FOR. The concurrent-bootstrap race returns one whole polynomial wrong -- every
// coefficient of every limb, shape intact, first_bad_limb 0 -- in about one repetition in eight,
// on one lane of four. That signature says the block backing that polynomial was written by one
// lane while another lane still owned or was still reading it: a pool hand-off whose fence did
// not cover the previous owner's last READER. A lock bisection can only say which scope makes the
// symptom go away; this says, for the blocks that actually came back wrong, WHO held each one,
// ON WHICH STREAM, and WHICH FENCE the next owner took -- i.e. the interleaving itself.
//
// TWO RINGS, both preallocated and both bounded:
//   * a PER-BLOCK ring (last kPoolTraceBlockEvents events), keyed by device pointer, of the
//     block's own history: cut, take, free, drain-move, boundary-move, and the output marks a
//     lane leaves on the polynomial its bootstrap has just finished writing.
//   * a GLOBAL FENCE ring (last kPoolTraceFenceEvents events) of every record / wait /
//     record_and_wait in the fence paths, including the ones that took an early out.
// Both carry the SAME monotonic sequence number, so a block event and the fence event it caused
// are adjacent in one order and the two dumps interleave by `seq`.
//
// OFF BY DEFAULT, and off in the default (non-concurrent) mode whatever the variable says: every
// hook below begins with one cached-bool read and returns. Nothing is allocated, no lock is
// taken, and the traced paths are the paths they were.
//
// COST WHEN ON: one mutex acquire and one ring write per pooled allocate / free (the block map),
// one atomic increment and one slot write per fence (the fence ring). Blocks are inserted into
// the map the first time they are seen -- i.e. as a chunk is cut -- so the only allocation is
// during warm-up; steady state writes into slots that already exist.

/// @brief Whether the provenance tracer is on: `FIDESLIB_POOL_TRACE=1` AND `ConcurrentOps()`.
/// Read once. False makes every hook below a single predictable branch.
bool PoolTraceEnabled();

/// What happened to a pooled BLOCK.
enum class PoolTraceKind : unsigned char {
	kCut,		   ///< Sliced out of a freshly cut chunk (its first appearance).
	kTake,		   ///< Handed to an allocator by GPUmalloc. `consumer` is the taker's stream.
	kFree,		   ///< Released by GPUfree. `consumer` is the releasing stream.
	kDrainMove,	   ///< Moved lane -> fresh by drainFreedLanes, behind a device synchronize.
	kBoundaryMove, ///< Moved lane -> fresh by MemPoolReleaseAllLanes, behind a device synchronize.
	kOutput,	   ///< An op declared this block part of an output it has just finished writing.
};

/// Which fence path issued the event.
enum class PoolTraceFence : unsigned char {
	kRecord,		 ///< Stream::record -- an event recorded on this stream, no wait.
	kWaitStreamObj,	 ///< Stream::wait(Stream&) -- record on `other`, wait on `self`.
	kWaitRaw,		 ///< Stream::wait(cudaStream_t) -- record on `other`, wait on `self`.
	kRecordAndWait,	 ///< Stream::record_and_wait -- the pool hand-off pair, one critical section.
};

/// @brief Append one event to the ring of the block at `ptr`. No-op unless PoolTraceEnabled().
///
/// `consumer` is the stream the event is ABOUT (the taker's consumer stream for a take, the
/// releasing stream for a free, the writing stream for an output mark); `owner` is the pool
/// handshake stream that was fenced on, if any; `ev` the fence event that carried it, if any.
/// `lane` is the pool lane of the thread performing the event -- kNoFenceLane (-1) is allowed and
/// printed as such.
///
/// Callers inside the pool hold `mempool_lock[id]`; this function takes its OWN lock and no
/// other, so it can also be called (by the output mark) with no pool lock held. It never takes
/// `mempool_lock`, so it cannot invert against it.
void PoolTraceBlock(PoolTraceKind kind, const void* ptr, int bytes, int lane, const void* consumer, const void* owner, const void* ev);

/// @brief Append one event to the global fence ring. No-op unless PoolTraceEnabled().
/// `early_out` records that the call RETURNED WITHOUT FENCING; `lock_held` that `fence_lock` was
/// held across the driver calls. Lock-free: one atomic sequence increment and one slot write.
void PoolTraceFenceEv(PoolTraceFence kind, const void* self, const void* other, const void* ev, bool early_out, bool lock_held);

/// @brief Print the provenance ring of the block that contains `device_ptr`, oldest event first.
///
/// Matches the EXACT pointer first; failing that, the pooled block with the greatest base address
/// at or below `device_ptr` whose size class contains it -- a limb's data pointer can be an
/// interior offset of the block that backs it. Prints one `no provenance` line if neither
/// matches, so a dump never silently omits a pointer it was asked about.
void PoolTraceDumpFor(const void* device_ptr, std::ostream& os);

/// @brief Print the last `n` fence events, oldest first (clamped to the ring's capacity).
void PoolTraceDumpRecent(std::ostream& os, size_t n);

/// @brief One line naming the lanes that last took and last freed the block behind `device_ptr`,
/// for a dump that wants the verdict without the whole ring. Empty if the block is unknown.
std::string PoolTraceOwnersLine(const void* device_ptr);

/// @brief An EXTRA report the pool's failure path appends to its own, supplied by the layer above.
///
/// The pool lives below CKKS in this library's layering and cannot see a ContextData, but the
/// auxiliary-polynomial lists are device memory the pool is HOLDING LIVE at the moment it fails --
/// the single most useful thing to read next to "the fresh tier was empty and the drain reclaimed
/// nothing". So the Context registers a reporter here and the failure path calls it.
///
/// CONTRACT, and it is a failure-path contract, not a general one:
///   * The reporter is called from inside `mempool_lock`, on the thread that just failed to
///     allocate, immediately before that thread aborts. It MUST NOT allocate device memory and
///     MUST NOT BLOCK -- the thread may already hold any lock the reporter would want (a Context
///     that is constructing an auxiliary polynomial holds `aux_poly_lock` while GPUmalloc runs),
///     so a reporter that takes one must `try_lock` and degrade to what it can read.
///   * Last registration wins. Two live contexts are a debugging configuration, not a deployment
///     one, and a single pointer keeps this to the one store it has to be.
///   * Registering `nullptr` (what ~ContextData does) turns it back off.
using PoolFailureReporter = std::string (*)();
void SetPoolFailureReporter(PoolFailureReporter fn);

/// @brief Print, to stderr, everything the pool knows -- for EVERY device -- plus whatever
/// SetPoolFailureReporter registered. The failure path of the chunk cut calls this before it
/// aborts; nothing on a successful path calls it.
///
/// @param id_locked The device whose `mempool_lock` the CALLER ALREADY HOLDS, or -1 for none.
/// That device is formatted without re-taking the lock (it is not recursive, and re-taking it
/// would deadlock rather than report); every other device is snapshotted normally.
void MemPoolFailureReport(int id_locked, int bytes, const char* what);

void initGPUprop();
int GetTargetThreads(int id);

enum NVTX_CATEGORIES { NONE, LIFETIME, FUNCTION };

/// @brief Whether the LIFETIME NVTX bookkeeping is enabled (`FIDESLIB_NVTX_LIFETIME=1`). Read once.
///
/// The LIFETIME category keeps a per-message refcounted NVTX range in a global `std::map` guarded
/// by one global mutex (see CudaUtils.cu). Every CKKS::Plaintext, Ciphertext, KeySwitchingKey and
/// Context holds a `CudaNvtxRange my_range` of that category, so *constructing or destroying* any
/// of those value types takes that one mutex and builds several std::strings. A pipelined workload
/// (24 prefetch worker threads building device plaintexts -- ~306k per pass -- while the consumer
/// thread builds its own ciphertexts: 25 threads) therefore serializes all of them on a single
/// lock, for bookkeeping that only an NVTX profiler ever reads -- nothing in the library reads the
/// map. So it is opt-in: unset (the default) the LIFETIME path returns before the lock, before the
/// map and before the strings; set to `1` it behaves exactly as it did before. The FUNCTION
/// category is untouched -- it is a per-thread NVTX push/pop and takes no lock.
bool NvtxLifetimeEnabled();

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0);
void CudaNvtxStop(const std::string msg = "", NVTX_CATEGORIES cat = FUNCTION);

class CudaNvtxRange {
	const std::string msg;
	const NVTX_CATEGORIES cat;
	bool valid = true;

  public:
	explicit CudaNvtxRange(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0) : msg(msg), cat(cat) {
		// Gated here as well as inside CudaNvtxStart so a disabled LIFETIME range does not even pay
		// the by-value std::string copy of the call. FUNCTION is unaffected.
		if (cat != LIFETIME || NvtxLifetimeEnabled())
			CudaNvtxStart(msg, cat, val);
	}

	CudaNvtxRange(CudaNvtxRange&& r) noexcept : msg(r.msg), cat(r.cat) {
		this->valid = r.valid;
		r.valid		= false;
	}

	~CudaNvtxRange() {
		// Symmetric with the constructor: the flag is read once per process, so a range that skipped
		// its Start always skips its Stop and the refcounts in `lifetimes_map` stay balanced.
		if (valid && (cat != LIFETIME || NvtxLifetimeEnabled()))
			CudaNvtxStop(msg, cat);
	}
};

int getNumDevices();

void CudaHostSync();

/// @brief Called by every CUDA failure path right before its exit(0). DIAGNOSTIC: with
/// FIDESLIB_FAILURE_HOLD_S=<seconds> the failing thread waits that long first, so a GPU core dump
/// the driver is still writing (CUDA_ENABLE_COREDUMP_ON_EXCEPTION) is whole when the process ends.
/// Unset, it returns at once.
void breakpoint();

// TODO: Remove the cudart unloading.
#define CudaCheckErrorMod                                                                    \
	do {                                                                                     \
		cudaDeviceSynchronize();                                                             \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e == cudaErrorCudartUnloading) {                                                 \
			exit(0);                                                                         \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {             \
                                                                                             \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

#define CudaCheckErrorModMGPU                                                                \
	do {                                                                                     \
		cudaStreamSynchronize(0);                                                            \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {                    \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

// TODO: FIX THE CUDARTUNLOADING ERROR, IT HAPPENS WHEN THE LIBRARY IS BEING UNLOADED, CAN BE IGNORED FOR NOW
#define CudaCheckErrorModNoSync                                                                                          \
	do {                                                                                                                 \
		/*cudaDeviceSynchronize();*/                                                                                     \
		cudaError_t e = cudaGetLastError();                                                                              \
		if (e == cudaErrorCudartUnloading) {                                                                             \
			exit(0);                                                                                                     \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && e != cudaErrorGraphExecUpdateFailure) { \
			void* array[10];                                                                                             \
			size_t size;                                                                                                 \
			size = backtrace(array, 10);                                                                                 \
			backtrace_symbols_fd(array, size, STDERR_FILENO);                                                            \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                             \
			FIDESlib::breakpoint();                                                                                      \
			exit(0);                                                                                                     \
		}                                                                                                                \
	} while (0)

#define NCCLCHECK(cmd)                                                                              \
	do {                                                                                            \
		ncclResult_t res = cmd;                                                                     \
		if (res != ncclSuccess) {                                                                   \
			printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
			exit(EXIT_FAILURE);                                                                     \
		}                                                                                           \
	} while (0)

class Event;

extern std::map<void*, int> free;

class Stream {
  private:
	cudaStream_t ptr_ = nullptr;
	bool owned_		  = false; ///< ptr_ was created by this object (kPrivStreams) and is destroyed with it
	/// Guards the record-then-wait pair below, and is TAKEN ONLY in the concurrent mode.
	/// A fence MUTATES THE STREAM BEING WAITED ON:
	/// `wait(Stream& s)` records `s`'s single event on `s`'s stream and flips `s.updated`. So two
	/// host threads that fence on the same producer -- two concurrent ops reading the same
	/// bootstrap precomputation plaintext, say -- would race on one `cudaEvent_t` and one `bool`,
	/// and could issue `cudaStreamWaitEvent` against an event the other thread had not yet
	/// recorded, i.e. drop the dependency. Uncontended this is one mutex acquire against two
	/// driver calls.
	std::mutex fence_lock;

	/// CONCURRENT MODE ONLY. One fence event PER POOL LANE, so that `ev` stops being a resource
	/// two host threads share.
	///
	/// A fence records an event on the PRODUCER's stream and then waits on it. With one event per
	/// Stream object, every lane that fences on a SHARED producer -- a bootstrap precomputation
	/// plaintext's `s` (LimbPartitionBatch.cu:433, :521), a key-switching key's stream
	/// (Bootstrap.cu:235, :239; LimbPartitionMGPU.cu:729-731) -- records and waits on the SAME
	/// `cudaEvent_t`. That is correct only while the record and the wait are one critical section
	/// on `fence_lock` AND nothing else touches the event; it makes every such fence a
	/// process-wide serialization point, and it leaves exactly one unguarded pair to get wrong
	/// (the pool handshake, see `record_and_wait`). A lane-private event removes the sharing
	/// instead of locking around it: two lanes fencing on one producer now record two different
	/// events at two points of the producer's stream and wait on their own.
	///
	/// Lazily grown under `fence_lock`, indexed by `MemPoolLane()` (which is 0 and never grows in
	/// the default mode). NEVER touched unless `ConcurrentOps()`: `fenceEvent()` returns `ev` at
	/// its first line otherwise, so the default path allocates nothing and records the same event
	/// on the same stream it always did.
	std::vector<cudaEvent_t> lane_ev;

	/// The event THIS fence must record and wait on: `ev` in the default mode, the calling
	/// thread's lane event in the concurrent mode. Creates the lane's event on first use.
	/// CALLER HOLDS `fence_lock` (it mutates `lane_ev`).
	cudaEvent_t fenceEvent();

  public:
	cudaEvent_t ev = nullptr;
	/// Whether `ev` currently holds a record of this stream's work. Atomic because ptr() clears it
	/// off the lock on the kernel-launch path; the record/wait pair itself runs under fence_lock.
	std::atomic<bool> updated = false;
	// Event ev;

	void init(int priority = 0);

	cudaStream_t ptr() {
		updated = false;
		return ptr_;
	}

	/// @brief The underlying stream WITHOUT `ptr()`'s side effect of clearing `updated`.
	///
	/// `ptr()` is the kernel-launch accessor and clearing `updated` is part of what launching
	/// means; a reader that only wants to NAME the stream (the provenance tracer, which prints
	/// which stream a fence or a hand-off ran on) must not perturb that flag, or the instrument
	/// changes the thing it is measuring. Nothing but tracing and assertions calls this.
	cudaStream_t raw() const { return ptr_; }

	void initDefault();

	// void wait(const Event &ev) const;
	void wait(Stream& s, bool external = false);
	void wait(cudaStream_t s);

	Stream();

	Stream(Stream& s) = delete;

	Stream(const Stream& s) = delete;

	Stream& operator=(const Stream&) = delete;

	Stream(Stream&& s) noexcept;

	~Stream();

	void record(bool external = false);

	/// @brief Record this stream's fence event and make `consumer` wait on it, AS ONE CRITICAL
	/// SECTION on `fence_lock`.
	///
	/// This is the device memory pool's handshake (GPUmalloc), which used to be written as
	/// `owner.record(); cudaStreamWaitEvent(stream, owner.ev);`. Those two statements read one
	/// `cudaEvent_t` across a gap in which `fence_lock` is NOT held -- the pair was serialized by
	/// `mempool_lock[id]` instead, a different mutex from the one every other reader and writer of
	/// that event takes. Splitting the pair also cannot work at all once the event is lane-private
	/// (`fenceEvent()`), because the caller can no longer name the event that was recorded.
	///
	/// Default mode: no lock is taken and the driver calls are the same two, in the same order,
	/// on the same `ev`, as the two statements it replaces.
	///
	/// @return the event the pair used -- `ev` in the default mode, this lane's event in the
	/// concurrent one -- so that the POOL can record in the block's provenance ring WHICH fence
	/// the taker took. The only caller (GPUmalloc) ignores it unless tracing is on; the driver
	/// calls and their order do not depend on it.
	cudaEvent_t record_and_wait(cudaStream_t consumer);

	void wait_recorded(const Stream& s);

	void capture_begin();

	void capture_end();
};

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache = false);
void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache = false);
/// @brief The default shared pool keyed by the EXACT byte count (a multiple of 4096 in [64 KB, 2 GB)):
/// the plaintext arena's class, so resident 5.5 MB arenas cost 5.5 MB each, not the 8 MB pow2
/// class. A block from here is returned ONLY through GPUfreeExact.
void* GPUmallocExact(int id, size_t bytes, cudaStream_t stream);
void GPUfreeExact(void* ptr, int id, size_t bytes, cudaStream_t stream);

// ---- Pinned host-to-device staging ring ----
//
// Host-to-device copies out of ordinary pageable memory are staged by the driver through an
// internal pinned buffer, which costs an extra host copy and caps the transfer at roughly half the
// bandwidth a pinned source reaches. Staging through our own pinned buffer instead recovers that.
//
// A single staging slot is not enough: it has to be free again before the next caller can use it,
// and making it free means waiting for the transfer, which serializes the stream and gives back in
// lost overlap everything the bandwidth bought (that is exactly what the first version measured).
// So the arena is a RING of slots per device, each with its own completion event. Acquiring a slot
// waits on that slot's event, never on the stream, so the host blocks only once it has lapped the
// ring -- by which point the DMA engine, not the staging, is the limit.
//
// Opt-in via FIDESLIB_PINNED_STAGING=1: the ring adds pinned allocations, so it stays off until GPU
// CI has exercised it. FIDESLIB_PINNED_SLOTS sets the ring depth (default 8).

/// @brief Whether the pinned staging ring is enabled (FIDESLIB_PINNED_STAGING=1). Read once.
bool PinnedStagingEnabled();

/// @brief Ring depth per device: FIDESLIB_PINNED_SLOTS, default 8, clamped to [1, 64]. Read once.
size_t PinnedStagingSlots();

/// @brief Upload `bytes` from host `src` to device `dst` through this device's pinned staging ring.
///
/// Reserves a slot, waits on that slot's completion event (the H2D copy issued out of it one lap
/// ago), copies `src` into it on the host, then issues the H2D copy on `stream` and records the
/// slot's event behind it. `src` therefore only has to survive this call, exactly as with a
/// pageable cudaMemcpyAsync; the transfer itself stays asynchronous, again exactly as with a
/// pageable cudaMemcpyAsync, and is ordered against the rest of `stream` in the usual way.
///
/// Slots are grown on demand to the largest request seen on that device (callers size it from
/// context parameters -- one limb is N * sizeof(T), 512 KB at logN = 16); nothing here is hardcoded.
///
/// Thread-safe: concurrent callers on the same device take different slots, and the arena lock is
/// held only while a slot is reserved and released -- never across the wait, the allocation or
/// either copy. Callers on different devices share nothing. `device` must be the current CUDA
/// device (the slot's event is recorded on `stream`, which belongs to it).
///
/// @return true if the upload was performed. false means the ring declined (disabled, device index
/// out of range, device not current, or a pinned allocation / event creation failed -- the last of
/// which is sticky for that device) and the CALLER MUST issue the copy itself straight out of
/// `src`.
bool PinnedStagingUpload(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream);

/// @brief Host -> device upload of a small host buffer (pointer table, constant vector, limb data)
/// on `stream`: through the pinned staging arena when FIDESLIB_PINNED_STAGING=1, else the plain
/// pageable cudaMemcpyAsync it has always been.
///
/// WHY ONE FUNCTION. Every upload the library makes on a partition stream used to be a bare
/// `cudaMemcpyAsync(dst, host_vector.data(), bytes, ..., s.ptr())` from PAGEABLE memory. For pageable
/// sources the runtime stages the bytes itself and performs the transfer on the stream; that path is
/// the one behaviour of a stream that two Stream OBJECTS holding the same cudaStream_t change for each
/// other (the concurrent-bootstrap mismatch exists exactly when two live objects share a
/// handle, and no fence or event in this library depends on the handle). Routing every upload through
/// the arena -- pinned source, explicit event per slot -- takes the runtime's pageable staging out of
/// the concurrent path. `PinnedStagingUpload` declines (returns false) when the flag is off, the arena
/// failed, or the device is not current; the pageable copy is the fallback in every case.
void UploadH2D(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream);

/// @brief Whether the fused key switch's digit-table uploads take the pinned arena
/// (`FIDESLIB_PINNED_MGPU=1`). Read once. DIAGNOSTIC (R4, candidate C2).
bool PinnedMGPU();

/// @brief UploadH2D for the three digit-pointer tables of the FUSED KEY SWITCH
/// (LimbPartitionMGPU.cu: dotKSKfusedMGPU, fusedHoistRotate, modup_ksk_moddown_mgpu). Takes the
/// pinned staging arena when EITHER FIDESLIB_PINNED_STAGING=1 or FIDESLIB_PINNED_MGPU=1, and the
/// plain pageable cudaMemcpyAsync otherwise.
///
/// WHY A SECOND GATE. `0f72c59` routed every H2D the library makes through the arena -- except
/// these three, because it touched LimbPartition.cu, LimbPartitionBatch.cu, RNSPoly.cpp and
/// VectorGPU.cu and NOT LimbPartitionMGPU.cu. A later run measured `pinned` (7/8) against `none`
/// (7/8) and recorded pageable staging as excluded -- but the op class the earlier bisection fingered, the
/// fused key switch, still had a raw pageable `cudaMemcpyAsync` on its digit table throughout that
/// run. The hypothesis was never tested on the sites it was about. The runtime stages a pageable
/// H2D through its own per-stream internal buffers (a stream sync, then a staging copy, then a
/// DMA, with small payloads pushed inline through the command buffer); two host threads driving
/// that on ONE cudaStream_t touch driver-internal state that no fence in this library names.
///
/// A SEPARATE variable rather than widening FIDESLIB_PINNED_STAGING, so `pinnedmgpu` covers these
/// three sites ALONE and its result is not confounded with the 20 sites `pinned` already covered.
void UploadH2DMGPU(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream);

/// @brief DIAGNOSTIC (stall). FIDESLIB_STALL_US=N launches an N-microsecond spin kernel on the WAITING
/// stream after every fence in the concurrent mode; FIDESLIB_STALL_IN="modup,keyswitch" restricts it to
/// fences issued while the thread is inside those ConcurrentOpsDiagScope kinds ("all" = every fence).
/// FIDESLIB_DEVSYNC_IN="modup_digits" (diagnostic) instead calls cudaDeviceSynchronize() on
/// ENTRY to every scope of the listed kinds; unset = off.
///
/// WHY. The concurrent-bootstrap mismatch exists exactly when two live Stream objects share
/// a cudaStream_t, no fence in this library depends on the handle, and the runtime's pageable staging is
/// not it (9l). What a shared handle DOES do is STALL: a cudaStreamWaitEvent one lane inserts delays every
/// later kernel another lane launches on the same handle. A missing dependency between two of a lane's own
/// streams stays hidden while the producer finishes before the consumer arrives -- always, with one issuing
/// thread -- and opens when the producer's stream is stalled. If that is the mechanism, stalling the
/// partition streams on purpose drives the mismatch rate towards 100%, WITH private streams too, and
/// restricting the stall to one op kind names the op that lacks the fence.
void DiagStall(cudaStream_t stream);

// ---- Pinned GATHER staging ring (batched device encode) ----
//
// The ring above stages ONE contiguous source. A batched device encode has `n` equally sized
// coefficient vectors that live in `n` unrelated host allocations and have to arrive on the device
// as one contiguous buffer, so its host copy is a GATHER: n memcpys into consecutive offsets of a
// single slot, then ONE H2D out of it. Doing that through PinnedStagingUpload would need the caller
// to build the contiguous host image first -- a second full copy of the batch -- and would drag the
// limb ring's slots up to the batch size (they grow to the largest request ever seen on the
// device), so the gather gets its OWN arena with its own depth.
//
// Enabled by default (FIDESLIB_PINNED_GATHER=0 turns it off), unlike the limb ring: the only caller
// is the batched device encode, which is itself opt-in, so nothing that runs today can reach it.
// FIDESLIB_PINNED_GATHER_SLOTS sets the depth (default 8, clamped to [1, 64]).

/// @brief Whether the pinned gather ring is enabled (FIDESLIB_PINNED_GATHER != 0). Read once.
bool PinnedGatherEnabled();

/// @brief Gather-ring depth per device: FIDESLIB_PINNED_GATHER_SLOTS, default 8, clamped to [1, 64].
size_t PinnedGatherSlots();

/// @brief Gather `n` host buffers of `bytes_each` into one pinned slot and issue ONE H2D to `dst`.
///
/// `srcs[i]` lands at `dst + i * bytes_each`. Same slot discipline, same lifetime contract and the
/// same decline semantics as PinnedStagingUpload: every `srcs[i]` only has to survive this call,
/// the transfer stays asynchronous and stream-ordered within `stream`, and a false return means the
/// CALLER MUST perform the copy itself (the caller then owes its own contiguous host image).
///
/// @return true if the gather and the upload were performed.
bool PinnedStagingUploadGather(void* dst, const void* const* srcs, size_t n, size_t bytes_each, int device, cudaStream_t stream);

} // namespace FIDESlib
#endif // FIDESLIB_CUDAUTILS_CUH
