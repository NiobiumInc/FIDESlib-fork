//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_CKKS_CONTEXT_CUH
#define FIDESLIB_CKKS_CONTEXT_CUH

#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "Parameters.cuh"
#include "RNSPoly.cuh"

#include <array>
#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef NCCL
#include "nccl.h"
#endif

namespace FIDESlib::CKKS {

struct Precomputations {
	std::vector<Constants> constants;
	std::unique_ptr<Global> globals;
	/// Bootstrap precomputations, keyed by (slots, levelsToDrop). levelsToDrop 0 is the
	/// full-height precomputation every ordinary Bootstrap uses; a non-zero key holds the
	/// same transforms re-encoded `levelsToDrop` towers lower, which lets Bootstrap raise
	/// to a shorter modulus chain (see Bootstrap(..., levelsToDrop)).
	std::map<std::pair<int, int>, BootstrapPrecomputation> boot;
	std::vector<RNSPoly> auxPoly;
	std::map<int, RNSPoly> monomialCache;
#ifdef NCCL
	std::map<int, ncclComm_t*> dev_to_communicator;
#else
	std::map<int, void*> dev_to_communicator;
#endif
	struct KeyPrecomputations {
		std::unique_ptr<KeySwitchingKey> eval_key;
		std::map<int, KeySwitchingKey> rot_keys;
	};

	std::map<KeyHash, KeyPrecomputations> keys;
};

enum RESCALE_TECHNIQUE { NO_RESCALE, FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT };

extern std::atomic_uint64_t next_uid;

class ContextData {
  public:
	static constexpr const char* loc{ "Context" };
	CudaNvtxRange my_range;
	Parameters param;
	Precomputations precom;
	const int logN;
	const int N;
	const RESCALE_TECHNIQUE rescaleTechnique;
	const int& L;
	const int logQ;
	int batch;
	const std::vector<int> GPUid;
	const int& dnum;
	std::vector<std::vector<int>> GPUdigits;
	const std::vector<PrimeRecord>& prime;
	std::vector<std::vector<LimbRecord>> meta;
	const std::vector<int> logQ_d;
	const int& K;
	const int logP;
	const std::vector<PrimeRecord>& specialPrime;

	std::vector<std::vector<LimbRecord>> specialMeta; // Make const maybe
	std::vector<std::vector<LimbRecord>> splitSpecialMeta;
	std::vector<std::vector<std::vector<LimbRecord>>> decompMeta; // Make const maybe
	std::vector<std::vector<std::vector<LimbRecord>>> digitMeta;  // Make const maybe
	std::vector<LimbRecord> gatherMeta;

	const std::vector<dim3> limbGPUid;
	const std::vector<int> digitGPUid;

#ifdef NCCL
	ncclUniqueId communicatorID;
	std::vector<ncclComm_t> GPUrank;
#else
	std::vector<int> GPUrank;
#endif

	std::unique_ptr<RNSPoly> key_switch_aux				= nullptr;
	std::unique_ptr<RNSPoly> key_switch_aux2			= nullptr;
	std::array<std::unique_ptr<RNSPoly>, 2> moddown_aux = { nullptr };
	std::vector<Stream> top_limb_stream;
	std::vector<uint64_t*> top_limb_buffer;
	std::vector<void*> top_limb_buffer_handle;
	std::vector<VectorGPU<void*>> top_limbptr;

	std::vector<Stream> top_limb_stream2;
	std::vector<uint64_t*> top_limb_buffer2;
	std::vector<void*> top_limb_buffer2_handle;
	std::vector<VectorGPU<void*>> top_limbptr2;

	std::vector<std::vector<Stream>> gatherStream;
	std::vector<std::vector<Stream>> digitStream;
	std::vector<std::vector<std::vector<Stream>>> digitStreamForMemcpyPeer;
	std::vector<std::vector<Stream>> digitStream2;

	// std::vector<RNSPoly> key_switch_digits;
	bool canP2P = false;
	std::list<uint64_t*> free_limb;

	/// PER-THREAD OP SCRATCH. Everything in here used to be ONE object per context, written by
	/// whichever op was running: `key_switch_aux`/`key_switch_aux2` (every key switch,
	/// relinearization, rotation, conjugation and the hoisted rotations), `moddown_aux` (every
	/// mod-down and every rescale), `precom.monomialCache` (multMonomial, i.e. every bootstrap),
	/// the `top_limb_*` rescale scratch and `digitStream` (the mod-up scheduling stream per
	/// digit). Two host threads issuing ops on the SAME context therefore overwrote each other's
	/// intermediate polynomials. One `OpScratch` per issuing thread removes that, WITHOUT changing
	/// what any single op computes: the same kernels run on the same values, only into a set of
	/// buffers nobody else is using.
	///
	/// Slot 0 is the context's own members, i.e. bit-for-bit today's behaviour and today's cost
	/// for an application that issues from one thread; a slot is created only when a SECOND
	/// thread actually asks for scratch, and each polynomial inside it is still built lazily on
	/// first use. Threads are handed slots by `ScratchSlot()` (CudaUtils), capped by
	/// FIDESLIB_SCRATCH_SLOTS.
	///
	/// THE PER-SLOT FAMILY, COMPLETE. Everything an op reads or writes that is not reachable from
	/// its own operands now lives here and is reached through the accessor of the same name:
	///   - the key-switch auxiliary polynomials  `key_switch_aux`, `key_switch_aux2`
	///       -> getKeySwitchAux(), getKeySwitchAux2()
	///   - the mod-down auxiliary polynomials    `moddown_aux`    -> getModdownAux(n)
	///   - the rescale top-limb scratch          `top_limb_*`, `top_limbptr*`
	///       -> getTopLimbStream/2(), getTopLimbBuffer/2(), getTopLimbPtr/2()
	///   - the monomial cache                    `monomialCache`  -> getMonomialCache()
	///   - the auxiliary-polynomial free list    `auxPoly`        -> auxPolyList()
	///   - the mod-up scheduling stream          `digitStream`    -> getDigitStream(d, id)
	///   - the key-switch launch stream          `digitStream2`   -> getDigitStream2(d, id)
	///
	/// THE RULE: inside an op, EVERY read of a launch stream or of scratch goes through the slot
	/// accessor -- never through the context member directly. A member read is not merely slower
	/// under FIDESLIB_CONCURRENT_OPS, it is wrong twice over: it queues one lane's kernels behind
	/// another's on a single FIFO stream, and where the op LAUNCHES on that stream (which
	/// `digitStream2` does -- `fusedDotKSK_2_` and the mod-down NTTs write their output on it) it
	/// also breaks the device pool's free/recycle ordering, because GPUfree fences only the stream
	/// it is handed. Both bugs already fixed in this fork were exactly this: doubleRescaleMGPU
	/// reading the `top_limb_*` members (LimbPartitionMGPU.cu:201-205) and the MGPU key-switch
	/// reading `digitStream`/`digitStream2` (whole-polynomial mismatch, every limb wrong, about
	/// one repetition in eight at four lanes).
	///
	/// Adding a new stream or scratch object to an op means adding it HERE with an accessor, not
	/// adding a member to the context.
	struct OpScratch {
		std::unique_ptr<RNSPoly> key_switch_aux				= nullptr;
		std::unique_ptr<RNSPoly> key_switch_aux2			= nullptr;
		std::array<std::unique_ptr<RNSPoly>, 2> moddown_aux = { nullptr };
		std::map<int, RNSPoly> monomialCache;
		// The rescale top-limb scratch, in the SAME two-set, 2 x GPUid.size() layout as the
		// context's own members above (top_limb_* and top_limb_*2), so every reader of the
		// context set can read a slot's set through the getters below with the same indices.
		std::vector<Stream> top_limb_stream;
		std::vector<uint64_t*> top_limb_buffer;
		std::vector<VectorGPU<void*>> top_limbptr;
		std::vector<Stream> top_limb_stream2;
		std::vector<uint64_t*> top_limb_buffer2;
		std::vector<VectorGPU<void*>> top_limbptr2;
		/// The auxiliary-polynomial free list of this slot's thread. In the concurrent mode a
		/// polynomial a thread returns is reused by THAT thread only: a recycled polynomial can still
		/// be read by kernels its previous owner enqueued on other streams, and within one thread the
		/// next use is later in that thread's own device order (the default-mode invariant); across
		/// threads it is not, and the new owner's first write raced the old owner's last read
		/// (Two concurrent bootstraps mismatch; one bootstrap at a time is clean.)
		std::vector<RNSPoly> auxPoly;
		std::vector<std::vector<Stream>> digitStream;
		/// The key-switch launch stream of `modup_ksk_moddown_mgpu` / `modupMGPU` / `moddownMGPU`,
		/// the twin of `digitStream` above. It is not just a scheduling stream: `fusedDotKSK_2_`
		/// and the post-mod-up NTTs LAUNCH on it, and the mod-down NTT writes the op's output on
		/// it, so two threads sharing it both serialized behind each other AND let a pooled block
		/// freed by one op be recycled while a kernel of the other was still writing it -- GPUfree
		/// fences only the stream it is handed (CudaUtils.cu, GPUfree), never this one.
		std::vector<std::vector<Stream>> digitStream2;
		bool device_ready = false;
	};
	std::vector<std::unique_ptr<OpScratch>> op_scratch;
	/// Guards the op_scratch VECTOR (slot creation), never the scratch contents: a slot belongs to
	/// exactly one thread, so its contents need no lock.
	std::mutex op_scratch_lock;
	/// Guards `precom.auxPoly`, the plain vector every Ciphertext constructor draws two polynomials
	/// from and every destructor returns them to.
	mutable std::mutex aux_poly_lock;

	/// Auxiliary-polynomial bookkeeping, all three written only inside `aux_poly_lock`'s critical
	/// sections (which are taken in the concurrent mode and are single-threaded in the default
	/// one), so plain counters are correct without adding an atomic to either path.
	///
	/// WHY THEY ARE HERE. These lists have the same producer/consumer shape the device-memory pool
	/// had: a Ciphertext takes two polynomials in its constructor and returns them in its
	/// destructor (Ciphertext.cpp:87, :100-108), and in the concurrent mode the list a thread draws
	/// from is its OWN slot's, so a polynomial built on one thread and destroyed on another lands
	/// on a list its maker never reads. `created` climbing while `returned` climbs on a different
	/// slot is that asymmetry, stated in numbers. This is INSTRUMENTATION ONLY -- nothing here
	/// changes which list a polynomial goes to.
	uint64_t aux_poly_created_	= 0; ///< A polynomial CONSTRUCTED because the slot's list was empty.
	uint64_t aux_poly_reused_	= 0; ///< A polynomial taken off the slot's list instead.
	uint64_t aux_poly_returned_ = 0; ///< A polynomial handed back to a slot's list.
	/// Of `returned`, the polynomials handed back to a list OTHER than the returning thread's own
	/// -- i.e. re-homed to the slot that created them (RNSPoly::aux_slot). In the default mode
	/// this is always 0. Under FIDESLIB_CONCURRENT_OPS it is the drift the routing fix absorbs:
	/// before it, every one of these landed on the destroying thread's list.
	uint64_t aux_poly_rehomed_	= 0;
	/// A polynomial DESTROYED by releaseParkedScratch(), i.e. taken off a slot's list and given
	/// back to the device pool rather than handed to another op. The one counter here that is not
	/// pure instrumentation: it is how much of `returned` the pass boundary took back, so
	/// `returned - released` is what the lists are still carrying across passes.
	uint64_t aux_poly_released_ = 0;

	OpScratch& opScratch(const int slot);
	std::vector<RNSPoly>& auxPolyList();
	/// The free list of a GIVEN slot, for returning a polynomial to the slot that created it:
	/// slot 0 (and any slot whose scratch is not materialised, or -1) answers the calling thread's
	/// list, so an unknown origin degrades to the behaviour this always had. Caller holds
	/// `aux_poly_lock` in the concurrent mode; this never materialises a slot's scratch.
	std::vector<RNSPoly>& auxPolyListFor(int slot);
	void ensureTopLimb(OpScratch& sc);
	void destroyOpScratch();

	//      std::array<Stream, 8> blockingStream;
	//      std::vector<std::vector<Stream>> asyncStream;
	RNSPoly& getKeySwitchAux();
	RNSPoly& getKeySwitchAux2();
	RNSPoly& getModdownAux(const int num);
	/// The mod-up scheduling stream for digit `d` on device `id`, of the calling thread's slot.
	Stream& getDigitStream(const int d, const int id);
	/// The key-switch launch stream for digit `d` on device `id`, of the calling thread's slot.
	Stream& getDigitStream2(const int d, const int id);
	/// Whether the two streams above may be per-slot at all: only with ONE GPU. See Context.cu.
	bool perSlotDigitStreams() const;
	/// The rescale top-limb scratch (stream + 1-entry pointer table) of the calling thread's slot.
	Stream& getTopLimbStream(const int id);
	VectorGPU<void*>& getTopLimbPtr(const int id);
	/// The second set and the raw buffers, same indexing as the members (id in [0, 2*GPUid.size())).
	/// Slot 0 returns the members; a slot > 0 returns its own set. EVERY reader of the rescale
	/// scratch must come through these: LimbPartition::doubleRescaleMGPU read the members directly
	/// (the path Ciphertext::rescale always takes, even on one GPU), which under
	/// FIDESLIB_CONCURRENT_OPS made every rescale of two threads share one staging buffer.
	Stream& getTopLimbStream2(const int id);
	uint64_t* getTopLimbBuffer(const int id);
	uint64_t* getTopLimbBuffer2(const int id);
	VectorGPU<void*>& getTopLimbPtr2(const int id);
	/// The monomial cache of the calling thread's slot (multMonomial).
	std::map<int, RNSPoly>& getMonomialCache();

	bool isValidPrimeId(const int i) const;

  public:
	ContextData(const Parameters& param_, const std::vector<int>& devs, const int secBits = 0);
	~ContextData();

	static int computeLogQ(const int L, std::vector<PrimeRecord>& primes);

	static const int& validateDnum(const std::vector<int>& GPUid, const int& dnum);

	static std::vector<std::vector<LimbRecord>>
	generateMeta(const std::vector<int>& GPUid, const int dnum, const std::vector<std::vector<int>> digitGPUid, const std::vector<PrimeRecord>& prime, const Parameters& param);

	static std::vector<int> computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta, const std::vector<PrimeRecord>& prime);

	static const int& computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, Parameters& param);

	static std::vector<std::vector<LimbRecord>>
	generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta, const std::vector<PrimeRecord>& specialPrime, const int ID0, const std::vector<int>& GPUid);

	static std::vector<std::vector<std::vector<LimbRecord>>>
	generateDecompMeta(const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> dnum, const std::vector<int>& vector, int L);

	static std::vector<std::vector<std::vector<LimbRecord>>> generateDigitMeta(const std::vector<std::vector<LimbRecord>>& meta,
	  const std::vector<std::vector<LimbRecord>>& splitSpecialMeta,
	  const std::vector<LimbRecord>& specialMeta,
	  const std::vector<std::vector<int>>& digitGPUid,
	  const std::vector<int>& GPUid);

	static std::vector<dim3>
	generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L, const std::vector<std::vector<LimbRecord>>& SPECIALmeta, int K);

	static std::vector<std::vector<int>> generateGPUdigits(const int dnum, const std::vector<int>& devs);
	static std::vector<std::vector<LimbRecord>> generateSplitSpecialMeta(std::vector<LimbRecord>& specialMeta, const std::vector<int> GPUid);
	static std::vector<LimbRecord> generateGatherMeta(const std::vector<std::vector<LimbRecord>>& meta, int L);

  public:
	std::vector<uint64_t> ElemForEvalMult(int level, const double operand, int level_in = -1);
	std::vector<uint64_t> ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg);
	std::vector<double>& GetCoeffsChebyshev();
	int GetDoubleAngleIts();
	void AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp);
	void AddBootPrecomputation(int slots, int levelsToDrop, BootstrapPrecomputation&& precomp);
	bool HasBootPrecomputation(int slots);
	bool HasBootPrecomputation(int slots, int levelsToDrop);
	BootstrapPrecomputation& GetBootPrecomputation(int slots);
	/// @brief The precomputation whose transform plaintexts sit `levelsToDrop` towers below
	/// the chain top. `levelsToDrop == 0` is the default set; a non-zero one must have been
	/// installed by AddBootstrapPrecomputationReduced first.
	BootstrapPrecomputation& GetBootPrecomputation(int slots, int levelsToDrop);
	void AddRotationKey(int index, KeySwitchingKey&& ksk);
	KeySwitchingKey& GetRotationKey(int index, const KeyHash& keyID);
	KeySwitchingKey& GetRotationKey(int index, const KeyHash& keyID, int slots, int& actual_index);
	bool HasRotationKey(int index, const KeyHash& keyID);
	void AddEvalKey(KeySwitchingKey&& ksk);
	bool HasEvalKey(const KeyHash& keyID);
	KeySwitchingKey& GetEvalKey(const KeyHash& keyID);
	int GetBootK();
	// int GetBootCorrectionFactor();
	static RESCALE_TECHNIQUE translateRescalingTechnique(lbcrypto::ScalingTechnique technique);
	void PrepareNCCLCommunication();
	const std::vector<int> generateDigitGPUid(std::vector<std::vector<LimbRecord>>& meta, const int L, const int dnum);

	bool hasAuxilarPoly() const;
	RNSPoly getAuxilarPoly();
	void returnAuxilarPoly(RNSPoly&& c);
	void trimAuxilarPoly(size_t size);
	void clearAuxilarPoly();
	/// @brief What releaseParkedScratch() gave back. Zeros, with `ran == false`, when it was a
	/// no-op (the default mode).
	struct ScratchRelease {
		/// Parked auxiliary polynomials destroyed, summed over slot 0 and every live slot.
		size_t polys_released = 0;
		/// Pooled blocks moved lane-freed -> shared fresh tier, summed over devices. Includes the
		/// blocks the polynomials above just freed, which is the point of doing them in this order.
		size_t blocks_moved = 0;
		/// Slots that held at least one parked polynomial. The drift this function exists to undo
		/// is one slot hoarding while the others starve, so the count of hoarding slots is the
		/// number worth printing next to the polynomial count.
		size_t slots_drained = 0;
		/// False = the concurrent mode is off and nothing was done, not "nothing needed doing".
		bool ran = false;
	};

	/// @brief AT A PASS BOUNDARY, give the device-memory pool back everything the concurrent mode
	/// has parked: every slot's auxiliary polynomials, and every pool lane's freed blocks.
	///
	/// THE PROBLEM. A Ciphertext draws two auxiliary polynomials in its constructor and returns
	/// them in its destructor, and the list is the CALLING THREAD'S slot's (see auxPolyList). An
	/// application that births intermediates on worker threads and destroys them on the thread
	/// that merges their results -- which is what a parallel-region scheduler does -- therefore
	/// moves polynomials one way: the merging thread's list grows without bound while the workers'
	/// stay empty and keep CONSTRUCTING (`aux_poly_created_` climbing against `aux_poly_reused_`
	/// is that asymmetry in numbers). Every parked polynomial is LIVE pool memory -- it is on no
	/// freed list, so no drain can reclaim it -- so the drift is a monotone claim on the card that
	/// nothing in the library ever gave back. At application scale that is what ran a 96 GB card out
	/// of memory in the third pass.
	///
	/// WHAT IT DOES, in this order and for this reason:
	///   1. Empties every slot's list. Destroying an RNSPoly returns its limbs through GPUfree, so
	///      the blocks land on the FREEING thread's lane freed list -- ours.
	///   2. Calls MemPoolReleaseAllLanes for each of this context's devices, which fences and then
	///      moves every lane's freed blocks of every class into the shared fresh tier. Step 1
	///      first is what puts the polynomials' blocks into step 2's reach in the same call
	///      instead of leaving them stranded on one lane until the next boundary.
	/// `precom.monomialCache` is deliberately NOT touched: it is bounded (two powers per slot),
	/// it is a CACHE rather than a free list, and dropping it makes the next bootstrap recompute.
	///
	/// DEFAULT MODE: returns immediately, `ran == false`. There is no per-slot list to drift (one
	/// list, one thread, recycled) and no lane pool to release, so the mode that has never had
	/// this problem does not pay a synchronize, a lock or a counter for it.
	///
	/// PRECONDITION, which this cannot check and the CALLER owns: no other thread is issuing ops
	/// on this context. It walks `op_scratch`, whose entries the allocating side creates with no
	/// lock at all (see opScratch and the note on auxPolyStatsReport -- taking `op_scratch_lock`
	/// here would exclude nothing and merely make the walk LOOK synchronized), and it destroys
	/// polynomials another thread's slot owns. A pass boundary -- the server between requests, a
	/// harness between passes -- satisfies it: the workers are joined or idle and the device is
	/// about to sit still anyway, which is also what makes the synchronize inside step 2 free.
	ScratchRelease releaseParkedScratch();

	/// @brief How many auxiliary polynomials are parked right now, summed over slot 0 and every
	/// live slot. The number releaseParkedScratch() is about, readable before and after it.
	///
	/// Same precondition as releaseParkedScratch, and for the same reason: it walks `op_scratch`.
	/// It is a quiescent-point read, not a monitor an app can poll from anywhere.
	size_t auxPolyParkedCount() const;
	/// @brief The auxiliary-polynomial lists in full: `created`/`reused`/`returned` and the length
	/// of every slot's list, one line per report. Instrumentation for the producer/consumer
	/// asymmetry described at the counters.
	///
	/// TEARDOWN ONLY. It walks `op_scratch`, which the allocating side (`opScratch`) writes with
	/// no lock because a slot belongs to one thread; the only point at which walking all of them
	/// is well defined is when no thread is issuing ops any more. Its single caller is
	/// ~ContextData(). Do not call it from a live pass -- see the definition for why no lock is
	/// taken to make that safe.
	std::string auxPolyStatsReport() const;
	/// @brief The same report, for the DEVICE-MEMORY POOL'S FAILURE PATH and nothing else.
	///
	/// The auxiliary-polynomial lists are device memory the pool is holding LIVE -- a polynomial on
	/// a slot's list is not on any freed list, so no drain can reclaim it -- which makes them the
	/// thing to read next to "the fresh tier was empty and the drain returned nothing". But the
	/// failure path reaches this from a thread that may ALREADY HOLD `aux_poly_lock`:
	/// getAuxilarPoly() takes it and then constructs an RNSPoly, i.e. allocates, i.e. can be the
	/// very call that failed. Blocking there would hang the process instead of reporting.
	///
	/// So this one never blocks. It try_locks; if it cannot get the lock it still prints the
	/// counters -- read without synchronisation, so in principle torn, and the report says so --
	/// and omits the slot-list walk, which needs the lock to mean anything. Nothing on a
	/// successful path calls it. Use auxPolyStatsReport() at teardown as before.
	std::string auxPolyStatsReportTry() const;
	/// @brief The text both of the above print. The CALLER has already decided what to do about
	/// `aux_poly_lock`; `with_lists` says whether it holds it, and therefore whether walking the
	/// slots' lists is defined. Not a lock-taking function, on purpose: one of its two callers
	/// must not take a lock at all.
	std::string auxPolyStatsText(bool with_lists) const;
	void clearAutomorphismKeys(const KeyHash& KeyID = {});
	void clearEvalMultKeys(const KeyHash& KeyID = {});
	void clearBootPrecomputation(int slots = -1);
	void clearParamSwitchKeys(const KeyHash& KeyID = {});

	friend Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs);
	friend void DeregisterCryptoContextGPU(const Parameters& param);
	friend void DeregisterCryptoContextGPU(Context cc);
	friend Context GetCurrentContext();
	friend void SetCurrentContext(Context&);
};

Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs);
void DeregisterCryptoContextGPU(const Parameters& param);
void DeregisterCryptoContextGPU(Context cc);
void DeregisterAllContexts();
Context GetCurrentContext();
void SetCurrentContext(Context& cc);
// Add/Has/GetSecretSwitchingKey removed — in-context ENCAPS stores both switching keys in the
// MAIN context's rotation-key map (2N-2 / 2N-4), so the dual-context key store is gone.

int32_t normalyzeIndex(int32_t index, int32_t slots, int32_t N);

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_CONTEXT_CUH