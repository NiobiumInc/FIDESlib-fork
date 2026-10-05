//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_LIMBPARTITION_CUH
#define FIDESLIB_CKKS_LIMBPARTITION_CUH

#include "Limb.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "PeerUtils.cuh"
#include <atomic>
#ifdef NCCL
#include "nccl.h"
#endif
namespace FIDESlib::CKKS {

extern bool MEMCPY_PEER;
extern bool GRAPH_CAPTURE;
extern bool PEER_ACCESS;

class LimbPartition {
  public:
	ContextData& cc;
	const uint64_t uid;
	int* level;
	const int id;
	const int device;
#ifdef NCCL
	const ncclComm_t rank; // For NCCL / RCCL
#else
	const int rank;
#endif
	Stream s;

	std::vector<LimbRecord>& meta;
	std::vector<LimbRecord>& SPECIALmeta;
	const std::vector<int>& digitid;
	std::vector<std::vector<LimbRecord>>& DECOMPmeta;
	std::vector<std::vector<LimbRecord>>& DIGITmeta;
	std::vector<LimbRecord>& GATHERmeta;

	std::vector<LimbImpl> limb;
	std::vector<LimbImpl> SPECIALlimb;
	std::vector<std::vector<LimbImpl>> DECOMPlimb;
	std::vector<std::vector<LimbImpl>> DIGITlimb;
	std::vector<LimbImpl> GATHERlimb;

	void** bufferAUXptrs;
	VectorGPU<void*> limbptr;
	VectorGPU<void*> auxptr;
	VectorGPU<void*> SPECIALlimbptr;
	VectorGPU<void*> SPECIALauxptr;

	std::vector<VectorGPU<void*>> DECOMPlimbptr;
	//  std::vector<VectorGPU<void*>> DECOMPauxptr;
	std::vector<VectorGPU<void*>> DIGITlimbptr;
	// std::vector<VectorGPU<void*>> DIGITauxptr;
	VectorGPU<void*> GATHERptr;

	uint64_t* bufferDECOMPandDIGIT	  = nullptr;
	uint64_t* bufferSPECIAL			  = nullptr;
	uint64_t* bufferLIMB			  = nullptr;
	uint64_t* bufferGATHER			  = nullptr;
	void* bufferDECOMPandDIGIT_handle = nullptr;
	void* bufferGATHER_handle		  = nullptr;

	/// @brief The byte size each of the four buffers above was allocated with. 0 == no buffer.
	///
	/// THE SIZE HAS TO TRAVEL WITH THE POINTER, for two reasons.
	///
	/// (1) `GPUfree` is not size-agnostic. Its prologue rewrites any `bytes < 64 * 1024` to 1024
	/// and forces `cache = true` (CudaUtils.cu, GPUfree), and every free list it can reach is
	/// keyed by `bytes` -- `size_to_memory[id][bytes]` in the default pool,
	/// `freed_mt[id][lane][bytes]` in the concurrent one. A free that passes 0 therefore takes a
	/// multi-megabyte block that `GPUmalloc` never pooled at all (cache defaults to false, so a
	/// >= 64 KiB allocation goes straight to `cudaMallocAsync`) and lists it in the 1024-byte size
	/// class, from which the next 1024-byte allocation will hand it out. Passing the TRUE size
	/// makes the two prologues agree: below 64 KiB both round to the same power of two and pool,
	/// at or above it neither pools, and the block is released exactly the way it was taken.
	///
	/// (2) The size is not always recoverable at the free site. `generateLimbSingleMalloc` sizes
	/// `bufferLIMB` from `meta.size()`, while `generateLimbConstant`, which frees it, computes its
	/// own `limbsize` as `getLimbSize(*level)` -- a different number for any polynomial not at the
	/// top level. Recomputing at the free site would be a second, quieter version of the same bug.
	///
	/// Every site that assigns one of the four pointers assigns its size field in the same
	/// statement, and every free site asserts the field is set before using it.
	size_t bufferDECOMPandDIGIT_bytes = 0;
	size_t bufferSPECIAL_bytes		  = 0;
	size_t bufferLIMB_bytes			  = 0;
	/// Whether `bufferLIMB` came from the exact-size pool class (GPUmallocExact: generateLimbArena) or
	/// from the non-pooled path (generateLimbSingleMalloc). The free MUST take the path the allocation
	/// took: cudaFreeAsync on a pointer inside a pool chunk is 'invalid argument' (observed in testing).
	bool bufferLIMB_pooled			  = false;
	size_t bufferGATHER_bytes		  = 0;

	/*
	LimbPartition(LimbPartition && lp) :
		device(lp.device),
		rank(lp.rank),
		meta(lp.meta),
		SPECIALmeta(lp.SPECIALmeta),
		DECOMPmeta(lp.DECOMPmeta),
		limb(std::move(lp.limb)),
		SPECIALlimb(std::move(lp.SPECIALlimb)),
		DECOMPlimb(std::move(lp.DECOMPlimb)),
		limbptr(std::move(lp.limbptr)),
		SPECIALlimbptr(std::move(lp.SPECIALlimbptr)),
		DECOMPlimbptr(std::move(lp.DECOMPlimbptr))
		{}
*/

	LimbPartition(LimbPartition&& l) noexcept;

	LimbPartition(ContextData& cc, const uint64_t& uid, int* level, const int id, bool def_stream = false);

	~LimbPartition();

	Global::Globals* getGlobals();
	void binomialDotProduct(LimbPartition& c1,
	  LimbPartition& c2,
	  const std::vector<const LimbPartition*>& c0s,
	  const std::vector<const LimbPartition*>& c1s,
	  const std::vector<const LimbPartition*>& d0s,
	  const std::vector<const LimbPartition*>& d1s,
	  bool ext);
	void binomialMult(LimbPartition& c1, LimbPartition& c2, const LimbPartition& d0, const LimbPartition& d1, bool extend_ins, bool square);
	void generateLimbToLevel(int new_level);

	/// @brief Release the NTT scratch of every Q limb of this partition, keeping the limb data.
	///
	/// For a partition that will never be transformed again — today, a device-encoded plaintext once
	/// its encode NTT has run (RNSPoly::loadCoefficients). Halves such a polynomial's device
	/// footprint: v + aux becomes v alone.
	///
	/// STREAM ORDERING IS THE CALLER'S: the releases are stream-ordered on `s`, so `s` must already
	/// be fenced against the limb streams. NTT(batch, /*sync=*/true) leaves exactly that fence
	/// behind on its way out (it closes with s.wait(STREAM(limb[i]))), which is the same guarantee
	/// loadCoefficients already relies on to free its staging buffer.
	///
	/// SPECIAL (P-basis) limbs are deliberately untouched: nothing that owns extended limbs is done
	/// transforming when this is called, and a device encode is Q-basis only.
	void freeNTTScratch();

	enum GENERATION_MODE { AUTOMATIC, SINGLE_BUFFER, DUAL_BUFFER };

	void generate(std::vector<LimbRecord>& records,
	  std::vector<LimbImpl>& limbs,
	  VectorGPU<void*>& ptrs,
	  int pos,
	  VectorGPU<void*>* auxptrs,
	  uint64_t* buffer	   = nullptr,
	  size_t offset		   = 0,
	  uint64_t* buffer_aux = nullptr,
	  size_t offset_aux	   = 0,
	  bool noptr		   = false);

	void generateLimb();

	void generateSpecialLimb(bool zero_out, bool for_communication);

	void add(const LimbPartition& p, const bool exta, bool extb);
	void add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b);

	void sub(const LimbPartition& p);

	void multElement(const LimbPartition& p);

	void multPt(const LimbPartition& p);

	void modup(LimbPartition& aux_partition);

	template <ALGO algo = ALGO_SHOUP> void moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs);

	void rescale();

	void freeSpecialLimbs();

	using OptReference		= LimbPartition*;
	using OptConstReference = const LimbPartition*;

	struct NTT_fusion_fields {
		OptReference op2;
		OptConstReference pt;
		OptReference res0;
		OptReference res1;
		OptConstReference kska;
		OptConstReference kskb;
	};

	template <ALGO algo, NTT_MODE mode>
	void ApplyNTT(int batch,
	  LimbPartition::NTT_fusion_fields fields,
	  std::vector<LimbImpl>& limb,
	  VectorGPU<void*>& limbptr,
	  VectorGPU<void*>& auxptr,
	  ContextData& cc,
	  const int primeid_init,
	  const int limbsize = -1);

	template <ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE> void NTT(int batch = 1, bool sync = false, NTT_fusion_fields fields = NTT_fusion_fields{});

	struct INTT_fusion_fields {
		OptReference res0;
		OptReference res1;
		OptConstReference kska;
		OptConstReference kskb;
		OptConstReference c0;
		OptConstReference c0tilde;
		OptConstReference c1;
		OptConstReference c1tilde;
	};

	template <ALGO algo, INTT_MODE mode>
	void ApplyINTT(int batch,
	  LimbPartition::INTT_fusion_fields fields,
	  std::vector<LimbImpl>& limb,
	  VectorGPU<void*>& limbptr,
	  VectorGPU<void*>& auxptr,
	  ContextData& cc,
	  const int primeid_init,
	  const int limbsize);

	template <ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE> void INTT(int batch = 1, bool sync = false, INTT_fusion_fields fields = INTT_fusion_fields{});

	static std::vector<VectorGPU<void*>> generateDecompLimbptr(void** buffer, const std::vector<std::vector<LimbRecord>>& DECOMPmeta, const int device, int offset);

	void generateAllDecompLimb(uint64_t* pInt, size_t offset);

	void generateAllDigitLimb(uint64_t* pInt, size_t offset);

	void copyLimb(const LimbPartition& partition);
	void copySpecialLimb(const LimbPartition& p);

	void generateAllDecompAndDigit(bool iskey);

	void mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2, const LimbPartition& partition3, const LimbPartition& partition4);

	void mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2);

	void generateLimbSingleMalloc();

	/// @brief One pooled buffer for `limbsize` value limbs (FIDESLIB_PT_ARENA): the limbs are views
/// into it (no Limb::aux; the encode runs its NTT through the chunk scratch), `bufferLIMB` owns it
/// and the destructor frees it whole. Fresh partition only (no limbs, no buffer).
void generateLimbArena(int limbsize);
	void generateLimbConstant();

	/// @brief Make this freshly constructed partition a NON-OWNING VIEW of `src`: its first
	/// getLimbSize(*level) Q limbs and all of its special limbs.
	///
	/// Every view limb is a `Limb` built with the buffer constructor (the one `generate` uses for a
	/// partition's single buffer) over the source limb's own `v.data`, so its `v` and `aux` are
	/// unmanaged VectorGPUs and its destructor frees nothing. The partition's own `bufferLIMB` and
	/// `bufferSPECIAL` stay null, so ~LimbPartition frees only what every partition allocates for
	/// itself in its constructor: the small device pointer table (`bufferAUXptrs`). That table is
	/// filled here -- `limbptr` and `SPECIALlimbptr`, which is how every kernel finds a limb -- and
	/// `auxptr` is left unfilled, as it is for a constant-grown plaintext (generateLimbConstant).
	///
	/// LIFETIME IS THE CALLER'S: the view reads `src`'s device memory for as long as it exists, so
	/// `src` must outlive it. Read-only use only: a view is a plaintext operand, never a destination.
	void aliasLimbsOf(const LimbPartition& src);

	void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli);

	void dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace = false, const LimbPartition* limbsrc = nullptr);

	void multElement(const LimbPartition& partition1, const LimbPartition& partition2);

	void multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0, const LimbPartition& c0tilde, const LimbPartition& ksk_a, const LimbPartition& ksk_b);

	size_t getLimbSize(int level);
	void automorph(const int index, const int br, LimbPartition* src, bool ext);

	void modupInto(LimbPartition& partition, LimbPartition& partition1);
	void multScalar(std::vector<uint64_t>& vector);
	void squareElement(const LimbPartition& p);
	void binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0, const LimbPartition& c2_key_switched_1);
	void addScalar(std::vector<uint64_t>& vector);
	void subScalar(std::vector<uint64_t>& vector);
	void dropLimb();
	void addMult(const LimbPartition& partition, const LimbPartition& partition1);
	void broadcastLimb0();
	void evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights);
	void rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b);
	void squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b);
	void rescaleMGPU();
	void moddownMGPU(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs, const std::vector<uint64_t*>& bufferSpecial_);
	void generatePartialSpecialLimb();
	void dotProductPt(LimbPartition& c1,
	  const std::vector<const LimbPartition*>& c0s,
	  const std::vector<const LimbPartition*>& c1s,
	  const std::vector<const LimbPartition*>& pts,
	  bool ext);

	void generateGatherLimb(bool iskey);
	void dotKSKfusedMGPU(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a, const LimbPartition& ksk_b, const LimbPartition& src);
	void fusedHoistRotate(int n,
	  std::vector<int> indexes,
	  std::vector<LimbPartition*>& c0,
	  std::vector<LimbPartition*>& c1,
	  const std::vector<LimbPartition*>& ksk_a,
	  const std::vector<LimbPartition*>& ksk_b,
	  const LimbPartition& src_c0,
	  const LimbPartition& src_c1,
	  bool c0_modup);

	void modup_ksk_moddown_mgpu(LimbPartition& c0,
	  const LimbPartition& ksk_a,
	  const LimbPartition& ksk_b,
	  LimbPartition& auxLimbs1,
	  LimbPartition& auxLimbs2,
	  const bool moddown,
	  const std::vector<uint64_t*>& bufferGather_,
	  const std::vector<uint64_t*>& bufferSpecial_c0,
	  const std::vector<uint64_t*>& bufferSpecial_c1,
	  const std::vector<Stream*>& external_s,
	  std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>>& signal,
	  std::vector<std::atomic_uint64_t*>& thread_stop,
	  const std::vector<Stream*>& external_s0);
	void broadcastLimb0_mgpu();
	void doubleRescaleMGPU(LimbPartition& partition);
	void scaleByP();

	void modupMGPU(LimbPartition& aux, const std::vector<uint64_t*>& bufferGather_, std::vector<std::atomic_uint64_t*>& thread_stop, std::vector<Stream*>& external_s);

	void multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1, const LimbPartition& in, const LimbPartition& aux);
	static void multScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
	  const std::vector<std::vector<unsigned long int>>& vector,
	  const std::vector<std::vector<unsigned long int>>& vector_shoup,
	  int stride,
	  double usage);

	static void addScalarBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage);

	static void multPtBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb, int stride, double usage);

	static void addBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb, int stride, double usage, bool sub, bool exta, bool extb);

	/// @param prepared_pt_q  A device diagonal table for this (partition, giant-step chunk)
	///   that was uploaded once and outlives the call — see CKKS/PreparedLT.cuh. When given (with
	///   @p prepared_pt_p alongside it for an `ext` call), the diagonal third of the pointer table
	///   is neither rebuilt on the host nor re-uploaded; the per-call output and baby-rotation
	///   thirds are built and uploaded exactly as before. nullptr (the default) is today's
	///   behaviour, byte for byte.
	static void LTdotProductPtBatch(std::vector<LimbPartition*>& out,
	  const std::vector<LimbPartition*>& in,
	  const std::vector<LimbPartition*>& pt,
	  int bStep,
	  int gStep,
	  int stride,
	  double usage,
	  bool ext,
	  void*** prepared_pt_q = nullptr,
	  void*** prepared_pt_p = nullptr);

	static void fusedHoistedRotateBatch(std::vector<LimbPartition*>& out,
	  const std::vector<LimbPartition*>& in,
	  const std::vector<LimbPartition*>& ksk_a,
	  const std::vector<LimbPartition*>& ksk_b,
	  const std::vector<int>& indexes,
	  int n,
	  int stride,
	  double usage,
	  bool c0_modup);

	Stream& getS() const {
		return const_cast<LimbPartition*>(this)->s;
	}
};

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_LIMBPARTITION_CUH
