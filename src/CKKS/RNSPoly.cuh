//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_RNSPOLY_CUH
#define FIDESLIB_CKKS_RNSPOLY_CUH

#include "CKKS/LimbPartition.cuh"
#include "CudaUtils.cuh"
#include <vector>

namespace FIDESlib::CKKS {
class KeySwitchingKey;

class RNSPoly {
	const uint64_t uid;
	ContextData& cc;
	int level;
	bool modUp = false;

  public:
	std::vector<LimbPartition> GPU;

	/// The scratch slot whose auxiliary-polynomial free list this polynomial was CREATED for, or -1
	/// for a polynomial that never came through ContextData::getAuxilarPoly. returnAuxilarPoly
	/// hands a polynomial back to THIS slot's list, not the returning thread's: under
	/// FIDESLIB_CONCURRENT_OPS a Ciphertext is routinely built on one thread and destroyed on
	/// another, and routing by the destroying thread grew that thread's list without bound while
	/// the makers' lists stayed empty and allocated fresh (the aux-poly drift: 902 created, 694
	/// parked across five lists, ~10 GB idle at an observed OOM). Carried by the move
	/// constructor; the default mode has one slot, so it is always 0 there and nothing changes.
	int aux_slot = -1;

	explicit RNSPoly(ContextData& context, int level = -1, bool single_malloc = false, bool def_stream = false);
	explicit RNSPoly(ContextData& context, const std::vector<std::vector<uint64_t>>& data);
	RNSPoly(RNSPoly&& src) noexcept;

	void grow(int level, bool single_malloc = false, bool constant = false);
	/// @brief grow for a batched device encode under FIDESLIB_PT_ARENA: every partition backs its
	/// value limbs with one pooled buffer (LimbPartition::generateLimbArena); no NTT scratch is
	/// allocated (the encode uses the chunk scratch). Fresh polynomial only (level -1).
	void growArena(int level);

	void load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli);

	void store(std::vector<std::vector<uint64_t>>& data);

	bool isModUp() const;
	void SetModUp(bool newValue);

	void scaleByP();

	int32_t getLevel() const;

	void add(const RNSPoly& p);
	void add(const RNSPoly& a, const RNSPoly& b);

	void sub(const RNSPoly& p);

	void multPt(const RNSPoly& p, bool rescale);

	void modup();

	template <ALGO algo = ALGO_SHOUP> void moddown(bool ntt = true, bool free = true, int aux_num = 0);
	int automorph_index_precomp(int idx) const;

	void rescale();

	void sync();

	void freeSpecialLimbs();

	template <ALGO algo = ALGO_SHOUP> void NTT(int batch, bool sync);

	template <ALGO algo = ALGO_SHOUP> void INTT(int batch, bool sync);

	// std::array<RNSPoly, 2> dotKSK(const KeySwitchingKey& ksk);

	void generateSpecialLimbs(bool zero_out, bool for_communication);

	void multElement(const RNSPoly& poly);

	void generateDecompAndDigit(bool iskey);

	void mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3, const RNSPoly& poly4);

	void mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2);

	void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli);

	void dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, const RNSPoly* limbsrc = nullptr);

	void multElement(const RNSPoly& poly1, const RNSPoly& poly2);

	void multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde, const KeySwitchingKey& key);

	void automorph(const int idx, const int br = 1, RNSPoly* src = nullptr);

	RNSPoly& dotKSKInPlace(const KeySwitchingKey& ksk, RNSPoly* limb_src);

	void hoistedRotationFused(std::vector<int> indexes,
	  std::vector<RNSPoly*>& c0,
	  std::vector<RNSPoly*>& c1,
	  const std::vector<RNSPoly*>& ksk_a,
	  const std::vector<RNSPoly*>& ksk_b,
	  const RNSPoly& src_c0,
	  const RNSPoly& src_c1);

	/** Change the polynomial level only superficially, be very careful as this should only be used for lower
	 * level optimizations.
	 */
	void setLevel(const int level);
	void modupInto(RNSPoly& poly);
	RNSPoly& dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, const RNSPoly* limbsrc = nullptr);
	void multScalar(std::vector<uint64_t>& vector1);
	void squareElement(const RNSPoly& poly);
	void binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1);
	void addScalar(std::vector<uint64_t>& vector1);
	void subScalar(std::vector<uint64_t>& vector1);
	void copy(const RNSPoly& poly);
	/** @brief Resize *this* to the limb layout of `poly` WITHOUT copying any device bytes.
	 *
	 * This is exactly the resizing half of copy(); it exists for callers whose very next operation
	 * writes every limb of *this* (an out-of-place kernel such as Mult_), for which the copy would
	 * be pure waste. The limb contents are left uninitialized. Not valid for mod-up polynomials.
	 */
	void shapeLike(const RNSPoly& poly);
	void dropToLevel(int level);
	void addMult(const RNSPoly& poly, const RNSPoly& poly1);
	void broadcastLimb0();
	void evalLinearWSum(uint32_t i, std::vector<const RNSPoly*>& vector1, std::vector<uint64_t>& vector2);
	void loadConstant(const std::vector<std::vector<uint64_t>>& vector1, const std::vector<uint64_t>& vector2);
	/**
	 * @brief Device encode: build every Q limb on the GPU from ONE uploaded coefficient vector.
	 *
	 * The counterpart of loadConstant, which takes an already-encoded EVALUATION-form tower per
	 * limb and uploads each separately — L transfers of N words for L*N words of traffic. Here the
	 * caller uploads N words ONCE, in OpenFHE's biased-integer coefficient form, and the device
	 * does the per-limb CRT reduction (Limb::loadCoefficients) followed by the forward NTT. Traffic
	 * drops by exactly the limb count.
	 *
	 * @param biased  Exactly cc.N biased coefficients: OpenFHE's `temp` array from
	 *                CKKSPackedEncoding::Encode, already scattered to full length (so the `gap`
	 *                stride of a sparse encoding is baked in, zeros included) rather than the
	 *                length-2*slots compact form.
	 * @param moduli  The Q primes, in tower order, that the coefficients are to be reduced against.
	 *                Q basis ONLY: unlike loadConstant this does NOT accept a P tail, because a
	 *                device-encoded extended plaintext would need the mod-up basis decided here
	 *                rather than by the encoder. Mismatches throw rather than being skipped —
	 *                loadConstant's silent drop of an unrecognized tail modulus is a known hazard.
	 * @param bigBound  OpenFHE's Max64BitValue() (2^63 - 2^9 - 1) — the bias constant the caller
	 *                  used. Passed rather than hardcoded so a mismatch with the host encoder is a
	 *                  parameter error, not a silent wrong answer.
	 *
	 * Leaves the polynomial at level moduli.size()-1 in EVALUATION format, exactly as a
	 * loadConstant of the same values would.
	 *
	 * @note Requires a freshly constructed polynomial (level -1), and grows it NON-constant: unlike
	 *       a loadConstant plaintext, this one has to be transformed, so it needs the NTT scratch
	 *       that constant-grown polynomials deliberately omit. That scratch is RELEASED as soon as
	 *       the encode's NTT is fenced (freeNTTScratch), so the finished plaintext occupies the
	 *       same device memory as a loadConstant one — v, no aux — and callers capping device
	 *       residency can size against a single figure whichever encoder produced the plaintext.
	 */
	void loadCoefficients(const std::vector<uint64_t>& biased, const std::vector<uint64_t>& moduli, uint64_t bigBound);

	/**
	 * @brief loadCoefficients for a BATCH of freshly constructed polynomials that share one
	 *        modulus chain, in one pass of kernels instead of one pass per polynomial.
	 *
	 * Identical result, polynomial for polynomial, to calling loadCoefficients on each in turn:
	 * same host-computed coefficients, same per-tower reduction, same forward NTT, same released
	 * scratch. What changes is the number of times the host talks to the driver. Per polynomial the
	 * single-call path issues its own coefficient upload, its own reduce launch per limb and its
	 * own NTT launch pair; here the whole batch shares ONE upload, ONE reduce launch per partition
	 * and ONE NTT launch pair per partition. With 24 worker threads encoding ~300k diagonals a pass
	 * that call count, not the arithmetic, is what the device-encode arm spends its time on.
	 *
	 * OWNERSHIP IS NOT BATCHED, deliberately: every polynomial still grows its own limbs through
	 * the ordinary pooled allocator, so each one is independently destructible and no polynomial's
	 * device memory is held alive by a sibling. A shared slab would cut the pool traffic too, but
	 * it would tie the lifetimes of the whole batch together and make an individual release
	 * impossible without refcounting; the pool's own contention is a separate problem with a
	 * separate fix.
	 *
	 * @param polys   Freshly constructed polynomials (level -1), all on the same context. Not empty.
	 * @param biased  One length-N biased coefficient vector per polynomial, same order, non-null.
	 * @param moduli  The Q primes of the shared target level, in tower order. Q basis only.
	 * @param bigBound  OpenFHE's Max64BitValue(), the bias constant every vector was built with.
	 *
	 * Every precondition loadCoefficients enforces is enforced here, per polynomial, and loudly.
	 */
	static void loadCoefficientsBatch(const std::vector<RNSPoly*>& polys,
	  const std::vector<const std::vector<uint64_t>*>& biased,
	  const std::vector<uint64_t>& moduli,
	  uint64_t bigBound);

	/**
	 * @brief loadCoefficientsBatch with the TRANSFORM on the device (FIDESLIB_DEVICE_IFFT): each
	 * polynomial is given its `slots` real slot values; the device runs the CKKS inverse special FFT
	 * (DiscreteFourierTransform::FFTSpecialInv's ladder, 1/slots, the scaling factor, llround,
	 * the Max64BitValue bias, the gap scatter), then the same per-limb reduction and NTT.
	 * Every precondition loadCoefficientsBatch enforces is enforced here too. With
	 * DeviceIfftCheck() a coefficient above 2^MAX_BITS_IN_WORD throws (one stream sync per batch).
	 */
	static void loadSlotsBatch(const std::vector<RNSPoly*>& polys,
	  const std::vector<const std::vector<double>*>& values,
	  int slots,
	  double scalingFactor,
	  const std::vector<uint64_t>& moduli,
	  uint64_t bigBound);

	/// @brief Release the NTT scratch of every Q limb, keeping the limb data.
	///
	/// For a polynomial that will never be transformed again — today, a device-encoded plaintext
	/// once loadCoefficients has run its NTT. Halves its device footprint.
	///
	/// STREAM ORDERING IS THE CALLER'S: see LimbPartition::freeNTTScratch. Idempotent. A polynomial
	/// whose scratch has been released must not be NTT'd/INTT'd again, nor used as the in-place
	/// destination of a mult/addMult; it may still be read as an operand, which is all a plaintext
	/// is ever used for.
	void freeNTTScratch();
	void rotateModupDotKSK(RNSPoly& poly, RNSPoly& poly1, const KeySwitchingKey& key);
	void squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key);
	void generatePartialSpecialLimbs();
	void dotKSKfused(RNSPoly& out2, const RNSPoly& digitSrc, const RNSPoly& ksk_a, const RNSPoly& ksk_b, const RNSPoly* source);
	void dotProductPt(RNSPoly& c1, const std::vector<const RNSPoly*>& c0s, const std::vector<const RNSPoly*>& c1s, const std::vector<const RNSPoly*>& pts, bool ext);
	RNSPoly& dotProduct(RNSPoly& c1,
	  const RNSPoly& kskb,
	  const RNSPoly& kska,
	  const std::vector<const RNSPoly*>& c0in,
	  const std::vector<const RNSPoly*>& c1in,
	  const std::vector<const RNSPoly*>& d0in,
	  const std::vector<const RNSPoly*>& d1in,
	  bool ext_in,
	  bool ext_out);
	void gatherAllLimbs();
	void generateGatherLimbs();
	void copyShallow(const RNSPoly& poly);
	/** @brief Make this freshly constructed polynomial (level -1) a NON-OWNING VIEW of `src`
	 * truncated to `level`: Q limbs 0..level and, for an extended `src`, all of its special limbs.
	 *
	 * Limb i of an RNS polynomial is the same integer polynomial reduced mod prime i, and the
	 * chain's primes are one prefix, so the first level + 1 limbs of `src` ARE that integer
	 * polynomial at `level`. Nothing is copied and no limb memory is allocated -- only each
	 * partition's pointer table (LimbPartition::aliasLimbsOf). `src` must outlive the view, and
	 * the view must only ever be read.
	 */
	void aliasPrefixOf(const RNSPoly& src, int level);
	RNSPoly& modup_ksk_moddown_mgpu(const KeySwitchingKey& key, bool moddown);
	void rescaleDouble(RNSPoly& poly);

	void multNoModdownEnd(RNSPoly& c0, const RNSPoly& bc0, const RNSPoly& bc1, const RNSPoly& in, const RNSPoly& aux);

	void binomialMult(RNSPoly& c1, RNSPoly& in, const RNSPoly& d0, const RNSPoly& d1, bool moddown, bool square);

	static void multScalarBatchManyToOne(std::vector<RNSPoly*>& polya,
	  const std::vector<std::vector<unsigned long int>>& vector,
	  const std::vector<std::vector<unsigned long int>>& vectors,
	  int stride,
	  double usage);
	static void addScalarBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage);
	static void multPtBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage);
	static void addBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage, bool sub, bool exta, bool extb);

	/// @param prepared  A resident device diagonal table for this group (CKKS/PreparedLT.cuh),
	///   or nullptr (the default) for today's behaviour, byte for byte. When given, its shape must
	///   match this call's (@p bStep, @p gStep, @p stride, @p ext and the transform count); a
	///   mismatch throws rather than indexing a table built for something else.
	static void LTdotProductPtBatch(std::vector<RNSPoly*>& out,
	  const std::vector<RNSPoly*>& in,
	  const std::vector<RNSPoly*>& pt,
	  int bStep,
	  int gStep,
	  int stride,
	  double usage,
	  bool ext,
	  const PreparedLTTable* prepared = nullptr);
	static void fusedHoistedRotateBatch(std::vector<RNSPoly*>& out,
	  const std::vector<RNSPoly*>& in,
	  const std::vector<RNSPoly*>& ksk_a,
	  const std::vector<RNSPoly*>& ksk_b,
	  const std::vector<int>& indexes,
	  int stride,
	  double usage,
	  bool c0_modup);
};
} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_RNSPOLY_CUH
