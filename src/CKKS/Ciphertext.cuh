//
// Created by carlosad on 24/04/24.
//

#ifndef FIDESLIB_CKKS_CIPHERTEXT_CUH
#define FIDESLIB_CKKS_CIPHERTEXT_CUH

#include "RNSPoly.cuh"
#include "forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"
#include <optional>
#include <source_location>

namespace FIDESlib::CKKS {

/**
 * @brief Global flag controlling whether hoisted rotation optimisation is enabled.
 *
 * This can be toggled at runtime to switch between the standard hoisted-rotation
 * implementation and the fused hoisted‑rotation variant.
 */
extern bool hoistRotateFused;

/**
 * @class Ciphertext
 * @brief Represents a ciphertext in the CKKS scheme.
 *
 * The ciphertext holds two RNSPoly components (c0 and c1) together with
 * associated metadata (key identifier, noise level, scaling factor, etc.).
 * It provides arithmetic operations, key‑switching, rotations and
 * rescaling.  Most operations maintain the level information and may
 * automatically perform rescaling or modulus‑down conversion as required
 * by the CKKS protocol.
 */
class Ciphertext {
	/** @brief Identifier string for NVTX profiling. */
	static constexpr const char* loc{ "Ciphertext" };
	/** @brief NVTX range object for profiling this class' methods. */
	CudaNvtxRange my_range;

  public:
	/** @brief Identifier for the key associated with this ciphertext. */
	KeyHash keyID{ "" };
	/** @brief Reference to the CKKS context owning this ciphertext. */
	Context& cc_;
	/** @brief Reference to the context data (metadata) for this ciphertext. */
	ContextData& cc;
	/** @brief The two polynomial components of the ciphertext (c0 and c1). */
	RNSPoly c0, c1;
	/**
	 * @brief Optional third polynomial component, present only while the ciphertext is degree 2
	 *        (i.e. between multNoRelin() and relinearize()).
	 *
	 * It is acquired from the context's auxiliary-polynomial POOL (Context::getAuxilarPoly) and
	 * returned to it on release, exactly like c0 and c1 — it is never the shared keyswitch scratch
	 * (Context::getKeySwitchAux), which every key-switch overwrites.
	 */
	std::optional<RNSPoly> c2;
	/** @brief Accumulated noise factor for this ciphertext. */
	double NoiseFactor = 0;
	/** @brief Discrete noise level indicator. */
	int NoiseLevel = { 1 };
	/** @brief Number of slots (vector length) encoded in the ciphertext. */
	int slots = 0;

	/**
	 * @brief Move constructor. Transfers ownership of the internal resources
	 *        from another Ciphertext instance.
	 *
	 * @param ct_moved Rvalue reference to the source Ciphertext.
	 */
	Ciphertext(Ciphertext&& ct_moved) noexcept;

	/**
	 * @brief Constructs an empty Ciphertext bound to a given context.
	 *
	 * The created ciphertext has uninitialised polynomial components and zero noise.
	 *
	 * @param cc Reference to the CKKS Context the ciphertext belongs to.
	 */
	explicit Ciphertext(Context& cc);

	/**
	 * @brief Constructs a Ciphertext from a serialized RawCipherText.
	 *
	 * This loads polynomial data and metadata from the raw representation.
	 *
	 * @param cc   Reference to the CKKS Context.
	 * @param rawct Serialized representation of a ciphertext.
	 */
	Ciphertext(Context& cc, const RawCipherText& rawct);

	/** @brief Destructor. Releases any allocated resources. */
	~Ciphertext();

	/**
	 * @brief OpenFHE NumberCiphertextElements: 3 while a third component is present, 2 otherwise.
	 */
	[[nodiscard]] int numElements() const {
		return c2.has_value() ? 3 : 2;
	}

	/**
	 * @brief Copies only the metadata fields from another ciphertext.
	 *
	 * This does **not** copy polynomial data.
	 *
	 * @param a Source ciphertext to copy metadata from.
	 */
	void copyMetadata(const Ciphertext& a);

	/**
	 * @brief Merges metadata from two ciphertexts into *this*.
	 *
	 * Used after operations that combine two ciphertexts.
	 *
	 * @param a First operand.
	 * @param b Second operand.
	 */
	void addMetadata(const Ciphertext& a, const Ciphertext& b);

	/**
	 * @brief Merges metadata from a ciphertext and a plaintext into *this*.
	 *
	 * @param a Ciphertext operand.
	 * @param b Plaintext operand.
	 */
	void addMetadata(const Ciphertext& a, const Plaintext& b);

	/**
	 * @brief Merges metadata appropriate for multiplication of two ciphertexts.
	 *
	 * @param a First operand.
	 * @param b Second operand.
	 */
	void multMetadata(const Ciphertext& a, const Ciphertext& b);

	/**
	 * @brief Merges metadata for multiplication of a ciphertext with a plaintext.
	 *
	 * @param a Ciphertext operand.
	 * @param b Plaintext operand.
	 */
	void multMetadata(const Ciphertext& a, const Plaintext& b);

	/**
	 * @brief Normalises a rotation index to the valid range for the current slot count.
	 *
	 * The function makes the index positive, maps it into `[0, slots)` and, if the
	 * index exceeds half the slot count, adds `N/2 - slots` to map it into the
	 * underlying NTT domain.
	 *
	 * @param index Desired rotation index (may be negative).
	 * @return Normalized index within `[0, slots)`.
	 */
	int normalyzeIndex(int index) const;

	/**
	 * @brief Loads ciphertext data from a RawCipherText representation.
	 *
	 * @param rawct Serialized ciphertext to load.
	 */
	void load(const RawCipherText& rawct);

	/**
	 * @brief Stores the current ciphertext into a RawCipherText representation.
	 *
	 * @param rawct Destination object that will receive the serialized data.
	 */
	void store(RawCipherText& rawct);

	/**
	 * @brief Adds another ciphertext to *this* (in‑place).
	 *
	 * The method ensures both ciphertexts are on the same level, performing
	 * rescaling or level adjustments when required by the current
	 * `rescaleTechnique`.  It may invoke `adjustForAddOrSub` and, if that
	 * fails, create a temporary copy of `b` with aligned scaling before
	 * retrying.  After the polynomial addition, metadata is merged via
	 * `addMetadata`.  The operation count for `ADD` is incremented.
	 *
	 * @param b Ciphertext to add.
	 */
	void add(const Ciphertext& b);

	/**
	 * @brief Subtracts another ciphertext from *this* (in‑place).
	 *
	 * Mirrors the behaviour of `add` but uses subtraction on the polynomial
	 * components.  Rescaling, level alignment and metadata handling follow the
	 * same logic as in `add`.
	 *
	 * @param b Ciphertext to subtract.
	 */
	void sub(const Ciphertext& b);

	/**
	 * @brief Adds a plaintext to *this* ciphertext (in‑place).
	 *
	 * Handles automatic rescaling when the plaintext and ciphertext have
	 * mismatched levels or noise.  For flexible rescaling techniques it may
	 * create a temporary adjusted plaintext before performing the addition.
	 * After the operation, metadata is merged via `addMetadata`.
	 *
	 * @param b Plaintext operand.
	 */
	void addPt(const Plaintext& b);

	/**
	 * @brief Subtracts a plaintext from *this* ciphertext (in‑place).
	 *
	 * The semantics are analogous to `addPt` but perform subtraction on `c0`.
	 *
	 * @param pt Plaintext operand.
	 */
	void subPt(const Plaintext& pt);

	/**
	 * @brief Adds a scalar (double) to both polynomial components.
	 *
	 * The scalar is first converted to the appropriate modulus representation
	 * via `ElemForEvalAddOrSub`.  The sign of the scalar is handled by
	 * converting the element to its complement when `c < 0`.  No metadata
	 * updates are performed.
	 *
	 * @param c Scalar value to add.
	 */
	void addScalar(const double c);

	/**
	 * @brief Multiplies *this* ciphertext by a plaintext.
	 *
	 * If the current rescaling technique requires matching noise levels,
	 * the method may rescale the ciphertext or adjust the plaintext
	 * (creating a temporary copy) before multiplication.  After the
	 * component‑wise multiplication, metadata is merged via `multMetadata`.
	 * If `rescale` is true and the technique is `FIXEDMANUAL`, the
	 * ciphertext is rescaled and the noise factor/level are updated.
	 *
	 * @param b Plaintext operand.
	 * @param rescale If true, performs rescaling after multiplication.
	 * @param ignore_scales
	 */
	void multPt(const Plaintext& b, bool rescale = false, bool ignore_scales = false);

	/**
	 * @brief Multiplies a ciphertext `c` by a plaintext and stores the result in *this*.
	 *
	 * Equivalent to copying `c` into *this* and then invoking `multPt(b, rescale)`.
	 *
	 * @param c Ciphertext operand.
	 * @param b Plaintext operand.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void multPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

	/**
	 * @brief Computes `*this + (c * b)` in a single fused operation.
	 *
	 * The method adds ciphertext `c * b` to *this* and optionally rescales.
	 * Metadata is merged via `multMetadata`.
	 *
	 * @param c Ciphertext to add.
	 * @param b Plaintext to multiply.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

	/**
	 * @brief Multiplies *this* by another ciphertext.
	 *
	 * Handles level alignment and possible rescaling according to the current
	 * `rescaleTechnique`.  If necessary, the method calls `adjustForMult`
	 * (creating a temporary copy of `b` when alignment fails).  The actual
	 * multiplication is performed before a key‑switch operation.  Metadata is
	 * merged via `multMetadata`.  The operation count for `MULT` is incremented.
	 *
	 * @param b Ciphertext multiplier.
	 * @param rescale Perform rescaling after multiplication if true.
	 * @param moddown Perform modulus down‑conversion if true.
	 */
	void mult(const Ciphertext& b, bool rescale = false, bool moddown = true);

	/**
	 * @brief Tensor product with NO key-switch, leaving a degree-2 ciphertext.
	 *
	 * Shares mult()'s level/scale prologue, then computes the three-term product
	 * (c0·b.c0, c0·b.c1 + c1·b.c0, c1·b.c1) with the third term kept in `c2`. Metadata is merged
	 * exactly as mult() does. The caller must relinearize() before any operation that needs a
	 * degree-1 ciphertext.
	 *
	 * @param b Ciphertext multiplier.
	 */
	void multNoRelin(const Ciphertext& b);

	/**
	 * @brief Key-switches `c2` with the evaluation key and folds it back into (c0, c1), returning the
	 *        ciphertext to degree 1.
	 *
	 * A no-op when the ciphertext is already degree 1, mirroring OpenFHE's RelinearizeInPlace. All
	 * metadata (level, noise level, scaling factor, slots) is left untouched.
	 */
	void relinearize();

	/**
	 * @brief Multiplies two ciphertexts and stores the result in *this*.
	 *
	 * This overload forwards to the two‑argument version, using `b` and `c`
	 * as the two multiplicands.  It exists for convenience when the two
	 * operands are already separate objects.
	 *
	 * @param b First operand.
	 * @param c Second operand.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void mult(const Ciphertext& b, const Ciphertext& c, bool rescale = false);

	/**
	 * @brief Multiplies both polynomial components by a scalar without pre‑checks.
	 *
	 * The scalar is converted to the appropriate modulus element via
	 * `ElemForEvalMult`.  If `rescale` is true and the technique is
	 * `FIXEDMANUAL`, the ciphertext is rescaled after multiplication.
	 *
	 * @param c Scalar multiplier.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void multScalarNoPrecheck(const double c, bool rescale = false);

	/**
	 * @brief Multiplies both polynomial components by a scalar, performing necessary checks.
	 *
	 * For flexible rescaling techniques the method may rescale the ciphertext
	 * when the noise level is 2.  The scalar is converted using
	 * `ElemForEvalMult` and the multiplication is applied to both components.
	 * Metadata (noise level/factor) is updated accordingly; if `rescale`
	 * is true and the technique is `FIXEDMANUAL`, a rescaling step follows.
	 *
	 * @param c Scalar multiplier.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void multScalar(const double c, bool rescale = false);

	/**
	 * @brief Multiplies a ciphertext `b` by a scalar and stores the result in *this*.
	 *
	 * The method copies `b` into *this* and then calls `multScalar(c,
	 * rescale)`.  It is useful when the caller wishes to keep `b` unchanged.
	 *
	 * @param b Ciphertext operand.
	 * @param c Scalar multiplier.
	 * @param rescale Perform rescaling after multiplication if true.
	 */
	void multScalar(const Ciphertext& b, const double c, bool rescale = false);

	/**
	 * @brief Squares the ciphertext (i.e., multiplies it by itself).
	 *
	 * After the multiplication the metadata (noise level/factor) is updated.
	 * If `rescale` is true and the technique is `FIXEDMANUAL`, a rescaling
	 * step follows.
	 *
	 * @param rescale Perform rescaling after squaring if true.
	 */
	void square(bool rescale = false);

	/**
	 * @brief Squares the source ciphertext `src` and stores the result in *this*.
	 *
	 * Equivalent to copying `src` into *this* and then calling `square(rescale)`.
	 *
	 * @param src Source ciphertext to square.
	 * @param rescale Perform rescaling after squaring if true.
	 */
	void square(const Ciphertext& src, bool rescale = false);

	/**
	 * @brief Rotates the slots of the ciphertext by `index` positions.
	 *
	 * The index is normalised using `normalyzeIndex`.  Depending on the
	 * `moddown` flag, a modulus‑down conversion may be performed after the
	 * rotation.
	 *
	 * @param index Number of slots to rotate (positive = left).
	 * @param moddown Perform modulus down‑conversion after rotation if true.
	 */
	void rotate(const int index, bool moddown = true);

	/**
	 * @brief Rotates ciphertext `c` by `index` slots and stores the result in *this*.
	 *
	 * This overload copies `c` into *this* and then calls the single‑argument
	 * `rotate(index, true)`.
	 *
	 * @param c Source ciphertext.
	 * @param index Number of slots to rotate.
	 */
	void rotate(const Ciphertext& c, const int index);

	/**
	 * @brief Computes the complex conjugate of ciphertext `c` and stores it in *this*.
	 *
	 * The operation is performed component‑wise and updates the operation
	 * count for `CONJUGATE`.
	 *
	 * @param c Source ciphertext.
	 */
	void conjugate(const Ciphertext& c);

	/**
	 * @brief Performs modulus down‑conversion on the ciphertext.
	 *
	 * Both polynomial components are down‑converted; when `free` is true the
	 * temporary special limbs are released.
	 *
	 * @param free If true, frees intermediate buffers after conversion.
	 */
	void modDown(bool free = false);

	/** @brief Increases the modulus level of the ciphertext (level up). */
	void modUp();

	/** @brief Rescales the ciphertext to the next level, adjusting scaling factor. */
	void rescale();

	/**
	 * @brief Grows the ciphertext to at least `level` by performing necessary modulus upgrades.
	 *
	 * The method ensures both components reach the requested level, generating
	 * special limbs if required.
	 *
	 * @param level Desired minimum level.
	 */
	void growToLevel(int level);

	/** @brief Drops the ciphertext to a lower level, discarding higher moduli.

		@todo adjust the scale on FLEXIBLE modes
	*/
	void dropToLevel(int level, bool skip_adjust = false);

	/**
	 * @brief Returns the current modulus level of the ciphertext.
	 *
	 * @return Current level index (both components share the same level).
	 */
	[[nodiscard]] int32_t getLevel() const;

	/**
	 * @brief Performs an automorphism on the ciphertext.
	 *
	 * The automorphism is applied to both polynomial components; the `br`
	 * argument is passed to the underlying `RNSPoly::automorph` method and
	 * is implementation‑specific.
	 *
	 * @param index Automorphism index.
	 * @param br    Additional parameter used by the underlying implementation.
	 */
	void automorph(const int index, const int br);

	/**
	 * @brief Performs a batch automorphism on multiple ciphertexts.
	 *
	 * Equivalent to calling `automorph` on each component; the `br` argument
	 * is forwarded unchanged.
	 *
	 * @param index Automorphism index.
	 * @param br    Additional parameter used by the underlying implementation.
	 */
	void automorph_multi(const int index, const int br);

	/**
	 * @brief Extends the ciphertext to a higher level if needed.
	 *
	 * If `init` is true, the method also (re)initialises internal structures
	 * and, when necessary, scales the components by the prime modulus.
	 *
	 * @param init If true, also (re)initialises internal structures.
	 */
	void extend(bool init = true);

	/**
	 * @brief Performs hoisted rotations for a set of indexes.
	 *
	 * The method normalises each index, allocates temporary results if they do
	 * not already exist, and optionally extends the ciphertexts before rotation.
	 *
	 * @param indexes Vector of rotation indexes.
	 * @param results Vector to receive resulting ciphertext pointers.
	 * @param ext     If true, the results are left in the extended Q*P basis
	 *                (isModUp() == true, values scaled by P) so that several of
	 *                them can be added and brought back with a single moddown().
	 *                Such results are not valid inputs for any other operation
	 *                (rescale, multPt, store, ...) until they, or the sum they
	 *                were added into, have been moddown'ed; see Accumulate() for
	 *                the intended pattern. If false, each result is moddown'ed
	 *                and ready to use.
	 */
	void rotate_hoisted(const std::vector<int>& indexes, std::vector<Ciphertext*> results, bool ext);

	/**
	 * @brief ONE index of rotate_hoisted, against a ModUp the CALLER owns and keeps alive.
	 *
	 * rotate_hoisted amortises the ModUp of c1 -- the digit decomposition and its NTTs, the
	 * expensive shared half of a key switch -- across a set of indexes known up front. The
	 * EvalFastRotationPrecompute / EvalFastRotation contract is the lazy shape of the same idea:
	 * decompose once, then rotate one index at a time. This is that per-index half; the ModUp
	 * lives in `src_c1_modup`, built once by the caller (api/engine/cuda/CudaEngine.cpp
	 * evalFastRotationPrecompute) from `src`'s c1.
	 *
	 * @param src           Source ciphertext. *this* must already be a DISTINCT copy of it at the
	 *                      same level (the precondition rotate_hoisted's `results` satisfy).
	 * @param src_c1_modup  ModUp of `src.c1`. Read-only in the fused and single-GPU paths, so one
	 *                      handle serves any number of indexes.
	 * @param index         Rotation index; 0 is the identity.
	 * @param ext           Leave the result in the extended (pre-ModDown) basis.
	 * @throws std::invalid_argument if the precompute's level no longer matches `src` (a stale
	 *         handle: the source was rescaled or levelled after the precompute was taken).
	 */
	void rotate_precomputed(const Ciphertext& src, RNSPoly& src_c1_modup, int index, bool ext);

	/**
	 * @brief The TRANSPOSE of rotate_hoisted: ONE index, MANY independent sources.
	 *
	 * rotate_hoisted shares one source's digit decomposition across many indexes. This shares one
	 * INDEX -- that is, one rotation key's digit-NTTs and one automorphism -- across many sources.
	 * Each source still needs its own ModUp (the decomposition is a property of the ciphertext, not
	 * of the key), but those ModUps and the key-switch inner products that follow them go out as
	 * ONE batched launch: RNSPoly::fusedHoistedRotateBatch with stride = B, n = 1, which loads the
	 * key into shared memory once per block and reuses it across the B sources of the block's z
	 * dimension.
	 *
	 * Every result is byte-identical to `Ciphertext::rotate(src, index)` on the same source: the
	 * same key, the same automorphism index, the same ModDown, only the launch shape differs.
	 *
	 * PRECONDITIONS (the batched kernel takes ONE limb layout for the whole batch):
	 *   - `srcs` and `results` are the same size, and every `results[i]` is a DISTINCT ciphertext
	 *     from every source (no aliasing: the kernel reads all sources and writes all results).
	 *   - every source is degree 1, at the SAME level, with the same keyID, slot count and ModUp
	 *     state. The caller (CudaEngine::evalRotateMany) checks this and falls back to a loop of
	 *     rotate() when it does not hold, so the api contract stays unconditional.
	 *   - `index` is non-zero and has a rotation key.
	 *
	 * @param srcs    Sources, all at one level. Not modified.
	 * @param results Destinations, one per source, distinct from them.
	 * @param index   The single rotation index, pre-normalisation.
	 */
	static void rotate_many(const std::vector<Ciphertext*>& srcs, const std::vector<Ciphertext*>& results, int index);

	/**
	 * @brief Evaluates a linear weighted sum (mutable version) over `n` ciphertexts.
	 *
	 * The method grows the result to the appropriate level, multiplies each
	 * ciphertext by the corresponding weight (converted via `ElemForEvalMult`),
	 * and accumulates the sum.  Metadata (noise level/factor) is updated to
	 * reflect the combined operation.
	 *
	 * @param n      Number of ciphertexts.
	 * @param ctxs   Vector of ciphertext pointers.
	 * @param weights Vector of weights applied to each ciphertext.
	 */
	void evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights);

	/**
	 * @brief Adds a scaled version of `ciphertext` (multiplied by `d`) to *this*.
	 *
	 * The scalar multiplication is performed on a temporary copy of the
	 * operand, which is then added component‑wise.  Metadata is merged via
	 * `addMetadata`.
	 *
	 * @param ciphertext Ciphertext to be scaled and added.
	 * @param d          Scalar multiplier.
	 */
	void addMultScalar(const Ciphertext& ciphertext, double d);

	/**
	 * @brief Adds scalar `c` to ciphertext `b` and stores result in *this*.
	 *
	 * Equivalent to copying `b` into *this* and invoking `addScalar(c)`.
	 *
	 * @param b Ciphertext operand.
	 * @param c Scalar to add.
	 */
	void addScalar(const Ciphertext& b, double c);

	/**
	 * @brief Adds two ciphertexts and stores the result in *this*.
	 *
	 * Handles self‑addition and aliasing cases; the method may copy the
	 * appropriate operand before performing the addition.
	 *
	 * @param ciphertext   First operand.
	 * @param ciphertext1  Second operand.
	 */
	void add(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);

	/**
	 * @brief Subtracts the second ciphertext from the first and stores the result in *this*.
	 *
	 * Handles self‑subtraction and aliasing cases similarly to `add`.
	 *
	 * @param ciphertext   Minuend.
	 * @param ciphertext1  Subtrahend.
	 */
	void sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);
	bool adjustScaleAndLevel(int scaleDegree, int level, double scaling_factor);

	/**
	 * @brief Copies the full content (polynomials and metadata) from another ciphertext.
	 *
	 * @param ciphertext Source ciphertext.
	 */
	void copy(const Ciphertext& ciphertext);

	/**
	 * @brief Adds a plaintext to a ciphertext and stores the result in *this*.
	 *
	 * @param ciphertext Ciphertext operand.
	 * @param plaintext  Plaintext operand.
	 */
	void addPt(const Ciphertext& ciphertext, const Plaintext& plaintext);

	/**
	 * @brief Reinterprets the internal data of `ciphertext` under the current context.
	 *
	 * The method checks that the source ciphertext belongs to a compatible
	 * context (same primes) and that its level does not exceed the current
	 * context's maximum.  It then copies the data.
	 *
	 * @param ciphertext Source ciphertext.
	 */
	// reinterpretContext removed — in-context ENCAPS has a single context (dual-context dance gone).

	/**
	 * @brief Performs a key‑switch operation using the provided switching key.
	 *
	 * @param ksk KeySwitchingKey to apply.
	 */
	void keySwitch(const KeySwitchingKey& ksk);

	/**
	 * @brief Adjusts internal scaling factors to make addition/subtraction possible.
	 *
	 * The method may rescale or multiply by 1.0 depending on the current
	 * `rescaleTechnique`.  It returns `true` when the adjustment succeeds.
	 *
	 * @param ciphertext Operand that may have a different scaling factor.
	 * @return `true` on successful adjustment.
	 */
	bool adjustForAddOrSub(const Ciphertext& ciphertext);

	/**
	 * @brief Adjusts scaling factors to enable multiplication with `ciphertext`.
	 *
	 * The method first calls `adjustForAddOrSub`; if the noise level of the
	 * operand is higher, the current ciphertext may be rescaled.  It returns
	 * `true` when the multiplication can proceed without further adjustment.
	 *
	 * @param ciphertext Operand to align scaling with.
	 * @return `true` on successful adjustment.
	 */
	bool adjustForMult(const Ciphertext& ciphertext);

	/**
	 * @brief Checks whether the ciphertext and a plaintext share the same scaling factor.
	 *
	 * The check accounts for a small tolerance (`1e-9` relative error).
	 *
	 * @param b Plaintext to compare against.
	 * @return `true` if scaling factors match within tolerance.
	 */
	[[nodiscard]] bool hasSameScalingFactor(const Plaintext& b) const;

	/** @brief Clears the operation record used for debugging/tracing. */
	static void clearOpRecord();

	/** @brief Prints the operation record to standard output. */
	static void printOpRecord();

	/**
	 * @brief Computes the dot product between arrays of ciphertexts and plaintexts.
	 *
	 * The operation multiplies corresponding components and accumulates the result
	 * in *this*.  When `ext` is true the method expects the inputs to have
	 * already performed modulus‑up conversion.
	 *
	 * @param ciphertexts Pointer to array of ciphertexts.
	 * @param plaintexts  Pointer to array of plaintexts.
	 * @param n Number of elements.
	 * @param ext If true, performs the operation on the extension libs aswell.
	 */
	void dotProductPt(Ciphertext* ciphertexts, Plaintext* plaintexts, const int n, bool ext);

	/**
	 * @brief Computes the dot product between ciphertexts and an array of plaintext pointers.
	 *
	 * @param ciphertexts Pointer to array of ciphertexts.
	 * @param plaintexts Pointer to array of pointers to plaintexts.
	 * @param n Number of elements.
	 * @param ext If true, performs the operation on the extension libs aswell.
	 */
	void dotProductPt(Ciphertext* ciphertexts, Plaintext** plaintexts, const int n, bool ext);

	/**
	 * @brief Computes the dot product between arrays of ciphertext pointers and plaintext pointers.
	 *
	 * @param ciphertexts Pointer to array of pointers to ciphertexts.
	 * @param plaintexts Pointer to array of pointers to plaintexts.
	 * @param n Number of elements.
	 * @param ext If true, performs the operation on the extension libs aswell.
	 */
	void dotProductPt(Ciphertext** ciphertexts, Plaintext** plaintexts, const int n, bool ext);
	void dotProduct(const std::vector<Ciphertext*>& a, const std::vector<Ciphertext*>& b, bool ext);

	/**
	 * @brief Multiplies the ciphertext by a monomial of the given `power`.
	 *
	 * The monomial is cached in the context; if not present, it is generated.
	 *
	 * @param power Exponent of the monomial.
	 */
	void multMonomial(int power);

  private:
	/**
	 * @brief The level/scale alignment that must precede a ciphertext-ciphertext tensor product,
	 *        shared by mult() and multNoRelin().
	 *
	 * Every path here is rescale / multScalar / dropToLevel — the helper never key-switches, so it
	 * cannot consume the shared keyswitch scratch.
	 *
	 * @param b The other multiplicand.
	 * @return `false` when *this could not be aligned to `b`; the caller must then retry against an
	 *         adjusted copy of `b` (see mult()).
	 */
	bool adjustBeforeMult(const Ciphertext& b);

	/**
	 * @brief The key-switch-and-fold tail shared by mult() and relinearize().
	 *
	 * Key-switches the degree-2 component held in `d2` with `kskEval` and folds both halves back
	 * into (c0, c1). `d2` is overwritten with the c1 half, so it must be a scratch polynomial that
	 * carries the DECOMP/DIGIT/GATHER buffers the hybrid key-switch needs — in practice always
	 * Context::getKeySwitchAux().
	 *
	 * @param d2      Degree-2 component to key-switch, in place.
	 * @param kskEval Evaluation (relinearization) key.
	 * @param moddown Perform the modulus down-conversion as part of the key-switch.
	 */
	void foldKeySwitched(RNSPoly& d2, const KeySwitchingKey& kskEval, bool moddown);

	/**
	 * @brief Acquire c2 from the context's auxiliary-polynomial pool, empty (level -1, no limbs) and
	 *        not mod-up, exactly as the constructor prepares c0 and c1.
	 */
	void acquireC2();

	/** @brief Return c2 to the context's auxiliary-polynomial pool and clear it. */
	void releaseC2();

	/**
	 * @brief Reject a degree-2 ciphertext for an operation defined only on degree-1 ones.
	 *
	 * A real throw rather than an assert: asserts are compiled out in Release, and several of the
	 * guarded entry points are reachable straight from the public api.
	 */
	void requireDegree1(const char* op) const;

	/**
	 * @brief Reject a plaintext that could not be brought onto this ciphertext's level and depth.
	 *
	 * @param adjusted Result of adjustPlaintextToCiphertext.
	 * @param original The plaintext handed to the operation, for the message.
	 * @param op       Operation name, for the message.
	 *
	 * A real throw rather than an assert, for the same reason as requireDegree1: the assert this
	 * replaces was compiled out in Release, leaving addPt/subPt to silently drop the plaintext.
	 */
	void requireAdjustedPt(bool ok, const Plaintext& adjusted, const Plaintext& original, const char* op) const;
};

/// @brief Record, in the BLOCK PROVENANCE TRACER, that the calling lane has just finished writing
/// `ct` -- i.e. that every block backing `ct`'s limbs is an OUTPUT of this lane, written on that
/// limb's stream.
///
/// WHY THIS EXISTS. The pool's own events say a block was freed by lane 0 and taken by lane 2.
/// That is only half an interleaving: it does not say that the block in question was the one
/// lane 2's bootstrap wrote its ANSWER into, which is what makes a dump readable as "lane 2 took
/// this block at seq N on stream S and wrote its output into it, while lane 0's last read of it
/// was fenced on stream S' at seq M". So the last op of a bootstrap marks its output here.
///
/// A no-op -- one cached-bool read -- unless FIDESLIB_POOL_TRACE=1 and FIDESLIB_CONCURRENT_OPS=1.
/// It reads pointers and stream handles only: no launch, no copy, no synchronize, and it does not
/// touch any stream's `updated` flag.
void PoolTraceMarkCiphertextOutput(const Ciphertext& ct);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_CIPHERTEXT_CUH
