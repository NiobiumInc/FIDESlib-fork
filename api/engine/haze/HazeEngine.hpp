#ifndef API_HAZEENGINE_HPP
#define API_HAZEENGINE_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/Engine.hpp"
#include "engine/haze/HazeKeyExtract.hpp"
#include "engine/haze/HazePayload.hpp"
#include "engine/haze/HazeScalarEncode.hpp"

#include <map>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fideslib {

/// @brief haze (FHETCH) record/replay backend targeting the Niobium accelerator.
///
/// haze is a recorder: compute calls append FHETCH IR and nothing executes until hazeFlush().
/// The engine records ONE program per context: H2D values are the program's
/// inputs, ciphertexts declared via MarkOutput (or implicitly by the first Decrypt of a
/// computed value) are its outputs, and every other computed value is a program-local.
/// The single flush happens at the first readback; tagging only declared outputs keeps
/// readback DMA minimal on real hardware. Any compute after the program has executed (the
/// first readback) is a hard error — flushes are exceptionally expensive in deployment.
///
/// CKKS algorithm logic (level/scale management, keyswitch, bootstrap staging) lives inside
/// this engine: haze's API is polynomial-level (per-residue MRP ops). Ciphertext data is
/// EVALUATION form, natural order, ordinary (non-Montgomery) representation, one hazeMalloc
/// allocation per RNS limb.
class HazeEngine final : public Engine {
  public:
	/// @param reducedNoise  Use the centered (ReducedNoise) FBC variant, matching an OpenFHE
	///                      reference built with WITH_REDUCED_NOISE — required for bit-exact parity.
	/// @param montgomery    Record in the Montgomery hardware data format (selects the 4-op
	///                      SwitchModulus FBC center shape). Only valid against a hardware/transport
	///                      target; the local simulator rejects Montgomery-form traces.
	/// Both default off so a bare HazeEngine matches the libhaze defaults; the facade selects them
	/// from CCParams (SetReducedNoise / SetMontgomery) at GenCryptoContext.
	explicit HazeEngine(bool reducedNoise = false, bool montgomery = false)
		: reducedNoise_(reducedNoise), montgomery_(montgomery) {}

	const char* name() const override {
		return "haze (FHETCH)";
	}

	Backend backend() const override {
		return Backend::HAZE;
	}

	// ---- Lifecycle + residency ----
	void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) override;
	void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) override;
	void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	void markOutput(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;

	// ---- Ciphertext backend hooks ----
	std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src) override;
	size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots) override;
	void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) override;

	// ---- Linear operations + scalar multiplication ----
	Ciphertext<DCRTPoly> evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	Ciphertext<DCRTPoly> evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) override;
	void evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;

	// ---- Multiplication family + rescale ----
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) override;
	void rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) override;

	// ---- Rotation + accumulate ----
	Ciphertext<DCRTPoly> evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	void evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) override;
	Ciphertext<DCRTPoly>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) override;
	std::vector<Ciphertext<DCRTPoly>>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) override;
	std::vector<Ciphertext<DCRTPoly>>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) override;
	Ciphertext<DCRTPoly> accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) override;

	// ---- Chebyshev series ----
	/// @brief Port of OpenFHE's EvalChebyshevSeries (linear/PS split) over the facade.
	/// Degree < 5: linear path (internalEvalChebyPolysLinear + LinearWithPrecomp).
	/// Degree >= 5: Paterson-Stockmeyer path (internalEvalChebyPolysPS + PSWithPrecomp).
	/// Coefficients and (a,b) interval handling ported exactly from
	/// deps/openfhe-src/src/pke/lib/scheme/ckksrns/ckksrns-advancedshe.cpp.
	Ciphertext<DCRTPoly> evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;
	void evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;

	// ---- Convolution transform ----
	/// @brief Port of cpuConvolutionTransform (OpenFheEngine.cpp:443-538) over the facade.
	/// kCpuInternalGStep=8, hoisted baby steps, block giant steps, cpuTreeAccumulate.
	void convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  const std::vector<int>& indexes,
	  int stride,
	  int rowSize) override;
	/// @brief Masked variant: each giant-step result is folded with two mask rotations before
	/// the intra-block rotation (mask != nullptr path in cpuConvolutionTransform).
	void specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride,
	  int maskRotationStride,
	  int rowSize) override;

	// ---- Bootstrap ----
	/// @brief Replicates the CPU policy verbatim (OpenFheEngine.cpp:400-408), including its
	/// flagged pre-existing arg-slot quirk — the host setup must match the CPU oracle.
	BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const override;
	void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) override;
	Ciphertext<DCRTPoly>
	evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;

	// ---- Context backend state ----
	bool isContextLoaded() const override;
	/// @brief No-op: haze records synchronously; hazeDeviceSynchronize is itself a no-op and
	/// the engine's single flush is triggered only by the first readback.
	void synchronize() const override;
	void teardown() override;
	void setDevices(const std::vector<int>& devices) override;
	std::vector<int> devices() const override;

	~HazeEngine() override;

  private:
	// ---- one-program execution ----
	/// @brief hazeTagOutput every residue chain of every declared output, then ONE hazeFlush;
	/// mark them Flushed and set executed_. All subsequent reads of declared outputs are pure
	/// shadow D2H: N decrypts, 1 flush, declared up front.
	void materialize();
	/// @brief Op guard: the context runs exactly one program. Any compute after the single
	/// flush (the first readback) is the one-program hard error — there is no fresh-input exception;
	/// multi-step host-in-the-loop work must use a fresh context.
	void requireComputable(hazebk::HazePayload& p, const char* op);

	/// @brief Ensure ct is device-resident (lazy load) and return its payload.
	std::shared_ptr<hazebk::HazePayload> ensureCt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);

	/// @brief Q-base prefix for the first `towers` limbs (the per-op MRP base argument).
	std::vector<uint64_t> qPrefix(size_t towers) const;

	// ---- FIXEDAUTO adjust + scalar-op cores (OpenFHE ckksrns-leveledshe is the oracle) ----

	/// @brief Adjusted read-only view of a ciphertext payload. `towers` may be smaller than
	/// p->towers (OpenFHE LevelReduce is a view truncation here — no IR, no copy), and the
	/// metadata fields are the post-adjust values. `p` is either the operand's own payload
	/// (no adjustment recorded) or a fresh payload produced by recorded compute — a const
	/// operand is never mutated.
	struct Operand {
		std::shared_ptr<hazebk::HazePayload> p;
		size_t towers;
		size_t noiseScaleDeg;
		double scalingFactor;
		size_t slots;
	};
	Operand asOperand(const std::shared_ptr<hazebk::HazePayload>& p) const;
	/// @brief OpenFHE level of a view: |Q| − towers.
	size_t levelOf(const Operand& x) const {
		return qBase_.size() - x.towers;
	}
	/// @brief Borrowed views of the engine's scale-factor caches for HazeScalarEncode.
	hazebk::ScalarEncodeParams scalarParams() const;
	/// @brief Recorded per-residue pass-through copy of the leading `towers` limbs
	/// (epoch.cpp copy_device_to_device) — for result components an op leaves unchanged.
	hazebk::LimbChain passThroughChain(const hazebk::LimbChain& src, size_t towers);

	/// @brief INTT → ModDown (drop the last Q prime; centered lift + q_l^{-1}, matching
	/// OpenFHE ModReduce exactly) → NTT on one chain (port of ops.cpp rescale_chain_one_tower).
	hazebk::LimbChain rescaleChainOneTower(const hazebk::LimbChain& src, size_t srcTowers);
	/// @brief OpenFHE EvalMultCoreInPlace analog into a fresh payload: per-limb CRT scalar
	/// multiply, NSD+1, sf ×= ScalingFactorReal[level]. No pre-rescale.
	Operand multScalarCore(const Operand& x, double operand);
	/// @brief OpenFHE ModReduceInternalInPlace (one level) analog into a fresh payload:
	/// rescale both chains, towers−1, NSD−1 (guard ≥1), sf ÷= ModReduceFactor[oldTowers−1].
	Operand rescaleCore(const Operand& x);
	/// @brief Exact EvalNegate into a fresh payload: per-limb scalar q_i − 1; metadata
	/// unchanged (do not copy CUDA's multScalar(-1.0), which bumps NSD).
	Operand negateCore(const Operand& x);
	/// @brief OpenFHE AdjustLevelsAndDepthInPlace port (all AUTO modes share one
	/// implementation; compositeDegree == 1): equalize (level, NSD) of the two views via
	/// recorded compute + view truncation. FIXEDMANUAL: level-align by truncation only.
	void adjustForAddOrSub(Operand& a, Operand& b);
	/// @brief Adjust `x` toward `tgt` (the c1lvl<c2lvl branch body of
	/// AdjustLevelsAndDepthInPlace). Works on 1-component operands too (morphed
	/// plaintexts: empty c1 chains are skipped by the cores). `ctRule` selects the depth2/depth2
	/// branch1 rule (parity #5): true = CUDA Ciphertext::adjustScaleAndLevel (q1=ModReduceFactor[c2lvl+1],
	/// trim-before-rescale, used for ciphertext operands); false = CUDA Plaintext::adjustScaleAndLevel
	/// (q1=ModReduceFactor[c1lvl], rescale-before-trim, used only by adjustPtToward).
	void adjustOperandToward(Operand& x, const Operand& tgt, bool ctRule = true);
	/// @brief GPU Plaintext::adjustPlaintextToCiphertext port: adjust a plaintext operand
	/// (single chain) toward the ciphertext's (level, NSD, scalingFactor). FIXEDAUTO uses
	/// the drop/rescale rules from src/CKKS/Plaintext.cu:195-225; FLEXIBLE modes reuse
	/// adjustOperandToward. Throws when the pt is deeper than the ct (cannot raise).
	void adjustPtToward(Operand& ptOp, const Operand& ct);
	/// @brief FIXEDAUTO multScalar precheck (Ciphertext.cpp:761-775): rescale first when
	/// NSD == 2, then multScalarCore.
	Operand multScalarWithPrecheck(const Operand& x, double scalar);
	/// @brief c0 ± pt with the FIXEDAUTO rescale-if rule (Ciphertext.cpp:250-252): rescale
	/// the ct when (pt.NSD==1, ct.NSD==2, ptTowers == ctTowers−1); then require level/NSD
	/// agreement with the pt chain trimmed to the ct's towers. c1 passes through (D2D).
	Operand applyPt(const Operand& ct, const hazebk::HazePtPayload& pt, bool subtract);
	/// @brief Encode ±scalar per ElemForEvalAddOrSub (negative: per-limb q_i − e_i flip)
	/// and add it onto c0; metadata unchanged. c1 passes through (D2D).
	Operand addScalarCore(const Operand& x, double scalar);

	/// @brief Materialize a view as a payload: if the view truncates its payload, record
	/// nothing — build the result payload that owns fresh chains is the OPS' job; this
	/// simply packages computed chains + view metadata into a fresh Recorded payload.
	std::shared_ptr<hazebk::HazePayload> finishPayload(hazebk::LimbChain c0, hazebk::LimbChain c1, const Operand& meta) const;
	/// @brief Rebind an existing payload's contents to a computed result (facade "InPlace"
	/// semantics: fresh chains per op result; the old chains are freed).
	void rebindPayload(hazebk::HazePayload& dst, hazebk::LimbChain c0, hazebk::LimbChain c1, const Operand& meta) const;
	/// @brief Ensure the plaintext is device-resident and return its payload.
	std::shared_ptr<hazebk::HazePtPayload> ensurePt(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt);

	// ---- Hybrid keyswitch (ops.cpp hybrid_keyswitch is the verified reference) ----

	/// @brief A HYBRID keyswitch key: host limbs + the lazily-uploaded device chains
	/// (full Q∥P rows per digit; Uploaded state, never tagged, reused across epochs).
	struct KsKey {
		hazebk::HybridKeyswitchLimbs host;
		std::vector<hazebk::LimbChain> aDigits; // [digit] -> |Q|+|P| rows
		std::vector<hazebk::LimbChain> bDigits;
		bool uploaded = false;
	};
	/// @brief H2D-upload the key's full Q∥P rows once (per-call trimming is pointer
	/// selection, not re-upload).
	void ensureKeyUploaded(KsKey& key);

	struct KsContribution {
		hazebk::LimbChain b;
		hazebk::LimbChain a;
	};

	/// @brief Hoisting handle (#19): the input-only prefix of hybridKeyswitch — the per-digit
	/// ModUp+NTT decomposition of one degree-1 component (c1), in EVAL form over Q∥P. This is the
	/// shared "ModUp once" of CUDA's rotate_hoisted (Ciphertext.cpp:965 c1.modupInto): it depends
	/// ONLY on (c1, towers), NOT on any key, so it is computed once and reused across every rotation
	/// index / key row (the per-key dot product is the cheap part). digitsEval holds numDigits
	/// blocks of qpTowers limbs; the metadata mirrors the values hybridKeyswitch would recompute.
	struct HoistedDigits {
		hazebk::LimbChain digitsEval; // [numDigits * qpTowers] EVAL-form digits over Q∥P
		size_t towers	= 0;		  // Q-prefix length the decomposition was built at
		size_t numDigits = 0;
		size_t qpTowers = 0;		  // towers + |P| (the Q∥P row count per digit)
		std::vector<uint64_t> qpBase; // Q-prefix ∥ P moduli (the per-op MRP base for step 3)
	};
	/// @brief #19 step 1-2: factor the key-independent ModUp/per-digit-NTT prefix of
	/// hybridKeyswitch out of the per-key work. `numPartQ` is the key's digit count (alpha
	/// partition is keyed off it); the result is the reusable HoistedDigits handle.
	HoistedDigits hoistedKeyswitchPrefix(const hazebk::LimbChain& src, size_t towers, size_t numPartQ);
	/// @brief #19 step 3: per-key dot product of a precomputed HoistedDigits handle against
	/// `key`, then (ext=false) ModDown(drop P)+NTT to base-Q, or (ext=true, #10) KEEP the
	/// extended Q∥P basis and SKIP the ModDown (CUDA rotate_hoisted ext=true,
	/// Ciphertext.cpp:994-998). The returned KsContribution chains are at `digits.towers` Q
	/// primes (ext=false) or `digits.qpTowers` Q∥P rows in EVAL form (ext=true).
	KsContribution hybridKeyswitchFromDigits(const HoistedDigits& digits, KsKey& key, bool ext = false);
	/// @brief Hybrid keyswitch of `src` (degree-1 component, EVAL form, first `towers` Q
	/// primes) against `key` — same math for relin (ct×ct) and automorphism (rotation)
	/// keys. Returns the (b, a) contribution in EVAL form at `towers` Q primes. Now a thin
	/// composition of hoistedKeyswitchPrefix + hybridKeyswitchFromDigits (so relin and
	/// non-hoisted rotation callers stay byte-identical).
	KsContribution hybridKeyswitch(const hazebk::LimbChain& src, size_t towers, KsKey& key);

	/// @brief OpenFHE AdjustLevelsAndDepthToOneInPlace: adjustForAddOrSub, then rescale both
	/// to NSD 1 when needed, so both enter the tensor product at depth 1.
	void adjustForMult(Operand& a, Operand& b);
	/// @brief Rotation (ops.cpp rotate): hybridKeyswitch(c1) against the step's
	/// automorphism key, c0' = c0 + ks.b / c1' = ks.a, then AutomorphMrp BOTH (keyswitch
	/// first, automorphism last — OpenFHE EvalAtIndex order). Metadata unchanged. Steps
	/// without a pre-extracted key fall back to FindAutomorphismIndex + the lazy auto-key
	/// cache (bootstrap rotations).
	Operand rotateCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int32_t step);
	/// @brief Resolve the slots-aware rotation key + automorphism index for `step` (the
	/// normalyzeIndex + rotKeyIndex_ alternate-key search from rotateCore, #7). Returns the
	/// RotKey* and its autoIndex when a pre-extracted key matches (fast-rotation path); returns
	/// nullptr when the step has no registered key (fast rotation has no host fallback — that is
	/// the plain-rotate path's job).
	struct RotKeyResolved {
		KsKey* key		   = nullptr;
		uint32_t autoIndex = 0;
	};
	RotKeyResolved resolveRotationKey(int32_t step, size_t slots);
	/// @brief Fast rotation of `x` reusing a precomputed HoistedDigits handle (#19): per index
	/// it does ONLY the cheap per-key step (hybridKeyswitchFromDigits) + the same c0+ks.b /
	/// AutomorphMrp assembly as rotateCore, so the result is byte-identical to evalRotate(x,step)
	/// while the expensive ModUp/NTT decomposition is shared across indexes. Requires a
	/// pre-extracted key for `step` (resolveRotationKey); throws if absent.
	Operand fastRotateFromDigits(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int32_t step, const HoistedDigits& digits);
	/// @brief ct×pt with the multPt adjust rules (Ciphertext.cpp:356-421): full polynomial
	/// MulMrp of both components against the pt chain (never the slot-constant shortcut).
	Operand multPtCore(const Operand& ct, const hazebk::HazePtPayload& pt);

	// ---- keys ----
	KsKey relinKey_;
	bool haveRelinKey_ = false;
	struct RotKey {
		uint32_t autoIndex = 0;
		KsKey key;
	};
	std::map<int32_t, RotKey> rotKeys_; // slot step -> key
	/// @brief CUDA-style normalized rotation index ((step mod N/2), made positive) -> the slot
	/// step in rotKeys_. Built from rotKeys_; used by the slots-aware GetRotationKey port (#7,
	/// incl. the alternate slot-compatible key search) so rotateCore can pick CUDA's actual_index.
	std::map<int32_t, int32_t> rotKeyIndex_;
	/// @brief Lazily-extracted automorphism keys by raw automorphism index (bootstrap
	/// rotations, conjugation 2N−1, and any rotation step not in rotKeys_).
	std::map<uint32_t, KsKey> autoKeys_;
	std::string keyTag_;
	/// @brief Resolve (extract + cache) the automorphism key for autoIndex from the host
	/// context's key map.
	KsKey& autoKeyFor(CryptoContextImpl<DCRTPoly>& ctx, uint32_t autoIndex);
	/// @brief Keyswitch+automorph by raw automorphism index (rotateCore generalization;
	/// conjugation = autoIndex 2N−1).
	Operand rotateByAutoIndex(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, uint32_t autoIndex);

	// ---- bootstrap precomputation (extracted at loadContext per slots_bootstrap) ----
	struct BootPrecom {
		uint32_t slots = 0;
		bool isLT	   = false; // levelBudget {1,1} (the acceptance path)
		// OpenFHE precom->BTSlotsEncoding (ckksrns-fhe.cpp:125). CUDA's Engine path is ALWAYS
		// ModRaise-first (Bootstrap.cu:169) and forwards btsfirstboot into this flag (default
		// false). bootstrapStaged selects the ModRaise-first port when false (CUDA-matching) and
		// keeps the StC-first port when true (the CPU oracle's BTSlotsEncoding=true configs).
		bool btSlotsEncoding = false;
		uint32_t bStep = 0;		// m_dim1 / m_paramsEnc.g; 0 -> ceil(sqrt(slots))
		uint32_t correctionFactor = 0;
		std::vector<Plaintext> u0hatTPre; // CoeffsToSlots linear-transform plaintexts (isLT)
		std::vector<Plaintext> u0Pre;	  // SlotsToCoeffs linear-transform plaintexts (isLT)
		// Multi-stage FFT precom (levelBudget != {1,1}, !isLT). Per-stage plaintext vectors
		// (m_U0hatTPreFFT / m_U0PreFFT) plus the BSGS params (m_paramsEnc / m_paramsDec). The
		// driver iterates stages with plain EvalRotate (Task 06 hoists). decode (StC) uses Dec.
		struct FFTParams {
			uint32_t lvlb = 0, layersCollapse = 0, remCollapse = 0;
			uint32_t numRotations = 0, b = 0, g = 0;
			uint32_t numRotationsRem = 0, bRem = 0, gRem = 0;
		};
		FFTParams paramsEnc, paramsDec;
		std::vector<std::vector<Plaintext>> u0hatTPreFFT; // CtS per-stage plaintexts (!isLT)
		std::vector<std::vector<Plaintext>> u0PreFFT;	  // StC per-stage plaintexts (!isLT)
		std::vector<double> coefficients; // Chebyshev table for the key distribution
		double k		 = 0.0;
		uint32_t numIter = 0; // double-angle iterations
	};
	std::unordered_map<uint32_t, BootPrecom> boot_;
	uint64_t plaintextModulus_ = 0;
	/// @brief Hybrid-keyswitch digit count (cryptoParams->GetNumPartQ()), a context constant
	/// shared by every relin/rotation key. Captured at loadContext so the hoisting prefix (#19)
	/// can build the shared digit decomposition without a specific key in hand.
	size_t numPartQ_ = 0;

	/// @brief Host-only extraction of the OpenFHE bootstrap precomputation for `slots`
	/// (m_bootPrecomMap fields + Chebyshev config; ExtractBootPrecom).
	void extractBootPrecom(CryptoContextImpl<DCRTPoly>& ctx, uint32_t slots);

	// ---- bootstrap cores (HazeBootstrap.cpp) ----
	/// @brief ModRaise: from the level-0 limb, INTT@{q0} → hazeBasisConvert({q0}→Q) →
	/// NTT@Q on both components; towers = |Q|, NSD/sf unchanged (OpenFHE raise semantics).
	Operand modRaiseCore(const Operand& x, size_t targetTowers);
	/// @brief Integer scalar multiply (OpenFHE MultByIntegerInPlace): scalar mod q_i per
	/// limb, metadata unchanged.
	Operand multIntCore(const Operand& x, uint64_t scalar);
	/// @brief BSGS linear transform with precomputed plaintexts (plain-rotation equivalent
	/// of OpenFHE EvalLinearTransform).
	Ciphertext<DCRTPoly> linearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Plaintext>& a, const Ciphertext<DCRTPoly>& ct, uint32_t bStep);
	/// @brief Shared staged bootstrap dispatcher. Selects the CUDA-matching ModRaise-first
	/// variant (Bootstrap.cu:169) when bp.btSlotsEncoding is false — the path the Engine always
	/// runs — and the StC-first variant (EvalBootstrapStCFirst port) when true.
	Ciphertext<DCRTPoly> bootstrapStaged(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext);
	/// @brief ModRaise-first (normal) bootstrap — CUDA FIDESlib::CKKS::Bootstrap (Bootstrap.cu:169);
	/// the variant CudaEngine::evalBootstrap always dispatches. Selected for btSlotsEncoding=false.
	Ciphertext<DCRTPoly> bootstrapModRaiseFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext);
	/// @brief StC-first (slim) bootstrap — OpenFHE EvalBootstrapStCFirst port; kept for the
	/// CPU-oracle's btSlotsEncoding=true configs (CUDA never selects this through the Engine).
	Ciphertext<DCRTPoly> bootstrapStCFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext);
	/// @brief Multi-stage FFT CoeffsToSlots / SlotsToCoeffs driver (levelBudget != {1,1}),
	/// iterating the per-stage LTstep vectors (CoeffsToSlots.cu:75-151). decode=false → CtS.
	Ciphertext<DCRTPoly> evalCoeffsToSlotsFFT(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, uint32_t slots, bool decode);

	// ---- Chebyshev Paterson-Stockmeyer cores (parity #12-#15; CUDA src/CKKS/ApproxModEval.cu
	// + Ciphertext.cpp evalLinearWSumMutable are the spec). ----

	/// @brief Fused weighted-sum producing a result Operand directly at `targetTowers` (NSD=2):
	/// out = sum_i weights[i]*ops[i], computed via multScalar (bootstrap-prescale encoding,
	/// towersIn = ops[i].towers) + add-accumulate, exactly reproducing CUDA
	/// Ciphertext::evalLinearWSumMutable (Ciphertext.cpp:1168). CUDA grows a fresh ct to the
	/// target level and fills it; haze cannot grow (setCiphertextLevel only drops), so this
	/// produces the result at `targetTowers` directly. NoiseFactor = ScalingFactorReal[level]^2
	/// where level = |Q| - targetTowers. All ops[i] must have towers >= targetTowers.
	Operand evalLinearWSumMutableCore(size_t targetTowers,
	  const std::vector<Operand>& ops,
	  const std::vector<double>& weights);
	/// @brief Facade wrapper: extract operands from `ctxs`, call evalLinearWSumMutableCore at the
	/// CUDA target level, and wrap the result back into a value-type Ciphertext parented to ctx.
	Ciphertext<DCRTPoly> evalLinearWSumMutableFacade(CryptoContextImpl<DCRTPoly>& ctx,
	  size_t targetTowers,
	  const std::vector<Ciphertext<DCRTPoly>>& ctxs,
	  const std::vector<double>& weights);
	/// @brief Port of CUDA innerEvalChebyshevPS (ApproxModEval.cu:138) — Paterson-Stockmeyer
	/// recursion threading level_offset (q@level_offset, s@level_offset+1) and max_m (cu caching
	/// when max_m-m<=1). Returns the recursive PS result; the caller subtracts T2km1.
	Ciphertext<DCRTPoly> hazeInnerEvalChebyshevPS(CryptoContextImpl<DCRTPoly>& ctx,
	  const std::vector<double>& coefficients,
	  uint32_t k, uint32_t m,
	  const std::vector<Ciphertext<DCRTPoly>>& T,
	  const std::vector<Ciphertext<DCRTPoly>>& T2,
	  int level_offset, int max_m);
	/// @brief Port of CUDA evalChebyshevSeries (ApproxModEval.cu:370) — always PS (no degree
	/// dispatch, #12), CUDA centering affine map (#13), T/T2/T2km1 build, then InnerEvalChebyshevPS.
	Ciphertext<DCRTPoly> hazeEvalChebyshevSeriesImpl(CryptoContextImpl<DCRTPoly>& ctx,
	  const Ciphertext<DCRTPoly>& ct,
	  std::vector<double>& coeffs,
	  double a, double b);

	// ---- FBC mode (constructor-selected, forwarded to libhaze at bring-up) ----
	bool reducedNoise_ = false;
	bool montgomery_   = false;

	// ---- context state ----
	bool loaded_	  = false;
	uint64_t ringDim_ = 0;
	size_t polyBytes_ = 0;
	std::vector<uint64_t> qBase_; // Q moduli, tower order (haze table order: Q then P)
	std::vector<uint64_t> pBase_; // P moduli
	int scalingTech_ = 0;		  // lbcrypto::ScalingTechnique (int keeps openfhe out of this header)
	// Per-level scale-factor caches in OpenFHE level orientation (do NOT copy
	// FIDESlib's flipped [L-i] indexing). modReduceFactor_ is tower-indexed
	// GetModReduceFactor(i) (approxSF for FIXED modes, double(q_i) for FLEXIBLE).
	std::vector<double> sfReal_, sfRealBig_, modReduceFactor_;
	/// @brief Outputs declared via markOutput (weak: a dropped ciphertext is not an output).
	std::vector<std::weak_ptr<hazebk::HazePayload>> outputs_;
	/// @brief The in-flight program has executed (its single flush happened).
	bool executed_ = false;
	/// @brief Program directory of this context (cleaned up at teardown unless kept).
	std::string programDir_;

	/// @brief haze configuration is process-global: one loaded haze context per process.
	static inline std::atomic<int> liveEngines_{ 0 };
};

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
#endif // API_HAZEENGINE_HPP
