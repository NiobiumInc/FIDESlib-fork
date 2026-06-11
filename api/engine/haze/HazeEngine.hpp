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
#include <vector>

namespace fideslib {

/// @brief haze (FHETCH) record/replay backend targeting the Niobium accelerator.
///
/// haze is a recorder: compute calls append FHETCH IR and nothing executes until hazeFlush().
/// The engine records ONE program per context: H2D values are the program's
/// inputs, ciphertexts declared via MarkOutput (or implicitly by the first Decrypt of a
/// computed value) are its outputs, and every other computed value is a program-local.
/// The single flush happens at the first readback; tagging only declared outputs keeps
/// readback DMA minimal on real hardware. Compute consuming a device-computed value after
/// the program executed is a hard error — flushes are exceptionally expensive in deployment.
///
/// CKKS algorithm logic (level/scale management, keyswitch, bootstrap staging) lives inside
/// this engine: haze's API is polynomial-level (per-residue MRP ops). Ciphertext data is
/// EVALUATION form, natural order, ordinary (non-Montgomery) representation, one hazeMalloc
/// allocation per RNS limb.
class HazeEngine final : public Engine {
  public:
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

	// ---- Bootstrap setup hooks (the compute itself lands in a later change) ----
	/// @brief Replicates the CPU policy verbatim (OpenFheEngine.cpp:400-408), including its
	/// flagged pre-existing arg-slot quirk — the host setup must match the CPU oracle.
	BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const override;
	void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) override;

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
	/// @brief Op guard: consuming a device-computed value (Recorded/Flushed) after the program
	/// executed is the one-program hard error. Ops whose device operands are all freshly Uploaded
	/// legitimately start a new program (host-in-the-loop re-encryption) — the caller resets
	/// executed_ via beginNewProgramIfExecuted() after guarding every operand.
	void requireComputable(const hazebk::HazePayload& p, const char* op) const;
	/// @brief After all operands of an op passed requireComputable while executed_ was set,
	/// the op starts a legitimately new program over fresh inputs only.
	void beginNewProgramIfExecuted();

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
	/// @brief OpenFHE AdjustLevelsAndDepthInPlace port (FIXEDAUTO; compositeDegree == 1):
	/// equalize (level, NSD) of the two views via recorded compute + view truncation.
	/// FIXEDMANUAL: level-align by truncation only. FLEXIBLE modes: notImplemented.
	void adjustForAddOrSub(Operand& a, Operand& b);
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
	/// @brief Hybrid keyswitch of `src` (degree-1 component, EVAL form, first `towers` Q
	/// primes) against `key` — same math for relin (ct×ct) and automorphism (rotation)
	/// keys. Returns the (b, a) contribution in EVAL form at `towers` Q primes.
	KsContribution hybridKeyswitch(const hazebk::LimbChain& src, size_t towers, KsKey& key);

	/// @brief OpenFHE AdjustLevelsAndDepthToOneInPlace: adjustForAddOrSub, then rescale both
	/// to NSD 1 when needed, so both enter the tensor product at depth 1.
	void adjustForMult(Operand& a, Operand& b);
	/// @brief Rotation (ops.cpp rotate): hybridKeyswitch(c1) against the step's
	/// automorphism key, c0' = c0 + ks.b / c1' = ks.a, then AutomorphMrp BOTH (keyswitch
	/// first, automorphism last — OpenFHE EvalAtIndex order). Metadata unchanged.
	Operand rotateCore(const Operand& x, int32_t step);
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
