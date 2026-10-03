#ifndef API_HAZEENGINE_HPP
#define API_HAZEENGINE_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/Engine.hpp"
#include "engine/haze/HazePayload.hpp"

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

	// ---- Operations (aligned ct+ct addition only; the FIXEDAUTO adjust family
	// and the remaining op surface land in later phases) ----
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;

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
