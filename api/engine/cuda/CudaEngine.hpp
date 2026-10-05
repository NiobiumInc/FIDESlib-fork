#ifndef API_CUDAENGINE_HPP
#define API_CUDAENGINE_HPP

#ifdef FIDESLIB_ENABLE_CUDA

#include "engine/Engine.hpp"

#include "CKKS/forwardDefs.cuh" // FIDESlib::CKKS::Context (= shared_ptr<ContextData>)

#include <memory>
#include <vector>

namespace fideslib {

/// @brief CUDA backend. Every operation runs FIDESlib's CUDA CKKS layer.
/// All CUDA operation code lives in CudaEngine.cpp.
class CudaEngine final : public Engine {
  public:
	const char* name() const override {
		return "FIDESlib (CUDA)";
	}

	Backend backend() const override {
		return Backend::CUDA;
	}

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
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalMultNoRelin(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> relinearize(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void relinearizeInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	void evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	Ciphertext<DCRTPoly>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) override;
	Ciphertext<DCRTPoly>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) override;
	std::vector<Ciphertext<DCRTPoly>> evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
	  const Ciphertext<DCRTPoly>& ct,
	  const std::vector<int32_t>& indices,
	  const uint32_t m,
	  const std::shared_ptr<void>& precomp) override;
	std::vector<Ciphertext<DCRTPoly>>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) override;
	std::vector<Ciphertext<DCRTPoly>> evalRotateMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& cts, int32_t index) override;
	Ciphertext<DCRTPoly> evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;
	void evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;
	Ciphertext<DCRTPoly> rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) override;
	void rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) override;
	Ciphertext<DCRTPoly> accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) override;
	BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const override;
	void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) override;
	Ciphertext<DCRTPoly>
	evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	/// @brief Variable-output-level bootstrap. Overridden here because the CUDA backend can
	/// make the refresh ITSELF cheaper at a reduced output level: the mod-raise stops short
	/// of the chain top, so CoeffsToSlots, EvalMod and SlotsToCoeffs all run on fewer limbs.
	Ciphertext<DCRTPoly>
	evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void evalBootstrapToLevelInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	/// @brief The one bootstrap implementation. @p levelsToDrop 0 is evalBootstrap; a positive
	/// value stops the mod-raise that many towers below the chain top (and, at more than one
	/// iteration, stops BOTH inner raises there, so the Meta-BTS glue still sees two operands
	/// on the same chain). Requires the matching reduced-level precomputation.
	Ciphertext<DCRTPoly>
	evalBootstrapInternal(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled, int levelsToDrop);
	/// @brief The two-iteration (Meta-BTS) composition with its glue on the device; taken by
	/// evalBootstrapInternal when FIDESLIB_METABTS_DEVICE=1 (MetaBtsDevice). Same steps as the
	/// host composition, no readbacks.
	Ciphertext<DCRTPoly>
	evalBootstrapMetaDevice(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t precision, bool prescaled, int levelsToDrop);
	/// @brief Install the reduced-level bootstrap precomputation for (slots, levelsToDrop) if
	/// it is not there yet. Host-scheme work, so it takes the host lock.
	void ensureReducedBootPrecomputation(CryptoContextImpl<DCRTPoly>& ctx, int slots, int levelsToDrop);
	/// @brief The level an ordinary device bootstrap returns a ciphertext at, in OpenFHE's
	/// convention (consumed towers) — the floor evalBootstrapToLevel clamps a request to.
	/// Read out of the SlotsToCoeffs diagonals, which are encoded for the chain the last
	/// transform of the classic bootstrap ends on, so it needs no duplicate of
	/// EvalBootstrapPrecompute's level arithmetic. Classic order only, which is the only
	/// order the device Bootstrap implements.
	uint32_t bootstrapOutputLevel(CryptoContextImpl<DCRTPoly>& ctx, int slots);
	void linearTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool ext) override;
	std::vector<Ciphertext<DCRTPoly>> linearTransformMany(CryptoContextImpl<DCRTPoly>& ctx,
	  const Ciphertext<DCRTPoly>& ct,
	  int rowSize,
	  int bStep,
	  const std::vector<std::vector<Plaintext>>& diagonalSets,
	  int stride,
	  int offset,
	  bool ext) override;

	// ---- Prepared linear transforms ----
	// This is the backend the prepared form exists for: the diagonal third of the MAC kernel's
	// device pointer table is built and uploaded ONCE here, into a FIDESlib::CKKS::PreparedLTTable
	// owned by the handle, instead of being rebuilt into a host vector and copied out of pageable
	// memory on every giant step. Numerics are untouched — the same pointers, same order, same
	// kernel; only who assembled the table and when changes.
	PreparedLinearTransform
	prepareLinearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<std::vector<Plaintext>>& diagonalSets, const LinearTransformShape& shape) override;
	uint64_t plaintextIdentity(CryptoContextImpl<DCRTPoly>& ctx, const Plaintext& pt) override;
	void linearTransformInPlacePrepared(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) override;
	std::vector<Ciphertext<DCRTPoly>>
	linearTransformManyPrepared(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) override;

	/// @brief True: the device holds the special (P) primes, so a plaintext can live in Q||P.
	bool supportsExtendedPlaintexts() const override;
	void convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  const std::vector<int>& indexes,
	  int stride,
	  int rowSize) override;
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
	/// @brief True: limbs live on the device, so building them there saves the transfer entirely.
	bool supportsDeviceEncode() const override;
	void encodeDevice(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const DeviceEncodedCoefficients& enc) override;
	/// @brief True: a batch shares one upload, one reduce launch and one NTT launch pair per device.
	bool supportsDeviceEncodeMany() const override;
	void encodeDeviceMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const std::vector<DeviceEncodedCoefficients>& encs) override;
	bool supportsDeviceSlotsEncode() const override;
	void encodeDeviceSlotsMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const DeviceEncodedSlots& enc) override;
	/// @brief The one backend that has something parked: forwards to ContextData::releaseParkedScratch.
	ScratchReleaseStats releaseParkedScratch(CryptoContextImpl<DCRTPoly>& ctx) override;
	void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) override;
	void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) override;
	void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;

	std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src) override;
	size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	double ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextNumElements(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	void refreshHostShadow(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct) override;
	void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots) override;
	void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) override;

	bool isContextLoaded() const override;
	void synchronize() const override;
	void teardown() override;
	void setDevices(const std::vector<int>& devices) override;
	std::vector<int> devices() const override;

	/// @brief Opt-in for harnesses that never read an uploaded input's host residues: drop them
	/// right after upload. Readback stays correct because refreshHostShadow rebuilds an emptied
	/// shell from the device residues. Uploaded values must not share their host shell with a
	/// value that stays host-only -- the drop mutates the shared shell in place.
	void setDropInputHostAfterUpload(bool drop) {
		dropInputHostAfterUpload_ = drop;
	}

	// ---- CUDA-owned per-context state ----
  private:
	/// @brief Negate in place by the per-limb scalar q_i - 1, preserving noise degree and scaling
	/// factor (OpenFHE's EvalNegate semantics). Applied to every component the ciphertext carries.
	void negateExact(FIDESlib::CKKS::Ciphertext& ct_gpu) const;

	/// @brief Throw unless a relinearization key is registered for this ciphertext's key tag.
	/// Queried from lbcrypto's own eval-mult key map — the same source loadContext consults when it
	/// decides whether to upload the key — and called BEFORE any degree branch, because OpenFHE's
	/// Relinearize/RelinearizeInPlace hit GetEvalMultKeyVector(tag) first and that accessor throws.
	void requireRelinKey(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct, const char* op) const;

	/// @brief CUDA CKKS context; null == not loaded. Owned by this engine (one engine per context).
	std::unique_ptr<FIDESlib::CKKS::Context> context_;
	/// @brief Devices this context is loaded on (default: device 0).
	std::vector<int> devices_ = { 0 };
	/// @brief See setDropInputHostAfterUpload.
	bool dropInputHostAfterUpload_ = false;

  public:
	~CudaEngine() override;
};

} // namespace fideslib

#endif // FIDESLIB_ENABLE_CUDA
#endif
