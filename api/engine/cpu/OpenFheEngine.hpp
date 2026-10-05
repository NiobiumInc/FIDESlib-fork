#ifndef API_OPENFHEENGINE_HPP
#define API_OPENFHEENGINE_HPP

#include "OpenFheVariant.hpp" // fideslib::LinkedOpenFheReducedNoise
#include "engine/Engine.hpp"

#include <string_view>

namespace fideslib {

/// @brief CPU backend. Most operations delegate to OpenFHE on the host; a few native ops
/// (convolution, accumulate) are reimplemented. All CPU operation code lives in OpenFheEngine.cpp.
/// Construction always succeeds regardless of reducedNoise: the linked OpenFHE fixes its FBC
/// variant at compile time, but only key-switching operations (ModUp/ModDown via
/// ApproxSwitchCRTBasis) are sensitive to it. Such a method refuses at call time instead, via
/// requireLinkedVariant(), when reducedNoise() disagrees with LinkedOpenFheReducedNoise().
class OpenFheEngine final : public Engine {
  public:
	explicit OpenFheEngine(bool reducedNoise);

	const char* name() const override {
		return "OpenFHE (CPU)";
	}

	Backend backend() const override {
		return Backend::CPU;
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
	/// @brief Variable-output-level bootstrap on the host backend: CORRECT, not cheaper.
	/// OpenFHE bootstraps to the top of the chain by construction — its CoeffsToSlots /
	/// SlotsToCoeffs plaintexts are precomputed once, at the levels a full-height raise
	/// meets them at (ckksrns-fhe.cpp, FHECKKSRNS::EvalBootstrapPrecompute), and
	/// EvalBootstrap has no argument that moves them — so the host path refreshes as usual
	/// and then spends the levels the caller does not want. Same ciphertext, same
	/// precision, same cost as EvalBootstrap plus the drops. Declared explicitly rather
	/// than inherited so the backend states what it does.
	Ciphertext<DCRTPoly>
	evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	/// `ext` is accepted and ignored — see the definition for why the CPU reference is unaffected.
	void linearTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool ext) override;
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
	void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) override;
	void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) override;
	void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;

	std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src) override;
	size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	double ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	void refreshHostShadow(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct) override;
	void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots) override;
	void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) override;

	bool isContextLoaded() const override;
	void synchronize() const override;
	void teardown() override;
	void setDevices(const std::vector<int>& devices) override;
	std::vector<int> devices() const override;

  private:
	/// @brief Throws std::runtime_error naming `op` if reducedNoise() disagrees with
	/// LinkedOpenFheReducedNoise(). Every method whose OpenFHE call can key-switch (ModUp/ModDown
	/// via ApproxSwitchCRTBasis) calls this first; see OpenFheEngine.cpp for the per-method evidence.
	void requireLinkedVariant(std::string_view op) const;
};

} // namespace fideslib

#endif
