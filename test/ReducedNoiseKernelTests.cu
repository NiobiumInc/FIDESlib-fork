// Exercises the CUDA ModUp/ModDown kernels directly against the reducedNoise flag: needs raw
// FIDESlib::CKKS GPU types (Context, Ciphertext, KeySwitchingKey), so it lives in its own .cu
// file rather than in test/ReducedNoiseTests.cpp.
#include "ParametrizedTest.cuh"

#include <gtest/gtest.h>
#include <openfhe.h>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LimbPartition.cuh" // FIDESlib::CKKS::PEER_ACCESS
#include "CKKS/Parameters.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "OpenFheVariant.hpp" // fideslib::LinkedOpenFheReducedNoise

namespace FIDESlib::Testing {

namespace {

// Restores PEER_ACCESS to its original value on scope exit, including when a test body unwinds
// early through an ASSERT_* failure.
class PeerAccessGuard {
  public:
	explicit PeerAccessGuard(bool value) : original_(FIDESlib::CKKS::PEER_ACCESS) {
		FIDESlib::CKKS::PEER_ACCESS = value;
	}

	~PeerAccessGuard() {
		FIDESlib::CKKS::PEER_ACCESS = original_;
	}

	PeerAccessGuard(const PeerAccessGuard&)			   = delete;
	PeerAccessGuard& operator=(const PeerAccessGuard&) = delete;

  private:
	bool original_;
};

// Replays the recorded pair through one GPU context per reducedNoise value and returns each
// variant's result, round-tripped back into an OpenFHE ciphertext via a clone of the shared
// template c3.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> RunVariant(bool reducedNoise,
  FIDESlib::CKKS::RawParams& raw_param,
  const FIDESlib::CKKS::RawCipherText& raw1,
  const FIDESlib::CKKS::RawCipherText& raw2,
  FIDESlib::CKKS::RawKeySwitchKey& rawKskEval,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& c3) {
	// adaptTo(raw) rebuilds every field (logN, L, dnum, primes...) from raw itself; only batch and
	// reducedNoise come from the object it is called on (Parameters.cu's adaptTo).
	FIDESlib::CKKS::Parameters base{ .batch = 100, .reducedNoise = reducedNoise };
	FIDESlib::CKKS::Context cc_gpu = FIDESlib::CKKS::GenCryptoContextGPU(base.adaptTo(raw_param), devices);

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_gpu);
	kskEval.Initialize(rawKskEval);
	cc_gpu->AddEvalKey(std::move(kskEval));

	FIDESlib::CKKS::Ciphertext GPUct1(cc_gpu, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_gpu, raw2);
	GPUct1.mult(GPUct2, false, true);

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	// Deep-copy the shared template, not a fresh Encrypt: lbcrypto::Ciphertext is a shared_ptr, so
	// re-encrypting per call would seed each variant from independent random noise, while Clone()'s
	// value-copy of m_elements makes the two starting points identical.
	auto cResGPU = c3->Clone();
	GetOpenFHECipherText(cResGPU, raw_res);
	return cResGPU;
}

// Runs the on/off comparison once under whichever PEER_ACCESS mode the caller has already set:
// false selects ModDown2/DecompAndModUpConv, true selects ModDown3/DecompAndModUpConv_spec2
// (LimbPartitionMGPU.cu's own dispatch).
void CheckFlagSelectsVariant(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle,
  FIDESlib::CKKS::RawParams& raw_param,
  const FIDESlib::CKKS::RawCipherText& raw1,
  const FIDESlib::CKKS::RawCipherText& raw2,
  FIDESlib::CKKS::RawKeySwitchKey& rawKskEval,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& c3) {
	auto resFalse = RunVariant(false, raw_param, raw1, raw2, rawKskEval, c3);
	auto resTrue  = RunVariant(true, raw_param, raw1, raw2, rawKskEval, c3);

	FIDESlib::CKKS::DeregisterAllContexts();

	// The two variants must disagree: a reduced-noise FBC correction that never fires would make
	// this pass trivially even with a broken kernel.
	bool anyCoefficientDiffers = false;
	for (size_t elem = 0; elem < 2 && !anyCoefficientDiffers; ++elem) {
		const auto& towersFalse = resFalse->GetElements().at(elem).GetAllElements();
		const auto& towersTrue	= resTrue->GetElements().at(elem).GetAllElements();
		for (size_t t = 0; t < towersFalse.size() && !anyCoefficientDiffers; ++t) {
			if (towersFalse.at(t).GetValues() != towersTrue.at(t).GetValues()) {
				anyCoefficientDiffers = true;
			}
		}
	}
	EXPECT_TRUE(anyCoefficientDiffers) << "reducedNoise=false and =true produced identical ciphertexts";

	// Whichever variant matches the linked OpenFHE must equal the oracle bit-for-bit; ASSERT_EQ_CIPHERTEXT
	// compares metadata plus every coefficient of every tower.
	auto& matching = fideslib::LinkedOpenFheReducedNoise() ? resTrue : resFalse;
	ASSERT_EQ_CIPHERTEXT(oracle, matching);
}

} // namespace

// One OpenFHE oracle, one recorded pair of ciphertexts and one eval-mult key, replayed through two
// FIDESlib GPU contexts built from the SAME RawParams and differing only in reducedNoise. A
// relinearizing multiply runs both ModUp (DecompAndModUpConv) and ModDown (ModDown2/3). The whole
// comparison runs once with PEER_ACCESS off and once with it on, covering both kernel pairs
// (ModDown2/DecompAndModUpConv and ModDown3/DecompAndModUpConv_spec2) in one test.
TEST(ReducedNoise, FlagSelectsVariant) {
	constexpr uint32_t kDepth	= 5;
	constexpr uint32_t kRingDim = 1u << 16;
	constexpr uint32_t kSlots	= 8;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(kDepth);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(kSlots);
	parameters.SetRingDim(kRingDim);
	parameters.SetScalingTechnique(lbcrypto::FIXEDAUTO);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);

	auto cc = lbcrypto::GenCryptoContext(parameters);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	const std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	const std::vector<double> x2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
	const std::vector<double> x3 = { 0.0 };
	auto pt1					 = cc->MakeCKKSPackedPlaintext(x1);
	auto pt2					 = cc->MakeCKKSPackedPlaintext(x2);
	auto pt3					 = cc->MakeCKKSPackedPlaintext(x3);
	auto c1						 = cc->Encrypt(keys.publicKey, pt1);
	auto c2						 = cc->Encrypt(keys.publicKey, pt2);
	// One template for the GPU->OpenFHE round trip, encrypted once so both variants start from
	// identical bytes; cloned per variant rather than re-encrypted (see RunVariant).
	auto c3 = cc->Encrypt(keys.publicKey, pt3);

	auto oracle = cc->EvalMult(c1, c2);

	FIDESlib::CKKS::RawParams raw_param		   = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::RawCipherText raw1		   = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2		   = FIDESlib::CKKS::GetRawCipherText(cc, c2);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

	for (const bool peerAccess : { false, true }) {
		SCOPED_TRACE(peerAccess ? "PEER_ACCESS=1 (ModDown3/DecompAndModUpConv_spec2)" : "PEER_ACCESS=0 (ModDown2/DecompAndModUpConv)");
		PeerAccessGuard guard(peerAccess);
		CheckFlagSelectsVariant(oracle, raw_param, raw1, raw2, rawKskEval, c3);
	}
}

} // namespace FIDESlib::Testing
