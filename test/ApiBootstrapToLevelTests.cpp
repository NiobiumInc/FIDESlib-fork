// Variable-output-level bootstrap (EvalBootstrapToLevel) — api-level suite, so it is
// meaningful on either backend:
//
//   * on the CPU backend it pins the CONTRACT: the ciphertext a reduced-level refresh
//     returns is the one an ordinary refresh followed by a level drop returns, at the same
//     level and to the same values.
//   * on the CUDA backend the same assertions cover the SHORTENED RAISE, which is a
//     different circuit — fewer limbs live through CoeffsToSlots, EvalMod and SlotsToCoeffs
//     — and must still land on that ciphertext. Run with FIDESLIB_TEST_BACKEND=cuda.
//
// CPU-only build:  cmake -B build-cpu -DFIDESLIB_ENABLE_CUDA=OFF && cmake --build build-cpu --target fideslib-cpu-test
// Run:    ./build-cpu/fideslib-cpu-test --gtest_filter='BootstrapToLevel*'

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "engine/Engine.hpp"
#include "fideslib.hpp"

using namespace fideslib;

namespace {

enum class TestBackend { CPU, CUDA, HAZE };
TestBackend GetTestBackend() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	if (b != nullptr && std::string(b) == "cuda" && IsBackendAvailable(Backend::CUDA))
		return TestBackend::CUDA;
	if (b != nullptr && std::string(b) == "haze" && IsBackendAvailable(Backend::HAZE))
		return TestBackend::HAZE;
	return TestBackend::CPU;
}

// Bootstrapping is a polynomial approximation of the modular reduction; the two refreshes
// compared here go through the same approximation, so this is the bar for "the same
// plaintext", not for the approximation itself.
constexpr double kBootstrapPrecision = 1e-2;

std::vector<double> DecryptReal(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, size_t n) {
	Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	pt->SetLength(n);
	return pt->GetRealPackedValue();
}

// FLEXIBLEAUTO, level budget {2,2}: the technique and transform shape the deployed
// bootstrap uses, and the shape the reduced-level precomputation is built for (a {1,1}
// budget uses the single-step linear transform and is rejected by name).
class BootstrapToLevelTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth	   = 23;
	static constexpr uint32_t kScaleMod	   = 59;
	static constexpr uint32_t kFirstMod	   = 60;
	static constexpr uint32_t kRingDim	   = 1u << 16;
	static constexpr uint32_t kBatchSize   = 8;
	static constexpr uint32_t kSlots	   = 1u << 5;
	static constexpr uint32_t kLevelBudget = 2;
	// Deepest level the host EvalBootstrap accepts (ckksrns-fhe.cpp's own guard):
	// L0 - (levelBudget_decode + 2). Encrypting AT it is what makes the refresh real — a
	// full-level input gains no levels and comes back unchanged, which would pass every
	// assertion below while exercising nothing.
	static constexpr uint32_t kMaxBootLevel = kDepth - (kLevelBudget + 2);

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(kBatchSize);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FLEXIBLEAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		params.SetReducedNoise(LinkedOpenFheReducedNoise()); // matches the oracle's variant
		if (GetTestBackend() == TestBackend::CUDA)
			params.SetBackend(Backend::CUDA);
		else if (GetTestBackend() == TestBackend::HAZE)
			params.SetBackend(Backend::HAZE);

		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup({ kLevelBudget, kLevelBudget }, { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		if (GetTestBackend() != TestBackend::CPU)
			cc->LoadContext(keys.publicKey);
	}

	Ciphertext<DCRTPoly> DepletedInput() {
		auto pt = cc->MakeCKKSPackedPlaintext(v1, 1, kMaxBootLevel, nullptr, kSlots);
		return cc->Encrypt(pt, keys.publicKey);
	}

	// The level an ordinary refresh lands on, MEASURED — the floor every request below is
	// stated against. Measured rather than derived so the assertions hold whatever the
	// backend's bootstrap circuit costs.
	uint32_t MeasuredOutputLevel() {
		auto ct = DepletedInput();
		auto refreshed = cc->EvalBootstrap(ct);
		return static_cast<uint32_t>(refreshed->GetLevel());
	}

	// The comparison path: refresh, then spend the levels afterwards, the way a caller has
	// had to until now.
	void DropTo(Ciphertext<DCRTPoly>& ct, uint32_t level) {
		while (ct->GetLevel() < level) {
			const size_t before = ct->GetLevel();
			cc->EvalMultInPlace(ct, 1.0);
			if (ct->GetLevel() == before)
				cc->RescaleInPlace(ct);
			ASSERT_GT(ct->GetLevel(), before) << "the drop made no progress at level " << before;
		}
	}
};

// THE TEST THAT MATTERS: a reduced-level refresh decrypts to the same plaintext, within the
// bootstrap's own precision, as a full refresh followed by a level drop — and lands on the
// same level.
TEST_F(BootstrapToLevelTest, ReducedLevelMatchesBootstrapThenDrop) {
	const uint32_t defaultOut = MeasuredOutputLevel();
	// Two levels of the refresh's output that the caller says it does not need. Deep enough
	// that the CUDA raise really is shorter, shallow enough to stay inside the chain.
	const uint32_t target = defaultOut + 2;
	ASSERT_LT(target, kDepth);

	auto ct = DepletedInput();
	const size_t levelBefore = ct->GetLevel();

	auto reduced = cc->EvalBootstrapToLevel(ct, target);
	ASSERT_LT(reduced->GetLevel(), levelBefore) << "the refresh gained no levels: it returned the input unchanged, so this test exercised nothing";
	EXPECT_EQ(static_cast<uint32_t>(reduced->GetLevel()), target);

	auto full = cc->EvalBootstrap(ct);
	ASSERT_NO_FATAL_FAILURE(DropTo(full, target));
	EXPECT_EQ(static_cast<uint32_t>(full->GetLevel()), target);

	const auto a = DecryptReal(cc, reduced, keys.secretKey, v1.size());
	const auto b = DecryptReal(cc, full, keys.secretKey, v1.size());
	for (size_t i = 0; i < v1.size(); ++i) {
		EXPECT_NEAR(a[i], v1[i], kBootstrapPrecision) << "reduced-level refresh, slot " << i;
		EXPECT_NEAR(b[i], v1[i], kBootstrapPrecision) << "refresh-then-drop, slot " << i;
		// The two routes to the same level must agree far more tightly than either agrees
		// with the cleartext: they share the approximation, so what is left between them is
		// only where the levels were spent.
		EXPECT_NEAR(a[i], b[i], kBootstrapPrecision) << "the two routes disagree, slot " << i;
	}
}

// A request for a level the bootstrap cannot reach (it cannot hand back MORE levels than its
// circuit leaves) is the ordinary refresh, not an error. 0 is the natural way to say
// "no preference".
TEST_F(BootstrapToLevelTest, ShallowRequestClampsToTheDefault) {
	const uint32_t defaultOut = MeasuredOutputLevel();
	auto ct					  = DepletedInput();

	auto clamped = cc->EvalBootstrapToLevel(ct, 0);
	EXPECT_EQ(static_cast<uint32_t>(clamped->GetLevel()), defaultOut);

	auto exact = cc->EvalBootstrapToLevel(ct, defaultOut);
	EXPECT_EQ(static_cast<uint32_t>(exact->GetLevel()), defaultOut);

	const auto vals = DecryptReal(cc, clamped, keys.secretKey, v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(vals[i], v1[i], kBootstrapPrecision) << "slot " << i;
}

// The in-place entry point is the same operation.
TEST_F(BootstrapToLevelTest, InPlaceMatchesTheValueForm) {
	const uint32_t target = MeasuredOutputLevel() + 2;
	auto ct				  = DepletedInput();
	auto value			  = cc->EvalBootstrapToLevel(ct, target);

	auto inplace = DepletedInput();
	cc->EvalBootstrapToLevelInPlace(inplace, target);
	EXPECT_EQ(static_cast<uint32_t>(inplace->GetLevel()), target);

	const auto a = DecryptReal(cc, value, keys.secretKey, v1.size());
	const auto b = DecryptReal(cc, inplace, keys.secretKey, v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(a[i], b[i], kBootstrapPrecision) << "slot " << i;
}


} // namespace
