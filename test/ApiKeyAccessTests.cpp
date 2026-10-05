// Key-access test suite for fideslib's static eval-key map accessors (GetEvalMultKeyVector,
// GetEvalAutomorphismKeyMap, FindAutomorphismIndex). Deliberately separate from ApiTests.cpp:
// these tests need lbcrypto types directly (EvalKey, the PublicKey pimpl, address identity
// against the lbcrypto static maps), so ApiTests.cpp itself can stay lbcrypto-free.
// CPU-only build:  cmake -B build-cpu -DFIDESLIB_ENABLE_CUDA=OFF && cmake --build build-cpu --target fideslib-cpu-test
// Run:    ./build-cpu/fideslib-cpu-test --gtest_filter='KeyAccessTest.*:NoEvalMultKeyTest.*'

#include <any>
#include <cstdlib>
#include <gtest/gtest.h>
#include <openfhe.h>
#include <string>
#include <vector>

#include "TestEngineConfig.hpp" // FIDESlib::Testing::{ConfigureTestEngine, GetTestBackend, TestBackend}
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;
using FIDESlib::Testing::GetTestBackend;
using FIDESlib::Testing::TestBackend;

static bool TestUseDevice() {
	return GetTestBackend() != TestBackend::CPU;
}

namespace {
// Encrypt() takes a Plaintext& lvalue, so bind the encoded plaintext first (mirrors
// ApiTests.cpp's helper of the same name).
Ciphertext<DCRTPoly> EncryptVec(CryptoContext<DCRTPoly>& cc, const std::vector<double>& v, const PublicKey<DCRTPoly>& pk) {
	auto pt = cc->MakeCKKSPackedPlaintext(v);
	return cc->Encrypt(pt, pk);
}
} // namespace

// ---------------------------------------------------------------------------
// Fixture with a generated EvalMultKey + rotation keys. One context per test: the facade's
// destructor clears the process-global eval-key maps, so tests cannot see each other's keys.
// ---------------------------------------------------------------------------
class KeyAccessTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth	= 5;
	static constexpr uint32_t kRingDim = 1u << 16; // 65536
	static constexpr uint32_t kSlots	= 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		ConfigureTestEngine(params);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2 });
		if (TestUseDevice())
			cc->LoadContext(keys.publicKey);
	}
};

// ---------------------------------------------------------------------------
// Fixture with NO EvalMultKeyGen: the relin key map is empty for this context's tag, so both
// the key accessors and Relinearize must fail cleanly (throw) rather than crash. Safe on device
// backends too — both engines guard the relin-key upload on a map lookup.
// ---------------------------------------------------------------------------
class NoEvalMultKeyTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth	= 5;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots	= 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		ConfigureTestEngine(params);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		keys = cc->KeyGen();
		// Deliberately no EvalMultKeyGen: exercises the "no key generated" path.
		if (TestUseDevice())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- Key accessors (KeyAccessTest: EvalMultKeyGen + EvalRotateKeyGen already ran) ----

TEST_F(KeyAccessTest, GetEvalMultKeyVectorForTag) {
	const auto& tag = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(keys.publicKey->pimpl)->GetKeyTag();
	const auto& vec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag);
	ASSERT_FALSE(vec.empty());
	EXPECT_EQ(vec.size(), 1u);
	EXPECT_EQ(vec[0]->GetKeyTag(), tag);
}

TEST_F(KeyAccessTest, GetEvalMultKeyVectorIsThePassthrough) {
	const auto& tag = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(keys.publicKey->pimpl)->GetKeyTag();
	const auto& vec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag);
	const auto& lb	 = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalMultKeyVector(tag);
	// Address identity: our accessor must return the lbcrypto static map's own storage, not a copy.
	EXPECT_EQ(&vec[0], &lb[0]);
}

TEST_F(KeyAccessTest, GetEvalAutomorphismKeyMapContainsGeneratedSteps) {
	const auto& tag					= std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(keys.publicKey->pimpl)->GetKeyTag();
	auto& map							= CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag);
	const std::vector<int32_t> steps	= { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2 };
	for (int32_t step : steps) {
		// FindAutomorphismIndex takes uint32_t; negative steps round-trip through the same
		// two's-complement bit pattern EvalRotateKeyGen used internally to generate the key,
		// so the cast below is exact, not lossy.
		uint32_t idx = cc->FindAutomorphismIndex(static_cast<uint32_t>(step));
		EXPECT_EQ(map.count(idx), 1u) << "step " << step;
	}
	EXPECT_GE(map.size(), 10u);
}

TEST_F(KeyAccessTest, KeyAccessorsUnknownTagThrow) {
	EXPECT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector("no-such-tag"), std::exception);
	EXPECT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap("no-such-tag"), std::exception);
}

// ---- Key accessors (NoEvalMultKeyTest: no EvalMultKeyGen ran) ----

TEST_F(NoEvalMultKeyTest, GetEvalMultKeyVectorBeforeKeyGenThrows) {
	const auto& tag = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(keys.publicKey->pimpl)->GetKeyTag();
	EXPECT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag), std::exception);
}

TEST_F(NoEvalMultKeyTest, RelinearizeWithoutEvalMultKeyThrows) {
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);
	Ciphertext<DCRTPoly> d;
	// EvalMultNoRelin needs no relin key at all (it is a pure tensor product).
	EXPECT_NO_THROW(d = cc->EvalMultNoRelin(a, b));
	// Relinearize does need the key; none was generated for this context.
	EXPECT_THROW(cc->RelinearizeInPlace(d), std::exception);
}

TEST_F(NoEvalMultKeyTest, RelinearizeInPlaceDegree1WithoutEvalMultKeyThrows) {
	// OpenFHE's RelinearizeInPlace calls GetEvalMultKeyVector(tag) BEFORE checking the degree
	// (cryptocontext.h), so even a degree-1 (already-relinearized) ciphertext must throw when no
	// relin key was generated — it is not a silent no-op. Pins the fix that hoists the
	// haveRelinKey_ check above the degree branch in HazeEngine::relinearizeCore.
	auto a = EncryptVec(cc, v1, keys.publicKey);
	EXPECT_THROW(cc->RelinearizeInPlace(a), std::exception);
}
