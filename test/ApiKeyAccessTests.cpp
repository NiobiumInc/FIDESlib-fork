// Key-access test suite for fideslib's static eval-key map accessors (GetEvalMultKeyVector,
// GetEvalAutomorphismKeyMap, FindAutomorphismIndex). Deliberately separate from ApiTests.cpp:
// these tests need lbcrypto types directly (EvalKey, the PublicKey pimpl, address identity
// against the lbcrypto static maps), so ApiTests.cpp itself can stay lbcrypto-free.
// CPU-only build:  cmake -B build-cpu -DFIDESLIB_ENABLE_CUDA=OFF && cmake --build build-cpu --target fideslib-cpu-test
// Run:    ./build-cpu/fideslib-cpu-test --gtest_filter='KeyAccessTest.*:NoEvalMultKeyTest.*'

#include <any>
#include <cstdlib>
#include <gtest/gtest.h>
#include <sstream>
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

// ---------------------------------------------------------------------------
// Context lifetime. OpenFHE keeps every evaluation key in process-wide maps, so the facade has
// to say when a context's keys go away: when the last facade context over the same OpenFHE
// context is destroyed, and not when another context is created or destroyed meanwhile.
// ---------------------------------------------------------------------------
namespace {

struct LiveContext {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
};

// A context with its multiplication and rotation keys. A different depth gives a different
// OpenFHE context; the same depth gives the same one, which OpenFHE hands out again.
// `onTestBackend` false builds the context on the CPU engine whatever backend the suite runs on:
// the haze backend loads one context per process, so a second context next to a loaded one has
// to be a CPU one. That is also the case this is about, a CPU reference next to a device context.
LiveContext MakeLiveContext(uint32_t depth, bool onTestBackend = true) {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(depth);
	params.SetScalingModSize(50);
	params.SetBatchSize(8);
	params.SetRingDim(1u << 13);
	params.SetSecurityLevel(HEStd_NotSet);
	params.SetScalingTechnique(FIXEDAUTO);
	if (onTestBackend)
		ConfigureTestEngine(params);
	else
		params.SetReducedNoise(LinkedOpenFheReducedNoise());
	LiveContext c;
	c.cc = GenCryptoContext(params);
	c.cc->Enable(PKE);
	c.cc->Enable(KEYSWITCH);
	c.cc->Enable(LEVELEDSHE);
	c.keys = c.cc->KeyGen();
	c.cc->EvalMultKeyGen(c.keys.secretKey);
	c.cc->EvalRotateKeyGen(c.keys.secretKey, { 1, 2 });
	if (onTestBackend && TestUseDevice())
		c.cc->LoadContext(c.keys.publicKey);
	return c;
}

// The context still multiplies (relinearization key) and rotates (rotation key).
void ExpectKeysWork(LiveContext& c) {
	const std::vector<double> v = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	auto ct	  = EncryptVec(c.cc, v, c.keys.publicKey);
	auto sq	  = c.cc->EvalMult(ct, ct);
	auto rot	  = c.cc->EvalRotate(ct, 1);
	// The haze backend needs a result declared as an output before it is decrypted; elsewhere this does nothing.
	c.cc->MarkOutput(sq);
	c.cc->MarkOutput(rot);
	Plaintext sqPt, rotPt;
	c.cc->Decrypt(sq, c.keys.secretKey, &sqPt);
	c.cc->Decrypt(rot, c.keys.secretKey, &rotPt);
	sqPt->SetLength(8);
	rotPt->SetLength(8);
	const auto sqv	= sqPt->GetRealPackedValue();
	const auto rotv = rotPt->GetRealPackedValue();
	for (size_t i = 0; i < v.size(); ++i) {
		EXPECT_NEAR(sqv[i], v[i] * v[i], 1e-3) << "slot " << i;
		EXPECT_NEAR(rotv[i], v[(i + 1) % v.size()], 1e-3) << "slot " << i;
	}
}

} // namespace

TEST(ContextLifetimeTest, CreatingAContextWithOtherParametersKeepsTheFirstContextsKeys) {
	auto a = MakeLiveContext(3);
	auto b = MakeLiveContext(4, false);
	ExpectKeysWork(a);
	ExpectKeysWork(b);
}

TEST(ContextLifetimeTest, CreatingAContextWithTheSameParametersKeepsTheFirstContextsKeys) {
	auto a = MakeLiveContext(3);
	auto b = MakeLiveContext(3, false);
	ExpectKeysWork(a);
	ExpectKeysWork(b);
}

TEST(ContextLifetimeTest, DestroyingAContextKeepsTheKeysOfAnotherOverTheSameParameters) {
	auto a = MakeLiveContext(3);
	{
		auto b = MakeLiveContext(3, false);
		ExpectKeysWork(b);
	}
	ExpectKeysWork(a);
}

TEST(ContextLifetimeTest, DestroyingAContextKeepsTheKeysOfAnotherOverOtherParameters) {
	auto a = MakeLiveContext(3);
	{
		auto b = MakeLiveContext(4, false);
		ExpectKeysWork(b);
	}
	ExpectKeysWork(a);
}

TEST(ContextLifetimeTest, DestroyingTheLastContextClearsItsKeys) {
	std::string tag;
	{
		auto a = MakeLiveContext(3);
		tag	= std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(a.keys.publicKey->pimpl)->GetKeyTag();
		ASSERT_FALSE(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag).empty());
		ASSERT_NO_THROW(CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag));
	}
	EXPECT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag), std::exception);
	EXPECT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag), std::exception);
}

// The serialized multiplication key goes into a second context once the key is gone from this
// process, as it would be in a fresh one. (OpenFHE refuses to insert a vector for a tag that
// already has one.)
TEST(ContextLifetimeTest, SerializedEvalMultKeyDeserializesIntoAnotherContext) {
	auto a = MakeLiveContext(3);
	const auto tag = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(a.keys.publicKey->pimpl)->GetKeyTag();
	std::ostringstream oss;
	ASSERT_TRUE(CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(oss, SerType::BINARY));
	EXPECT_GT(oss.str().size(), 0u);

	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys(tag);
	ASSERT_THROW(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag), std::exception);

	auto b = MakeLiveContext(3, false);
	std::istringstream iss(oss.str());
	ASSERT_TRUE(b.cc->DeserializeEvalMultKey(iss, SerType::BINARY));
	EXPECT_FALSE(CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag).empty());
	// The original context multiplies again with the deserialized key.
	ExpectKeysWork(a);
}

// A bootstrap-ready context keeps working when another context with the same parameters is created.
// The second context is a CPU context: the haze backend loads one context per process, and a CPU
// reference next to a device context is the case this is about.
TEST(ContextLifetimeTest, BootstrapSurvivesASecondContextWithTheSameParameters) {
	auto make = [](KeyPair<DCRTPoly>& keys, bool bootstrap, bool onTestBackend) {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(23);
		params.SetScalingModSize(59);
		params.SetFirstModSize(60);
		params.SetBatchSize(8);
		params.SetRingDim(1u << 16);
		params.SetScalingTechnique(FLEXIBLEAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		if (onTestBackend)
			ConfigureTestEngine(params);
		else
			params.SetReducedNoise(LinkedOpenFheReducedNoise());
		auto cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		if (bootstrap) {
			cc->EvalBootstrapSetup({ 2, 2 }, { 0, 0 }, 32, 0);
			cc->EvalBootstrapKeyGen(keys.secretKey, 32);
		}
		if (onTestBackend && TestUseDevice())
			cc->LoadContext(keys.publicKey);
		return cc;
	};
	KeyPair<DCRTPoly> ka, kb;
	auto a = make(ka, true, true);
	auto b = make(kb, false, false); // same parameters: the same OpenFHE context as `a`
	auto pt = a->MakeCKKSPackedPlaintext(std::vector<double>{ 0.1, 0.2, 0.3, 0.4 }, 1, 21, nullptr, 32);
	auto ct = a->Encrypt(pt, ka.publicKey);
	Ciphertext<DCRTPoly> out;
	ASSERT_NO_THROW(out = a->EvalBootstrap(ct));
	a->MarkOutput(out);
	Plaintext res;
	a->Decrypt(out, ka.secretKey, &res);
	res->SetLength(4);
	const auto v = res->GetRealPackedValue();
	EXPECT_NEAR(v[0], 0.1, 1e-2);
	EXPECT_NEAR(v[3], 0.4, 1e-2);
}
