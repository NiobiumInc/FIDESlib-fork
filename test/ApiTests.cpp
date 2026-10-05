// API test suite for fideslib — drives the public CryptoContextImpl api and verifies
// results by decryption against expected values, so it is meaningful on either backend.
// Default backend is CPU; set FIDESLIB_TEST_BACKEND=cuda on a CUDA build to exercise the
// CUDA engine end-to-end (see TestUseCuda below).
// CPU-only build:  cmake -B build-cpu -DFIDESLIB_ENABLE_CUDA=OFF && cmake --build build-cpu --target fideslib-cpu-test
// Run:    ./build-cpu/fideslib-cpu-test [--gtest_filter=-*Bootstrap*]

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <openfhe.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <random>
#include <sstream>
#include <vector>

#include "Serialize.hpp"
#include "engine/Engine.hpp"		 // Engine::backend(), for the hard backend guard
#include "engine/EngineCommon.hpp" // bootstrapModEvalLevels
#include "engine/cuda/CudaEngine.hpp" // setDropInputHostAfterUpload; empty header when !FIDESLIB_ENABLE_CUDA
#include "engine/haze/HazeEngine.hpp" // setDropInputHostAfterUpload; empty header when !FIDESLIB_ENABLE_HAZE
#include "fideslib.hpp"

using namespace fideslib;

// The api test fixtures run against any backend: set FIDESLIB_TEST_BACKEND=cuda (on a
// CUDA build) or =haze (on a haze build) to exercise that engine end-to-end. Builds
// without the requested backend fall back to CPU, since IsBackendAvailable is false
// when it is not compiled in.
enum class TestBackend { CPU, CUDA, HAZE };
static TestBackend GetTestBackend() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	if (b != nullptr && std::string(b) == "cuda" && IsBackendAvailable(Backend::CUDA))
		return TestBackend::CUDA;
	if (b != nullptr && std::string(b) == "haze" && IsBackendAvailable(Backend::HAZE))
		return TestBackend::HAZE;
	return TestBackend::CPU;
}
static bool TestUseCuda() {
	return GetTestBackend() == TestBackend::CUDA;
}
// Any device backend (CUDA or haze): gates SetBackend + LoadContext in the fixtures.
static bool TestUseDevice() {
	return GetTestBackend() != TestBackend::CPU;
}

// Precision tolerance for standard CKKS operations (encrypt, add, mult, rotate).
// With scalingModSize=50, theoretical single-op precision is ~2^{-50} ~ 1e-15.
// Accumulated error through depth-5 circuits and encoding roundtrip gives ~1e-6.
static constexpr double CKKS_PRECISION = 1e-6;

// Bootstrapping introduces polynomial approximation error for modular reduction.
// OpenFHE's iterative bootstrap with levelBudget={1,1} achieves ~10-14 bits of
// precision; 1e-2 (~6.6 bits) provides margin for the single-iteration case.
static constexpr double CKKS_BOOTSTRAP_PRECISION = 1e-2;

// Decrypt ct into *pt and check real slots against expected within tol.
static void CheckPrecision(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, const std::vector<double>& expected, double tol) {
	Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	pt->SetLength(expected.size());
	auto vals = pt->GetRealPackedValue();
	for (size_t i = 0; i < expected.size(); ++i)
		EXPECT_NEAR(vals[i], expected[i], tol) << "slot " << i;
}

// ---------------------------------------------------------------------------
// Standard fixture: depth=5, ringDim=65536, slots=8, FIXEDAUTO, UNIFORM_TERNARY
// ---------------------------------------------------------------------------

class CKKSTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 5;
	static constexpr uint32_t kRingDim = 1u << 16; // 65536
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	// Test vectors (8 slots).
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		else if (GetTestBackend() == TestBackend::HAZE) {
			params.SetBackend(Backend::HAZE);
			params.SetReducedNoise(true); // parity asserts vs a WITH_REDUCED_NOISE OpenFHE oracle
		}
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		// Generate every rotation index the tests use up front. In device mode the keys are
		// uploaded by LoadContext, so they must all exist before it; the convolution and
		// accumulate ops need the full 1..8 range (incl. 3,5,6,7), not just powers of two.
		// -4 and -8 complete the NEGATIVE half of the +-2^i doubling ladder, which is the index
		// set EvalRotateMany is exercised over (head_bcast walks it downward, head_sum upward).
		cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -4, -8 });
		if (TestUseDevice())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 1. Context ----

TEST_F(CKKSTest, ContextSetup) {
	EXPECT_EQ(cc->GetRingDimension(), kRingDim);
	EXPECT_EQ(cc->GetCyclotomicOrder(), 2 * kRingDim);
}

// ---- 2. Encoding ----

TEST_F(CKKSTest, EncodingReal) {
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto out = pt->GetRealPackedValue();
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(out[i], v1[i], CKKS_PRECISION) << "slot " << i;
}

TEST_F(CKKSTest, EncodingComplex) {
	std::vector<std::complex<double>> cv(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		cv[i] = { v1[i], 0.0 };
	auto pt	 = cc->MakeCKKSPackedPlaintext(cv);
	auto out = pt->GetCKKSPackedValue();
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(out[i].real(), v1[i], CKKS_PRECISION) << "slot " << i;
}

// ---- 3. Encrypt / Decrypt ----

TEST_F(CKKSTest, EncryptDecryptPublicKey) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_PRECISION);
}

TEST_F(CKKSTest, EncryptDecryptPrivateKey) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.secretKey);
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_PRECISION);
}

// ---- 4. EvalAdd ----

TEST_F(CKKSTest, EvalAddCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalAdd(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalAdd(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddCtScalar) {
	double scalar = 10.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalAdd(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalAddInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	cc->EvalAddInPlace(ct1, pt2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtScalar) {
	double scalar = 5.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalAddInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMany) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i] + v1[i];
	auto pt1							  = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2							  = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1							  = cc->Encrypt(pt1, keys.publicKey);
	auto ct2							  = cc->Encrypt(pt2, keys.publicKey);
	auto ct3							  = cc->Encrypt(pt1, keys.publicKey);
	std::vector<Ciphertext<DCRTPoly>> cts = { ct1, ct2, ct3 };
	auto res							  = cc->EvalAddMany(cts);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalSub ----

TEST_F(CKKSTest, EvalSubCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalSub(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalSub(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubCtScalar) {
	double scalar = 0.5;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSub(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalSubInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubInPlaceCtScalar) {
	double scalar = 3.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalSubInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalMult ----

TEST_F(CKKSTest, EvalMultCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMult(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalMult(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultCtScalar) {
	double scalar = 2.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalMult(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultInPlaceCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	cc->EvalMultInPlace(ct1, pt2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultInPlaceCtScalar) {
	double scalar = 3.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalMultInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalNegate / EvalSquare ----

TEST_F(CKKSTest, EvalNegate) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = -v1[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalNegate(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquare) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSquare(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquareInPlace) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt = cc->MakeCKKSPackedPlaintext(v2);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalSquareInPlace(ct);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 5. Rotation ----

TEST_F(CKKSTest, EvalRotate) {
	// Rotate left by 1: slot i gets v1[(i+1) % slots] (circular).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalRotate(ct, 1);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalFastRotation) {
	// Same rotation as EvalRotate but via precompute + EvalFastRotation.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt		 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct		 = cc->Encrypt(pt, keys.publicKey);
	auto precomp = cc->EvalFastRotationPrecompute(ct);
	uint32_t m	 = cc->GetCyclotomicOrder();
	auto res	 = cc->EvalFastRotation(ct, 1, m, precomp);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 5b. EvalRotateMany ----
//
// The contract is byte-equality with EvalRotate, not closeness, so these compare RNS limbs and
// not decrypted slots: a batched key switch that produced a differently-rounded-but-valid answer
// would pass a precision check and still break the "numerics identical to the serial path"
// guarantee the application's server arm is built on. Run on whatever backend the suite is
// pointed at: on CPU it pins the reference loop and the facade's validation, on CUDA it is the
// real assertion about the batched kernel.

// Sync both ciphertexts back from the device and compare every RNS limb of every component, plus
// the scaling metadata. `where` names the case in the failure message.
static void ExpectCiphertextBitEqual(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& got, Ciphertext<DCRTPoly>& want, const std::string& where) {
	cc->RecoverHostCiphertext(got);
	cc->RecoverHostCiphertext(want);
	auto g = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(got->host);
	auto w = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(want->host);

	ASSERT_EQ(g->GetLevel(), w->GetLevel()) << where;
	ASSERT_EQ(g->GetNoiseScaleDeg(), w->GetNoiseScaleDeg()) << where;
	ASSERT_EQ(g->GetScalingFactor(), w->GetScalingFactor()) << where;
	ASSERT_EQ(g->GetElements().size(), w->GetElements().size()) << where;
	for (size_t j = 0; j < g->GetElements().size(); ++j) {
		ASSERT_EQ(g->GetElements().at(j).GetAllElements().size(), w->GetElements().at(j).GetAllElements().size()) << where << " component " << j;
		for (size_t i = 0; i < g->GetElements().at(j).GetAllElements().size(); ++i)
			ASSERT_EQ(g->GetElements().at(j).GetAllElements().at(i).GetValues(), w->GetElements().at(j).GetAllElements().at(i).GetValues()) << where << " component " << j << " tower " << i;
	}
}

TEST_F(CKKSTest, EvalRotateManyMatchesEvalRotate) {
	// The head ladder's +-2^i, which is the index set the score path actually walks, over the batch
	// widths that bracket a page's live-body count: 1 (degenerate), 2, 7 (not a power of two, and
	// not a divisor of the engine's batch cap), 32 (a full page of diagonals).
	const std::vector<int32_t> indices = { 1, 2, 4, 8, -1, -2, -4, -8 };
	for (uint32_t B : { 1u, 2u, 7u, 32u }) {
		// Distinct contents per entry: a batch that silently rotated one input B times, or emitted
		// its outputs in the wrong order, would pass with identical inputs.
		std::vector<Ciphertext<DCRTPoly>> cts;
		cts.reserve(B);
		for (uint32_t b = 0; b < B; ++b) {
			std::vector<double> v(kSlots);
			for (uint32_t i = 0; i < kSlots; ++i)
				v[i] = static_cast<double>(b + 1) + static_cast<double>(i) / 16.0;
			auto pt = cc->MakeCKKSPackedPlaintext(v);
			cts.push_back(cc->Encrypt(pt, keys.publicKey));
		}
		for (int32_t idx : indices) {
			auto many = cc->EvalRotateMany(cts, idx);
			ASSERT_EQ(many.size(), cts.size()) << "B=" << B << " index=" << idx;
			for (uint32_t b = 0; b < B; ++b) {
				auto one = cc->EvalRotate(cts[b], idx);
				ASSERT_NO_FATAL_FAILURE(ExpectCiphertextBitEqual(cc, many[b], one, "B=" + std::to_string(B) + " index=" + std::to_string(idx) + " entry=" + std::to_string(b)));
			}
		}
	}
}

TEST_F(CKKSTest, EvalRotateManyEmptyBatch) {
	std::vector<Ciphertext<DCRTPoly>> none;
	EXPECT_TRUE(cc->EvalRotateMany(none, 1).empty());
}

TEST_F(CKKSTest, EvalRotateManyRejectsNullEntry) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	std::vector<Ciphertext<DCRTPoly>> cts{ cc->Encrypt(pt, keys.publicKey), nullptr };
	EXPECT_THROW(cc->EvalRotateMany(cts, 1), lbcrypto::OpenFHEException);
}

// Mixed levels are the fallback case, not an error: the api contract is unconditional, so the
// engine is required to answer correctly even when its batched kernel cannot take the batch.
TEST_F(CKKSTest, EvalRotateManyMixedLevels) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto a	= cc->Encrypt(pt, keys.publicKey);
	auto b	= cc->Encrypt(pt, keys.publicKey);
	cc->RescaleInPlace(b); // b is now one level deeper than a
	std::vector<Ciphertext<DCRTPoly>> cts{ a, b };
	auto many = cc->EvalRotateMany(cts, 1);
	ASSERT_EQ(many.size(), 2u);
	auto ra = cc->EvalRotate(a, 1);
	auto rb = cc->EvalRotate(b, 1);
	ASSERT_NO_FATAL_FAILURE(ExpectCiphertextBitEqual(cc, many[0], ra, "mixed-levels entry=0"));
	ASSERT_NO_FATAL_FAILURE(ExpectCiphertextBitEqual(cc, many[1], rb, "mixed-levels entry=1"));
}

// ---- 6. Level management ----

TEST_F(CKKSTest, RescaleInPlace) {
	// Verify that the ciphertext remains decryptable and accurate after rescaling.
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto ct2 = cc->EvalMult(ct, ct); // ct-ct mult: scale^2
	cc->RescaleInPlace(ct2);
	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[i] * v1[i];
	CheckPrecision(cc, ct2, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, SetLevel) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	// Consume one level so SetLevel has somewhere to go.
	cc->RescaleInPlace(ct);
	size_t cur = ct->GetLevel();
	// SetLevel drops towers (increases level number toward max).
	CryptoContextImpl<DCRTPoly>::SetLevel(ct, cur + 1);
	EXPECT_EQ(ct->GetLevel(), cur + 1);
}

// ---- 7. Advanced operations ----

TEST_F(CKKSTest, EvalChebyshevSeries) {
	// The GPU Chebyshev domain-map fix is deliberately off this branch (lives on
	// chebyshev-domain-map-fix); the device path is off by ~1 on non-[-1,1] intervals.
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	// Approximate the identity f(x)=x on [0,1] with degree-5 Chebyshev.
	std::function<double(double)> f = [](double x) { return x; };
	auto coeffs						= CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(f, 0.0, 1.0, 5);
	// Inputs must lie inside [0,1].
	std::vector<double> input(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		input[i] = static_cast<double>(i + 1) / (kSlots + 1);
	auto pt	 = cc->MakeCKKSPackedPlaintext(input);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalChebyshevSeries(ct, coeffs, 0.0, 1.0);
	// Degree-5 approximation of identity should be accurate to ~1e-4.
	CheckPrecision(cc, res, keys.secretKey, input, 1e-4);
}

// Regression test: HazeEngine::hazeEvalChebyshevSeriesImpl silently omitted the
// FIXEDMANUAL-only manual rescale/level-drop branches present in the CUDA/OpenFHE
// reference (src/CKKS/ApproxModEval.cu), so under FIXEDMANUAL the Haze backend decrypted
// to a different result than the CPU/OpenFHE oracle for the same series. Not a CKKSTest
// (TEST_F) case: the fixture's SetUp already loads a device context, and haze context
// state is process-global (only one may be loaded at a time), so both contexts here are
// built and torn down explicitly, oracle first.
TEST(ChebyshevFixedManual, HazeMatchesOpenFheOracle) {
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	if (GetTestBackend() != TestBackend::HAZE)
		GTEST_SKIP() << "requires FIDESLIB_TEST_BACKEND=haze";

	constexpr uint32_t kDepth	 = 5;
	constexpr uint32_t kRingDim = 1u << 16;
	constexpr uint32_t kSlots	 = 8;

	// Approximate the identity f(x)=x on [0,1] with degree-5 Chebyshev (k>=2, so
	// Paterson-Stockmeyer's power-basis/T2 chain and inner-PS recursion are exercised).
	std::function<double(double)> f = [](double x) { return x; };
	auto coeffs						 = CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(f, 0.0, 1.0, 5);
	std::vector<double> input(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		input[i] = static_cast<double>(i + 1) / (kSlots + 1);

	auto setCommonParams = [&](CCParams<CryptoContextCKKSRNS>& params) {
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDMANUAL);
	};

	// --- OpenFHE CPU oracle ---
	std::vector<double> cpuVals;
	{
		CCParams<CryptoContextCKKSRNS> params;
		setCommonParams(params);
		auto ccCpu = GenCryptoContext(params);
		ccCpu->Enable(PKE);
		ccCpu->Enable(KEYSWITCH);
		ccCpu->Enable(LEVELEDSHE);
		ccCpu->Enable(ADVANCEDSHE);
		auto keysCpu = ccCpu->KeyGen();
		ccCpu->EvalMultKeyGen(keysCpu.secretKey);
		auto pt	 = ccCpu->MakeCKKSPackedPlaintext(input);
		auto ct	 = ccCpu->Encrypt(pt, keysCpu.publicKey);
		auto res = ccCpu->EvalChebyshevSeries(ct, coeffs, 0.0, 1.0);
		Plaintext out;
		ccCpu->Decrypt(res, keysCpu.secretKey, &out);
		out->SetLength(kSlots);
		cpuVals = out->GetRealPackedValue();
	}

	// --- Haze backend ---
	std::vector<double> hazeVals;
	{
		CCParams<CryptoContextCKKSRNS> params;
		setCommonParams(params);
		params.SetBackend(Backend::HAZE);
		params.SetReducedNoise(true); // parity asserts vs a WITH_REDUCED_NOISE OpenFHE oracle
		auto ccHaze = GenCryptoContext(params);
		ccHaze->Enable(PKE);
		ccHaze->Enable(KEYSWITCH);
		ccHaze->Enable(LEVELEDSHE);
		ccHaze->Enable(ADVANCEDSHE);
		auto keysHaze = ccHaze->KeyGen();
		ccHaze->EvalMultKeyGen(keysHaze.secretKey);
		ccHaze->LoadContext(keysHaze.publicKey);
		auto pt	 = ccHaze->MakeCKKSPackedPlaintext(input);
		auto ct	 = ccHaze->Encrypt(pt, keysHaze.publicKey);
		auto res = ccHaze->EvalChebyshevSeries(ct, coeffs, 0.0, 1.0);
		Plaintext out;
		ccHaze->Decrypt(res, keysHaze.secretKey, &out);
		out->SetLength(kSlots);
		hazeVals = out->GetRealPackedValue();
	}

	ASSERT_EQ(hazeVals.size(), cpuVals.size());
	for (size_t i = 0; i < kSlots; ++i)
		EXPECT_NEAR(hazeVals[i], cpuVals[i], 1e-4) << "slot " << i;
}

// Regression coverage for the same bug, contributed by a coworker who hit it independently via a
// degree-13 sign/comparator approximation used in production: encrypt x=0.5, evaluate a fixed
// Chebyshev series on [-1,1] under FIXEDMANUAL on Haze, and compare against the closed-form value
// computed directly from the Chebyshev recurrence (self-checking, no CPU oracle round-trip needed).
// Four coefficient sets exercise different (k,m) Paterson-Stockmeyer shapes: "identity" is the
// trivial k=2/m=1 case on the addScalar-only affine fast path; "simple"/"simple7" straddle the
// dc==1-vs-general-weighted-sum split in the cu branch; "cmp13" is the real degree-13 case that
// originally surfaced this bug.
struct ChebyshevManualCase {
	std::string name;
	std::vector<double> coeffs;
};
class ChebyshevFixedManualCoworker : public ::testing::TestWithParam<ChebyshevManualCase> {};

TEST_P(ChebyshevFixedManualCoworker, MatchesClosedForm) {
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	if (GetTestBackend() != TestBackend::HAZE)
		GTEST_SKIP() << "requires FIDESLIB_TEST_BACKEND=haze";
	const auto& tc = GetParam();
	const double x = 0.5;

	// c0/2 + sum_{j>=1} c_j T_j(x), evaluated via the Chebyshev recurrence T_{j+1}=2x*T_j-T_{j-1}.
	double expected = 0.5 * tc.coeffs[0];
	{
		double Tprev = 1.0, Tcur = x;
		for (size_t j = 1; j < tc.coeffs.size(); ++j) {
			expected += tc.coeffs[j] * Tcur;
			const double Tnext = 2.0 * x * Tcur - Tprev;
			Tprev = Tcur;
			Tcur  = Tnext;
		}
	}

	CCParams<CryptoContextCKKSRNS> params;
	params.SetSecurityLevel(HEStd_128_classic);
	params.SetMultiplicativeDepth(9);
	params.SetScalingModSize(45);
	params.SetRingDim(1u << 16);
	params.SetBatchSize(1u << 15);
	params.SetScalingTechnique(FIXEDMANUAL);
	params.SetBackend(Backend::HAZE);
	params.SetReducedNoise(true);
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->LoadContext(keys.publicKey);

	std::vector<double> vx(4, x);
	auto pt		= cc->MakeCKKSPackedPlaintext(vx);
	auto ct		= cc->Encrypt(pt, keys.publicKey);
	auto coeffs = tc.coeffs;
	auto res	= cc->EvalChebyshevSeries(ct, coeffs, -1.0, 1.0);
	Plaintext out;
	cc->Decrypt(res, keys.secretKey, &out);
	out->SetLength(1);
	const double got = out->GetRealPackedValue()[0];
	EXPECT_NEAR(got, expected, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(
  Repro, ChebyshevFixedManualCoworker,
  ::testing::Values(
    ChebyshevManualCase{ "identity", { 0.0, 1.0 } }, ChebyshevManualCase{ "simple", { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 } },
    ChebyshevManualCase{ "simple7", { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 } },
    ChebyshevManualCase{ "cmp13",
      { -0.5714285714285714, 1.1495595991805079, 0.5019312556142438, -0.09624801724554126, -0.32099708624535217,
        -0.1674146646899354, 0.09941364095727803, 0.2020305089104422, 0.0792797331553387, -0.10519360586987322,
        -0.15458404944693244, -0.03367863784660501, 0.11456253368640568, 0.1295242596300609 } }),
  [](const ::testing::TestParamInfo<ChebyshevManualCase>& info) { return info.param.name; });

// Regression coverage for HazeEngine::setDropInputHostAfterUpload: with the opt-in on, an
// uploaded input's host residues are freed right after upload, so decrypting that same input
// later must still work via refreshHostShadow's empty-shell rebuild. EvalRotate also exercises
// the (unconditional) per-key host-limb drop in ensureKeyUploaded (HazeEngine.cpp:1476).
TEST(DropInputHost, DecryptSurvivesDroppedInput) {
	if (GetTestBackend() != TestBackend::HAZE)
		GTEST_SKIP() << "requires FIDESLIB_TEST_BACKEND=haze";
#ifdef FIDESLIB_ENABLE_HAZE
	constexpr uint32_t kDepth	  = 5;
	constexpr uint32_t kRingDim  = 1u << 16;
	constexpr uint32_t kSlots	  = 8;
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetBackend(Backend::HAZE);
	params.SetReducedNoise(true); // parity asserts vs a WITH_REDUCED_NOISE OpenFHE oracle
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });
	cc->LoadContext(keys.publicKey);

	auto* engine = dynamic_cast<HazeEngine*>(cc->engine_.get());
	ASSERT_NE(engine, nullptr);
	engine->setDropInputHostAfterUpload(true);

	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);

	auto sum = cc->EvalAdd(ct1, ct2);
	auto rot = cc->EvalRotate(sum, 1);
	cc->MarkOutput(rot);

	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[(i + 1) % kSlots] + v2[(i + 1) % kSlots];
	CheckPrecision(cc, rot, keys.secretKey, expected, CKKS_PRECISION);

	// The dropped input still decrypts: refreshHostShadow rebuilds its empty host shell from
	// the still-resident device residues.
	CheckPrecision(cc, ct1, keys.secretKey, v1, CKKS_PRECISION);
	EXPECT_FALSE(ct1->GetElements().empty());
#endif
}

// Same regression, CUDA backend: CudaEngine::setDropInputHostAfterUpload frees an uploaded
// input's host residues right after upload; refreshHostShadow's empty-shell rebuild (needsFresh
// path, CudaEngine.cpp:637-642) must still recover it for decryption.
TEST(DropInputHost, DecryptSurvivesDroppedInputCuda) {
	if (GetTestBackend() != TestBackend::CUDA)
		GTEST_SKIP() << "requires FIDESLIB_TEST_BACKEND=cuda";
#ifdef FIDESLIB_ENABLE_CUDA
	constexpr uint32_t kDepth	  = 5;
	constexpr uint32_t kRingDim  = 1u << 16;
	constexpr uint32_t kSlots	  = 8;
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetBackend(Backend::CUDA);
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });
	cc->LoadContext(keys.publicKey);

	auto* engine = dynamic_cast<CudaEngine*>(cc->engine_.get());
	ASSERT_NE(engine, nullptr);
	engine->setDropInputHostAfterUpload(true);

	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);

	auto sum = cc->EvalAdd(ct1, ct2);
	EXPECT_EQ(ct1->GetNumElementsHost(), 0u);
	auto rot = cc->EvalRotate(sum, 1);
	cc->MarkOutput(rot);

	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[(i + 1) % kSlots] + v2[(i + 1) % kSlots];
	CheckPrecision(cc, rot, keys.secretKey, expected, CKKS_PRECISION);

	// The dropped input still decrypts: refreshHostShadow rebuilds its empty host shell from
	// the still-resident device residues.
	CheckPrecision(cc, ct1, keys.secretKey, v1, CKKS_PRECISION);
	EXPECT_EQ(ct1->GetNumElementsHost(), 2u);

	// A dropped input uploaded at level > 0 pins the fresh shell's absolute-level reset:
	// without it, GetOpenFHECipherText's relative correction double-counts the level.
	auto pt3	= cc->MakeCKKSPackedPlaintext(v1, 1, 2);
	auto ct3	= cc->Encrypt(pt3, keys.publicKey);
	auto sum3 = cc->EvalAdd(ct3, ct3);
	EXPECT_EQ(ct3->GetNumElementsHost(), 0u);
	std::vector<double> doubled(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		doubled[i] = 2.0 * v1[i];
	CheckPrecision(cc, sum3, keys.secretKey, doubled, CKKS_PRECISION);
	CheckPrecision(cc, ct3, keys.secretKey, v1, CKKS_PRECISION);
	EXPECT_EQ(ct3->GetLevelHost(), 2u);
#endif
}

TEST_F(CKKSTest, AccumulateSum) {
	// AccumulateSum(ct, slots) sums all slots into each slot position.
	// With v1={1..8}, sum = 36.
	std::vector<double> expected(kSlots, 36.0);
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->AccumulateSum(ct, static_cast<int>(kSlots));
	// Only slot 0 is guaranteed; the rest depend on rotation key availability.
	Plaintext out;
	cc->Decrypt(res, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

// ---- 10. GetConvolutionTransformRotationIndices ----

TEST_F(CKKSTest, ConvRotIndices_gStep0) {
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 0);
	EXPECT_TRUE(idx.empty());
}

TEST_F(CKKSTest, ConvRotIndices_gStep1) {
	// gStep=1: maxIntra = min(1,8) = 1, loop k in [1,1) → empty intra.
	// blockCount = 1 → no inter. Result: empty.
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 1);
	EXPECT_TRUE(idx.empty());
}

TEST_F(CKKSTest, ConvRotIndices_gStep8) {
	// gStep=8, stride=1: maxIntra=8, intra = {1,2,3,4,5,6,7}.
	// blockCount=1 → no inter entries.
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 8);
	ASSERT_EQ(idx.size(), 7u);
	for (int k = 1; k <= 7; ++k)
		EXPECT_EQ(idx[static_cast<size_t>(k - 1)], k);
}

TEST_F(CKKSTest, ConvRotIndices_gStep16) {
	// gStep=16, stride=2, internal_gstep=8:
	//   maxIntra=8, intra = {2,4,6,8,10,12,14} (k*stride for k=1..7)
	//   blockCount=2, inter = {16} (baseRotation=8*2=16, k=1)
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(32, 4, 2, 16);
	// 7 intra + 1 inter = 8 entries
	ASSERT_EQ(idx.size(), 8u);
	for (int k = 1; k <= 7; ++k)
		EXPECT_EQ(idx[static_cast<size_t>(k - 1)], k * 2);
	EXPECT_EQ(idx[7], 16);
}

// ---- 11. In-place variants ----

TEST_F(CKKSTest, EvalNegateInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = -v1[i];
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalNegateInPlace(ct);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalRotateInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalRotateInPlace(ct, 1);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Rescale) {
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto ct2 = cc->EvalMult(ct, ct);
	auto ct3 = cc->Rescale(ct2);
	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[i] * v1[i];
	CheckPrecision(cc, ct3, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalChebyshevSeriesInPlace) {
	// GPU Chebyshev domain-map fix deferred to branch chebyshev-domain-map-fix (off by ~1 on
	// non-[-1,1] intervals); CPU (OpenFHE) is correct, so this still runs there.
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	std::function<double(double)> f = [](double x) { return x; };
	auto coeffs						= CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(f, 0.0, 1.0, 5);
	std::vector<double> input(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		input[i] = static_cast<double>(i + 1) / (kSlots + 1);
	auto pt = cc->MakeCKKSPackedPlaintext(input);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalChebyshevSeriesInPlace(ct, coeffs, 0.0, 1.0);
	CheckPrecision(cc, ct, keys.secretKey, input, 1e-4);
}

TEST_F(CKKSTest, AccumulateSumInPlace) {
	std::vector<double> expected(kSlots, 36.0);
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->AccumulateSumInPlace(ct, static_cast<int>(kSlots));
	Plaintext out;
	cc->Decrypt(ct, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

// ---- 11b. ConvolutionTransform ----

TEST_F(CKKSTest, ConvolutionTransformIdentity) {
	// Identity transform: bStep=1, gStep=1, single plaintext of ones.
	// With indexes={0} (no baby-step rotation), the result should be
	// ct * pt (scaled by 1), rotated by stride*(1-0)=stride, then rescaled.
	// Use a single "1.0" weight and gStep=1 so the output = rot(ct*1, stride).
	std::vector<double> ones(kSlots, 1.0);
	auto ptWeight				   = cc->MakeCKKSPackedPlaintext(ones);
	std::vector<Plaintext> weights = { ptWeight };
	std::vector<int> indexes	   = { 0 };

	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);

	// gStep=1, bStep=1, stride=1 => rotation by stride*(1-0)=1, then rescale.
	cc->ConvolutionTransformInPlace(ct, 1, 1, weights, indexes, 1);

	// Expected: v1 rotated left by 1, then ct*1.0 = v1 (precision loss from mult+rescale).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-4);
}

// ---- 12. SPARSE_ENCAPSULATED ----

TEST_F(CKKSTest, SparseEncapsulatedContext) {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(5);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecretKeyDist(SPARSE_ENCAPSULATED);
	EXPECT_NO_THROW({
		auto cc2 = GenCryptoContext(params);
		cc2->Enable(PKE);
		cc2->Enable(KEYSWITCH);
		cc2->Enable(LEVELEDSHE);
		auto kp = cc2->KeyGen();
		auto pt = cc2->MakeCKKSPackedPlaintext(v1);
		auto ct = cc2->Encrypt(pt, kp.publicKey);
		Plaintext result;
		cc2->Decrypt(ct, kp.secretKey, &result);
	});
}

// ---- 13. Serialization ----

TEST_F(CKKSTest, SerializeDeserializeEvalMultKey) {
	// Serialize the eval mult key (generated in SetUp) to a buffer.
	std::ostringstream oss;
	ASSERT_TRUE(CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(oss, SerType::BINARY));
	EXPECT_GT(oss.str().size(), 0u);

	// Create a second context with the same parameters but no mult key.
	// Deserializing into it verifies the binary format is self-consistent.
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecurityLevel(HEStd_NotSet);
	auto cc2 = GenCryptoContext(params);
	cc2->Enable(PKE);
	cc2->Enable(KEYSWITCH);
	cc2->Enable(LEVELEDSHE);

	std::istringstream iss(oss.str());
	ASSERT_TRUE(cc2->DeserializeEvalMultKey(iss, SerType::BINARY));

	// Confirm the original context can still multiply (its keys are still live).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMult(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// THE HOST SHADOW'S COPY-ON-WRITE, UNDER CONCURRENCY.
//
// A CiphertextImpl copy SHARES the underlying lbcrypto::Ciphertext with its source and detaches on
// first host mutation (api/Ciphertext.cpp). Whether a value is private is decided by
// `need_lazy_copy` plus `use_count() <= 1` -- and a reference count is a snapshot, not a claim, so
// with several host threads issuing ops on one context the test and the decision can straddle
// another thread's copy. What follows is one lbcrypto ciphertext being rewritten while another
// thread reads or copies it.
//
// THIS IS BACKEND-INDEPENDENT -- the shadow and its copy-on-write are api code -- so this case
// runs on the CPU backend, where it needs no GPU and no device at all. It is the regression test
// for the lock that closes it, and the reason a laptop can hold the line on a defect that was
// found on a GPU.
//
// It skips unless FIDESLIB_CONCURRENT_OPS=1, exactly like the CUDA suite: the default mode is one
// issuing thread per context, so several threads here are out of contract, not a failure.
TEST_F(CKKSTest, ConcurrentHostShadowCopyOnWrite) {
	if (!ConcurrentOpsEnabled())
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op cases";

	constexpr int kLanes = 4;
	constexpr int kReps	 = 40;

	auto pt		 = cc->MakeCKKSPackedPlaintext(v1, 1, 0, nullptr, kSlots);
	auto shared	 = cc->Encrypt(pt, keys.publicKey);

	// What each lane does: take its OWN clone of the one shared ciphertext and then mutate it
	// through the host path (SetLevelHost/EnsureLazyHostCopy is what a readback and every host
	// leveled op go through). Every lane must end with its own value; a lane that mutated the
	// shared source, or copied from it mid-mutation, shows up as a wrong decryption -- and the
	// shared source itself must come back unchanged.
	auto lane_work = [&](int lane, std::vector<double>* out) {
		for (int rep = 0; rep < kReps; ++rep) {
			Ciphertext<DCRTPoly> mine = shared->Clone();
			cc->EvalMultInPlace(mine, static_cast<double>(lane + 1));
			Plaintext got;
			cc->Decrypt(mine, keys.secretKey, &got);
			got->SetLength(kSlots);
			out->assign(kSlots, 0.0);
			for (uint32_t i = 0; i < kSlots; ++i)
				(*out)[i] = got->GetCKKSPackedValue()[i].real();
		}
	};

	std::vector<std::vector<double>> serial(kLanes);
	for (int lane = 0; lane < kLanes; ++lane)
		lane_work(lane, &serial[lane]);

	std::vector<std::vector<double>> parallel(kLanes);
	std::atomic<int> failures{ 0 };
	{
		std::vector<std::thread> pool;
		pool.reserve(kLanes);
		for (int lane = 0; lane < kLanes; ++lane)
			pool.emplace_back([&, lane] {
				try {
					lane_work(lane, &parallel[lane]);
				} catch (const std::exception& e) {
					++failures;
					std::cerr << "lane " << lane << " threw: " << e.what() << std::endl;
				}
			});
		for (auto& th : pool)
			th.join();
	}
	ASSERT_EQ(failures.load(), 0) << "a lane threw under concurrency";

	for (int lane = 0; lane < kLanes; ++lane) {
		ASSERT_EQ(parallel[lane].size(), serial[lane].size()) << "lane " << lane;
		for (uint32_t i = 0; i < kSlots; ++i)
			EXPECT_NEAR(parallel[lane][i], serial[lane][i], CKKS_PRECISION) << "lane " << lane << " slot " << i;
	}
	// The source every lane cloned must be untouched.
	CheckPrecision(cc, shared, keys.secretKey, v1, CKKS_PRECISION);
}

// Bootstrap fixture (FIXEDAUTO): depth=25, ringDim=65536, slots=8, UNIFORM_TERNARY
// ---------------------------------------------------------------------------

class CKKSBootstrapTest : public ::testing::Test {
  protected:
	// Context parameters mirror the CUDA backend's bootstrap tests exactly
	// (test/ParametrizedTest.cuh:377-380 logNboot/firstmodboot/scalemodboot/depthboot, driving
	// OpenFHEBootstrapTest over TTALL64BOOT), so the haze and CUDA bootstrap paths are compared
	// under one parameter set. numLargeDigits mirrors gparams64_13_3 (dnum=3); the CUDA suite
	// sweeps dnum 1..4, this fixture pins one. The old depth=25/scaleMod=50 pairing was not a CUDA
	// reference and could not reach 1e-2 even on pure OpenFHE once the input was really depleted.
	static constexpr uint32_t kDepth	= 23; // depthboot
	static constexpr uint32_t kScaleMod = 59; // scalemodboot
	static constexpr uint32_t kFirstMod = 60; // firstmodboot
	static constexpr uint32_t kRingDim	= 1u << 16; // 1 << logNboot
	static constexpr uint32_t kBatchSize = 8; // gparams64_13_*.batchSize
	static constexpr uint32_t kSlots	= 1u << 5; // OpenFHEBootstrap's `int slots = 1 << 5`
	static constexpr uint32_t kLevelBudget = 2; // EvalBootstrapSetup({2,2})
	// Deepest level EvalBootstrap accepts: L0 - (levelBudget_decode + 2), from the guard at
	// ckksrns-fhe.cpp:1252-1256 (FIXEDAUTO, so no extra FLEXIBLEAUTOEXT tower). The CUDA test
	// depletes to L-1, which it can only do because it bootstraps on the GPU and never calls the
	// host EvalBootstrap that owns this guard; these tests do, so they deplete to the guard's limit
	// instead — as depleted as the host path legally allows. Encrypting AT this level is what makes
	// bootstrap real: a full-level input gains no levels and returns unchanged (a false pass).
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
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		else if (GetTestBackend() == TestBackend::HAZE) {
			params.SetBackend(Backend::HAZE);
			params.SetReducedNoise(true); // parity asserts vs a WITH_REDUCED_NOISE OpenFHE oracle
		}
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		// FIXEDAUTO EvalBootstrapSetup({1,1}) segfaults on GPU (pre-existing on main, tracked
		// separately); the tests below skip on GPU, so set up bootstrap only on the CPU backend.
		// Under haze the setup runs (the segfault is CUDA-specific).
		if (!TestUseCuda()) {
			// levelBudget {2,2}, dim1 {0,0}, derived correction factor — OpenFHEBootstrap's
			// EvalBootstrapSetup({ 2, 2 }, { 0, 0 }, slots) verbatim.
			cc->EvalBootstrapSetup({ kLevelBudget, kLevelBudget }, { 0, 0 }, kSlots, 0);
			cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		}
		if (GetTestBackend() == TestBackend::HAZE)
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 8. Bootstrap ----

TEST_F(CKKSBootstrapTest, Bootstrap) {
	if (TestUseCuda())
		GTEST_SKIP() << "FIXEDAUTO bootstrap setup segfaults on GPU (pre-existing on main, tracked separately)";
	// Encrypt DEPLETED so EvalBootstrap runs the real pipeline. A full-level input gains no levels,
	// so bootstrap returns a clone of the input unchanged (HazeBootstrap.cpp:1237, mirroring OpenFHE
	// ckksrns-fhe.cpp:835) — which still decrypts to v1 and passes: a false pass. kMaxBootLevel is
	// the deepest level bootstrap accepts (ckksrns-fhe.cpp:1252-1256), so this is as depleted as the
	// input can legally be; going further throws "Not enough levels to perform Bootstrapping".
	auto pt		   = cc->MakeCKKSPackedPlaintext(v1, 1, kMaxBootLevel, nullptr, kSlots);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	const size_t levelBefore = ct->GetLevel();
	auto refreshed = cc->EvalBootstrap(ct);
	// Guard against the false pass: if bootstrap gained no levels it returns the input unchanged,
	// which still decrypts to v1 and would satisfy CheckPrecision while proving nothing.
	ASSERT_LT(refreshed->GetLevel(), levelBefore) << "bootstrap gained no levels: it returned the input unchanged, so this test exercised nothing";
	CheckPrecision(cc, refreshed, keys.secretKey, v1, CKKS_BOOTSTRAP_PRECISION);
}

TEST_F(CKKSBootstrapTest, BootstrapInPlace) {
	if (TestUseCuda())
		GTEST_SKIP() << "FIXEDAUTO bootstrap setup segfaults on GPU (pre-existing on main, tracked separately)";
	// Depleted input, same reason as CKKSBootstrapTest.Bootstrap above.
	auto pt = cc->MakeCKKSPackedPlaintext(v1, 1, kMaxBootLevel, nullptr, kSlots);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	const size_t levelBefore = ct->GetLevel();
	cc->EvalBootstrapInPlace(ct);
	ASSERT_LT(ct->GetLevel(), levelBefore) << "bootstrap gained no levels: it returned the input unchanged, so this test exercised nothing";
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_BOOTSTRAP_PRECISION);
}

// ---- 9. GetPreScaleFactor (FIXEDAUTO) ----

TEST_F(CKKSBootstrapTest, GetPreScaleFactorFIXEDAUTO) {
	if (TestUseCuda())
		GTEST_SKIP() << "FIXEDAUTO bootstrap setup segfaults on GPU (pre-existing on main, tracked separately)";
	// EvalBootstrapSetup sets m_correctionFactor; without it the result underflows.
	double r = cc->GetPreScaleFactor(kSlots);
	EXPECT_TRUE(std::isfinite(r)) << "result is not finite: " << r;
	EXPECT_GT(r, 0.0);
	EXPECT_LT(r, 1.0);
}

// ---------------------------------------------------------------------------
// Bootstrap fixture (FLEXIBLEAUTO): identical parameters but FLEXIBLEAUTO
// ---------------------------------------------------------------------------

class CKKSFlexBootstrapTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 25;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FLEXIBLEAUTO);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		else if (GetTestBackend() == TestBackend::HAZE) {
			params.SetBackend(Backend::HAZE);
			params.SetReducedNoise(true); // parity asserts vs a WITH_REDUCED_NOISE OpenFHE oracle
		}
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup({ 1, 1 }, { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		if (TestUseDevice())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 9. GetPreScaleFactor (FLEXIBLEAUTO) ----

TEST_F(CKKSFlexBootstrapTest, GetPreScaleFactorFLEXIBLEAUTO) {
	double r = cc->GetPreScaleFactor(kSlots);
	EXPECT_TRUE(std::isfinite(r)) << "result is not finite: " << r;
	EXPECT_GT(r, 0.0);
	// FLEXIBLEAUTO result can exceed 1.
}

// ===========================================================================
// Additional coverage tests
// ===========================================================================

// ---- Mutable variants ----

TEST_F(CKKSTest, EvalAddMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalAddMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalAddMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalAddMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalSubMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalSubMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalSubMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMultMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalMultMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalMultMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquareMutable) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSquareMutable(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- Multi-level operations ----

TEST_F(CKKSTest, MultChainedAllLevels) {
	// Multiply through all 5 depth levels, verifying at each step.
	auto pt					= cc->MakeCKKSPackedPlaintext(v2);
	auto ct					= cc->Encrypt(pt, keys.publicKey);
	std::vector<double> cur = v2;
	for (uint32_t d = 0; d < kDepth; ++d) {
		ct = cc->EvalMult(ct, ct);
		for (size_t i = 0; i < cur.size(); ++i)
			cur[i] = cur[i] * cur[i];
	}
	// Precision degrades with depth; use a looser tolerance.
	CheckPrecision(cc, ct, keys.secretKey, cur, 1e-2);
}

TEST_F(CKKSTest, RotateAfterMult) {
	// Consume a level via multiplication, then rotate at the lower level.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto ct	 = cc->EvalMult(ct1, ct2);
	auto res = cc->EvalRotate(ct, 1);
	// Expected: (v1*v2) rotated left by 1.
	std::vector<double> rotated(expected.size());
	for (size_t i = 0; i < expected.size(); ++i)
		rotated[i] = expected[(i + 1) % expected.size()];
	CheckPrecision(cc, res, keys.secretKey, rotated, CKKS_PRECISION);
}

// ---- Ciphertext copy and clone ----

TEST_F(CKKSTest, CiphertextCopy) {
	// Ciphertext<DCRTPoly> is a shared_ptr; copying shares the underlying object.
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	Ciphertext<DCRTPoly> alias(ct);
	// Both alias and ct point to the same ciphertext.
	cc->EvalAddInPlace(ct, ct);
	// alias sees the modification (shared semantics).
	std::vector<double> doubled(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		doubled[i] = v1[i] * 2.0;
	CheckPrecision(cc, alias, keys.secretKey, doubled, CKKS_PRECISION);
}

TEST_F(CKKSTest, CiphertextClone) {
	auto pt	   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	   = cc->Encrypt(pt, keys.publicKey);
	auto clone = ct->Clone();
	cc->EvalAddInPlace(ct, ct);
	CheckPrecision(cc, clone, keys.secretKey, v1, CKKS_PRECISION);
}

// ---- Getters / setters ----

TEST_F(CKKSTest, GetLevel) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	EXPECT_EQ(ct->GetLevel(), 0u);
	auto ct2 = cc->EvalMult(ct, ct);
	// After mult, level may have increased.
	EXPECT_GE(ct2->GetLevel(), 0u);
}

TEST_F(CKKSTest, GetNoiseScaleDeg) {
	auto pt	   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	   = cc->Encrypt(pt, keys.publicKey);
	size_t deg = ct->GetNoiseScaleDeg();
	EXPECT_GE(deg, 1u);
}

// ---- Automorphism key serialization ----

TEST_F(CKKSTest, SerializeDeserializeEvalAutomorphismKey) {
	std::ostringstream oss;
	ASSERT_TRUE(CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(oss, SerType::BINARY));
	EXPECT_GT(oss.str().size(), 0u);

	// Deserialize into a second context with the same parameters.
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecurityLevel(HEStd_NotSet);
	auto cc2 = GenCryptoContext(params);
	cc2->Enable(PKE);
	cc2->Enable(KEYSWITCH);
	cc2->Enable(LEVELEDSHE);

	std::istringstream iss(oss.str());
	ASSERT_TRUE(cc2->DeserializeEvalAutomorphismKey(iss, SerType::BINARY));

	// Verify original context can still rotate.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalRotate(ct, 1);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- EvalAddManyInPlace ----

TEST_F(CKKSTest, EvalAddManyInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i] + v1[i];
	auto pt1							  = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2							  = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1							  = cc->Encrypt(pt1, keys.publicKey);
	auto ct2							  = cc->Encrypt(pt2, keys.publicKey);
	auto ct3							  = cc->Encrypt(pt1, keys.publicKey);
	std::vector<Ciphertext<DCRTPoly>> cts = { ct1, ct2, ct3 };
	cc->EvalAddManyInPlace(cts);
	// Result is in cts[0].
	CheckPrecision(cc, cts[0], keys.secretKey, expected, CKKS_PRECISION);
}

// ---------------------------------------------------------------------------
// 14. Linear transforms: ConvolutionTransform / SpecialConvolutionTransform
//
// The CPU baby-step/giant-step path is compared against an independent
// clear-text simulation of the documented algorithm (which mirrors
// src/CKKS/LinearTransform.cu). Parameters are chosen to cover the branchy
// pieces: single even block, single odd block (linear accumulate), and a
// two-block case (inter-block rotation + even-tree accumulate), plus masking.
// ---------------------------------------------------------------------------

namespace {

// Clear-text reference. rot(v, k)[i] = v[(i + k) mod n] (OpenFHE left rotation).
std::vector<double> SimulateConvolution(const std::vector<double>& x,
  const std::vector<std::vector<double>>& pts,
  const std::vector<int>& indexes,
  int bStep,
  int gStep,
  int stride,
  int rowSize,
  const std::vector<double>* mask,
  int maskRotationStride) {
	const int n = static_cast<int>(x.size());
	auto rot	= [n](const std::vector<double>& v, int k) {
		   std::vector<double> out(n);
		   for (int i = 0; i < n; ++i)
			   out[i] = v[(((i + k) % n) + n) % n];
		   return out;
	};
	if (rowSize == 0)
		rowSize = bStep * gStep;

	std::vector<std::vector<double>> fr(bStep);
	for (int i = 0; i < bStep; ++i)
		fr[i] = rot(x, indexes[i]);

	constexpr uint32_t INTERNAL = 8;
	uint32_t blockCount			= (static_cast<uint32_t>(gStep) + INTERNAL - 1) / INTERNAL;
	std::vector<std::vector<double>> blockResults;
	for (uint32_t b = 0; b < blockCount; ++b) {
		uint32_t start = b * INTERNAL;
		uint32_t end   = std::min(start + INTERNAL, static_cast<uint32_t>(gStep));
		uint32_t cur   = end - start;
		std::vector<std::vector<double>> results(cur, std::vector<double>(n, 0.0));
		for (uint32_t j = 0; j < cur; ++j) {
			uint32_t gj = start + j;
			for (int s = 0; s < n; ++s)
				results[j][s] = pts[bStep * gj][s] * fr[0][s];
			for (int i = 1; i < bStep; ++i) {
				int ptIdx = bStep * static_cast<int>(gj) + i;
				if (ptIdx < rowSize)
					for (int s = 0; s < n; ++s)
						results[j][s] += pts[ptIdx][s] * fr[i][s];
			}
			if (mask != nullptr) {
				auto r1 = rot(results[j], maskRotationStride);
				auto r2 = rot(results[j], 2 * maskRotationStride);
				for (int s = 0; s < n; ++s)
					results[j][s] = (results[j][s] + r1[s] + r2[s]) * (*mask)[s];
			}
			int rotation = stride * static_cast<int>(cur - j);
			if (rotation != 0)
				results[j] = rot(results[j], rotation);
		}
		std::vector<double> acc(n, 0.0);
		for (uint32_t j = 0; j < cur; ++j)
			for (int s = 0; s < n; ++s)
				acc[s] += results[j][s];
		blockResults.push_back(acc);
	}
	if (blockCount > 1) {
		int baseRotation = static_cast<int>(INTERNAL) * stride;
		for (uint32_t b = 0; b < blockCount - 1; ++b) {
			int rotation = static_cast<int>(blockCount - 1 - b) * baseRotation;
			if (rotation != 0)
				blockResults[b] = rot(blockResults[b], rotation);
		}
	}
	std::vector<double> out(n, 0.0);
	for (const auto& br : blockResults)
		for (int s = 0; s < n; ++s)
			out[s] += br[s];
	return out;
}

// Deterministic, per-slot-varying weights so a rotation/index bug cannot hide.
std::vector<std::vector<double>> MakeWeights(int count, int n) {
	std::vector<std::vector<double>> w(count, std::vector<double>(n));
	for (int j = 0; j < count; ++j)
		for (int s = 0; s < n; ++s)
			w[j][s] = 0.1 * (((j * 3 + s) % 7) + 1);
	return w;
}

// Encrypt() takes a Plaintext& lvalue, so bind the encoded plaintext first.
Ciphertext<DCRTPoly> EncryptVec(CryptoContext<DCRTPoly>& cc, const std::vector<double>& v, const PublicKey<DCRTPoly>& pk) {
	auto pt = cc->MakeCKKSPackedPlaintext(v);
	return cc->Encrypt(pt, pk);
}

} // namespace

TEST_F(CKKSTest, ConvolutionTransformSingleBlockEven) {
	// bStep=2, gStep=2: single block, even-tree accumulate.
	const int gStep = 2, bStep = 2, stride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-3);
}

TEST_F(CKKSTest, ConvolutionTransformSingleBlockOdd) {
	// bStep=2, gStep=3: single block, odd count exercises the linear-accumulate branch.
	const int gStep = 3, bStep = 2, stride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-3);
}

TEST_F(CKKSTest, ConvolutionTransformMultiBlock) {
	// bStep=1, gStep=9: two giant-step blocks, exercising inter-block rotation
	// and accumulation on top of the per-block (size-8 even) accumulate.
	const int gStep = 9, bStep = 1, stride = 1;
	const std::vector<int> indexes = { 0 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 5e-3);
}

TEST_F(CKKSTest, SpecialConvolutionTransform) {
	// Masked variant: bStep=2, gStep=2, maskRotationStride=1.
	const int gStep = 2, bStep = 2, stride = 1, maskRotationStride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);
	std::vector<double> maskVals(kSlots);
	for (size_t s = 0; s < kSlots; ++s)
		maskVals[s] = (s % 2 == 0) ? 1.0 : 0.5;

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto mask = cc->MakeCKKSPackedPlaintext(maskVals);
	auto ct	  = EncryptVec(cc, v1, keys.publicKey);

	cc->SpecialConvolutionTransformInPlace(ct, gStep, bStep, pts, mask, indexes, stride, maskRotationStride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, &maskVals, maskRotationStride);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-2);
}

// ---------------------------------------------------------------------------
// 14b. LinearTransformInPlace (fused BSGS matrix-vector product)
// ---------------------------------------------------------------------------
// These do NOT replay the algorithm: they assert the api's actual promise, that a
// caller who packs diagonals per the documented layout gets M*v back. So the
// oracle is an independent host matrix-vector product, and a wrong layout
// contract fails here even if CPU and CUDA agree with each other.
//
// Runs on whichever backend the fixture selected; the CUDA path is exercised by
// the same cases under FIDESLIB_TEST_BACKEND=cuda.
// ---------------------------------------------------------------------------

namespace {

// Left rotation with OpenFHE's sign: HostRot(v, r)[i] = v[(i + r) mod n].
std::vector<double> HostRot(const std::vector<double>& v, int r) {
	const int n = static_cast<int>(v.size());
	std::vector<double> out(static_cast<size_t>(n));
	for (int i = 0; i < n; ++i)
		out[static_cast<size_t>(i)] = v[static_cast<size_t>((((i + r) % n) + n) % n)];
	return out;
}

// Deterministic pseudo-random n-by-n matrix in [-1, 1]; `band` (0 = dense) zeroes every
// entry whose cyclic diagonal index falls outside [-band, band], which is what a transform
// with rowSize < n and a negative offset expects.
std::vector<std::vector<double>> MakeRandomMatrix(int n, uint32_t seed, int band = 0) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::vector<std::vector<double>> M(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int d = ((j - i) % n + n) % n; // cyclic diagonal index in [0, n)
			if (d > n / 2)
				d -= n;
			if (band != 0 && std::abs(d) > band)
				continue;
			M[static_cast<size_t>(i)][static_cast<size_t>(j)] = dist(rng);
		}
	}
	return M;
}

std::vector<double> HostMatVec(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
	const size_t n = v.size();
	std::vector<double> out(n, 0.0);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			out[i] += M[i][j] * v[j];
	return out;
}

// Pack M's diagonals exactly as CryptoContextImpl::LinearTransformInPlace documents:
//   index k  <->  the diagonal that multiplies Rot(ct, k*stride + offset),
//                 i.e. D_k[i] = M[i][(i + k*stride + offset) mod n],
//   stored pre-rotated by GetLinearTransformPlaintextRotationIndices(...)[k].
// This is the reference the application port should copy.
//
// `extended` switches ONLY the encoder — MakeCKKSPackedPlaintextExtended instead of
// MakeCKKSPackedPlaintext — and nothing else. That is the point of the flag: the packing, the
// index-to-rotation map and the pre-rotation are identical, so an application that already builds
// diagonals this way opts into the extended Q||P basis by changing one call.
std::vector<Plaintext>
PackDiagonals(CryptoContext<DCRTPoly>& cc, const std::vector<std::vector<double>>& M, int rowSize, int bStep, int stride, int offset, bool extended = false) {
	const int n		 = static_cast<int>(M.size());
	const auto ptRot = CryptoContextImpl<DCRTPoly>::GetLinearTransformPlaintextRotationIndices(rowSize, bStep, stride, offset);
	std::vector<Plaintext> pts;
	pts.reserve(static_cast<size_t>(rowSize));
	for (int k = 0; k < rowSize; ++k) {
		const int r = k * stride + offset;
		std::vector<double> d(static_cast<size_t>(n));
		for (int i = 0; i < n; ++i)
			d[static_cast<size_t>(i)] = M[static_cast<size_t>(i)][static_cast<size_t>((((i + r) % n) + n) % n)];
		auto diag = HostRot(d, ptRot[static_cast<size_t>(k)]);
		pts.push_back(extended ? cc->MakeCKKSPackedPlaintextExtended(diag) : cc->MakeCKKSPackedPlaintext(diag));
	}
	return pts;
}

} // namespace

TEST_F(CKKSTest, LinearTransformRotationIndices) {
	// Ciphertext keys: {i*stride : i in [1, bStep]}, plus offset when non-zero.
	auto idx = CryptoContextImpl<DCRTPoly>::GetLinearTransformRotationIndices(4, 1, 0);
	ASSERT_EQ(idx.size(), 4u);
	for (int i = 1; i <= 4; ++i)
		EXPECT_EQ(idx[static_cast<size_t>(i) - 1], i);

	auto idxOff = CryptoContextImpl<DCRTPoly>::GetLinearTransformRotationIndices(2, 3, -5);
	ASSERT_EQ(idxOff.size(), 3u);
	EXPECT_EQ(idxOff[0], 3);
	EXPECT_EQ(idxOff[1], 6);
	EXPECT_EQ(idxOff[2], -5);

	// Plaintext pre-rotation: constant within a giant step, -bStep*j*stride - offset.
	auto ptRot = CryptoContextImpl<DCRTPoly>::GetLinearTransformPlaintextRotationIndices(5, 2, 1, -2);
	ASSERT_EQ(ptRot.size(), 5u);
	EXPECT_EQ(ptRot[0], 2); // j=0: -0 + 2
	EXPECT_EQ(ptRot[1], 2);
	EXPECT_EQ(ptRot[2], 0); // j=1: -2 + 2
	EXPECT_EQ(ptRot[3], 0);
	EXPECT_EQ(ptRot[4], -2); // j=2: -4 + 2
}

TEST_F(CKKSTest, LinearTransformMatVecSingleGiantStep) {
	// bStep = rowSize = 8 -> gStep = 1: no giant-step fold at all, pure baby steps.
	// Keys: {1..8}, all generated by the fixture.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 8, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 20260829u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformMatVecBabyGiantSplit) {
	// bStep = 4, rowSize = 8 -> gStep = 2: exercises the backwards fold and its
	// bStep*stride = 4 rotation between giant steps.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 991u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformMatVecShortLastGiantStep) {
	// bStep = 3, rowSize = 8 -> gStep = 3 with a 2-wide last giant step: the branch
	// where the diagonal index runs past rowSize (CUDA null-pads it, CPU stops early).
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 3, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 4242u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformMatVecWithOffset) {
	// The bert-tiny sigma shape: a banded matrix whose diagonals run -2..+2, covered by
	// rowSize = 5 with offset = -2 so index k maps to ciphertext rotation k - 2.
	// Keys: {1, 2} from bStep, plus the offset -2.
	const int n = static_cast<int>(kSlots), rowSize = 5, bStep = 2, stride = 1, offset = -2;
	auto M	 = MakeRandomMatrix(n, 777u, /*band=*/2);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformRejectsBadArguments) {
	const int n = static_cast<int>(kSlots);
	auto M		= MakeRandomMatrix(n, 1u);
	auto pts	= PackDiagonals(cc, M, n, 4, 1, 0);
	auto ct		= EncryptVec(cc, v1, keys.publicKey);

	EXPECT_THROW(cc->LinearTransformInPlace(ct, 0, 4, pts, 1, 0), std::exception);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, n, 0, pts, 1, 0), std::exception);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, n + 1, 4, pts, 1, 0), std::exception);

	auto withNull = pts;
	withNull[3]	  = Plaintext{};
	EXPECT_THROW(cc->LinearTransformInPlace(ct, n, 4, withNull, 1, 0), std::exception);
}

// ---- Extended (mod-up, Q||P) diagonals ----
// The extended basis exists so the CUDA transform can keep the baby rotations and the MAC
// accumulation in Q||P and pay one ModDown per giant step instead of one per baby rotation. It is
// invisible to the CKKS math, so these cases assert exactly that: the same host oracle, to the same
// tolerance, off diagonals that differ only in which encoder built them. On the CPU backend
// MakeCKKSPackedPlaintextExtended returns the ordinary encoding and the engine ignores the flag,
// which is what makes "same answer" the right assertion rather than a weaker one — and the same
// cases run the real ext path under FIDESLIB_TEST_BACKEND=cuda.

TEST_F(CKKSTest, MakeCKKSPackedPlaintextExtendedMatchesOrdinaryEncoding) {
	auto plain = cc->MakeCKKSPackedPlaintext(v1);
	auto ext   = cc->MakeCKKSPackedPlaintextExtended(v1);

	// The extended request is recorded on every backend, including the ones that ignore it.
	EXPECT_TRUE(ext->extended);
	EXPECT_FALSE(plain->extended);

	// Only the RNS basis may differ: slots, level and the packed values must not.
	EXPECT_EQ(ext->GetSlots(), plain->GetSlots());
	EXPECT_EQ(ext->GetLevel(), plain->GetLevel());
	ext->SetLength(v1.size());
	auto vals = ext->GetRealPackedValue();
	ASSERT_EQ(vals.size(), v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(vals[i], v1[i], CKKS_PRECISION) << "slot " << i;

	EXPECT_THROW(cc->MakeCKKSPackedPlaintextExtended(std::vector<double>{}), std::exception);
}

// ---- Device encode ----
// MakeCKKSPackedPlaintextDevice moves the per-tower half of the encode onto the backend so the
// transfer carries one coefficient vector instead of L encoded towers. On a backend with no device
// there is nothing to move, so the facade falls back to the ordinary encoder — and THAT is what
// this case pins down, because the fallback is what makes the call safe for a caller to make
// unconditionally, and what makes an application's device-encode flag a behavioural no-op on CPU
// rather than a different answer. The bit-for-bit device-vs-host comparison needs a device and
// lives in test/DeviceEncodeTests.cu.
TEST_F(CKKSTest, MakeCKKSPackedPlaintextDeviceFallsBackOnHostBackends) {
	if (cc->SupportsDeviceEncode())
		GTEST_SKIP() << "backend encodes on the device; the fallback under test is the host-backend path.";

	auto plain = cc->MakeCKKSPackedPlaintext(v1);
	auto dev   = cc->MakeCKKSPackedPlaintextDevice(v1);

	// The fallback is the ordinary encoding, so the plaintext is an ordinary one in every respect:
	// it keeps a host value, and every accessor answers rather than throwing.
	EXPECT_FALSE(dev->device_only);
	EXPECT_TRUE(dev->host.has_value());
	EXPECT_EQ(dev->GetSlots(), plain->GetSlots());
	EXPECT_EQ(dev->GetLevel(), plain->GetLevel());

	dev->SetLength(v1.size());
	auto vals = dev->GetRealPackedValue();
	ASSERT_EQ(vals.size(), v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(vals[i], v1[i], CKKS_PRECISION) << "slot " << i;

	EXPECT_THROW(cc->MakeCKKSPackedPlaintextDevice(std::vector<double>{}), std::exception);
}

// MakeCKKSPackedPlaintextDeviceMany claims ONE thing: element for element, the batch is what the
// single call would have produced. On a HOST backend it honours that by construction -- the facade
// loops MakeCKKSPackedPlaintextDevice, which itself falls back to MakeCKKSPackedPlaintext -- and
// this case pins it, because that fallback chain is what lets an application turn the batched arm
// on unconditionally and still get a behavioural no-op out of its CPU gates.
//
// On a device-encoding backend the batch takes genuinely different kernels (one upload, one reduce
// launch, one NTT launch pair for the whole group), so the claim needs a limb-by-limb comparison
// against the single call; that needs a device and lives in test/DeviceEncodeTests.cu. The obvious
// api-level substitute -- multiply one ciphertext by each plaintext and compare the products -- is
// NOT usable as a parity oracle here: the api's Clone copies the handle, and EvalMult adjusts its
// operands, so the two products would not start from the same ciphertext.
TEST_F(CKKSTest, MakeCKKSPackedPlaintextDeviceManyMatchesSingleOnHostBackends) {
	if (cc->SupportsDeviceEncode())
		GTEST_SKIP() << "backend encodes on the device; the limb-level batch-vs-single comparison lives in DeviceEncodeTests.cu.";

	// Mixed signs and magnitudes, and a SHORT vector among them: the batch shares one `slots`
	// resolution, so a member shorter than the rest is exactly the case where a batched encoder
	// could silently pad differently from the single one.
	const std::vector<std::vector<double>> batch = {
		v1,
		v2,
		{ -1.5, 2.25, -3.125, 4.0, -5.5, 6.75, -7.875, 8.0 },
		{ 0.0, -0.0, 1.0, -1.0 },
	};

	for (uint32_t level = 0; level <= 2; ++level) {
		std::vector<Plaintext> many = cc->MakeCKKSPackedPlaintextDeviceMany(batch, 1, level);
		ASSERT_EQ(many.size(), batch.size()) << "level " << level;

		for (size_t i = 0; i < batch.size(); ++i) {
			Plaintext one = cc->MakeCKKSPackedPlaintextDevice(batch[i], 1, level);

			// A host-backend result is an ordinary plaintext in every respect, exactly as the
			// single call's fallback is.
			ASSERT_FALSE(many[i]->device_only) << "level " << level << ", element " << i;
			ASSERT_TRUE(many[i]->host.has_value()) << "level " << level << ", element " << i;

			EXPECT_EQ(many[i]->GetSlots(), one->GetSlots()) << "level " << level << ", element " << i;
			EXPECT_EQ(many[i]->GetLevel(), one->GetLevel()) << "level " << level << ", element " << i;

			// EXACT, not approximate: both went through the same encoder, so any difference is a
			// batching bug rather than precision.
			one->SetLength(batch[i].size());
			many[i]->SetLength(batch[i].size());
			const auto va = one->GetRealPackedValue();
			const auto vb = many[i]->GetRealPackedValue();
			ASSERT_EQ(va.size(), vb.size()) << "level " << level << ", element " << i;
			for (size_t j = 0; j < va.size(); ++j) {
				ASSERT_EQ(va[j], vb[j]) << "level " << level << ", element " << i << ", slot " << j;
			}
		}
	}
}

// The refusals the batch adds to the single call's own, each of which has to fail loudly rather
// than encode something else.
TEST_F(CKKSTest, MakeCKKSPackedPlaintextDeviceManyRefusesBadBatches) {
	// An empty batch is a caller bug, not a no-op: a caller that meant to encode a group and passed
	// none would otherwise get an empty vector and a silent MISS downstream.
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDeviceMany(std::vector<std::vector<double>>{}), std::exception);
	// An empty member is the single call's own refusal, reached through the batch.
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDeviceMany({ v1, std::vector<double>{} }), std::exception);
	// noiseScaleDeg > 1 and an out-of-range level, likewise.
	if (cc->SupportsDeviceEncode()) {
		EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDeviceMany({ v1, v2 }, 2, 0), std::exception);
		EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDeviceMany({ v1, v2 }, 1, kDepth + 8), std::exception);
	}
}

TEST_F(CKKSTest, LinearTransformMatVecExtendedBabyGiantSplit) {
	// Same case as LinearTransformMatVecBabyGiantSplit (gStep = 2, so the backwards fold and its
	// bStep*stride rotation both run), with extended diagonals and ext = true.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 991u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset, /*extended=*/true);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset, /*ext=*/true);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformMatVecExtendedWithOffset) {
	// The bert-tiny sigma shape again (rowSize = 5, offset = -2, short last giant step), extended:
	// the offset branch of the fold is the one that moddowns before its final rotation.
	const int n = static_cast<int>(kSlots), rowSize = 5, bStep = 2, stride = 1, offset = -2;
	auto M	 = MakeRandomMatrix(n, 777u, /*band=*/2);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset, /*extended=*/true);
	auto ct	 = EncryptVec(cc, v1, keys.publicKey);

	cc->LinearTransformInPlace(ct, rowSize, bStep, pts, stride, offset, /*ext=*/true);

	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

TEST_F(CKKSTest, LinearTransformExtRequiresExtendedDiagonals) {
	// ext = true is a requirement, not a hint: the fused kernel would silently take the slow
	// per-rotation-ModDown path off ordinary diagonals. The facade rejects the mismatch instead,
	// on every backend, so the mistake surfaces on CPU too.
	const int n = static_cast<int>(kSlots);
	auto M		= MakeRandomMatrix(n, 31337u);
	auto ct		= EncryptVec(cc, v1, keys.publicKey);

	auto plainPts = PackDiagonals(cc, M, n, 4, 1, 0);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, n, 4, plainPts, 1, 0, /*ext=*/true), std::exception);

	// One ordinary diagonal in an otherwise extended set is rejected just the same.
	auto mixed = PackDiagonals(cc, M, n, 4, 1, 0, /*extended=*/true);
	mixed[2]   = cc->MakeCKKSPackedPlaintext(std::vector<double>(kSlots, 1.0));
	EXPECT_THROW(cc->LinearTransformInPlace(ct, n, 4, mixed, 1, 0, /*ext=*/true), std::exception);

	// And extended diagonals with ext = false are still legal — the flag only ever adds a
	// requirement, so it can be turned off without repacking.
	auto extPts = PackDiagonals(cc, M, n, 4, 1, 0, /*extended=*/true);
	EXPECT_NO_THROW(cc->LinearTransformInPlace(ct, n, 4, extPts, 1, 0, /*ext=*/false));
	CheckPrecision(cc, ct, keys.secretKey, HostMatVec(M, v1), 1e-3);
}

// ---------------------------------------------------------------------------
// 14c. LinearTransformMany (N independent transforms of ONE source)
// ---------------------------------------------------------------------------
// The promise is an EQUIVALENCE, not just an approximation: LinearTransformMany
// must return exactly what N independent LinearTransformInPlace calls on clones
// of the same source return. So these check both ends — the host matrix-vector
// oracle (the contract) and slot-for-slot equality against the single-call path
// (the equivalence). A batched kernel that shares the hoisted rotations across
// transforms but gets the per-transform fold wrong fails the second check even
// though the first still looks plausible.
// ---------------------------------------------------------------------------

namespace {

// How close two decryptions of the SAME computation have to be to count as the same result.
//
// Not zero, and not because the two paths differ: OpenFHE's CKKS Decrypt is itself nondeterministic
// at the 1e-11 level (decrypting one ciphertext twice does not give the same doubles), so equality
// of the underlying ciphertexts is not observable through it. 1e-8 is three orders above that floor
// and five below the 1e-3 the transform's own contract is checked at, so it still catches any real
// disagreement — a per-transform fold applied to the wrong partials is off by O(1), not O(1e-8).
constexpr double kSameResult = 1e-8;

// Decrypt to `len` slots. The Many tests compare two ciphertexts against each other, which needs the
// values, not a tolerance against an oracle.
std::vector<double> DecryptSlots(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, size_t len) {
	Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	pt->SetLength(len);
	return pt->GetRealPackedValue();
}

} // namespace

TEST_F(CKKSTest, LinearTransformManyMatchesIndependentCalls) {
	// The q/k/v shape: THREE transforms of one source. bStep = 4, rowSize = n = 8 -> gStep = 2, so
	// the per-transform backwards fold and its bStep*stride rotation both run and cannot be shared.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	const std::vector<uint32_t> seeds = { 991u, 20260829u, 4242u };

	// Two independently encoded copies of the same diagonals, one per path. Sharing the objects
	// would compare a first consumption against a second consumption of the same plaintexts (which
	// a backend is free to leave device-resident, level-adjusted, or otherwise touched) rather than
	// one algorithm against the other. Encoding is deterministic, so the two copies are equal.
	std::vector<std::vector<std::vector<double>>> Ms;
	std::vector<std::vector<Plaintext>> sets, refSets;
	for (uint32_t s : seeds) {
		Ms.push_back(MakeRandomMatrix(n, s));
		sets.push_back(PackDiagonals(cc, Ms.back(), rowSize, bStep, stride, offset));
		refSets.push_back(PackDiagonals(cc, Ms.back(), rowSize, bStep, stride, offset));
	}

	auto src = EncryptVec(cc, v1, keys.publicKey);
	auto got = cc->LinearTransformMany(src, rowSize, bStep, sets, stride, offset);
	ASSERT_EQ(got.size(), sets.size());

	for (size_t t = 0; t < sets.size(); ++t) {
		// (a) the contract: M_t * v.
		CheckPrecision(cc, got[t], keys.secretKey, HostMatVec(Ms[t], v1), 1e-3);

		// (b) the equivalence: the same source through the single-transform call.
		auto one = src->Clone();
		cc->LinearTransformInPlace(one, rowSize, bStep, refSets[t], stride, offset);
		auto a = DecryptSlots(cc, got[t], keys.secretKey, kSlots);
		auto b = DecryptSlots(cc, one, keys.secretKey, kSlots);
		ASSERT_EQ(a.size(), b.size());
		for (size_t i = 0; i < a.size(); ++i)
			EXPECT_NEAR(a[i], b[i], kSameResult) << "transform " << t << ", slot " << i;

		// Same result metadata too — level and noise scale degree are part of what downstream code
		// reads off the ciphertext.
		EXPECT_EQ(got[t]->GetLevel(), one->GetLevel()) << "transform " << t;
		EXPECT_EQ(got[t]->GetNoiseScaleDeg(), one->GetNoiseScaleDeg()) << "transform " << t;
	}

	// The source is an input, not a workspace: the batch must not consume it.
	CheckPrecision(cc, src, keys.secretKey, v1, CKKS_PRECISION);
}

TEST_F(CKKSTest, LinearTransformManyWithOffsetAndExtendedDiagonals) {
	// The gate/up shape: TWO transforms, this time on the banded rowSize = 5 / offset = -2 / short
	// last giant step case, with extended diagonals — the fold branch that moddowns before its final
	// offset rotation, run once per transform out of one batch.
	const int n = static_cast<int>(kSlots), rowSize = 5, bStep = 2, stride = 1, offset = -2;
	const std::vector<uint32_t> seeds = { 777u, 31337u };

	std::vector<std::vector<std::vector<double>>> Ms;
	std::vector<std::vector<Plaintext>> sets, refSets; // one encoding per path; see the test above
	for (uint32_t s : seeds) {
		Ms.push_back(MakeRandomMatrix(n, s, /*band=*/2));
		sets.push_back(PackDiagonals(cc, Ms.back(), rowSize, bStep, stride, offset, /*extended=*/true));
		refSets.push_back(PackDiagonals(cc, Ms.back(), rowSize, bStep, stride, offset, /*extended=*/true));
	}

	auto src = EncryptVec(cc, v1, keys.publicKey);
	auto got = cc->LinearTransformMany(src, rowSize, bStep, sets, stride, offset, /*ext=*/true);
	ASSERT_EQ(got.size(), sets.size());

	for (size_t t = 0; t < sets.size(); ++t) {
		CheckPrecision(cc, got[t], keys.secretKey, HostMatVec(Ms[t], v1), 1e-3);

		auto one = src->Clone();
		cc->LinearTransformInPlace(one, rowSize, bStep, refSets[t], stride, offset, /*ext=*/true);
		auto a = DecryptSlots(cc, got[t], keys.secretKey, kSlots);
		auto b = DecryptSlots(cc, one, keys.secretKey, kSlots);
		for (size_t i = 0; i < a.size(); ++i)
			EXPECT_NEAR(a[i], b[i], kSameResult) << "transform " << t << ", slot " << i;
	}
}

TEST_F(CKKSTest, LinearTransformManySingleSetEqualsSingleCall) {
	// Degenerate batch of one: the Many path must not be a different algorithm.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 8, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 20260830u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto ref = PackDiagonals(cc, M, rowSize, bStep, stride, offset); // one encoding per path
	auto src = EncryptVec(cc, v1, keys.publicKey);

	auto got = cc->LinearTransformMany(src, rowSize, bStep, { pts }, stride, offset);
	ASSERT_EQ(got.size(), 1u);

	auto one = src->Clone();
	cc->LinearTransformInPlace(one, rowSize, bStep, ref, stride, offset);
	auto a = DecryptSlots(cc, got[0], keys.secretKey, kSlots);
	auto b = DecryptSlots(cc, one, keys.secretKey, kSlots);
	for (size_t i = 0; i < a.size(); ++i)
		EXPECT_NEAR(a[i], b[i], kSameResult) << "slot " << i;
}

TEST_F(CKKSTest, LinearTransformManyRejectsBadArguments) {
	// Every set is validated before ANY of them runs: one call is one launch, so a bad set must not
	// leave the batch half-computed. The messages name which set failed.
	const int n = static_cast<int>(kSlots);
	auto M		= MakeRandomMatrix(n, 1u);
	auto good	= PackDiagonals(cc, M, n, 4, 1, 0);
	auto src	= EncryptVec(cc, v1, keys.publicKey);

	EXPECT_THROW(cc->LinearTransformMany(src, n, 4, {}, 1, 0), std::exception);
	EXPECT_THROW(cc->LinearTransformMany(src, 0, 4, { good }, 1, 0), std::exception);
	EXPECT_THROW(cc->LinearTransformMany(src, n, 0, { good }, 1, 0), std::exception);
	EXPECT_THROW(cc->LinearTransformMany(src, n + 1, 4, { good, good }, 1, 0), std::exception);

	// A hole in the SECOND set is caught, not just the first.
	auto withNull = good;
	withNull[3]	  = Plaintext{};
	EXPECT_THROW(cc->LinearTransformMany(src, n, 4, { good, withNull }, 1, 0), std::exception);

	// ext = true against ordinary diagonals: one launch takes one basis, so this is a whole-batch
	// failure and is refused up front on every backend.
	EXPECT_THROW(cc->LinearTransformMany(src, n, 4, { good, good }, 1, 0, /*ext=*/true), std::exception);
}

// ---- 15. Additional operation paths ----

TEST_F(CKKSTest, EvalFastRotationVector) {
	// Vector overload returns one clean rotation per requested index.
	const std::vector<int32_t> indices = { 1, 2 };
	auto ct							   = EncryptVec(cc, v1, keys.publicKey);
	auto precomp					   = cc->EvalFastRotationPrecompute(ct);
	uint32_t m						   = cc->GetCyclotomicOrder();
	auto results					   = cc->EvalFastRotation(ct, indices, m, precomp);
	ASSERT_EQ(results.size(), indices.size());
	// Declare every result an output before the first decrypt: a record/replay backend
	// (haze) executes the recorded program once at the first readback, so all results that
	// will be read back must be declared up front (no-op on CPU/CUDA).
	for (auto& r : results)
		cc->MarkOutput(r);
	for (size_t k = 0; k < indices.size(); ++k) {
		std::vector<double> expected(v1.size());
		for (size_t i = 0; i < v1.size(); ++i)
			expected[i] = v1[(i + indices[k]) % v1.size()];
		CheckPrecision(cc, results[k], keys.secretKey, expected, CKKS_PRECISION);
	}
}

TEST_F(CKKSTest, EvalFastRotationExt) {
	// EvalFastRotationExt yields an extended-basis intermediate (not a plain
	// rotation), so just exercise the CPU dispatch and ensure it succeeds.
	auto ct		 = EncryptVec(cc, v1, keys.publicKey);
	auto precomp = cc->EvalFastRotationPrecompute(ct);
	Ciphertext<DCRTPoly> res;
	EXPECT_NO_THROW(res = cc->EvalFastRotationExt(ct, 1, precomp, true));
	EXPECT_NE(res.get(), nullptr);
}

TEST_F(CKKSTest, AccumulateSumInPlaceWithStart) {
	// The 4-arg overload (start, then doubling): start=1 reduces all 8 slots into
	// slot 0 (= 36). Like the AccumulateSum siblings, only slot 0 is guaranteed —
	// the CUDA backend reduces into slot 0 and leaves the rest 0.
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->AccumulateSumInPlace(ct, kSlots, 1, 1);
	Plaintext out;
	cc->Decrypt(ct, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

TEST_F(CKKSTest, SerializeDeserializeContextRoundTrip) {
	// Exercises the full-context serializer, including the CPU-only device
	// metadata block ("0 { }") written/read by api/Serialize.cpp.
	const std::string path = "/tmp/fideslib_cpu_ctx_roundtrip";
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	CryptoContext<DCRTPoly> cc2;
	ASSERT_TRUE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
	ASSERT_NE(cc2.get(), nullptr);
	EXPECT_EQ(cc2->GetRingDimension(), cc->GetRingDimension());

	std::remove(path.c_str());
	std::remove((path + ".dev").c_str());
}

// ---- 16. EvalMultNoRelin / Relinearize ----

namespace {
// Decrypt ct and return its real slots (length n). Kept lbcrypto-free like EncryptVec/CheckPrecision.
std::vector<double> DecryptVec(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, size_t n) {
	Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	pt->SetLength(n);
	return pt->GetRealPackedValue();
}
} // namespace

TEST_F(CKKSTest, EvalMultNoRelinDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);
	auto q = cc->EvalMultNoRelin(a, b);
	cc->MarkOutput(q);
	CheckPrecision(cc, q, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(q->GetNumElements(), 3u);
}

TEST_F(CKKSTest, RelinearizeInPlaceMatchesEvalMult) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);
	auto q = cc->EvalMultNoRelin(a, b);
	cc->RelinearizeInPlace(q);
	auto p = cc->EvalMult(a, b);
	cc->MarkOutput(q);
	cc->MarkOutput(p);
	auto qv = DecryptVec(cc, q, keys.secretKey, v1.size());
	auto pv = DecryptVec(cc, p, keys.secretKey, v1.size());
	for (size_t i = 0; i < v1.size(); ++i) {
		EXPECT_NEAR(qv[i], expected[i], CKKS_PRECISION) << "slot " << i;
		EXPECT_NEAR(pv[i], expected[i], CKKS_PRECISION) << "slot " << i;
		EXPECT_NEAR(qv[i], pv[i], CKKS_PRECISION) << "slot " << i;
	}
	EXPECT_EQ(q->GetNumElements(), 2u);
}

TEST_F(CKKSTest, Degree2EvalAddDecrypt) {
	// NoRelin(a,b) has no relinearized counterpart here, so add it to itself: 2*v1*v2.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = 2.0 * v1[i] * v2[i];
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d1	 = cc->EvalMultNoRelin(a, b);
	auto d2	 = cc->EvalMultNoRelin(a, b);
	auto sum = cc->EvalAdd(d1, d2);
	cc->MarkOutput(sum);
	CheckPrecision(cc, sum, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2EvalSubDecrypt) {
	// Distinct products, so the expected difference is nonzero: subtracting a ciphertext from an
	// identical one yields zeros whatever the tail components hold, which proves nothing.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v1[i] - v1[i] * v2[i];
	auto a	  = EncryptVec(cc, v1, keys.publicKey);
	auto b	  = EncryptVec(cc, v2, keys.publicKey);
	auto d1	  = cc->EvalMultNoRelin(a, a);
	auto d2	  = cc->EvalMultNoRelin(a, b);
	auto diff = cc->EvalSub(d1, d2);
	cc->MarkOutput(diff);
	CheckPrecision(cc, diff, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2EvalNegateDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = -(v1[i] * v2[i]);
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d	 = cc->EvalMultNoRelin(a, b);
	auto neg = cc->EvalNegate(d);
	cc->MarkOutput(neg);
	CheckPrecision(cc, neg, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2EvalMultPtDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i] * v1[i];
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d	 = cc->EvalMultNoRelin(a, b);
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto res = cc->EvalMult(d, pt1);
	cc->MarkOutput(res);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2EvalMultScalarDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = 0.5 * v1[i] * v2[i];
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d2	 = cc->EvalMultNoRelin(a, b);
	auto d2b = cc->EvalMultNoRelin(a, b);
	auto res = cc->EvalMult(d2, 0.5);
	cc->EvalMultInPlace(d2b, 0.5);
	cc->MarkOutput(res);
	cc->MarkOutput(d2b);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
	CheckPrecision(cc, d2b, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2EvalAddPtDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i] + v1[i];
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d2	 = cc->EvalMultNoRelin(a, b);
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto res = cc->EvalAdd(d2, pt);
	cc->MarkOutput(res);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(res->GetNumElements(), 3u);
}

TEST_F(CKKSTest, Degree2EvalAddScalarDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i] + 0.25;
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d2	 = cc->EvalMultNoRelin(a, b);
	auto d2b = cc->EvalMultNoRelin(a, b);
	auto res = cc->EvalAdd(d2, 0.25);
	cc->EvalAddInPlace(d2b, 0.25);
	cc->MarkOutput(res);
	cc->MarkOutput(d2b);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
	CheckPrecision(cc, d2b, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(res->GetNumElements(), 3u);
	EXPECT_EQ(d2b->GetNumElements(), 3u);
}

TEST_F(CKKSTest, Degree2EvalSubPtDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i] - v1[i];
	auto a	 = EncryptVec(cc, v1, keys.publicKey);
	auto b	 = EncryptVec(cc, v2, keys.publicKey);
	auto d2	 = cc->EvalMultNoRelin(a, b);
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto res = cc->EvalSub(d2, pt);
	cc->MarkOutput(res);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(res->GetNumElements(), 3u);
}

TEST_F(CKKSTest, Degree2EvalSubScalarDecrypt) {
	// Two orders: ct - scalar leaves cv[0] shifted; scalar - ct negates all
	// components (cv[0] and cv[1]) before adding, so this also covers the
	// negate-all-components path on a degree-2 ciphertext.
	std::vector<double> expectedCtMinusScalar(v1.size());
	std::vector<double> expectedScalarMinusCt(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) {
		expectedCtMinusScalar[i] = v1[i] * v2[i] - 0.25;
		expectedScalarMinusCt[i] = 0.25 - v1[i] * v2[i];
	}
	auto a	  = EncryptVec(cc, v1, keys.publicKey);
	auto b	  = EncryptVec(cc, v2, keys.publicKey);
	auto d2	  = cc->EvalMultNoRelin(a, b);
	auto d2b  = cc->EvalMultNoRelin(a, b);
	auto res  = cc->EvalSub(d2, 0.25);
	auto res2 = cc->EvalSub(0.25, d2b);
	cc->MarkOutput(res);
	cc->MarkOutput(res2);
	CheckPrecision(cc, res, keys.secretKey, expectedCtMinusScalar, CKKS_PRECISION);
	CheckPrecision(cc, res2, keys.secretKey, expectedScalarMinusCt, CKKS_PRECISION);
	EXPECT_EQ(res->GetNumElements(), 3u);
	EXPECT_EQ(res2->GetNumElements(), 3u);
}

TEST_F(CKKSTest, Degree2RescaleDecrypt) {
	// api Rescale mod-reduces under EVERY scaling technique on all backends (BITCOMPAT O3,
	// resolved): the CPU engine routes the AUTO techniques through OpenFHE's internal ModReduce
	// (whose public Rescale only acts under FIXEDMANUAL), matching CUDA and haze. A Rescale
	// therefore always consumes a level, on every backend.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto a				= EncryptVec(cc, v1, keys.publicKey);
	auto b				= EncryptVec(cc, v2, keys.publicKey);
	auto d				= cc->EvalMultNoRelin(a, b);
	size_t levelBefore	= d->GetLevel();
	cc->RescaleInPlace(d);
	cc->MarkOutput(d);
	CheckPrecision(cc, d, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(d->GetLevel(), levelBefore + 1);
}

TEST_F(CKKSTest, Degree2EvalAddManyDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = 3.0 * v1[i] * v2[i];
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);
	std::vector<Ciphertext<DCRTPoly>> ds;
	for (int k = 0; k < 3; ++k)
		ds.push_back(cc->EvalMultNoRelin(a, b));
	auto sum = cc->EvalAddMany(ds);
	cc->MarkOutput(sum);
	CheckPrecision(cc, sum, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2DotProductAccumulate) {
	// 4 deterministic (lhs, rhs) pairs built from the fixture's v1/v2.
	const std::vector<std::vector<double>> lv = { v1, v2, v1, v2 };
	const std::vector<std::vector<double>> rv = { v2, v1, v1, v2 };

	std::vector<double> expected(v1.size(), 0.0);
	for (size_t k = 0; k < 4; ++k)
		for (size_t i = 0; i < v1.size(); ++i)
			expected[i] += lv[k][i] * rv[k][i];

	std::vector<Ciphertext<DCRTPoly>> lhs, rhs;
	for (size_t k = 0; k < 4; ++k) {
		lhs.push_back(EncryptVec(cc, lv[k], keys.publicKey));
		rhs.push_back(EncryptVec(cc, rv[k], keys.publicKey));
	}

	// Lazy: accumulate degree-2 products, relinearize once at the end.
	auto lazy = cc->EvalMultNoRelin(lhs[0], rhs[0]);
	for (size_t k = 1; k < 4; ++k) {
		auto term = cc->EvalMultNoRelin(lhs[k], rhs[k]);
		cc->EvalAddInPlace(lazy, term);
	}
	cc->RelinearizeInPlace(lazy);

	// Eager: relinearize after every product.
	auto eager = cc->EvalMult(lhs[0], rhs[0]);
	for (size_t k = 1; k < 4; ++k) {
		auto term = cc->EvalMult(lhs[k], rhs[k]);
		cc->EvalAddInPlace(eager, term);
	}

	cc->MarkOutput(lazy);
	cc->MarkOutput(eager);
	auto lazyVals  = DecryptVec(cc, lazy, keys.secretKey, v1.size());
	auto eagerVals = DecryptVec(cc, eager, keys.secretKey, v1.size());
	for (size_t i = 0; i < v1.size(); ++i) {
		EXPECT_NEAR(lazyVals[i], expected[i], 1e-5) << "slot " << i;
		EXPECT_NEAR(eagerVals[i], expected[i], 1e-5) << "slot " << i;
		EXPECT_NEAR(lazyVals[i], eagerVals[i], 1e-5) << "slot " << i;
	}
}

TEST_F(CKKSTest, Degree2CloneDecrypt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);
	auto p = cc->EvalMultNoRelin(a, b);
	auto q = p->Clone();
	cc->MarkOutput(p);
	cc->MarkOutput(q);
	CheckPrecision(cc, p, keys.secretKey, expected, CKKS_PRECISION);
	CheckPrecision(cc, q, keys.secretKey, expected, CKKS_PRECISION);
	EXPECT_EQ(q->GetNumElements(), 3u);
}

TEST_F(CKKSTest, RepeatedNoRelinRelinearizeRecyclesMemory) {
	// Pins that repeated NoRelin -> Relinearize cycles keep the degree contract (3 then 2) on every
	// cycle and that the final result still decrypts correctly, exercising acquire/release of the
	// third component across many cycles. 25 cycles on fresh encrypts.
	static constexpr int kCycles = 25;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];

	Ciphertext<DCRTPoly> last;
	for (int k = 0; k < kCycles; ++k) {
		auto a = EncryptVec(cc, v1, keys.publicKey);
		auto b = EncryptVec(cc, v2, keys.publicKey);
		auto q = cc->EvalMultNoRelin(a, b);
		ASSERT_EQ(q->GetNumElements(), 3u) << "cycle " << k;
		cc->RelinearizeInPlace(q);
		ASSERT_EQ(q->GetNumElements(), 2u) << "cycle " << k;
		last = q; // every earlier cycle's ciphertexts are released here
	}
	// Only the final cycle's value is read back, so the loop stays one program on record/replay
	// backends instead of 25 flushes.
	cc->MarkOutput(last);
	CheckPrecision(cc, last, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Degree2UnsupportedOpsThrow) {
	// Every sub-check builds its own fresh degree-2 ciphertext, so a throw in one
	// sub-check cannot leave a stale ciphertext for the next. No readback anywhere:
	// these are all expected to fail before producing a result.
	auto a = EncryptVec(cc, v1, keys.publicKey);
	auto b = EncryptVec(cc, v2, keys.publicKey);

	{
		auto d2 = cc->EvalMultNoRelin(a, b);
		EXPECT_THROW(cc->EvalRotate(d2, 1), std::exception);
	}
	{
		auto d2 = cc->EvalMultNoRelin(a, b);
		EXPECT_THROW(cc->EvalRotateInPlace(d2, 1), std::exception);
	}
	{
		auto d2 = cc->EvalMultNoRelin(a, b);
		EXPECT_THROW(cc->EvalSquare(d2), std::exception);
	}
	{
		auto d2 = cc->EvalMultNoRelin(a, b);
		EXPECT_THROW(cc->EvalMult(d2, d2), std::exception);
	}
	{
		auto d2 = cc->EvalMultNoRelin(a, b);
		EXPECT_THROW(cc->EvalMultNoRelin(d2, b), std::exception);
	}
}

// ===========================================================================
// Sparse-ternary secret keys: the mod-eval approximation must shrink
// ===========================================================================
//
// A sparse ternary secret key has low Hamming weight, so the argument range of the bootstrap's
// modular reduction is smaller and the Chebyshev approximation needs fewer terms. OpenFHE encodes
// that as three tables (ckksrns-fhe.h): g_coefficientsUniform is degree 88 with K_UNIFORM=512 and
// R_UNIFORM=6 double-angle iterations, g_coefficientsSparse is degree 44 with R_SPARSE=3, and
// g_coefficientsSparseEncapsulated is degree 32, also R_SPARSE. Fed through GetDepthByDegree that
// is 8+6=14 mod-eval levels for uniform and 7+3=10 for either sparse form.
//
// These tests are the evidence that the CUDA backend genuinely runs the smaller approximation
// rather than merely accepting the key distribution.

// Tier 1: the table selection itself, on the host. No device involved, so this pins the numbers
// independently of any backend and fails loudly if the OpenFHE tables are ever regenerated.
TEST(SparseSecretModEval, LevelsPerKeyDistribution) {
	EXPECT_EQ(bootstrapModEvalLevels(UNIFORM_TERNARY), 14) << "degree-88 g_coefficientsUniform + R_UNIFORM";
	EXPECT_EQ(bootstrapModEvalLevels(SPARSE_TERNARY), 10) << "degree-44 g_coefficientsSparse + R_SPARSE";
	EXPECT_EQ(bootstrapModEvalLevels(SPARSE_ENCAPSULATED), 10) << "degree-32 table, same depth bucket as sparse";
	EXPECT_EQ(bootstrapModEvalLevels(UNIFORM_TERNARY) - bootstrapModEvalLevels(SPARSE_TERNARY), 4)
	  << "a sparse main key must free exactly four mod-eval levels";
}

// Shared configuration for the end-to-end sparse tests. This is the compat suite's bootstrap
// configuration, which is the one already known to work on the GPU: FLEXIBLEAUTO, N=4096,
// depth 25, level budget {3,3}. Depth fits both distributions (14+3+3=20 and 10+3+3=16 <= 25).
namespace sparse_modeval {

static constexpr uint32_t kDepth   = 25;
static constexpr uint32_t kRingDim = 1u << 12;
static constexpr uint32_t kSlots   = 8;
static constexpr uint32_t kBudget  = 3;

static Backend RequestedBackend() {
	switch (GetTestBackend()) {
	case TestBackend::CUDA: return Backend::CUDA;
	case TestBackend::HAZE: return Backend::HAZE;
	default: return Backend::CPU;
	}
}

// Guard the silent-fallback failure mode. GetTestBackend() resolves to CPU whenever
// FIDESLIB_TEST_BACKEND names a backend this build lacks, so asking for cuda on a CPU-only build
// would run these tests entirely on OpenFHE and still report a pass - the level accounting below
// adapts per backend, so nothing else would notice. These tests exist to prove what the device
// does, so an unmet request is a failure, not a fallback.
static void RequireRequestedBackend(CryptoContext<DCRTPoly>& cc) {
	if (const char* want = std::getenv("FIDESLIB_TEST_BACKEND"); want != nullptr) {
		const std::string requested(want);
		if (requested == "cuda")
			EXPECT_TRUE(IsBackendAvailable(Backend::CUDA))
			  << "FIDESLIB_TEST_BACKEND=cuda but this build has no CUDA backend; refusing to fall back to the CPU";
		else if (requested == "haze")
			EXPECT_TRUE(IsBackendAvailable(Backend::HAZE))
			  << "FIDESLIB_TEST_BACKEND=haze but this build has no haze backend; refusing to fall back to the CPU";
	}
	// And the engine actually constructed is the one asked for.
	EXPECT_EQ(cc->engine_->backend(), RequestedBackend()) << "context was built on a different backend than requested";
}

// Build a bootstrap-ready context for one secret-key distribution, on whichever backend the
// environment selected. Call order is forced: EvalBootstrapSetup and EvalBootstrapKeyGen both
// throw once the context is loaded, and CudaEngine::loadContext needs the bootstrap keys.
static CryptoContext<DCRTPoly> MakeContext(SecretKeyDist dist, KeyPair<DCRTPoly>& keys) {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetSecurityLevel(HEStd_NotSet);
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetSecretKeyDist(dist);
	params.SetBackend(RequestedBackend());
	if (RequestedBackend() == Backend::HAZE)
		params.SetReducedNoise(true);

	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);
	keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalBootstrapSetup({ kBudget, kBudget }, { 0, 0 }, kSlots, 0);
	cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
	if (RequestedBackend() != Backend::CPU)
		cc->LoadContext(keys.publicKey);
	RequireRequestedBackend(cc);
	return cc;
}

} // namespace sparse_modeval

namespace sparse_modeval {

// Bootstrap one depleted ciphertext and report the level of the refreshed result plus the worst
// slot error. The input is encrypted at the deepest level EvalBootstrap accepts,
// kDepth - (kBudget + 2); a non-depleted input is returned unchanged, which still decrypts
// correctly and would make every assertion below vacuous.
static size_t BootstrapAndReportLevel(SecretKeyDist dist, double& maxErr) {
	KeyPair<DCRTPoly> keys;
	auto cc = MakeContext(dist, keys);

	const std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	auto pt						= cc->MakeCKKSPackedPlaintext(x, 1, kDepth - (kBudget + 2), nullptr, kSlots);
	auto ct						= cc->Encrypt(pt, keys.publicKey);

	const size_t levelBefore = ct->GetLevel();
	auto refreshed			 = cc->EvalBootstrap(ct);
	EXPECT_LT(refreshed->GetLevel(), levelBefore) << "bootstrap gained no levels: it returned the input unchanged, so this measured nothing";

	Plaintext out;
	cc->Decrypt(refreshed, keys.secretKey, &out);
	out->SetLength(x.size());
	auto vals = out->GetRealPackedValue();
	maxErr	  = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
		maxErr = std::max(maxErr, std::abs(vals[i] - x[i]));

	return refreshed->GetLevel();
}

} // namespace sparse_modeval

// Tier 2: the observable, end to end. At otherwise identical parameters a sparse main key must
// leave the refreshed ciphertext four levels shallower than a uniform one, because its mod-eval
// Chebyshev approximation is four levels cheaper. This is the black-box proof that the device ran
// the smaller approximation and not merely that it accepted the key distribution -  no
// instrumentation, no private accessors, just the level the caller can see.
TEST(SparseSecretModEval, SparseKeyFreesFourLevelsEndToEnd) {
	using namespace sparse_modeval;

	double uniformErr = 0.0, sparseErr = 0.0;
	const size_t uniformLevel = BootstrapAndReportLevel(UNIFORM_TERNARY, uniformErr);
	const size_t sparseLevel  = BootstrapAndReportLevel(SPARSE_TERNARY, sparseErr);

	EXPECT_LT(sparseLevel, uniformLevel) << "sparse bootstrap consumed at least as much depth as uniform: "
										 << "sparse=" << sparseLevel << " uniform=" << uniformLevel;
	EXPECT_EQ(uniformLevel - sparseLevel, bootstrapModEvalLevels(UNIFORM_TERNARY) - bootstrapModEvalLevels(SPARSE_TERNARY))
	  << "level delta does not match the mod-eval table difference: sparse=" << sparseLevel << " uniform=" << uniformLevel;

	// Pin the absolute depths too, not just their difference -  asserting only the delta would still
	// pass if both arms drifted together. A bootstrap reserves CtS + EvalMod + StC; the executed
	// chain lands under that reservation by a backend-dependent amount, measured here rather than
	// derived. The device backends come in 1 level under (uniform 19, sparse 15 at kBudget=3); the
	// CPU backend comes in 4 under (16 and 12) because OpenFheEngine::bootstrapSetupPolicy routes
	// modEvalLevels into OpenFHE's BTSlotsEncoding slot and passes -1 for modevallevels, so
	// ckksrns-fhe.cpp takes the branch that derives lDec from the budget alone and ignores the
	// approximation depth entirely. That divergence is pre-existing and flagged in place; it changes
	// the offset on both arms equally, which is why the delta above is the real assertion.
	const size_t budgetSlack = (RequestedBackend() == Backend::CPU) ? 4 : 1;
	EXPECT_EQ(uniformLevel, bootstrapModEvalLevels(UNIFORM_TERNARY) + 2 * kBudget - budgetSlack) << "uniform bootstrap consumed an unexpected depth; got " << uniformLevel;
	EXPECT_EQ(sparseLevel, bootstrapModEvalLevels(SPARSE_TERNARY) + 2 * kBudget - budgetSlack) << "sparse bootstrap consumed an unexpected depth; got " << sparseLevel;

	// Both must still be correct -  a cheaper approximation that does not decrypt proves nothing.
	EXPECT_LT(uniformErr, CKKS_BOOTSTRAP_PRECISION) << "uniform bootstrap lost precision";
	EXPECT_LT(sparseErr, CKKS_BOOTSTRAP_PRECISION) << "sparse bootstrap lost precision";
}

// The sparse path must also simply work: construct, bootstrap, decrypt. Before the CUDA guard was
// removed this threw at GenCryptoContext on the CUDA backend.
TEST(SparseSecretModEval, SparseTernaryBootstrapIsCorrect) {
	using namespace sparse_modeval;

	double err				= 0.0;
	const size_t afterLevel = BootstrapAndReportLevel(SPARSE_TERNARY, err);
	EXPECT_LT(err, CKKS_BOOTSTRAP_PRECISION) << "sparse-ternary bootstrap decrypted outside tolerance";
	EXPECT_GT(afterLevel, 0u);
}


