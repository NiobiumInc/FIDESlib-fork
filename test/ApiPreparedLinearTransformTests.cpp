// Prepared linear transforms — api suite.
//
// WHAT THIS PINS. PrepareLinearTransform freezes (shape + diagonals) into a handle that owns the
// backend's per-call setup, so that the device pointer table the batched MAC kernel reads is built
// ONCE per resident diagonal group instead of being rebuilt and re-uploaded on every giant step.
// Two properties have to hold for that to be a legal optimisation, and this file asserts both:
//
//   1. EQUALITY. The prepared overloads produce BIT-IDENTICAL ciphertexts to the vector overloads
//      on the same inputs. On a host backend that is structural — Engine's reference prepare
//      forwards to the vector entry point with the diagonals the handle holds — and these tests
//      are what keeps it structural as the CUDA override lands beside it.
//
//   2. STALENESS IS A THROW. A handle whose diagonals have been re-encoded or released must throw,
//      not compute. A prepared table holds raw device pointers into the diagonals; reading a stale
//      one is a WRONG ANSWER, not a slowdown, so there is no fallback path to take.
//
// Runs on any backend (default CPU): set FIDESLIB_TEST_BACKEND=cuda on a CUDA build to exercise
// the CUDA override of the same asserts.

#include <gtest/gtest.h>
#include <openfhe.h>

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

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
bool TestUseDevice() {
	return GetTestBackend() != TestBackend::CPU;
}

// Exact ciphertext equality: scaling metadata + every RNS limb of both polynomials.
void ExpectSameCiphertext(Ciphertext<DCRTPoly>& a, Ciphertext<DCRTPoly>& b) {
	ASSERT_EQ(a->GetNoiseScaleDeg(), b->GetNoiseScaleDeg());
	ASSERT_EQ(a->GetScalingFactor(), b->GetScalingFactor());
	ASSERT_EQ(a->GetLevel(), b->GetLevel());
	const auto& ea = a->GetElements();
	const auto& eb = b->GetElements();
	ASSERT_EQ(ea.size(), eb.size());
	for (size_t j = 0; j < ea.size(); ++j) {
		ASSERT_EQ(ea[j].GetAllElements().size(), eb[j].GetAllElements().size()) << "component " << j;
		for (size_t i = 0; i < ea[j].GetAllElements().size(); ++i) {
			ASSERT_EQ(ea[j].GetAllElements()[i].GetValues(), eb[j].GetAllElements()[i].GetValues()) << "component " << j << " tower " << i;
		}
	}
}

std::vector<double> HostRot(const std::vector<double>& v, int r) {
	const int n = static_cast<int>(v.size());
	std::vector<double> out(v.size());
	for (int i = 0; i < n; ++i)
		out[static_cast<size_t>(i)] = v[static_cast<size_t>((((i + r) % n) + n) % n)];
	return out;
}

std::vector<std::vector<double>> MakeRandomMatrix(int n, uint32_t seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::vector<std::vector<double>> M(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			M[static_cast<size_t>(i)][static_cast<size_t>(j)] = dist(rng);
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

// Diagonal packing exactly as CryptoContextImpl::LinearTransformInPlace documents it.
std::vector<Plaintext> PackDiagonals(CryptoContext<DCRTPoly>& cc, const std::vector<std::vector<double>>& M, int rowSize, int bStep, int stride, int offset) {
	const int n		 = static_cast<int>(M.size());
	const auto ptRot = CryptoContextImpl<DCRTPoly>::GetLinearTransformPlaintextRotationIndices(rowSize, bStep, stride, offset);
	std::vector<Plaintext> pts;
	pts.reserve(static_cast<size_t>(rowSize));
	for (int k = 0; k < rowSize; ++k) {
		const int r = k * stride + offset;
		std::vector<double> d(static_cast<size_t>(n));
		for (int i = 0; i < n; ++i)
			d[static_cast<size_t>(i)] = M[static_cast<size_t>(i)][static_cast<size_t>((((i + r) % n) + n) % n)];
		pts.push_back(cc->MakeCKKSPackedPlaintext(HostRot(d, ptRot[static_cast<size_t>(k)])));
	}
	return pts;
}

class PreparedLtTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 5;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { -0.5, 1.5, 0.25, -2.0, 3.5, 0.75, -1.25, 2.25 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		if (GetTestBackend() == TestBackend::CUDA)
			params.SetBackend(Backend::CUDA);
		else if (GetTestBackend() == TestBackend::HAZE) {
			params.SetBackend(Backend::HAZE);
			params.SetReducedNoise(true);
		}
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

	Ciphertext<DCRTPoly> Encrypt(const std::vector<double>& v) {
		auto pt = cc->MakeCKKSPackedPlaintext(v);
		return cc->Encrypt(pt, keys.publicKey);
	}
};

} // namespace

// ---- 1. Equality with the vector overloads ----

TEST_F(PreparedLtTest, PreparedSingleEqualsVectorOverload) {
	// gStep = 2: the backwards Horner fold and its bStep*stride rotation are live, which is the
	// shape the application's matmul actually runs.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 20260921u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);

	// ONE encryption, cloned: CKKS encryption is randomised, so two fresh encryptions of the same
	// vector are different ciphertexts and "bit-identical" would be untestable against them.
	auto src	   = Encrypt(v1);
	auto viaVector = src->Clone();
	cc->LinearTransformInPlace(viaVector, rowSize, bStep, pts, stride, offset);

	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);
	ASSERT_TRUE(prepared != nullptr);
	auto viaPrepared = src->Clone();
	cc->LinearTransformInPlace(viaPrepared, prepared);

	ExpectSameCiphertext(viaVector, viaPrepared);
	// ...and it is the right answer, not merely the same wrong one.
	Plaintext out;
	cc->Decrypt(viaPrepared, keys.secretKey, &out);
	out->SetLength(v1.size());
	auto got	  = out->GetRealPackedValue();
	auto expected = HostMatVec(M, v1);
	for (size_t i = 0; i < expected.size(); ++i)
		EXPECT_NEAR(got[i], expected[i], 1e-3) << "slot " << i;
}

TEST_F(PreparedLtTest, PreparedSingleEqualsVectorOverloadAcrossShapes) {
	// Every branch the single transform has: gStep == 1, a short last giant step, and a non-zero
	// offset. Random diagonals in each, so equality is not an artefact of one matrix.
	struct Shape {
		int rowSize, bStep, stride, offset;
		uint32_t seed;
	};
	const std::vector<Shape> shapes = {
		{ 8, 8, 1, 0, 11u },  // one giant step, no fold
		{ 8, 3, 1, 0, 22u },  // gStep = 3 with a 2-wide last giant step (null padding)
		{ 8, 2, 1, -2, 33u }, // non-zero offset: the final rotation branch
	};
	for (const auto& s : shapes) {
		auto M	 = MakeRandomMatrix(static_cast<int>(kSlots), s.seed);
		auto pts = PackDiagonals(cc, M, s.rowSize, s.bStep, s.stride, s.offset);

		auto src	   = Encrypt(v2);
		auto viaVector = src->Clone();
		cc->LinearTransformInPlace(viaVector, s.rowSize, s.bStep, pts, s.stride, s.offset);

		auto viaPrepared = src->Clone();
		cc->LinearTransformInPlace(viaPrepared, cc->PrepareLinearTransform(pts, s.rowSize, s.bStep, s.stride, s.offset));

		SCOPED_TRACE("rowSize=" + std::to_string(s.rowSize) + " bStep=" + std::to_string(s.bStep) + " offset=" + std::to_string(s.offset));
		ExpectSameCiphertext(viaVector, viaPrepared);
	}
}

TEST_F(PreparedLtTest, PreparedManyEqualsVectorMany) {
	// The batched shape the application runs: several independent transforms of ONE source.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	std::vector<std::vector<std::vector<double>>> Ms;
	std::vector<std::vector<Plaintext>> sets;
	for (uint32_t t = 0; t < 3; ++t) {
		Ms.push_back(MakeRandomMatrix(n, 700u + t));
		sets.push_back(PackDiagonals(cc, Ms.back(), rowSize, bStep, stride, offset));
	}
	// ONE handle for the whole batch: that is what keeps the batched call one call.
	auto prepared = cc->PrepareLinearTransformMany(sets, rowSize, bStep, stride, offset);
	ASSERT_EQ(prepared->Sets(), sets.size());

	auto src	   = Encrypt(v1);
	auto viaVector = cc->LinearTransformMany(src->Clone(), rowSize, bStep, sets, stride, offset);

	auto viaPrepared = cc->LinearTransformMany(src->Clone(), prepared);

	ASSERT_EQ(viaVector.size(), sets.size());
	ASSERT_EQ(viaPrepared.size(), sets.size());
	for (size_t t = 0; t < sets.size(); ++t) {
		SCOPED_TRACE("transform " + std::to_string(t));
		ExpectSameCiphertext(viaVector[t], viaPrepared[t]);
	}
}

TEST_F(PreparedLtTest, PreparedHandleIsReusableAcrossCalls) {
	// The whole point of the handle: prepare once, call many times. Every call must equal the
	// vector overload, not just the first — a table that is consumed or invalidated by use would
	// pass the single-call test and fail here.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M		  = MakeRandomMatrix(n, 5150u);
	auto pts	  = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);

	for (const auto& v : { v1, v2 }) {
		for (int rep = 0; rep < 2; ++rep) {
			auto src	   = Encrypt(v);
			auto viaVector = src->Clone();
			cc->LinearTransformInPlace(viaVector, rowSize, bStep, pts, stride, offset);
			auto viaPrepared = src->Clone();
			cc->LinearTransformInPlace(viaPrepared, prepared);
			SCOPED_TRACE("rep " + std::to_string(rep));
			ExpectSameCiphertext(viaVector, viaPrepared);
		}
	}
}

TEST_F(PreparedLtTest, PreparedHandleKeepsItsDiagonalsAlive) {
	// The handle holds the diagonals BY SHARED HANDLE, so a caller that drops its own vector (the
	// application's diagonal pool evicting an entry while a prepared transform still references it)
	// cannot free the buffers the table points at.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 8675309u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);

	auto src	   = Encrypt(v1);
	auto viaVector = src->Clone();
	cc->LinearTransformInPlace(viaVector, rowSize, bStep, pts, stride, offset);

	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);
	ASSERT_EQ(prepared->Diagonals().size(), static_cast<size_t>(rowSize));
	PlaintextImpl* first = pts[0].get();
	pts.clear(); // the caller lets go
	ASSERT_EQ(prepared->Diagonals()[0].get(), first);
	ASSERT_TRUE(prepared->Diagonals()[0].use_count() >= 1);

	auto viaPrepared = src->Clone();
	cc->LinearTransformInPlace(viaPrepared, prepared);
	ExpectSameCiphertext(viaVector, viaPrepared);
}

// ---- 2. Staleness is a throw, never a stale read ----

TEST_F(PreparedLtTest, PreparedHandleThrowsOnReEncodedDiagonal) {
	// A diagonal RE-ENCODED IN PLACE: the same PlaintextImpl, different contents. This is the
	// aliasing hazard the identity check exists for — the handle still holds a live, valid
	// plaintext, so nothing crashes and nothing is null; it is simply no longer the plaintext the
	// table was built from, and using it would silently compute a different matrix.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 314159u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);

	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);

	auto other			 = MakeRandomMatrix(n, 271828u);
	auto replacement	 = PackDiagonals(cc, other, rowSize, bStep, stride, offset);
	pts[2]->host		 = replacement[2]->host; // in-place re-encode of diagonal 2
	pts[2]->device.reset();

	auto ct = Encrypt(v1);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, prepared), std::exception);

	auto src = Encrypt(v1);
	EXPECT_THROW(cc->LinearTransformMany(src, prepared), std::exception);
}

TEST_F(PreparedLtTest, PreparedHandleThrowsOnReleasedDiagonal) {
	// A diagonal whose payload was RELEASED — the state a backend free leaves behind. The shared
	// handle means the object is still there; its buffers are not. The call must throw rather than
	// read what used to be at those addresses.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M	 = MakeRandomMatrix(n, 161803u);
	auto pts = PackDiagonals(cc, M, rowSize, bStep, stride, offset);

	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);
	pts[rowSize - 1]->host.reset();
	pts[rowSize - 1]->device.reset();

	auto ct = Encrypt(v1);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, prepared), std::exception);
}

TEST_F(PreparedLtTest, PreparedHandleAcceptsAnUntouchedDiagonalSet) {
	// The negative control for the two throws above: the identity check must not be so eager that
	// an ordinary reuse trips it. Encrypting, transforming and decrypting around the handle leaves
	// the diagonals alone, so the second call still runs.
	const int n = static_cast<int>(kSlots), rowSize = n, bStep = 4, stride = 1, offset = 0;
	auto M		  = MakeRandomMatrix(n, 42u);
	auto pts	  = PackDiagonals(cc, M, rowSize, bStep, stride, offset);
	auto prepared = cc->PrepareLinearTransform(pts, rowSize, bStep, stride, offset);

	auto a = Encrypt(v1);
	EXPECT_NO_THROW(cc->LinearTransformInPlace(a, prepared));
	Plaintext scratch;
	cc->Decrypt(a, keys.secretKey, &scratch);
	auto b = Encrypt(v2);
	EXPECT_NO_THROW(cc->LinearTransformInPlace(b, prepared));
}

// ---- 3. Argument checking ----

TEST_F(PreparedLtTest, PrepareRejectsBadArguments) {
	const int n = static_cast<int>(kSlots);
	auto M		= MakeRandomMatrix(n, 1u);
	auto pts	= PackDiagonals(cc, M, n, 4, 1, 0);

	EXPECT_THROW(cc->PrepareLinearTransform(pts, 0, 4), std::exception);	 // rowSize
	EXPECT_THROW(cc->PrepareLinearTransform(pts, n, 0), std::exception);	 // bStep
	EXPECT_THROW(cc->PrepareLinearTransform(pts, n + 1, 4), std::exception); // too few diagonals

	auto withNull = pts;
	withNull[3]	  = Plaintext{};
	EXPECT_THROW(cc->PrepareLinearTransform(withNull, n, 4), std::exception);

	// ext is a REQUIREMENT, not a hint, at prepare time too.
	EXPECT_THROW(cc->PrepareLinearTransform(pts, n, 4, 1, 0, /*ext=*/true), std::exception);
}

TEST_F(PreparedLtTest, PreparedCallsRejectNullAndTheWrongHandleKind) {
	const int n = static_cast<int>(kSlots);
	auto M		= MakeRandomMatrix(n, 2u);
	auto pts	= PackDiagonals(cc, M, n, 4, 1, 0);
	auto single = cc->PrepareLinearTransform(pts, n, 4, 1, 0);
	auto batch	= cc->PrepareLinearTransformMany({ pts, pts }, n, 4, 1, 0);

	auto ct = Encrypt(v1);
	EXPECT_THROW(cc->LinearTransformInPlace(ct, PreparedLinearTransform{}), std::exception);
	// A batch handle is not a single-transform handle: the in-place call would have to pick one of
	// its sets, and picking silently is how a caller gets the wrong matrix.
	EXPECT_THROW(cc->LinearTransformInPlace(ct, batch), std::exception);
	EXPECT_THROW(batch->Diagonals(), std::exception);

	auto src = Encrypt(v1);
	EXPECT_THROW(cc->LinearTransformMany(src, PreparedLinearTransform{}), std::exception);
	EXPECT_THROW(cc->PrepareLinearTransformMany({}, n, 4, 1, 0), std::exception);
	// ...but a single handle IS a one-wide batch, which is the shape the application's
	// width-1 matmul group uses.
	EXPECT_NO_THROW(cc->LinearTransformMany(src, single));
}
