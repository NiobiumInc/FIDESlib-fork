// api-vs-OpenFHE parity suite.
//
// Drives the public fideslib api on the CUDA or haze backend and asserts the resulting ciphertext
// is bit-for-bit identical to the same computation run directly through OpenFHE (the oracle). This
// is the api-level counterpart of the raw-GPU parity tests in OpenFheInterfaceTests.cu: instead of
// hand-plumbing FIDESlib::CKKS device objects, every op goes through CryptoContextImpl, and the
// result is recovered with CryptoContextImpl::RecoverHostCiphertext (the backend readback seam).
//
// Because it compares the active backend against OpenFHE, it is only meaningful on a non-CPU
// backend: on the CPU backend the api *is* OpenFHE, so the comparison is trivial and the fixture
// skips. Set FIDESLIB_TEST_BACKEND=cuda (CUDA build) or =haze (haze build) to run it against that
// engine.
//
// Oracle note: when several ops are chained, the api runs them on the device and only the final
// result is synced back, so intermediate host shadows are stale. The oracle therefore replays the
// whole chain in OpenFHE from the freshly-encrypted inputs (never from a mid-chain api ciphertext).

#include <gtest/gtest.h>
#include <openfhe.h>

#include <any>
#include <cmath>
#include <string>
#include <vector>

#include "TestEngineConfig.hpp" // FIDESlib::Testing::{ConfigureTestEngine, GetTestBackend, TestBackend}
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;
using FIDESlib::Testing::GetTestBackend;
using FIDESlib::Testing::TestBackend;

// Exact ciphertext equality: scaling metadata + every RNS limb of both polynomials. Mirrors the
// ASSERT_EQ_CIPHERTEXT in test/ParametrizedTest.cuh, minus the per-limb debug printing.
#define ASSERT_EQ_CIPHERTEXT(ct1, ct2)                                                                                                                   \
	do {                                                                                                                                                 \
		ASSERT_EQ((ct1)->GetNoiseScaleDeg(), (ct2)->GetNoiseScaleDeg());                                                                                 \
		ASSERT_EQ((ct1)->GetScalingFactor(), (ct2)->GetScalingFactor());                                                                                 \
		ASSERT_EQ((ct1)->GetEncodingType(), (ct2)->GetEncodingType());                                                                                   \
		ASSERT_EQ((ct1)->GetElements().size(), (ct2)->GetElements().size());                                                                             \
		for (size_t j = 0; j < (ct1)->GetElements().size(); ++j) {                                                                                       \
			ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().size(), (ct2)->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < (ct1)->GetElements().at(j).GetAllElements().size(); ++i) {                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                     \
				  (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues(), (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                            \
		}                                                                                                                                                \
	} while (0)

// ASSERT_EQ_CIPHERTEXT, plus pinning the expected component count up front (e.g. degree-2 vs.
// degree-3 ciphertexts after a multiplication that has not been relinearized).
#define ASSERT_EQ_CIPHERTEXT_N(ct1, ct2, n)                            \
	do {                                                               \
		ASSERT_EQ((ct1)->GetElements().size(), static_cast<size_t>(n)); \
		ASSERT_EQ_CIPHERTEXT(ct1, ct2);                                 \
	} while (0)

// Component-count and per-limb value equality only, no NoiseScaleDeg/ScalingFactor/EncodingType
// asserts. For ops where a documented metadata divergence exists (Rescale family) but limb
// bit-exactness must still be pinned.
#define ASSERT_EQ_CIPHERTEXT_LIMBS(ct1, ct2)                                                                                                             \
	do {                                                                                                                                                 \
		ASSERT_EQ((ct1)->GetElements().size(), (ct2)->GetElements().size());                                                                             \
		for (size_t j = 0; j < (ct1)->GetElements().size(); ++j) {                                                                                       \
			ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().size(), (ct2)->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < (ct1)->GetElements().at(j).GetAllElements().size(); ++i) {                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                     \
				  (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues(), (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                            \
		}                                                                                                                                                \
	} while (0)

static bool TestUseCuda() {
	return GetTestBackend() == TestBackend::CUDA;
}

// The OpenFHE context the api wraps (its CPU shadow). It holds the same keys the api generated, so
// it is the parity oracle.
static lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& LbCc(CryptoContext<DCRTPoly>& cc) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->host);
}

// The lbcrypto ciphertext behind a freshly-encrypted api ciphertext (host-resident before any op).
// Only valid on encrypt outputs — a mid-chain api ciphertext's shadow is stale on the CUDA backend.
static lbcrypto::Ciphertext<lbcrypto::DCRTPoly> LbCt(Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

// The lbcrypto plaintext behind an api plaintext.
static lbcrypto::Plaintext LbPt(Plaintext& pt) {
	return std::any_cast<lbcrypto::Plaintext>(pt->host);
}

// Recover the lbcrypto ciphertext of an api result, syncing it back from the device first (the device
// copy stays resident, so the ciphertext can still be used on-device afterward).
static lbcrypto::Ciphertext<lbcrypto::DCRTPoly> HostCt(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct) {
	cc->RecoverHostCiphertext(ct);
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

// Assert both ciphertexts have exactly `comps` elements and that each corresponding component
// pair has the same tower count. A structural precondition check, not a value comparison.
static void ExpectStructure(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle,
  size_t comps) {
	ASSERT_EQ(got->GetElements().size(), comps);
	ASSERT_EQ(oracle->GetElements().size(), comps);
	for (size_t j = 0; j < comps; ++j) {
		ASSERT_EQ(got->GetElements().at(j).GetAllElements().size(), oracle->GetElements().at(j).GetAllElements().size());
	}
}

// Decrypt two lbcrypto ciphertexts and compare their real slots within tol. For ops that are
// numerically correct but not bit-identical to OpenFHE (the raw OpenFheInterfaceTests use
// ASSERT_ERROR_OK for the same ops, for the same reason).
static void ExpectSlotsNear(CryptoContext<DCRTPoly>& cc,
  const PrivateKey<DCRTPoly>& sk,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle,
  size_t slots,
  double tol) {
	auto& lbcc	 = LbCc(cc);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext pgot, porc;
	lbcc->Decrypt(skImpl, got, &pgot);
	lbcc->Decrypt(skImpl, oracle, &porc);
	pgot->SetLength(slots);
	porc->SetLength(slots);
	auto vg = pgot->GetRealPackedValue();
	auto vo = porc->GetRealPackedValue();
	for (size_t i = 0; i < slots; ++i)
		EXPECT_NEAR(vg[i], vo[i], tol) << "slot " << i;
}

// Same parameters as the api correctness fixture (ApiTests.cpp): depth 5, ringDim 65536, 8 slots,
// FIXEDAUTO (default secret-key dist, as in ApiTests).
class ApiParityTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 5;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
	// Near-1 values keep repeated squaring well-conditioned (above the noise floor, below overflow).
	const std::vector<double> vsq = { 0.93, 0.96, 0.99, 1.0, 1.01, 1.04, 1.07, 0.95 };

	void SetUp() override {
		if (GetTestBackend() == TestBackend::CPU)
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend (api == OpenFHE); "
							"set FIDESLIB_TEST_BACKEND=cuda or =haze to run it.";

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
		// -4/-8 complete the negative half of the +-2^i doubling ladder EvalRotateMany is tested over.
		cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -4, -8 });
		cc->LoadContext(keys.publicKey);
	}

	// Encrypt v1 and v2 with the public key.
	void EncryptInputs(Ciphertext<DCRTPoly>& a1, Ciphertext<DCRTPoly>& a2) {
		auto p1 = cc->MakeCKKSPackedPlaintext(v1);
		auto p2 = cc->MakeCKKSPackedPlaintext(v2);
		a1		= cc->Encrypt(p1, keys.publicKey);
		a2		= cc->Encrypt(p2, keys.publicKey);
	}

	// Encrypt just v1 (for single-input ops).
	Ciphertext<DCRTPoly> EncryptV1() {
		auto p = cc->MakeCKKSPackedPlaintext(v1);
		return cc->Encrypt(p, keys.publicKey);
	}

	// Approximate comparison for the single-op tests (negate/rotate/rescale).
	void ExpectApproxEq(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle) {
		ExpectSlotsNear(cc, keys.secretKey, got, oracle, kSlots, 1e-6);
	}

};

TEST_F(ApiParityTest, EvalAdd) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalAdd(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalAdd(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalSub) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalSub(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalSub(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalMult) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalMult(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalMult(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalSquare) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalSquare(a1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalSquare(lb1);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalNegate) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalNegate(a1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalNegate(lb1);
	// Approximate, not bit-identical: the CUDA backend implements negate as multScalar(-1.0), which
	// bumps NoiseScaleDeg (1->2) relative to OpenFHE's degree-preserving sign flip. The decrypted
	// values match; the scale-degree bookkeeping differs (a known GPU EvalNegate fidelity gap).
	ExpectApproxEq(got, oracle);
}

TEST_F(ApiParityTest, EvalRotate) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalRotate(a1, 1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalRotate(lb1, 1);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 2);
}

TEST_F(ApiParityTest, EvalRotateSlotsAware) {
	// Rotate an 8-slot ciphertext by 5/6/7 — steps > slots/2 — to lock the slots-aware rotation
	// (parity #7: normalyzeIndex + the actual_index/alternate-key selection) to CUDA/OpenFHE. The
	// EvalRotate case above rotates by 1, where normalyzeIndex is the identity and the gap is hidden.
	// One program per context: declare all rotations as outputs before any readback.
	auto a1	 = EncryptV1();
	auto lb1 = LbCt(a1);
	const std::vector<int> steps = { 5, 6, 7 };
	std::vector<Ciphertext<DCRTPoly>> rots;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> oracles;
	for (int s : steps) {
		auto r = cc->EvalRotate(a1, s);
		cc->MarkOutput(r);
		rots.push_back(r);
		oracles.push_back(LbCc(cc)->EvalRotate(lb1, s));
	}
	for (size_t i = 0; i < steps.size(); ++i) {
		SCOPED_TRACE("rotate by " + std::to_string(steps[i]));
		auto got = HostCt(cc, rots[i]);
		ExpectSlotsNear(cc, keys.secretKey, got, oracles[i], kSlots, 1e-5);
	}
}

// EvalRotateMany on the device backend: the batched same-index key switch
// against THIS backend's own EvalRotate, bit for bit.
//
// The oracle is deliberately the backend's serial EvalRotate and not OpenFHE: the CUDA key switch
// is a different (equally valid) rounding of the same value than OpenFHE's — the pre-existing gap
// the EvalRotate case above documents — so an OpenFHE comparison could only ever be approximate,
// and approximate is not the contract. What EvalRotateMany promises is that BATCHING changes
// nothing, which is exactly the serial-vs-batched comparison made here. The decrypted values are
// still checked against OpenFHE so a batch that was self-consistently wrong cannot pass.
TEST_F(ApiParityTest, EvalRotateMany) {
	const std::vector<int32_t> indices = { 1, 2, 4, 8, -1, -2 };
	for (uint32_t B : { 1u, 2u, 7u, 32u }) {
		SCOPED_TRACE("batch width " + std::to_string(B));
		std::vector<Ciphertext<DCRTPoly>> cts;
		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> lbs;
		for (uint32_t b = 0; b < B; ++b) {
			std::vector<double> v(kSlots);
			for (uint32_t i = 0; i < kSlots; ++i)
				v[i] = static_cast<double>(b + 1) + static_cast<double>(i) / 16.0;
			auto pt = cc->MakeCKKSPackedPlaintext(v);
			cts.push_back(cc->Encrypt(pt, keys.publicKey));
			lbs.push_back(LbCt(cts.back()));
		}
		// One program per context: every rotation is declared an output before any readback.
		std::vector<std::vector<Ciphertext<DCRTPoly>>> batched, serial;
		std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> oracles;
		for (int32_t idx : indices) {
			auto many = cc->EvalRotateMany(cts, idx);
			std::vector<Ciphertext<DCRTPoly>> one;
			std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> orc;
			for (uint32_t b = 0; b < B; ++b) {
				cc->MarkOutput(many[b]);
				one.push_back(cc->EvalRotate(cts[b], idx));
				cc->MarkOutput(one.back());
				orc.push_back(LbCc(cc)->EvalRotate(lbs[b], idx));
			}
			batched.push_back(std::move(many));
			serial.push_back(std::move(one));
			oracles.push_back(std::move(orc));
		}
		for (size_t k = 0; k < indices.size(); ++k) {
			for (uint32_t b = 0; b < B; ++b) {
				SCOPED_TRACE("index " + std::to_string(indices[k]) + " entry " + std::to_string(b));
				auto gb = HostCt(cc, batched[k][b]);
				auto gs = HostCt(cc, serial[k][b]);
				ASSERT_EQ_CIPHERTEXT(gb, gs);
				ExpectSlotsNear(cc, keys.secretKey, gb, oracles[k][b], kSlots, 1e-5);
			}
		}
	}
}

// EvalFastRotationPrecompute + the SINGLE-INDEX EvalFastRotation: the hoisting contract the way
// callers actually use it -- precompute once, then rotate one index at a time, lazily. That shape is
// what the CUDA engine used to drop on the floor: evalFastRotationPrecompute returned nullptr and
// the single-index evalFastRotation ignored the handle and did a full key switch per call, so the
// hoisting was a no-op there while the CPU backend (OpenFHE's own EvalFastRotation) really hoisted.
// The vector overload was unaffected -- it reaches Ciphertext::rotate_hoisted -- which is why
// OpenFheInterfaceTests.cu's HoistedRotate never caught this.
//
// The bar is VALUE IDENTITY: sharing the ModUp must not change a single bit against the same
// backend's EvalRotate. One program per context: every rotation is declared an output before any
// readback.
TEST_F(ApiParityTest, FastRotationPrecompute) {
	auto a1			 = EncryptV1();
	auto lb1		 = LbCt(a1);
	const uint32_t m = cc->GetCyclotomicOrder();

	// A negative index is in here on purpose: normalyzeIndex + the alternate-key selection run on the
	// precomputed path too, and a sign slip there is silent (see EvalRotateSlotsAware).
	const std::vector<int32_t> steps = { 1, 2, 3, 4, -1 };

	// ONE precompute, every index -- the whole point of the arm.
	auto pre = cc->EvalFastRotationPrecompute(a1);

	std::vector<Ciphertext<DCRTPoly>> fast, plain;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> oracles;
	for (int32_t s : steps) {
		auto f = cc->EvalFastRotation(a1, s, m, pre);
		cc->MarkOutput(f);
		fast.push_back(f);

		auto r = cc->EvalRotate(a1, s);
		cc->MarkOutput(r);
		plain.push_back(r);

		// The OpenFHE oracle, through ITS hoisted path (upstream EvalRotate and EvalFastRotation are
		// the same formulation since the BITCOMPAT H1 patch, so this pins the value, not a variant).
		auto lbpre = LbCc(cc)->EvalFastRotationPrecompute(lb1);
		oracles.push_back(LbCc(cc)->EvalFastRotation(lb1, s, m, lbpre));
	}

	for (size_t i = 0; i < steps.size(); ++i) {
		SCOPED_TRACE("fast rotate by " + std::to_string(steps[i]));
		auto got = HostCt(cc, fast[i]);
		auto ref = HostCt(cc, plain[i]);
		// SAME backend, SAME formulation, only the shared half moved: bit-for-bit.
		ASSERT_EQ_CIPHERTEXT(ref, got);
		// vs OpenFHE: approximate only. The backend's key switch is a different (equally valid)
		// rounding of the same value -- the pre-existing gap EvalRotate above documents.
		ExpectSlotsNear(cc, keys.secretKey, got, oracles[i], kSlots, 1e-5);
	}

	// A handle decomposes ONE modulus chain. Used after the source has descended a level it is a
	// wrong answer CKKS would never flag, so the CUDA engine refuses it rather than computing
	// garbage. (The haze engine chose the other horn and silently falls back to a plain rotate --
	// api/engine/haze/HazeEngine.cpp evalFastRotation -- so this contract is CUDA's alone.)
	if (TestUseCuda()) {
		Ciphertext<DCRTPoly> b1, b2;
		EncryptInputs(b1, b2);
		auto stale	 = cc->EvalFastRotationPrecompute(b1);
		auto lowered = cc->EvalMult(b1, b2);
		cc->RescaleInPlace(lowered);
		EXPECT_THROW((void)cc->EvalFastRotation(lowered, 1, m, stale), std::exception);
	}
}

TEST_F(ApiParityTest, EvalAddPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	= LbCt(a1);
	auto pt		= cc->MakeCKKSPackedPlaintext(v2);
	auto lbpt	= LbPt(pt);
	auto res	= cc->EvalAdd(a1, pt);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalAdd(lb1, lbpt);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalMultPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	= LbCt(a1);
	auto pt		= cc->MakeCKKSPackedPlaintext(v2);
	auto lbpt	= LbPt(pt);
	auto res	= cc->EvalMult(a1, pt);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalMult(lb1, lbpt);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalAddScalar) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	 = LbCt(a1);
	auto resP	 = cc->EvalAdd(a1, 2.0);
	auto resN	 = cc->EvalAdd(a1, -2.0);
	// One program per context: declare both outputs, then read them back. Computing
	// resN after resP's readback would be compute-after-flush on the haze backend.
	cc->MarkOutput(resP);
	cc->MarkOutput(resN);
	auto gotP	 = HostCt(cc, resP);
	auto oracleP = LbCc(cc)->EvalAdd(lb1, 2.0);
	ASSERT_EQ_CIPHERTEXT(oracleP, gotP);
	auto gotN	 = HostCt(cc, resN);
	auto oracleN = LbCc(cc)->EvalAdd(lb1, -2.0);
	ASSERT_EQ_CIPHERTEXT(oracleN, gotN);
}

TEST_F(ApiParityTest, EvalMultScalar) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1);
	// 2^-7 matches the raw MultScalar test (a clean power-of-two scalar).
	const double s = std::pow(2.0, -7.0);
	auto res	   = cc->EvalMult(a1, s);
	auto got	   = HostCt(cc, res);
	auto oracle	   = LbCc(cc)->EvalMult(lb1, s);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, RescaleAfterMult) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto prod = cc->EvalMult(a1, a2);
	auto res  = cc->Rescale(prod);
	auto got  = HostCt(cc, res);
	// Oracle replays the whole chain in OpenFHE (the mid-chain product's host shadow is stale).
	auto oracle = LbCc(cc)->Rescale(LbCc(cc)->EvalMult(lb1, lb2));
	// Approximate, not bit-identical: GPU and OpenFHE track NoiseScaleDeg differently across an
	// explicit Rescale under FIXEDAUTO (deferred-rescale accounting). The decrypted values match.
	ExpectApproxEq(got, oracle);
}

// ---- AllLevels depth tests: exercise an op at each level down the modulus chain ----
// These walk the ciphertext down the chain via mult+rescale, which crosses the api-Rescale
// NoiseScaleDeg divergence, so they compare decrypted values (approximate). Bit-identity at each
// individual level remains covered by the retained raw OpenFheInterfaceTests *AllLevels tests.

// Program shape note (record/replay backends): the AllLevels tests compute every level's
// result first, declaring each as an output via MarkOutput, and only then read them all
// back. A record/replay backend (haze) executes the recorded program exactly once at the
// first readback, so per-level decrypt-then-keep-computing would be compute-after-readback.
// The op sequence and order are identical to the previous interleaved form — only the
// readback placement moved — so CPU/CUDA results are unchanged (MarkOutput is a no-op there).

TEST_F(ApiParityTest, EvalMultAllLevels) {
	Ciphertext<DCRTPoly> a, b;
	EncryptInputs(a, b);
	auto lbA = LbCt(a);
	auto lbB = LbCt(b);
	std::vector<Ciphertext<DCRTPoly>> perLevel;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> oracles;
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		a = cc->EvalMult(a, b);
		cc->RescaleInPlace(a);
		cc->MarkOutput(a);
		perLevel.push_back(a);
		lbA = LbCc(cc)->EvalMult(lbA, lbB);
		LbCc(cc)->RescaleInPlace(lbA);
		oracles.push_back(lbA);
	}
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		auto got = HostCt(cc, perLevel[lvl]);
		ExpectSlotsNear(cc, keys.secretKey, got, oracles[lvl], kSlots, 1e-5);
	}
}

TEST_F(ApiParityTest, EvalSquareAllLevels) {
	auto p	 = cc->MakeCKKSPackedPlaintext(vsq);
	auto a	 = cc->Encrypt(p, keys.publicKey);
	auto lbA = LbCt(a);
	std::vector<Ciphertext<DCRTPoly>> perLevel;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> oracles;
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		a = cc->EvalSquare(a);
		cc->RescaleInPlace(a);
		cc->MarkOutput(a);
		perLevel.push_back(a);
		lbA = LbCc(cc)->EvalSquare(lbA);
		LbCc(cc)->RescaleInPlace(lbA);
		oracles.push_back(lbA);
	}
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		auto got = HostCt(cc, perLevel[lvl]);
		ExpectSlotsNear(cc, keys.secretKey, got, oracles[lvl], kSlots, 1e-5);
	}
}

TEST_F(ApiParityTest, EvalRotateAllLevels) {
	Ciphertext<DCRTPoly> a, b;
	EncryptInputs(a, b);
	auto lbA = LbCt(a);
	auto lbB = LbCt(b);
	std::vector<Ciphertext<DCRTPoly>> rotations;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> oracles;
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		// Rotate at the current level; the per-level descent chain stays a program-local.
		auto r = cc->EvalRotate(a, 1);
		cc->MarkOutput(r);
		rotations.push_back(r);
		oracles.push_back(LbCc(cc)->EvalRotate(lbA, 1));
		// Descend a level for the next iteration.
		a = cc->EvalMult(a, b);
		cc->RescaleInPlace(a);
		lbA = LbCc(cc)->EvalMult(lbA, lbB);
		LbCc(cc)->RescaleInPlace(lbA);
	}
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		auto gotR = HostCt(cc, rotations[lvl]);
		ExpectSlotsNear(cc, keys.secretKey, gotR, oracles[lvl], kSlots, 1e-5);
	}
}

// ---- Degree-2 (EvalMultNoRelin / Relinearize) parity ----
// EvalMultNoRelin is the tensor product without the trailing key-switch, so its result carries
// three polynomials until Relinearize folds the third back in. OpenFHE's keyed EvalMult is
// literally EvalMultNoRelin + KeySwitchCore(cv[2]) + fold + resize(2)
// (deps/openfhe-src/src/pke/lib/schemebase/base-leveledshe.cpp:202-215,315-329), which is what
// EvalMultEqualsRelinearizeNoRelin pins bit-exactly.
//
// Degree-2 operand rules the group below locks to OpenFHE: add/sub are elementwise over the
// components both operands have, with the larger operand's tail copied (add) or negated when it
// belongs to the subtrahend (sub); negate, plaintext multiply, scalar multiply and rescale all
// loop over EVERY component.

TEST_F(ApiParityTest, EvalMultNoRelin) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	const size_t nsd1 = a1->GetNoiseScaleDeg(), nsd2 = a2->GetNoiseScaleDeg();
	const double sf1 = a1->GetScalingFactor(), sf2 = a2->GetScalingFactor();

	auto res = cc->EvalMultNoRelin(a1, a2);
	// NumberCiphertextElements is the OpenFHE spelling of GetNumElements; exercised once here.
	EXPECT_EQ(res->NumberCiphertextElements(), 3u);
	// Metadata computed independently of the oracle: NSD adds, the scaling factor multiplies.
	EXPECT_EQ(res->GetNoiseScaleDeg(), nsd1 + nsd2);
	EXPECT_EQ(res->GetScalingFactor(), sf1 * sf2);

	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
}

TEST_F(ApiParityTest, RelinearizeAfterNoRelin) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto d2		= cc->EvalMultNoRelin(a1, a2);
	auto res	= cc->Relinearize(d2); // out-of-place: d2 stays degree 2
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->Relinearize(LbCc(cc)->EvalMultNoRelin(lb1, lb2));
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 2);
}

TEST_F(ApiParityTest, RelinearizeInPlaceAfterNoRelin) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto q = cc->EvalMultNoRelin(a1, a2);
	cc->RelinearizeInPlace(q);
	cc->MarkOutput(q); // declared after the mutation: the output is the relinearized value

	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	LbCc(cc)->RelinearizeInPlace(oracle);
	auto got = HostCt(cc, q);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 2);
}

TEST_F(ApiParityTest, RelinearizeDegree1NoOp) {
	// OpenFHE's Relinearize/RelinearizeInPlace on a degree-1 (2-component) ciphertext is a no-op
	// (RelinearizeCore returns early). Pins that no-op both out-of-place and in-place, and that the
	// in-place path never rebinds a payload onto itself (which would corrupt or double-free it).
	auto p	 = EncryptV1();
	auto lb1 = LbCt(p);

	auto q = cc->Relinearize(p); // out-of-place: p is untouched, q is an independent ciphertext
	auto r = p->Clone();
	cc->RelinearizeInPlace(r); // in-place no-op on the clone
	cc->MarkOutput(p);
	cc->MarkOutput(q);
	cc->MarkOutput(r);

	auto oracle = LbCc(cc)->Relinearize(lb1);
	auto gotP	= HostCt(cc, p);
	auto gotQ	= HostCt(cc, q);
	auto gotR	= HostCt(cc, r);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotP, 2);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotQ, 2);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotR, 2);
	ASSERT_EQ_CIPHERTEXT(gotP, gotQ);
	ASSERT_EQ_CIPHERTEXT(gotP, gotR);
}

TEST_F(ApiParityTest, EvalMultEqualsRelinearizeNoRelin) {
	// The primary gate for splitting evalMult into tensor product + relinearize: the composition
	// must be BIT-IDENTICAL to the fused op, and both must match OpenFHE's EvalMult.
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto p = cc->EvalMult(a1, a2);
	auto q = cc->EvalMultNoRelin(a1, a2);
	cc->RelinearizeInPlace(q);
	cc->MarkOutput(p);
	cc->MarkOutput(q);

	auto oracle = LbCc(cc)->EvalMult(lb1, lb2);
	auto gotP	= HostCt(cc, p);
	auto gotQ	= HostCt(cc, q);
	// The primary gate first, and BIT-EXACT on EVERY backend including CUDA: the split composition
	// must reproduce the fused op exactly, which is what proves mult() and multNoRelin()+relinearize()
	// share one tensor product and one key-switch fold rather than two drifting copies. The
	// comparison against the OpenFHE oracle below is likewise bit-exact on every backend.
	ASSERT_EQ_CIPHERTEXT(gotP, gotQ);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotP, 2);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotQ, 2);
}

TEST_F(ApiParityTest, Degree2EvalAdd) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto d2a	 = cc->EvalMultNoRelin(a1, a2);
	auto d2b	 = cc->EvalMultNoRelin(a2, a1);
	auto sum	 = cc->EvalAdd(d2a, d2b);
	auto mixedLo = cc->EvalAdd(a1, d2a); // degree 1 + degree 2: the third component is copied through
	auto mixedHi = cc->EvalAdd(d2a, a1); // same sum, operands swapped (see the adjust note below)
	cc->MarkOutput(sum);
	cc->MarkOutput(mixedLo);
	cc->MarkOutput(mixedHi);

	auto od2a	  = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	auto od2b	  = LbCc(cc)->EvalMultNoRelin(lb2, lb1);
	auto oSum	  = LbCc(cc)->EvalAdd(od2a, od2b);
	auto oMixedLo = LbCc(cc)->EvalAdd(lb1, od2a);
	auto oMixedHi = LbCc(cc)->EvalAdd(od2a, lb1);
	{
		SCOPED_TRACE("degree 2 + degree 2");
		auto got = HostCt(cc, sum);
		ASSERT_EQ_CIPHERTEXT_N(oSum, got, 3);
	}
	{
		SCOPED_TRACE("degree 1 + degree 2 (tail copied from the second operand)");
		auto got = HostCt(cc, mixedLo);
		ASSERT_EQ_CIPHERTEXT_N(oMixedLo, got, 3);
	}
	{
		// Swapping the operands crosses a PRE-EXISTING, degree-independent divergence in the
		// FIXEDAUTO add adjust: at equal levels with mismatched depth, the device backends port
		// CUDA's depth fix, which RESCALES the depth-2 operand (result depth 1, one level lower),
		// while OpenFHE's AdjustLevelsAndDepthInPlace bumps the depth-1 operand to depth 2 (result
		// depth 2, same level). Both are valid; they are not bit-identical, and the tower counts
		// differ, so only the decrypted values can be compared. The `mixedLo` order above happens
		// to take the branch that agrees with OpenFHE, which is what pins the tail copy exactly.
		SCOPED_TRACE("degree 2 + degree 1 (CUDA-ported depth fix; values only)");
		auto got = HostCt(cc, mixedHi);
		ASSERT_EQ(got->GetElements().size(), 3u);
		ExpectApproxEq(got, oMixedHi);
	}
}

TEST_F(ApiParityTest, Degree2EvalAddInPlace) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto acc  = cc->EvalMultNoRelin(a1, a2);
	auto term = cc->EvalMultNoRelin(a2, a1);
	cc->EvalAddInPlace(acc, term);
	cc->MarkOutput(acc);

	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	auto oTerm	= LbCc(cc)->EvalMultNoRelin(lb2, lb1);
	LbCc(cc)->EvalAddInPlace(oracle, oTerm);
	auto got = HostCt(cc, acc);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
}

TEST_F(ApiParityTest, Degree2EvalAddMany) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	// EvalAddMany's association is bit-significant; OpenFHE folds left from the first non-null
	// entry (base-advancedshe.cpp:49-68), which is what the backend reproduces.
	std::vector<Ciphertext<DCRTPoly>> ds = { cc->EvalMultNoRelin(a1, a2), cc->EvalMultNoRelin(a2, a1), cc->EvalMultNoRelin(a1, a1) };
	auto sum							 = cc->EvalAddMany(ds);
	cc->MarkOutput(sum);

	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ods = { LbCc(cc)->EvalMultNoRelin(lb1, lb2), LbCc(cc)->EvalMultNoRelin(lb2, lb1), LbCc(cc)->EvalMultNoRelin(lb1, lb1) };
	auto oracle = LbCc(cc)->EvalAddMany(ods);
	auto got	= HostCt(cc, sum);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
}

TEST_F(ApiParityTest, Degree2EvalSub) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto d2a  = cc->EvalMultNoRelin(a1, a2);
	auto d2b  = cc->EvalMultNoRelin(a2, a1);
	auto diff = cc->EvalSub(d2a, d2b);
	// degree 1 - degree 2: the tail comes from the SUBTRAHEND, so it is negated rather than copied.
	auto mixed = cc->EvalSub(a1, d2a);
	cc->MarkOutput(diff);
	cc->MarkOutput(mixed);

	auto od2a	= LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	auto od2b	= LbCc(cc)->EvalMultNoRelin(lb2, lb1);
	auto oDiff	= LbCc(cc)->EvalSub(od2a, od2b);
	auto oMixed = LbCc(cc)->EvalSub(lb1, od2a);
	{
		SCOPED_TRACE("degree 2 - degree 2");
		auto got = HostCt(cc, diff);
		ASSERT_EQ_CIPHERTEXT_N(oDiff, got, 3);
	}
	{
		SCOPED_TRACE("degree 1 - degree 2 (negated tail)");
		auto got = HostCt(cc, mixed);
		ASSERT_EQ_CIPHERTEXT_N(oMixed, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2EvalSubMixedMinuend) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto d2a = cc->EvalMultNoRelin(a1, a2);
	// degree 2 - degree 1: the tail comes from the MINUEND, so it is copied rather than negated
	// (the mirror of Degree2EvalSub's degree-1-minuend case above, addSubCore's "minuend tail:
	// unchanged" branch).
	auto mixed = cc->EvalSub(d2a, a1);
	cc->MarkOutput(mixed);

	auto od2a	= LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	auto oMixed = LbCc(cc)->EvalSub(od2a, lb1);
	auto got	= HostCt(cc, mixed);
	// This operand order (degree-2 first) crosses the same PRE-EXISTING, degree-independent
	// divergence documented on Degree2EvalAdd's mixedHi case: adjustForAddOrSub is symmetric in
	// add/sub, so ordering the degree-2 operand first makes the device backends take CUDA's depth
	// fix (rescale the depth-2 operand), while OpenFHE's AdjustLevelsAndDepthInPlace bumps the
	// depth-1 operand instead (same values, different tower counts). Values-only comparison.
	SCOPED_TRACE("degree 2 - degree 1 (CUDA-ported depth fix; values only)");
	ASSERT_EQ(got->GetElements().size(), 3u);
	ExpectApproxEq(got, oMixed);
}

TEST_F(ApiParityTest, Degree2EvalNegate) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto d2		= cc->EvalMultNoRelin(a1, a2);
	auto neg	= cc->EvalNegate(d2);
	auto got	= HostCt(cc, neg);
	auto oracle = LbCc(cc)->EvalNegate(LbCc(cc)->EvalMultNoRelin(lb1, lb2));
	// Bit-exact on both device backends: negate is OpenFHE's degree-preserving per-limb q_i − 1 on
	// every component (CUDA's negateExact, CudaEngine.cpp; haze mirrors it).
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
}

TEST_F(ApiParityTest, Degree2EvalMultPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto pt	  = cc->MakeCKKSPackedPlaintext(v1);
	auto lbpt = LbPt(pt);
	// The same values encoded one level down, so the pre-rescaled case below meets its ciphertext
	// at equal level and depth — the configuration in which both sides skip the adjust entirely.
	auto ptL1	= cc->MakeCKKSPackedPlaintext(v1, 1, 1);
	auto lbptL1 = LbPt(ptL1);

	auto deep	  = cc->EvalMult(cc->EvalMultNoRelin(a1, a2), pt);				  // depth-2 ct × plaintext
	auto rescaled = cc->EvalMult(cc->Rescale(cc->EvalMultNoRelin(a1, a2)), ptL1); // depth-1 ct × plaintext
	cc->MarkOutput(deep);
	cc->MarkOutput(rescaled);

	auto oDeep	   = LbCc(cc)->EvalMult(LbCc(cc)->EvalMultNoRelin(lb1, lb2), lbpt);
	auto oRescaled = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	LbCc(cc)->GetScheme()->ModReduceInternalInPlace(oRescaled, 1); // the api Rescale is unconditional
	oRescaled = LbCc(cc)->EvalMult(oRescaled, lbptL1);
	{
		// A depth-2 ciphertext must be rescaled before the plaintext multiply, and the two
		// implementations spend that rescale differently — a PRE-EXISTING, degree-independent
		// divergence: the device backends rescale the ciphertext and trim the plaintext chain (GPU
		// Ciphertext.cpp:356-421), while OpenFHE morphs the plaintext, scales it by an encoded 1.0
		// and rescales THAT too (rns-leveledshe.cpp:266-278 -> AdjustLevelsAndDepthToOneInPlace).
		// Level, depth and scaling factor still agree exactly; the limbs differ by that rounding.
		SCOPED_TRACE("depth-2 ciphertext (values + metadata only)");
		auto got = HostCt(cc, deep);
		ExpectStructure(got, oDeep, 3);
		EXPECT_EQ(got->GetNoiseScaleDeg(), oDeep->GetNoiseScaleDeg());
		EXPECT_EQ(got->GetScalingFactor(), oDeep->GetScalingFactor());
		ExpectApproxEq(got, oDeep);
	}
	{
		// Depth 1 and equal levels: both sides multiply straight through, so every one of the three
		// components must match bit-for-bit.
		SCOPED_TRACE("depth-1 ciphertext (bit-exact)");
		auto got = HostCt(cc, rescaled);
		ASSERT_EQ_CIPHERTEXT_N(oRescaled, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2EvalMultScalar) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	const double s = std::pow(2.0, -7.0); // clean power of two, as in the degree-1 scalar test

	auto res = cc->EvalMult(cc->EvalMultNoRelin(a1, a2), s);
	auto ip	 = cc->EvalMultNoRelin(a1, a2);
	cc->EvalMultInPlace(ip, s);
	cc->MarkOutput(res);
	cc->MarkOutput(ip);

	auto oracle = LbCc(cc)->EvalMult(LbCc(cc)->EvalMultNoRelin(lb1, lb2), s);
	{
		SCOPED_TRACE("out of place");
		auto got = HostCt(cc, res);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
	{
		SCOPED_TRACE("in place");
		auto got = HostCt(cc, ip);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2EvalAddSubPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto pt	  = cc->MakeCKKSPackedPlaintext(v1);
	auto lbpt = LbPt(pt);

	// Depth-2 ciphertext + depth-1 plaintext at equal level. Unlike the plaintext MULTIPLY above,
	// add/sub does NOT rescale the ciphertext: OpenFHE's AdjustLevelsAndDepthInPlace scales the
	// lower-depth side up instead (ckksrns-leveledshe.cpp:726-732), which is an encoded 1.0 on the
	// morphed plaintext and nothing at all on the ciphertext. Both device backends now do exactly
	// that, so all three components must match bit-for-bit — including the two the plaintext never
	// touches, which have to be passed through unchanged.
	auto sum  = cc->EvalAdd(cc->EvalMultNoRelin(a1, a2), pt);
	auto diff = cc->EvalSub(cc->EvalMultNoRelin(a1, a2), pt);
	cc->MarkOutput(sum);
	cc->MarkOutput(diff);

	auto oSum  = LbCc(cc)->EvalAdd(LbCc(cc)->EvalMultNoRelin(lb1, lb2), lbpt);
	auto oDiff = LbCc(cc)->EvalSub(LbCc(cc)->EvalMultNoRelin(lb1, lb2), lbpt);
	{
		SCOPED_TRACE("degree 2 + plaintext");
		auto got = HostCt(cc, sum);
		ASSERT_EQ_CIPHERTEXT_N(oSum, got, 3);
	}
	{
		SCOPED_TRACE("degree 2 - plaintext");
		auto got = HostCt(cc, diff);
		ASSERT_EQ_CIPHERTEXT_N(oDiff, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2EvalSubScalarReversed) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	const double s = 0.25; // clean power of two, exactly representable at every scale

	// scalar − ct, the reversed operand order: OpenFHE evaluates it as
	// EvalAdd(EvalNegate(ct), scalar) (cryptocontext.h:1835-1837), i.e. a degree-preserving sign
	// flip of EVERY component followed by the scalar encoded at the ciphertext's own noise degree.
	// Neither step changes level or depth, so the parity is bit-exact on all three components.
	auto res = cc->EvalSub(s, cc->EvalMultNoRelin(a1, a2));
	auto ip	 = cc->EvalMultNoRelin(a1, a2);
	cc->EvalSubInPlace(s, ip);
	cc->MarkOutput(res);
	cc->MarkOutput(ip);

	auto oracle = LbCc(cc)->EvalSub(s, LbCc(cc)->EvalMultNoRelin(lb1, lb2));
	{
		SCOPED_TRACE("out of place");
		auto got = HostCt(cc, res);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
	{
		SCOPED_TRACE("in place");
		auto got = HostCt(cc, ip);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2Rescale) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto res = cc->Rescale(cc->EvalMultNoRelin(a1, a2));
	auto ip	 = cc->EvalMultNoRelin(a1, a2);
	cc->RescaleInPlace(ip);
	cc->MarkOutput(res);
	cc->MarkOutput(ip);

	// The api's Rescale is UNCONDITIONAL on the device backends (parity #3: CUDA mod-reduces under
	// every scaling technique), while OpenFHE's public Rescale/ModReduce only mod-reduces under
	// FIXEDMANUAL and clones under FIXEDAUTO. The oracle therefore calls the internal
	// ModReduceInternalInPlace directly — the operation the device backends actually perform.
	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	LbCc(cc)->GetScheme()->ModReduceInternalInPlace(oracle, 1);
	{
		SCOPED_TRACE("out of place");
		auto got = HostCt(cc, res);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
	{
		SCOPED_TRACE("in place");
		auto got = HostCt(cc, ip);
		ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
	}
}

TEST_F(ApiParityTest, Degree2HostReadback) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	const size_t nsd1 = a1->GetNoiseScaleDeg(), nsd2 = a2->GetNoiseScaleDeg();
	const double sf1 = a1->GetScalingFactor(), sf2 = a2->GetScalingFactor();
	const size_t level = a1->GetLevel();

	auto d2	 = cc->EvalMultNoRelin(a1, a2);
	auto got = HostCt(cc, d2);
	// The readback shell must be RESHAPED from 2 elements to 3, all at the same tower count.
	ASSERT_EQ(got->GetElements().size(), 3u);
	const size_t towers = got->GetElements().at(0).GetNumOfElements();
	for (size_t j = 1; j < 3; ++j)
		EXPECT_EQ(got->GetElements().at(j).GetNumOfElements(), towers) << "component " << j;
	EXPECT_EQ(got->GetLevel(), level);
	EXPECT_EQ(got->GetNoiseScaleDeg(), nsd1 + nsd2);
	EXPECT_EQ(got->GetScalingFactor(), sf1 * sf2);
	// The facade getter sees the same three elements once the value has been flushed.
	EXPECT_EQ(d2->GetElements().size(), 3u);
}

TEST_F(ApiParityTest, Degree2HostReadbackAtReducedLevel) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);

	// Descend one level first, so the degree-2 readback must both reshape (2 -> 3 elements) and
	// TRIM the fresh full-level shell down to the reduced tower count.
	auto lower = cc->EvalMult(a1, a2);
	cc->RescaleInPlace(lower);
	const size_t lowerLevel = lower->GetLevel();
	ASSERT_GT(lowerLevel, a1->GetLevel()) << "the descent did not consume a level";

	auto d2	 = cc->EvalMultNoRelin(lower, lower);
	auto got = HostCt(cc, d2);
	ASSERT_EQ(got->GetElements().size(), 3u);
	EXPECT_EQ(got->GetLevel(), lowerLevel);
	const size_t towers = got->GetElements().at(0).GetNumOfElements();
	for (size_t j = 1; j < 3; ++j)
		EXPECT_EQ(got->GetElements().at(j).GetNumOfElements(), towers) << "component " << j;
}

TEST_F(ApiParityTest, Degree2HostLoad) {
	// Mirror of Degree2HostReadback in the opposite direction: build a degree-2 ciphertext
	// entirely on the HOST via the OpenFHE oracle (EvalMultNoRelin never runs on the device for
	// this value), wrap it in a fresh facade ciphertext by assigning ->host directly -- the same
	// idiom CryptoContextImpl::Encrypt uses internally (api/CryptoContext.cpp) -- then push it to
	// the device with the public LoadCiphertext and read it back. Exercises the host->device load
	// path for a 3-component ciphertext through the public api.
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	ASSERT_EQ(oracle->GetElements().size(), 3u);

	Ciphertext<DCRTPoly> wrapped = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc));
	wrapped->host				 = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(oracle);
	cc->LoadCiphertext(wrapped);

	auto got = HostCt(cc, wrapped);
	ASSERT_EQ_CIPHERTEXT_N(oracle, got, 3);
}

TEST_F(ApiParityTest, Degree2CloneKeepsThirdComponent) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);

	auto p = cc->EvalMultNoRelin(a1, a2);
	auto q = p->Clone();
	cc->MarkOutput(p);
	cc->MarkOutput(q);

	auto oracle = LbCc(cc)->EvalMultNoRelin(lb1, lb2);
	auto gotP	= HostCt(cc, p);
	auto gotQ	= HostCt(cc, q);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotP, 3);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotQ, 3);
	ASSERT_EQ_CIPHERTEXT(gotP, gotQ);
}

TEST_F(ApiParityTest, Degree2DotProductLazyRelin) {
	// Lazy relinearization: accumulate degree-2 products and key-switch ONCE at the end, versus
	// key-switching after every product.
	const std::vector<std::vector<double>> lv = { v1, v2, v1, v2 };
	const std::vector<std::vector<double>> rv = { v2, v1, v1, v2 };

	std::vector<Ciphertext<DCRTPoly>> lhs, rhs;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> lbL, lbR;
	for (size_t k = 0; k < 4; ++k) {
		auto pl = cc->MakeCKKSPackedPlaintext(lv[k]);
		auto pr = cc->MakeCKKSPackedPlaintext(rv[k]);
		lhs.push_back(cc->Encrypt(pl, keys.publicKey));
		rhs.push_back(cc->Encrypt(pr, keys.publicKey));
		lbL.push_back(LbCt(lhs.back()));
		lbR.push_back(LbCt(rhs.back()));
	}

	auto lazy = cc->EvalMultNoRelin(lhs[0], rhs[0]);
	for (size_t k = 1; k < 4; ++k)
		cc->EvalAddInPlace(lazy, cc->EvalMultNoRelin(lhs[k], rhs[k]));
	cc->RelinearizeInPlace(lazy);

	auto eager = cc->EvalMult(lhs[0], rhs[0]);
	for (size_t k = 1; k < 4; ++k)
		cc->EvalAddInPlace(eager, cc->EvalMult(lhs[k], rhs[k]));

	cc->MarkOutput(lazy);
	cc->MarkOutput(eager);

	auto oracle = LbCc(cc)->EvalMultNoRelin(lbL[0], lbR[0]);
	for (size_t k = 1; k < 4; ++k) {
		auto term = LbCc(cc)->EvalMultNoRelin(lbL[k], lbR[k]);
		LbCc(cc)->EvalAddInPlace(oracle, term);
	}
	LbCc(cc)->RelinearizeInPlace(oracle);

	auto gotLazy  = HostCt(cc, lazy);
	auto gotEager = HostCt(cc, eager);
	ASSERT_EQ_CIPHERTEXT_N(oracle, gotLazy, 2);
	// Lazy and eager agree numerically but NOT bit-for-bit: KeySwitchCore is not additive, so one
	// key-switch of the summed third component is a different (equally valid) rounding of the same
	// value as four key-switches summed. Comparing decrypted slots is the only meaningful check.
	ExpectSlotsNear(cc, keys.secretKey, gotLazy, gotEager, kSlots, 1e-5);
}

// ---- Bootstrap parity ----
// EvalBootstrap on GPU/haze vs the OpenFHE host oracle (LbCc(cc) shares the api context's precom).
// Bootstrap is heavily approximate, so it compares decrypted values within the bootstrap precision
// (~1e-2). With the CUDA-matching policy (btsfirstboot forwarded -> BTSlotsEncoding=false) BOTH the
// api backend and the oracle run the ModRaise-first variant, so they stay in lockstep.
// Parametrized over levelBudget: {1,1} (isLT single BSGS), {2,2}, {3,3} (multi-stage FFT CtS/StC).
class ApiParityBootstrapTest : public ::testing::TestWithParam<std::vector<uint32_t>> {
  protected:
	// Context parameters mirror the CUDA backend's bootstrap tests exactly
	// (test/ParametrizedTest.cuh:377-380 logNboot/firstmodboot/scalemodboot/depthboot, driving
	// OpenFHEBootstrapTest over TTALL64BOOT), so haze and CUDA bootstrap under one parameter set.
	// numLargeDigits mirrors gparams64_13_3 (dnum=3); the CUDA suite sweeps dnum 1..4. This fixture
	// sweeps levelBudget instead ({1,1} single-BSGS vs {2,2}/{3,3} multi-stage FFT), where the CUDA
	// test pins {2,2}. The old depth=25/scaleMod=50 pairing was not a CUDA reference at all.
	static constexpr uint32_t kDepth	 = 23; // depthboot — exact CUDA match, viable up to {2,2}
	// A {3,3} budget consumes more levels than depthboot can recover, so bootstrap would gain
	// nothing and early-return (caught by the guard in the test body). Give the larger budget the
	// headroom it needs; {1,1} and {2,2} stay on the CUDA depth verbatim.
	static constexpr uint32_t kDepthWideBudget = 28;
	static constexpr uint32_t kScaleMod	 = 59; // scalemodboot
	static constexpr uint32_t kFirstMod	 = 60; // firstmodboot
	static constexpr uint32_t kRingDim	 = 1u << 16; // 1 << logNboot
	static constexpr uint32_t kBatchSize = 8; // gparams64_13_*.batchSize
	static constexpr uint32_t kSlots	 = 1u << 5; // OpenFHEBootstrap's `int slots = 1 << 5`

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	/// @brief Deepest level EvalBootstrap accepts for this instantiation:
	/// L0 - (levelBudget_decode + 2), from the guard at ckksrns-fhe.cpp:1252-1256. FIXEDAUTO, so no
	/// extra FLEXIBLEAUTOEXT tower. The CUDA test depletes to L-1, which it can only do because it
	/// bootstraps on the GPU and never calls the host EvalBootstrap that owns this guard; this
	/// fixture compares against that host call, so it deplete to the guard's limit instead — as
	/// depleted as the host path legally allows, and enough to make bootstrap do real work.
	/// @brief Multiplicative depth for this instantiation: the CUDA reference depth, widened for
	/// budgets it cannot support (see kDepthWideBudget).
	uint32_t depth() const {
		return GetParam().at(1) > 2 ? kDepthWideBudget : kDepth;
	}

	uint32_t maxBootLevel() const {
		return depth() - (GetParam().at(1) + 2);
	}

	void SetUp() override {
		if (GetTestBackend() == TestBackend::CPU)
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend (api == OpenFHE); "
							"set FIDESLIB_TEST_BACKEND=cuda or =haze to run it.";
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(depth());
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(kBatchSize);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		ConfigureTestEngine(params);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup(GetParam(), { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		cc->LoadContext(keys.publicKey);
	}
};

TEST_P(ApiParityBootstrapTest, EvalBootstrap) {
	// Encrypt DEPLETED (level kDepth-1) so EvalBootstrap runs the real pipeline on both sides. A
	// full-level input gains no levels, so bootstrap returns the input clone unchanged
	// (HazeBootstrap.cpp:1237, mirroring OpenFHE ckksrns-fhe.cpp:835); haze and oracle then agree
	// trivially without either having bootstrapped — a false pass. Same reason the fully-packed
	// fixture below depletes its input.
	auto pt		   = cc->MakeCKKSPackedPlaintext(v1, 1, maxBootLevel(), nullptr, kSlots);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	auto lbCt	   = LbCt(ct);
	const size_t levelBefore = ct->GetLevel();
	auto refreshed = cc->EvalBootstrap(ct);
	// Guard against the false pass: if bootstrap gained no levels it returns the input unchanged, so
	// haze and the oracle agree trivially without either having bootstrapped. At kDepth this holds
	// for budgets up to {2,2}; a larger budget consumes more levels than it recovers.
	ASSERT_LT(refreshed->GetLevel(), levelBefore) << "bootstrap gained no levels at levelBudget {" << GetParam().at(0) << "," << GetParam().at(1)
												 << "} and depth " << depth() << ": it returned the input unchanged, so this test exercised nothing";
	auto got	   = HostCt(cc, refreshed);
	auto oracle	   = LbCc(cc)->EvalBootstrap(lbCt);
	ExpectSlotsNear(cc, keys.secretKey, got, oracle, kSlots, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(LevelBudgets, ApiParityBootstrapTest,
  ::testing::Values(std::vector<uint32_t>{ 1, 1 }, std::vector<uint32_t>{ 2, 2 }, std::vector<uint32_t>{ 3, 3 }),
  [](const ::testing::TestParamInfo<std::vector<uint32_t>>& info) {
	  return "lvlb" + std::to_string(info.param[0]) + "_" + std::to_string(info.param[1]);
  });

// ---- Fully-packed bootstrap parity (slots == N/2; the COMPLEX modular-reduction branch) ----
// Exercises the fully-packed COMPLEX path in HazeEngine::bootstrapModRaiseFirst (the real/imag
// split via Conjugate + EvalSub/EvalAdd, MultByMonomial(3N/2), dual Chebyshev + double-angle on
// BOTH halves, MultByMonomial(N/2), recombine, scalar mult; skips the sparse post-StC fold). The
// oracle is the OpenFHE host context (LbCc), which runs the identical ModRaise-first bootstrap the
// CUDA backend mirrors.
//
// Config: N=4096 (slots == N/2 == 2048), fully packed. Depth/scaleMod/firstMod follow the CUDA
// reference (depth=28, scaleMod=58, firstMod=60, numLargeDigits=3) so EvalBootstrapSetup/keygen
// has comfortably more levels than the {3,3}/{4,4} budget consumes — the earlier "setup segfault"
// at depth 24/25 was an insufficient-depth parameter bug, not a haze defect (it reproduces on the
// CPU backend with the bad params and disappears at adequate depth). UNIFORM_TERNARY,
// HEStd_NotSet. Tolerance is the bootstrap precision (1e-2). The input is DEPLETED so EvalBootstrap
// runs the real pipeline rather than early-returning unchanged (a false pass).
//
// Parametrized over (levelBudget, scalingTechnique): the haze bootstrap supports
// FIXEDAUTO/FLEXIBLEAUTO/FLEXIBLEAUTOEXT (not FIXEDMANUAL). N=4096 fully-packed runs in-process on
// the local FHETCH simulator (OOM is no longer a concern at this ring).
struct FullPackParam {
	std::vector<uint32_t> levelBudget;
	ScalingTechnique scalingTech;
	const char* techName;
};

class ApiParityFullPackBootstrapTest : public ::testing::TestWithParam<FullPackParam> {
  protected:
	static constexpr uint32_t kDepth	= 28; // CUDA-reference depth; > {3,3}/{4,4} budget + EvalMod
	static constexpr uint32_t kScaleMod = 58; // CUDA-reference scaleMod (< 60 for device validation)
	static constexpr uint32_t kFirstMod = 60; // CUDA-reference firstMod
	// N defaults to 4096 (the validated in-process target). FIDESLIB_FULLPACK_RING overrides it
	// (e.g. 65536 for the CUDA-original ring) for diagnosis on machines with the RAM/time budget.
	uint32_t kRingDim = 1u << 12; // N = 4096
	uint32_t kSlots	  = kRingDim / 2; // fully packed: slots == N/2 == 2048

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	std::vector<double> v1;

	void SetUp() override {
		if (GetTestBackend() == TestBackend::CPU && std::getenv("FIDESLIB_FULLPACK_RUN_CPU") == nullptr)
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend (api == OpenFHE). Set "
							"FIDESLIB_FULLPACK_RUN_CPU=1 to run the OpenFHE-oracle isolation path (setup/keygen "
							"+ bootstrap on pure OpenFHE, no haze) for parameter diagnosis.";
		if (TestUseCuda())
			GTEST_SKIP() << "fully-packed parity is validated on haze; run with FIDESLIB_TEST_BACKEND=haze.";

		const FullPackParam& p = GetParam();

		if (const char* r = std::getenv("FIDESLIB_FULLPACK_RING")) {
			kRingDim = static_cast<uint32_t>(std::strtoul(r, nullptr, 10));
			kSlots	 = kRingDim / 2;
		}

		// FLEXIBLE* cannot run through the in-process FHETCH replay bridge at small rings: the bridge's
		// per-input template synthesis re-validates the FLEXIBLE scaling primes through OpenFHE's
		// gen-cryptocontext-params-validation, which rejects FLEXIBLE scaling-mod sizes outside (15,60)
		// at sub-2^16 rings ("scalingModSize should be greater than 15 and less than 60"). This is a
		// replay-bridge synthesis limitation, NOT a haze-engine defect (the haze trace records fine; it
		// fails only when the bridge re-derives input contexts). FIXEDAUTO uses fixed primes the bridge
		// accepts, so it is the validated in-process path. Skip FLEXIBLE* on the local in-process target.
		if (GetTestBackend() == TestBackend::HAZE && p.scalingTech != FIXEDAUTO) {
			const char* tgt = std::getenv("FIDESLIB_HAZE_TARGET");
			const bool isLocal = (tgt == nullptr || tgt[0] == '\0' || std::string(tgt) == "local");
			if (isLocal)
				GTEST_SKIP() << p.techName
							 << " cannot run through the in-process replay bridge at N=4096 (FLEXIBLE "
								"scaling-prime validation rejects sub-2^16 templates). Validated in-process "
								"path is FIXEDAUTO; set FIDESLIB_HAZE_TARGET to a non-local target for FLEXIBLE.";
		}

		// A full slot vector (length N/2): smooth, well-conditioned magnitudes < 1.
		v1.resize(kSlots);
		for (uint32_t i = 0; i < kSlots; ++i)
			v1[i] = 0.25 * std::sin(0.01 * static_cast<double>(i)) + 0.5;

		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(p.scalingTech);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		ConfigureTestEngine(params);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup(p.levelBudget, { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		cc->LoadContext(keys.publicKey);
	}
};

TEST_P(ApiParityFullPackBootstrapTest, EvalBootstrapFullyPacked) {
	if (GetTestBackend() == TestBackend::CPU) {
		// OpenFHE-oracle isolation path: setup/keygen already ran in SetUp(); just bootstrap on the
		// host context. If this path passes, the parameters are sound and any failure is haze-only.
		auto pt	  = cc->MakeCKKSPackedPlaintext(v1, 1, kDepth - 1);
		auto ct	  = cc->Encrypt(pt, keys.publicKey);
		auto lbCt = LbCt(ct);
		auto oracle = LbCc(cc)->EvalBootstrap(lbCt);
		auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(keys.secretKey->pimpl);
		lbcrypto::Plaintext porc;
		LbCc(cc)->Decrypt(skImpl, oracle, &porc);
		porc->SetLength(kSlots);
		auto vo = porc->GetRealPackedValue();
		double maxErr = 0.0;
		for (uint32_t i = 0; i < kSlots; ++i)
			maxErr = std::max(maxErr, std::abs(vo[i] - v1[i]));
		std::cout << "[CPU-oracle] " << GetParam().techName << " maxErr vs plaintext = " << maxErr << std::endl;
		EXPECT_LT(maxErr, 1e-2);
		return;
	}
	// Encrypt DEPLETED (level kDepth-1) so EvalBootstrap runs the real pipeline (a full-level input
	// would early-return unchanged, a false pass — see the early-out guard at the end of bootstrap).
	auto pt		   = cc->MakeCKKSPackedPlaintext(v1, 1, kDepth - 1);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	auto lbCt	   = LbCt(ct);
	auto refreshed = cc->EvalBootstrap(ct);
	auto got	   = HostCt(cc, refreshed);

	auto& lbcc	 = LbCc(cc);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(keys.secretKey->pimpl);

	std::cout << std::scientific;
	std::cout << "[haze] got level=" << got->GetLevel() << " NSD=" << got->GetNoiseScaleDeg()
			  << " sf=" << got->GetScalingFactor() << " towers=" << got->GetElements().at(0).GetNumOfElements() << std::endl;

	// Primary correctness check: decrypt the haze bootstrap result and compare to the KNOWN
	// plaintext (the true refresh target). This is independent of the OpenFHE oracle's own numeric
	// health at these params.
	lbcrypto::Plaintext pgot;
	std::vector<double> vg;
	double maxErrPlain = 0.0;
	try {
		lbcc->Decrypt(skImpl, got, &pgot);
		pgot->SetLength(kSlots);
		vg = pgot->GetRealPackedValue();
		for (uint32_t i = 0; i < kSlots; ++i)
			maxErrPlain = std::max(maxErrPlain, std::abs(vg[i] - v1[i]));
		std::cout << "[haze] " << GetParam().techName << " lvlb" << GetParam().levelBudget[0]
				  << " maxErr(haze vs plaintext) = " << maxErrPlain << " first slots:";
		for (uint32_t i = 0; i < 6; ++i) std::cout << " " << vg[i] << "(" << v1[i] << ")";
		std::cout << std::endl;
	} catch (const std::exception& e) {
		std::cout << "[haze] DECRYPT of haze result FAILED: " << e.what() << std::endl;
		FAIL() << "haze result decrypt failed";
	}

	// Secondary: compare against the OpenFHE ModRaise-first oracle when it decrypts cleanly.
	try {
		auto oracle = lbcc->EvalBootstrap(lbCt);
		lbcrypto::Plaintext porc;
		lbcc->Decrypt(skImpl, oracle, &porc);
		porc->SetLength(kSlots);
		auto vo			 = porc->GetRealPackedValue();
		double maxErrOrc = 0.0;
		for (uint32_t i = 0; i < kSlots; ++i)
			maxErrOrc = std::max(maxErrOrc, std::abs(vg[i] - vo[i]));
		std::cout << "[haze] " << GetParam().techName << " maxErr(haze vs oracle)   = " << maxErrOrc << std::endl;
		EXPECT_LT(maxErrOrc, 1e-2);
	} catch (const std::exception& e) {
		std::cout << "[haze] oracle EvalBootstrap unavailable at these params: " << e.what() << std::endl;
	}

	EXPECT_LT(maxErrPlain, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(LevelBudgets, ApiParityFullPackBootstrapTest,
  ::testing::Values(FullPackParam{ { 3, 3 }, FIXEDAUTO, "FIXEDAUTO" },
	FullPackParam{ { 3, 3 }, FLEXIBLEAUTO, "FLEXIBLEAUTO" },
	FullPackParam{ { 3, 3 }, FLEXIBLEAUTOEXT, "FLEXIBLEAUTOEXT" },
	FullPackParam{ { 4, 4 }, FIXEDAUTO, "FIXEDAUTO" },
	FullPackParam{ { 4, 4 }, FLEXIBLEAUTO, "FLEXIBLEAUTO" }),
  [](const ::testing::TestParamInfo<FullPackParam>& info) {
	  return "lvlb" + std::to_string(info.param.levelBudget[0]) + "_" + std::to_string(info.param.levelBudget[1]) + "_" + info.param.techName;
  });

// ---- Sparse-ternary secret key: parity of the shorter mod-eval approximation ----
// A genuinely sparse main key (lbcrypto::SPARSE_TERNARY, propagated by CCParams::SetSecretKeyDist)
// makes the OpenFHE oracle select g_coefficientsSparse -  degree 44 with R_SPARSE=3 double-angle
// iterations, against uniform's degree 88 with R_UNIFORM=6. A device that quietly fell back to the
// uniform table with K_UNIFORM=512 could not track the oracle slot-for-slot, so agreement here is
// evidence that the device ran the same shorter approximation the oracle did.
//
// Mirrors ApiParityBootstrapTest at level budget {3,3}; only the key distribution differs.
class ApiParitySparseSecretBootstrapTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth	 = 28; // budget {3,3} + sparse EvalMod, per kDepthWideBudget
	static constexpr uint32_t kScaleMod	 = 59;
	static constexpr uint32_t kFirstMod	 = 60;
	static constexpr uint32_t kRingDim	 = 1u << 16;
	static constexpr uint32_t kBatchSize = 8;
	static constexpr uint32_t kSlots	 = 1u << 5;
	static constexpr uint32_t kBudget	 = 3;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	uint32_t maxBootLevel() const {
		return kDepth - (kBudget + 2);
	}

	void SetUp() override {
		if (GetTestBackend() == TestBackend::CPU)
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend (api == OpenFHE); "
							"set FIDESLIB_TEST_BACKEND=cuda or =haze to run it.";
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(kBatchSize);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(3);
		params.SetSecretKeyDist(SPARSE_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		ConfigureTestEngine(params);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup({ kBudget, kBudget }, { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		cc->LoadContext(keys.publicKey);
	}
};

TEST_F(ApiParitySparseSecretBootstrapTest, EvalBootstrap) {
	auto pt					 = cc->MakeCKKSPackedPlaintext(v1, 1, maxBootLevel(), nullptr, kSlots);
	auto ct					 = cc->Encrypt(pt, keys.publicKey);
	auto lbCt				 = LbCt(ct);
	const size_t levelBefore = ct->GetLevel();
	auto refreshed			 = cc->EvalBootstrap(ct);
	ASSERT_LT(refreshed->GetLevel(), levelBefore) << "bootstrap gained no levels: it returned the input unchanged, so this test exercised nothing";
	auto got	= HostCt(cc, refreshed);
	auto oracle = LbCc(cc)->EvalBootstrap(lbCt);
	ExpectSlotsNear(cc, keys.secretKey, got, oracle, kSlots, 1e-2);
}
