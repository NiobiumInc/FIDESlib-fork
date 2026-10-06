// Tests for the reducedNoise (FBC variant) plumbing: cpu's lazy key-switch refusal,
// GenCryptoContext's unset-value refusal, and the serialized .dev field's round trip. Deliberately
// separate from ApiTests.cpp so this file stays the one place that exercises engine construction
// and serialization directly against the flag, rather than through a fixture's already-built
// context.
// Run:    ./build/fideslib-test --gtest_filter='ReducedNoise.*'          (CUDA-enabled configure)
// Run:    ./build-cpu/fideslib-cpu-test --gtest_filter='ReducedNoise.*'  (CPU-only configure)

#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h> // getpid
#include <vector>

#include "Serialize.hpp"
#include "TestEngineConfig.hpp" // FIDESlib::Testing::ConfigureTestEngine
#include "engine/Engine.hpp"	// Engine::reducedNoise(), for cc->engine_->reducedNoise()
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;

namespace {

// A unique directory under the system temp root, recursively removed on destruction -- including
// when a test body exits early via an ASSERT_* failure, which unwinds the stack through this
// guard same as any other exception-driven unwind.
class ScopedTempDir {
  public:
	ScopedTempDir() {
		static std::atomic<int> counter{ 0 };
		path_ = std::filesystem::temp_directory_path() / ("fideslib_reduced_noise_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
		std::filesystem::create_directories(path_);
	}

	~ScopedTempDir() {
		std::error_code ec;
		std::filesystem::remove_all(path_, ec);
	}

	ScopedTempDir(const ScopedTempDir&)			   = delete;
	ScopedTempDir& operator=(const ScopedTempDir&) = delete;

	std::filesystem::path file(const std::string& name) const {
		return path_ / name;
	}

  private:
	std::filesystem::path path_;
};

// A minimal, cheap-to-build CCParams: these tests exercise engine construction and
// serialization, not FHE correctness, so depth/ring/slots are the smallest workable values.
// CCParams has no move/copy constructor, so this configures in place rather than returning one.
void ConfigureMinimalParams(CCParams<CryptoContextCKKSRNS>& params) {
	params.SetMultiplicativeDepth(1);
	params.SetScalingModSize(50);
	params.SetBatchSize(8);
	params.SetRingDim(1u << 12);
	params.SetSecurityLevel(HEStd_NotSet); // small ring on purpose; not meant to be secure
}

// A cpu context built with the requested FBC variant, with mult and rotate-by-1 eval keys
// already generated -- the shared fixture for the mismatched-variant tests below.
struct MinimalCpuContext {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
};

MinimalCpuContext BuildMinimalCpuContext(bool reducedNoise) {
	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	params.SetReducedNoise(reducedNoise);
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	return MinimalCpuContext{ std::move(cc), std::move(keys) };
}

// Asserts fn() throws std::runtime_error whose message is requireLinkedVariant's exact text for
// opName ("cpu <opName> cannot honour ..."), not merely a substring match on opName -- which would
// let "evalMult" match the "evalMultInPlace" message too.
void ExpectThrowsNaming(const std::function<void()>& fn, const std::string& opName) {
	try {
		fn();
		FAIL() << "expected " << opName << " to throw";
	} catch (const std::runtime_error& e) {
		const std::string what	 = e.what();
		const std::string needle = "cpu " + opName + " cannot honour";
		EXPECT_NE(what.find(needle), std::string::npos) << "op=" << opName << " what=" << what;
	}
}

// One row per OpenFheEngine method guarded by requireLinkedVariant: the op name it was called
// with, and a callable that drives it (through the public cc facade) far enough to reach the
// guard. A future guarded method is one more row in CpuMismatchedVariantRefusesEveryGuard below.
struct GuardCase {
	std::string opName;
	std::function<void()> call;
};
} // namespace

// MakeEngine(CPU, ...) constructs regardless of variant (see CpuConstructsMismatchedVariant
// below); this just confirms the oracle's own value is one of the values it accepts.
TEST(ReducedNoise, CpuAcceptsOracleVariant) {
	EXPECT_NO_THROW(MakeEngine(Backend::CPU, LinkedOpenFheReducedNoise()));
}

// (a) Construction never checks the variant: cpu has exactly one compiled-in OpenFHE to run, but
// refusal is deferred to the first operation that actually key-switches (see below).
TEST(ReducedNoise, CpuConstructsMismatchedVariant) {
	EXPECT_NO_THROW(MakeEngine(Backend::CPU, !LinkedOpenFheReducedNoise()));
}

// (b) A cpu context built with the non-linked variant still runs a full client workload --
// keygen (including EvalMultKeyGen / EvalRotateKeyGen, which only generate keys), encode,
// encrypt, ct+pt, ct*pt, rescale, decrypt, decode -- because none of these key-switch.
TEST(ReducedNoise, CpuMismatchedVariantRunsClientWorkload) {
	auto ctx = BuildMinimalCpuContext(!LinkedOpenFheReducedNoise());

	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
	auto pt1					 = ctx.cc->MakeCKKSPackedPlaintext(v1);
	auto pt2					 = ctx.cc->MakeCKKSPackedPlaintext(v2);
	auto ct1					 = ctx.cc->Encrypt(pt1, ctx.keys.publicKey);

	ctx.cc->EvalAddInPlace(ct1, pt2);		   // ct + pt: no key-switch
	auto product = ctx.cc->EvalMult(ct1, pt2); // ct * pt: no key-switch
	ctx.cc->RescaleInPlace(product);		   // mod-switch within Q: no key-switch

	Plaintext decrypted;
	ctx.cc->Decrypt(product, ctx.keys.secretKey, &decrypted);
	decrypted->SetLength(v1.size());
	auto out = decrypted->GetRealPackedValue();
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(out[i], (v1[i] + v2[i]) * v2[i], 1e-6) << "slot " << i;
}

// (c) Every OpenFheEngine method guarded by requireLinkedVariant refuses under the mismatched
// variant, naming itself exactly rather than as a substring of a longer op name. One row per
// guarded method (see the guarded-method list in OpenFheEngine.cpp); a future guard is one more row.
TEST(ReducedNoise, CpuMismatchedVariantRefusesEveryGuard) {
	auto ctx = BuildMinimalCpuContext(!LinkedOpenFheReducedNoise());

	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	auto pt1					 = ctx.cc->MakeCKKSPackedPlaintext(v1);
	auto ct1					 = ctx.cc->Encrypt(pt1, ctx.keys.publicKey);
	auto ct2					 = ctx.cc->Encrypt(pt1, ctx.keys.publicKey);

	// EvalMultNoRelin does not key-switch (no relinearize), so it is unguarded and is the one way
	// to mint a real degree-2 ciphertext to drive Relinearize/RelinearizeInPlace below.
	auto ctDeg2a = ctx.cc->EvalMultNoRelin(ct1, ct2);
	auto ctDeg2b = ctx.cc->EvalMultNoRelin(ct1, ct2);

	// A precomp/digits object for the fast-rotation family, minted under a separate matched-variant
	// donor context: every case below throws before touching its contents, so a correctly-typed
	// object from elsewhere serves as well as one from ctx -- whose own EvalFastRotationPrecompute
	// is itself one of the guards under test, so it cannot produce one.
	auto donor		  = BuildMinimalCpuContext(LinkedOpenFheReducedNoise());
	auto donorPt	  = donor.cc->MakeCKKSPackedPlaintext(v1);
	auto donorCt	  = donor.cc->Encrypt(donorPt, donor.keys.publicKey);
	auto dummyPrecomp = donor.cc->EvalFastRotationPrecompute(donorCt);
	const uint32_t m  = ctx.cc->GetCyclotomicOrder();

	std::vector<double> coeffs = { 0.0, 1.0 };
	const std::vector<Plaintext> noPts;
	const std::vector<int> noIndexes;
	auto maskPt = ctx.cc->MakeCKKSPackedPlaintext(v1);

	const std::vector<GuardCase> cases = {
		{ "evalMult", [&] { ctx.cc->EvalMult(ct1, ct2); } },
		{ "evalMultInPlace",
		  [&] {
			  auto a				   = ct1, b = ct2;
			  ctx.cc->EvalMultMutableInPlace(a, b);
		  } },
		{ "relinearize", [&] { ctx.cc->Relinearize(ctDeg2a); } },
		{ "relinearizeInPlace",
		  [&] {
			  auto d				   = ctDeg2b;
			  ctx.cc->RelinearizeInPlace(d);
		  } },
		{ "evalSquare", [&] { ctx.cc->EvalSquare(ct1); } },
		{ "evalSquareInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->EvalSquareInPlace(a);
		  } },
		{ "evalRotate", [&] { ctx.cc->EvalRotate(ct1, 1); } },
		{ "evalRotateInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->EvalRotateInPlace(a, 1);
		  } },
		{ "evalFastRotation", [&] { ctx.cc->EvalFastRotation(ct1, 1, m, dummyPrecomp); } },
		{ "evalFastRotationExt", [&] { ctx.cc->EvalFastRotationExt(ct1, 1, dummyPrecomp, false); } },
		{ "evalFastRotation", [&] { ctx.cc->EvalFastRotation(ct1, std::vector<int32_t>{ 1 }, m, dummyPrecomp); } },
		{ "evalFastRotationExt", [&] { ctx.cc->EvalFastRotationExt(ct1, std::vector<int32_t>{ 1 }, dummyPrecomp, false); } },
		{ "evalFastRotationPrecompute", [&] { ctx.cc->EvalFastRotationPrecompute(ct1); } },
		{ "evalChebyshevSeries", [&] { ctx.cc->EvalChebyshevSeries(ct1, coeffs, -1.0, 1.0); } },
		{ "evalChebyshevSeriesInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->EvalChebyshevSeriesInPlace(a, coeffs, -1.0, 1.0);
		  } },
		{ "accumulateSum", [&] { ctx.cc->AccumulateSum(ct1, 8, 1); } },
		{ "accumulateSumInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->AccumulateSumInPlace(a, 8, 1);
		  } },
		{ "accumulateSumInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->AccumulateSumInPlace(a, 8, 1, 1);
		  } },
		{ "evalBootstrap", [&] { ctx.cc->EvalBootstrap(ct1); } },
		{ "evalBootstrapInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->EvalBootstrapInPlace(a);
		  } },
		{ "convolutionTransformInPlace",
		  [&] {
			  auto a				   = ct1;
			  ctx.cc->ConvolutionTransformInPlace(a, 0, 0, noPts, noIndexes);
		  } },
		{ "specialConvolutionTransformInPlace",
		  [&] {
			  auto a				   = ct1;
			  auto mask				   = maskPt;
			  ctx.cc->SpecialConvolutionTransformInPlace(a, 0, 0, noPts, mask, noIndexes, 1, 0, 0);
		  } },
	};

	for (const auto& c : cases)
		ExpectThrowsNaming(c.call, c.opName);
}

// (d) The matching (linked) variant does none of that throwing, and the key-switching ops it runs
// (ct*ct mult, rotate, relinearize-after-no-relin) produce correct results.
TEST(ReducedNoise, CpuMatchedVariantRunsKeySwitchCorrectly) {
	auto ctx = BuildMinimalCpuContext(LinkedOpenFheReducedNoise());

	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	auto pt1					 = ctx.cc->MakeCKKSPackedPlaintext(v1);
	auto ct1					 = ctx.cc->Encrypt(pt1, ctx.keys.publicKey);
	auto ct2					 = ctx.cc->Encrypt(pt1, ctx.keys.publicKey);

	Ciphertext<DCRTPoly> product, rotated, relinearized;
	EXPECT_NO_THROW(product = ctx.cc->EvalMult(ct1, ct2));
	EXPECT_NO_THROW(rotated = ctx.cc->EvalRotate(ct1, 1));
	EXPECT_NO_THROW(relinearized = ctx.cc->Relinearize(ctx.cc->EvalMultNoRelin(ct1, ct2)));

	Plaintext productPt, rotatedPt, relinearizedPt;
	ctx.cc->Decrypt(product, ctx.keys.secretKey, &productPt);
	ctx.cc->Decrypt(rotated, ctx.keys.secretKey, &rotatedPt);
	ctx.cc->Decrypt(relinearized, ctx.keys.secretKey, &relinearizedPt);
	productPt->SetLength(v1.size());
	rotatedPt->SetLength(v1.size());
	relinearizedPt->SetLength(v1.size());
	auto productOut		 = productPt->GetRealPackedValue();
	auto rotatedOut		 = rotatedPt->GetRealPackedValue();
	auto relinearizedOut = relinearizedPt->GetRealPackedValue();
	for (size_t i = 0; i < v1.size(); ++i) {
		EXPECT_NEAR(productOut[i], v1[i] * v1[i], 1e-6) << "product slot " << i;
		EXPECT_NEAR(rotatedOut[i], v1[(i + 1) % v1.size()], 1e-6) << "rotated slot " << i;
		EXPECT_NEAR(relinearizedOut[i], v1[i] * v1[i], 1e-6) << "relinearized slot " << i;
	}
}

// CCParams::reducedNoise has no default; GenCryptoContext must refuse to build from an unset one
// rather than silently picking a variant.
TEST(ReducedNoise, GenCryptoContextRefusesUnsetReducedNoise) {
	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	// Deliberately no SetReducedNoise call.
	EXPECT_THROW(GenCryptoContext(params), std::runtime_error);
}

// A .dev file predating the ReducedNoise field cannot say which FBC variant the engine was built
// for, so deserializing one must fail rather than silently default.
TEST(ReducedNoise, DeserializeRefusesMissingReducedNoise) {
	ScopedTempDir dir;
	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	ConfigureTestEngine(params);
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);

	const std::string path = dir.file("ctx_missing_reduced_noise").string();
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	{
		std::ifstream in(path + ".dev");
		std::vector<std::string> lines;
		std::string line;
		while (std::getline(in, line))
			lines.push_back(line);
		in.close();
		ASSERT_FALSE(lines.empty());
		ASSERT_EQ(lines.back().rfind(Serial::kReducedNoiseLabel, 0), 0u) << "last .dev line is not " << Serial::kReducedNoiseLabel;
		lines.pop_back();

		std::ofstream out(path + ".dev", std::ios::trunc);
		for (const auto& l : lines)
			out << l << "\n";
	}

	CryptoContext<DCRTPoly> cc2;
	EXPECT_FALSE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
}

// Round-trips a cuda context built with the OPPOSITE of the oracle's variant (cuda has no
// refusal gate, so this is a legal, distinct configuration), and confirms the deserialized
// engine reports the same reducedNoise() the original was built with -- i.e. that the .dev
// field, not some default, is what determines it on read-back.
TEST(ReducedNoise, SerializeDeserializeCudaReducedNoiseRoundTrip) {
	if (!IsBackendAvailable(Backend::CUDA))
		GTEST_SKIP() << "requires a CUDA build";

	ScopedTempDir dir;
	const bool kOppositeOfOracle = !LinkedOpenFheReducedNoise();

	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	params.SetBackend(Backend::CUDA);
	params.SetReducedNoise(kOppositeOfOracle);
	auto cc = GenCryptoContext(params);
	ASSERT_EQ(cc->engine_->reducedNoise(), kOppositeOfOracle);

	const std::string path = dir.file("cuda_reduced_noise_roundtrip").string();
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	CryptoContext<DCRTPoly> cc2;
	ASSERT_TRUE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
	ASSERT_NE(cc2.get(), nullptr);
	EXPECT_EQ(cc2->engine_->reducedNoise(), kOppositeOfOracle);
}
