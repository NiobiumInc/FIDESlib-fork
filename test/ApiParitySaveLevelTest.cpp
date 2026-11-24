// The level-saving StC-first bootstrap (OpenFHE PR #1374) beyond the shape matrix of ApiParityExactBootstrapTest.cpp: the correction a
// context folds at setup against the one applied at each bootstrap, the checks on the input, and the archive format. A device is compared
// with OpenFHE limb for limb, and the oracle is decrypted so that two equally wrong results cannot match.

#include <gtest/gtest.h>
#include <openfhe.h>

#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <any>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ExactBootstrapShared.hpp" // MakeExactContext, ExpectRefusal, ExpectBootstrapExact, ExpectBootstrapPrecision, the small-ring constants
#include "ParityHelpers.hpp"		// LbCc, LbCt, HostCt
#include "TestEngineConfig.hpp"		// FIDESlib::Testing::GetTestBackend
#include "fideslib.hpp"

using namespace fideslib;
using namespace FIDESlib::Testing;

namespace {

// The level saving folds 2^-(factor - degree) into SlotsToCoeffs at setup. These moduli give degree 10 (kSmallFirstMod - kSmallScaleMod),
// so a factor of 12 folds 2^-2, and the factor is then changed on the host before or after the context reaches a device.
constexpr uint32_t kDegree		  = kSmallFirstMod - kSmallScaleMod;
constexpr uint32_t kFoldedFactor  = 12;
constexpr uint32_t kLoweredFactor = 11;
constexpr uint32_t kRaisedFactor  = 13;
constexpr uint32_t kFolded		  = kFoldedFactor - kDegree;
constexpr uint32_t kLowered		  = kLoweredFactor - kDegree;

// A factor below the degree 20 that kRefusalScaleMod gives, which OpenFHE accepts at setup and refuses at bootstrap.
constexpr uint32_t kBelowDegreeFactor = 12;

constexpr uint32_t kPackedSlots = kSmallRing / 2;
constexpr uint32_t kFewSlots	= 1u << 3;

// 1e-6 is far above the 1e-9 tolerance OpenFHE checks the input's scaling factor against.
constexpr double kScalingFactorPerturbation = 1e-6;

// Enough differing bytes to name a field, few enough to read.
constexpr size_t kReportedDifferences = 8;

// The polymorphic type name the scheme section of a context archive starts at.
constexpr const char* kSchemeTypeName = "lbcrypto::SchemeCKKSRNS";

constexpr const char* kRaisedRefusal	  = "raised after EvalBootstrapSetup";
constexpr const char* kReencodedAfterLoad = "re-encoded after the context reached the device";
constexpr const char* kScalingFactorCheck = "differs from the scaling factor of its level";

const std::vector<uint32_t> kLevelBudget = { 3, 3 };

constexpr SecretKeyDist kBaseKeyDist = UNIFORM_TERNARY;
constexpr uint32_t kSingleIteration	 = 1;
constexpr SecretKeyDist kMetaKeyDist = SPARSE_TERNARY;
constexpr uint32_t kMetaIterations	 = 2;

uint32_t SaveLevelDepth(SecretKeyDist keyDist, uint32_t numIterations) {
	return kSmallLevelsAfter + lbcrypto::FHECKKSRNS::GetBootstrapDepth(kLevelBudget, OpenFheKeyDist(keyDist)) + (numIterations - 1);
}

uint32_t SaveLevelInputLevel(ScalingTechnique technique, SecretKeyDist keyDist, uint32_t numIterations) {
	return SaveLevelEncryptLevel(SaveLevelDepth(keyDist, numIterations), kLevelBudget.at(1), technique);
}

struct SmallContextSpec {
	ScalingTechnique technique;
	SecretKeyDist keyDist;
	uint32_t numIterations;
	uint32_t scaleMod;
	uint32_t batchSize;
};

CryptoContext<DCRTPoly> MakeSmallContext(const SmallContextSpec& spec) {
	const ExactContextParams params = { .multiplicativeDepth = SaveLevelDepth(spec.keyDist, spec.numIterations),
		.scaleMod											 = spec.scaleMod,
		.firstMod											 = kSmallFirstMod,
		.batchSize											 = spec.batchSize,
		.ringDim											 = kSmallRing,
		.numLargeDigits										 = kSmallLargeDigits,
		.scalingTech										 = spec.technique,
		.secretKeyDist										 = spec.keyDist };
	return MakeExactContext(params);
}

std::shared_ptr<lbcrypto::FHECKKSRNS> BootstrapScheme(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	return std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
}

// OpenFHE's own reading of the correction pair for `slots` right now, which throws where its bootstrap would.
lbcrypto::FHECKKSRNS::StCFirstCorrection OracleCorrection(CryptoContext<DCRTPoly>& cc, uint32_t slots) {
	auto& host = LbCc(cc);
	return BootstrapScheme(host)->GetStCFirstCorrection(slots, std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(host->GetCryptoParameters()));
}

// A small-ring context set up for the level-saving circuit at kFoldedFactor, keyed for bootstrapping, not yet loaded on a device.
struct SaveLevelContext {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	std::vector<double> values;
	uint32_t encryptLevel;
	Ciphertext<DCRTPoly> ct;
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> lbCt;
};

SaveLevelContext SetUpSaveLevelContext(ScalingTechnique technique, SecretKeyDist keyDist, uint32_t numIterations) {
	SaveLevelContext c;
	c.cc   = MakeSmallContext({ technique, keyDist, numIterations, kSmallScaleMod, kSparseSlots });
	c.keys = c.cc->KeyGen();
	c.cc->EvalMultKeyGen(c.keys.secretKey);
	c.cc->EvalBootstrapSetup(kLevelBudget, { 0, 0 }, kSparseSlots, kFoldedFactor, /*precompute=*/true, /*btsfirstboot=*/true, /*stcFirstSavesLevel=*/true);
	c.cc->EvalBootstrapKeyGen(c.keys.secretKey, kSparseSlots);
	c.values	   = InputValues(kSparseSlots);
	c.encryptLevel = SaveLevelInputLevel(technique, keyDist, numIterations);
	return c;
}

// Encrypts the input at the deepest level the circuit accepts, once, because encryption is randomised, and keeps a clone for the oracle.
void EncryptInput(SaveLevelContext& c) {
	auto pt = c.cc->MakeCKKSPackedPlaintext(c.values, 1, c.encryptLevel, nullptr, kSparseSlots);
	c.ct	= c.cc->Encrypt(pt, c.keys.publicKey);
	c.lbCt	= LbCt(c.ct)->Clone();
}

void LoadAndEncryptInput(SaveLevelContext& c) {
	c.cc->LoadContext(c.keys.publicKey);
	EncryptInput(c);
}

// A bootstrap of the level-saving circuit as the api reaches it: one call, the two-iteration Meta-BTS composition, or the in-place form.
enum class BootstrapPath { Single, MetaBts, InPlace };

const char* PathName(BootstrapPath path) {
	switch (path) {
	case BootstrapPath::Single: return "Single";
	case BootstrapPath::MetaBts: return "MetaBts";
	case BootstrapPath::InPlace: return "InPlace";
	}
	throw std::invalid_argument("PathName: unsupported bootstrap path");
}

// The Meta-BTS path is the sparse-key two-iteration shape s32_lvlb3_3_FLEXIBLEAUTO_sparse_it2_stcsave of ApiParityExactBootstrapTest.cpp.
struct PathSpec {
	SecretKeyDist keyDist;
	uint32_t numIterations;
	uint32_t precision;
};

PathSpec SpecOf(BootstrapPath path) {
	switch (path) {
	case BootstrapPath::Single:
	case BootstrapPath::InPlace: return { kBaseKeyDist, kSingleIteration, kNoPrecision };
	case BootstrapPath::MetaBts: return { kMetaKeyDist, kMetaIterations, kMetaPrecisionFlexible };
	}
	throw std::invalid_argument("SpecOf: unsupported bootstrap path");
}

SaveLevelContext SetUpPathContext(BootstrapPath path) {
	const PathSpec spec = SpecOf(path);
	return SetUpSaveLevelContext(FLEXIBLEAUTO, spec.keyDist, spec.numIterations);
}

// Bootstraps `c.ct` through `path` and returns the result, which is `c.ct` itself for the in-place form.
Ciphertext<DCRTPoly> BootstrapOnPath(SaveLevelContext& c, BootstrapPath path) {
	const PathSpec spec = SpecOf(path);
	if (path == BootstrapPath::InPlace) {
		c.cc->EvalBootstrapInPlace(c.ct, spec.numIterations, spec.precision);
		return c.ct;
	}
	return c.cc->EvalBootstrap(c.ct, spec.numIterations, spec.precision);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> OracleOnPath(SaveLevelContext& c, BootstrapPath path) {
	const PathSpec spec = SpecOf(path);
	return LbCc(c.cc)->EvalBootstrap(c.lbCt, spec.numIterations, spec.precision);
}

// The device result is limb-exact against OpenFHE's and decrypts to the input, for a context whose factor has been lowered by now.
void ExpectLoweredFactorApplied(SaveLevelContext& c, BootstrapPath path) {
	const auto pair = OracleCorrection(c.cc, kSparseSlots);
	ASSERT_EQ(pair.folded, kFolded);
	ASSERT_EQ(pair.applied, kLowered) << "the lowered factor did not reach OpenFHE's selector, so this test exercised nothing";

	auto refreshed = BootstrapOnPath(c, path);
	auto oracle	   = OracleOnPath(c, path);
	ASSERT_LT(oracle->GetLevel(), c.lbCt->GetLevel()) << "the bootstrap gained no levels, so this test exercised nothing";
	ExpectBootstrapExact(oracle, HostCt(c.cc, refreshed));
	ExpectBootstrapPrecision(c.cc, c.keys.secretKey, oracle, c.values);
}

// Raising the factor above the folded one needs a re-fold that only setup or EvalBootstrapPrecompute does, so the bootstrap refuses as OpenFHE's does.
void ExpectRaisedFactorRefused(SaveLevelContext& c) {
	ExpectRefusal<lbcrypto::OpenFHEException>("the oracle", kRaisedRefusal, [&] { LbCc(c.cc)->EvalBootstrap(c.lbCt); });
	ExpectRefusal<std::exception>("the device", kRaisedRefusal, [&] { c.cc->EvalBootstrap(c.ct); });
}

} // namespace

class ApiParitySaveLevelCorrectionTest : public ::testing::TestWithParam<ScalingTechnique> {};

// A factor lowered before the context loads is applied as an integer multiply after the last SlotsToCoeffs rescale, exactly as OpenFHE does.
TEST_P(ApiParitySaveLevelCorrectionTest, LoweredFactorIsExact) {
	auto c = SetUpSaveLevelContext(GetParam(), kBaseKeyDist, kSingleIteration);
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kLoweredFactor);
	LoadAndEncryptInput(c);
	ExpectLoweredFactorApplied(c, BootstrapPath::Single);
}

// The applied exponent is read at every bootstrap, so a factor lowered after the device holds the folded matrices takes effect there too.
TEST_P(ApiParitySaveLevelCorrectionTest, CorrectionLoweredAfterLoadIsApplied) {
	auto c = SetUpSaveLevelContext(GetParam(), kBaseKeyDist, kSingleIteration);
	LoadAndEncryptInput(c);
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kLoweredFactor);
	ExpectLoweredFactorApplied(c, BootstrapPath::Single);
}

// The refusal is OpenFHE's own, raised at the bootstrap on every backend whether the factor was raised before the context loaded or after.
TEST_P(ApiParitySaveLevelCorrectionTest, RaisedFactorBeforeLoadIsRefused) {
	auto c = SetUpSaveLevelContext(GetParam(), kBaseKeyDist, kSingleIteration);
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kRaisedFactor);
	LoadAndEncryptInput(c);
	ExpectRaisedFactorRefused(c);
}

TEST_P(ApiParitySaveLevelCorrectionTest, RaisedFactorAfterLoadIsRefused) {
	auto c = SetUpSaveLevelContext(GetParam(), kBaseKeyDist, kSingleIteration);
	LoadAndEncryptInput(c);
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kRaisedFactor);
	ExpectRaisedFactorRefused(c);
}

INSTANTIATE_TEST_SUITE_P(SaveLevelCorrection,
  ApiParitySaveLevelCorrectionTest,
  ::testing::Values(FIXEDAUTO, FIXEDMANUAL, FLEXIBLEAUTO, FLEXIBLEAUTOEXT),
  [](const ::testing::TestParamInfo<ScalingTechnique>& info) { return std::string(ExactTechniqueName(info.param)); });

class ApiParitySaveLevelPathTest : public ::testing::TestWithParam<BootstrapPath> {};

// Each path reads the correction itself, so each must apply a factor lowered after load: a path that skipped the read would run the folded one.
TEST_P(ApiParitySaveLevelPathTest, LoweredAfterLoadIsExact) {
	auto c = SetUpPathContext(GetParam());
	LoadAndEncryptInput(c);
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kLoweredFactor);
	ExpectLoweredFactorApplied(c, GetParam());
}

// A re-fold changes the folded exponent and not the applied one, so only a device, whose matrices are the old fold, must refuse it; the host
// backend is the oracle and bootstraps on the new matrices.
TEST_P(ApiParitySaveLevelPathTest, RefoldAfterLoadIsRefusedOnDevice) {
	auto c = SetUpPathContext(GetParam());
	LbCc(c.cc)->SetCKKSBootCorrectionFactor(kLoweredFactor);
	LoadAndEncryptInput(c);
	const auto before = OracleCorrection(c.cc, kSparseSlots);
	LbCc(c.cc)->EvalBootstrapPrecompute(kSparseSlots);
	const auto after = OracleCorrection(c.cc, kSparseSlots);
	ASSERT_NE(before.folded, after.folded) << "the precompute did not re-fold, so this test exercised nothing";
	ASSERT_EQ(before.applied, after.applied) << "the precompute changed the applied exponent, so this test did not isolate the fold";

	if (GetTestBackend() == TestBackend::CPU) {
		auto refreshed = BootstrapOnPath(c, GetParam());
		ExpectBootstrapPrecision(c.cc, c.keys.secretKey, HostCt(c.cc, refreshed), c.values);
		return;
	}
	ExpectRefusal<std::exception>("the device", kReencodedAfterLoad, [&] { BootstrapOnPath(c, GetParam()); });
}

INSTANTIATE_TEST_SUITE_P(SaveLevelPath,
  ApiParitySaveLevelPathTest,
  ::testing::Values(BootstrapPath::Single, BootstrapPath::MetaBts, BootstrapPath::InPlace),
  [](const ::testing::TestParamInfo<BootstrapPath>& info) { return std::string(PathName(info.param)); });

// A factor left at its default depends on the slot count, so after two setups the last one's factor is not the one the first folded.
// The device must apply the first set's own, as OpenFHE does.
TEST(ApiParitySaveLevelTest, MultiSlotDefaultFactors) {
	const std::vector<double> values = InputValues(kPackedSlots);
	auto cc							 = MakeSmallContext({ FLEXIBLEAUTO, kBaseKeyDist, kSingleIteration, kSmallScaleMod, kPackedSlots });
	auto keys						 = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalBootstrapSetup(kLevelBudget, { 0, 0 }, kPackedSlots, kDerivedCorrection, /*precompute=*/true, /*btsfirstboot=*/true, /*stcFirstSavesLevel=*/true);
	cc->EvalBootstrapSetup(kLevelBudget, { 0, 0 }, kFewSlots, kDerivedCorrection, /*precompute=*/true, /*btsfirstboot=*/true, /*stcFirstSavesLevel=*/true);
	cc->EvalBootstrapKeyGen(keys.secretKey, kPackedSlots);
	cc->EvalBootstrapKeyGen(keys.secretKey, kFewSlots);
	const auto packed = OracleCorrection(cc, kPackedSlots);
	const auto few	  = OracleCorrection(cc, kFewSlots);
	ASSERT_NE(packed.folded, few.folded) << "the two default factors agree, so this test exercised nothing";
	ASSERT_EQ(packed.applied, packed.folded);
	cc->LoadContext(keys.publicKey);

	auto pt		   = cc->MakeCKKSPackedPlaintext(values, 1, SaveLevelInputLevel(FLEXIBLEAUTO, kBaseKeyDist, kSingleIteration), nullptr, kPackedSlots);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	auto lbCt	   = LbCt(ct)->Clone();
	auto refreshed = cc->EvalBootstrap(ct);
	auto oracle	   = LbCc(cc)->EvalBootstrap(lbCt);
	ASSERT_LT(oracle->GetLevel(), lbCt->GetLevel()) << "the bootstrap gained no levels, so this test exercised nothing";
	ExpectBootstrapExact(oracle, HostCt(cc, refreshed));
	ExpectBootstrapPrecision(cc, keys.secretKey, oracle, values);
}

// The scaling factor reaching ModRaise is fixed at setup, so an input whose own differs from its level's is refused, not silently
// mis-scaled. The input stays on the host until the bootstrap so that the edited factor is the one a device uploads.
TEST(ApiParitySaveLevelTest, InputScalingFactorIsChecked) {
	auto c = SetUpSaveLevelContext(FLEXIBLEAUTO, kBaseKeyDist, kSingleIteration);
	c.cc->LoadContext(c.keys.publicKey);
	c.cc->SetAutoLoadCiphertexts(false);
	EncryptInput(c);
	auto host = LbCt(c.ct);
	host->SetScalingFactor(host->GetScalingFactor() * (1.0 + kScalingFactorPerturbation));
	c.lbCt = host->Clone();

	ExpectRefusal<lbcrypto::OpenFHEException>("the oracle", kScalingFactorCheck, [&] { LbCc(c.cc)->EvalBootstrap(c.lbCt); });
	ExpectRefusal<std::exception>("the device", kScalingFactorCheck, [&] { c.cc->EvalBootstrap(c.ct); });
}

// The degree is checked when the bootstrap runs, not at setup or load, so a factor below it is accepted until then on every backend.
TEST(ApiParityBootstrapRefusalTest, SaveLevelBootstrapRefusesDegreeAboveCorrection) {
	auto cc	  = MakeSmallContext({ FIXEDAUTO, kBaseKeyDist, kSingleIteration, kRefusalScaleMod, kSparseSlots });
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	ASSERT_NO_THROW(
	  cc->EvalBootstrapSetup(kLevelBudget, { 0, 0 }, kSparseSlots, kBelowDegreeFactor, /*precompute=*/true, /*btsfirstboot=*/true, /*stcFirstSavesLevel=*/true));
	cc->EvalBootstrapKeyGen(keys.secretKey, kSparseSlots);
	cc->LoadContext(keys.publicKey);

	const std::vector<double> values = InputValues(kSparseSlots);
	auto pt	  = cc->MakeCKKSPackedPlaintext(values, 1, SaveLevelInputLevel(FIXEDAUTO, kBaseKeyDist, kSingleIteration), nullptr, kSparseSlots);
	auto ct	  = cc->Encrypt(pt, keys.publicKey);
	auto lbCt = LbCt(ct)->Clone();
	ExpectRefusal<lbcrypto::OpenFHEException>("the oracle", kDegreeRefusal, [&] { LbCc(cc)->EvalBootstrap(lbCt); });
	ExpectDegreeRefusal("the api level-saving bootstrap", [&] { cc->EvalBootstrap(ct); });
}

namespace {

// OpenFHE's archive of a context is released from its registry first, or a deserialize returns the registered context instead of the archive's.
void ReleaseRegisteredContexts() {
	lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
}

} // namespace

// An archive written before the level saving existed holds no switch, so it must load as a switch-off context and write back as the same
// bytes: the switch rides in a bit of lDec that a switch-off precomputation leaves clear. Only the scheme section is compared, because a
// statically linked OpenFHE (this binary) drops two cereal class-version registrations, which the crypto parameters before it then write as 0.
TEST(ApiParitySaveLevelSerializationTest, SwitchOffArchiveIsByteIdentical) {
	const std::string path = std::string(FIDESLIB_TEST_DATA_DIR) + "/precom-switch-off-ae9797f4.bin";
	std::ifstream file(path, std::ios::binary);
	ASSERT_TRUE(file.is_open()) << path;
	const std::string expected((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	ASSERT_FALSE(expected.empty()) << path;

	ReleaseRegisteredContexts();
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
	ASSERT_TRUE(lbcrypto::Serial::DeserializeFromFile(path, cc, lbcrypto::SerType::BINARY)) << path;
	const auto fhe = BootstrapScheme(cc);
	ASSERT_TRUE(fhe) << "the archive holds no CKKS bootstrapping scheme";
	ASSERT_FALSE(fhe->m_bootPrecomMap.empty()) << "the archive holds no bootstrapping precomputation";
	bool slotsEncoding = false;
	for (const auto& [slots, precom] : fhe->m_bootPrecomMap) {
		EXPECT_FALSE(precom->StCFirstSavesLevel) << "the precomputation for " << slots << " slots loaded as saving a level";
		slotsEncoding = slotsEncoding || precom->BTSlotsEncoding;
	}
	EXPECT_TRUE(slotsEncoding) << "no precomputation is StC-first, so the switch-off StC-first layout was not exercised";

	std::stringstream written;
	lbcrypto::Serial::Serialize(cc, written, lbcrypto::SerType::BINARY);
	const std::string actual = written.str();
	ASSERT_EQ(actual.size(), expected.size()) << "the re-serialized archive changed size";
	const size_t expectedScheme = expected.find(kSchemeTypeName);
	const size_t actualScheme	= actual.find(kSchemeTypeName);
	ASSERT_NE(expectedScheme, std::string::npos) << "the archive holds no " << kSchemeTypeName;
	ASSERT_EQ(actualScheme, expectedScheme);
	std::ostringstream differences;
	size_t differing = 0;
	for (size_t i = expectedScheme; i < expected.size(); ++i) {
		if (actual[i] == expected[i])
			continue;
		if (differing++ < kReportedDifferences)
			differences << " [" << i << "]: " << static_cast<int>(static_cast<unsigned char>(expected[i])) << " -> "
						<< static_cast<int>(static_cast<unsigned char>(actual[i]));
	}
	EXPECT_EQ(differing, 0u) << differing << " of " << expected.size() - expectedScheme << " scheme bytes differ (offset: archive -> rewritten):" << differences.str();
}

// The switch survives an archive, which carries the factor but not the encoded matrices: precomputing again on the reader must fold the
// same correction.
TEST(ApiParitySaveLevelSerializationTest, SwitchOnArchiveRoundTrips) {
	auto cc	   = MakeSmallContext({ FLEXIBLEAUTO, kBaseKeyDist, kSingleIteration, kSmallScaleMod, kSparseSlots });
	auto& host = LbCc(cc);
	host->EvalBootstrapSetup(
	  kLevelBudget, { 0, 0 }, kSparseSlots, kFoldedFactor, /*precompute=*/true, /*BTSlotsEncoding=*/true, /*modEvalLevels=*/-1, /*StCFirstSavesLevel=*/true);
	const auto written = BootstrapScheme(host)->m_bootPrecomMap.at(kSparseSlots);
	ASSERT_TRUE(written->StCFirstSavesLevel);
	ASSERT_NE(written->m_foldedCorrection, 0u) << "nothing is folded, so an equal reading proves nothing";

	std::stringstream archive;
	lbcrypto::Serial::Serialize(host, archive, lbcrypto::SerType::BINARY);
	ReleaseRegisteredContexts();
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> loaded;
	lbcrypto::Serial::Deserialize(loaded, archive, lbcrypto::SerType::BINARY);
	ASSERT_NE(loaded.get(), host.get()) << "the archive resolved to the registered context, so it was not read";

	loaded->EvalBootstrapPrecompute(kSparseSlots);
	const auto read = BootstrapScheme(loaded)->m_bootPrecomMap.at(kSparseSlots);
	EXPECT_TRUE(read->StCFirstSavesLevel);
	EXPECT_EQ(read->m_foldedCorrection, written->m_foldedCorrection);
}
