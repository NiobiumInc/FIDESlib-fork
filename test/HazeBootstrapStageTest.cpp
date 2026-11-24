// Stage-by-stage exactness of the haze ModRaise-first bootstrap against OpenFHE's EvalBootstrap.
//
// HazeEngine::bootstrapModRaiseFirst returns its intermediate after stage k when FIDESLIB_HAZE_BOOT_DEBUG_STAGE=k
// (1 raise, 2 partial sum and rescale, 3 CoeffsToSlots, conjugate fold and rescale, 4 EvalMod and scalar,
// 5 SlotsToCoeffs and sparse fold). The oracle's intermediate at each stage is rebuilt from the same encryption by
// replaying FHECKKSRNS::EvalBootstrap (ckksrns-fhe.cpp) with the calls it makes, so the first stage that differs
// names the op. Haze refuses compute after a readback, so every stage runs in a context of its own.
// Run: FIDESLIB_TEST_BACKEND=haze ./fideslib-test --gtest_filter='HazeBootstrapStageTest.*'

#include <gtest/gtest.h>
#include <openfhe.h>

#include <any>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ParityHelpers.hpp"
#include "TestEngineConfig.hpp"
#include "TestRegistration.hpp"
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;
using FIDESlib::Testing::OnHaze;

namespace {

using LbCtPtr = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using LbCcPtr = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using LbDcrt  = lbcrypto::DCRTPoly;
using Fhe	  = lbcrypto::FHECKKSRNS;

constexpr const char* kStageEnv					= "FIDESLIB_HAZE_BOOT_DEBUG_STAGE";
constexpr int kFirstStage						= 1;
constexpr int kLastStage						= 5;
constexpr uint32_t kRingDim						= 1u << 12;
constexpr uint32_t kScaleMod					= 50;
constexpr uint32_t kFirstMod					= 60;
constexpr uint32_t kNumLargeDigits				= 3;
constexpr uint32_t kLevelsAfter					= 3;
constexpr uint32_t kSparseSlots					= 1u << 5;
constexpr uint32_t kFullSlots					= kRingDim / 2;
constexpr uint32_t kFixedAutoFullPackCorrection = 10; // the derived 9 is below the degree 10 these moduli give

struct StageShape {
	const char* name;
	uint32_t slots;
	std::vector<uint32_t> levelBudget;
	ScalingTechnique scalingTech;
	uint32_t correctionFactor; // 0 derives

	uint32_t depth() const {
		return kLevelsAfter + Fhe::GetBootstrapDepth(levelBudget, lbcrypto::UNIFORM_TERNARY);
	}
};

// FHECKKSRNS::AdjustCiphertext (private in OpenFHE), composite degree 1.
void AdjustCiphertext(LbCtPtr& ciphertext, double correction, uint32_t lvl) {
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(ciphertext->GetCryptoParameters());
	auto cc					= ciphertext->GetCryptoContext();
	auto algo				= cc->GetScheme();
	const auto st			= cryptoParams->GetScalingTechnique();
	if (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) {
		const double targetSF		  = cryptoParams->GetScalingFactorReal(lvl);
		const double sourceSF		  = ciphertext->GetScalingFactor();
		const uint32_t numTowers	  = ciphertext->GetElements()[0].GetNumOfElements();
		const double modToDrop		  = cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
		const double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * correction;
		cc->EvalMultInPlace(ciphertext, adjustmentFactor);
		algo->ModReduceInternalInPlace(ciphertext, 1);
		ciphertext->SetScalingFactor(targetSF);
	} else {
		cc->EvalMultInPlace(ciphertext, correction);
		algo->ModReduceInternalInPlace(ciphertext, 1);
	}
}

// The oracle's ciphertext after each marker of the engine (index = stage, 0 unused), for sparse or fully packed
// UNIFORM_TERNARY keys. Follows ckksrns-fhe.cpp EvalBootstrap (the BTSlotsEncoding = false circuit) call for call.
std::array<LbCtPtr, kLastStage + 1> ReplayOracle(LbCcPtr& cc, const LbCtPtr& input, uint32_t slots) {
	std::array<LbCtPtr, kLastStage + 1> at;
	auto algo = cc->GetScheme();
	auto fhe  = std::dynamic_pointer_cast<Fhe>(algo->m_FHE);
	if (!fhe)
		throw std::runtime_error("ReplayOracle: the scheme's bootstrap object is not FHECKKSRNS");
	const auto& p			= *fhe->m_bootPrecomMap.at(slots);
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(input->GetCryptoParameters());
	const auto st			= cryptoParams->GetScalingTechnique();
	if (cryptoParams->GetSecretKeyDist() != lbcrypto::UNIFORM_TERNARY)
		throw std::runtime_error("ReplayOracle: only UNIFORM_TERNARY keys are replayed");
	const uint32_t N  = cc->GetRingDimension();
	const uint32_t L0 = cryptoParams->GetElementParams()->GetParams().size();

	auto elementParamsRaised = *(cryptoParams->GetElementParams());
	if (st == lbcrypto::FLEXIBLEAUTOEXT)
		elementParamsRaised.PopLastParam();
	const auto& paramsQ = elementParamsRaised.GetParams();
	std::vector<lbcrypto::NativeInteger> moduli(paramsQ.size()), roots(paramsQ.size());
	for (size_t i = 0; i < paramsQ.size(); ++i) {
		moduli[i] = paramsQ[i]->GetModulus();
		roots[i]  = paramsQ[i]->GetRootOfUnity();
	}
	auto elementParamsRaisedPtr = std::make_shared<lbcrypto::ILDCRTParams<LbDcrt::Integer>>(cc->GetCyclotomicOrder(), moduli, roots);

	const double qDouble	  = cryptoParams->GetElementParams()->GetParams()[0]->GetModulus().ConvertToDouble(); // composite degree 1
	const double powP		  = std::pow(2, cryptoParams->GetPlaintextModulus());
	const int32_t deg		  = std::round(std::log2(qDouble / powP));
	const uint32_t correction = fhe->GetCKKSBootCorrectionFactor() - deg;
	const double post		  = std::pow(2, static_cast<double>(deg));
	const double pre		  = 1. / post;
	const uint64_t scalar	  = std::llround(post);
	const bool isLT			  = (p.m_paramsEnc.lvlb == 1) && (p.m_paramsDec.lvlb == 1);
	const bool fullyPacked	  = slots == cc->GetCyclotomicOrder() / 4;

	auto raised = input->Clone();
	algo->ModReduceInternalInPlace(raised, raised->GetNoiseScaleDeg() - 1);
	AdjustCiphertext(raised, std::pow(2, -static_cast<int32_t>(correction)), st != lbcrypto::FLEXIBLEAUTOEXT ? 0 : 1);
	for (auto& dcrt : raised->GetElements()) {
		dcrt.SetFormat(::COEFFICIENT);
		LbDcrt tmp(dcrt.GetElementAtIndex(0), elementParamsRaisedPtr);
		tmp.SetFormat(::EVALUATION);
		dcrt = std::move(tmp);
	}
	raised->SetLevel(L0 - raised->GetElements()[0].GetNumOfElements());
	at[1] = raised->Clone();

	cc->EvalMultInPlace(raised, pre * (1.0 / (Fhe::K_UNIFORM * N)));
	if (!fullyPacked)
		Fhe::EvalPartialSumInPlace(raised, slots);
	algo->ModReduceInternalInPlace(raised, 1);
	at[2] = raised->Clone();

	auto ctxtEnc	 = isLT ? fhe->EvalLinearTransform(p.m_U0hatTPre, raised) : fhe->EvalCoeffsToSlots(p.m_U0hatTPreFFT, raised);
	auto& evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
	LbCtPtr ctxtEncI;
	if (fullyPacked) {
		auto conj = Fhe::Conjugate(ctxtEnc, evalKeyMap);
		ctxtEncI  = cc->EvalSub(ctxtEnc, conj);
		cc->EvalAddInPlace(ctxtEnc, conj);
		algo->MultByMonomialInPlace(ctxtEncI, 3 * slots);
	} else {
		cc->EvalAddInPlace(ctxtEnc, Fhe::Conjugate(ctxtEnc, evalKeyMap));
	}
	if (st == lbcrypto::FIXEDMANUAL) {
		while (ctxtEnc->GetNoiseScaleDeg() > 1) {
			cc->ModReduceInPlace(ctxtEnc);
			if (fullyPacked)
				cc->ModReduceInPlace(ctxtEncI);
		}
	} else if (ctxtEnc->GetNoiseScaleDeg() == 2) {
		algo->ModReduceInternalInPlace(ctxtEnc, 1);
		if (fullyPacked)
			algo->ModReduceInternalInPlace(ctxtEncI, 1);
	}
	at[3] = ctxtEnc->Clone();

	auto evalMod = [&](LbCtPtr& c) {
		c = algo->EvalChebyshevSeries(c, Fhe::g_coefficientsUniform, -1.0, 1.0);
		if (st != lbcrypto::FIXEDMANUAL)
			algo->ModReduceInternalInPlace(c, 1);
		fhe->ApplyDoubleAngleIterations(c, Fhe::R_UNIFORM);
	};
	if (fullyPacked)
		evalMod(ctxtEncI);
	evalMod(ctxtEnc);
	if (fullyPacked) {
		algo->MultByMonomialInPlace(ctxtEncI, slots);
		cc->EvalAddInPlaceNoCheck(ctxtEnc, ctxtEncI);
	}
	algo->MultByIntegerInPlace(ctxtEnc, scalar);
	at[4] = ctxtEnc->Clone();

	if (st != lbcrypto::FIXEDMANUAL)
		algo->ModReduceInternalInPlace(ctxtEnc, 1);
	auto ctxtDec = isLT ? fhe->EvalLinearTransform(p.m_U0Pre, ctxtEnc) : fhe->EvalSlotsToCoeffs(p.m_U0PreFFT, ctxtEnc);
	if (!fullyPacked)
		cc->EvalAddInPlaceNoCheck(ctxtDec, cc->EvalRotate(ctxtDec, slots));
	at[5] = ctxtDec->Clone();
	return at;
}

std::string Describe(const LbCtPtr& ct) {
	char line[192];
	std::snprintf(line,
	  sizeof line,
	  "{level %zu nsd %zu sf %.17g towers %u slots %u}",
	  ct->GetLevel(),
	  ct->GetNoiseScaleDeg(),
	  ct->GetScalingFactor(),
	  ct->GetElements().at(0).GetNumOfElements(),
	  ct->GetSlots());
	return line;
}

// log2 of the largest centred coefficient-domain difference in limb 0 of one component: near the modulus size for
// unrelated values, a few bits for a rounding difference.
double LimbZeroDifferenceBits(const LbCtPtr& oracle, const LbCtPtr& got, size_t component) {
	LbDcrt diff = oracle->GetElements().at(component) - got->GetElements().at(component);
	diff.SetFormat(::COEFFICIENT);
	const auto& limb = diff.GetElementAtIndex(0);
	const uint64_t q = limb.GetModulus().ConvertToInt();
	uint64_t worst	 = 0;
	for (size_t k = 0; k < limb.GetValues().GetLength(); ++k) {
		const uint64_t x = limb.GetValues()[k].ConvertToInt();
		worst			 = std::max(worst, x > q / 2 ? q - x : x);
	}
	return worst == 0 ? -1.0 : std::log2(static_cast<double>(worst));
}

// Whether the two ciphertexts agree in every limb of both components and in level, noise scale degree, scaling
// factor and slot count; on a mismatch prints the first differing limb, the differing-limb counts and both sides' metadata.
bool StageIsExact(const std::string& label, const LbCtPtr& oracle, const LbCtPtr& got) {
	const bool sameShape =
	  oracle->GetElements().size() == got->GetElements().size() && oracle->GetElements().at(0).GetNumOfElements() == got->GetElements().at(0).GetNumOfElements();
	const bool sameMeta = oracle->GetLevel() == got->GetLevel() && oracle->GetNoiseScaleDeg() == got->GetNoiseScaleDeg() &&
	  oracle->GetScalingFactor() == got->GetScalingFactor() && oracle->GetSlots() == got->GetSlots();
	std::string detail;
	bool limbsEqual = sameShape;
	if (sameShape) {
		for (size_t j = 0; j < oracle->GetElements().size(); ++j) {
			const auto& a		  = oracle->GetElements().at(j).GetAllElements();
			const auto& b		  = got->GetElements().at(j).GetAllElements();
			size_t limbsDiffering = 0;
			for (size_t i = 0; i < a.size(); ++i) {
				size_t differing = 0, first = 0;
				for (size_t k = 0; k < a[i].GetValues().GetLength(); ++k) {
					if (a[i].GetValues()[k] != b[i].GetValues()[k] && differing++ == 0)
						first = k;
				}
				if (differing == 0)
					continue;
				if (limbsDiffering++ == 0 && detail.empty()) {
					char buf[160];
					std::snprintf(buf,
					  sizeof buf,
					  " first differing limb: component %zu limb %zu, %zu of %zu coefficients, first at %zu;",
					  j,
					  i,
					  differing,
					  a[i].GetValues().GetLength(),
					  first);
					detail = buf;
				}
			}
			char buf[96];
			std::snprintf(
			  buf, sizeof buf, " c%zu limbs differing %zu/%zu (limb-0 difference 2^%.1f);", j, limbsDiffering, a.size(), LimbZeroDifferenceBits(oracle, got, j));
			detail += buf;
			limbsEqual = limbsEqual && limbsDiffering == 0;
		}
	} else {
		detail = " limb structure differs;";
	}
	const bool exact = sameMeta && limbsEqual;
	std::cout << "[stage] " << label << ": "
			  << (exact ? "exact" : "diverges" + detail + "\n          oracle " + Describe(oracle) + "\n          haze   " + Describe(got)) << std::endl;
	return exact;
}

// Sets FIDESLIB_HAZE_BOOT_DEBUG_STAGE for the life of the object and restores its previous value, or its absence.
class ScopedStage {
  public:
	explicit ScopedStage(int stage) {
		if (const char* previous = std::getenv(kStageEnv); previous != nullptr) {
			previous_ = previous;
		}
		::setenv(kStageEnv, std::to_string(stage).c_str(), 1);
	}

	~ScopedStage() {
		if (previous_.has_value()) {
			::setenv(kStageEnv, previous_->c_str(), 1);
		} else {
			::unsetenv(kStageEnv);
		}
	}

	ScopedStage(const ScopedStage&)			   = delete;
	ScopedStage& operator=(const ScopedStage&) = delete;

  private:
	std::optional<std::string> previous_;
};

class HazeBootstrapStageTest : public ::testing::Test {
  protected:
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	std::vector<double> v1;

	void Build(const StageShape& s) {
		v1.resize(s.slots);
		for (uint32_t i = 0; i < s.slots; ++i)
			v1[i] = 0.25 * std::sin(0.01 * static_cast<double>(i)) + 0.5;
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(s.depth());
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(s.slots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(s.scalingTech);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(kNumLargeDigits);
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
		cc->EvalBootstrapSetup(s.levelBudget, { 0, 0 }, s.slots, s.correctionFactor, /*precompute=*/true, /*btsfirstboot=*/false);
		cc->EvalBootstrapKeyGen(keys.secretKey, s.slots);
		cc->LoadContext(keys.publicKey);
	}

	void RunStage(const StageShape& s, int stage) {
		Build(s);
		// The input sits at the deepest level the bootstrap accepts, so it gains levels.
		auto pt	  = cc->MakeCKKSPackedPlaintext(v1, 1, s.depth() - 1, nullptr, s.slots);
		auto ct	  = cc->Encrypt(pt, keys.publicKey);
		auto lbCt = LbCt(ct);

		Ciphertext<DCRTPoly> out;
		{
			ScopedStage scoped(stage);
			out = cc->EvalBootstrap(ct);
		}
		cc->RecoverHostCiphertext(out);
		const auto oracle = ReplayOracle(LbCc(cc), lbCt, s.slots);
		EXPECT_TRUE(StageIsExact(std::string(s.name) + " stage " + std::to_string(stage), oracle[stage], LbCt(out)))
		  << s.name << ": stage " << stage << " differs from the oracle";
	}

	void RunStages(const StageShape& s) {
		for (int stage = kFirstStage; stage <= kLastStage; ++stage)
			RunStage(s, stage);
	}
};

const StageShape kShapes[] = {
	{ "s32_lvlb1_1_FIXEDAUTO", kSparseSlots, { 1, 1 }, FIXEDAUTO, 0 },
	{ "s32_lvlb1_1_FIXEDMANUAL", kSparseSlots, { 1, 1 }, FIXEDMANUAL, 0 },
	{ "s32_lvlb2_2_FIXEDAUTO", kSparseSlots, { 2, 2 }, FIXEDAUTO, 0 },
	{ "s32_lvlb3_3_FIXEDAUTO", kSparseSlots, { 3, 3 }, FIXEDAUTO, 0 },
	{ "s32_lvlb3_3_FIXEDMANUAL", kSparseSlots, { 3, 3 }, FIXEDMANUAL, 0 },
	{ "full_lvlb3_3_FIXEDAUTO", kFullSlots, { 3, 3 }, FIXEDAUTO, kFixedAutoFullPackCorrection },
};

} // namespace

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Sparse_1_1_FixedAuto) {
	RunStages(kShapes[0]);
}

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Sparse_1_1_FixedManual) {
	RunStages(kShapes[1]);
}

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Sparse_2_2_FixedAuto) {
	RunStages(kShapes[2]);
}

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Sparse_3_3_FixedAuto) {
	RunStages(kShapes[3]);
}

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Sparse_3_3_FixedManual) {
	RunStages(kShapes[4]);
}

FIDES_TEST_F_IF(OnHaze(), HazeBootstrapStageTest, Full_3_3_FixedAuto) {
	RunStages(kShapes[5]);
}
