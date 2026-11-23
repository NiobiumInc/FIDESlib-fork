//
// The constants and checks the exact bootstrap parity suites share (ApiParityExactBootstrapTest.cpp, ApiParitySaveLevelTest.cpp): the
// small-ring context every shape is built from, the limb-for-limb comparison with OpenFHE, the precision floor and the refusals.
//
#ifndef FIDESLIB_TEST_EXACTBOOTSTRAPSHARED_HPP
#define FIDESLIB_TEST_EXACTBOOTSTRAPSHARED_HPP

#include <gtest/gtest.h>
#include <openfhe.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ParityHelpers.hpp"	// LbCc
#include "TestEngineConfig.hpp" // ConfigureTestEngine
#include "fideslib.hpp"

namespace FIDESlib::Testing {

inline constexpr uint32_t kSmallRing		 = 1u << 12;
inline constexpr uint32_t kSmallScaleMod	 = 50;
inline constexpr uint32_t kSmallFirstMod	 = 60;
inline constexpr uint32_t kSmallLargeDigits	 = 3;
inline constexpr uint32_t kSmallLevelsAfter	 = 3;
inline constexpr uint32_t kSparseSlots		 = 1u << 5;
inline constexpr uint32_t kDerivedCorrection = 0;
inline constexpr uint32_t kNoPrecision			 = 0;
inline constexpr uint32_t kMetaPrecisionFlexible = 7;
inline constexpr uint32_t kMetaPrecisionManual	 = 8;

// The bootstrap circuit a shape runs. StCFirstSaveLevel is StC-first with EvalBootstrapSetup's StCFirstSavesLevel (OpenFHE PR #1374).
enum class Circuit { ModRaiseFirst, StCFirst, StCFirstSaveLevel };

// The suffix a circuit adds to a shape's base name in its test name.
inline const char* CircuitSuffix(Circuit circuit) {
	switch (circuit) {
	case Circuit::ModRaiseFirst: return "_raise";
	case Circuit::StCFirst: return "_stc";
	case Circuit::StCFirstSaveLevel: return "_stcsave";
	}
	throw std::invalid_argument("CircuitSuffix: unsupported circuit");
}

// OpenFHE's enumerator for a fideslib one, which the two enums do not guarantee to share a value for.
inline lbcrypto::SecretKeyDist OpenFheKeyDist(fideslib::SecretKeyDist dist) {
	switch (dist) {
	case fideslib::UNIFORM_TERNARY: return lbcrypto::UNIFORM_TERNARY;
	case fideslib::SPARSE_TERNARY: return lbcrypto::SPARSE_TERNARY;
	case fideslib::SPARSE_ENCAPSULATED: return lbcrypto::SPARSE_ENCAPSULATED;
	default: throw std::invalid_argument("OpenFheKeyDist: unsupported secret key distribution");
	}
}

// The scaling technique's name as it appears in a test name.
inline const char* ExactTechniqueName(fideslib::ScalingTechnique technique) {
	switch (technique) {
	case fideslib::FIXEDMANUAL: return "FIXEDMANUAL";
	case fideslib::FIXEDAUTO: return "FIXEDAUTO";
	case fideslib::FLEXIBLEAUTO: return "FLEXIBLEAUTO";
	case fideslib::FLEXIBLEAUTOEXT: return "FLEXIBLEAUTOEXT";
	default: throw std::invalid_argument("ExactTechniqueName: unsupported scaling technique");
	}
}

// The plaintext every exact bootstrap shape encrypts: `slots` values in [0.25, 0.75].
inline std::vector<double> InputValues(uint32_t slots) {
	std::vector<double> v(slots);
	for (uint32_t i = 0; i < slots; ++i)
		v[i] = 0.25 * std::sin(0.01 * static_cast<double>(i)) + 0.5;
	return v;
}

// The deepest level a level-saving StC-first input is encrypted at in a context of multiplicative depth `depth` and SlotsToCoeffs level
// budget `decodeBudget`: SlotsToCoeffs ends on the raising tower, and FLEXIBLEAUTOEXT's extra tower no longer rides through it.
inline uint32_t SaveLevelEncryptLevel(uint32_t depth, uint32_t decodeBudget, fideslib::ScalingTechnique technique) {
	return depth - decodeBudget + (technique == fideslib::FLEXIBLEAUTOEXT ? 1 : 0);
}

// What the exact bootstrap suites vary in a context; every other parameter is fixed by the harness.
struct ExactContextParams {
	uint32_t multiplicativeDepth, scaleMod, firstMod, batchSize, ringDim, numLargeDigits;
	fideslib::ScalingTechnique scalingTech;
	fideslib::SecretKeyDist secretKeyDist;
};

// A CKKS context on the active test backend with the bootstrapping features enabled, not yet keyed.
inline fideslib::CryptoContext<fideslib::DCRTPoly> MakeExactContext(const ExactContextParams& p) {
	fideslib::CCParams<fideslib::CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(p.multiplicativeDepth);
	params.SetScalingModSize(p.scaleMod);
	params.SetFirstModSize(p.firstMod);
	params.SetBatchSize(p.batchSize);
	params.SetRingDim(p.ringDim);
	params.SetScalingTechnique(p.scalingTech);
	params.SetKeySwitchTechnique(fideslib::HYBRID);
	params.SetNumLargeDigits(p.numLargeDigits);
	params.SetSecretKeyDist(p.secretKeyDist);
	params.SetSecurityLevel(fideslib::HEStd_NotSet);
	ConfigureTestEngine(params);
	auto cc = fideslib::GenCryptoContext(params);
	cc->Enable(fideslib::PKE);
	cc->Enable(fideslib::KEYSWITCH);
	cc->Enable(fideslib::LEVELEDSHE);
	cc->Enable(fideslib::ADVANCEDSHE);
	cc->Enable(fideslib::FHE);
	return cc;
}

// Every limb of both components plus the metadata that places the ciphertext, so a circuit that lands on the right
// bits at the wrong level cannot pass. A mismatch names the first differing limb and never prints a vector.
inline void ExpectBootstrapExact(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got) {
	EXPECT_EQ(oracle->GetLevel(), got->GetLevel());
	EXPECT_EQ(oracle->GetNoiseScaleDeg(), got->GetNoiseScaleDeg());
	EXPECT_EQ(oracle->GetScalingFactor(), got->GetScalingFactor()); // exact double
	EXPECT_EQ(oracle->GetSlots(), got->GetSlots());
	EXPECT_EQ(oracle->GetEncodingType(), got->GetEncodingType());

	constexpr size_t kComponents = 2;
	if (oracle->GetElements().size() != kComponents || got->GetElements().size() != kComponents) {
		ADD_FAILURE() << "component counts differ from " << kComponents << ": oracle " << oracle->GetElements().size() << ", got " << got->GetElements().size();
		return;
	}
	for (size_t j = 0; j < kComponents; ++j) {
		const auto& limbsOracle = oracle->GetElements().at(j).GetAllElements();
		const auto& limbsGot	= got->GetElements().at(j).GetAllElements();
		if (limbsOracle.size() != limbsGot.size()) {
			ADD_FAILURE() << "component " << j << " has " << limbsOracle.size() << " limbs in the oracle and " << limbsGot.size() << " in the result";
			return;
		}
		for (size_t i = 0; i < limbsOracle.size(); ++i) {
			const auto& a = limbsOracle.at(i).GetValues();
			const auto& b = limbsGot.at(i).GetValues();
			if (a.GetLength() != b.GetLength()) {
				ADD_FAILURE() << "component " << j << ", limb " << i << " has " << a.GetLength() << " coefficients in the oracle and " << b.GetLength() << " in the result";
				return;
			}
			size_t differing = 0, first = 0;
			for (size_t k = 0; k < a.GetLength(); ++k) {
				if (a[k] != b[k] && differing++ == 0)
					first = k;
			}
			if (differing > 0) {
				ADD_FAILURE() << "component " << j << ", limb " << i << " of " << limbsOracle.size() << ": " << differing << " of " << a.GetLength()
							  << " coefficients differ, the first at index " << first;
				return;
			}
		}
	}
}

// A broken SlotsToCoeffs fold decodes at under 2 bits, so this floor separates a working bootstrap from a limb match of two equally wrong ones.
inline constexpr double kMinBootstrapPrecisionBits = 3.0;

// Decrypts the OpenFHE bootstrap result `oracle` with `sk` and requires its precision against `expected` to reach the floor.
inline void ExpectBootstrapPrecision(fideslib::CryptoContext<fideslib::DCRTPoly>& cc,
  const fideslib::PrivateKey<fideslib::DCRTPoly>& sk,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle,
  const std::vector<double>& expected) {
	lbcrypto::Plaintext decrypted;
	LbCc(cc)->Decrypt(std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl), oracle, &decrypted);
	decrypted->SetLength(expected.size());
	const std::vector<double> values = decrypted->GetRealPackedValue();
	ASSERT_EQ(values.size(), expected.size());
	double maxError = 0.0;
	for (size_t i = 0; i < expected.size(); ++i)
		maxError = std::max(maxError, std::abs(values[i] - expected[i]));
	const double bits = -std::log2(maxError);
	EXPECT_GE(bits, kMinBootstrapPrecisionBits) << "the bootstrap decrypts at " << bits << " bits (max abs error " << maxError << ")";
	std::cout << "[parity] precision " << bits << " bits\n";
}

// q0 / 2^40 is 2^20 here, above any correction factor OpenFHE derives for this ring, so every bootstrap circuit must refuse the context.
inline constexpr uint32_t kRefusalScaleMod	= 40;
inline constexpr const char* kDegreeRefusal = "must be less than or equal to the correction factor";

// Calls `call` and expects it to throw an `Exception` whose message holds `text`.
template <typename Exception, typename Call> void ExpectRefusal(const char* what, const char* text, Call&& call) {
	try {
		call();
	} catch (const std::exception& e) {
		EXPECT_NE(dynamic_cast<const Exception*>(&e), nullptr) << what << " threw an unexpected exception type: " << e.what();
		EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << what << " threw without \"" << text << "\": " << e.what();
		return;
	}
	ADD_FAILURE() << what << " returned instead of refusing with \"" << text << "\"";
}

// Calls `call` and expects the engine's own exception, whatever its type, to carry OpenFHE's wording of the refusal.
template <typename Call> void ExpectDegreeRefusal(const char* what, Call&& call) {
	ExpectRefusal<std::exception>(what, kDegreeRefusal, std::forward<Call>(call));
}

} // namespace FIDESlib::Testing

#endif // FIDESLIB_TEST_EXACTBOOTSTRAPSHARED_HPP
