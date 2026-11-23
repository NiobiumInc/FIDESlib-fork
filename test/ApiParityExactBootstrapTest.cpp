// Exact bootstrap parity: the bootstrap is compared with OpenFHE's limb for limb, level, noise scale degree and
// scaling factor included, not by decrypted value. The StC-first shapes set btsfirstboot, which selects OpenFHE's
// EvalBootstrapStCFirst.

#include <gtest/gtest.h>
#include <openfhe.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ParityHelpers.hpp"	// LbCc, LbCt, HostCt
#include "TestEngineConfig.hpp" // FIDESlib::Testing::ConfigureTestEngine
#include "TestRegistration.hpp" // OnDevice, OnlyIf
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;
using FIDESlib::Testing::OnDevice;
using FIDESlib::Testing::OnlyIf;

// Every limb of both components plus the metadata that places the ciphertext, so a circuit that lands on the right
// bits at the wrong level cannot pass. A mismatch names the first differing limb and never prints a vector.
static void ExpectBootstrapExact(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got) {
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

struct ExactBootstrapShape {
	std::string name; // gtest-safe; empty names the shape after its fields, and the mode (_stc or _raise) is appended either way
	uint32_t ringDim, slots, scaleMod, firstMod, numLargeDigits;
	std::vector<uint32_t> levelBudget;
	ScalingTechnique scalingTech;
	SecretKeyDist secretKeyDist;
	uint32_t correctionFactor; // 0 derives
	uint32_t numIterations, precision;
	bool stcFirst;
	uint32_t levelsAfter;		 // levels the context keeps once the bootstrap has run
	uint32_t levelsBelowDeepest; // 0: the input sits at the deepest level the circuit accepts
};

// Prints the shape by name instead of as a byte dump.
void PrintTo(const ExactBootstrapShape& shape, std::ostream* os) {
	*os << shape.name;
}

namespace {

const char* ExactTechniqueName(ScalingTechnique technique) {
	switch (technique) {
	case FIXEDMANUAL: return "FIXEDMANUAL";
	case FIXEDAUTO: return "FIXEDAUTO";
	case FLEXIBLEAUTO: return "FLEXIBLEAUTO";
	case FLEXIBLEAUTOEXT: return "FLEXIBLEAUTOEXT";
	default: throw std::invalid_argument("ExactTechniqueName: unsupported scaling technique");
	}
}

const char* ExactKeyName(SecretKeyDist dist) {
	switch (dist) {
	case UNIFORM_TERNARY: return "uniform";
	case SPARSE_TERNARY: return "sparse";
	case SPARSE_ENCAPSULATED: return "encaps";
	default: throw std::invalid_argument("ExactKeyName: unsupported secret key distribution");
	}
}

// OpenFHE's enumerator for a fideslib one, which the two enums do not guarantee to share a value for.
lbcrypto::SecretKeyDist ExactOpenFheKeyDist(SecretKeyDist dist) {
	switch (dist) {
	case UNIFORM_TERNARY: return lbcrypto::UNIFORM_TERNARY;
	case SPARSE_TERNARY: return lbcrypto::SPARSE_TERNARY;
	case SPARSE_ENCAPSULATED: return lbcrypto::SPARSE_ENCAPSULATED;
	default: throw std::invalid_argument("ExactOpenFheKeyDist: unsupported secret key distribution");
	}
}

// Spells every field that varies across the small-ring matrix, so no two of its shapes share a name.
std::string ExactNameFromFields(const ExactBootstrapShape& shape) {
	std::string name = (shape.slots == shape.ringDim / 2 ? std::string("full") : "s" + std::to_string(shape.slots)) + "_lvlb" +
	  std::to_string(shape.levelBudget.at(0)) + "_" + std::to_string(shape.levelBudget.at(1)) + "_" + ExactTechniqueName(shape.scalingTech) + "_" +
	  ExactKeyName(shape.secretKeyDist);
	if (shape.numIterations > 1)
		name += "_it" + std::to_string(shape.numIterations);
	if (shape.correctionFactor != 0)
		name += "_corr" + std::to_string(shape.correctionFactor);
	if (shape.levelsBelowDeepest > 0)
		name += "_dep" + std::to_string(shape.levelsBelowDeepest);
	return name;
}

// Names the shape and appends it, refusing a name that is already taken.
void AddShape(std::vector<ExactBootstrapShape>& shapes, ExactBootstrapShape shape) {
	shape.name = (shape.name.empty() ? ExactNameFromFields(shape) : shape.name) + (shape.stcFirst ? "_stc" : "_raise");
	for (const auto& existing : shapes) {
		if (existing.name == shape.name)
			throw std::logic_error("duplicate exact bootstrap shape name: " + shape.name);
	}
	shapes.push_back(std::move(shape));
}

constexpr uint32_t kSmallRing			  = 1u << 12;
constexpr uint32_t kSmallScaleMod		  = 50;
constexpr uint32_t kSmallFirstMod		  = 60;
constexpr uint32_t kSmallLargeDigits	  = 3;
constexpr uint32_t kSmallLevelsAfter	  = 3;
constexpr uint32_t kSparseSlots			  = 1u << 5;
constexpr uint32_t kFullSlots			  = kSmallRing / 2;
constexpr uint32_t kLargeRing			  = 1u << 16;
constexpr uint32_t kDerivedCorrection	  = 0;
constexpr uint32_t kNoPrecision			  = 0;
constexpr uint32_t kMetaPrecisionFlexible = 7;
constexpr uint32_t kMetaPrecisionManual	  = 8;
// OpenFHE derives 9 for this ring and packing (ckksrns-fhe.cpp:165-167), below the degree 10 these moduli give, so the derived
// value is refused; 10 is the O6d precedent.
constexpr uint32_t kFixedAutoFullPackCorrection = 10;
// The derived FIXED* factor equals the degree at {1,1} and 32 slots, which makes the final correction multiply one; 12 makes it four.
constexpr uint32_t kCorrectionAboveDegree = 12;
// One level above the deepest input reaches the EvalMult-and-rescale branch of the FLEXIBLE deplete, three reach the LevelReduce truncation as
// well, and that needs levelsAfter 8 for the input to still gain levels under every technique.
constexpr uint32_t kShallowDeplete		   = 1;
constexpr uint32_t kDeepDeplete			   = 3;
constexpr uint32_t kDeepDepleteLevelsAfter = 8;

std::vector<ExactBootstrapShape> BuildSmallShapes() {
	const std::vector<ScalingTechnique> fourTechniques = { FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDMANUAL };
	const std::vector<SecretKeyDist> sparseKeys		   = { SPARSE_TERNARY, SPARSE_ENCAPSULATED };
	std::vector<ExactBootstrapShape> shapes;

	// A: the ModRaise-first baseline, sparse packing.
	for (const auto& budget : std::vector<std::vector<uint32_t>>{ { 1, 1 }, { 2, 2 }, { 3, 3 } }) {
		for (const auto tech : fourTechniques) {
			AddShape(shapes,
			  { .ringDim			= kSmallRing,
				.slots				= kSparseSlots,
				.scaleMod			= kSmallScaleMod,
				.firstMod			= kSmallFirstMod,
				.numLargeDigits		= kSmallLargeDigits,
				.levelBudget		= budget,
				.scalingTech		= tech,
				.secretKeyDist		= UNIFORM_TERNARY,
				.correctionFactor	= kDerivedCorrection,
				.numIterations		= 1,
				.precision			= kNoPrecision,
				.stcFirst			= false,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0 });
		}
	}
	// A: one with a correction factor above the degree.
	AddShape(shapes,
	  { .ringDim			= kSmallRing,
		.slots				= kSparseSlots,
		.scaleMod			= kSmallScaleMod,
		.firstMod			= kSmallFirstMod,
		.numLargeDigits		= kSmallLargeDigits,
		.levelBudget		= { 1, 1 },
		.scalingTech		= FIXEDAUTO,
		.secretKeyDist		= UNIFORM_TERNARY,
		.correctionFactor	= kCorrectionAboveDegree,
		.numIterations		= 1,
		.precision			= kNoPrecision,
		.stcFirst			= false,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0 });
	// A': ModRaise-first, fully packed.
	AddShape(shapes,
	  { .ringDim			= kSmallRing,
		.slots				= kFullSlots,
		.scaleMod			= kSmallScaleMod,
		.firstMod			= kSmallFirstMod,
		.numLargeDigits		= kSmallLargeDigits,
		.levelBudget		= { 3, 3 },
		.scalingTech		= FIXEDAUTO,
		.secretKeyDist		= UNIFORM_TERNARY,
		.correctionFactor	= kFixedAutoFullPackCorrection,
		.numIterations		= 1,
		.precision			= kNoPrecision,
		.stcFirst			= false,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0 });
	AddShape(shapes,
	  { .ringDim			= kSmallRing,
		.slots				= kFullSlots,
		.scaleMod			= kSmallScaleMod,
		.firstMod			= kSmallFirstMod,
		.numLargeDigits		= kSmallLargeDigits,
		.levelBudget		= { 3, 3 },
		.scalingTech		= FLEXIBLEAUTO,
		.secretKeyDist		= UNIFORM_TERNARY,
		.correctionFactor	= kDerivedCorrection,
		.numIterations		= 1,
		.precision			= kNoPrecision,
		.stcFirst			= false,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0 });
	// A'': ModRaise-first, sparse secret keys.
	for (const auto key : sparseKeys) {
		AddShape(shapes,
		  { .ringDim			= kSmallRing,
			.slots				= kSparseSlots,
			.scaleMod			= kSmallScaleMod,
			.firstMod			= kSmallFirstMod,
			.numLargeDigits		= kSmallLargeDigits,
			.levelBudget		= { 3, 3 },
			.scalingTech		= FLEXIBLEAUTO,
			.secretKeyDist		= key,
			.correctionFactor	= kDerivedCorrection,
			.numIterations		= 1,
			.precision			= kNoPrecision,
			.stcFirst			= false,
			.levelsAfter		= kSmallLevelsAfter,
			.levelsBelowDeepest = 0 });
	}

	// B: StC-first, sparse packing.
	for (const auto& budget : std::vector<std::vector<uint32_t>>{ { 1, 1 }, { 2, 2 }, { 3, 3 } }) {
		for (const auto tech : fourTechniques) {
			AddShape(shapes,
			  { .ringDim			= kSmallRing,
				.slots				= kSparseSlots,
				.scaleMod			= kSmallScaleMod,
				.firstMod			= kSmallFirstMod,
				.numLargeDigits		= kSmallLargeDigits,
				.levelBudget		= budget,
				.scalingTech		= tech,
				.secretKeyDist		= UNIFORM_TERNARY,
				.correctionFactor	= kDerivedCorrection,
				.numIterations		= 1,
				.precision			= kNoPrecision,
				.stcFirst			= true,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0 });
		}
	}
	// B': StC-first, sparse secret keys, one and two iterations.
	const std::pair<uint32_t, uint32_t> iterationRuns[] = { { 1, kNoPrecision }, { 2, kMetaPrecisionFlexible } };
	for (const auto key : sparseKeys) {
		for (const auto& [iterations, precision] : iterationRuns) {
			AddShape(shapes,
			  { .ringDim			= kSmallRing,
				.slots				= kSparseSlots,
				.scaleMod			= kSmallScaleMod,
				.firstMod			= kSmallFirstMod,
				.numLargeDigits		= kSmallLargeDigits,
				.levelBudget		= { 3, 3 },
				.scalingTech		= FLEXIBLEAUTO,
				.secretKeyDist		= key,
				.correctionFactor	= kDerivedCorrection,
				.numIterations		= iterations,
				.precision			= precision,
				.stcFirst			= true,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0 });
		}
	}

	// C: StC-first, fully packed.
	for (const auto& budget : std::vector<std::vector<uint32_t>>{ { 1, 1 }, { 3, 3 }, { 4, 3 }, { 4, 4 } }) {
		for (const auto tech : fourTechniques) {
			for (const auto key : { UNIFORM_TERNARY, SPARSE_ENCAPSULATED }) {
				AddShape(shapes,
				  { .ringDim			= kSmallRing,
					.slots				= kFullSlots,
					.scaleMod			= kSmallScaleMod,
					.firstMod			= kSmallFirstMod,
					.numLargeDigits		= kSmallLargeDigits,
					.levelBudget		= budget,
					.scalingTech		= tech,
					.secretKeyDist		= key,
					.correctionFactor	= kDerivedCorrection,
					.numIterations		= 1,
					.precision			= kNoPrecision,
					.stcFirst			= true,
					.levelsAfter		= kSmallLevelsAfter,
					.levelsBelowDeepest = 0 });
			}
		}
	}
	// C': StC-first on an input that needs depleting, by one tower and by three.
	for (const auto slots : { kFullSlots, kSparseSlots }) {
		for (const auto tech : fourTechniques) {
			AddShape(shapes,
			  { .ringDim			= kSmallRing,
				.slots				= slots,
				.scaleMod			= kSmallScaleMod,
				.firstMod			= kSmallFirstMod,
				.numLargeDigits		= kSmallLargeDigits,
				.levelBudget		= { 3, 3 },
				.scalingTech		= tech,
				.secretKeyDist		= UNIFORM_TERNARY,
				.correctionFactor	= kDerivedCorrection,
				.numIterations		= 1,
				.precision			= kNoPrecision,
				.stcFirst			= true,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = kShallowDeplete });
		}
		for (const auto tech : fourTechniques) {
			AddShape(shapes,
			  { .ringDim			= kSmallRing,
				.slots				= slots,
				.scaleMod			= kSmallScaleMod,
				.firstMod			= kSmallFirstMod,
				.numLargeDigits		= kSmallLargeDigits,
				.levelBudget		= { 3, 3 },
				.scalingTech		= tech,
				.secretKeyDist		= UNIFORM_TERNARY,
				.correctionFactor	= kDerivedCorrection,
				.numIterations		= 1,
				.precision			= kNoPrecision,
				.stcFirst			= true,
				.levelsAfter		= kDeepDepleteLevelsAfter,
				.levelsBelowDeepest = kDeepDeplete });
		}
	}
	// C'': StC-first Meta-BTS.
	AddShape(shapes,
	  { .ringDim			= kSmallRing,
		.slots				= kFullSlots,
		.scaleMod			= kSmallScaleMod,
		.firstMod			= kSmallFirstMod,
		.numLargeDigits		= kSmallLargeDigits,
		.levelBudget		= { 4, 3 },
		.scalingTech		= FLEXIBLEAUTO,
		.secretKeyDist		= SPARSE_ENCAPSULATED,
		.correctionFactor	= kDerivedCorrection,
		.numIterations		= 2,
		.precision			= kMetaPrecisionFlexible,
		.stcFirst			= true,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0 });
	AddShape(shapes,
	  { .ringDim			= kSmallRing,
		.slots				= kFullSlots,
		.scaleMod			= kSmallScaleMod,
		.firstMod			= kSmallFirstMod,
		.numLargeDigits		= kSmallLargeDigits,
		.levelBudget		= { 4, 3 },
		.scalingTech		= FIXEDMANUAL,
		.secretKeyDist		= SPARSE_ENCAPSULATED,
		.correctionFactor	= kDerivedCorrection,
		.numIterations		= 2,
		.precision			= kMetaPrecisionManual,
		.stcFirst			= true,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0 });
	return shapes;
}

// The ModRaise-first twin of a row: the same context, with the input at the deepest level the baseline circuit accepts.
ExactBootstrapShape RaiseTwin(ExactBootstrapShape row) {
	row.stcFirst		   = false;
	row.levelsBelowDeepest = 0;
	return row;
}

// The production-ring shapes (N = 65536), each row as its StC-first shape and its ModRaise-first twin.
std::vector<ExactBootstrapShape> BuildLargeShapes() {
	const std::vector<ExactBootstrapShape> rows = {
		{ .name				  = "upstream_r1",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 56,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 5,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 5,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r2",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 54,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 7,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 10,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r3",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 50,
		  .firstMod			  = 56,
		  .numLargeDigits	  = 6,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 2,
		  .precision		  = kMetaPrecisionFlexible,
		  .stcFirst			  = true,
		  .levelsAfter		  = 9,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r5",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 52,
		  .firstMod			  = 58,
		  .numLargeDigits	  = 6,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FIXEDMANUAL,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 2,
		  .precision		  = kMetaPrecisionManual,
		  .stcFirst			  = true,
		  .levelsAfter		  = 8,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r7",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 12,
		  .scaleMod			  = 59,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 6,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 7,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r8",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 5,
		  .scaleMod			  = 59,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 6,
		  .levelBudget		  = { 1, 1 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 11,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r9",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 58,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 9,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = UNIFORM_TERNARY,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 4,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r10",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 55,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 3,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = UNIFORM_TERNARY,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 1,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "upstream_r11",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 14,
		  .scaleMod			  = 50,
		  .firstMod			  = 53,
		  .numLargeDigits	  = 7,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_TERNARY,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 10,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "sweep_a",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 59,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 5,
		  .levelBudget		  = { 3, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 3,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "sweep_b",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 46,
		  .firstMod			  = 47,
		  .numLargeDigits	  = 6,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = 5,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 13,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "sweep_c",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 59,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 5,
		  .levelBudget		  = { 4, 4 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = UNIFORM_TERNARY,
		  .correctionFactor	  = kDerivedCorrection,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .stcFirst			  = true,
		  .levelsAfter		  = 3,
		  .levelsBelowDeepest = 0 },
		{ .name				  = "sweep_d",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 50,
		  .firstMod			  = 51,
		  .numLargeDigits	  = 8,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = UNIFORM_TERNARY,
		  .correctionFactor	  = 5,
		  .numIterations	  = 2,
		  .precision		  = kMetaPrecisionFlexible,
		  .stcFirst			  = true,
		  .levelsAfter		  = 9,
		  .levelsBelowDeepest = 0 },
	};
	std::vector<ExactBootstrapShape> shapes;
	for (const auto& row : rows) {
		AddShape(shapes, row);
		AddShape(shapes, RaiseTwin(row));
	}
	return shapes;
}

std::string ExactBootstrapShapeName(const ::testing::TestParamInfo<ExactBootstrapShape>& info) {
	return info.param.name;
}

} // namespace

class ApiParityExactBootstrapTest : public ::testing::TestWithParam<ExactBootstrapShape> {
  protected:
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	std::vector<double> v1;

	// The levels the context keeps after the bootstrap plus the ones it consumes, as upstream sizes its contexts.
	uint32_t Depth() const {
		const ExactBootstrapShape& s = GetParam();
		return s.levelsAfter + lbcrypto::FHECKKSRNS::GetBootstrapDepth(s.levelBudget, ExactOpenFheKeyDist(s.secretKeyDist)) + (s.numIterations - 1);
	}

	// The StC-first circuit spends its decode budget on the input before the raise, so its deepest input sits that much higher.
	uint32_t EncryptLevel() const {
		const ExactBootstrapShape& s = GetParam();
		const uint32_t deepest		 = Depth() - 1 - (s.stcFirst ? s.levelBudget.at(1) : 0);
		return deepest - s.levelsBelowDeepest;
	}

	void SetUp() override {
		const ExactBootstrapShape& s = GetParam();

		v1.resize(s.slots);
		for (uint32_t i = 0; i < s.slots; ++i)
			v1[i] = 0.25 * std::sin(0.01 * static_cast<double>(i)) + 0.5;

		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(Depth());
		params.SetScalingModSize(s.scaleMod);
		params.SetFirstModSize(s.firstMod);
		params.SetBatchSize(s.slots);
		params.SetRingDim(s.ringDim);
		params.SetScalingTechnique(s.scalingTech);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(s.numLargeDigits);
		params.SetSecretKeyDist(s.secretKeyDist);
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
		cc->EvalBootstrapSetup(s.levelBudget, { 0, 0 }, s.slots, s.correctionFactor, /*precompute=*/true, /*btsfirstboot=*/s.stcFirst);
		cc->EvalBootstrapKeyGen(keys.secretKey, s.slots);
		cc->LoadContext(keys.publicKey);
	}
};

// One input is encrypted once, because encryption is randomised, and bootstrapped on both sides. On the CPU backend `refreshed` is
// itself an independent OpenFHE bootstrap, so the exact comparison there is the determinism check.
TEST_P(ApiParityExactBootstrapTest, EvalBootstrap) {
	const ExactBootstrapShape& s = GetParam();
	auto pt						 = cc->MakeCKKSPackedPlaintext(v1, 1, EncryptLevel(), nullptr, s.slots);
	auto ct						 = cc->Encrypt(pt, keys.publicKey);
	auto lbCt					 = LbCt(ct)->Clone();
	const size_t levelBefore	 = ct->GetLevel();
	auto refreshed				 = cc->EvalBootstrap(ct, s.numIterations, s.precision);
	// A refresh that gained no levels returns its input, and the comparison would hold vacuously.
	ASSERT_LT(refreshed->GetLevel(), levelBefore)
	  << s.name << ": the bootstrap gained no levels (depth " << Depth() << ", encrypted at level " << EncryptLevel() << "), so this test exercised nothing";
	auto oracle = LbCc(cc)->EvalBootstrap(lbCt, s.numIterations, s.precision);

	char line[256];
	std::snprintf(line,
	  sizeof line,
	  "[parity] %s: input level %zu, oracle level %zu nsd %zu scaling factor %.17g towers %zu\n",
	  s.name.c_str(),
	  levelBefore,
	  oracle->GetLevel(),
	  oracle->GetNoiseScaleDeg(),
	  oracle->GetScalingFactor(),
	  oracle->GetElements().at(0).GetNumOfElements());
	std::cout << line;

	ExpectBootstrapExact(oracle, HostCt(cc, refreshed));
}

INSTANTIATE_TEST_SUITE_P(Shapes, ApiParityExactBootstrapTest, ::testing::ValuesIn(BuildSmallShapes()), ExactBootstrapShapeName);

// Production-ring shapes: a case costs tens of seconds on the host oracle alone, so they run on the device backends only.
INSTANTIATE_TEST_SUITE_P(Large, ApiParityExactBootstrapTest, OnlyIf<ExactBootstrapShape>(OnDevice(), ::testing::ValuesIn(BuildLargeShapes())), ExactBootstrapShapeName);
