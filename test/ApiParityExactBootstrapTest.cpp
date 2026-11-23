// Exact bootstrap parity: the bootstrap is compared with OpenFHE's limb for limb, level, noise scale degree and
// scaling factor included, not by decrypted value. The StC-first shapes set btsfirstboot, which selects OpenFHE's
// EvalBootstrapStCFirst.

#include <gtest/gtest.h>
#include <openfhe.h>

#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ExactBootstrapShared.hpp" // Circuit, MakeExactContext, ExpectBootstrapExact, ExpectBootstrapPrecision, ExpectDegreeRefusal, the small-ring constants
#include "ParityHelpers.hpp"		// LbCc, LbCt, HostCt
#include "TestRegistration.hpp" // OnDevice, OnlyIf
#include "fideslib.hpp"

using namespace fideslib;
using namespace FIDESlib::Testing;

struct ExactBootstrapShape {
	std::string name; // gtest-safe base name; empty takes the one spelled from the fields, and the circuit's suffix follows it in the test name
	uint32_t ringDim, slots, scaleMod, firstMod, numLargeDigits;
	std::vector<uint32_t> levelBudget;
	ScalingTechnique scalingTech;
	SecretKeyDist secretKeyDist;
	uint32_t correctionFactor; // 0 derives
	uint32_t numIterations, precision;
	Circuit circuit;			 // StCFirstSaveLevel is EvalBootstrapSetup's StCFirstSavesLevel (OpenFHE PR #1374), so the input sits one level higher
	uint32_t levelsAfter;		 // levels the context keeps once the bootstrap has run
	uint32_t levelsBelowDeepest; // 0: the input sits at the deepest level the circuit accepts
	bool degreeTwoInput;		 // the input is multiplied by 1.0 first, so it arrives at noise degree 2 on both sides
	bool fullHeightInput;		 // the input is encrypted at level 0, the chain top, where a bootstrap gains nothing and returns it

	std::string TestName() const {
		return name + CircuitSuffix(circuit);
	}
};

// Prints the shape by name instead of as a byte dump.
void PrintTo(const ExactBootstrapShape& shape, std::ostream* os) {
	*os << shape.TestName();
}

namespace {

const char* ExactKeyName(SecretKeyDist dist) {
	switch (dist) {
	case UNIFORM_TERNARY: return "uniform";
	case SPARSE_TERNARY: return "sparse";
	case SPARSE_ENCAPSULATED: return "encaps";
	default: throw std::invalid_argument("ExactKeyName: unsupported secret key distribution");
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
	if (shape.degreeTwoInput)
		name += "_deg2";
	if (shape.fullHeightInput)
		name += "_full";
	return name;
}

// Names the shape after its fields when it has no name, then appends it, refusing a test name that is already taken.
void AddShape(std::vector<ExactBootstrapShape>& shapes, ExactBootstrapShape shape) {
	if (shape.name.empty())
		shape.name = ExactNameFromFields(shape);
	for (const auto& existing : shapes) {
		if (existing.TestName() == shape.TestName())
			throw std::logic_error("duplicate exact bootstrap shape name: " + shape.TestName());
	}
	shapes.push_back(std::move(shape));
}

constexpr uint32_t kFullSlots			  = kSmallRing / 2;
constexpr uint32_t kLargeRing			  = 1u << 16;
// OpenFHE derives 9 for this ring and packing (ckksrns-fhe.cpp:165-167), below the degree 10 these moduli give, so the derived
// value is refused; 10 is the O6d precedent.
constexpr uint32_t kFixedAutoFullPackCorrection = 10;
// The derived FIXED* factor equals the degree at {1,1} and 32 slots, which makes the final correction multiply one; 12 makes it four.
constexpr uint32_t kCorrectionAboveDegree = 12;
// One level above the deepest input reaches the EvalMult-and-rescale branch of the FLEXIBLE deplete, three reach the LevelReduce truncation as
// well, and that needs levelsAfter 8 for the input to still gain levels under every technique.
constexpr uint32_t kShallowDeplete = 1;
// A degree-2 input is depleted by its pending rescale, so the oracle accepts it one level shallower than a degree-1 one.
constexpr uint32_t kDegreeTwoDepth		   = 1;
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
				.circuit			= Circuit::ModRaiseFirst,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0,
				.degreeTwoInput		= false,
				.fullHeightInput	= false });
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
		.circuit			= Circuit::ModRaiseFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
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
		.circuit			= Circuit::ModRaiseFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
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
		.circuit			= Circuit::ModRaiseFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
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
			.circuit			= Circuit::ModRaiseFirst,
			.levelsAfter		= kSmallLevelsAfter,
			.levelsBelowDeepest = 0,
			.degreeTwoInput		= false,
			.fullHeightInput	= false });
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
				.circuit			= Circuit::StCFirst,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0,
				.degreeTwoInput		= false,
				.fullHeightInput	= false });
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
				.circuit			= Circuit::StCFirst,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = 0,
				.degreeTwoInput		= false,
				.fullHeightInput	= false });
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
					.circuit			= Circuit::StCFirst,
					.levelsAfter		= kSmallLevelsAfter,
					.levelsBelowDeepest = 0,
					.degreeTwoInput		= false,
					.fullHeightInput	= false });
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
				.circuit			= Circuit::StCFirst,
				.levelsAfter		= kSmallLevelsAfter,
				.levelsBelowDeepest = kShallowDeplete,
				.degreeTwoInput		= false,
				.fullHeightInput	= false });
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
				.circuit			= Circuit::StCFirst,
				.levelsAfter		= kDeepDepleteLevelsAfter,
				.levelsBelowDeepest = kDeepDeplete,
				.degreeTwoInput		= false,
				.fullHeightInput	= false });
		}
	}
	// C''': StC-first on inputs the oracle treats specially: at noise degree 2 (one level shallower, as the pending rescale takes one),
	// at the chain top where a bootstrap gains nothing and returns its input, and with a correction factor above the degree.
	for (const auto tech : fourTechniques) {
		AddShape(shapes,
		  { .ringDim			= kSmallRing,
			.slots				= kSparseSlots,
			.scaleMod			= kSmallScaleMod,
			.firstMod			= kSmallFirstMod,
			.numLargeDigits		= kSmallLargeDigits,
			.levelBudget		= { 3, 3 },
			.scalingTech		= tech,
			.secretKeyDist		= UNIFORM_TERNARY,
			.correctionFactor	= kDerivedCorrection,
			.numIterations		= 1,
			.precision			= kNoPrecision,
			.circuit			= Circuit::StCFirst,
			.levelsAfter		= kSmallLevelsAfter,
			.levelsBelowDeepest = kDegreeTwoDepth,
			.degreeTwoInput		= true,
			.fullHeightInput	= false });
	}
	for (const auto tech : { FLEXIBLEAUTO, FIXEDAUTO, FLEXIBLEAUTOEXT, FIXEDMANUAL }) {
		AddShape(shapes,
		  { .ringDim			= kSmallRing,
			.slots				= kSparseSlots,
			.scaleMod			= kSmallScaleMod,
			.firstMod			= kSmallFirstMod,
			.numLargeDigits		= kSmallLargeDigits,
			.levelBudget		= { 3, 3 },
			.scalingTech		= tech,
			.secretKeyDist		= UNIFORM_TERNARY,
			.correctionFactor	= kDerivedCorrection,
			.numIterations		= 1,
			.precision			= kNoPrecision,
			.circuit			= Circuit::StCFirst,
			.levelsAfter		= kSmallLevelsAfter,
			.levelsBelowDeepest = 0,
			.degreeTwoInput		= false,
			.fullHeightInput	= true });
	}
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
		.circuit			= Circuit::StCFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
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
		.circuit			= Circuit::StCFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
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
		.circuit			= Circuit::StCFirst,
		.levelsAfter		= kSmallLevelsAfter,
		.levelsBelowDeepest = 0,
		.degreeTwoInput		= false,
		.fullHeightInput	= false });
	return shapes;
}

// The ModRaise-first twin of a row: the same context, with the input at the deepest level the baseline circuit accepts.
ExactBootstrapShape RaiseTwin(ExactBootstrapShape row) {
	row.circuit			   = Circuit::ModRaiseFirst;
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 5,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 10,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 9,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 8,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 7,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 11,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 4,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 1,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 10,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 3,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 13,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 3,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
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
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 9,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
		// The first passing set of the 128-bit dcf CPU sweep, at a degree (10) above the old default factor.
		{ .name				  = "dcf_e_3_50",
		  .ringDim			  = kLargeRing,
		  .slots			  = 1u << 15,
		  .scaleMod			  = 50,
		  .firstMod			  = 60,
		  .numLargeDigits	  = 2,
		  .levelBudget		  = { 4, 3 },
		  .scalingTech		  = FLEXIBLEAUTO,
		  .secretKeyDist	  = SPARSE_ENCAPSULATED,
		  .correctionFactor	  = 10,
		  .numIterations	  = 1,
		  .precision		  = kNoPrecision,
		  .circuit			  = Circuit::StCFirst,
		  .levelsAfter		  = 3,
		  .levelsBelowDeepest = 0,
		  .degreeTwoInput	  = false,
		  .fullHeightInput	  = false },
	};
	std::vector<ExactBootstrapShape> shapes;
	for (const auto& row : rows) {
		AddShape(shapes, row);
		AddShape(shapes, RaiseTwin(row));
	}
	return shapes;
}

// Every StC-first shape again with the level saving on, its name ending _stcsave instead of _stc.
std::vector<ExactBootstrapShape> SaveLevelTwins(const std::vector<ExactBootstrapShape>& shapes) {
	std::vector<ExactBootstrapShape> twins;
	for (auto shape : shapes) {
		if (shape.circuit != Circuit::StCFirst)
			continue;
		shape.circuit = Circuit::StCFirstSaveLevel;
		AddShape(twins, std::move(shape));
	}
	return twins;
}

std::string ExactBootstrapShapeName(const ::testing::TestParamInfo<ExactBootstrapShape>& info) {
	return info.param.TestName();
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
		return s.levelsAfter + lbcrypto::FHECKKSRNS::GetBootstrapDepth(s.levelBudget, OpenFheKeyDist(s.secretKeyDist)) + (s.numIterations - 1);
	}

	// The StC-first circuit spends its decode budget on the input before the raise, so its deepest input sits that much higher.
	uint32_t EncryptLevel() const {
		const ExactBootstrapShape& s = GetParam();
		if (s.fullHeightInput)
			return 0;
		const uint32_t deepest = s.circuit == Circuit::StCFirstSaveLevel ? SaveLevelEncryptLevel(Depth(), s.levelBudget.at(1), s.scalingTech) :
																		   Depth() - 1 - (s.circuit == Circuit::StCFirst ? s.levelBudget.at(1) : 0);
		return deepest - s.levelsBelowDeepest;
	}

	void SetUp() override {
		const ExactBootstrapShape& s = GetParam();

		v1 = InputValues(s.slots);

		const ExactContextParams params = { .multiplicativeDepth = Depth(),
			.scaleMod											 = s.scaleMod,
			.firstMod											 = s.firstMod,
			.batchSize											 = s.slots,
			.ringDim											 = s.ringDim,
			.numLargeDigits										 = s.numLargeDigits,
			.scalingTech										 = s.scalingTech,
			.secretKeyDist										 = s.secretKeyDist };

		cc	 = MakeExactContext(params);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup(s.levelBudget,
		  { 0, 0 },
		  s.slots,
		  s.correctionFactor,
		  /*precompute=*/true,
		  /*btsfirstboot=*/s.circuit != Circuit::ModRaiseFirst,
		  /*stcFirstSavesLevel=*/s.circuit == Circuit::StCFirstSaveLevel);
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
	if (s.degreeTwoInput) {
		// The oracle's input is OpenFHE's own constant multiply on the clone taken before the api call, so no device result is read back
		// before the bootstrap (haze refuses compute after a readback); both multiplies are limb-exact against OpenFHE.
		lbCt = LbCc(cc)->EvalMult(lbCt, 1.0);
		ct	 = cc->EvalMult(ct, 1.0);
		ASSERT_EQ(lbCt->GetNoiseScaleDeg(), 2u) << s.TestName() << ": the input did not arrive at noise degree 2, so this shape exercised nothing";
	}
	const size_t levelBefore = ct->GetLevel();
	auto refreshed			 = cc->EvalBootstrap(ct, s.numIterations, s.precision);
	if (s.fullHeightInput) {
		// A chain-top input gains no levels, so the bootstrap returns it unchanged; the exact comparison below checks that against the oracle.
		ASSERT_EQ(refreshed->GetLevel(), levelBefore) << s.TestName() << ": a chain-top input must come back at its own level";
	} else {
		// A refresh that gained no levels returns its input, and the comparison would hold vacuously.
		ASSERT_LT(refreshed->GetLevel(), levelBefore) << s.TestName() << ": the bootstrap gained no levels (depth " << Depth() << ", encrypted at level "
													  << EncryptLevel() << "), so this test exercised nothing";
	}
	auto oracle = LbCc(cc)->EvalBootstrap(lbCt, s.numIterations, s.precision);

	char line[256];
	std::snprintf(line,
	  sizeof line,
	  "[parity] %s: input level %zu, oracle level %zu nsd %zu scaling factor %.17g towers %u\n",
	  s.TestName().c_str(),
	  levelBefore,
	  oracle->GetLevel(),
	  oracle->GetNoiseScaleDeg(),
	  oracle->GetScalingFactor(),
	  oracle->GetElements().at(0).GetNumOfElements());
	std::cout << line;

	ExpectBootstrapExact(oracle, HostCt(cc, refreshed));
	ExpectBootstrapPrecision(cc, keys.secretKey, oracle, v1);
}

INSTANTIATE_TEST_SUITE_P(Shapes, ApiParityExactBootstrapTest, ::testing::ValuesIn(BuildSmallShapes()), ExactBootstrapShapeName);

// Production-ring shapes: a case costs tens of seconds on the host oracle alone, so they run on the device backends only.
INSTANTIATE_TEST_SUITE_P(Large, ApiParityExactBootstrapTest, OnlyIf<ExactBootstrapShape>(OnDevice(), ::testing::ValuesIn(BuildLargeShapes())), ExactBootstrapShapeName);

INSTANTIATE_TEST_SUITE_P(SaveLevel, ApiParityExactBootstrapTest, ::testing::ValuesIn(SaveLevelTwins(BuildSmallShapes())), ExactBootstrapShapeName);

INSTANTIATE_TEST_SUITE_P(LargeSaveLevel, ApiParityExactBootstrapTest, OnlyIf<ExactBootstrapShape>(OnDevice(), ::testing::ValuesIn(SaveLevelTwins(BuildLargeShapes()))), ExactBootstrapShapeName);

namespace {

// The context is the small-ring fixture's, at moduli whose degree exceeds the correction factor; the input sits at the deepest level the
// circuit would accept, so the refusal is the only thing that can stop the bootstrap.
void ExpectBootstrapRefusesDegree(Circuit circuit) {
	if (circuit == Circuit::StCFirstSaveLevel)
		throw std::invalid_argument("ExpectBootstrapRefusesDegree: the level-saving circuit is covered by ApiParitySaveLevelTest.cpp");
	const bool stcFirst						= circuit == Circuit::StCFirst;
	const std::vector<uint32_t> levelBudget = { 3, 3 };
	const uint32_t depth					= kSmallLevelsAfter + lbcrypto::FHECKKSRNS::GetBootstrapDepth(levelBudget, lbcrypto::UNIFORM_TERNARY);
	const uint32_t encryptLevel				= depth - 1 - (stcFirst ? levelBudget.at(1) : 0);

	const ExactContextParams params = { .multiplicativeDepth = depth,
		.scaleMod											 = kRefusalScaleMod,
		.firstMod											 = kSmallFirstMod,
		.batchSize											 = kSparseSlots,
		.ringDim											 = kSmallRing,
		.numLargeDigits										 = kSmallLargeDigits,
		.scalingTech										 = FIXEDAUTO,
		.secretKeyDist										 = UNIFORM_TERNARY };

	auto cc	  = MakeExactContext(params);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, kSparseSlots, kDerivedCorrection, /*precompute=*/true, /*btsfirstboot=*/stcFirst);
	cc->EvalBootstrapKeyGen(keys.secretKey, kSparseSlots);
	cc->LoadContext(keys.publicKey);

	const std::vector<double> v(kSparseSlots, 0.5);
	auto pt	  = cc->MakeCKKSPackedPlaintext(v, 1, encryptLevel, nullptr, kSparseSlots);
	auto ct	  = cc->Encrypt(pt, keys.publicKey);
	auto lbCt = LbCt(ct)->Clone();

	EXPECT_THROW(LbCc(cc)->EvalBootstrap(lbCt), lbcrypto::OpenFHEException);
	ExpectDegreeRefusal(stcFirst ? "the api StC-first bootstrap" : "the api ModRaise-first bootstrap", [&] { cc->EvalBootstrap(ct); });
}

} // namespace

// Runs on every backend: the engines refuse a degree above the correction factor as OpenFHE does, instead of wrapping the unsigned difference.
TEST(ApiParityBootstrapRefusalTest, StcFirstRefusesDegreeAboveCorrection) {
	ExpectBootstrapRefusesDegree(Circuit::StCFirst);
}

TEST(ApiParityBootstrapRefusalTest, ModRaiseFirstRefusesDegreeAboveCorrection) {
	ExpectBootstrapRefusesDegree(Circuit::ModRaiseFirst);
}
