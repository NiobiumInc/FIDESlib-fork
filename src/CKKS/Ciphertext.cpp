//
// Created by carlosad on 24/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include "ConcurrentOps.hpp"
#include "CudaUtils.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include <omp.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>
#include <string>
#if defined(__clang__)
#include <experimental/source_location>
using sc                  = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc                  = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

namespace {
// Operand adjustment (level / scaling-degree matching) can legitimately fail, e.g. a plaintext with fewer RNS limbs
// than the ciphertext or a NoiseLevel-2 plaintext at the ciphertext's level. Report it instead of returning the
// unmodified ciphertext (the previous assert(false) is compiled out in Release builds).
[[noreturn]] void throwAdjustFailure(const char* op, const Ciphertext& c, int operandLevel, int operandNoiseLevel, const char* operandKind) {
	throw std::runtime_error(std::string("FIDESlib::CKKS::Ciphertext::") + op + ": cannot adjust the " + operandKind +
							 " operand to the ciphertext (ciphertext level " + std::to_string(c.getLevel()) + ", NoiseLevel " +
							 std::to_string(c.NoiseLevel) + "; operand level " + std::to_string(operandLevel) + ", NoiseLevel " +
							 std::to_string(operandNoiseLevel) + "). The operand needs at least as many RNS limbs as the ciphertext and a " +
							 "NoiseLevel not above the ciphertext's; encode plaintexts at the ciphertext's level.");
}
} // namespace

bool hoistRotateFused         = true;
constexpr bool RESCALE_DOUBLE = true;

enum OPS {
	NOP,
	ADD,
	ADDPT,
	MULT,
	MULTPT,
	RESCALE,
	ROTATE,
	COPY,
	SQUARE,
	ADDSCALAR,
	MULTSCALAR,
	ADDMULTPT,
	ADDMULTPTINS,
	WSUM,
	WSUMINPUTS,
	CONJUGATE,
	HOISTEDROTATE,
	HOISTEDROTATEOUTS,
	PRECOMPROTATE,
};

constexpr std::array<const char*, 19> opstr{ "                   Noop: ",
                                             "                   HAdd: ",
                                             "                  AddPt: ",
                                             "                   Mult: ",
                                             "                 MultPt: ", // 5
                                             "                Rescale: ",
                                             "                 Rotate: ",
                                             "                   Copy: ",
                                             "                 Square: ",
                                             "              ScalarAdd: ", // 10
                                             "             ScalarMult: ",
                                             "              AddMultPt: ",
                                             "     AddMultPt (inputs): ",
                                             "                   WSum: ",
                                             "          WSum (inputs): ", // 15
                                             "              Conjugate: ",
                                             "          HoistedRotate: ",
                                             "HoistedRotate (outputs): ",
                                             "        PrecompRotate: " };

/// Op tally. One counter per op kind, ATOMIC because ops are issued from more than one host thread
/// once an application overlaps independent branches of its circuit: a `std::map<OPS,int>` was both
/// a lost-increment race and a tree-corrupting insertion race, and the gates that read these
/// numbers compare them for exact equality. Fixed-size, so no node is ever inserted.
std::array<std::atomic<long long>, opstr.size()> op_count{};

Ciphertext::Ciphertext(Ciphertext&& ct_moved) noexcept
	: my_range(std::move(ct_moved.my_range)), keyID(std::move(ct_moved.keyID)), cc_(ct_moved.cc_), cc(*cc_), c0(std::move(ct_moved.c0)),
	  c1(std::move(ct_moved.c1)), c2(std::move(ct_moved.c2)),
	  NoiseFactor(ct_moved.NoiseFactor), NoiseLevel(ct_moved.NoiseLevel), slots(ct_moved.slots) {
}

Ciphertext::Ciphertext(Context& cc)
	: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
	  c0(cc->getAuxilarPoly()), c1(cc->getAuxilarPoly()) {
	c0.dropToLevel(-1);
	c1.dropToLevel(-1);
	c0.SetModUp(false);
	c1.SetModUp(false);
	CudaNvtxStop();
}

Ciphertext::Ciphertext(Context& cc, const RawCipherText& rawct)
	: Ciphertext(cc) {
	this->load(rawct);
}

Ciphertext::~Ciphertext() {
	// LIFO against the pool: c2 was acquired last, so it goes back first.
	if (c2.has_value() && !c2->GPU.empty())
		cc.returnAuxilarPoly(std::move(*c2));
	if (!c1.GPU.empty())
		cc.returnAuxilarPoly(std::move(c1));
	if (!c0.GPU.empty())
		cc.returnAuxilarPoly(std::move(c0));
}

void Ciphertext::acquireC2() {
	if (c2.has_value())
		return;
	c2.emplace(cc.getAuxilarPoly());
	c2->dropToLevel(-1);
	c2->SetModUp(false);
}

void Ciphertext::releaseC2() {
	if (!c2.has_value())
		return;
	if (!c2->GPU.empty())
		cc.returnAuxilarPoly(std::move(*c2));
	c2.reset();
}

void Ciphertext::requireDegree1(const char* op) const {
	if (c2.has_value())
		throw std::runtime_error(std::string(op) + ": ciphertext should be relinearized before (has 3 components; call Relinearize)");
}

void Ciphertext::requireAdjustedPt(bool ok, const Plaintext& adjusted, const Plaintext& original, const char* op) const {
	const auto where = [](int level, int noiseLevel) {
		return "level " + std::to_string(level) + " depth " + std::to_string(noiseLevel);
	};
	const std::string target = " ciphertext at " + where(getLevel(), NoiseLevel);
	if (!ok)
		throw std::runtime_error(std::string(op) + ": cannot adjust plaintext at " + where(original.c0.getLevel(), original.NoiseLevel) + " to" + target +
		  " (a plaintext can only be scaled up; rescale the ciphertext first)");
	// The caller re-enters with the adjusted plaintext and expects to take the direct path, so a
	// residual mismatch would recurse rather than fail.
	if (adjusted.c0.getLevel() != getLevel() || adjusted.NoiseLevel != NoiseLevel)
		throw std::runtime_error(std::string(op) + ": plaintext adjust left the operands mismatched — plaintext at " + where(adjusted.c0.getLevel(), adjusted.NoiseLevel) + " vs" + target);
}

void Ciphertext::copyMetadata(const Ciphertext& a) {
	assert(this->getLevel() == a.getLevel());
	this->slots       = a.slots;
	this->keyID       = a.keyID;
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::addMetadata(const Ciphertext& a, const Ciphertext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.getLevel());
		// assert(abs(a.NoiseFactor - b.NoiseFactor) < a.NoiseFactor / 1e9);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.keyID == b.keyID);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel = a.NoiseLevel;

	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::addMetadata(const Ciphertext& a, const Plaintext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.c0.getLevel());
		// assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::multMetadata(const Ciphertext& a, const Ciphertext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.getLevel());
		// assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.keyID == b.keyID);
	this->keyID = a.keyID;
	// assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel + b.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

void Ciphertext::multMetadata(const Ciphertext& a, const Plaintext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		// assert(a.getLevel() == b.c0.getLevel());
		//  assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	this->keyID = a.keyID;
	// assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel = a.NoiseLevel + b.NoiseLevel;

	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

/**
 * Negate one component exactly: multiply limb i by q_i - 1 (== -1 mod q_i). Unlike multScalar(-1.0)
 * this leaves the noise degree and scaling factor alone, which is what OpenFHE's per-element Negate
 * does. Same scalar construction as CudaEngine::negateExact.
 */
static void negatePolyExact(RNSPoly& poly, ContextData& cc) {
	std::vector<uint64_t> negOne(poly.getLevel() + 1);
	for (size_t i = 0; i < negOne.size(); ++i)
		negOne[i] = cc.prime[i].p - 1;
	poly.multScalar(negOne);
}

int Ciphertext::normalyzeIndex(int index) const {

	return FIDESlib::CKKS::normalyzeIndex(index, slots, cc.N);
}

void Ciphertext::add(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp() || (c2.has_value() && c2->isModUp()) || (b.c2.has_value() && b.c2->isModUp())) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (!adjustForAddOrSub(b)) {
			Ciphertext b_(cc_);
			b_.copy(b);
			if (b_.adjustForAddOrSub(*this))
				add(b_);
			else
				throwAdjustFailure("add", *this, b.getLevel(), b.NoiseLevel, "ciphertext");
			return;
		}
	}

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(this->getLevel() == b.getLevel());
	} else if (getLevel() > b.getLevel()) {

		if (c0.isModUp() || c1.getLevel() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}
		assert(this->getLevel() <= b.getLevel());
		dropToLevel(b.getLevel());
	}
	op_count[OPS::ADD]++;

	c0.add(b.c0);
	c1.add(b.c1);
	// OpenFHE EvalAddCoreInPlace over operands of different degree: elementwise over the components
	// both carry, then the longer operand's tail is COPIED in unchanged.
	if (c2.has_value() && b.c2.has_value()) {
		c2->add(*b.c2);
	} else if (b.c2.has_value()) {
		acquireC2();
		c2->copy(*b.c2);
		// The (c0, c1) adds above take only b's low limbs when b sits at a higher level, so the
		// copied tail is trimmed the same way.
		c2->dropToLevel(c0.getLevel());
	}
	// A tail we already own and b does not is left alone: b contributes nothing to it.

	this->addMetadata(*this, b);
}

void Ciphertext::sub(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp() || (c2.has_value() && c2->isModUp()) || (b.c2.has_value() && b.c2->isModUp())) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (!adjustForAddOrSub(b)) {
			Ciphertext b_(cc_);
			b_.copy(b);
			if (b_.adjustForAddOrSub(*this))
				sub(b_);
			else
				throwAdjustFailure("sub", *this, b.getLevel(), b.NoiseLevel, "ciphertext");
			return;
		}
	}

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(this->getLevel() == b.getLevel());
	} else if (getLevel() > b.getLevel()) {
		c0.dropToLevel(b.getLevel());
		c1.dropToLevel(b.getLevel());
		if (c2.has_value())
			c2->dropToLevel(b.getLevel());
	}
	op_count[OPS::ADD]++;

	c0.sub(b.c0);
	c1.sub(b.c1);
	// OpenFHE EvalSubCoreInPlace over operands of different degree: elementwise over the components
	// both carry; a tail belonging to the MINUEND is kept unchanged, a tail belonging to the
	// SUBTRAHEND is NEGATED.
	if (c2.has_value() && b.c2.has_value()) {
		c2->sub(*b.c2);
	} else if (b.c2.has_value()) {
		acquireC2();
		c2->copy(*b.c2);
		c2->dropToLevel(c0.getLevel());
		negatePolyExact(*c2, cc);
	}

	this->addMetadata(*this, b);
}

void Ciphertext::addPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescale();
		}

		// A depth mismatch needs the adjust just as much as a level mismatch does: adding a
		// plaintext encoded at one scaling factor onto a depth-2 ciphertext contributes ~2^-50 of
		// its value, so gating on the level alone silently discards the plaintext.
		if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel != this->NoiseLevel) {
			Plaintext b_(cc_);
			requireAdjustedPt(b_.adjustPlaintextToCiphertext(b, *this), b_, b, "addPt");
			addPt(b_);
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	if (b.c0.getLevel() < this->getLevel())   // no adjust step ran (FIXEDMANUAL): the kernels would read past the plaintext
		throwAdjustFailure("addPt", *this, b.c0.getLevel(), b.NoiseLevel, "plaintext");
	op_count[OPS::ADDPT]++;

	c0.add(b.c0);

	this->addMetadata(*this, b);
}

void Ciphertext::subPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
			assert(NoiseLevel == 1);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescale();
		}

		// Same depth-mismatch dispatch as addPt above, and the adjusted plaintext must be
		// SUBTRACTED: recursing into addPt here computed ct + pt on every adjusted path.
		if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel != this->NoiseLevel) {
			Plaintext b_(cc_);
			requireAdjustedPt(b_.adjustPlaintextToCiphertext(b, *this), b_, b, "subPt");
			subPt(b_);
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	if (b.c0.getLevel() < this->getLevel())   // no adjust step ran (FIXEDMANUAL): the kernels would read past the plaintext
		throwAdjustFailure("subPt", *this, b.c0.getLevel(), b.NoiseLevel, "plaintext");
	op_count[OPS::ADDPT]++;

	c0.sub(b.c0);

	this->addMetadata(*this, b);
}

void Ciphertext::load(const RawCipherText& rawct) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kCtLoad);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	keyID = rawct.keyid;
	c0.load(rawct.sub_0, rawct.moduli);
	c1.load(rawct.sub_1, rawct.moduli);
	// A non-empty sub_2 is a degree-2 ciphertext; RNSPoly::load grows/drops to the data's own level.
	if (!rawct.sub_2.empty()) {
		acquireC2();
		c2->load(rawct.sub_2, rawct.moduli);
	} else {
		releaseC2();
	}

	NoiseLevel  = rawct.NoiseLevel;
	NoiseFactor = rawct.Noise;
	slots       = rawct.slots;
}

void Ciphertext::store(RawCipherText& rawct) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kCtLoad);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

	CKKS::SetCurrentContext(cc_);
	cudaDeviceSynchronize();
	rawct.numRes = c0.getLevel() + 1;
	rawct.sub_0.resize(rawct.numRes);
	rawct.sub_1.resize(rawct.numRes);
	if (c2.has_value()) {
		rawct.sub_2.resize(rawct.numRes);
	} else {
		rawct.sub_2.clear(); // degree 1: the raw ciphertext carries two components
	}
	c0.store(rawct.sub_0);
	c1.store(rawct.sub_1);
	if (c2.has_value())
		c2->store(rawct.sub_2);
	rawct.N = cc.N;
	c0.sync();
	c1.sync();
	if (c2.has_value())
		c2->sync();

	rawct.NoiseLevel = NoiseLevel;
	rawct.Noise      = NoiseFactor;
	rawct.keyid      = keyID;
	rawct.slots      = slots; // TODO store other interesting metadata
	cudaDeviceSynchronize();
}

void Ciphertext::modDown(bool free) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("modDown");
	c0.moddown(true, false, 0);
	c1.moddown(true, false, 1);
	if (free) {
		c0.freeSpecialLimbs();
		c1.freeSpecialLimbs();
	}
}

void Ciphertext::modUp() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("modUp");
	// c0.modup();
	c1.modup();
}

void Ciphertext::multPt(const Plaintext& b, bool rescale, bool ignore_scale) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kMultPt);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	constexpr bool PRINT = false;

	if (!ignore_scale) {
		if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

			if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || (c2.has_value() && c2->isModUp())) {
				assert(getLevel() == b.c0.getLevel());
				assert(NoiseLevel == b.NoiseLevel);
				assert(NoiseLevel == 1);
			}

			if constexpr (PRINT)
				std::cout << "multPt: Rescale input ciphertext" << std::endl;
			if (NoiseLevel == 2)
				this->rescale();
		}

		if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

			if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || (c2.has_value() && c2->isModUp())) {
				assert(getLevel() == b.c0.getLevel());
				assert(NoiseLevel == b.NoiseLevel);
				assert(NoiseLevel == 1);
			}

			if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
				Plaintext b_(cc_);
				if constexpr (PRINT)
					std::cout << "multPt: adjust input plaintext" << std::endl;

				// if (!this->adju)
				if (!b_.adjustPlaintextToCiphertext(b, *this)) {
					if constexpr (PRINT)
						std::cout << "multPt: FAILED!" << std::endl;
					throwAdjustFailure("multPt", *this, b.c0.getLevel(), b.NoiseLevel, "plaintext");
				} else {
					if (NoiseLevel == 2)
						this->rescale();
					if (b_.NoiseLevel == 2) {
						if constexpr (PRINT)
							std::cout << "multPt: Rescale input plaintext" << std::endl;
						b_.rescale();
					}
					multPt(b_, rescale);
				}
				return;
			}
		}

		assert(NoiseLevel < 2);
		assert(b.NoiseLevel < 2);
	}
	if (b.c0.getLevel() < this->getLevel())   // no adjust step ran (FIXEDMANUAL / ignore_scale): the kernels would read past the plaintext
		throwAdjustFailure("multPt", *this, b.c0.getLevel(), b.NoiseLevel, "plaintext");
	op_count[OPS::MULTPT]++;

	// OpenFHE's plaintext multiply loops over EVERY component of the ciphertext.
	c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);
	c1.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);
	if (c2.has_value())
		c2->multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);

	this->multMetadata(*this, b);
	if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
		NoiseLevel -= 1;
	}
}

void Ciphertext::rescale() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	// assert(this->NoiseLevel == 2);
	if (cc.rescaleTechnique != FIXEDMANUAL) {
		// this wouldn't do anything in OpenFHE
	}
	const int32_t levelBefore = getLevel();

	if (c2.has_value() && levelBefore == cc.L + 1) {
		// rescaleDouble alone knows how to rescale off the FLEXIBLEAUTOEXT extra level: it takes the
		// top limb from SPECIALlimb and drops the level by 2 (RNSPoly::rescaleDouble ->
		// LimbPartition::doubleRescaleMGPU). The single-polynomial RNSPoly::rescale() reads the top
		// limb from `limb` and drops by 1, so it cannot follow c0/c1 there. Refuse rather than let the
		// three components silently desynchronize.
		throw std::runtime_error("rescale: a degree-2 ciphertext at the FLEXIBLEAUTOEXT extra level is not supported; relinearize first");
	}

	op_count[OPS::RESCALE]++;
	NoiseFactor /= (levelBefore == cc.L + 1) ? cc.specialPrime[0].p : cc.param.ModReduceFactor.at(c0.getLevel());

	if constexpr (RESCALE_DOUBLE) {
		c0.rescaleDouble(c1);
	} else {
		c0.rescale();
		c1.rescale();
	}
	// OpenFHE's mod-reduce loops over EVERY component. rescaleDouble only FUSES the two components'
	// top-limb streams and buffers: LimbPartition::rescale, ::rescaleMGPU and ::doubleRescaleMGPU all
	// issue the same kernel pair per component — INTT_<false/true, ALGO_SHOUP, INTT_NONE> on the top
	// limb, then NTT_<false/true, ALGO_SHOUP, NTT_RESCALE> over the limbs below it, M=4 — so a third
	// single-polynomial rescale is the same arithmetic as a third fused one would be.
	if (c2.has_value())
		c2->rescale();

	// Manage metadata
	NoiseLevel -= 1;
	assert(NoiseFactor == (NoiseLevel == 1 ? cc.param.ScalingFactorReal[this->getLevel()] : cc.param.ScalingFactorRealBig[this->getLevel()]));
}

/** "in" needs to have Digit and Gather limbs pre-generated */
RNSPoly& MGPUkeySwitchCore(RNSPoly& in, const KeySwitchingKey& kskEval, const bool moddown) {
	{
		RNSPoly& aux = in.modup_ksk_moddown_mgpu(kskEval, moddown);

		return aux;
	}
}

void Ciphertext::foldKeySwitched(RNSPoly& d2, const KeySwitchingKey& kskEval, const bool moddown) {
	RNSPoly& aux = MGPUkeySwitchCore(d2, kskEval, moddown);
	c0.add(aux);
	c1.add(d2);
}

bool Ciphertext::adjustBeforeMult(const Ciphertext& b) {
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		return adjustForMult(b);
	}
	if (cc.rescaleTechnique == FIXEDMANUAL && getLevel() > b.getLevel()) {
		dropToLevel(b.getLevel(), true);
	}
	return true;
}

void Ciphertext::mult(const Ciphertext& b, bool rescale, const bool moddown) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kMult);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("mult");
	b.requireDegree1("mult (second operand)");
	if (!adjustBeforeMult(b)) {
		Ciphertext b_(cc_);
		b_.copy(b);
		if (b_.adjustForMult(*this))
			mult(b_, rescale, moddown);
		else
			throwAdjustFailure("mult", *this, b.getLevel(), b.NoiseLevel, "ciphertext");
		return;
	}
	assert(NoiseLevel == 1);
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::MULT]++;
	// assert(c0.getLevel() <= b.c0.getLevel());
	// assert(c1.getLevel() <= b.c1.getLevel());
	Out(KEYSWITCH, " start ");
	assert(this->NoiseLevel == 1);
	assert(b.NoiseLevel == 1);

	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

	if (0 && cc.GPUid.size() == 1) {
		if constexpr (0) {
			constexpr bool PRINT = true;
			bool SELECT			 = true;

			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().multElement(c1, b.c1);

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Input data ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].limb.size(); ++j) {
						std::cout << cc.getKeySwitchAux().GPU[0].meta[j].id;
						SWITCH(cc.getKeySwitchAux().GPU[0].limb[j], printThisLimb(2));
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			cc.getKeySwitchAux().modup();

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
						for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
							std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
							SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU out KSK specials: ";
					for (const auto& j : { &aux0, &cc.getKeySwitchAux() }) {
						for (const auto& k : j->GPU) {
							for (auto& i : k.SPECIALlimb) {
								SWITCH(i, printThisLimb(2));
							}
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}
			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU out KSK limbs: ";
					for (const auto& j : { &aux0, &cc.getKeySwitchAux() }) {
						for (const auto& k : j->GPU) {
							for (auto& i : k.limb) {
								SWITCH(i, printThisLimb(2));
							}
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			cc.getKeySwitchAux().moddown(true, false, 0);
			aux0.moddown(true, false, 1);
			c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());
			c0.mult1Add2(b.c0, aux0);
			// c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
			if (rescale) {
				this->rescale();
			}
			/*
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().multElement(c1, b.c1);
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().modup();
			cudaDeviceSynchronize();
			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, c0.getLevel(), nullptr);
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().moddown(true, false);
			cudaDeviceSynchronize();
			c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());  // Read 4 first for better cache locality.
			cudaDeviceSynchronize();
			aux0.moddown(true, false);
			cudaDeviceSynchronize();
			c0.mult1Add2(b.c0, aux0);
			cudaDeviceSynchronize();

			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
			 */
		} else if (false) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().multModupDotKSK(c1, b.c1, c0, b.c0, kskEval);
			{ // TODO MAD Figure 4: add before fused ModDown+Rescale
			}
			if (moddown)
				c1.moddown(true, false);
			if (moddown)
				c0.moddown(true, false);

			// Manage metadata
			this->multMetadata(*this, b);

			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				this->rescale();
		} else if (false) {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.multElement(c1, b.c1);

			RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, moddown);

			if (moddown) {
				c1.mult1AddMult23Add4(b.c0, c0, b.c1, in); // Read 4 first for better cache locality.
				c0.mult1Add2(b.c0, aux);
			} else {
				c1.multNoModdownEnd(c0, b.c0, b.c1, in, aux);
			}

			// Manage metadata
			this->multMetadata(*this, b);
			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
		} else {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());

			c0.binomialMult(c1, in, b.c0, b.c1, moddown, &b == this);

			foldKeySwitched(in, kskEval, moddown);

			// Manage metadata
			this->multMetadata(*this, b);
			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
		}
	} else {
		constexpr bool PRINT = false;

		if constexpr (PRINT)
			std::cout << "Init mult" << std::endl;
		RNSPoly& in = cc.getKeySwitchAux();
		in.setLevel(c1.getLevel());
		c0.binomialMult(c1, in, b.c0, b.c1, moddown, &b == this);

		foldKeySwitched(in, kskEval, moddown);

		// Manage metadata
		this->multMetadata(*this, b);
		if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
			this->rescale();
		}
		if constexpr (PRINT)
			std::cout << "End mult" << std::endl;
		if constexpr (PRINT)
			CudaCheckErrorMod;
	}

	Out(KEYSWITCH, " finish ");
}

void Ciphertext::multNoRelin(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("multNoRelin");
	b.requireDegree1("multNoRelin (second operand)");
	if (!adjustBeforeMult(b)) {
		Ciphertext b_(cc_);
		b_.copy(b);
		if (b_.adjustForMult(*this))
			multNoRelin(b_);
		else
			assert(false);
		return;
	}
	assert(NoiseLevel == 1);
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::MULT]++;

	// The third component is OWNED, acquired from the auxiliary-polynomial pool: it has to survive
	// until relinearize(), and every key-switch in between overwrites the shared scratch. It cannot BE
	// that scratch by construction — getKeySwitchAux()/getKeySwitchAux2() hand out ContextData's own
	// unique_ptrs, which are never returned to the pool getAuxilarPoly() draws from. (Comparing
	// against them here would mean calling those accessors, and they allocate as a side effect, so the
	// check would only run in Debug.)
	acquireC2();
	c2->grow(c1.getLevel());
	c2->SetModUp(false);
	assert(&*c2 != &c0 && &*c2 != &c1 && &*c2 != &b.c0 && &*c2 != &b.c1);

	// Same kernel mult() uses for the tensor product, with moddown=true so the result stays in the
	// base RNS basis (never mod-up) — the state relinearize() expects. binomialMult_ reads all four
	// inputs into registers before writing c0/c1/c2, so writing into the destination in place is
	// safe, and a self-multiply (&b == this) routes to binomialSquare_.
	c0.binomialMult(c1, *c2, b.c0, b.c1, /*moddown=*/true, &b == this);

	this->multMetadata(*this, b);
}

void Ciphertext::relinearize() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (!c2.has_value()) {
		// OpenFHE's RelinearizeInPlace on a degree-1 ciphertext is a no-op (cv.resize(2)). The api
		// level requirement that a relinearization key exist is enforced by the engine, which has to
		// check it BEFORE this degree branch to match OpenFHE's accessor-throws-first order.
		return;
	}

	// GetEvalKey only asserts (a no-op in Release), so low-level callers get an indeterminate crash
	// or garbage key instead of a diagnosable error. Direct low-level consumers of this class (not
	// routed through the api layer's own checks) are exactly who this is for.
	if (!cc.HasEvalKey(keyID)) {
		throw std::runtime_error("relinearize: no eval-mult key registered for key tag " + keyID);
	}
	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

	// Copy c2 into the shared key-switch scratch instead of switching it in place: the scratch is the
	// polynomial that carries the DECOMP/DIGIT/GATHER buffers the hybrid key-switch needs
	// (Context::getKeySwitchAux generates them), and giving every degree-2 ciphertext its own set
	// would cost an extra dnum*(L+K) limbs each. The no-copy alternative is to call
	// generateDecompAndDigit()/generateGatherLimbs() on c2 and key-switch it in place; that trades
	// this one copy for that per-ciphertext allocation.
	RNSPoly& in = cc.getKeySwitchAux();
	in.copy(*c2);

	foldKeySwitched(in, kskEval, /*moddown=*/true);

	releaseC2();
	// Metadata (level, NoiseLevel, NoiseFactor, slots) is deliberately untouched, as in OpenFHE.
}

void Ciphertext::square(bool rescale) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kMult);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("square");
	Out(KEYSWITCH, " start ");

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp()) {
			assert(NoiseLevel == 1);
		}

		if (NoiseLevel == 2)
			this->rescale();
	}
	assert(this->NoiseLevel == 1);
	op_count[OPS::SQUARE]++;

	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

	if (cc.GPUid.size() == 1) {
		if constexpr (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().squareElement(c1);
			cc.getKeySwitchAux().modup();
			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);
			cc.getKeySwitchAux().moddown(true, false);
			aux0.moddown(true, false);
			// c1.mult1AddMult23Add4(c0, c0, c1, cc.getKeySwitchAux());
			c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
			this->multMetadata(*this, *this);
			if (rescale) {
				this->rescale();
			}
		} else if constexpr (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().squareModupDotKSK(c0, c1, kskEval);

			c1.moddown(true, false);

			c0.moddown(true, false);
			this->multMetadata(*this, *this);
			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				this->rescale();

		} else {
			this->mult(*this, rescale);
		}
	} else {
		if (0) {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.squareElement(c1);

			RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, true);

			c1.binomialSquareFold(c0, aux, in);
			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
			this->multMetadata(*this, *this);
		} else {
			this->mult(*this, rescale);
		}
	}
	// Manage metadata
	Out(KEYSWITCH, " finish ");
}

void Ciphertext::multScalarNoPrecheck(const double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::MULTSCALAR]++;

	auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
	// OpenFHE's scalar multiply loops over EVERY component of the ciphertext.
	c0.multScalar(elem);
	c1.multScalar(elem);
	if (c2.has_value())
		c2->multScalar(elem);

	// Manage metadata
	NoiseLevel += 1;
	NoiseFactor *= getLevel() == cc.L + 1 ? cc.specialPrime[0].p : cc.param.ScalingFactorReal.at(c0.getLevel());
	if (rescale && cc.rescaleTechnique == FIXEDAUTO) {
		this->rescale();
	}
}

void Ciphertext::multScalar(const double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp()) {
			assert(NoiseLevel == 1);
		}

		if (NoiseLevel == 2)
			this->rescale();
	}
	assert(this->NoiseLevel == 1);
	multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == FIXEDMANUAL);
}

void Ciphertext::addScalar(const double c) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::ADDSCALAR]++;

	auto elem = cc.ElemForEvalAddOrSub(c0.getLevel(), std::abs(c), this->NoiseLevel);

	if (c < 0.0) {
		for (auto i = 0u; i < elem.size(); ++i) {
			elem[i] = cc.prime[i].p - elem[i];
		}
	}
	// if (c >= 0.0) {
	if (c != 0.0)
		c0.addScalar(elem);
	//} else {
	//    c0.subScalar(elem);
	//}
}

void Ciphertext::automorph(const int index, const int br) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("automorph");
	auto& aux0 = cc.getModdownAux(0);
	auto& aux1 = cc.getModdownAux(1);
	aux0.copy(c0);
	aux1.copy(c1);
	c0.automorph(index, br, &aux0);
	c1.automorph(index, br, &aux1);
}

void Ciphertext::extend(bool init) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("extend");
	c0.generateSpecialLimbs(init && !c0.isModUp(), false);
	c1.generateSpecialLimbs(init && !c1.isModUp(), false);

	if (init) {
		if (!c0.isModUp())
			c0.scaleByP();
		if (!c1.isModUp())
			c1.scaleByP();
	}
}

void Ciphertext::rotate(const int index__, const bool moddown) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("rotate");
	op_count[OPS::ROTATE]++;
	int index = normalyzeIndex(index__);

	assert(index != 0);
	//constexpr bool PRINT = false;
	assert(!c1.isModUp());
	{
		/*
		RNSPoly& in = cc.getKeySwitchAux();
		in.setLevel(c1.getLevel());
		in.copy(c1);

		RNSPoly& aux = MGPUkeySwitchCore(in, kskRot, moddown);
		if (!moddown) {
			//in.moddown(true, false);
			c1.SetModUp(false);
			c1.generateSpecialLimbs(false);
		}
		c1.automorph(index, 1, &in);
		//c1.moddown(true, false);

		if (!moddown) {
			//aux.moddown(true, false);
			c0.SetModUp(false);
			c0.generateSpecialLimbs(true);
		}
		aux.add(aux, c0);
		//c0.add(aux);
		c0.automorph(index, 1, &aux);
		//c0.moddown();
		*/

		auto& in0 = cc.getKeySwitchAux2();
		auto& in1 = cc.getKeySwitchAux();
		in1.copy(c1);
		in1.modup();
		// c1.modupInto(cc.getKeySwitchAux());
		in0.copy(c0);

		std::vector<int> index_;
		std::vector<RNSPoly*> c0_out;
		std::vector<RNSPoly*> c1_out;
		std::vector<RNSPoly*> ksk_a;
		std::vector<RNSPoly*> ksk_b;
		{
			{
				c0_out.push_back(&c0);
				c1_out.push_back(&c1);
				int32_t actual_index;
				auto& ksk = cc.GetRotationKey(index, keyID, slots, actual_index);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index_.push_back(actual_index);
			}
		}

		in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, in0, in1);

		if (moddown) {
			modDown(false);
		}
	}
}

void Ciphertext::rotate(const Ciphertext& c, const int index) {
	this->copy(c);
	this->rotate(index, true);
}

void Ciphertext::conjugate(const Ciphertext& c) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kConj);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("conjugate");
	c.requireDegree1("conjugate (source)");
	op_count[OPS::CONJUGATE]++;

	int index = 2 * cc.N - 1;
	// auto& in0 = cc.getKeySwitchAux2();
	auto& in1 = cc.getKeySwitchAux();
	in1.copy(c.c1);
	in1.modup();
	// c1.modupInto(cc.getKeySwitchAux());
	// in0.copy(c0);

	std::vector<int> index_;
	std::vector<RNSPoly*> c0_out;
	std::vector<RNSPoly*> c1_out;
	std::vector<RNSPoly*> ksk_a;
	std::vector<RNSPoly*> ksk_b;
	{
		{
			c0_out.push_back(&c0);
			c1_out.push_back(&c1);
			int actual_index;
			auto& ksk = cc.GetRotationKey(index, c.keyID, slots, actual_index);
			ksk_a.push_back(&ksk.a);
			ksk_b.push_back(&ksk.b);
			index_.push_back(actual_index);
		}
	}

	dropToLevel(c.getLevel(), true);
	c0.grow(c.c0.getLevel());
	c1.grow(c.c1.getLevel());

	in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, c.c0, in1);

	if (1) {
		modDown(false);
	}

	this->copyMetadata(c);
}

void Ciphertext::rotate_hoisted(const std::vector<int>& indexes_, std::vector<Ciphertext*> results, const bool ext) {
	std::vector<int> indexes;
	for (auto i : indexes_) {
		indexes.push_back(normalyzeIndex(i));
	}

	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("rotate_hoisted");
	op_count[OPS::HOISTEDROTATE]++;
	op_count[OPS::HOISTEDROTATEOUTS] += static_cast<long long>(indexes.size());

	constexpr bool PRINT = false;
	assert(indexes.size() == results.size());

	bool grow_full = false;
	for (auto& i : results) {
		i->growToLevel(grow_full ? cc.L : this->c0.getLevel());
		i->dropToLevel(getLevel(), true);
		if (ext)
			i->extend(false);

		// i->c0.setLevel(c0.getLevel());
		// i->c1.setLevel(c1.getLevel());
	}

	if (!hoistRotateFused) {
		if (cc.GPUid.size() == 1) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			c1.modupInto(cc.getKeySwitchAux());

			if constexpr (PRINT) {
				{
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
						for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
							std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
							SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] == 0) {
					results[i]->copy(*this);
					if (ext) {
						results[i]->extend();
					}
				} else {
					int actual_index;
					RNSPoly& aux0 = results[i]->c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), cc.GetRotationKey(indexes[i], keyID, slots, actual_index), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext)
						results[i]->c1.moddown(true, false, 0);
					results[i]->c1.automorph(actual_index, 1);

					if (!ext)
						aux0.moddown(true, false, 1);

					// results[i]->c0.generateSpecialLimbs(true);
					results[i]->c0.add(c0, aux0);
					results[i]->c0.automorph(actual_index, 1);
					// if (!ext)
					//     results[i]->c0.moddown(true, false);

					results[i]->copyMetadata(*this); // also copies `slots` (was keyID/NoiseLevel/NoiseFactor only -> slots stayed 0)
				}
			}
		} else {

			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.copy(c1);
			in.modup();

			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] == 0) {
					results[i]->copy(*this);
					if (ext) {
						results[i]->extend();
					}
				} else {
					int actual_index;
					RNSPoly& aux0 = in.dotKSKInPlace(cc.GetRotationKey(indexes[i], keyID, slots, actual_index), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext) {
						in.moddown(true, false, 0);
					}
					// std::cout << "in ismodup: " << in.isModUp() << std::endl;
					results[i]->c1.automorph(actual_index, 1, &in);
					// std::cout << "results[i] c1 ismodup: " << results[i]->c1.isModUp() << std::endl;
					if (!ext) {
						aux0.moddown(true, false, 1);
					}
					// std::cout << "aux0 ismodup: " << aux0.isModUp() << std::endl;
					// std::cout << "c0 ismodup: " << c0.isModUp() << std::endl;
					results[i]->c0.add(c0, aux0);
					// std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;
					results[i]->c0.automorph(actual_index, 1, nullptr);
					// std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;

					results[i]->copyMetadata(*this);
				}
			}
		}
	} else {
		RNSPoly& in = cc.getKeySwitchAux();
		if (cc.GPUid.size() == 1) {
			in.setLevel(c1.getLevel());
			c1.modupInto(in);
		} else {
			in.setLevel(c1.getLevel());
			in.copy(c1);
			in.modup();
		}

		if constexpr (PRINT) {
			{
				cudaDeviceSynchronize();
				std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
				for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
					for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
						std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
						SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}

		std::vector<int> index;
		std::vector<RNSPoly*> c0_out;
		std::vector<RNSPoly*> c1_out;
		std::vector<RNSPoly*> ksk_a;
		std::vector<RNSPoly*> ksk_b;
		for (size_t i = 0; i < indexes.size(); ++i) {
			if (indexes[i] == 0) {
				results[i]->copy(*this);
				if (ext) {
					results[i]->extend();
				}
			} else {
				c0_out.push_back(&results[i]->c0);
				c1_out.push_back(&results[i]->c1);
				int actual_index;
				auto& ksk = cc.GetRotationKey(indexes[i], keyID, slots, actual_index);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index.push_back(actual_index);

				results[i]->copyMetadata(*this);
			}
		}

		in.hoistedRotationFused(index, c0_out, c1_out, ksk_a, ksk_b, c0, c1);

		if (!ext) {
			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] != 0) {
					results[i]->modDown();
				}
			}
		}
	}
}

/** @brief ONE index of rotate_hoisted, against a ModUp computed EARLIER and kept alive by the caller.
 *
 * rotate_hoisted (above) does the ModUp of c1 -- the digit decomposition plus its NTTs, i.e. the
 * expensive half of a key switch -- once and then applies it to every index it was handed. That only
 * helps a caller who knows all its indices up front. The API's hoisting contract is the other shape:
 * EvalFastRotationPrecompute once, then EvalFastRotation per index, lazily. This is the per-index
 * half of rotate_hoisted, with the ModUp lifted out into `src_c1_modup` (built by
 * CudaEngine::evalFastRotationPrecompute, which owns it for the lifetime of the handle).
 *
 * `*this` is the destination and must already be a distinct copy of `src` at src's level -- the same
 * precondition rotate_hoisted's `results` entries satisfy; aliasing `*this` with `src` is what
 * rotate() has to spend two scratch polynomials on, and is not supported here.
 *
 * Read side by side with rotate_hoisted: the prologue, the index == 0 case, the key lookup, the
 * fused kernel call and the modDown are line for line the same, with `indexes.size() == 1`.
 */
void Ciphertext::rotate_precomputed(const Ciphertext& src, RNSPoly& src_c1_modup, const int index__, const bool ext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("rotate_precomputed");
	src.requireDegree1("rotate_precomputed (source)");
	op_count[OPS::PRECOMPROTATE]++;

	// A handle decomposes ONE modulus chain. rotate_hoisted re-does the ModUp on every call and so
	// cannot go stale; this one can -- rescale/dropToLevel on the source moves c1's level and the
	// digits stop describing it. The fused kernel would read them anyway and return plausible
	// numbers, which CKKS never flags, so refuse instead of computing garbage.
	if (src_c1_modup.getLevel() != src.c1.getLevel())
		throw std::invalid_argument("rotate_precomputed: the precompute was built at level " + std::to_string(src_c1_modup.getLevel()) +
									" but the source ciphertext is at level " + std::to_string(src.c1.getLevel()) + " (stale handle)");

	const int index = src.normalyzeIndex(index__);

	// rotate_hoisted's per-result prologue (it runs for index 0 too).
	this->growToLevel(src.c0.getLevel());
	this->dropToLevel(src.getLevel(), true);
	if (ext)
		this->extend(false);

	// Index 0 is the identity, exactly as in rotate_hoisted's loop.
	if (index == 0) {
		this->copy(src);
		if (ext)
			this->extend();
		return;
	}

	int actual_index;
	auto& ksk = cc.GetRotationKey(index, src.keyID, src.slots, actual_index);

	if (!hoistRotateFused) {
		if (cc.GPUid.size() == 1) {
			// dotKSKInPlaceFrom reads the digits (dotKSKfused takes digitSrc by const&) and writes
			// *this*, so the handle survives the call and can serve the next index.
			RNSPoly& aux0 = this->c1.dotKSKInPlaceFrom(src_c1_modup, ksk, &src.c1);
			if (!ext)
				this->c1.moddown(true, false, 0);
			this->c1.automorph(actual_index, 1);

			if (!ext)
				aux0.moddown(true, false, 1);

			this->c0.add(src.c0, aux0);
			this->c0.automorph(actual_index, 1);

			this->keyID       = src.keyID;
			this->NoiseLevel  = src.NoiseLevel;
			this->NoiseFactor = src.NoiseFactor;
		} else {
			// The multi-GPU non-fused path consumes its ModUp poly in place (dotKSKInPlace then
			// moddown), so it CANNOT be handed a shared handle -- it would destroy it for every
			// later index. It recomputes into the context scratch instead, which is exactly what
			// rotate_hoisted does: correct, and no slower than the code this replaces. The handle
			// is left untouched. Nothing in this deployment reaches here (hoistRotateFused is true
			// and the GPU count is one), and a correctness regression there would be silent.
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(src.c1.getLevel());
			in.copy(src.c1);
			in.modup();

			// dotKSKInPlace declares its limb source as RNSPoly* but forwards it to dotKSKfused's
			// `const RNSPoly* source` (RNSPoly.cpp:684) -- it is read-only in fact, not in type.
			RNSPoly& aux0 = in.dotKSKInPlace(ksk, const_cast<RNSPoly*>(&src.c1));
			if (!ext)
				in.moddown(true, false, 0);
			this->c1.automorph(actual_index, 1, &in);
			if (!ext)
				aux0.moddown(true, false, 1);
			this->c0.add(src.c0, aux0);
			this->c0.automorph(actual_index, 1, nullptr);

			this->copyMetadata(src);
		}
	} else {
		// hoistedRotationFused takes c0/c1 by NON-const reference, so these have to be named
		// vectors rather than braced temporaries.
		std::vector<int> indexes{ actual_index };
		std::vector<RNSPoly*> c0_out{ &this->c0 };
		std::vector<RNSPoly*> c1_out{ &this->c1 };
		std::vector<RNSPoly*> ksk_a{ &ksk.a };
		std::vector<RNSPoly*> ksk_b{ &ksk.b };

		this->copyMetadata(src);

		src_c1_modup.hoistedRotationFused(indexes, c0_out, c1_out, ksk_a, ksk_b, src.c0, src.c1);

		if (!ext)
			this->modDown();
	}
}

/** @brief ONE index over MANY sources -- the transpose of rotate_hoisted. See Ciphertext.cuh.
 *
 * Shape, next to the two neighbours it is built from:
 *   rotate()          1 source, 1 index  -> hoistedRotationFused with n = 1.
 *   rotate_hoisted()  1 source, k indexes -> hoistedRotationFused with n = k; ONE ModUp, k keys.
 *   rotate_many()     B sources, 1 index  -> fusedHoistedRotateBatch with stride = B, n = 1;
 *                                            B ModUps, ONE key.
 * The batched kernel (ElemenwiseBatchKernels.cu hoistedRotateDotKSKBatched___) spans the sources
 * on threadIdx.z and the indexes on its j loop: the source digits are read per z, the key digits
 * are staged in shared memory once per j and reused by every z. With n = 1 that is exactly "load
 * the key once, B inner products, B outputs".
 */
void Ciphertext::rotate_many(const std::vector<Ciphertext*>& srcs, const std::vector<Ciphertext*>& results, const int index__) {
	assert(srcs.size() == results.size());
	if (srcs.empty())
		return;

	Ciphertext& src0 = *srcs[0];
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(src0.cc_);
	ContextData& cc = src0.cc;

	const auto B = static_cast<uint32_t>(srcs.size());
	// Counted as B rotations, not as one op: the arm must be comparable to the serial path's
	// rotation count line for line (a batched launch that showed up as one rotation would make the
	// A/B rotation totals differ and the comparison meaningless).
	op_count[OPS::ROTATE] += B;

	const int index = src0.normalyzeIndex(index__);
	assert(index != 0);

	// The whole batch shares one key lookup. GetRotationKey resolves the index against this
	// ciphertext's key tag and slot count; the caller has already established that every source
	// agrees on both, so one lookup is the key for all of them.
	int actual_index;
	auto& ksk = cc.GetRotationKey(index, src0.keyID, src0.slots, actual_index);

	const bool c0_modup = src0.c0.isModUp();

	std::vector<std::unique_ptr<RNSPoly>> modups;
	modups.reserve(B);
	std::vector<RNSPoly*> in, out;
	in.reserve(2 * B);
	out.reserve(2 * B);

	for (uint32_t i = 0; i < B; ++i) {
		Ciphertext& s = *srcs[i];
		Ciphertext& d = *results[i];
		assert(&s != &d);
		s.requireDegree1("rotate_many (source)");
		d.requireDegree1("rotate_many (destination)");
		assert(s.c1.getLevel() == src0.c1.getLevel());
		assert(s.c0.isModUp() == c0_modup);

		// Each source needs its OWN ModUp: the digit decomposition describes the ciphertext, not
		// the key, so this is the one part of the key switch the batch cannot share. It is built
		// exactly as CudaEngine::evalFastRotationPrecompute builds its handle -- a private poly,
		// never the context-wide getKeySwitchAux scratch, because all B must be live at once when
		// the batched launch reads them. modupInto copies the base limbs in as well, so the poly
		// carries both halves the kernel wants from an input c1 (the undecomposed limb for the
		// digit that already sits in the right basis, and the NTTs of the others).
		auto up = std::make_unique<RNSPoly>(cc, cc.L, false);
		up->generateDecompAndDigit(false);
		up->generateSpecialLimbs(false, false);
		up->setLevel(s.c1.getLevel());
		s.c1.modupInto(*up);

		// rotate_hoisted's per-result prologue, once per destination. fusedHoistedRotateBatch reads
		// the limb layout off out[0] and does NOT generate the outputs' special limbs itself (its
		// single-source sibling hoistedRotationFused does), so that happens here.
		d.growToLevel(s.c0.getLevel());
		d.dropToLevel(s.getLevel(), true);
		d.c0.generateSpecialLimbs(false, false);
		d.c1.generateSpecialLimbs(false, false);
		d.copyMetadata(s);

		in.push_back(&s.c0);
		in.push_back(up.get());
		out.push_back(&d.c0);
		out.push_back(&d.c1);
		modups.push_back(std::move(up));
	}

	// n = 1: one index, one key pair, one shared-memory key stage per block.
	std::vector<int> indexes{ actual_index };
	std::vector<RNSPoly*> ksk_a{ &ksk.a };
	std::vector<RNSPoly*> ksk_b{ &ksk.b };

	RNSPoly::fusedHoistedRotateBatch(out, in, ksk_a, ksk_b, indexes, static_cast<int>(B), 1.0, c0_modup);

	for (uint32_t i = 0; i < B; ++i)
		results[i]->modDown();
}

void Ciphertext::mult(const Ciphertext& b, const Ciphertext& c, bool rescale) {
	b.requireDegree1("mult (first operand)");
	c.requireDegree1("mult (second operand)");
	if (this == &b && this == &c) {
		this->square(rescale);
	} else if (this == &b) {
		this->mult(c, rescale);
	} else if (this == &c) {
		this->mult(b, rescale);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->mult(c, rescale);
		} else {
			this->copy(c);
			this->mult(b, rescale);
		}
	}
}

void Ciphertext::square(const Ciphertext& src, bool rescale) {
	if (this == &src) {
		this->square(rescale);
	} else {
		this->copy(src);
		this->square(rescale);
	}
}

void Ciphertext::dropToLevel(const int level, bool skip_adjust) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	if (c0.getLevel() > level) {
		assert(c1.getLevel() > level);
		if (!skip_adjust && (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT)) {
			assert(NoiseLevel == 1 || NoiseLevel == 2);
			bool ok = adjustScaleAndLevel(this->NoiseLevel, level, this->NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : cc.param.ScalingFactorRealBig[level]);
			assert(ok);
			(void)ok;
		} else {
			c0.dropToLevel(level);
			c1.dropToLevel(level);
			if (c2.has_value())
				c2->dropToLevel(level);
		}
	}
}

int32_t Ciphertext::getLevel() const {
	assert(c0.getLevel() == c1.getLevel());
	assert(!c2.has_value() || c2->getLevel() == c0.getLevel());
	return c0.getLevel();
}

void Ciphertext::multScalar(const Ciphertext& b, const double c, bool rescale) {
	this->copy(b);
	this->multScalar(c, rescale);
}

void Ciphertext::evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("evalLinearWSumMutable");
	for (uint32_t i = 0; i < n; ++i)
		ctxs[i]->requireDegree1("evalLinearWSumMutable (input)");
	op_count[OPS::WSUM]++;
	op_count[OPS::WSUMINPUTS] += static_cast<long long>(n);

	if constexpr (1) {
		if (static_cast<int32_t>(this->getLevel()) == -1) {
			this->c0.grow(ctxs[0]->getLevel());
			this->c1.grow(ctxs[0]->getLevel());
			this->NoiseLevel = 1;
		}

		for (size_t i = 0; i < n; ++i) {
			if (cc.rescaleTechnique == FIXEDMANUAL) {
				assert(ctxs[i]->NoiseLevel == 1);
				assert(getLevel() <= ctxs[i]->getLevel());
			} else {
				assert(ctxs[i]->NoiseLevel == 1);
			}
			assert(ctxs[0]->keyID == ctxs[i]->keyID);
		}

		std::vector<uint64_t> elem(MAXP * n);

		// #pragma omp parallel for
		// double scalingFactor;
		for (size_t i = 0; i < n; ++i) {
			auto aux = cc.ElemForEvalMult(c0.getLevel(), weights[i], ctxs[i]->getLevel());
			for (size_t j          = 0; j < aux.size(); ++j)
				elem[i * MAXP + j] = aux[j];
		}

		std::vector<const RNSPoly*> c0s(n), c1s(n);

		for (size_t i = 0; i < n; ++i) {
			c0s[i] = &ctxs[i]->c0;
			c1s[i] = &ctxs[i]->c1;
		}
		c0.evalLinearWSum(n, c0s, elem);
		c1.evalLinearWSum(n, c1s, elem);

		// this->copyMetadata(*ctxs[0]);
		this->slots = ctxs[0]->slots;
		this->keyID = ctxs[0]->keyID;
		for (uint32_t i = 1; i < n; ++i) {
			assert(this->keyID == ctxs[i]->keyID);
			slots = std::max(slots, ctxs[i]->slots);
		}
		this->NoiseLevel  = 2;
		this->NoiseFactor = cc.param.ScalingFactorReal.at(getLevel()) * cc.param.ScalingFactorReal.at(getLevel());
	} else {
		this->multScalar(*ctxs[0], weights[0], false);
		for (int i = 1; i < n; ++i) {
			assert(getLevel() <= ctxs[i]->getLevel());
		}
		for (int i = 1; i < n; ++i) {
			this->addMultScalar(*ctxs[i], weights[i]);
		}
	}
}

void Ciphertext::addMultScalar(const Ciphertext& b, double d) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::MULTSCALAR]++;
	op_count[OPS::COPY]++;
	op_count[OPS::ADD]++;

	assert(NoiseLevel == 2);
	assert(b.NoiseLevel == 1);
	assert(b.getLevel() >= getLevel());
	assert(keyID == b.keyID);
	auto elem = cc.ElemForEvalMult(c0.getLevel(), d);

	RNSPoly aux0(cc);
	RNSPoly aux1(cc);
	aux0.copy(b.c0);
	aux0.multScalar(elem);
	c0.add(aux0);
	aux1.copy(b.c1);
	aux1.multScalar(elem);
	c1.add(aux1);
}

void Ciphertext::addScalar(const Ciphertext& b, double c) {
	this->copy(b);
	this->addScalar(c);
}

void Ciphertext::add(const Ciphertext& b, const Ciphertext& c) {
	assert(NoiseLevel <= 2);
	if (this == &b && this == &c) {
		this->add(c);
	} else if (this == &b) {
		this->add(c);
	} else if (this == &c) {
		this->add(b);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->add(c);
		} else {
			this->copy(c);
			this->add(b);
		}
	}
}

void Ciphertext::growToLevel(int level) {

	c0.grow(level);
	c1.grow(level);
	if (c2.has_value())
		c2->grow(level);
	if (c0.isModUp())
		c0.generateSpecialLimbs(false, false);
	if (c1.isModUp())
		c1.generateSpecialLimbs(false, false);
	if (c2.has_value() && c2->isModUp())
		c2->generateSpecialLimbs(false, false);
}

void Ciphertext::copy(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (this == &ciphertext) {
		return;
	}
	assert(this != &ciphertext);
	op_count[OPS::COPY]++;
	c0.copy(ciphertext.c0);
	c1.copy(ciphertext.c1);
	// Match the source's degree: acquire c2 when the source has one, release ours when it does not.
	if (ciphertext.c2.has_value()) {
		acquireC2();
		c2->copy(*ciphertext.c2);
	} else {
		releaseC2();
	}
	this->copyMetadata(ciphertext);
}

void Ciphertext::multPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kMultPt);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	// The straightforward implementation is copy(c) followed by the in-place multPt below. That
	// deep-copies the whole ciphertext on device (~29 MB at level 13, two RNSPolys) and then
	// overwrites every byte of it, because the element-wise plaintext multiply is a genuinely
	// out-of-place kernel: Mult_ reads l1/l2 and writes l. So when the in-place multPt would fall
	// straight through to that kernel we size the destination and let the kernel write into it,
	// skipping the copy entirely. The fast path is refused when the in-place path would do
	// anything else first:
	//   - fused_rescale: FIXEDMANUAL mult+rescale goes through LimbPartition::multPt, which is
	//     in-place only (it multiplies the top limb, INTTs it, and fuses the rest into an NTT).
	//   - modup_operand: mod-up operands carry special limbs that shapeLike() does not allocate.
	//   - needs_fixup: under the automatic techniques the in-place path may rescale the ciphertext
	//     or build an adjusted copy of the plaintext before multiplying.
	// Every refused case falls back to the original copy-then-multiply, so semantics are unchanged.
	const bool auto_scaling	 = cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT;
	const bool fused_rescale = rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL;
	const bool modup_operand = c.c0.isModUp() || c.c1.isModUp() || b.c0.isModUp() || (c.c2.has_value() && c.c2->isModUp());
	const bool needs_fixup	 = auto_scaling && (c.NoiseLevel == 2 || b.NoiseLevel == 2 || b.c0.getLevel() != c.getLevel());

	if (this != &c && !fused_rescale && !modup_operand && !needs_fixup) {
		assert(c.NoiseLevel < 2);
		assert(b.NoiseLevel < 2);
		op_count[OPS::MULTPT]++;

		// Match the source's shape (level, degree) without moving any device bytes; the multiply
		// below writes every limb we just allocated.
		c0.shapeLike(c.c0);
		c1.shapeLike(c.c1);
		if (c.c2.has_value()) {
			acquireC2();
			c2->shapeLike(*c.c2);
		} else {
			releaseC2();
		}

		// OpenFHE's plaintext multiply loops over EVERY component of the ciphertext.
		c0.multElement(c.c0, b.c0);
		c1.multElement(c.c1, b.c0);
		if (c2.has_value())
			c2->multElement(*c.c2, b.c0);

		// Same merge the in-place path performs, with `c` standing in for the copy of it we did
		// not make: slots/keyID/NoiseLevel/NoiseFactor come out identical.
		this->multMetadata(c, b);
		return;
	}

	this->copy(c);
	multPt(b, rescale);
}

void Ciphertext::addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("addMultPt");
	c.requireDegree1("addMultPt (input)");
	op_count[OPS::ADDMULTPT]++;

	assert(NoiseLevel == 2);
	assert(c.NoiseLevel == 1);
	assert(b.NoiseLevel == 1);

	c0.addMult(c.c0, b.c0);
	c1.addMult(c.c1, b.c0);

	if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		this->rescale();
	}
}

void Ciphertext::addPt(const Ciphertext& ciphertext, const Plaintext& plaintext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);
	this->addPt(plaintext);
}

// Ciphertext::reinterpretContext removed. It moved a ciphertext's data between the main and
// the "switchable" sparse GPU contexts for the dual-context mod-raise dance; in-context ENCAPS has
// a single context, so nothing reinterprets any more (its only users were the ENCAPS Bootstrap
// path and the ExtractContextCreateSwitch test, both gone).

void Ciphertext::keySwitch(const KeySwitchingKey& ksk) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKeySwitch);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("keySwitch");
	assert(ksk.keyID == this->keyID);

	RNSPoly& aux = cc.getKeySwitchAux();
	aux.copy(c1); // This is to save on memory allocations for keyswitching, not best performance but not expected for relinearize or rotation

	RNSPoly& aux0 = MGPUkeySwitchCore(aux, ksk, true); // c1.dotKSKInPlaceFrom(aux, ksk, &aux);

	c1.copy(aux);
	c0.add(aux0);
}

void Ciphertext::sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(ciphertext.getLevel() <= ciphertext1.getLevel());
	this->copy(ciphertext);
	this->sub(ciphertext1);
}

bool Ciphertext::adjustScaleAndLevel(const int scaleDegree, const int level, const double scaling_factor) {
	assert(scaleDegree == 2 ? std::abs(cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] - cc.param.ScalingFactorRealBig[level]) /
		(cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] + cc.param.ScalingFactorRealBig[level]) <
		1e-11 :
		true);
	assert(scaleDegree < 3 ? scaling_factor == (scaleDegree == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) : true);

	uint32_t c1lvl   = getLevel();
	uint32_t c2lvl   = level;
	uint32_t c1depth = this->NoiseLevel;
	uint32_t c2depth = scaleDegree;
	auto sizeQl1     = c1lvl + 1;
	// auto sizeQl2 = c2lvl + 1;

	if (c1lvl > c2lvl) {
		if (c1depth == 2) {
			if (c2depth == 2) {
				// FIDESlib bit-compat: mirror OpenFHE AdjustLevelsAndDepthInPlace exactly --
				// same floating-point evaluation order for the adjustment factor, the
				// mod-reduce factor of the CURRENT top level, and rescale before dropping.
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf  = cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
				double q1   = cc.param.ModReduceFactor[c1lvl];   // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
				multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
				NoiseFactor = cc.param.ScalingFactorRealBig[getLevel() - 1] * cc.param.ModReduceFactor[getLevel()];
				rescale();
				if (getLevel() > static_cast<int32_t>(c2lvl)) {
					this->dropToLevel(c2lvl, true);
				}
				NoiseFactor = scaling_factor;
			} else {
				if (c1lvl - 1 == c2lvl) {
					rescale();
				} else {
					double scf1 = NoiseFactor;
					double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
					double scf  = cc.param.ScalingFactorReal[c1lvl];        // cryptoParams->GetScalingFactorReal(c1lvl);
					double q1   = cc.param.ModReduceFactor[sizeQl1 - 1];    // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
					multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
					NoiseFactor = cc.param.ScalingFactorRealBig[this->getLevel() - 1] * cc.param.ModReduceFactor[c1lvl];
					rescale();
					if (getLevel() - 1 > static_cast<int32_t>(c2lvl)) {
						this->dropToLevel(c2lvl + 1, true);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
					}
					NoiseFactor = cc.param.ScalingFactorRealBig[this->getLevel()];
					// NoiseFactor *= scf2 / scf1 * q1 / scf;
					rescale();
					// assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);

					NoiseFactor = scaling_factor;
				}
			}
		} else {
			if (c2depth == 2) {
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf  = cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
				multScalarNoPrecheck(scf2 / scf1 / scf);
				this->dropToLevel(c2lvl, true);
				// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
				assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scf2;
			} else {
				double scf1 = NoiseFactor;
				double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
				double scf  = cc.param.ScalingFactorReal[c1lvl];        // cryptoParams->GetScalingFactorReal(c1lvl);
				multScalarNoPrecheck(scf2 / scf1 / scf);
				if (c1lvl - 1 > c2lvl) {
					this->dropToLevel(c2lvl + 1, true);
					// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
				}
				NoiseFactor *= scf2 / scf1 / scf;
				rescale();
				// assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scaling_factor;
			}
		}
		assert(scaleDegree < 3 ? this->NoiseFactor == (NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) : true);
		return true;
	} else if (c1lvl < c2lvl) {
		return false;
	} else {
		if (c1depth < c2depth) {
			multScalar(1.0, false);
		} else if (c2depth < c1depth) {
			return false;
		}
		return true;
	}
}

bool Ciphertext::adjustForAddOrSub(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	/*
	if (cc.rescaleTechnique == FIXEDMANUAL) {
		if (b.NoiseLevel > NoiseLevel || (b.getLevel() < getLevel()))
			return false;
		else
			return true;
	} else
	*/
	if (cc.rescaleTechnique == FIXEDMANUAL || cc.rescaleTechnique == FIXEDAUTO) {
		if (getLevel() - NoiseLevel > b.getLevel() - b.NoiseLevel) {
			if (b.NoiseLevel == 1 && NoiseLevel == 2) {
				rescale();
			} else if (b.NoiseLevel == 2 && NoiseLevel == 1) {
				this->multScalar(1.0);
			}
			return true;
		} else if (b.NoiseLevel == 1 && NoiseLevel == 2) {
			rescale();
			return true;
		} else if (NoiseLevel == 1 && b.NoiseLevel == 2) {
			return false;
		} else {
			return true;
		}
	} else if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		return adjustScaleAndLevel(b.NoiseLevel, b.getLevel(), b.NoiseFactor);
	}
	assert("This never happens" == nullptr);
	return false;
}

bool Ciphertext::adjustForMult(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	if (adjustForAddOrSub(ciphertext)) {
		if (NoiseLevel == 2)
			rescale();
		if (ciphertext.NoiseLevel == 2)
			return false;
		else
			return true;
	} else {
		if (NoiseLevel == 2)
			rescale();
		return false;
	}
}

bool Ciphertext::hasSameScalingFactor(const Plaintext& b) const {
	return NoiseFactor > b.NoiseFactor * (1 - 1e-9) && NoiseFactor < b.NoiseFactor * (1 + 1e-9);
}

void Ciphertext::clearOpRecord() {
	for (auto& c : op_count)
		c.store(0, std::memory_order_relaxed);
}

void Ciphertext::dotProductPt(Ciphertext* ciphertexts, Plaintext* plaintexts, const int n, const bool ext) {

	std::vector<Plaintext*> pts(n);
	for (int i = 0; i < n; ++i) {
		pts[i] = &plaintexts[i];
	}
	dotProductPt(ciphertexts, pts.data(), n, ext);
}

void Ciphertext::dotProductPt(Ciphertext* ciphertexts, Plaintext** plaintexts, const int n, const bool ext) {
	std::vector<Ciphertext*> pts(n);
	for (int i = 0; i < n; ++i) {
		pts[i] = &ciphertexts[i];
	}
	dotProductPt(pts.data(), plaintexts, n, ext);
}

void Ciphertext::dotProductPt(Ciphertext** ciphertexts, Plaintext** plaintexts, const int n, const bool ext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	requireDegree1("dotProductPt");
	for (int i = 0; i < n; ++i)
		ciphertexts[i]->requireDegree1("dotProductPt (input)");
	std::vector<const RNSPoly*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);

	for (int i = 0; i < n; ++i) {
		c0s[i] = &(ciphertexts[i]->c0);
		c1s[i] = &(ciphertexts[i]->c1);
		pts[i] = &(plaintexts[i]->c0);
		assert(getLevel() <= ciphertexts[i]->getLevel());
		assert(getLevel() <= plaintexts[i]->c0.getLevel());
		if (ext) {
			assert(ciphertexts[i]->c0.isModUp());
			assert(plaintexts[i]->c0.isModUp());
		}
		assert(ciphertexts[0]->keyID == ciphertexts[i]->keyID);
	}
	c0.dotProductPt(c1, c0s, c1s, pts, ext);

	// Manage metadata
	this->multMetadata(*ciphertexts[0], *plaintexts[0]);
	for (int i = 1; i < n; ++i) {
		this->slots = std::max(this->slots, ciphertexts[i]->slots);
		this->slots = std::max(this->slots, plaintexts[i]->slots);
	}
}

void Ciphertext::dotProduct(const std::vector<Ciphertext*>& a, const std::vector<Ciphertext*>& b, const bool ext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	requireDegree1("dotProduct");
	for (const Ciphertext* i : a)
		i->requireDegree1("dotProduct (lhs)");
	for (const Ciphertext* i : b)
		i->requireDegree1("dotProduct (rhs)");
	assert(a.size() == b.size());
	assert(a.size() > 0);
	assert(a[0]->c0.isModUp() == b[0]->c1.isModUp());
	bool in_ext = a[0]->c0.isModUp();

	this->growToLevel(a[0]->getLevel());
	this->dropToLevel(a[0]->getLevel(), true);

	std::vector<const RNSPoly*> c0s(a.size(), nullptr), c1s(a.size(), nullptr), d0s(a.size(), nullptr), d1s(a.size(), nullptr);

	for (size_t i = 0; i < a.size(); ++i) {
		assert(this->cc_ == a[i]->cc_);
		assert(this->cc_ == b[i]->cc_);
		// assert(a[0]->NoiseFactor == a[i]->NoiseFactor);
		// assert(a[0]->NoiseFactor == b[i]->NoiseFactor);
		assert(a[0]->keyID == a[i]->keyID);
		assert(a[0]->keyID == b[i]->keyID);
		// assert(a[i]->NoiseLevel == 1);
		// assert(b[i]->NoiseLevel == 1);
		assert(a[0]->getLevel() == a[i]->getLevel());
		assert(a[0]->getLevel() == b[i]->getLevel());

		assert(a[0]->c0.isModUp() == a[i]->c0.isModUp());
		assert(a[0]->c1.isModUp() == a[i]->c1.isModUp());
		assert(a[0]->c0.isModUp() == b[i]->c0.isModUp());
		assert(a[0]->c1.isModUp() == b[i]->c1.isModUp());
		c0s[i] = &(a[i]->c0);
		c1s[i] = &(a[i]->c1);
		d0s[i] = &(b[i]->c0);
		d1s[i] = &(b[i]->c1);
	}

	RNSPoly& c2 = c0.dotProduct(c1, cc.GetEvalKey(a[0]->keyID).a, cc.GetEvalKey(a[0]->keyID).b, c0s, c1s, d0s, d1s, in_ext, ext);

	if (c2.isModUp()) {
		c2.moddown(true, false, 0);
	}
	RNSPoly& aux = MGPUkeySwitchCore(c2, cc.GetEvalKey(a[0]->keyID), !(ext || in_ext));

	c0.add(aux);
	c1.add(c2);

	if (!ext && in_ext) {
		modDown();
	}

	this->multMetadata(*a[0], *b[0]);
	for (size_t i = 1; i < a.size(); ++i) {
		this->slots = std::max(this->slots, a[i]->slots);
		this->slots = std::max(this->slots, b[i]->slots);
	}
}

void Ciphertext::multMonomial(/*Ciphertext& ctxt,*/ int power) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kMonomial);
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	CKKS::SetCurrentContext(cc_);
	requireDegree1("multMonomial");

	// Per-issuing-thread cache (slot 0 is cc.precom.monomialCache, i.e. today's object): the entry
	// is ERASED and rebuilt whenever the level moves, so two threads bootstrapping at once would
	// have one destroying the polynomial the other is multiplying by. The monomials themselves are
	// a pure function of (power, level), so a per-thread copy computes the same thing.
	std::map<int, RNSPoly>& monomialCache = cc.getMonomialCache();
	if (!monomialCache.contains(power) || monomialCache.find(power)->second.getLevel() != this->getLevel()) {
		// TODO compute fully as a GPU function.
		RNSPoly monomial(cc.getAuxilarPoly());
		monomial.grow(c0.getLevel());
		monomial.dropToLevel(c0.getLevel());
		std::vector<uint64_t> coefs(cc.N, 0);

		if (power < cc.N) {
			coefs[power] = 1;

			for (auto& g : monomial.GPU) {
				cudaSetDevice(g.device);
				int limb_size = g.getLimbSize(monomial.getLevel());
				for (int i = 0; i < limb_size; ++i) {
					SWITCH(g.limb[i], load(coefs));
					g.s.wait(STREAM(g.limb[i]));
				}

				// for (auto& l : g.limb) {
				//     SWITCH(l, load(coefs));
				//     g.s.wait(STREAM(l));
				// }

				for (int i = 0; i < limb_size; i += cc.batch) {
					STREAM(g.limb[i]).wait(g.s);
				}
			}
		} else {
			for (auto& g : monomial.GPU) {
				cudaSetDevice(g.device);
				int limb_size = g.getLimbSize(monomial.getLevel());
				for (int i = 0; i < limb_size; ++i) {
					coefs[power % cc.N] = (cc.prime[PRIMEID(g.limb[i])].p - 1) /*% ctxt.cc.prime[PRIMEID(l)].p*/;
					SWITCH(g.limb[i], load(coefs));
					g.s.wait(STREAM(g.limb[i]));
				}
				// for (auto& l : g.limb) {
				//     coefs[power % cc.N] = (cc.prime[PRIMEID(l)].p - 1) /*% ctxt.cc.prime[PRIMEID(l)].p*/;
				//     SWITCH(l, load(coefs));
				//     g.s.wait(STREAM(l));
				// }
				for (int i = 0; i < limb_size; i += cc.batch) {
					STREAM(g.limb[i]).wait(g.s);
				}
			}
		}

		// cudaDeviceSynchronize();
		monomial.NTT(cc.batch, true);
		// cudaDeviceSynchronize();

		monomialCache.erase(power);
		monomialCache.emplace(power, std::move(monomial));
	}

	RNSPoly& monomial = monomialCache.find(power)->second;

	c0.multElement(monomial);
	c1.multElement(monomial);

	/* Based on this (OpenFHE):
std::vector<DCRTPoly>& cv = ciphertext->GetElements();
const auto elemParams     = cv[0].GetParams();
auto paramsNative         = elemParams->GetParams()[0];
uint32_t N                   = elemParams->GetRingDimension();
uint32_t M                   = 2 * N;

	NativePoly monomial(paramsNative, Format::COEFFICIENT, true);

	uint32_t powerReduced = power % M;
	uint32_t index        = power % N;
	monomial[index]    = powerReduced < N ? NativeInteger(1) : paramsNative->GetModulus() - NativeInteger(1);

	DCRTPoly monomialDCRT(elemParams, Format::COEFFICIENT, true);
	monomialDCRT = monomial;
	monomialDCRT.SetFormat(Format::EVALUATION);

	for (uint32_t i = 0; i < ciphertext->NumberCiphertextElements(); i++) {
		cv[i] *= monomialDCRT;
	}
	*/
}

void Ciphertext::printOpRecord() {
	std::cout << "|-------------- OP COUNT --------------|\n";
	for (size_t op = 0; op < op_count.size(); ++op) {
		// Only ops that actually happened, as the std::map this replaced did.
		const long long c = op_count[op].load(std::memory_order_relaxed);
		if (c != 0)
			std::cout << opstr[op] << c << "\n";
	}
	std::cout << "|--------------------------------------|" << std::endl;
}

} // namespace FIDESlib::CKKS

// ---- BLOCK PROVENANCE: TURNING A WRONG CIPHERTEXT INTO POINTERS ------------------------------
//
// The tracer keys everything on device pointers, and a test that has a wrong ciphertext in hand
// has only the ciphertext. The walk from one to the other crosses four private types
// (Ciphertext -> RNSPoly -> LimbPartition -> Limb -> VectorGPU); doing it here, once, is what
// keeps api/ConcurrentOps.hpp free of them.
//
// Nothing below launches, copies or synchronizes anything: it reads pointers and stream handles
// that are already there. `Stream::raw()` rather than `Stream::ptr()` on purpose -- ptr() clears
// the stream's `updated` flag as part of what launching means, and an instrument must not do
// that to the run it is measuring.

namespace {

/// Append every Q limb of `poly` (`which` = 0 for c0, 1 for c1) to `out`, in partition-then-limb
/// order. SPECIAL (P-basis) limbs are left out: a ciphertext that has been stored has none.
void collectPolyLimbPointers(const FIDESlib::CKKS::RNSPoly& poly, const int which, std::vector<fideslib::LimbBlockRef>& out) {
	for (size_t p = 0; p < poly.GPU.size(); ++p) {
		const FIDESlib::CKKS::LimbPartition& part = poly.GPU[p];
		for (size_t l = 0; l < part.limb.size(); ++l) {
			const FIDESlib::CKKS::LimbImpl& li = part.limb[l];
			fideslib::LimbBlockRef ref;
			ref.poly	  = which;
			ref.partition = static_cast<int>(p);
			ref.limb	  = static_cast<int>(l);
			if (li.index() == FIDESlib::TYPE::U32) {
				const FIDESlib::CKKS::Limb<uint32_t>& L = std::get<FIDESlib::TYPE::U32>(li);
				ref.data								= L.v.data;
				ref.stream								= L.stream.raw();
				ref.primeid								= L.primeid;
			} else {
				const FIDESlib::CKKS::Limb<uint64_t>& L = std::get<FIDESlib::TYPE::U64>(li);
				ref.data								= L.v.data;
				ref.stream								= L.stream.raw();
				ref.primeid								= L.primeid;
			}
			out.push_back(ref);
		}
	}
}

} // namespace

namespace fideslib {

/// See api/ConcurrentOps.hpp. Pointers only -- no device traffic, no stream mutation.
std::vector<LimbBlockRef> CiphertextLimbPointers(const FIDESlib::CKKS::Ciphertext& ct) {
	std::vector<LimbBlockRef> out;
	collectPolyLimbPointers(ct.c0, 0, out);
	collectPolyLimbPointers(ct.c1, 1, out);
	return out;
}

} // namespace fideslib

namespace FIDESlib::CKKS {

/// See the declaration in Ciphertext.cuh. This is the fourth thing the tracer records and the one
/// that makes the other three interpretable: without it a dump says a block changed hands, but
/// not that the block it changed hands over was the one a lane's bootstrap had just written its
/// answer into.
void PoolTraceMarkCiphertextOutput(const Ciphertext& ct) {
	// One cached-bool read in every run that has not asked for the tracer, which is all of them
	// by default.
	if (!FIDESlib::PoolTraceEnabled())
		return;
	const int lane								   = FIDESlib::MemPoolLane();
	const std::vector<fideslib::LimbBlockRef> refs = fideslib::CiphertextLimbPointers(ct);
	for (const fideslib::LimbBlockRef& r : refs) {
		// `bytes` 0 means "I am not claiming to know the size class": an output mark witnesses
		// that this lane finished writing the block, it is not an allocation. PoolTraceBlock
		// keeps whatever size the block's cut recorded.
		FIDESlib::PoolTraceBlock(FIDESlib::PoolTraceKind::kOutput, r.data, /*bytes=*/0, lane, /*consumer=*/r.stream, /*owner=*/nullptr,
								 /*ev=*/nullptr);
	}
}

} // namespace FIDESlib::CKKS
