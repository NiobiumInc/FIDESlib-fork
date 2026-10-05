//
// Created by carlosad on 12/11/24.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Checksum.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"
#include <cassert>
#include <cmath>
#include <memory>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

constexpr bool PRINT = false;

using namespace FIDESlib::CKKS;

void evalChebyshevSeries(Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, std::vector<double>& coefficients, double lower_bound, double upper_bound);
class FusedPowers;
void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval, double seriesScale = 1.0);

namespace {
/// The series factor c = outputScale^(1/2^its) that makes the double-angle iterations deliver
/// outputScale x EvalMod (see approxModReduction's declaration), and the stock coefficients
/// scaled by it. outputScale == 1.0 returns 1.0 and leaves `scaled` empty: the caller then hands
/// the stock vector to the series exactly as it always has.
double FoldSeriesScale(ContextData& cc, const double outputScale, std::vector<double>& scaled) {
	if (outputScale == 1.0)
		return 1.0;
	const double c = std::pow(outputScale, std::ldexp(1.0, -cc.GetDoubleAngleIts()));
	scaled		   = cc.GetCoeffsChebyshev();
	for (double& a : scaled)
		a *= c;
	return c;
}
} // namespace

void FIDESlib::CKKS::approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post, double outputScale) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kEvalMod);
	CudaNvtxRange r(std::string{ sc::current().function_name() });

	// cudaDeviceSynchronize();
	if constexpr (PRINT)
		std::cout << "Approx mod red start " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;

	bool constexpr COMPLEX = true;
	ContextData& cc        = ctxtEnc.cc;

	// Both halves take the same fold: they are added after the monomial shift below, so the sum
	// carries it once.
	std::vector<double> scaledCoeffs;
	const double seriesScale	  = FoldSeriesScale(cc, outputScale, scaledCoeffs);
	std::vector<double>& coeffs = seriesScale == 1.0 ? cc.GetCoeffsChebyshev() : scaledCoeffs;

	ChecksumProbe(ctxtEnc, "em-in");
	ChecksumProbe(ctxtEncI, "em-inI");
	if constexpr (COMPLEX)
		evalChebyshevSeries(ctxtEncI, coeffs, -1.0, 1.0);
	ChecksumProbe(ctxtEncI, "em-chebI");
	evalChebyshevSeries(ctxtEnc, coeffs, -1.0, 1.0);
	ChecksumProbe(ctxtEnc, "em-cheb");
	if constexpr (PRINT) {
		std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;

		std::cout << "ctxtEncI res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
		for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}

	applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey, seriesScale);
	ChecksumProbe(ctxtEnc, "em-da");
	if constexpr (COMPLEX)
		applyDoubleAngleIterations(ctxtEncI, cc.GetDoubleAngleIts(), keySwitchingKey, seriesScale);
	ChecksumProbe(ctxtEncI, "em-daI");
	if constexpr (PRINT) {
		std::cout << "ctxtEnc DA res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;

		std::cout << "ctxtEncI DA res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
		for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	// cudaDeviceSynchronize();
	if constexpr (COMPLEX)
		ctxtEncI.multMonomial(cc.N / 2);
	ChecksumProbe(ctxtEncI, "em-monoI");
	// cudaDeviceSynchronize();
	if constexpr (COMPLEX)
		ctxtEnc.add(ctxtEncI);
	ChecksumProbe(ctxtEnc, "em-add");
	if constexpr (!COMPLEX)
		ctxtEnc.add(ctxtEnc);
	// cudaDeviceSynchronize();
	multIntScalar(ctxtEnc, post);
	ChecksumProbe(ctxtEnc, "em-out");
	// cudaDeviceSynchronize();
	if constexpr (PRINT) {
		std::cout << "ctxtEnc final res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
}

void FIDESlib::CKKS::approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post, double outputScale) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kEvalMod);
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	ContextData& cc = ctxtEnc.cc;

	KeySwitchingKey& keySwitchingKey = cc.GetEvalKey(ctxtEnc.keyID);

	std::vector<double> scaledCoeffs;
	const double seriesScale = FoldSeriesScale(cc, outputScale, scaledCoeffs);
	evalChebyshevSeries(ctxtEnc, seriesScale == 1.0 ? cc.GetCoeffsChebyshev() : scaledCoeffs, (double)-1.0, (double)1.0);

	if constexpr (PRINT) {
		std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey, seriesScale);
	if constexpr (PRINT) {
		std::cout << "ctxtEnc DA " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	multIntScalar(ctxtEnc, post);
	if constexpr (PRINT) {
		std::cout << "ctxtEnc final " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	// cudaDeviceSynchronize();
}

void FIDESlib::CKKS::multIntScalar(Ciphertext& ctxt, uint64_t op) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	std::vector<uint64_t> op_(ctxt.getLevel() + 1, op);
	ctxt.c0.multScalar(op_);
	ctxt.c1.multScalar(op_);
}

// FIDESlib bit-compat: direct transcription of OpenFHE EvalPartialLinearWSum.
static void evalPartialLinearWSumCompat(Ciphertext& out, const std::vector<Ciphertext*>& ciphertexts,
                                        const std::vector<double>& constants, uint32_t limit) {
	FIDESlib::CKKS::Context& cc_ = ciphertexts[0]->cc_;
	ContextData& cc              = ciphertexts[0]->cc;

	std::vector<Ciphertext> cts;
	cts.reserve(limit);
	for (uint32_t i = 0; i < limit; ++i) {
		cts.emplace_back(cc_);
		cts[i].copy(*ciphertexts[i]);
	}

	if (cc.rescaleTechnique != FIDESlib::CKKS::FIXEDMANUAL) {
		// OpenFHE GetLevel() counts consumed levels; FIDESlib getLevel() counts remaining
		// towers, so OpenFHE "higher level" corresponds to a smaller getLevel() here.
		uint32_t maxIdx = 0;
		for (uint32_t i = 1; i < limit; ++i) {
			if ((cts[i].getLevel() < cts[maxIdx].getLevel()) ||
			    ((cts[i].getLevel() == cts[maxIdx].getLevel()) && (cts[i].NoiseLevel == 2))) {
				maxIdx = i;
			}
		}
		for (uint32_t i = 0; i < maxIdx; ++i)
			if (!cts[i].adjustForAddOrSub(cts[maxIdx]))
				cts[maxIdx].adjustForAddOrSub(cts[i]);
		for (uint32_t i = maxIdx + 1; i < limit; ++i)
			if (!cts[i].adjustForAddOrSub(cts[maxIdx]))
				cts[maxIdx].adjustForAddOrSub(cts[i]);

		if (cts[maxIdx].NoiseLevel == 2) {
			for (uint32_t i = 0; i < limit; ++i)
				cts[i].rescale();
		}
	}

	cts[0].multScalar(constants[1], false);
	for (uint32_t i = 1; i < limit; ++i) {
		cts[i].multScalar(constants[i + 1], false);
		cts[0].add(cts[i]);
	}
	if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
		cts[0].rescale();

	out.copy(cts[0]);
}

// Operands are fusible when they are mutually interchangeable as far as
// evalPartialLinearWSumCompat is concerned: equal tower count, equal noise degree and equal
// scaling factor. Its adjust loop is then a no-op (adjustScaleAndLevel returns true without
// touching data when level and depth match) and its rescale loop either fires for all operands
// or for none, so the whole block reduces to "rescale every operand iff the common noise degree
// is 2". Ragged tower counts -- what FIXEDAUTO produces, because its adjustForAddOrSub rescales
// without level-reducing -- are not fusible: evalLinearWSum reads limbsize limbs from every
// operand, so an operand shallower than the output would be read past its end.
static bool wsumOperandsFusible(const std::vector<Ciphertext*>& ctxs, uint32_t limit) {
	if (ctxs.size() < limit || limit == 0)
		return false;
	for (uint32_t i = 0; i < limit; ++i) {
		if (ctxs[i]->getLevel() != ctxs[0]->getLevel() || ctxs[i]->NoiseLevel != ctxs[0]->NoiseLevel || ctxs[i]->NoiseFactor != ctxs[0]->NoiseFactor)
			return false;
		if (ctxs[i]->keyID != ctxs[0]->keyID)
			return false;
	}
	return true;
}

// Degree-1 counterparts of the stored powers, for the fused weighted sums. The powers themselves
// stay pristine, so at noise degree 2 each one needs a private rescaled copy. Which powers a given
// series actually reads is only known once the Paterson-Stockmeyer divisions have run -- a
// degree-5 series reads T[0] alone -- so the copies are materialized on first use rather than all
// k up front. Materialization order is irrelevant: each copy is an independent rescale of one
// stored power, so any subset built in any order holds the same values the eager loop produced.
class FusedPowers {
	FIDESlib::CKKS::Context* cc_        = nullptr;
	const std::vector<Ciphertext*>* src = nullptr;
	bool direct                         = false; // powers already at degree 1: usable as they are
	bool owned                          = false; // powers at degree 2: private rescaled copies
	std::vector<std::unique_ptr<Ciphertext>> store;
	std::vector<Ciphertext*> view;

  public:
	void init(FIDESlib::CKKS::Context& cc, const std::vector<Ciphertext*>& T, uint32_t k);
	bool ready(uint32_t i) const { return direct || (owned && store[i] != nullptr); }
	Ciphertext* at(uint32_t i);
	// Materializes entries 0..limit-1 and returns the operand view, or nullptr when the fused
	// path does not apply to these powers.
	const std::vector<Ciphertext*>* prepare(uint32_t limit);
};

// Fused counterpart of evalPartialLinearWSumCompat, for operands already at noise degree 1 and a
// common tower count. `out` is empty, so evalLinearWSumMutable sizes it from ctxs[0], which under
// that precondition is also the shallowest operand -- the tower-count half of the fused kernel's
// contract. ElemForEvalMult then sees level_in == level and yields the same per-tower scalar
// multScalar would, and the kernel accumulates in the same ascending order, so the sum is
// bit-identical to the unfused copy/multScalar/add chain.
static void evalPartialLinearWSumFused(Ciphertext& out, const std::vector<Ciphertext*>& ctxs, const std::vector<double>& constants, uint32_t limit) {
	ContextData& cc = ctxs[0]->cc;
	assert(ctxs[0]->NoiseLevel == 1);
	assert(out.getLevel() == -1);

	std::vector<double> weights(constants.begin() + 1, constants.begin() + 1 + limit);
	out.evalLinearWSumMutable(limit, ctxs, weights);
	if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
		out.rescale();
}

void FusedPowers::init(FIDESlib::CKKS::Context& cc, const std::vector<Ciphertext*>& T, uint32_t k) {
	cc_ = &cc;
	src = &T;
	if (!wsumOperandsFusible(T, k))
		return;
	if (T[0]->NoiseLevel == 1) {
		direct = true;
		view   = T;
		return;
	}
	if (T[0]->NoiseLevel != 2)
		return;
	owned = true;
	store.resize(k);
	view.assign(k, nullptr);
}

Ciphertext* FusedPowers::at(uint32_t i) {
	if (direct)
		return (*src)[i];
	if (!owned)
		return nullptr;
	if (store[i] == nullptr) {
		store[i] = std::make_unique<Ciphertext>(*cc_);
		store[i]->copy(*(*src)[i]);
		store[i]->rescale();
		view[i] = store[i].get();
	}
	return view[i];
}

const std::vector<Ciphertext*>* FusedPowers::prepare(uint32_t limit) {
	if ((!direct && !owned) || limit == 0 || view.size() < limit)
		return nullptr;
	for (uint32_t i = 0; i < limit; ++i)
		if (at(i) == nullptr)
			return nullptr;
	return &view;
}

// `fused` holds the same powers as `powers`, already at noise degree 1, when the caller could
// establish that; it holds nothing otherwise and the unfused transcription runs. An `out` that
// already carries towers is also left to the unfused path: the fused primitive would then size
// itself from `out` rather than from the operands, which is how the tower-count contract gets
// violated.
static void evalPartialLinearWSum(Ciphertext& out,
                                  const std::vector<Ciphertext*>& powers,
                                  FusedPowers& fused,
                                  const std::vector<double>& constants,
                                  uint32_t limit) {
	const std::vector<Ciphertext*>* f = out.getLevel() == -1 ? fused.prepare(limit) : nullptr;
	if (f != nullptr && (*f)[0]->NoiseLevel == 1 && wsumOperandsFusible(*f, limit))
		evalPartialLinearWSumFused(out, *f, constants, limit);
	else
		evalPartialLinearWSumCompat(out, powers, constants, limit);
}

// FIDESlib bit-compat: direct transcription of OpenFHE InnerEvalChebyshevPS.
void innerEvalChebyshevPS(const Ciphertext& x,
                          Ciphertext& out,
                          const std::vector<double>& coefficients,
                          const uint32_t k,
                          uint32_t m,
                          const std::vector<Ciphertext*>& T,
                          FusedPowers& Tw,
                          const std::vector<Ciphertext*>& T2,
                          int level_offset = 0,
                          int max_m        = 1000) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
	FIDESlib::CKKS::Context& cc_ = x.cc_;
	ContextData& cc              = x.cc;
	(void)level_offset;
	(void)max_m;

	// Compute k*2^{m-1}-k because we use it a lot
	uint32_t k2m2k = k * (1 << (m - 1)) - k;

	// Divide coefficients by T^{k*2^{m-1}}
	std::vector<double> Tkm(k2m2k + k + 1, 0.0);
	Tkm.back() = 1;
	auto divqr = lbcrypto::LongDivisionChebyshev(coefficients, Tkm);

	// Subtract x^{k(2^{m-1} - 1)} from r
	std::vector<double> r2 = divqr->r;
	if (uint32_t n = lbcrypto::Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
		r2.resize(n + 1);
		r2[k2m2k] -= 1;
	} else {
		r2.resize(k2m2k + 1, 0.0);
		r2.back() = -1;
	}

	auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

	Ciphertext cu(cc_), qu(cc_), su(cc_);
	bool has_cu = false;

	{
		// Evaluate q and s2 at u. If their degrees are larger than k, then recursively
		// apply the Paterson-Stockmeyer algorithm.
		if (lbcrypto::Degree(divqr->q) > k) {
			innerEvalChebyshevPS(x, qu, divqr->q, k, m - 1, T, Tw, T2);
		} else {
			// the highest order coefficient is always a power of two up to 2^{m-1}:
			// scale T[k-1] by repeated doubling instead of a multiplication.
			qu.copy(*T[k - 1]);
			ChecksumProbe(qu, "ps-qu-copy");
			// ROUND, do not truncate. The exponent is recovered from a double that the Chebyshev
			// long division produced; at 1.9999999999999998 a truncating conversion returns 0 and
			// silently drops the whole leading term instead of doubling it once. The power-of-two
			// invariant is asserted so a table that breaks it fails loudly rather than quietly.
			const double q_back	 = divqr->q.back();
			const uint32_t limit = static_cast<uint32_t>(std::lround(std::log2(q_back)));
			assert(q_back > 0.0 && std::abs(q_back - std::exp2((double)limit)) < 1e-9 * q_back);
			for (uint32_t i = 0; i < limit; ++i)
				qu.add(qu);
			ChecksumProbe(qu, "ps-qu-add");

			// adds the free term (at x^0)
			qu.addScalar(divqr->q.front() / 2.0);
			ChecksumProbe(qu, "ps-qu-a");

			divqr->q.resize(k);
			if (uint32_t n = lbcrypto::Degree(divqr->q); n > 0) {
				Ciphertext ws(cc_);
				evalPartialLinearWSum(ws, T, Tw, divqr->q, n);
				qu.add(ws);
				ChecksumProbe(qu, "ps-qu-b");
			}
		}
	}

	{
		// Add x^{k(2^{m-1} - 1)} to s
		std::vector<double>& s2 = divcs->r;
		s2.resize(k2m2k + 1, 0.0);
		s2.back() = 1;

		if (lbcrypto::Degree(s2) > k) {
			innerEvalChebyshevPS(x, su, s2, k, m - 1, T, Tw, T2);
		} else {
			// the highest order coefficient is always 1 because s2 is monic.
			su.copy(*T[k - 1]);
			ChecksumProbe(su, "ps-su-copy");

			s2.resize(k);
			if (uint32_t n = lbcrypto::Degree(s2); n > 0) {
				Ciphertext ws(cc_);
				evalPartialLinearWSum(ws, T, Tw, s2, n);
				su.add(ws);
			}
			ChecksumProbe(su, "ps-su-add");

			// adds the free term (at x^0)
			su.addScalar(s2.front() / 2.0);
			ChecksumProbe(su, "ps-su");

			// LevelReduceInPlace(su, nullptr): only FIXEDMANUAL actually drops a level;
			// it is a no-op for the FLEXIBLE/FIXEDAUTO scaling techniques.
			if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
				su.dropToLevel(su.getLevel() - 1, true);
		}
	}

	if (uint32_t n = lbcrypto::Degree(divcs->q); n >= 1) {
		if (n == 1) {
			if (lbcrypto::IsNotEqualOne(divcs->q[1])) {
				// multScalar(*T[0], c) copies T[0] and, at noise degree 2, rescales that copy to
				// degree 1 before scaling it. When the weighted sums have already materialized
				// exactly that rescale of T[0], copy it instead of computing it a second time.
				// FIXEDMANUAL is excluded because multScalar does not pre-rescale there.
				const bool reuse = cc.rescaleTechnique != FIDESlib::CKKS::FIXEDMANUAL && Tw.ready(0);
				if (reuse) {
					cu.copy(*Tw.at(0));
					cu.multScalarNoPrecheck(divcs->q[1], false);
				} else {
					cu.multScalar(*T[0], divcs->q[1], false);
				}
				if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
					cu.rescale();
			} else {
				cu.copy(*T[0]);
			}
		} else {
			evalPartialLinearWSum(cu, T, Tw, divcs->q, n);
		}

		// adds the free term (at x^0)
		ChecksumProbe(cu, "ps-cu-pre");
		cu.addScalar(divcs->q.front() / 2.0);
		ChecksumProbe(cu, "ps-cu");

		// LevelReduceInPlace(cu, ...): only FIXEDMANUAL actually drops levels;
		// it is a no-op for the FLEXIBLE/FIXEDAUTO scaling techniques.
		if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL && cu.getLevel() > T2[m - 1]->getLevel())
			cu.dropToLevel(T2[m - 1]->getLevel(), true);
		has_cu = true;
	}

	// T2[m-1] + cu. Addition over the RNS limbs is exact and commutative, so accumulating into cu
	// -- a temporary this call owns -- yields the same ciphertext without copying T2[m-1] out. Only
	// taken when the two are already aligned, so neither add() adjusts anything.
	Ciphertext t2store(cc_);
	Ciphertext* t2sum = &t2store;
	if (has_cu && cu.getLevel() == T2[m - 1]->getLevel() && cu.NoiseLevel == T2[m - 1]->NoiseLevel &&
	    cu.NoiseFactor == T2[m - 1]->NoiseFactor) {
		cu.add(*T2[m - 1]);
		t2sum = &cu;
	} else {
		t2store.copy(*T2[m - 1]);
		if (has_cu)
			t2store.add(cu);
		else
			t2store.addScalar(divcs->q.front() / 2.0);
	}

	// mult(a, b) copies the shallower operand into `out` and, when both are at noise degree 2,
	// rescales `out` and then a private clone of the other operand. Both operands are temporaries
	// this call owns, so rescaling them in place computes the same two degree-1 values and drops
	// the clone.
	if (cc.rescaleTechnique != FIDESlib::CKKS::FIXEDMANUAL && t2sum->getLevel() == qu.getLevel() &&
	    t2sum->NoiseLevel == 2 && qu.NoiseLevel == 2 && t2sum != &qu) {
		out.copy(*t2sum);
		out.rescale();
		qu.rescale();
		out.mult(qu, false);
	} else {
		out.mult(*t2sum, qu, false);
	}
	if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
		out.rescale();

	// add() clones su whenever su sits above out, because it may not modify a const operand. su is
	// a temporary this call owns; adjusting it in place runs the same adjustment on the same value.
	if (out.getLevel() < su.getLevel() &&
	    (cc.rescaleTechnique == FIDESlib::CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == FIDESlib::CKKS::FLEXIBLEAUTOEXT))
		su.adjustForAddOrSub(out);
	out.add(su);
	ChecksumProbe(out, "ps-out");
}

/**
 * Adaptation of OpenFHE's implementation.
 */
void FIDESlib::CKKS::evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound, double upper_bound) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
	/*
	Ciphertext<DCRTPoly> AdvancedSHECKKSRNS::EvalChebyshevSeriesPS(ConstCiphertext<DCRTPoly> x,
const std::vector<double>& coefficients, double a, double b) const {
	*/

	if (abs(lower_bound + 1.0) > 1e-9 || abs(upper_bound - 1.0) > 1e-9) {
		// Domain map [a, b] -> [-1, 1]: t = (x - (a+b)/2) * 2/(b-a).
		// (Fixed: this subtracted the half-WIDTH (b-a)/2 instead of the MEAN
		// (a+b)/2 — correct only for symmetric intervals, so the bootstrap's
		// internal (-1,1) calls never exercised the bug, while asymmetric
		// application intervals evaluated the series at shifted points. The
		// width-2 shortcut below had the matching sign error, -a+1 for -(a+1).
		// Reference: lbcrypto::AdvancedSHECKKSRNS::EvalChebyshevSeriesPS.)
		if (abs(upper_bound - lower_bound - 2.0) < 1e-8) {
			ctxt.addScalar(-(lower_bound + 1.0));
		} else {
			if (abs(lower_bound + upper_bound) > 1e-8)
				ctxt.addScalar(-(upper_bound + lower_bound) / 2.0); // center on 0 (subtract mean)
			if (ctxt.cc.rescaleTechnique == CKKS::FIXEDMANUAL && ctxt.NoiseLevel == 2)
				ctxt.rescale();
			ctxt.multScalar(2.0 / (upper_bound - lower_bound));
		}
	}

	constexpr bool sync = false;

	uint32_t n             = lbcrypto::Degree(coefficients);
	std::vector<double> f2 = coefficients;
	f2.resize(n + 1);
	/*
	 uint32_t n = Degree(coefficients);
	std::vector<double> f2 = coefficients;
	// Make sure the coefficients do not have the zero dominant terms
	if (coefficients[coefficients.size() - 1] == 0)
		f2.resize(n + 1);
	*/

	std::vector<uint32_t> degs = lbcrypto::ComputeDegreesPS(n);
	uint32_t k                 = degs[0];
	uint32_t m                 = degs[1];
	if (false) {
		if (n <= 36) {
			k = 12;
			m = 2;
		} else if (n <= 92) {
			k = 13;
			m = 3;
		}
	}
	/*
	std::vector<uint32_t> degs = ComputeDegreesPS(n);
	uint32_t k                 = degs[0];
	uint32_t m                 = degs[1];
	// std::cerr << "\n Degree: n = " << n << ", k = " << k << ", m = " << m << std::endl;
	*/
	// assert((lower_bound - std::round(lower_bound) < 1e-10) && (upper_bound - std::round(upper_bound) < 1e-10) &&
	//        (std::round(lower_bound) == -1) && (std::round(upper_bound) == 1));

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc              = ctxt.cc;
	/*
	std::vector<Ciphertext> T_;
	T_.emplace_back(cc);
	for (uint32_t i = 1; i < k; ++i)
		T_.emplace_back(cc);
	std::vector<Ciphertext> T2_;
	T2_.emplace_back(cc);
	for (uint32_t i = 1; i < m; i++) {
		T2_.emplace_back(cc);
	}
*/
	std::vector<Ciphertext> aux;
	for (size_t i = aux.size(); i < k + m; i++) {
		aux.emplace_back(cc_);
	}

	std::vector<Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i]        = &aux[i];
	std::vector<Ciphertext*> T2(m);
	for (uint32_t i = 0; i < m; i++)
		T2[i]       = &aux[i + k];
	/*
	std::vector<Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i] = &T_[i];
	std::vector<Ciphertext*> T2(m);
	for (uint32_t i = 0; i < m; i++)
		T2[i] = &T2_[i];
*/
	T[0]->copy(ctxt);
	ChecksumProbe(*T[0], "cheb-T0");
	/*
	// computes linear transformation y = -1 + 2 (x-a)/(b-a)
	// consumes one level when a <> -1 && b <> 1
	auto cc = x->GetCryptoContext();
	std::vector<Ciphertext<DCRTPoly>> T(k);
	if ((a - std::round(a) < 1e-10) && (b - std::round(b) < 1e-10) && (std::round(a) == -1) && (std::round(b) == 1)) {
		// no linear transformation is needed if a = -1, b = 1
		// T_1(y) = y
		T[0] = x->Clone();
	}
	else {
		// linear transformation is needed
		double alpha = 2 / (b - a);
		double beta  = 2 * a / (b - a);

		T[0] = cc->EvalMult(x, alpha);
		cc->ModReduceInPlace(T[0]);
		cc->EvalAddInPlace(T[0], -1.0 - beta);
	}
	*/
	// Ciphertext y(cc);
	// y.copy(T[0]);

	// if (ctxt.NoiseLevel == 1)
	//     ctxt.multScalar(1.0);
	// Rescale the base power before the basis is derived from it, as upstream does. Every
	// power then carries one fewer tower for the rest of the series, including the whole
	// T2/T2km1 chain. FIXEDMANUAL keeps the transcribed placement: it rescales explicitly
	// inside the loop below, so an eager rescale here would double-count.
	if (cc.rescaleTechnique != CKKS::FIXEDMANUAL && T[0]->NoiseLevel == 2)
		T[0]->rescale();
	// FIDESlib bit-compat: transcribed from OpenFHE internalEvalChebyPolysPS. The stored
	// powers stay pristine: mult/square/sub clone and adjust their operands internally,
	// exactly like EvalMult/EvalSquare/EvalSubInPlace do on const operands.
	for (uint32_t i = 2; i <= k; i++) {
		if (i > 2)
			ChecksumProbe(*T[i - 2], "cheb-T" + std::to_string(i - 2));
		if constexpr (sync)
			cudaDeviceSynchronize();

		if (i % 2 == 1) {
			// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
			T[i - 1]->mult(*T[i / 2 - 1], *T[i / 2], false);
			T[i - 1]->add(*T[i - 1]);
			if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				T[i - 1]->rescale();
			T[i - 1]->sub(*T[0]);
		} else {
			// compute T_{2i}(y) = 2*T_i(y)^2 - 1
			T[i - 1]->square(*T[i / 2 - 1], false);
			T[i - 1]->add(*T[i - 1]);
			if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				T[i - 1]->rescale();
			T[i - 1]->addScalar(-1.0);
		}
	}
	ChecksumProbe(*T[k - 1], "cheb-T" + std::to_string(k - 1));

	if constexpr (PRINT) {
		for (size_t j = 0; j < k; ++j) {
			std::cout << "T[" << j << "]: " << std::endl;
			for (auto& i : T[j]->c0.GPU.at(0).limb) {
				cudaSetDevice(T[j]->c0.GPU.at(0).device);
				SWITCH(i, printThisLimb(1));
			}
			std::cout << std::endl;
		}
	}
	// cudaDeviceSynchronize();
	/*
	Ciphertext<DCRTPoly> y = T[0]->Clone();

	// Computes Chebyshev polynomials up to degree k
	// for y: T_1(y) = y, T_2(y), ... , T_k(y)
	// uses binary tree multiplication
	for (uint32_t i = 2; i <= k; i++) {
		// if i is a power of two
		if (!(i & (i - 1))) {
			// compute T_{2i}(y) = 2*T_i(y)^2 - 1
			auto square = cc->EvalSquare(T[i / 2 - 1]);
			T[i - 1] = cc->EvalAdd(square, square);
			cc->ModReduceInPlace(T[i - 1]);
			cc->EvalAddInPlace(T[i - 1], -1.0);
		} else {
			// non-power of 2
			if (i % 2 == 1) {
				// if i is odd
				// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
				auto prod = cc->EvalMult(T[i / 2 - 1], T[i / 2]);
				T[i - 1] = cc->EvalAdd(prod, prod);

				cc->ModReduceInPlace(T[i - 1]);
				cc->EvalSubInPlace(T[i - 1], y);
			} else {
				// i is even but not power of 2
				// compute T_{2i}(y) = 2*T_i(y)^2 - 1
				auto square = cc->EvalSquare(T[i / 2 - 1]);
				T[i - 1] = cc->EvalAdd(square, square);
				cc->ModReduceInPlace(T[i - 1]);
				cc->EvalAddInPlace(T[i - 1], -1.0);
			}
		}
	}
	*/
	if (cc.rescaleTechnique == CKKS::FIXEDMANUAL) {

		for (size_t i = 1; i < k; i++) {
			T[i - 1]->dropToLevel(T[k - 1]->getLevel());
		}
	} else {
		// Settle every power to noise degree 1 here, as upstream does, so the weighted sums
		// and the T2 chain all read one-tower-shallower operands.
		for (size_t i = 1; i <= k; i++) {
			if (T[i - 1]->NoiseLevel == 2)
				T[i - 1]->rescale();
		}
		// FIDESlib bit-compat: mirror OpenFHE AdjustLevelsAndDepthInPlace(T[i-1], T[k-1]).
		for (size_t i = 1; i < k; i++) {
			if (!T[i - 1]->adjustForAddOrSub(*T[k - 1]))
				T[k - 1]->adjustForAddOrSub(*T[i - 1]);
		}
		for (size_t i = 1; i <= k; i++)
			ChecksumProbe(*T[i - 1], "cheb-Tadj" + std::to_string(i - 1));

		/*
		assert(k >= 2);
		assert(T[k - 1]->getLevel() - T[k - 1]->NoiseLevel == T[k - 2]->getLevel() - T[k - 2]->NoiseLevel - 1);
		for (size_t i = 1; i < k - 1; i++) {
			if (!T[i - 1]->adjustForMult(*T[k - 2])) {
				//if (!T[k - 1]->adjustForAddOrSub(*T[i - 1])) {
				assert("false");
				//  std::cerr << "PANIC" << std::endl;
				//}
			}
			//algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
		}
		*/
	}
	/*
	const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(T[k - 1]->GetCryptoParameters());

	auto algo = cc->GetScheme();

	if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
		// brings all powers of x to the same level
		for (size_t i = 1; i < k; i++) {
			uint32_t levelDiff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
			cc->LevelReduceInPlace(T[i - 1], nullptr, levelDiff);
		}
	} else {
		for (size_t i = 1; i < k; i++) {
			algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
		}
	}
	 */



	// Compute the Chebyshev polynomials T_k(y), T_{2k}(y), ..., T_{2^{m-1}k}(y) and
	// T_{k(2m-1)}(y), interleaved exactly like OpenFHE internalEvalChebyPolysPS.
	Ciphertext T2km1(cc_);
	// T2[0] is only ever read, and its one assignment makes it a copy of T[k-1]; point at T[k-1]
	// instead of duplicating it.
	T2[0] = T.back();
	T2km1.copy(*T.back());
	for (uint32_t i = 1; i < m; i++) {
		T2[i]->square(*T2[i - 1], false);
		T2[i]->add(*T2[i]);
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			T2[i]->rescale();
		T2[i]->addScalar(-1.0);

		// compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
		T2km1.mult(*T2[i], false);
		T2km1.add(T2km1);
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			T2km1.rescale();
		T2km1.sub(*T2[0]);
		ChecksumProbe(*T2[i], "cheb-T2-" + std::to_string(i));
		ChecksumProbe(T2km1, "cheb-T2km1-" + std::to_string(i));

		if constexpr (sync)
			cudaDeviceSynchronize();
	}

	if constexpr (PRINT) {
		for (size_t j = 0; j < m; ++j) {
			std::cout << "T2[" << j << "]: " << std::endl;

			for (auto& i : T2[j]->c0.GPU.at(0).limb) {
				cudaSetDevice(T2[j]->c0.GPU.at(0).device);
				SWITCH(i, printThisLimb(1));
			}
			std::cout << std::endl;
		}
		std::cout << "T2kmi cheby " << T2km1.getLevel() << " " << T2km1.NoiseLevel << std::endl;
		for (auto& i : T2km1.c0.GPU.at(0).limb) {
			cudaSetDevice(T2km1.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}



	// Operands for the fused weighted sums inside innerEvalChebyshevPS. The stored powers stay
	// pristine: every other consumer (T2, T2km1, the qu/su/cu copies) must keep seeing them
	// exactly as the bit-compat transcription left them. At noise degree 1 they are already
	// valid fused operands and are used directly; at degree 2 the fused primitive needs degree-1
	// inputs, so a power is rescaled once into a private copy on first use rather than once per
	// weighted sum, which is the same rescale evalPartialLinearWSumCompat performs on its own
	// copies. Powers that are not mutually interchangeable leave the cache unusable, keeping the
	// unfused path.
	FusedPowers Tw;
	Tw.init(cc_, T, k);

	if constexpr (true) {
		uint32_t k2m2k = k * (1 << (m - 1)) - k;
		f2.resize(2 * k2m2k + k + 1, 0.0);
		f2.back() = 1;
		innerEvalChebyshevPS(ctxt, ctxt, f2, k, m, T, Tw, T2, 0, m);
		ChecksumProbe(ctxt, "cheb-ps");
	}


	ctxt.sub(T2km1);
	ChecksumProbe(ctxt, "cheb-out");
	/*

	Ciphertext<DCRTPoly> result;

	if (flag_c) {
		result = cc->EvalAdd(T2[m - 1], cu);
	} else {
		result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);
	}

	result = cc->EvalMult(result, qu);
	cc->ModReduceInPlace(result);

	cc->EvalAddInPlace(result, su);
	cc->EvalSubInPlace(result, T2km1);

	return result;
	*/
	if constexpr (sync)
		cudaDeviceSynchronize();
}

/// @param seriesScale  The factor c the series' coefficients were multiplied by (FoldSeriesScale).
///   Iteration j squares a value that carries c^(2^(j-1)), so its constant is multiplied by
///   c^(2^j) to stay in step; the factor is carried by repeated squaring of the same double so
///   the constants agree with what the squarings actually produce. 1.0 adds the stock constants.
void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval, double seriesScale) {
	FIDESlib::CudaNvtxRange r_(std::string{ sc::current().function_name() });
	ContextData& cc = ctxt.cc;
	int32_t r       = its;
	double carried	= seriesScale;
	// std::cout << "Its: " << its << std::endl;
	for (int32_t j = 1; j < r + 1; j++) {
		ctxt.square(false);
		ctxt.add(ctxt);
		double scalar = -std::pow(2.0 * M_PI, -std::pow(2.0, j - r));
		if (seriesScale != 1.0) {
			carried *= carried;
			scalar *= carried;
		}
		ctxt.addScalar(scalar);
		if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
			ctxt.rescale();
	}
}
