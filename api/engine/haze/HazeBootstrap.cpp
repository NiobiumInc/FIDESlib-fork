#ifdef FIDESLIB_ENABLE_HAZE

// Bootstrap orchestration. The CPU oracle is OpenFHE
// FHECKKSRNS::EvalBootstrap (ckksrns-fhe.cpp; sparsely-packed branch, levelBudget {1,1}
// = single BSGS linear transform per direction); FIDESlib src/CKKS/Bootstrap.cu:169-320
// is the cross-reference for the stage list. Precision bar is 1e-2, not bit parity.

#include "engine/haze/HazeEngine.hpp"

#include "CryptoContext.hpp"

#include <haze/haze.h>

#include <openfhe.h>

#include <any>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fideslib {

namespace {

using hazebk::HazePayload;
using hazebk::LimbChain;
using hazebk::Residency;

void hazeCheck(hazeError_t err, const char* what) {
	if (err != HAZE_SUCCESS) {
		throw std::runtime_error(std::string("haze backend: ") + what + " failed: " + hazeGetErrorString(err));
	}
}

std::shared_ptr<HazePayload> devicePayload(const Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<std::shared_ptr<HazePayload>>(ct->device);
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& hostContext(CryptoContextImpl<DCRTPoly>& ctx) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
}

// Wrap a device-computed result in a new value-type Ciphertext (same helper as
// HazeEngine.cpp's file-local wrapDeviceResult; see the rationale there).
Ciphertext<DCRTPoly> wrapDeviceResult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& proto, std::shared_ptr<HazePayload> payload) {
	Ciphertext<DCRTPoly> res = std::make_shared<CiphertextImpl<DCRTPoly>>(ctx.self_reference.lock());
	res->host				 = proto->host;
	res->need_lazy_copy		 = true;
	res->device				 = std::make_any<std::shared_ptr<HazePayload>>(std::move(payload));
	return res;
}

} // namespace

// ---- precomputation extraction (host-only; ExtractBootPrecom) ----

void HazeEngine::extractBootPrecom(CryptoContextImpl<DCRTPoly>& ctx, uint32_t slots) {
	if (boot_.count(slots) != 0) {
		return;
	}
	auto& context = hostContext(ctx);
	const auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE);
	if (!fhe) {
		throw std::runtime_error("haze backend: context has no CKKS FHE scheme (EvalBootstrapSetup not run?)");
	}
	const auto precomIt = fhe->m_bootPrecomMap.find(slots);
	if (precomIt == fhe->m_bootPrecomMap.end()) {
		throw std::runtime_error("haze backend: no bootstrap precomputation for " + std::to_string(slots) + " slots (run EvalBootstrapSetup before LoadContext)");
	}
	const auto& precom = precomIt->second;

	BootPrecom bp;
	bp.slots			= slots;
	bp.btSlotsEncoding	= precom->BTSlotsEncoding;
	bp.isLT				= (precom->m_paramsEnc.lvlb == 1) && (precom->m_paramsDec.lvlb == 1);
	bp.bStep = (precom->m_paramsEnc.g == 0) ? static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(slots)))) : static_cast<uint32_t>(precom->m_paramsEnc.g);
	bp.correctionFactor = fhe->GetCKKSBootCorrectionFactor();

	// Wrap the precomputed linear-transform plaintexts as facade values so the engine's
	// pt machinery (lazy H2D, trimmed multPt) applies unchanged. Deliberately default-
	// constructed (no parent_context): the engine's pt path never reads it, and a parent
	// reference here would cycle context -> engine -> boot_ -> Plaintext -> context,
	// leaking the context (its teardown would never run and the process-global haze
	// engine guard would reject every later LoadContext).
	auto wrapPts = [](const std::vector<lbcrypto::ReadOnlyPlaintext>& src) {
		std::vector<Plaintext> out;
		out.reserve(src.size());
		for (const auto& roPt : src) {
			Plaintext pt = std::make_shared<PlaintextImpl>();
			pt->host	 = std::make_any<lbcrypto::Plaintext>(std::const_pointer_cast<lbcrypto::PlaintextImpl>(roPt));
			out.push_back(std::move(pt));
		}
		return out;
	};
	if (bp.isLT) {
		bp.u0hatTPre = wrapPts(precom->m_U0hatTPre);
		bp.u0Pre	 = wrapPts(precom->m_U0Pre);
	} else {
		// Multi-stage FFT path (levelBudget != {1,1}): wrap m_U0hatTPreFFT / m_U0PreFFT per
		// stage and copy the BSGS params, so evalCoeffsToSlotsFFT can iterate the per-stage LT
		// vectors (Bootstrap.cu:261-296; CoeffsToSlots.cu:75-151). Plaintext shells stay
		// default-constructed (no parent_context) for the same lifetime reason as above.
		auto wrapStages = [&](const std::vector<std::vector<lbcrypto::ReadOnlyPlaintext>>& src) {
			std::vector<std::vector<Plaintext>> out;
			out.reserve(src.size());
			for (const auto& stage : src) {
				out.push_back(wrapPts(stage));
			}
			return out;
		};
		bp.u0hatTPreFFT = wrapStages(precom->m_U0hatTPreFFT);
		bp.u0PreFFT		= wrapStages(precom->m_U0PreFFT);
		auto copyParams = [](const lbcrypto::ckks_boot_params& s) {
			BootPrecom::FFTParams d;
			d.lvlb = s.lvlb, d.layersCollapse = s.layersCollapse, d.remCollapse = s.remCollapse;
			d.numRotations = s.numRotations, d.b = s.b, d.g = s.g;
			d.numRotationsRem = s.numRotationsRem, d.bRem = s.bRem, d.gRem = s.gRem;
			return d;
		};
		bp.paramsEnc = copyParams(precom->m_paramsEnc);
		bp.paramsDec = copyParams(precom->m_paramsDec);
	}

	// Chebyshev configuration by secret-key distribution (mirrors EvalBootstrap's table
	// selection; cross-checked with GetRawParams' bootConfig and EngineCommon.cpp).
	switch (ctx.keyDist) {
	case SPARSE_TERNARY:
		bp.coefficients = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		bp.k			= 1.0; // pre-divided during precomputation
		bp.numIter		= lbcrypto::FHECKKSRNS::R_SPARSE;
		break;
	case SPARSE_ENCAPSULATED:
		bp.coefficients = lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated;
		bp.k			= 1.0;
		bp.numIter		= lbcrypto::FHECKKSRNS::R_SPARSE;
		break;
	default: // UNIFORM_TERNARY
		bp.coefficients = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		bp.k			= lbcrypto::FHECKKSRNS::K_UNIFORM;
		bp.numIter		= lbcrypto::FHECKKSRNS::R_UNIFORM;
		break;
	}

	boot_.emplace(slots, std::move(bp));
}

// ---- device cores ----

HazeEngine::Operand HazeEngine::modRaiseCore(const Operand& x, size_t targetTowers) {
	// OpenFHE raise: only the level-0 limb is used; it is reinterpreted at the raised chain
	// (DCRTPoly(tmp, elementParamsRaised); FLEXIBLEAUTOEXT pops the extra modulus, hence
	// the parameterized target). Device-side: INTT@{q0} -> hazeBasisConvert ({q0} -> Q') ->
	// NTT@Q' on both components. Metadata (NSD/sf) carries through.
	const std::vector<uint64_t> base1	  = { qBase_.front() };
	const std::vector<uint64_t> raisedBase = qPrefix(targetTowers);
	const hazeBasisConvertParams convParams = {
		/*.src_base =*/base1.data(),
		/*.src_base_len =*/base1.size(),
		/*.dst_base =*/raisedBase.data(),
		/*.dst_base_len =*/raisedBase.size(),
	};
	const size_t fullTowers = targetTowers;

	auto raiseChain = [&](const LimbChain& src) {
		LimbChain intt(1, polyBytes_);
		hazeCheck(hazeINTTMrp(intt.data(), src.asConst().data(), base1.data(), base1.size(), nullptr), "hazeINTTMrp");
		LimbChain conv(fullTowers, polyBytes_);
		hazeCheck(hazeBasisConvert(conv.data(), intt.asConst().data(), &convParams, nullptr), "hazeBasisConvert");
		LimbChain ntt(fullTowers, polyBytes_);
		hazeCheck(hazeNTTMrp(ntt.data(), conv.asConst().data(), raisedBase.data(), raisedBase.size(), nullptr), "hazeNTTMrp");
		return ntt;
	};

	LimbChain out0 = raiseChain(x.p->c0);
	LimbChain out1 = raiseChain(x.p->c1);

	Operand res = x;
	res.towers	= fullTowers;
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
	return res;
}

HazeEngine::Operand HazeEngine::multIntCore(const Operand& x, uint64_t scalar) {
	// OpenFHE MultByIntegerInPlace: per-limb scalar mod q_i; no metadata change.
	std::vector<uint64_t> scalars(x.towers);
	for (size_t i = 0; i < x.towers; ++i) {
		scalars[i] = scalar % qBase_[i];
	}
	const auto base = qPrefix(x.towers);
	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeMulScalarMrp(out0.data(), x.p->c0.asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	hazeCheck(hazeMulScalarMrp(out1.data(), x.p->c1.asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");

	Operand res = x;
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
	return res;
}

// ---- BSGS linear transform (plain-rotation equivalent of OpenFHE EvalLinearTransform) ----

Ciphertext<DCRTPoly> HazeEngine::linearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Plaintext>& a, const Ciphertext<DCRTPoly>& ct, uint32_t bStep) {
	// result = sum_j rot_{bStep*j}( sum_i rot_i(ct) * A[bStep*j + i] ) — identical math to
	// OpenFHE's hoisted-Ext implementation, with plain rotations (no hoisting in v1).
	const uint32_t slots = static_cast<uint32_t>(a.size());
	const uint32_t gStep = static_cast<uint32_t>(std::ceil(static_cast<double>(slots) / bStep));

	std::vector<Ciphertext<DCRTPoly>> rot(bStep);
	rot[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	for (uint32_t i = 1; i < bStep; ++i) {
		rot[i] = ctx.EvalRotate(ct, static_cast<int32_t>(i));
	}

	Ciphertext<DCRTPoly> result;
	for (uint32_t j = 0; j < gStep; ++j) {
		auto inner = ctx.EvalMult(rot[0], const_cast<Plaintext&>(a[bStep * j]));
		for (uint32_t i = 1; i < bStep; ++i) {
			if (bStep * j + i < slots) {
				auto term = ctx.EvalMult(rot[i], const_cast<Plaintext&>(a[bStep * j + i]));
				ctx.EvalAddInPlace(inner, term);
			}
		}
		if (j == 0) {
			result = inner;
		} else {
			auto rotated = ctx.EvalRotate(inner, static_cast<int32_t>(bStep * j));
			ctx.EvalAddInPlace(result, rotated);
		}
	}
	return result;
}

// ---- staged bootstrap (sparse path) ----

namespace {
// Debug bisection hook: FIDESLIB_HAZE_BOOT_DEBUG_STAGE=<k> makes bootstrapStaged return the
// intermediate ciphertext right after stage k (1-based; see the markers below), so a probe
// can compare each stage limb-exactly against a host-replicated oracle stage.
int bootDebugStage() {
	const char* s = std::getenv("FIDESLIB_HAZE_BOOT_DEBUG_STAGE");
	return (s != nullptr) ? std::atoi(s) : 0;
}
} // namespace

Ciphertext<DCRTPoly> HazeEngine::bootstrapStaged(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// CUDA parity (#9): the Engine-dispatched bootstrap is ALWAYS the ModRaise-first (normal)
	// variant (FIDESlib::CKKS::Bootstrap, Bootstrap.cu:169) — the precom's BTSlotsEncoding flag
	// (forwarded from btsfirstboot, default false) only steers OpenFHE's host setup, not the
	// runtime variant. We select ModRaise-first for BTSlotsEncoding=false (CUDA-matching, the
	// path the parity oracle also runs) and keep the StC-first port for BTSlotsEncoding=true.
	auto pIn = ensureCt(ctx, ciphertext);
	requireComputable(*pIn, "EvalBootstrap");
	const uint32_t slots = static_cast<uint32_t>(pIn->slots);
	const auto bpIt		 = boot_.find(slots);
	if (bpIt == boot_.end()) {
		throw std::runtime_error("haze backend: no bootstrap precomputation for " + std::to_string(slots) + " slots (run EvalBootstrapSetup + EvalBootstrapKeyGen before LoadContext)");
	}
	if (bpIt->second.btSlotsEncoding) {
		return bootstrapStCFirst(ctx, ciphertext); // BTSlotsEncoding=true (CPU-oracle slim path)
	}
	return bootstrapModRaiseFirst(ctx, ciphertext); // BTSlotsEncoding=false (CUDA Engine path)
}

// Multi-stage FFT CoeffsToSlots / SlotsToCoeffs (levelBudget != {1,1}; #11). Plain-rotation port
// of OpenFHE EvalCoeffsToSlots / EvalSlotsToCoeffs (ckksrns-fhe.cpp:1885-2200) — mathematically the
// non-hoisted form of CUDA EvalCoeffsToSlots (CoeffsToSlots.cu:75-151), iterating the per-stage LT
// vectors. Hoisting is Task 06; here every baby/giant step is a plain EvalRotate.
Ciphertext<DCRTPoly> HazeEngine::evalCoeffsToSlotsFFT(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, uint32_t slots, bool decode) {
	const BootPrecom& bp = boot_.at(slots);
	const auto& p		 = decode ? bp.paramsDec : bp.paramsEnc;
	const auto& A		 = decode ? bp.u0PreFFT : bp.u0hatTPreFFT;
	const uint32_t M4	 = static_cast<uint32_t>(ringDim_) / 2; // cyclotomicOrder/4 = N/2

	auto reduceRot = [](int32_t index, uint32_t mod) { return static_cast<int32_t>(lbcrypto::ReduceRotation(index, mod)); };

	// Per-stage inner (baby, rot_in) and outer (giant, rot_out) rotation tables — ckksrns-fhe.cpp:1892.
	std::vector<std::vector<int32_t>> rot_in(p.lvlb), rot_out(p.lvlb);
	int32_t stop	= -1;
	int32_t flagRem = 0;
	if (p.remCollapse != 0) {
		stop	= 0;
		flagRem = 1;
	}
	for (int32_t s = 0; s < static_cast<int32_t>(p.lvlb); ++s) {
		rot_in[s].assign(p.g, 0);
		rot_out[s].assign(p.b, 0);
	}
	if (decode) {
		// EvalSlotsToCoeffs ordering (ckksrns-fhe.cpp:2042+): stage s ascending, scale 1<<(s*layersColl).
		int32_t offset = static_cast<int32_t>((p.numRotations + 1) / 2) - 1;
		for (int32_t s = 0; s < static_cast<int32_t>(p.lvlb) - flagRem; ++s) {
			int32_t scale = (1 << (s * static_cast<int32_t>(p.layersCollapse)));
			for (uint32_t i = 0; i < p.b; ++i)
				rot_out[s][i] = reduceRot(scale * static_cast<int32_t>(p.g) * static_cast<int32_t>(i), M4);
			for (uint32_t j = 0; j < p.g; ++j)
				rot_in[s][j] = reduceRot(scale * (static_cast<int32_t>(j) - offset), M4);
		}
		if (flagRem == 1) {
			int32_t s	  = static_cast<int32_t>(p.lvlb) - flagRem;
			int32_t scale = (1 << (s * static_cast<int32_t>(p.layersCollapse)));
			rot_out[s].assign(p.bRem, 0);
			rot_in[s].assign(p.gRem, 0);
			offset = static_cast<int32_t>((p.numRotationsRem + 1) / 2) - 1;
			for (uint32_t i = 0; i < p.bRem; ++i)
				rot_out[s][i] = reduceRot(scale * static_cast<int32_t>(p.gRem) * static_cast<int32_t>(i), M4);
			for (uint32_t j = 0; j < p.gRem; ++j)
				rot_in[s][j] = reduceRot(scale * (static_cast<int32_t>(j) - offset), M4);
		}
	} else {
		// EvalCoeffsToSlots ordering (ckksrns-fhe.cpp:1909): stage s descending from lvlb-1.
		int32_t offset = static_cast<int32_t>((p.numRotations + 1) / 2) - 1;
		for (int32_t s = static_cast<int32_t>(p.lvlb) - 1; s > stop; --s) {
			int32_t scale = (1 << ((s - flagRem) * static_cast<int32_t>(p.layersCollapse) + static_cast<int32_t>(p.remCollapse)));
			for (uint32_t i = 0; i < p.b; ++i)
				rot_out[s][i] = reduceRot(scale * static_cast<int32_t>(p.g) * static_cast<int32_t>(i), M4);
			for (uint32_t j = 0; j < p.g; ++j)
				rot_in[s][j] = reduceRot(scale * (static_cast<int32_t>(j) - offset), static_cast<uint32_t>(slots));
		}
		if (flagRem == 1) {
			rot_out[stop].assign(p.bRem, 0);
			rot_in[stop].assign(p.gRem, 0);
			offset = static_cast<int32_t>((p.numRotationsRem + 1) / 2) - 1;
			for (uint32_t i = 0; i < p.bRem; ++i)
				rot_out[stop][i] = reduceRot(static_cast<int32_t>(p.gRem) * static_cast<int32_t>(i), M4);
			for (uint32_t j = 0; j < p.gRem; ++j)
				rot_in[stop][j] = reduceRot(static_cast<int32_t>(j) - offset, static_cast<uint32_t>(slots));
		}
	}

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

	// Iterate the stages in evaluation order: CtS descends s = lvlb-1..stop+1 then remainder@stop;
	// StC ascends s = 0..lvlb-1. A[stage] is the per-stage plaintext block; OpenFHE's A index runs
	// 0..lvlb-1 in storage order, matching paramsEnc/Dec's s here (CUDA reverses CtS storage, but
	// it reads the SAME mathematical stage — we keep OpenFHE's storage order with OpenFHE's pts).
	std::vector<int32_t> stageOrder;
	if (decode) {
		for (int32_t s = 0; s < static_cast<int32_t>(p.lvlb); ++s)
			stageOrder.push_back(s);
	} else {
		for (int32_t s = static_cast<int32_t>(p.lvlb) - 1; s > stop; --s)
			stageOrder.push_back(s);
		if (flagRem == 1)
			stageOrder.push_back(stop);
	}

	bool first = true;
	for (int32_t s : stageOrder) {
		if (!first) {
			auto pr = devicePayload(result);
			if (pr->noiseScaleDeg > 1) {
				Operand r = rescaleCore(asOperand(pr));
				rebindPayload(*pr, std::move(r.p->c0), std::move(r.p->c1), r);
			}
		}
		first = false;

		const bool rem	 = (flagRem == 1) && ((decode && s == static_cast<int32_t>(p.lvlb) - 1) || (!decode && s == stop));
		const uint32_t g = rem ? p.gRem : p.g;
		const uint32_t b = rem ? p.bRem : p.b;
		const uint32_t numRot = rem ? p.numRotationsRem : p.numRotations;

		// Baby-step rotations (rot_in): plain EvalRotate (hoisted in Task 06).
		std::vector<Ciphertext<DCRTPoly>> fastRotation(g);
		for (uint32_t j = 0; j < g; ++j) {
			fastRotation[j] = (rot_in[s][j] != 0) ? ctx.EvalRotate(result, rot_in[s][j]) : std::make_shared<CiphertextImpl<DCRTPoly>>(*result);
		}

		Ciphertext<DCRTPoly> outer;
		for (uint32_t i = 0; i < b; ++i) {
			const uint32_t G = g * i;
			Ciphertext<DCRTPoly> inner = ctx.EvalMult(fastRotation[0], const_cast<Plaintext&>(A[s][G]));
			for (uint32_t j = 1; j < g; ++j) {
				if ((G + j) != numRot) {
					auto term = ctx.EvalMult(fastRotation[j], const_cast<Plaintext&>(A[s][G + j]));
					ctx.EvalAddInPlace(inner, term);
				}
			}
			if (i == 0) {
				outer = inner;
			} else {
				auto rotated = (rot_out[s][i] != 0) ? ctx.EvalRotate(inner, rot_out[s][i]) : inner;
				ctx.EvalAddInPlace(outer, rotated);
			}
		}
		result = outer;
	}
	return result;
}

Ciphertext<DCRTPoly> HazeEngine::bootstrapStCFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// Port of OpenFHE FHECKKSRNS::EvalBootstrapStCFirst (the path the CPU policy's
	// BTSlotsEncoding=true selects), single iteration, REAL data, sparse packing,
	// levelBudget {1,1}: deplete -> SlotsToCoeffs -> raise -> CoeffsToSlots -> EvalMod ->
	// scale-back. Like the oracle, an input that would not gain levels is returned as a
	// clone (the suite's full-level inputs take exactly that path on CPU too).
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDAUTO && st != lbcrypto::FLEXIBLEAUTO && st != lbcrypto::FLEXIBLEAUTOEXT) {
		notImplemented("EvalBootstrapStCFirst under FIXEDMANUAL");
	}

	auto pIn = ensureCt(ctx, ciphertext);
	requireComputable(*pIn, "EvalBootstrap");

	const uint32_t slots = static_cast<uint32_t>(pIn->slots);
	const uint32_t N	 = static_cast<uint32_t>(ringDim_);
	if (slots == N / 2) {
		notImplemented("fully-packed EvalBootstrapStCFirst (slots == N/2)");
	}
	const auto bpIt = boot_.find(slots);
	if (bpIt == boot_.end()) {
		throw std::runtime_error("haze backend: no bootstrap precomputation for " + std::to_string(slots) + " slots (run EvalBootstrapSetup + EvalBootstrapKeyGen before LoadContext)");
	}
	const BootPrecom& bp = bpIt->second;
	if (!bp.isLT) {
		notImplemented("EvalBootstrapStCFirst with levelBudget != {1,1}");
	}

	// Constants (ckksrns-fhe.cpp:940-954, 64-bit branch).
	const uint32_t L0	 = static_cast<uint32_t>(qBase_.size());
	const double qDouble = static_cast<double>(qBase_.front());
	const double powP	 = std::pow(2.0, static_cast<double>(plaintextModulus_));
	const int32_t deg	 = static_cast<int32_t>(std::round(std::log2(qDouble / powP)));
	const uint32_t correction = bp.correctionFactor - static_cast<uint32_t>(deg);
	const double post		  = std::pow(2.0, static_cast<double>(deg));
	const double pre		  = 1.0 / post;
	const uint64_t scalar	  = static_cast<uint64_t>(std::llround(post));

	const size_t initTowers = pIn->towers;

	// Early-out (oracle ckksrns-fhe.cpp:1222: output towers <= input towers -> return the
	// input clone; on the CPU oracle a full-level input takes exactly this path). The
	// pipeline consumes AT LEAST 3 mod-reduces + (numIter-1) double-angle rescales + 2
	// Chebyshev levels after the raise, so any input with at least L0 - that many towers
	// cannot gain levels — clone without recording the (discarded) program. Borderline
	// inputs below the bound still run the pipeline and hit the same comparison after it.
	const uint32_t minPipelineLevels = 3 + (bp.numIter - 1) + 2;
	if (initTowers + minPipelineLevels >= L0) {
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}

	// In-place helpers on a working facade ct.
	auto rescaleHard = [&](Ciphertext<DCRTPoly>& c) { // ModReduceInternalInPlace(1)
		auto p	  = devicePayload(c);
		Operand r = rescaleCore(asOperand(p));
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	};
	auto toDepthOne = [&](Ciphertext<DCRTPoly>& c) { // ModReduceInternalInPlace(NSD-1)
		while (devicePayload(c)->noiseScaleDeg > 1) {
			rescaleHard(c);
		}
	};
	auto multInt = [&](Ciphertext<DCRTPoly>& c, uint64_t s) {
		auto p	  = devicePayload(c);
		Operand r = multIntCore(asOperand(p), s);
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	};

	// ---- Dropping unnecessary towers (deplete; ckksrns-fhe.cpp:995-1033) ----
	Ciphertext<DCRTPoly> work = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	const size_t expectedLevel = L0 - (/*lvlbDec=*/1 + 2 + (st == lbcrypto::FLEXIBLEAUTOEXT ? 1 : 0));
	{
		auto p = devicePayload(work);
		if ((L0 - p->towers) + (p->noiseScaleDeg - 1) > expectedLevel) {
			throw std::runtime_error("haze backend: not enough levels to perform bootstrapping");
		}
		toDepthOne(work);
		p = devicePayload(work);
		const size_t ctxtLevel = L0 - p->towers;
		if (st == lbcrypto::FIXEDAUTO) {
			if (ctxtLevel < expectedLevel) {
				// LevelReduceInternalInPlace: pure truncation, no IR.
				const size_t targetTowers = L0 - expectedLevel;
				p->c0.truncate(targetTowers);
				p->c1.truncate(targetTowers);
				p->towers = targetTowers;
			}
		} else if (ctxtLevel < expectedLevel) {
			// FLEXIBLE deplete (oracle :1016-1029): scale to the expected level's factor,
			// level-reduce, mod-reduce, then pin the scaling factor.
			const double scf2 = p->scalingFactor;
			const double scf1 = sfRealBig_[ctxtLevel - 1 + (p->noiseScaleDeg - 1)];
			const double scf  = sfReal_[expectedLevel];
			ctx.EvalMultInPlace(work, scf1 / scf2 / scf);
			p = devicePayload(work);
			if ((L0 - p->towers) + (p->noiseScaleDeg - 1) < expectedLevel) {
				const size_t drop = expectedLevel - ctxtLevel - (p->noiseScaleDeg - 1);
				p->c0.truncate(p->towers - drop);
				p->c1.truncate(p->towers - drop);
				p->towers -= drop;
			}
			toDepthOne(work);
			devicePayload(work)->scalingFactor = scf;
		}
	}
	if (bootDebugStage() == 1) {
		return work;
	}

	// ---- SlotsToCoeffs first (LT path), then the sparse fold ----
	work = linearTransform(ctx, bp.u0Pre, work, bp.bStep);
	{
		auto rotated = ctx.EvalRotate(work, static_cast<int32_t>(slots));
		ctx.EvalAddInPlace(work, rotated);
	}
	if (bootDebugStage() == 2) {
		return work;
	}

	// ---- Raising the modulus ----
	toDepthOne(work); // ModReduceInternalInPlace(NSD-1)
	// AdjustCiphertext(work, 2^-correction, lvl) with the modReduce=true default, so the
	// raise is entered at depth 1. FIXED modes: public EvalMult by the correction scale.
	// FLEXIBLE modes (oracle ckksrns-fhe.cpp:2237-2260): scale to the target level's
	// factor folded with the correction, then pin the scaling factor.
	if (st == lbcrypto::FIXEDAUTO) {
		ctx.EvalMultInPlace(work, std::pow(2.0, -static_cast<double>(correction)));
		rescaleHard(work);
	} else {
		auto p				   = devicePayload(work);
		const uint32_t lvl	   = (st == lbcrypto::FLEXIBLEAUTOEXT) ? 1 : 0;
		const double targetSF  = sfReal_[lvl];
		const double sourceSF  = p->scalingFactor;
		const double modToDrop = modReduceFactor_[p->towers - 1]; // double(q_{towers-1}) under FLEXIBLE
		const double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * std::pow(2.0, -static_cast<double>(correction));
		ctx.EvalMultInPlace(work, adjustmentFactor);
		rescaleHard(work);
		devicePayload(work)->scalingFactor = targetSF;
	}
	{
		auto p					  = devicePayload(work);
		const size_t raiseTowers  = L0 - (st == lbcrypto::FLEXIBLEAUTOEXT ? 1 : 0);
		Operand r				  = modRaiseCore(asOperand(p), raiseTowers);
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	}
	if (bootDebugStage() == 3) {
		return work;
	}

	// ---- Scaling adjustment before CoeffsToSlots ----
	ctx.EvalMultInPlace(work, pre * (1.0 / (bp.k * static_cast<double>(N))));
	if (bootDebugStage() == 4) {
		return work;
	}

	// ---- Partial sum (sparse packing) ----
	const uint32_t limit = N / (2 * slots);
	for (uint32_t j = 1; j < limit; j <<= 1) {
		auto rotated = ctx.EvalRotate(work, static_cast<int32_t>(j * slots));
		ctx.EvalAddInPlace(work, rotated);
	}
	rescaleHard(work); // ModReduceInternalInPlace(1) before CoeffsToSlots
	if (bootDebugStage() == 5) {
		return work;
	}

	// ---- CoeffsToSlots (LT path) ----
	auto ctxtEnc = linearTransform(ctx, bp.u0hatTPre, work, bp.bStep);

	// ctxtEnc += Conjugate(ctxtEnc): keyswitch with the conjugation key (2N-1).
	{
		auto pEnc	 = devicePayload(ctxtEnc);
		Operand conj = rotateByAutoIndex(ctx, asOperand(pEnc), 2 * N - 1);
		auto conjCt	 = wrapDeviceResult(ctx, ctxtEnc, std::move(conj.p));
		ctx.EvalAddInPlace(ctxtEnc, conjCt);
	}
	if (devicePayload(ctxtEnc)->noiseScaleDeg == 2) {
		rescaleHard(ctxtEnc);
	}
	if (bootDebugStage() == 6) {
		return ctxtEnc;
	}

	// ---- Approximate modular reduction: Chebyshev + double-angle ----
	std::vector<double> coeffs = bp.coefficients;
	ctx.EvalChebyshevSeriesInPlace(ctxtEnc, coeffs, -1.0, 1.0);
	toDepthOne(ctxtEnc); // ModReduceInternalInPlace(NSD-1)
	if (bootDebugStage() == 7) {
		return ctxtEnc;
	}
	{
		constexpr double twoPi = 2.0 * M_PI;
		for (int32_t i = 1 - static_cast<int32_t>(bp.numIter); i <= 0; ++i) {
			const double angleScalar = -std::pow(twoPi, -std::pow(2.0, static_cast<double>(i)));
			ctx.EvalSquareInPlace(ctxtEnc);
			auto shifted = ctx.EvalAdd(ctxtEnc, angleScalar);
			ctx.EvalAddInPlace(ctxtEnc, shifted);
			ctx.RescaleInPlace(ctxtEnc); // public ModReduce: FIXEDAUTO no-op (oracle-exact)
		}
	}
	if (bootDebugStage() == 8) {
		return ctxtEnc;
	}

	// ---- Scale the message back up; 64-bit correction ----
	multInt(ctxtEnc, scalar);
	multInt(ctxtEnc, static_cast<uint64_t>(1) << correction);

	// Oracle line 1222: if bootstrapping did not gain levels, return the input clone (the
	// recorded pipeline becomes a discarded program-local — never tagged, never read).
	if (devicePayload(ctxtEnc)->towers <= initTowers) {
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}
	return ctxtEnc;
}

Ciphertext<DCRTPoly> HazeEngine::bootstrapModRaiseFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// CUDA-matching ModRaise-first (normal) bootstrap: FIDESlib::CKKS::Bootstrap (Bootstrap.cu:169),
	// the variant CudaEngine::evalBootstrap always runs. Cross-ref OpenFHE FHECKKSRNS::EvalBootstrap
	// (ckksrns-fhe.cpp:430, BTSlotsEncoding=false branch): ModReduce(NSD-1) -> AdjustCiphertext ->
	// ModRaise -> scale by pre/(k*N) -> [sparse partial sum] -> ModReduce -> CtS -> +Conjugate ->
	// approxMod (Chebyshev + double-angle + scalar) -> StC -> [sparse rotate+add] -> *2^correction.
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool fixedManual = (st == lbcrypto::FIXEDMANUAL);

	auto pIn = ensureCt(ctx, ciphertext);
	requireComputable(*pIn, "EvalBootstrap");

	const uint32_t slots = static_cast<uint32_t>(pIn->slots);
	const uint32_t N	 = static_cast<uint32_t>(ringDim_);
	if (slots == N / 2) {
		// Fully-packed COMPLEX path (Bootstrap.cu:268-278, ApproxModEval.cu:24-92) needs
		// MultByMonomial (ctxtEncI = (ct-conj) * X^{3N/2}). haze CAN express this via
		// hazeRotAutomorphCoeffMrp (mult by X^{-offset} in coeff form, with the X^N=-1 sign flip),
		// so it is not a hard primitive gap — but the path is untestable in this environment
		// (slots==N/2==32768 OOMs the in-process simulator, multi-GB), so it is deliberately
		// deferred rather than shipped unvalidated. The CUDA-correct sequence is the COMPLEX
		// branch of approxModReduction.
		notImplemented("fully-packed ModRaise-first EvalBootstrap (slots == N/2): COMPLEX path deferred (untestable in-process)");
	}
	const BootPrecom& bp = boot_.at(slots);

	// Constants (Bootstrap.cu:182-210; OpenFHE ckksrns-fhe.cpp:533-547, 64-bit branch).
	const uint32_t L0	 = static_cast<uint32_t>(qBase_.size());
	const double qDouble = static_cast<double>(qBase_.front());
	const double powP	 = std::pow(2.0, static_cast<double>(plaintextModulus_));
	const int32_t deg	 = static_cast<int32_t>(std::round(std::log2(qDouble / powP)));
	// CUDA leaves the deg>correctionFactor throw commented out (Bootstrap.cu:53-58); match it (#11).
	const uint32_t correction = bp.correctionFactor - static_cast<uint32_t>(deg);
	const double post		  = std::pow(2.0, static_cast<double>(deg));
	const double pre		  = 1.0 / post;
	const uint64_t scalar	  = static_cast<uint64_t>(std::llround(post));

	const size_t initTowers = pIn->towers;

	auto rescaleHard = [&](Ciphertext<DCRTPoly>& c) {
		auto p	  = devicePayload(c);
		Operand r = rescaleCore(asOperand(p));
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	};
	auto toDepthOne = [&](Ciphertext<DCRTPoly>& c) {
		while (devicePayload(c)->noiseScaleDeg > 1)
			rescaleHard(c);
	};
	auto multInt = [&](Ciphertext<DCRTPoly>& c, uint64_t s) {
		auto p	  = devicePayload(c);
		Operand r = multIntCore(asOperand(p), s);
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	};

	Ciphertext<DCRTPoly> raised = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);

	// ---- ModReduce(NSD-1), then AdjustCiphertext(2^-correction, lvl) (Bootstrap.cu:510-572) ----
	toDepthOne(raised);
	if (st == lbcrypto::FIXEDAUTO || fixedManual) {
		ctx.EvalMultInPlace(raised, std::pow(2.0, -static_cast<double>(correction)));
		// FIXEDMANUAL rescale gating (#11; Bootstrap.cu:122-125,274-277): under FIXEDMANUAL the
		// rescale is the explicit ModReduce after the public mult; FIXEDAUTO rescales eagerly too
		// (parity #3 — haze rescale is unconditional, unlike OpenFHE's technique gate).
		rescaleHard(raised);
	} else { // FLEXIBLE* (Bootstrap.cu:428-509 / ckksrns AdjustCiphertext)
		auto p				   = devicePayload(raised);
		const uint32_t lvl	   = (st == lbcrypto::FLEXIBLEAUTOEXT) ? 1 : 0;
		const double targetSF  = sfReal_[lvl];
		const double sourceSF  = p->scalingFactor;
		const double modToDrop = modReduceFactor_[p->towers - 1];
		const double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * std::pow(2.0, -static_cast<double>(correction));
		ctx.EvalMultInPlace(raised, adjustmentFactor);
		rescaleHard(raised);
		devicePayload(raised)->scalingFactor = targetSF;
	}

	// ---- RAISE THE MODULUS to Q (Bootstrap.cu:380-710 ModRaise; OpenFHE :592-602) ----
	{
		auto p					 = devicePayload(raised);
		const size_t raiseTowers = L0 - (st == lbcrypto::FLEXIBLEAUTOEXT ? 1 : 0);
		Operand r				 = modRaiseCore(asOperand(p), raiseTowers);
		rebindPayload(*p, std::move(r.p->c0), std::move(r.p->c1), r);
	}
	if (bootDebugStage() == 1) {
		return raised;
	}

	// ---- Scale before CoeffsToSlots (Bootstrap.cu:225,235; OpenFHE :639) ----
	ctx.EvalMultInPlace(raised, pre * (1.0 / (bp.k * static_cast<double>(N))));

	// ---- Sparse partial sum (Bootstrap.cu sparse Accumulate; OpenFHE :744-746) ----
	const uint32_t limit = N / (2 * slots);
	for (uint32_t j = 1; j < limit; j <<= 1) {
		auto rotated = ctx.EvalRotate(raised, static_cast<int32_t>(j * slots));
		ctx.EvalAddInPlace(raised, rotated);
	}
	// ModReduceInternal before CoeffsToSlots (OpenFHE :756). Unconditional (eager) — CUDA rescale.
	rescaleHard(raised);
	if (bootDebugStage() == 2) {
		return raised;
	}

	// ---- CoeffsToSlots (Bootstrap.cu:261-265; isLT -> EvalLinearTransform, else FFT) ----
	Ciphertext<DCRTPoly> ctxtEnc = bp.isLT ? linearTransform(ctx, bp.u0hatTPre, raised, bp.bStep) : evalCoeffsToSlotsFFT(ctx, raised, slots, /*decode=*/false);

	// ctxtEnc += Conjugate(ctxtEnc) (Bootstrap.cu:280-281; OpenFHE :762).
	{
		auto pEnc	 = devicePayload(ctxtEnc);
		Operand conj = rotateByAutoIndex(ctx, asOperand(pEnc), 2 * N - 1);
		auto conjCt	 = wrapDeviceResult(ctx, ctxtEnc, std::move(conj.p));
		ctx.EvalAddInPlace(ctxtEnc, conjCt);
	}
	// FIXEDMANUAL: drain to depth 1; else rescale only if NSD==2 (Bootstrap.cu:282-283 / OpenFHE :764-773).
	if (fixedManual) {
		toDepthOne(ctxtEnc);
	} else if (devicePayload(ctxtEnc)->noiseScaleDeg == 2) {
		rescaleHard(ctxtEnc);
	}
	if (bootDebugStage() == 3) {
		return ctxtEnc;
	}

	// ---- Approximate modular reduction (sparse; ApproxModEval.cu:94-129) ----
	std::vector<double> coeffs = bp.coefficients;
	ctx.EvalChebyshevSeriesInPlace(ctxtEnc, coeffs, -1.0, 1.0);
	if (!fixedManual) {
		toDepthOne(ctxtEnc); // ModReduceInternal after Chebyshev (OpenFHE :790-791)
	}
	{
		constexpr double twoPi = 2.0 * M_PI;
		for (int32_t i = 1 - static_cast<int32_t>(bp.numIter); i <= 0; ++i) {
			const double angleScalar = -std::pow(twoPi, -std::pow(2.0, static_cast<double>(i)));
			ctx.EvalSquareInPlace(ctxtEnc);
			auto shifted = ctx.EvalAdd(ctxtEnc, angleScalar);
			ctx.EvalAddInPlace(ctxtEnc, shifted);
			ctx.RescaleInPlace(ctxtEnc); // eager (parity #3); FIXEDAUTO no-op on OpenFHE, eager here
		}
	}
	multInt(ctxtEnc, scalar); // scale message back up after Chebyshev (Bootstrap.cu:119 / OpenFHE :798)
	if (bootDebugStage() == 4) {
		return ctxtEnc;
	}

	// ModReduceInternal before SlotsToCoeffs (OpenFHE :814-815). Skipped under FIXEDMANUAL.
	if (!fixedManual) {
		toDepthOne(ctxtEnc);
	}

	// ---- SlotsToCoeffs (Bootstrap.cu:293-296; isLT -> EvalLinearTransform, else FFT) ----
	Ciphertext<DCRTPoly> ctxtDec = bp.isLT ? linearTransform(ctx, bp.u0Pre, ctxtEnc, bp.bStep) : evalCoeffsToSlotsFFT(ctx, ctxtEnc, slots, /*decode=*/true);

	// Sparse fold (Bootstrap.cu:299-301; OpenFHE :819).
	{
		auto rotated = ctx.EvalRotate(ctxtDec, static_cast<int32_t>(slots));
		ctx.EvalAddInPlace(ctxtDec, rotated);
	}
	if (bootDebugStage() == 5) {
		return ctxtDec;
	}

	// ---- 64-bit correction scale-back (Bootstrap.cu:304-305; OpenFHE :824-825) ----
	multInt(ctxtDec, static_cast<uint64_t>(1) << correction);

	// If bootstrapping did not gain levels, return the input clone (OpenFHE :835).
	if (devicePayload(ctxtDec)->towers <= initTowers) {
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}
	return ctxtDec;
}

Ciphertext<DCRTPoly>
HazeEngine::evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t /*numIterations*/, uint32_t /*precision*/, bool /*prescaled*/) {
	return bootstrapStaged(ctx, ciphertext);
}

void HazeEngine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	ciphertext = evalBootstrap(ctx, ciphertext, numIterations, precision, prescaled);
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
