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
#include <optional>
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
	// the parameterized target). Device-side: INTT@{q0} -> hazeBasisConvertCentered ({q0} -> Q',
	// centered because that constructor's SwitchModulus always is) -> NTT@Q' on both components.
	// Metadata (NSD/sf) carries through.
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
		hazeCheck(hazeBasisConvertCentered(conv.data(), intt.asConst().data(), &convParams, nullptr), "hazeBasisConvertCentered");
		LimbChain ntt(fullTowers, polyBytes_);
		hazeCheck(hazeNTTMrp(ntt.data(), conv.asConst().data(), raisedBase.data(), raisedBase.size(), nullptr), "hazeNTTMrp");
		return ntt;
	};

	LimbChain out0 = raiseChain(x.p->c0());
	LimbChain out1 = raiseChain(x.p->c1());

	Operand res = x;
	res.towers	= fullTowers;
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
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
	hazeCheck(hazeMulScalarMrp(out0.data(), x.p->c0().asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	hazeCheck(hazeMulScalarMrp(out1.data(), x.p->c1().asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");

	Operand res = x;
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
	return res;
}

HazeEngine::Operand HazeEngine::multByMonomialCore(const Operand& x, uint32_t power) {
	// OpenFHE MultByMonomialInPlace / CUDA Ciphertext::multMonomial (Ciphertext.cpp:1623):
	// multiply both components by the literal monomial X^power in R_q = Z_q[X]/(X^N+1).
	//
	// We use the dedicated coefficient-domain primitive hazeRotAutomorphCoeffMrp(offset), which is
	// multiplication by X^{-offset} (negacyclic LEFT shift, offset in [0,N), X^N = -1 sign flip on
	// wraparound — haze.h:281). This records a single IR op on the EXISTING ciphertext and creates
	// no synthetic monomial input (a host-built plaintext-monomial would need a replay-bridge input
	// template, which fails to synthesize for a literal monomial — its bytes are not a valid CKKS
	// ciphertext shape). Map X^power onto X^{-offset}:
	//   r = power mod 2N.
	//   r == 0          -> X^1 == identity ... handled as r in [0,N) below (offset 0, no negate).
	//   r in [N, 2N)    -> X^r = X^{-(2N-r)};  offset = 2N - r in (0,N], no extra sign.
	//   r in [0, N)     -> X^r = -X^{-(N-r)};  offset = N - r in (0,N], with a per-limb negate
	//                      (because X^{-(N-r)} = X^{(2N-(N-r))} ... = X^{N+r} = X^N*X^r = -X^r).
	// offset == N collapses to offset 0 with the sign already accounted for (X^{-N} = -1).
	// For our two call sites: power = 3N/2 -> r=3N/2 -> offset=N/2, no negate; power = N/2 ->
	// r=N/2 -> offset=N/2, negate. (Consistent: X^{3N/2} = -X^{N/2}.)
	const uint32_t N	= static_cast<uint32_t>(ringDim_);
	const uint32_t mTwo = 2 * N;
	const uint32_t r	= power % mTwo; // X^{2N} = 1

	uint32_t offset = 0;
	bool negate		= false;
	if (r >= N) {
		offset = mTwo - r; // in (0, N]
	} else {
		offset = N - r; // in (0, N]
		negate = true;
	}
	if (offset == N) { // X^{-N} = -1: fold to offset 0 and flip the accumulated sign
		offset = 0;
		negate = !negate;
	}

	const auto base = qPrefix(x.towers);

	std::vector<uint64_t> negScalars(x.towers);
	for (size_t t = 0; t < x.towers; ++t) {
		negScalars[t] = qBase_[t] - 1; // ≡ -1 mod q_t
	}

	// hazeRotAutomorphCoeffMrp is a COEFFICIENT-form negacyclic shift (haze.h:281), but ciphertext
	// limbs travel the bootstrap pipeline in EVALUATION form. Bracket the monomial multiply in
	// INTT (eval -> coeff) ... shift [... optional -1] ... NTT (coeff -> eval), mirroring CUDA
	// Ciphertext::multMonomial which builds the monomial in coeff form, NTTs it, then multiplies
	// pointwise in eval form (Ciphertext.cpp:1623-1685). Skipping the INTT/NTT applies the shift to
	// eval-domain samples, which is NOT a monomial multiply and corrupts the ciphertext.
	auto shiftChain = [&](const LimbChain& src) {
		LimbChain coeff(x.towers, polyBytes_);
		hazeCheck(hazeINTTMrp(coeff.data(), src.asConst().data(), base.data(), base.size(), nullptr), "hazeINTTMrp(monomial)");
		LimbChain shifted(x.towers, polyBytes_);
		hazeCheck(hazeRotAutomorphCoeffMrp(shifted.data(), coeff.asConst().data(), offset, base.data(), base.size(), nullptr), "hazeRotAutomorphCoeffMrp(monomial)");
		if (negate) {
			LimbChain neg(x.towers, polyBytes_);
			hazeCheck(hazeMulScalarMrp(neg.data(), shifted.asConst().data(), negScalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp(monomial negate)");
			shifted = std::move(neg);
		}
		LimbChain ev(x.towers, polyBytes_);
		hazeCheck(hazeNTTMrp(ev.data(), shifted.asConst().data(), base.data(), base.size(), nullptr), "hazeNTTMrp(monomial)");
		return ev;
	};

	LimbChain out0 = shiftChain(x.p->c0());
	LimbChain out1 = shiftChain(x.p->c1());

	Operand res = x; // metadata unchanged (literal monomial: NSD/sf/level preserved)
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
	return res;
}

// ---- Extended-basis (Q∥P) BSGS for the bootstrap linear transforms ----
//
// Port of OpenFHE's hoisted Horner linear-transform inner loop (EvalCoeffsToSlots /
// EvalSlotsToCoeffs / EvalLinearTransform, ckksrns-fhe.cpp:1918-2186, post-#1209 single-giant-
// step form): every baby-step rotation is an EvalFastRotationExt — the per-key dot product
// against the ONE shared ModUp handle, kept in the extended Q∥P basis with NO per-baby ModDown —
// the plaintext multiply and the cross-baby-step accumulation happen in Q∥P (the boot precompute
// plaintexts are encoded in the extended P·Q basis for exactly this:
// EvalCoeffsToSlotsPrecompute, "extended basis P*Q"). Giant blocks are consumed from the TOP
// down, and between blocks the ACCUMULATOR is rotated by the single constant stride
// (EvalHornerGiantRotate: KeySwitchDown of both components, ModUp of the a part, ext keyswitch
// with the one giant key, addFirst c0·PModq fold, automorph both), plus one final KeySwitchDown.
// Only baby keys 1..g-1 and the ONE giant-stride key are required — the post-#1209 reduced
// rotation-key footprint EvalBootstrapKeyGen now generates.

std::vector<uint64_t> HazeEngine::pModqPrefix(size_t towers) const {
	// OpenFHE cryptoParams->GetPModq() (P = prod pBase_), per surviving Q prime.
	std::vector<uint64_t> out(towers);
	for (size_t i = 0; i < towers; ++i) {
		const lbcrypto::NativeInteger q(qBase_[i]);
		lbcrypto::NativeInteger prod(1);
		for (const uint64_t p : pBase_) {
			prod = prod.ModMul(lbcrypto::NativeInteger(p), q);
		}
		out[i] = prod.ConvertToInt();
	}
	return out;
}

hazebk::LimbChain HazeEngine::extModDownChain(const hazebk::LimbChain& ext, size_t towers) {
	// KeySwitchDown of one component (keyswitch-hybrid.cpp:246 ApproxModDown): INTT ONLY the |P|
	// tail rows, then evalModDown into the Q survivors — the identical chain (and centered-FBC
	// lift under ReducedNoise) as hybridKeyswitchFromDigits' ext=false tail.
	const size_t numP = pBase_.size();
	if (ext.size() != towers + numP) {
		throw std::runtime_error("haze backend: extModDownChain got a non-extended chain");
	}
	LimbChain pCoeff(numP, polyBytes_);
	std::vector<const void*> pSrc(numP);
	for (size_t j = 0; j < numP; ++j) {
		pSrc[j] = ext[towers + j];
	}
	hazeCheck(hazeINTTMrp(pCoeff.data(), pSrc.data(), pBase_.data(), numP, nullptr), "hazeINTTMrp(extModDown)");
	std::vector<const void*> surv(towers);
	for (size_t i = 0; i < towers; ++i) {
		surv[i] = ext[i];
	}
	return evalModDown(surv, pCoeff, pBase_, qPrefix(towers), ModDownLift::Configured);
}

std::optional<HazeEngine::Operand> HazeEngine::extBsgsStage(CryptoContextImpl<DCRTPoly>& ctx,
  const Operand& x,
  const std::vector<Plaintext>& pts,
  const std::vector<int32_t>& rotIn,
  int32_t giantStride,
  uint32_t numBlocks,
  uint32_t numRotSkip) {
	// Kill switch for A/B and bisection: force the plain-rotation (per-baby ModDown) path.
	// ("cts"/"stc" are DIRECTIONAL switches handled at the evalCoeffsToSlotsFFT call site.)
	if (const char* noExt = std::getenv("FIDESLIB_HAZE_BOOT_NO_EXT"); noExt != nullptr) {
		const std::string v(noExt);
		if (v != "cts" && v != "stc") {
			return std::nullopt;
		}
	}
	if (x.noiseScaleDeg != 1 || pts.empty() || rotIn.empty() || numBlocks == 0 || x.p->components.size() != 2) {
		return std::nullopt; // stage drivers rescale to depth 1 first; anything else -> plain path
	}
	const size_t towers	 = x.towers;
	const size_t numP	 = pBase_.size();
	const auto st		 = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool fixedMode = (st == lbcrypto::FIXEDAUTO || st == lbcrypto::FIXEDMANUAL);

	// Ext preconditions, probed on the first plaintext (a stage block shares one encoding): the
	// precompute must be depth-1 and in the extended Q∥P basis. Trimming the pt's Q part to the
	// ct's towers is exact only under the FIXED scaling rules (multPtCore's fixedTrimOk); an
	// exact-level pt is fine under every technique.
	auto pt0 = ensurePt(ctx, const_cast<Plaintext&>(pts.front()));
	if (pt0->noiseScaleDeg != 1 || pt0->towers < towers + numP) {
		return std::nullopt; // not extended-encoded (or too shallow): plain path
	}
	const size_t ptQ = pt0->towers - numP;
	if (!(ptQ == towers || (fixedMode && ptQ > towers))) {
		return std::nullopt;
	}

	const uint32_t g = static_cast<uint32_t>(rotIn.size());
	const uint32_t b = numBlocks;
	const auto qSub	 = qPrefix(towers);
	std::vector<uint64_t> qpBase = qSub;
	qpBase.insert(qpBase.end(), pBase_.begin(), pBase_.end());
	const size_t qpTowers = qpBase.size();

	// Row view of an extended-encoded pt over (Q-prefix ∥ P): its first `towers` Q rows plus its
	// |P| tail rows (per-prime CRT residues of one integer vector — the trim is row selection).
	auto ptRowsQP = [&](const hazebk::HazePtPayload& pt) {
		if (pt.noiseScaleDeg != 1 || pt.towers < towers + numP || (pt.towers - numP) != ptQ) {
			throw std::runtime_error("haze backend: bootstrap LT plaintext block is not uniformly extended-encoded");
		}
		std::vector<const void*> rows;
		rows.reserve(qpTowers);
		for (size_t t = 0; t < towers; ++t) {
			rows.push_back(pt.chain[t]);
		}
		for (size_t t = 0; t < numP; ++t) {
			rows.push_back(pt.chain[ptQ + t]);
		}
		return rows;
	};

	auto mulRows = [&](const std::vector<const void*>& xr, const std::vector<const void*>& yr, const std::vector<uint64_t>& base) {
		LimbChain out(base.size(), polyBytes_);
		hazeCheck(hazeMulMrp(out.data(), xr.data(), yr.data(), base.data(), base.size(), nullptr), "hazeMulMrp(extLT)");
		return out;
	};
	auto addInPlace = [&](LimbChain& dst, const LimbChain& src, const std::vector<uint64_t>& base) {
		hazeCheck(hazeAddMrp(dst.data(), dst.asConst().data(), src.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp(extLT)");
	};
	// In-place add of a base-Q chain into the leading Q rows of an extended chain (a term whose
	// P rows are identically zero contributes nothing there — no zero rows are materialized).
	auto addQIntoExt = [&](LimbChain& ext, const LimbChain& srcQ) {
		std::vector<void*> dst(towers);
		std::vector<const void*> s1(towers);
		for (size_t t = 0; t < towers; ++t) {
			dst[t] = ext[t];
			s1[t]  = ext[t];
		}
		hazeCheck(hazeAddMrp(dst.data(), s1.data(), srcQ.asConst().data(), qSub.data(), towers, nullptr), "hazeAddMrp(extLT Q)");
	};
	auto automorphChain = [&](const LimbChain& src, uint32_t autoIndex, const std::vector<uint64_t>& base) {
		LimbChain out(base.size(), polyBytes_);
		hazeCheck(hazeAutomorphMrp(out.data(), src.asConst().data(), autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp(extLT)");
		return out;
	};
	// Same key + automorphism-index resolution as rotateCore/fastRotateFromDigits: the
	// pre-extracted slots-aware key when registered, else the lazy FindAutomorphismIndex cache.
	auto keyFor = [&](int32_t step) -> std::pair<KsKey*, uint32_t> {
		const RotKeyResolved rk = resolveRotationKey(step, x.slots);
		if (rk.key != nullptr) {
			return { rk.key, rk.autoIndex };
		}
		auto& context			 = hostContext(ctx);
		const uint32_t autoIndex = context->FindAutomorphismIndex(static_cast<uint32_t>(step));
		return { &autoKeyFor(ctx, autoIndex), autoIndex };
	};

	// KeySwitchExt raise (keyswitch-hybrid.cpp:217): both components × (P mod q_i) on the Q rows.
	// P·c ≡ 0 mod every special prime, so the P rows of a raised (identity) term are identically
	// zero and are never materialized. raised0 doubles as the hoisted addFirst fold source
	// (OpenFHE recomputes c0·PModq per baby rotation; the values are identical).
	const auto pmodq = pModqPrefix(towers);
	LimbChain raised0(towers, polyBytes_);
	LimbChain raised1(towers, polyBytes_);
	hazeCheck(hazeMulScalarMrp(raised0.data(), x.p->c0().asConst().data(), pmodq.data(), qSub.data(), towers, nullptr), "hazeMulScalarMrp(extLT raise)");
	hazeCheck(hazeMulScalarMrp(raised1.data(), x.p->c1().asConst().data(), pmodq.data(), qSub.data(), towers, nullptr), "hazeMulScalarMrp(extLT raise)");

	// ONE shared ModUp of c1 per stage (EvalFastRotationPrecompute; ckksrns-fhe.cpp:1941).
	const HoistedDigits digits = hoistedKeyswitchPrefix(x.p->c1(), towers, numPartQ_);

	// Baby-step rotations (EvalFastRotationExt(result, rot_in, digits, addFirst=true)): per-key
	// dot product in Q∥P, addFirst c0·P fold on the Q rows, automorph BOTH extended components.
	// NO ModDown here — that is the point of the extended-basis form.
	struct ExtRot {
		LimbChain b, a; // qpTowers rows each; empty for the identity index
	};
	std::vector<ExtRot> fastRot(g);
	std::vector<char> isIdentity(g, 0);
	for (uint32_t j = 0; j < g; ++j) {
		if (rotIn[j] == 0) {
			isIdentity[j] = 1; // KeySwitchExt(result, true): Q rows = raised0/raised1, P rows = 0
			continue;
		}
		const auto [key, autoIndex] = keyFor(rotIn[j]);
		KsContribution ks			= hybridKeyswitchFromDigits(digits, *key, /*ext=*/true);
		addQIntoExt(ks.b, raised0); // addFirst: cTilda[0] += c0·PModq (Q rows; P rows of c0·P are 0)
		fastRot[j].b = automorphChain(ks.b, autoIndex, qpBase);
		fastRot[j].a = automorphChain(ks.a, autoIndex, qpBase);
	}

	// Giant-block builder: Ext-accumulate the g baby products of block i (Q∥P rows).
	// Keyswitched terms first, identity terms (Q-rows-only) last: modular adds are exact and
	// commutative, so this reordering of the oracle's j-ascending sum is value-identical.
	// The TOP block mirrors the oracle's `(Gtop + j) != numRotations` guard via numRotSkip
	// (j = 0 is always included, exactly like the oracle's unconditional A[Gtop] term).
	auto buildBlock = [&](uint32_t i, bool topBlock) -> std::optional<std::pair<LimbChain, LimbChain>> {
		const size_t G = static_cast<size_t>(g) * i;
		auto include   = [&](uint32_t j) {
			const size_t k = G + j;
			if (k >= pts.size()) {
				return false;
			}
			return !(topBlock && j != 0 && k == static_cast<size_t>(numRotSkip));
		};
		LimbChain innB, innA;
		bool started = false;
		for (uint32_t j = 0; j < g; ++j) {
			if (isIdentity[j] != 0 || !include(j)) {
				continue;
			}
			const size_t k = G + j;
			auto ptp	   = ensurePt(ctx, const_cast<Plaintext&>(pts[k]));
			const auto pr  = ptRowsQP(*ptp);
			if (!started) {
				innB	= mulRows(fastRot[j].b.asConst(), pr, qpBase);
				innA	= mulRows(fastRot[j].a.asConst(), pr, qpBase);
				started = true;
			} else {
				addInPlace(innB, mulRows(fastRot[j].b.asConst(), pr, qpBase), qpBase);
				addInPlace(innA, mulRows(fastRot[j].a.asConst(), pr, qpBase), qpBase);
			}
		}
		for (uint32_t j = 0; j < g; ++j) {
			if (isIdentity[j] == 0 || !include(j)) {
				continue;
			}
			const size_t k = G + j;
			auto ptp	   = ensurePt(ctx, const_cast<Plaintext&>(pts[k]));
			const auto pr  = ptRowsQP(*ptp);
			if (started) {
				const std::vector<const void*> prQ(pr.begin(), pr.begin() + towers);
				addQIntoExt(innB, mulRows(raised0.asConst(), prQ, qSub));
				addQIntoExt(innA, mulRows(raised1.asConst(), prQ, qSub));
			} else {
				// Degenerate all-identity block (never produced by well-formed BSGS shapes):
				// materialize the zero P rows the raised term formally carries, then multiply
				// raised∥0 by the full extended pt rows.
				const std::vector<uint64_t> zeros(numP, 0);
				const std::vector<const void*> anyRows(numP, x.p->c0()[0]);
				LimbChain zb(numP, polyBytes_);
				LimbChain za(numP, polyBytes_);
				hazeCheck(hazeMulScalarMrp(zb.data(), anyRows.data(), zeros.data(), pBase_.data(), numP, nullptr), "hazeMulScalarMrp(extLT zeroP)");
				hazeCheck(hazeMulScalarMrp(za.data(), anyRows.data(), zeros.data(), pBase_.data(), numP, nullptr), "hazeMulScalarMrp(extLT zeroP)");
				std::vector<const void*> xb = raised0.asConst();
				std::vector<const void*> xa = raised1.asConst();
				const auto zbRows			= zb.asConst();
				const auto zaRows			= za.asConst();
				xb.insert(xb.end(), zbRows.begin(), zbRows.end());
				xa.insert(xa.end(), zaRows.begin(), zaRows.end());
				innB	= mulRows(xb, pr, qpBase);
				innA	= mulRows(xa, pr, qpBase);
				started = true;
			}
		}
		if (!started) {
			return std::nullopt; // empty giant block (cannot happen for a well-formed BSGS split)
		}
		return std::make_pair(std::move(innB), std::move(innA));
	};

	// The one giant-stride key, resolved once per stage (oracle GetGiantStepRotation).
	KsKey* gKey	   = nullptr;
	uint32_t gAuto = 0;
	if (giantStride != 0 && b > 1) {
		const auto resolved = keyFor(giantStride);
		gKey				= resolved.first;
		gAuto				= resolved.second;
	}

	// EvalHornerGiantRotate: outer = KeySwitchDown(outer) on both components, then ONE ModUp of
	// the a part, ext keyswitch with the giant key, addFirst fold of the moddowned b part
	// (cTilda[0] += b·PModq on the Q rows; P rows of b·P are 0), automorph both Ext components.
	auto hornerRotate = [&](LimbChain& outB, LimbChain& outA) {
		LimbChain bQ		   = extModDownChain(outB, towers);
		LimbChain aQ		   = extModDownChain(outA, towers);
		const HoistedDigits gd = hoistedKeyswitchPrefix(aQ, towers, numPartQ_);
		KsContribution ks	   = hybridKeyswitchFromDigits(gd, *gKey, /*ext=*/true);
		LimbChain bRaised(towers, polyBytes_);
		hazeCheck(hazeMulScalarMrp(bRaised.data(), bQ.asConst().data(), pmodq.data(), qSub.data(), towers, nullptr), "hazeMulScalarMrp(extLT horner)");
		addQIntoExt(ks.b, bRaised);
		outB = automorphChain(ks.b, gAuto, qpBase);
		outA = automorphChain(ks.a, gAuto, qpBase);
	};

	// Horner backward accumulation (ckksrns-fhe.cpp:2011-2026): start from the TOP giant block,
	// rotate the ACCUMULATOR by the constant stride between blocks, Ext-add each lower block;
	// ONE final KeySwitchDown of both components.
	auto topBlock = buildBlock(b - 1, /*topBlock=*/true);
	if (!topBlock) {
		return std::nullopt;
	}
	LimbChain outerB = std::move(topBlock->first);
	LimbChain outerA = std::move(topBlock->second);
	for (int32_t i = static_cast<int32_t>(b) - 2; i >= 0; --i) {
		if (gKey != nullptr) {
			hornerRotate(outerB, outerA);
		}
		if (auto blk = buildBlock(static_cast<uint32_t>(i), /*topBlock=*/false)) {
			addInPlace(outerB, blk->first, qpBase);
			addInPlace(outerA, blk->second, qpBase);
		}
	}

	// result = KeySwitchDown(outer) (ckksrns-fhe.cpp:2026).
	LimbChain resB = extModDownChain(outerB, towers);
	LimbChain resA = extModDownChain(outerA, towers);

	// EvalMultExt metadata (NSD += pt.NSD, sf ×= pt.sf) — the ext dance itself is metadata-
	// neutral, matching the plain path's multPtCore(fixedTrimOk): slots stays x.slots. pt0 is a
	// bootstrap linear-transform plaintext encoded at full ring width, so its own GetSlots() must
	// not widen a sparsely-packed x here.
	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + pt0->noiseScaleDeg;
	res.scalingFactor = x.scalingFactor * pt0->scalingFactor;
	res.p			  = finishPayload(hazebk::makeComponents(std::move(resB), std::move(resA)), res);
	return res;
}

// ---- BSGS linear transform (plain-rotation equivalent of OpenFHE EvalLinearTransform) ----

Ciphertext<DCRTPoly> HazeEngine::linearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Plaintext>& a, const Ciphertext<DCRTPoly>& ct, uint32_t bStep) {
	// Horner form of OpenFHE EvalLinearTransform (ckksrns-fhe.cpp:1918-1957, post-#1209): baby
	// index i is the rotation i (i=0 identity); giant blocks are consumed from the top and the
	// accumulator rotates by the single stride bStep between blocks. Term k = bStep*j + i is
	// included iff k < slots == a.size(). Run the hoisted-Ext implementation
	// (extBsgsStage); the plain form below (#19: shared precompute handle, per-index ModDown)
	// remains as the fallback for non-extended precompute or FIDESLIB_HAZE_BOOT_NO_EXT.
	const uint32_t slots = static_cast<uint32_t>(a.size());
	const uint32_t gStep = static_cast<uint32_t>(std::ceil(static_cast<double>(slots) / bStep));
	const uint32_t m	 = static_cast<uint32_t>(ctx.GetCyclotomicOrder());

	{
		auto p = ensureCt(ctx, ct);
		std::vector<int32_t> rotIn(bStep);
		for (uint32_t i = 0; i < bStep; ++i) {
			rotIn[i] = static_cast<int32_t>(i);
		}
		const int32_t stride = (gStep > 1) ? static_cast<int32_t>(bStep) : 0;
		// numRotSkip = a.size(): no oracle skip term in LT — the k < a.size() bound is the guard.
		if (auto ext = extBsgsStage(ctx, asOperand(p), a, rotIn, stride, gStep, slots)) {
			return wrapDeviceResult(ctx, ct, std::move(ext->p));
		}
	}

	auto precomp = ctx.EvalFastRotationPrecompute(ct);
	std::vector<Ciphertext<DCRTPoly>> rot(bStep);
	rot[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	for (uint32_t i = 1; i < bStep; ++i) {
		rot[i] = ctx.EvalFastRotation(ct, static_cast<int32_t>(i), m, precomp);
	}

	auto evalBlock = [&](uint32_t j) {
		auto acc = ctx.EvalMult(rot[0], const_cast<Plaintext&>(a[bStep * j]));
		for (uint32_t i = 1; i < bStep; ++i) {
			if (bStep * j + i < slots) {
				auto term = ctx.EvalMult(rot[i], const_cast<Plaintext&>(a[bStep * j + i]));
				ctx.EvalAddInPlace(acc, term);
			}
		}
		return acc;
	};

	Ciphertext<DCRTPoly> result = evalBlock(gStep - 1);
	for (int32_t j = static_cast<int32_t>(gStep) - 2; j >= 0; --j) {
		result	   = ctx.EvalRotate(result, static_cast<int32_t>(bStep));
		auto inner = evalBlock(static_cast<uint32_t>(j));
		ctx.EvalAddInPlace(result, inner);
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
	// Bootstrap bypasses the facade (it drives the engine cores directly), so the degree-1
	// precondition of every stage — rotation, keyswitch, monomial multiply — is enforced here.
	requireRelinearized(*pIn, "EvalBootstrap");
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

// Multi-stage FFT CoeffsToSlots / SlotsToCoeffs (levelBudget != {1,1}; #11). Port of OpenFHE
// EvalCoeffsToSlots / EvalSlotsToCoeffs (ckksrns-fhe.cpp:1885-2200), iterating the per-stage LT
// vectors. Each stage runs the extended-basis BSGS (extBsgsStage — baby steps in Q∥P,
// one KeySwitchDown per giant step, matching the oracle's hoisted form); the plain-EvalRotate
// body below remains as the fallback for non-extended precompute or FIDESLIB_HAZE_BOOT_NO_EXT.
Ciphertext<DCRTPoly> HazeEngine::evalCoeffsToSlotsFFT(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, uint32_t slots, bool decode) {
	const BootPrecom& bp = boot_.at(slots);
	const auto& p		 = decode ? bp.paramsDec : bp.paramsEnc;
	const auto& A		 = decode ? bp.u0PreFFT : bp.u0hatTPreFFT;
	const uint32_t M4	 = static_cast<uint32_t>(ringDim_) / 2; // cyclotomicOrder/4 = N/2
	// Post-#1209 index domain: every rotation index is reduced into [0, min(M/4, 2*slots))
	// (ckksrns-fhe.cpp:1976/2088).
	const uint32_t reduceMod = std::min(M4, 2 * slots);

	auto reduceRot = [](int32_t index, uint32_t mod) { return static_cast<int32_t>(lbcrypto::ReduceRotation(index, mod)); };

	int32_t stop	= -1;
	int32_t flagRem = 0;
	if (p.remCollapse != 0) {
		stop	= 0;
		flagRem = 1;
	}

	// Per-stage zero-based baby-rotation table (rot_in[s][0] = 0 identity) and the SINGLE Horner
	// giant stride per stage — ckksrns-fhe.cpp:1990-2001 (CtS) / 2101-2112 (StC).
	std::vector<std::vector<int32_t>> rot_in(p.lvlb);
	std::vector<int32_t> stride(p.lvlb, 0);
	if (decode) {
		// EvalSlotsToCoeffs ordering: stage s ascending, scale 1<<(s*layersCollapse); the
		// remainder stage (last) continues the scale progression with gRem babies.
		const int32_t smax = static_cast<int32_t>(p.lvlb) - flagRem;
		for (int32_t s = 0; s < smax; ++s) {
			const int32_t scale = 1 << (s * static_cast<int32_t>(p.layersCollapse));
			rot_in[s].resize(p.g);
			for (uint32_t j = 0; j < p.g; ++j)
				rot_in[s][j] = reduceRot(scale * static_cast<int32_t>(j), reduceMod);
			stride[s] = reduceRot(scale * static_cast<int32_t>(p.g), reduceMod);
		}
		if (flagRem == 1) {
			const int32_t scaleRem = 1 << (smax * static_cast<int32_t>(p.layersCollapse));
			rot_in[smax].resize(p.gRem);
			for (uint32_t j = 0; j < p.gRem; ++j)
				rot_in[smax][j] = reduceRot(scaleRem * static_cast<int32_t>(j), reduceMod);
			stride[smax] = reduceRot(scaleRem * static_cast<int32_t>(p.gRem), reduceMod);
		}
	} else {
		// EvalCoeffsToSlots ordering: stage s descending from lvlb-1; remainder at s = stop
		// with scale 1 (gRem babies, stride gRem).
		for (int32_t s = static_cast<int32_t>(p.lvlb) - 1; s > stop; --s) {
			const int32_t scale = 1 << ((s - flagRem) * static_cast<int32_t>(p.layersCollapse) + static_cast<int32_t>(p.remCollapse));
			rot_in[s].resize(p.g);
			for (uint32_t j = 0; j < p.g; ++j)
				rot_in[s][j] = reduceRot(scale * static_cast<int32_t>(j), reduceMod);
			stride[s] = reduceRot(scale * static_cast<int32_t>(p.g), reduceMod);
		}
		if (flagRem == 1) {
			rot_in[stop].resize(p.gRem);
			for (uint32_t j = 0; j < p.gRem; ++j)
				rot_in[stop][j] = reduceRot(static_cast<int32_t>(j), reduceMod);
			stride[stop] = reduceRot(static_cast<int32_t>(p.gRem), reduceMod);
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
				rebindPayload(*pr, std::move(r.p->components), r);
			}
		}
		first = false;

		const bool rem		  = (flagRem == 1) && ((decode && s == static_cast<int32_t>(p.lvlb) - 1) || (!decode && s == stop));
		const uint32_t g	  = rem ? p.gRem : p.g;
		const uint32_t b	  = rem ? p.bRem : p.b;
		const uint32_t numRot = rem ? p.numRotationsRem : p.numRotations;
		const int32_t t		  = stride[s];

		// Extended-basis Horner BSGS stage (OpenFHE's hoisted form): baby rotations stay
		// in Q∥P via the shared ModUp handle, the accumulator rotates by the single stride t
		// between giant blocks. Falls back to the plain path when the ext preconditions fail
		// (non-extended pts, depth != 1, or FIDESLIB_HAZE_BOOT_NO_EXT).
		{
			// Directional bisection: FIDESLIB_HAZE_BOOT_NO_EXT=cts|stc disables the ext form for
			// that direction only (any other non-empty value disables it globally, inside
			// extBsgsStage).
			const char* noExt	  = std::getenv("FIDESLIB_HAZE_BOOT_NO_EXT");
			const bool dirBlocked = (noExt != nullptr) && ((std::string(noExt) == "cts" && !decode) || (std::string(noExt) == "stc" && decode));
			auto px = ensureCt(ctx, result);
			if (!dirBlocked) {
				if (auto ext = extBsgsStage(ctx, asOperand(px), A[s], rot_in[s], t, b, numRot)) {
					result = wrapDeviceResult(ctx, result, std::move(ext->p));
					continue;
				}
			}
		}

		// Plain-rotation fallback (#19 hoisting kept): the same Horner accumulation with plain
		// per-rotation ModDown. Mathematically identical; bit-parity with the oracle is owned by
		// the ext path above.
		const uint32_t M = static_cast<uint32_t>(ctx.GetCyclotomicOrder());
		auto precomp	 = ctx.EvalFastRotationPrecompute(result);
		std::vector<Ciphertext<DCRTPoly>> fastRotation(g);
		for (uint32_t j = 0; j < g; ++j) {
			fastRotation[j] = (rot_in[s][j] != 0) ? ctx.EvalFastRotation(result, rot_in[s][j], M, precomp) : std::make_shared<CiphertextImpl<DCRTPoly>>(*result);
		}

		auto evalBlock = [&](uint32_t i, bool topBlock) {
			const uint32_t G		 = g * i;
			Ciphertext<DCRTPoly> acc = ctx.EvalMult(fastRotation[0], const_cast<Plaintext&>(A[s][G]));
			for (uint32_t j = 1; j < g; ++j) {
				if ((G + j) >= A[s].size() || (topBlock && (G + j) == numRot)) {
					continue;
				}
				auto term = ctx.EvalMult(fastRotation[j], const_cast<Plaintext&>(A[s][G + j]));
				ctx.EvalAddInPlace(acc, term);
			}
			return acc;
		};

		Ciphertext<DCRTPoly> outer = evalBlock(b - 1, /*topBlock=*/true);
		for (int32_t i = static_cast<int32_t>(b) - 2; i >= 0; --i) {
			if (t != 0) {
				outer = ctx.EvalRotate(outer, t);
			}
			auto inner = evalBlock(static_cast<uint32_t>(i), /*topBlock=*/false);
			ctx.EvalAddInPlace(outer, inner);
		}
		result = outer;
	}

	// Undo the per-stage accumulated pre-rotation: Aut_{-(slots-1)}, applied by the oracle at the
	// end of BOTH EvalCoeffsToSlots and EvalSlotsToCoeffs (ckksrns-fhe.cpp:2070/2183). The oracle
	// goes through EvalAtIndex — FindAutomorphismIndex key, no slots-aware remap — so mirror that
	// exact automorphism via rotateByAutoIndex.
	const int32_t delta = reduceRot(-static_cast<int32_t>(slots - 1), reduceMod);
	if (delta != 0) {
		auto px					 = ensureCt(ctx, result);
		auto& context			 = hostContext(ctx);
		const uint32_t autoIndex = context->FindAutomorphismIndex(static_cast<uint32_t>(delta));
		Operand rot				 = rotateByAutoIndex(ctx, asOperand(px), autoIndex);
		result					 = wrapDeviceResult(ctx, result, std::move(rot.p));
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
		rebindPayload(*p, std::move(r.p->components), r);
	};
	auto toDepthOne = [&](Ciphertext<DCRTPoly>& c) { // ModReduceInternalInPlace(NSD-1)
		while (devicePayload(c)->noiseScaleDeg > 1) {
			rescaleHard(c);
		}
	};
	auto multInt = [&](Ciphertext<DCRTPoly>& c, uint64_t s) {
		auto p	  = devicePayload(c);
		Operand r = multIntCore(asOperand(p), s);
		rebindPayload(*p, std::move(r.p->components), r);
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
				for (auto& comp : p->components) {
					comp.truncate(targetTowers);
				}
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
				for (auto& comp : p->components) {
					comp.truncate(p->towers - drop);
				}
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
		rebindPayload(*p, std::move(r.p->components), r);
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
	// Fully-packed (slots == N/2): the COMPLEX modular-reduction branch runs (Bootstrap.cu:268-278,
	// ApproxModEval.cu:24-92). The slots != N/2 (sparse) branch keeps the single-Chebyshev path.
	const bool fullyPacked = (slots == N / 2);
	const BootPrecom& bp   = boot_.at(slots);

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
		rebindPayload(*p, std::move(r.p->components), r);
	};
	auto toDepthOne = [&](Ciphertext<DCRTPoly>& c) {
		while (devicePayload(c)->noiseScaleDeg > 1)
			rescaleHard(c);
	};
	auto multInt = [&](Ciphertext<DCRTPoly>& c, uint64_t s) {
		auto p	  = devicePayload(c);
		Operand r = multIntCore(asOperand(p), s);
		rebindPayload(*p, std::move(r.p->components), r);
	};
	auto multMonomial = [&](Ciphertext<DCRTPoly>& c, uint32_t power) {
		auto p	  = devicePayload(c);
		Operand r = multByMonomialCore(asOperand(p), power);
		rebindPayload(*p, std::move(r.p->components), r);
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
		rebindPayload(*p, std::move(r.p->components), r);
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

	// Conjugate split (Bootstrap.cu:268-285; OpenFHE :762). conj = Conjugate(ctxtEnc).
	// Sparse (slots != N/2): ctxtEnc += conj (real part only).
	// Fully packed (slots == N/2): ctxtEnc = ctxtEnc + conj (real), ctxtEncI = ctxtEnc - conj (imag),
	//   ctxtEncI *= X^{3N/2}. Both go through the COMPLEX dual approxModReduction.
	Ciphertext<DCRTPoly> ctxtEncI; // program-local imaginary part (fully-packed only; never markOutput)
	{
		auto pEnc	 = devicePayload(ctxtEnc);
		Operand conj = rotateByAutoIndex(ctx, asOperand(pEnc), 2 * N - 1);
		auto conjCt	 = wrapDeviceResult(ctx, ctxtEnc, std::move(conj.p));
		if (fullyPacked) {
			ctxtEncI = ctx.EvalSub(ctxtEnc, conjCt); // imag = ct - conj (Bootstrap.cu:271)
			ctx.EvalAddInPlace(ctxtEnc, conjCt);	 // real = ct + conj (Bootstrap.cu:272)
			multMonomial(ctxtEncI, 3 * N / 2);		 // x X^{3N/2} (Bootstrap.cu:273)
		} else {
			ctx.EvalAddInPlace(ctxtEnc, conjCt);
		}
	}
	// FIXEDMANUAL: drain to depth 1; else rescale only if NSD==2 (Bootstrap.cu:274-283 / OpenFHE :764-773).
	// Apply to both parts so they stay level-aligned through the dual Chebyshev (ct ± conj share NSD,
	// and multMonomial preserves it, so the NSD==2 test agrees for both).
	auto conjRescale = [&](Ciphertext<DCRTPoly>& c) {
		if (fixedManual) {
			toDepthOne(c);
		} else if (devicePayload(c)->noiseScaleDeg == 2) {
			rescaleHard(c);
		}
	};
	conjRescale(ctxtEnc);
	if (fullyPacked) {
		conjRescale(ctxtEncI);
	}
	if (bootDebugStage() == 3) {
		return ctxtEnc;
	}

	// ---- Approximate modular reduction (Chebyshev + double-angle). Sparse: ApproxModEval.cu:94-129
	// on ctxtEnc only. COMPLEX: ApproxModEval.cu:24-92 — same Chebyshev+double-angle on BOTH parts,
	// then ctxtEncI *= X^{N/2} and ctxtEnc += ctxtEncI before the final scalar mult. ----
	auto chebyshevAndDoubleAngle = [&](Ciphertext<DCRTPoly>& c) {
		std::vector<double> coeffs = bp.coefficients;
		ctx.EvalChebyshevSeriesInPlace(c, coeffs, -1.0, 1.0);
		if (!fixedManual) {
			toDepthOne(c); // ModReduceInternal after Chebyshev (OpenFHE :790-791)
		}
		constexpr double twoPi = 2.0 * M_PI;
		for (int32_t i = 1 - static_cast<int32_t>(bp.numIter); i <= 0; ++i) {
			const double angleScalar = -std::pow(twoPi, -std::pow(2.0, static_cast<double>(i)));
			ctx.EvalSquareInPlace(c);
			auto shifted = ctx.EvalAdd(c, angleScalar);
			ctx.EvalAddInPlace(c, shifted);
			ctx.RescaleInPlace(c); // eager (parity #3); FIXEDAUTO no-op on OpenFHE, eager here
		}
	};
	// CUDA orders the imaginary Chebyshev first (ApproxModEval.cu:35-36), then real; mirror it.
	if (fullyPacked) {
		chebyshevAndDoubleAngle(ctxtEncI);
	}
	chebyshevAndDoubleAngle(ctxtEnc);
	if (fullyPacked) {
		// ctxtEncI *= X^{N/2} (== x i), then recombine (ApproxModEval.cu:73-76).
		multMonomial(ctxtEncI, N / 2);
		ctx.EvalAddInPlace(ctxtEnc, ctxtEncI);
	}
	multInt(ctxtEnc, scalar); // scale message back up (Bootstrap.cu:119 / OpenFHE :798 / ApproxModEval.cu:80)
	if (bootDebugStage() == 4) {
		return ctxtEnc;
	}

	// ModReduceInternal before SlotsToCoeffs (OpenFHE :814-815). Skipped under FIXEDMANUAL.
	if (!fixedManual) {
		toDepthOne(ctxtEnc);
	}

	// ---- SlotsToCoeffs (Bootstrap.cu:293-296; isLT -> EvalLinearTransform, else FFT) ----
	Ciphertext<DCRTPoly> ctxtDec = bp.isLT ? linearTransform(ctx, bp.u0Pre, ctxtEnc, bp.bStep) : evalCoeffsToSlotsFFT(ctx, ctxtEnc, slots, /*decode=*/true);

	// Sparse fold (Bootstrap.cu:299-302; OpenFHE :819). Skipped when fully packed (slots == N/2).
	if (!fullyPacked) {
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
