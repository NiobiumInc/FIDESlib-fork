#ifdef FIDESLIB_ENABLE_HAZE

// Bootstrap orchestration. The oracle is OpenFHE FHECKKSRNS::EvalBootstrap and EvalBootstrapStCFirst (ckksrns-fhe.cpp); both
// circuits run on every packing, level budget and scaling technique, and the bar is limb-exact equality with the oracle.
// FIDESlib src/CKKS/Bootstrap.cu is the cross-reference for the stage list.

#include "engine/haze/HazeEngine.hpp"

#include "CKKS/StCFirstSaveLevel.hpp"
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

// The oracle's correction, m_correctionFactor - deg (ckksrns-fhe.cpp:789-794, 1194-1199): the degree the moduli give must fit
// the correction factor, else the unsigned difference wraps into a scale that decodes to garbage.
uint32_t bootstrapCorrection(uint32_t correctionFactor, int32_t deg) {
	if (deg > static_cast<int32_t>(correctionFactor)) {
		throw std::runtime_error(
		  "haze backend: degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" + std::to_string(correctionFactor) + "]");
	}
	return correctionFactor - static_cast<uint32_t>(deg);
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
	if (precom->StCFirstSavesLevel) {
		const auto raise = lbcrypto::FHECKKSRNS::GetStCFirstRaiseScaling(std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters()));
		bp.savedLevel = BootPrecom::SavedLevel{ precom->m_foldedCorrection, raise.stcSF, raise.targetSF };
	}
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

HazeEngine::Operand HazeEngine::raiseModulusCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, size_t targetTowers) {
	if (ctx.keyDist != SPARSE_ENCAPSULATED) {
		return modRaiseCore(x, targetTowers);
	}
	// OpenFHE EvalBootstrap, SPARSE_ENCAPSULATED: KeySwitchSparse (2N-4) to the sparse secret, the raise from its
	// first prime, then the hybrid KeySwitchInPlace (2N-2) back to the dense secret.
	const uint32_t N = static_cast<uint32_t>(ringDim_);
	Operand sparse	 = keySwitchSparseCore(ctx, x);
	Operand raised	 = modRaiseCore(sparse, targetTowers);
	return keySwitchByAutoIndex(ctx, raised, 2 * N - 2);
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

HazeEngine::Operand HazeEngine::extBsgsStage(CryptoContextImpl<DCRTPoly>& ctx,
  const Operand& x,
  const std::vector<Plaintext>& pts,
  const std::vector<int32_t>& rotIn,
  int32_t giantStride,
  uint32_t numBlocks,
  uint32_t numRotSkip,
  int32_t deltaStride) {
	// Only this extended form rounds once per giant step like the oracle; a plain-rotation evaluation decodes correctly
	// but differs in the low bits. A stage that cannot run it is therefore refused, never evaluated another way.
	if (pts.empty() || rotIn.empty() || numBlocks == 0) {
		throw std::runtime_error("haze backend: extended bsgs stage needs plaintexts, baby steps and giant blocks (got " + std::to_string(pts.size()) + ", " + std::to_string(rotIn.size()) + ", " + std::to_string(numBlocks) + ")");
	}
	if (x.noiseScaleDeg != 1) {
		throw std::runtime_error("haze backend: extended bsgs stage needs a depth-1 input (noise scale degree " + std::to_string(x.noiseScaleDeg) + ")");
	}
	if (x.p->components.size() != 2) {
		throw std::runtime_error("haze backend: extended bsgs stage needs a degree-1 input (" + std::to_string(x.p->components.size()) + " components)");
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
		throw std::runtime_error("haze backend: extended bsgs stage needs plaintexts encoded in the extended basis at depth 1 (noise scale degree " + std::to_string(pt0->noiseScaleDeg) + ", "
								 + std::to_string(pt0->towers) + " towers for " + std::to_string(towers + numP) + " required)");
	}
	const size_t ptQ = pt0->towers - numP;
	if (!(ptQ == towers || (fixedMode && ptQ > towers))) {
		throw std::runtime_error("haze backend: extended bsgs stage needs plaintexts at the input's level (" + std::to_string(ptQ) + " base towers for an input of " + std::to_string(towers)
								 + "; a deeper encoding is exact only under the fixed scaling techniques)");
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
	auto buildBlock = [&](uint32_t i, bool topBlock) -> std::pair<LimbChain, LimbChain> {
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
			throw std::runtime_error("haze backend: extended bsgs stage has an empty giant block (block " + std::to_string(i) + " of " + std::to_string(b) + ")");
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

	// EvalHornerGiantRotate (ckksrns-fhe.cpp:97-128): only the a part is moddowned, since its digit decomposition needs
	// base-Q rows; the b part stays extended and is folded into the key-switch product on every Q∥P row. Rounding b
	// early would differ from the oracle's single final KeySwitchDown by a carry.
	auto hornerRotate = [&](LimbChain& outB, LimbChain& outA, KsKey& key, uint32_t autoIndex) {
		LimbChain aQ		   = extModDownChain(outA, towers);
		const HoistedDigits gd = hoistedKeyswitchPrefix(aQ, towers, numPartQ_);
		KsContribution ks	   = hybridKeyswitchFromDigits(gd, key, /*ext=*/true);
		addInPlace(ks.b, outB, qpBase);
		outB = automorphChain(ks.b, autoIndex, qpBase);
		outA = automorphChain(ks.a, autoIndex, qpBase);
	};

	// Horner backward accumulation (ckksrns-fhe.cpp:2011-2026): start from the TOP giant block,
	// rotate the ACCUMULATOR by the constant stride between blocks, Ext-add each lower block;
	// ONE final KeySwitchDown of both components.
	auto topBlock	 = buildBlock(b - 1, /*topBlock=*/true);
	LimbChain outerB = std::move(topBlock.first);
	LimbChain outerA = std::move(topBlock.second);
	for (int32_t i = static_cast<int32_t>(b) - 2; i >= 0; --i) {
		if (gKey != nullptr) {
			hornerRotate(outerB, outerA, *gKey, gAuto);
		}
		const auto blk = buildBlock(static_cast<uint32_t>(i), /*topBlock=*/false);
		addInPlace(outerB, blk.first, qpBase);
		addInPlace(outerA, blk.second, qpBase);
	}

	// The CtS/StC delta rotation, folded into the last stage's extended output (ckksrns-fhe.cpp:2216-2221,
	// 2262-2267, 2338-2343, 2386-2391). The oracle resolves its key by FindAutomorphismIndex, with no slots-aware remap.
	if (deltaStride != 0) {
		const uint32_t deltaAuto = hostContext(ctx)->FindAutomorphismIndex(static_cast<uint32_t>(deltaStride));
		hornerRotate(outerB, outerA, autoKeyFor(ctx, deltaAuto), deltaAuto);
	}

	// result = KeySwitchDown(outer) (ckksrns-fhe.cpp:2026).
	LimbChain resB = extModDownChain(outerB, towers);
	LimbChain resA = extModDownChain(outerA, towers);

	// EvalMultExt metadata (NSD += pt.NSD, sf ×= pt.sf) — the ext dance itself is metadata-
	// neutral, as in multPtCore: slots stays x.slots. pt0 is a
	// bootstrap linear-transform plaintext encoded at full ring width, so its own GetSlots() must
	// not widen a sparsely-packed x here.
	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + pt0->noiseScaleDeg;
	res.scalingFactor = x.scalingFactor * pt0->scalingFactor;
	res.p			  = finishPayload(hazebk::makeComponents(std::move(resB), std::move(resA)), res);
	return res;
}

// ---- BSGS linear transform (OpenFHE EvalLinearTransform) ----

Ciphertext<DCRTPoly> HazeEngine::linearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Plaintext>& a, const Ciphertext<DCRTPoly>& ct, uint32_t bStep) {
	// Horner form of OpenFHE EvalLinearTransform (ckksrns-fhe.cpp:1918-1957, post-#1209): baby index i is the rotation i
	// (i=0 identity); giant blocks are consumed from the top and the accumulator rotates by the single stride bStep
	// between blocks. Term k = bStep*j + i is included iff k < slots == a.size().
	const uint32_t slots = static_cast<uint32_t>(a.size());
	const uint32_t gStep = static_cast<uint32_t>(std::ceil(static_cast<double>(slots) / bStep));

	auto p = ensureCt(ctx, ct);
	std::vector<int32_t> rotIn(bStep);
	for (uint32_t i = 0; i < bStep; ++i) {
		rotIn[i] = static_cast<int32_t>(i);
	}
	const int32_t stride = (gStep > 1) ? static_cast<int32_t>(bStep) : 0;
	// numRotSkip = a.size(): no oracle skip term in LT — the k < a.size() bound is the guard.
	Operand res = extBsgsStage(ctx, asOperand(p), a, rotIn, stride, gStep, slots, /*deltaStride=*/0);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

// ---- staged bootstrap ----

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
	// The precom's BTSlotsEncoding flag (forwarded from btsfirstboot, default false) selects the circuit, as it does in
	// OpenFHE's EvalBootstrap and in CUDA's Bootstrap(): ModRaise-first when false, StC-first when true.
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
		return bootstrapStCFirst(ctx, ciphertext); // BTSlotsEncoding=true: OpenFHE's EvalBootstrapStCFirst
	}
	return bootstrapModRaiseFirst(ctx, ciphertext); // BTSlotsEncoding=false: OpenFHE's EvalBootstrap
}

// Multi-stage FFT CoeffsToSlots / SlotsToCoeffs (levelBudget != {1,1}; #11). Port of OpenFHE
// EvalCoeffsToSlots / EvalSlotsToCoeffs (ckksrns-fhe.cpp:1885-2200), iterating the per-stage LT
// vectors. Each stage runs the extended-basis BSGS (extBsgsStage — baby steps in Q∥P,
// one KeySwitchDown per giant step, matching the oracle's hoisted form), which refuses what it cannot run.
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

	// Aut_{-(slots-1)}, the accumulated per-stage pre-rotation, applied by the oracle at the end of both
	// EvalCoeffsToSlots and EvalSlotsToCoeffs (ckksrns-fhe.cpp:2216-2221, 2262-2267, 2338-2343, 2386-2391): inside the last
	// evaluated stage, before its final ModDown.
	const int32_t delta = reduceRot(-static_cast<int32_t>(slots - 1), reduceMod);

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
		const uint32_t b	  = rem ? p.bRem : p.b;
		const uint32_t numRot = rem ? p.numRotationsRem : p.numRotations;
		const int32_t t		  = stride[s];
		const int32_t stageDelta = (s == stageOrder.back()) ? delta : 0;

		// Extended-basis Horner BSGS stage (OpenFHE's hoisted form): baby rotations stay in Q∥P via the shared ModUp
		// handle, the accumulator rotates by the single stride t between giant blocks.
		auto px		= ensureCt(ctx, result);
		Operand ext = extBsgsStage(ctx, asOperand(px), A[s], rot_in[s], t, b, numRot, stageDelta);
		result		= wrapDeviceResult(ctx, result, std::move(ext.p));
	}
	return result;
}

Ciphertext<DCRTPoly> HazeEngine::bootstrapStCFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// Port of OpenFHE FHECKKSRNS::EvalBootstrapStCFirst (ckksrns-fhe.cpp:1091-1476), the circuit BTSlotsEncoding=true
	// selects: deplete -> SlotsToCoeffs -> raise -> CoeffsToSlots -> EvalMod -> scale-back, single iteration, REAL data,
	// every packing, level budget and scaling technique. An input that would not gain levels runs the whole pipeline and
	// is then returned as a clone by the oracle's no-gain check (:1472), except a full-height one, which returns before the circuit.
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool fixedManual = (st == lbcrypto::FIXEDMANUAL);

	auto pIn = ensureCt(ctx, ciphertext);
	requireComputable(*pIn, "EvalBootstrap");

	const uint32_t slots = static_cast<uint32_t>(pIn->slots);
	const uint32_t N	 = static_cast<uint32_t>(ringDim_);
	const auto bpIt = boot_.find(slots);
	if (bpIt == boot_.end()) {
		throw std::runtime_error("haze backend: no bootstrap precomputation for " + std::to_string(slots) + " slots (run EvalBootstrapSetup + EvalBootstrapKeyGen before LoadContext)");
	}
	const BootPrecom& bp = bpIt->second;
	const std::optional<BootPrecom::SavedLevel>& saved = bp.savedLevel;

	// Constants (ckksrns-fhe.cpp:1191-1205, 64-bit branch). A saved-level set applies the exponent OpenFHE reads at this bootstrap,
	// which also raises OpenFHE's own refusals.
	const uint32_t L0	 = static_cast<uint32_t>(qBase_.size());
	const double qDouble = static_cast<double>(qBase_.front());
	const double powP	 = std::pow(2.0, static_cast<double>(plaintextModulus_));
	const int32_t deg	 = static_cast<int32_t>(std::round(std::log2(qDouble / powP)));
	const uint32_t correction =
	  saved ? FIDESlib::CKKS::ReadStCFirstAppliedCorrection(hostContext(ctx), slots, saved->foldedCorrection) : bootstrapCorrection(bp.correctionFactor, deg);
	const double post		  = std::pow(2.0, static_cast<double>(deg));
	const double pre		  = 1.0 / post;
	const uint64_t scalar	  = static_cast<uint64_t>(std::llround(post));

	const size_t initTowers = pIn->towers;

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

	// ---- Level guard (ckksrns-fhe.cpp:1250-1257) ----
	// m_paramsDec.lvlb: BootPrecom copies the FFT parameters only when !isLT, and isLT means both budgets are 1.
	const uint32_t lvlbDec	   = bp.isLT ? 1 : bp.paramsDec.lvlb;
	const size_t expectedLevel = saved ? L0 - (lvlbDec + 1) : L0 - (lvlbDec + 2 + (st == lbcrypto::FLEXIBLEAUTOEXT ? 1 : 0));
	if ((L0 - pIn->towers) + (pIn->noiseScaleDeg - 1) > expectedLevel) {
		throw std::runtime_error("haze backend: not enough levels to perform bootstrapping");
	}
	// The oracle runs the circuit on a full-height input and returns the input clone at its no-gain check, since every
	// stage after the deplete consumes levels and the output is always below the chain top. Skipping the circuit is
	// therefore exact, and it avoids the FLEXIBLE deplete's read of the scaling-factor tables below level 0.
	if (pIn->towers == L0) {
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}

	// ---- Dropping unnecessary towers (deplete; ckksrns-fhe.cpp:1259-1284) ----
	Ciphertext<DCRTPoly> work = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	{
		toDepthOne(work);
		auto p				   = devicePayload(work);
		const size_t ctxtLevel = L0 - p->towers;
		if (st == lbcrypto::FIXEDAUTO || fixedManual) {
			if (ctxtLevel < expectedLevel) {
				// LevelReduceInternalInPlace: pure truncation, no IR (oracle :1262-1266).
				const size_t targetTowers = L0 - expectedLevel;
				for (auto& comp : p->components) {
					comp.truncate(targetTowers);
				}
				p->towers = targetTowers;
			}
		} else if (ctxtLevel < expectedLevel) {
			// FLEXIBLE deplete (oracle :1268-1280): scale to the expected level's factor,
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
	// CheckStCFirstInputScalingFactor: the factor that reaches the raise is fixed at setup.
	if (saved && st != lbcrypto::FIXEDAUTO && !fixedManual) {
		const auto p = devicePayload(work);
		if (std::abs(p->scalingFactor / sfReal_[L0 - p->towers] - 1.0) > FIDESlib::CKKS::kStCFirstScalingFactorTolerance) {
			throw std::runtime_error("haze backend: the scaling factor of the bootstrap input differs from the scaling factor of its level; provide an "
									 "input with at least one more level");
		}
	}
	if (bootDebugStage() == 1) {
		return work;
	}

	// ---- SlotsToCoeffs first (isLT -> EvalLinearTransform, else FFT), then the sparse fold (oracle :1292-1300) ----
	work = bp.isLT ? linearTransform(ctx, bp.u0Pre, work, bp.bStep) : evalCoeffsToSlotsFFT(ctx, work, slots, /*decode=*/true);
	if (slots != N / 2) {
		auto rotated = ctx.EvalRotate(work, static_cast<int32_t>(slots));
		ctx.EvalAddInPlace(work, rotated);
	}
	if (bootDebugStage() == 2) {
		return work;
	}

	// ---- Raising the modulus ----
	if (saved) {
		// RescaleStCFirstOutputInPlace: the last SlotsToCoeffs rescale lands on the raising tower; the correction is in its matrix.
		if (st == lbcrypto::FIXEDAUTO || fixedManual) {
			toDepthOne(work);
		} else {
			const auto p = devicePayload(work);
			if (p->noiseScaleDeg != 2 || std::abs(p->scalingFactor / saved->stcSF - 1.0) > FIDESlib::CKKS::kStCFirstScalingFactorTolerance) {
				throw std::runtime_error("haze backend: unexpected scaling factor of the SlotsToCoeffs output in bootstrapping");
			}
			rescaleHard(work);
			devicePayload(work)->scalingFactor = saved->targetSF;
		}
		if (correction < saved->foldedCorrection) {
			multInt(work, uint64_t{ 1 } << (saved->foldedCorrection - correction));
		}
	} else {
		toDepthOne(work); // ModReduceInternalInPlace(NSD-1)
		// AdjustCiphertext(work, 2^-correction, lvl) with the modReduce=true default, so the
		// raise is entered at depth 1. FIXED modes: public EvalMult by the correction scale.
		// FLEXIBLE modes (oracle ckksrns-fhe.cpp:2435-2461): scale to the target level's
		// factor folded with the correction, then pin the scaling factor.
		if (st == lbcrypto::FIXEDAUTO || fixedManual) {
			ctx.EvalMultInPlace(work, std::pow(2.0, -static_cast<double>(correction)));
			rescaleHard(work);
		} else {
			auto p						  = devicePayload(work);
			const uint32_t lvl			  = (st == lbcrypto::FLEXIBLEAUTOEXT) ? 1 : 0;
			const double targetSF		  = sfReal_[lvl];
			const double sourceSF		  = p->scalingFactor;
			const double modToDrop		  = modReduceFactor_[p->towers - 1]; // double(q_{towers-1}) under FLEXIBLE
			const double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * std::pow(2.0, -static_cast<double>(correction));
			ctx.EvalMultInPlace(work, adjustmentFactor);
			rescaleHard(work);
			devicePayload(work)->scalingFactor = targetSF;
		}
	}
	{
		auto p					  = devicePayload(work);
		const size_t raiseTowers  = L0 - (st == lbcrypto::FLEXIBLEAUTOEXT ? 1 : 0);
		Operand r				  = raiseModulusCore(ctx, asOperand(p), raiseTowers);
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

	// ---- Partial sum (sparse packing only; oracle :1369-1378): EvalPartialSumInPlace(work, slots), the
	// AccumulateSum core with stride = slots and size = N / (2 * slots) ----
	if (slots != N / 2) {
		accumulateSumInPlace(ctx, work, static_cast<int>(N / (2 * slots)), static_cast<int>(slots));
	}
	rescaleHard(work); // ModReduceInternalInPlace(1) before CoeffsToSlots (oracle :1388)
	if (bootDebugStage() == 5) {
		return work;
	}

	// ---- CoeffsToSlots (isLT -> EvalLinearTransform, else FFT; oracle :1390-1391) ----
	auto ctxtEnc = bp.isLT ? linearTransform(ctx, bp.u0hatTPre, work, bp.bStep) : evalCoeffsToSlotsFFT(ctx, work, slots, /*decode=*/false);

	// ctxtEnc += Conjugate(ctxtEnc): keyswitch with the conjugation key (2N-1). REAL data has no imaginary
	// branch at any packing: the raised CtS output's imaginary part is q0 * integer noise (oracle :1393-1400).
	{
		auto pEnc	 = devicePayload(ctxtEnc);
		Operand conj = rotateByAutoIndex(ctx, asOperand(pEnc), 2 * N - 1);
		auto conjCt	 = wrapDeviceResult(ctx, ctxtEnc, std::move(conj.p));
		ctx.EvalAddInPlace(ctxtEnc, conjCt);
	}
	// FIXEDMANUAL: drain to depth 1; else rescale only if NSD==2 (oracle :1402-1415).
	if (fixedManual) {
		toDepthOne(ctxtEnc);
	} else if (devicePayload(ctxtEnc)->noiseScaleDeg == 2) {
		rescaleHard(ctxtEnc);
	}
	if (bootDebugStage() == 6) {
		return ctxtEnc;
	}

	// ---- Approximate modular reduction: Chebyshev + double-angle ----
	std::vector<double> coeffs = bp.coefficients;
	ctx.EvalChebyshevSeriesInPlace(ctxtEnc, coeffs, -1.0, 1.0);
	if (!fixedManual) {
		toDepthOne(ctxtEnc); // ModReduceInternalInPlace(NSD-1) (oracle :1438-1439)
	}
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
			if (fixedManual) {
				rescaleHard(ctxtEnc); // public ModReduceInPlace: rescales only under FIXEDMANUAL
			}
		}
	}
	if (bootDebugStage() == 8) {
		return ctxtEnc;
	}

	// ---- Scale the message back up; 64-bit correction ----
	multInt(ctxtEnc, scalar);
	multInt(ctxtEnc, static_cast<uint64_t>(1) << correction);

	// Oracle :1472-1473: if bootstrapping did not gain levels, return the input clone (the
	// recorded pipeline becomes a discarded program-local — never tagged, never read).
	if (devicePayload(ctxtEnc)->towers <= initTowers) {
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}
	return ctxtEnc;
}

Ciphertext<DCRTPoly> HazeEngine::bootstrapModRaiseFirst(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// ModRaise-first (normal) bootstrap: OpenFHE's EvalBootstrap, which CUDA's Bootstrap() (Bootstrap.cu:169) mirrors.
	// Cross-ref OpenFHE FHECKKSRNS::EvalBootstrap
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
	const uint32_t correction = bootstrapCorrection(bp.correctionFactor, deg);
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
		Operand r = multByMonomialCore(ctx, asOperand(p), power);
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
		Operand r				 = raiseModulusCore(ctx, asOperand(p), raiseTowers);
		rebindPayload(*p, std::move(r.p->components), r);
	}
	if (bootDebugStage() == 1) {
		return raised;
	}

	// ---- Scale before CoeffsToSlots (Bootstrap.cu:225,235; OpenFHE :639) ----
	ctx.EvalMultInPlace(raised, pre * (1.0 / (bp.k * static_cast<double>(N))));

	// ---- Sparse partial sum (Bootstrap.cu sparse Accumulate; OpenFHE :744-746): EvalPartialSumInPlace(raised, slots),
	// the AccumulateSum core with stride = slots and size = N / (2 * slots); a no-op when fully packed ----
	accumulateSumInPlace(ctx, raised, static_cast<int>(N / (2 * slots)), static_cast<int>(slots));
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
			if (fixedManual) {
				rescaleHard(c); // public ModReduceInPlace: rescales only under FIXEDMANUAL
			}
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
HazeEngine::evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool /*prescaled*/) {
	if (numIterations <= 1) {
		return bootstrapStaged(ctx, ciphertext);
	}
	if (numIterations != 2) {
		throw std::runtime_error("haze backend: CKKS bootstrapping supports 1 or 2 iterations");
	}
	// Meta-BTS (iterated bootstrapping): OpenFHE's EvalBootstrap for numIterations == 2
	// (ckksrns-fhe.cpp; the step numbers below are the oracle's), in the order
	// CudaEngine::evalBootstrapMetaDevice runs it on the device. The two bootstraps are recorded
	// like any single one and the glue -- the exact scale-ups by 2^precision, the error
	// subtraction, the recombination and the scale-down -- runs on the recorded operands, so
	// nothing is read back. Until here the two parameters were ignored and a caller asking for
	// two iterations got one refresh, with its precision, silently (BITCOMPAT O11).
	auto& context			= hostContext(ctx);
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	const auto st			= cryptoParams->GetScalingTechnique();
	const uint64_t pow2		= uint64_t{ 1 } << precision;

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
	auto multInt = [&](Ciphertext<DCRTPoly>& c, uint64_t s) { // MultByIntegerInPlace: exact, scale metadata unchanged
		auto p	  = devicePayload(c);
		Operand r = multIntCore(asOperand(p), s);
		rebindPayload(*p, std::move(r.p->components), r);
	};

	auto pIn			  = ensureCt(ctx, ciphertext);
	const size_t inTowers = pIn->towers;

	// Step 3: the initial bootstrap. bootstrapStaged works on a copy, the input stays intact.
	Ciphertext<DCRTPoly> b1 = bootstrapStaged(ctx, ciphertext);
	toDepthOne(b1);
	if (devicePayload(b1)->towers <= inTowers) {
		// Bootstrapping gained no towers: the oracle returns the input (ckksrns-fhe.cpp :835).
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	}
	// Step 4: scale the bootstrapped ciphertext up by 2^precision.
	multInt(b1, pow2);
	// Step 2: scale the ORIGINAL input up by 2^precision (the copy constructor clones the payload).
	Ciphertext<DCRTPoly> up = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	if (st == lbcrypto::FIXEDMANUAL) {
		toDepthOne(up);
	}
	multInt(up, pow2);
	// Steps 5-7: the bootstrapping error. Under FIXEDAUTO the oracle drops the bootstrapped
	// ciphertext to the input's towers first (DropLastElements, a pure truncation); under the
	// FLEXIBLE techniques the subtraction's own level-and-scale adjustment brings it down.
	Ciphertext<DCRTPoly> err = std::make_shared<CiphertextImpl<DCRTPoly>>(*b1);
	if (st == lbcrypto::FIXEDAUTO) {
		auto p = devicePayload(err);
		for (auto& comp : p->components) {
			comp.truncate(inTowers);
		}
		p->towers = inTowers;
	}
	ctx.EvalSubInPlace(err, up);
	// Step 8: bootstrap the error.
	Ciphertext<DCRTPoly> b2 = bootstrapStaged(ctx, err);
	toDepthOne(b2);
	// Steps 9-10: subtract the bootstrapped error, scale back down by 2^precision.
	ctx.EvalSubInPlace(b1, b2);
	ctx.EvalMultInPlace(b1, 1.0 / static_cast<double>(pow2));
	return b1;
}

void HazeEngine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	ciphertext = evalBootstrap(ctx, ciphertext, numIterations, precision, prescaled);
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
