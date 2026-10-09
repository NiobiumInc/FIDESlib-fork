#include "engine/cpu/OpenFheEngine.hpp"

#include "CryptoContext.hpp"
#include "engine/EngineCommon.hpp"
#include "engine/SparseEncapsulation.h"

#include <openfhe.h>

#include <algorithm>
#include <any>
#include <cassert>
#include <stdexcept>
#include <string>

namespace fideslib {

OpenFheEngine::OpenFheEngine(bool reducedNoise) : Engine(reducedNoise) {
	if (reducedNoise != LinkedOpenFheReducedNoise()) {
		throw std::runtime_error("cpu backend cannot honour reducedNoise=" + std::string(reducedNoise ? "true" : "false") +
		  ": the linked openfhe was built with WITH_REDUCED_NOISE=" + (LinkedOpenFheReducedNoise() ? "ON" : "OFF"));
	}
}

// Most methods here are thin shims over OpenFHE, which makes OpenFHE the reference oracle: if CPU and
// CUDA disagree on, say, evalMult, the CUDA kernel is the suspect. The exceptions are the ops OpenFHE has
// no equivalent for — linearTransform / convolutionTransform / specialConvolutionTransform — which
// re-trace the CUDA algorithm by hand. The convolution tests compare against a clear-text replay of that
// same algorithm, so they catch a rotation/rescale/encoding slip but not a flaw in the algorithm itself;
// the linear transform is checked against an independent host matrix-vector product instead, which does
// pin the algorithm. Keep these as thin as possible. accumulateSum used to be in that list; it now
// delegates to OpenFHE's EvalPartialSumInPlace, which folds with the same association the CUDA path uses.

namespace {
// Unwrap the host-side OpenFHE objects from the value types' / context's `host` std::any (parity with
// engine/cuda's deviceCt/devicePt). On the CPU backend `host` always carries the value, so these are the
// single access point; callers that mutate in place still EnsureLazyHostCopy() first.
lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& hostContext(CryptoContextImpl<DCRTPoly>& ctx) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& hostCt(const Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
}

lbcrypto::Plaintext& hostPt(const Plaintext& pt) {
	return std::any_cast<lbcrypto::Plaintext&>(pt->host);
}

// Store a freshly-computed host OpenFHE ciphertext into an existing value type's `host` slot
// (parity with engine/cuda writing back a device result). The single place that packaging lives.
void setHostCt(const Ciphertext<DCRTPoly>& ct, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res) {
	ct->host = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(res));
}

// Wrap a host result in a new value-type Ciphertext parented to `ctx`, then set its `host` slot.
Ciphertext<DCRTPoly> wrapHostCt(CryptoContextImpl<DCRTPoly>& ctx, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res) {
	Ciphertext<DCRTPoly> ct = std::make_shared<CiphertextImpl<DCRTPoly>>(ctx.self_reference.lock());
	setHostCt(ct, std::move(res));
	return ct;
}

// As above, but the new value type is copy-constructed from `proto` (carrying its metadata) rather
// than freshly parented — used where a result inherits an existing ciphertext's state.
Ciphertext<DCRTPoly> wrapHostCt(const Ciphertext<DCRTPoly>& proto, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res) {
	Ciphertext<DCRTPoly> ct = std::make_shared<CiphertextImpl<DCRTPoly>>(*proto);
	setHostCt(ct, std::move(res));
	return ct;
}
} // namespace

Ciphertext<DCRTPoly> OpenFheEngine::evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalNegate(ctImpl));
}

void OpenFheEngine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);
	context->EvalNegateInPlace(ctImpl);
}

Ciphertext<DCRTPoly> OpenFheEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	return wrapHostCt(ctx, context->EvalAdd(ct1Impl, ct2Impl));
}

void OpenFheEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	context->EvalAddInPlace(ct1Impl, ct2Impl);
}

Ciphertext<DCRTPoly> OpenFheEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto& ptImpl  = hostPt(pt);
	return wrapHostCt(ctx, context->EvalAdd(ctImpl, ptImpl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalAdd(ctImpl, scalar));
}

void OpenFheEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	auto& ptImpl  = hostPt(pt);
	context->EvalAddInPlace(ct1Impl, ptImpl);
	return;
}

void OpenFheEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	context->EvalAddInPlace(ct1Impl, scalar);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
	}

	auto& context = hostContext(ctx);
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
	ctImpls.reserve(ciphertexts.size());
	for (const auto& ct : ciphertexts) {
		// Null entries pass through as null impls; the OpenFHE EvalAddMany skips them.
		ctImpls.push_back(ct ? hostCt(ct) : lbcrypto::Ciphertext<lbcrypto::DCRTPoly>());
	}
	return wrapHostCt(ctx, context->EvalAddMany(ctImpls));
}

void OpenFheEngine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	auto& context = hostContext(ctx);
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
	ctImpls.reserve(ciphertexts.size());
	for (const auto& ct : ciphertexts) {
		// Null entries pass through as null impls; the OpenFHE EvalAddManyInPlace skips them.
		ctImpls.push_back(ct ? hostCt(ct) : lbcrypto::Ciphertext<lbcrypto::DCRTPoly>());
	}
	context->EvalAddManyInPlace(ctImpls);
	// The result lands in slot 0 (the in-place contract); wrap it if slot 0 was null.
	if (ciphertexts[0] == nullptr) {
		ciphertexts[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(ctx.self_reference.lock());
	}
	setHostCt(ciphertexts[0], ctImpls[0]);
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	return wrapHostCt(ctx, context->EvalSub(ct1Impl, ct2Impl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto& ptImpl  = hostPt(pt);
	return wrapHostCt(ctx, context->EvalSub(ctImpl, ptImpl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto& ptImpl  = hostPt(pt);
	return wrapHostCt(ctx, context->EvalSub(ptImpl, ctImpl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalSub(ctImpl, scalar));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalSub(scalar, ctImpl));
}

void OpenFheEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	context->EvalSubInPlace(ct1Impl, ct2Impl);
	return;
}

void OpenFheEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	context->EvalSubInPlace(ct1Impl, scalar);
	return;
}

void OpenFheEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	context->EvalSubInPlace(scalar, ct1Impl);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	return wrapHostCt(ctx, context->EvalMult(ct1Impl, ct2Impl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	auto& ptImpl  = hostPt(pt);
	return wrapHostCt(ctx, context->EvalMult(ct1Impl, ptImpl));
}

Ciphertext<DCRTPoly> OpenFheEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	return wrapHostCt(ctx, context->EvalMult(ct1Impl, scalar));
}

void OpenFheEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	auto& ptImpl  = hostPt(pt);
	setHostCt(ct1, context->EvalMult(ct1Impl, ptImpl));
	return;
}

void OpenFheEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	context->EvalMultInPlace(ct1Impl, scalar);
	return;
}

void OpenFheEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	ct1->EnsureLazyHostCopy();
	ct2->EnsureLazyHostCopy();
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	context->EvalMultMutableInPlace(ct1Impl, ct2Impl);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalMultNoRelin(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto& context = hostContext(ctx);
	auto& ct1Impl = hostCt(ct1);
	auto& ct2Impl = hostCt(ct2);
	return wrapHostCt(ctx, context->EvalMultNoRelin(ct1Impl, ct2Impl));
}

Ciphertext<DCRTPoly> OpenFheEngine::relinearize(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->Relinearize(ctImpl));
}

void OpenFheEngine::relinearizeInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);
	context->RelinearizeInPlace(ctImpl);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalSquare(ctImpl));
}

void OpenFheEngine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);
	context->EvalSquareInPlace(ctImpl);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	return wrapHostCt(ctx, context->EvalRotate(ctImpl, index));
}

void OpenFheEngine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	setHostCt(ciphertext, context->EvalRotate(ctImpl, index));
	return;
}

Ciphertext<DCRTPoly>
OpenFheEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPoly>>(precomp);
	return wrapHostCt(ctx, context->EvalFastRotation(ctImpl, index, m, casted));
}

Ciphertext<DCRTPoly>
OpenFheEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPoly>>(digits);
	return wrapHostCt(ctx, context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
}

std::vector<Ciphertext<DCRTPoly>> OpenFheEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const uint32_t m,
  const std::shared_ptr<void>& precomp) {
	std::vector<Ciphertext<DCRTPoly>> results;

	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPoly>>(precomp);

	for (const auto& index : indices) {
		results.push_back(wrapHostCt(ct, context->EvalFastRotation(ctImpl, index, m, casted)));
	}
	return results;
}

std::vector<Ciphertext<DCRTPoly>> OpenFheEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const std::shared_ptr<void>& digits,
  bool addFirst) {
	std::vector<Ciphertext<DCRTPoly>> results;

	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPoly>>(digits);

	for (const auto& index : indices) {
		results.push_back(wrapHostCt(ct, context->EvalFastRotationExt(ctImpl, index, casted, addFirst)));
	}
	return results;
}

Ciphertext<DCRTPoly> OpenFheEngine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return wrapHostCt(ctx, context->EvalChebyshevSeries(ctImpl, coeffs, a, b));
}

void OpenFheEngine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);
	setHostCt(ct, context->EvalChebyshevSeries(ctImpl, coeffs, a, b));
	return;
}

namespace {
// api Rescale contract: mod-reduce eagerly under EVERY scaling technique — towers−1, NSD−1
// (guard ≥1), sf ÷= ModReduceFactor — matching the CUDA and haze engines (Ciphertext::rescale,
// HazeEngine::rescaleCore; BITCOMPAT.md O3/O12, resolved on the device semantics). OpenFHE's
// public Rescale only mod-reduces under FIXEDMANUAL and CLONES under the AUTO techniques, so
// the AUTO path goes through the scheme's internal ModReduce — the same call OpenFHE itself
// makes inside EvalMult's operand adjustment, which is what makes a hoisted api Rescale
// equivalent to (and cheaper than) the per-multiply hidden rescales it replaces.
void modReduceOneLevelInPlace(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                              lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct) {
	const auto params =
		std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(context->GetCryptoParameters());
	const auto tech = params->GetScalingTechnique();
	// Parity guard (BITCOMPAT.md O13): a degree-2 ciphertext at the FLEXIBLEAUTOEXT extra
	// level is refused on CUDA and haze; refuse here too rather than diverge on an untested state.
	if (tech == lbcrypto::FLEXIBLEAUTOEXT && ct->GetElements().size() == 3 && ct->GetLevel() == 0) {
		throw std::runtime_error(
			"cpu backend: rescale of a degree-2 ciphertext at the FLEXIBLEAUTOEXT extra level is not supported; relinearize first");
	}
	if (tech == lbcrypto::FIXEDMANUAL) {
		context->ModReduceInPlace(ct); // already a real mod-reduce here; keep OpenFHE's own path
	} else {
		context->GetScheme()->ModReduceInternalInPlace(ct, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
		// HazeEngine::rescaleCore clamps NSD at 1 (CUDA decrements unconditionally); OpenFHE's
		// internal call can leave 0 when rescaling an NSD-1 ciphertext — clamp for parity.
		if (ct->GetNoiseScaleDeg() < 1)
			ct->SetNoiseScaleDeg(1);
	}
}
} // namespace

Ciphertext<DCRTPoly> OpenFheEngine::rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res = ctImpl->Clone();
	modReduceOneLevelInPlace(context, res);
	return wrapHostCt(ctx, std::move(res));
}

void OpenFheEngine::rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) {
	auto& context = hostContext(ctx);
	ciphertext->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ciphertext);
	modReduceOneLevelInPlace(context, ctImpl);
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	auto& ctImpl = hostCt(ct);

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ctImpl);

	lbcrypto::FHECKKSRNS::EvalPartialSumInPlace(result_ct, stride, slots, ACCUMULATE_SUM_RADIX);

	return wrapHostCt(ctx, result_ct);
}

void OpenFheEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);

	lbcrypto::FHECKKSRNS::EvalPartialSumInPlace(ctImpl, stride, slots, ACCUMULATE_SUM_RADIX);
}

void OpenFheEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);

	for (int s = start; s < slots; s <<= 1) {
		int rot_idx = stride * s;
		auto tmp	= context->EvalRotate(ctImpl, rot_idx);
		context->EvalAddInPlace(ctImpl, tmp);
	}

	return;
}

BootstrapSetupPolicy OpenFheEngine::bootstrapSetupPolicy(bool /*precompute*/, bool btsfirstboot, int32_t /*modEvalLevels*/) const {
	// BTSlotsEncoding selects OpenFHE's bootstrap variant: false is the ModRaise-first circuit the
	// CUDA and haze engines run, true is StC-first. It follows the caller's btsfirstboot, so the
	// default setup refreshes to the same level on every backend. (This policy used to pass the
	// mod-eval level count in that position, which is never 0, so the CPU engine always ran
	// StC-first and landed levels away from the device backends.) modevallevels stays at OpenFHE's
	// -1 default and the host precomputation is always built.
	return BootstrapSetupPolicy{ /*precompute=*/true, /*btSlotsEncoding=*/btsfirstboot, /*modEvalLevels=*/-1 };
}

void OpenFheEngine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	if (isContextLoaded()) {
		OPENFHE_THROW("Context is already loaded");
	}
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);

	auto& context = hostContext(ctx);

	// In-context encapsulation: CCParams now maps SPARSE_ENCAPSULATED onto the STOCK lbcrypto
	// SPARSE_ENCAPSULATED enum, so stock's EvalBootstrapKeyGen emits the two switching keys itself
	// — the 2N-4 two-prime (q0,p) sparse-switch key via KeySwitchGenSparse and the 2N-2 ordinary
	// hybrid key — at the correct IN-CONTEXT parameters (ckksrns-fhe.cpp EvalBootstrapKeyGen). They
	// land in the process-global automorphism map at M-4 / M-2 and travel in the existing
	// SerializeEvalAutomorphismKey stream, exactly like the rotation and conjugation keys.
	//
	// This REPLACES the previous FIDESlib::CKKS::AddSparseEncapsulationKeys(skImpl, 32) call, which
	// built a *second* "switchable" CryptoContext and generated dual-context-shaped switching keys
	// — the design the in-context rework removes. That path is no longer needed on either backend: the main key is
	// kept uniform by CryptoContextImpl::KeyGen() (see the security guard there), and the keys stock
	// now emits are the ones the in-context GPU bootstrap consumes.
	context->EvalBootstrapKeyGen(skImpl, slots);
	return;
}

Ciphertext<DCRTPoly>
OpenFheEngine::evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	return wrapHostCt(ctx, context->EvalBootstrap(ctImpl, numIterations, precision));
}

void OpenFheEngine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	ciphertext	  = wrapHostCt(ctx, context->EvalBootstrap(ctImpl, numIterations, precision));
}

Ciphertext<DCRTPoly>
OpenFheEngine::evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	// The reference path, and on this backend it is the only one available: see the header
	// for why the host bootstrap cannot be shortened. The ciphertext and its precision are
	// the same as the device backend produces at the same output level; the WALL is not.
	return Engine::evalBootstrapToLevel(ctx, ciphertext, outputLevel, numIterations, precision, prescaled);
}

void OpenFheEngine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	// CPU backend: the ciphertext is never device-resident, so ct->host already holds the current
	// value. Nothing to read back.
}

namespace {
// Internal giant-step block size. Must match INTERNAL_GSTEP in
// FIDESlib::CKKS::ConvolutionTransform (src/CKKS/LinearTransform.cu) so the keys from
// GetConvolutionTransformRotationIndices (api/HostMath.cpp) cover the rotations used here.
constexpr uint32_t kCpuInternalGStep = 8;

// Tree-reduce v[0..count) into v[0]: pairwise halving when even, linear when odd,
// matching the GPU accumulation order in src/CKKS/LinearTransform.cu.
void cpuTreeAccumulate(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& v, uint32_t count) {
	if (count % 2 == 0) {
		for (uint32_t active = count; active > 1; active /= 2)
			for (uint32_t j = 0; j < active / 2; ++j)
				context->EvalAddInPlace(v[j], v[j + active / 2]);
	} else {
		for (uint32_t j = 1; j < count; ++j)
			context->EvalAddInPlace(v[0], v[j]);
	}
}

// Baby-step/giant-step homomorphic linear transform on CPU via OpenFHE, mirroring
// FIDESlib::CKKS::{Special,}ConvolutionTransform. A non-null `mask` selects the
// "special" variant: each giant-step result is folded with two mask-rotations and
// masked before the intra-block rotation. Returns the transformed (rescaled) ciphertext.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpuConvolutionTransform(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct,
  const std::vector<lbcrypto::Plaintext>& pts,
  const std::vector<int>& indexes,
  int bStep,
  int gStep,
  int stride,
  int rowSize,
  const lbcrypto::Plaintext* mask,
  int maskRotationStride) {
	if (rowSize == 0)
		rowSize = bStep * gStep;
	assert(static_cast<int>(pts.size()) >= rowSize);

	// Match the GPU path: rescale a freshly-multiplied (noise level 2) input.
	if (ct->GetNoiseScaleDeg() == 2)
		ct = context->Rescale(ct);

	// Phase 1: baby-step rotations via hoisted key switching.
	auto precomp = context->EvalFastRotationPrecompute(ct);
	uint32_t m	 = context->GetCyclotomicOrder();
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> fastRotation(bStep);
	for (int i = 0; i < bStep; ++i)
		fastRotation[i] = (indexes[i] == 0) ? std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ct) : context->EvalFastRotation(ct, indexes[i], m, precomp);

	// Phase 2: process blocks of kCpuInternalGStep giant steps.
	uint32_t blockCount = (static_cast<uint32_t>(gStep) + kCpuInternalGStep - 1) / kCpuInternalGStep;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> blockResults;
	blockResults.reserve(blockCount);

	for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
		uint32_t blockStart		   = blockIdx * kCpuInternalGStep;
		uint32_t blockEnd		   = std::min(blockStart + kCpuInternalGStep, static_cast<uint32_t>(gStep));
		uint32_t currentBlockGStep = blockEnd - blockStart;

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> results(currentBlockGStep);
		for (uint32_t j = 0; j < currentBlockGStep; ++j) {
			uint32_t globalJ = blockStart + j;
			// Dot product: results[j] = sum_i fastRotation[i] * pts[bStep*globalJ + i].
			results[j] = context->EvalMult(fastRotation[0], pts[bStep * globalJ]);
			for (int i = 1; i < bStep; ++i) {
				int ptIdx = bStep * static_cast<int>(globalJ) + i;
				if (ptIdx < rowSize)
					context->EvalAddInPlace(results[j], context->EvalMult(fastRotation[i], pts[ptIdx]));
			}

			if (mask != nullptr) {
				// temp = result + rot(result, s) + rot(result, 2s), then temp *= mask.
				auto temp = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(results[j]);
				context->EvalAddInPlace(temp, context->EvalRotate(results[j], maskRotationStride));
				context->EvalAddInPlace(temp, context->EvalRotate(results[j], 2 * maskRotationStride));
				results[j] = context->EvalMult(temp, *mask);
			}

			// Intra-block rotation by stride * (currentBlockGStep - j).
			int rotation = stride * static_cast<int>(currentBlockGStep - j);
			if (rotation != 0)
				results[j] = context->EvalRotate(results[j], rotation);
		}

		cpuTreeAccumulate(context, results, currentBlockGStep);
		blockResults.push_back(results[0]);
	}

	// Phase 3: inter-block rotation and accumulation.
	if (blockCount > 1) {
		int baseRotation = static_cast<int>(kCpuInternalGStep) * stride;
		for (uint32_t blockIdx = 0; blockIdx < blockCount - 1; ++blockIdx) {
			int rotation = static_cast<int>(blockCount - 1 - blockIdx) * baseRotation;
			if (rotation != 0)
				blockResults[blockIdx] = context->EvalRotate(blockResults[blockIdx], rotation);
		}
		cpuTreeAccumulate(context, blockResults, blockCount);
	}

	return context->Rescale(blockResults[0]);
}

// Baby-step/giant-step matrix-vector product on CPU via OpenFHE, mirroring
// FIDESlib::CKKS::LinearTransform (src/CKKS/LinearTransform.cu:63) step for step. The CUDA path
// fuses the whole thing into one hoisted ModUp plus one MAC kernel over every diagonal and limb;
// here the same arithmetic is spelled out as rotations, plaintext mults and adds. Slower by a long
// way, and that is the point: it makes the api portable and gives the CUDA kernel an oracle.
//
// Semantics (see CryptoContextImpl::LinearTransformInPlace for the caller-facing contract):
//   result = sum_{k < rowSize} Rot(ct, k*stride + offset) * D_k
// with pts[k] holding D_k already counter-rotated by
//   -bStep*(k/bStep)*stride - offset.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpuLinearTransform(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct,
  const std::vector<lbcrypto::Plaintext>& pts,
  int rowSize,
  int bStep,
  int stride,
  int offset) {
	assert(bStep > 0 && rowSize > 0);
	assert(static_cast<int>(pts.size()) >= rowSize);
	const int gStep = (rowSize + bStep - 1) / bStep;

	// Match the GPU path: rescale a freshly-multiplied (noise level 2) input, so the diagonals'
	// level requirement is stated against the post-rescale level.
	if (ct->GetNoiseScaleDeg() == 2)
		ct = context->Rescale(ct);

	// Phase 1: the baby steps. One hoisted key-switch precompute (the CUDA path's single ModUp)
	// feeds all bStep rotations; index 0 is the identity and consumes no key.
	auto precomp	 = context->EvalFastRotationPrecompute(ct);
	const uint32_t m = context->GetCyclotomicOrder();
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> fastRotation(static_cast<size_t>(bStep));
	for (int i = 0; i < bStep; ++i)
		fastRotation[static_cast<size_t>(i)] =
		  (i * stride == 0) ? std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ct) : context->EvalFastRotation(ct, i * stride, m, precomp);

	// A giant-step rotation that is a whole number of full slot cycles is the identity; the CUDA
	// path skips it on exactly this condition ((bStep*stride) % (N/2) != 0).
	const int fullCycle			= static_cast<int>(context->GetRingDimension() / 2);
	const bool giantStepIsNoOp	= (bStep * stride) % fullCycle == 0;
	const int giantStepRotation = bStep * stride;

	// Phases 2 and 3: one dot product per giant step, folded backwards (Horner). Walking j down and
	// rotating the running sum by bStep*stride each time is what turns the per-diagonal rotation
	// k*stride into i*stride only — the giant part is paid once per step, not once per diagonal.
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> acc;
	for (int j = gStep - 1; j >= 0; --j) {
		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> inner;
		for (int i = 0; i < bStep; ++i) {
			const int k = bStep * j + i;
			if (k >= rowSize)
				break; // the last giant step is short; CUDA null-pads it, here we stop
			auto term = context->EvalMult(fastRotation[static_cast<size_t>(i)], pts[static_cast<size_t>(k)]);
			if (!inner)
				inner = term;
			else
				context->EvalAddInPlace(inner, term);
		}
		// gStep = ceil(rowSize/bStep) guarantees bStep*j < rowSize, so `inner` is never null.
		if (!acc)
			acc = inner;
		else
			context->EvalAddInPlace(acc, inner);

		if (j > 0) {
			if (!giantStepIsNoOp)
				acc = context->EvalRotate(acc, giantStepRotation);
		} else if (offset != 0) {
			acc = context->EvalRotate(acc, offset);
		}
	}

	// No trailing rescale: the CUDA kernel hands back a noise-scale-degree-2 ciphertext at the input
	// level, and the api contract promises the same on both backends.
	return acc;
}
} // namespace

// `ext` is deliberately unused. It asks for the extended (mod-up, Q||P) basis, which is a property
// of how a device holds its RNS limbs — the win is that the baby rotations and the MAC accumulation
// stay in Q||P and pay ONE ModDown per giant step instead of one per rotation. The CPU reference
// composes OpenFHE ops that each key-switch down to Q on their own, so there is no basis to stay in
// and nothing to hoist; the math and the result (down to the same scale and level) are identical
// either way. MakeCKKSPackedPlaintextExtended likewise hands this backend an ordinary Q-basis
// encoding, so the diagonals here are ordinary plaintexts whatever the caller asked for. The
// parameter exists so the same call, with the same flag, is valid and correct on both backends.
void OpenFheEngine::linearTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool /*ext*/) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	std::vector<lbcrypto::Plaintext> ptImpls;
	ptImpls.reserve(static_cast<size_t>(rowSize));
	for (int k = 0; k < rowSize; ++k)
		ptImpls.push_back(hostPt(diagonals[static_cast<size_t>(k)]));
	setHostCt(ct, cpuLinearTransform(context, ctImpl, ptImpls, rowSize, bStep, stride, offset));
}

void OpenFheEngine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	std::vector<lbcrypto::Plaintext> ptImpls;
	ptImpls.reserve(pts.size());
	for (const auto& pt : pts)
		ptImpls.push_back(hostPt(pt));
	setHostCt(ct, cpuConvolutionTransform(context, ctImpl, ptImpls, indexes, bStep, gStep, stride, rowSize, nullptr, 0));
}

void OpenFheEngine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	auto& context  = hostContext(ctx);
	auto& ctImpl   = hostCt(ct);
	auto& maskImpl = hostPt(mask);
	std::vector<lbcrypto::Plaintext> ptImpls;
	ptImpls.reserve(pts.size());
	for (const auto& pt : pts)
		ptImpls.push_back(hostPt(pt));
	setHostCt(ct, cpuConvolutionTransform(context, ctImpl, ptImpls, indexes, bStep, gStep, stride, rowSize, &maskImpl, maskRotationStride));
}

// Loading objects to a device is a CUDA-only concept; on the CPU backend these are no-ops.
void OpenFheEngine::loadContext(CryptoContextImpl<DCRTPoly>&, const PublicKey<DCRTPoly>&) {
}

void OpenFheEngine::loadPlaintext(CryptoContextImpl<DCRTPoly>&, Plaintext&) {
}

void OpenFheEngine::loadCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
}

std::shared_ptr<void> OpenFheEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {

	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	return context->EvalFastRotationPrecompute(ctImpl);
}

// ---- Ciphertext backend hooks ----
// The CPU backend keeps no device-resident payload, so these operate on the host value via the value
// type's host primitives (the host computation lives there, once).

std::any OpenFheEngine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	return std::any{};
}

size_t OpenFheEngine::ciphertextLevel(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	return ct.GetLevelHost();
}

size_t OpenFheEngine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	return ct.GetNoiseScaleDegHost();
}

double OpenFheEngine::ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	return ct.GetScalingFactorHost();
}

size_t OpenFheEngine::ciphertextSlots(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	return ct.GetSlotsHost();
}

void OpenFheEngine::refreshHostShadow(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&) {
	// CPU backend: the ciphertext is never device-resident, so the host value is already current.
}

void OpenFheEngine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	ct.SetSlotsHost(slots);
}

void OpenFheEngine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>& ct, size_t level) {
	ct.SetLevelHost(level);
}

// ---- Context backend state ----
// The CPU backend is never device-loaded and manages no devices.

bool OpenFheEngine::isContextLoaded() const {
	return false;
}

void OpenFheEngine::synchronize() const {
}

void OpenFheEngine::teardown() {
}

void OpenFheEngine::setDevices(const std::vector<int>&) {
}

std::vector<int> OpenFheEngine::devices() const {
	return {};
}

} // namespace fideslib
