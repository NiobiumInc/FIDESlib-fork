#include "engine/cpu/OpenFheEngine.hpp"

#include "CryptoContext.hpp"
#include "engine/EngineCommon.hpp"

#include <openfhe.h>

#include <algorithm>
#include <any>
#include <cassert>

namespace fideslib {

// Most methods here are thin shims over OpenFHE, which makes OpenFHE the reference oracle: if CPU and
// CUDA disagree on, say, evalMult, the CUDA kernel is the suspect. The exceptions are the ops OpenFHE has
// no equivalent for — convolutionTransform / specialConvolutionTransform, and more simply accumulateSum
// — which re-trace the CUDA algorithm by hand. Their tests compare against a clear-text replay of that
// same algorithm, so they catch a rotation/rescale/encoding slip but not a flaw in the algorithm
// itself. Keep these as thin as possible.

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
		ctImpls.push_back(hostCt(ct));
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
		ctImpls.push_back(hostCt(ct));
	}
	context->EvalAddManyInPlace(ctImpls);
	setHostCt(ciphertexts[0], ctImpls[0]);
	return;
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
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);
	return wrapHostCt(ctx, context->EvalFastRotation(ctImpl, index, m, casted));
}

Ciphertext<DCRTPoly>
OpenFheEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);
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
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);

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
	auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);

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

Ciphertext<DCRTPoly> OpenFheEngine::rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ciphertext);
	return wrapHostCt(ctx, context->Rescale(ctImpl));
}

void OpenFheEngine::rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) {
	auto& context = hostContext(ctx);
	ciphertext->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ciphertext);
	setHostCt(ciphertext, context->Rescale(ctImpl));
	return;
}

Ciphertext<DCRTPoly> OpenFheEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	auto& context = hostContext(ctx);
	auto& ctImpl  = hostCt(ct);

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ctImpl);

	for (int i = 0; i < log2(slots); i++) {
		int rot_idx = stride * (1 << i);
		auto tmp	= context->EvalRotate(result_ct, rot_idx);
		context->EvalAddInPlace(result_ct, tmp);
	}

	return wrapHostCt(ctx, result_ct);
}

void OpenFheEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	auto& context = hostContext(ctx);
	ct->EnsureLazyHostCopy();
	auto& ctImpl = hostCt(ct);

	for (int i = 0; i < log2(slots); i++) {
		int rot_idx = stride * (1 << i);
		auto tmp	= context->EvalRotate(ctImpl, rot_idx);
		context->EvalAddInPlace(ctImpl, tmp);
	}

	return;
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

BootstrapSetupPolicy OpenFheEngine::bootstrapSetupPolicy(bool /*precompute*/, bool /*btsfirstboot*/, int32_t modEvalLevels) const {
	// Behaviour preserved verbatim from the previous CPU evalBootstrapSetup, which called the 6-arg
	// EvalBootstrapSetup(levelBudget, dim1, slots, correctionFactor, /*precompute=*/true, modall).
	// NOTE (pre-existing, flagged — not a behavior change in this refactor): in OpenFHE's single
	// signature (..., precompute, BTSlotsEncoding, modevallevels=-1), that 6th `modall` argument lands
	// in BTSlotsEncoding, NOT modevallevels — so the CPU "configure mod-eval levels" intent is not
	// actually in effect (modevallevels stays -1). Reproduced exactly here; needs a separate decision.
	return BootstrapSetupPolicy{ /*precompute=*/true, /*btSlotsEncoding=*/modEvalLevels != 0, /*modEvalLevels=*/-1 };
}

void OpenFheEngine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	if (isContextLoaded()) {
		OPENFHE_THROW("Context is already loaded");
	}
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);

	auto& context = hostContext(ctx);
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
} // namespace

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
