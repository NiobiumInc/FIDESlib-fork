#ifdef FIDESLIB_ENABLE_CUDA

#include "engine/cuda/CudaEngine.hpp"

#include "CryptoContext.hpp"
#include "engine/EngineCommon.hpp"

// GPU CKKS implementation headers for the migrated device-side ops.
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/PreparedLT.cuh"
#include "CKKS/RNSPoly.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CudaUtils.cuh"

#include <algorithm>
#include <any>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <openfhe.h> // OPENFHE_THROW in shared precondition checks
#include <string>

namespace fideslib {

namespace {
// The neutral value types store the GPU-resident payload as a shared_ptr inside their `device` slot
// (the device object is not copyable). These unwrap it; an empty slot means "not resident", which the
// engine ensures via LoadCiphertext/LoadPlaintext before every use.
std::shared_ptr<FIDESlib::CKKS::Ciphertext> deviceCt(const CiphertextImpl<DCRTPoly>& ct) {
	return std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
}

std::shared_ptr<FIDESlib::CKKS::Ciphertext> deviceCt(const Ciphertext<DCRTPoly>& ct) {
	return deviceCt(*ct);
}

std::shared_ptr<FIDESlib::CKKS::Plaintext> devicePt(const Plaintext& pt) {
	return std::any_cast<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(pt->device);
}

/// @brief Widest EvalRotateMany batch the CUDA engine issues in one launch (0 = never batch).
///
/// A batch holds 2*B ciphertexts and B digit decompositions on the device at once, and the kernel
/// spends the block's z dimension on the batch (block = {x, 2, B}), so B is bounded by the block
/// size as well as by memory. 32 is well inside both at ring 2^16; the environment override exists
/// so the GPU gate can sweep it and so 0 can switch the batched path off without a rebuild.
size_t rotateManyMaxBatch() {
	static const size_t cap = [] {
		if (const char* v = std::getenv("FIDESLIB_ROTATE_MANY_BATCH")) {
			const long n = std::strtol(v, nullptr, 10);
			if (n >= 0 && n <= 64)
				return static_cast<size_t>(n);
		}
		return static_cast<size_t>(32);
	}();
	return cap;
}

/// @brief What EvalFastRotationPrecompute hands back on CUDA: the ModUp of the source's c1.
///
/// A key switch is (1) ModUp -- the digit decomposition of c1 and its NTTs -- and (2) the per-index
/// inner product with the rotation key plus the automorphism. Step (1) depends on the CIPHERTEXT
/// only, so many rotations of ONE source can share it; that is the whole content of the hoisting
/// contract, and it is what FIDESlib::CKKS::Ciphertext::rotate_hoisted already exploits for a set of
/// indexes known up front (src/CKKS/Ciphertext.cpp:1138). Until now this engine returned nullptr
/// here and the single-index evalFastRotation below fell through to Ciphertext::rotate, which redoes
/// the ModUp into the context-wide scratch on EVERY call -- so a caller following the
/// precompute-once/rotate-lazily contract (the shape OpenFHE's CPU path really hoists) paid a full
/// key switch per index and the hoisting was a no-op on this backend.
///
/// The polynomial is OURS, not ContextData::getKeySwitchAux(): that scratch is overwritten by the
/// next rotate/conjugate/relinearize on the context, and a handle is expected to outlive them. It is
/// prepared exactly as getKeySwitchAux prepares the shared one (Context.cu:377) and released by RAII
/// -- ~RNSPoly -> ~LimbPartition returns every buffer to the GPUmalloc pool -- when the last
/// shared_ptr to this handle dies.
struct FastRotationPrecompute {
	FIDESlib::CKKS::RNSPoly modup; ///< ModUp of src.c1; read-only per index, so it is reusable.
	int32_t level;				   ///< c1's level when the digits were built (staleness check).
	FIDESlib::CKKS::KeyHash keyID; ///< key tag of the source (diagnostic cross-check).
	int slots;					   ///< slot count of the source (diagnostic cross-check).

	FastRotationPrecompute(FIDESlib::CKKS::ContextData& cc, FIDESlib::CKKS::Ciphertext& src)
		: modup(cc, cc.L, false), level(src.c1.getLevel()), keyID(src.keyID), slots(src.slots) {
		// Same preparation ContextData::getKeySwitchAux() does before handing out its scratch.
		modup.generateDecompAndDigit(false);
		modup.generateSpecialLimbs(false, false);
		modup.setLevel(level); // modupInto asserts the levels match

		// Line for line rotate_hoisted's ModUp (Ciphertext.cpp:1251-1258).
		if (cc.GPUid.size() == 1) {
			src.c1.modupInto(modup);
		} else {
			modup.copy(src.c1);
			modup.modup();
		}
	}
};

/// @brief The handle, or nullptr for a caller that passed none (every pre-existing caller).
FastRotationPrecompute* fastRotationPrecompute(const std::shared_ptr<void>& precomp) {
	return static_cast<FastRotationPrecompute*>(precomp.get());
}

/// @brief Refuse a handle taken from a DIFFERENT ciphertext. The level check that catches a stale
/// handle (the source was rescaled since) lives in Ciphertext::rotate_precomputed; these two are the
/// cheap cross-checks the handle itself can make.
void requireMatchingPrecompute(const FastRotationPrecompute& pre, const FIDESlib::CKKS::Ciphertext& ct, const char* op) {
	if (pre.keyID != ct.keyID || pre.slots != ct.slots) {
		OPENFHE_THROW(std::string(op) + ": the fast-rotation precompute was taken from a ciphertext with a different key tag or slot count");
	}
}

/// @brief Build the value CiphertextImpl's copy constructor would build, minus the device deep copy.
///
/// The copy constructor delegates device cloning to CloneCiphertextBackend, which duplicates the
/// entire GPU payload (~29 MB at level 13). When the caller is about to install a freshly computed
/// payload, that clone is written and then thrown away. This reproduces the host half of the copy
/// constructor exactly — the host shadow shared_ptr is shared with the source and need_lazy_copy is
/// set, so the first host mutation still detaches — and leaves `device` empty for the caller to fill.
Ciphertext<DCRTPoly> makeResultShell(const CiphertextImpl<DCRTPoly>& src) {
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>{ src.parent_context });
	result->host				= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(src.host));
	result->need_lazy_copy		= true;
	return result;
}
} // namespace

// CKKS prime tables for the GPU context parameters (used by loadContext).
static std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
	{ .p = 2251799661248513 },
	{ .p = 2251799661641729 },
	{ .p = 2251799665180673 },
	{ .p = 2251799682088961 },
	{ .p = 2251799678943233 },
	{ .p = 2251799717609473 },
	{ .p = 2251799710138369 },
	{ .p = 2251799708827649 },
	{ .p = 2251799707385857 },
	{ .p = 2251799713677313 },
	{ .p = 2251799712366593 },
	{ .p = 2251799716691969 },
	{ .p = 2251799714856961 },
	{ .p = 2251799726522369 },
	{ .p = 2251799726129153 },
	{ .p = 2251799747493889 },
	{ .p = 2251799741857793 },
	{ .p = 2251799740416001 },
	{ .p = 2251799746707457 },
	{ .p = 2251799756013569 },
	{ .p = 2251799775805441 },
	{ .p = 2251799763091457 },
	{ .p = 2251799767154689 },
	{ .p = 2251799765975041 },
	{ .p = 2251799770562561 },
	{ .p = 2251799769776129 },
	{ .p = 2251799772266497 },
	{ .p = 2251799775281153 },
	{ .p = 2251799774887937 },
	{ .p = 2251799797432321 },
	{ .p = 2251799787995137 },
	{ .p = 2251799787601921 },
	{ .p = 2251799791403009 },
	{ .p = 2251799789568001 },
	{ .p = 2251799795466241 },
	{ .p = 2251799807131649 },
	{ .p = 2251799806345217 },
	{ .p = 2251799805165569 },
	{ .p = 2251799813554177 },
	{ .p = 2251799809884161 },
	{ .p = 2251799810670593 },
	{ .p = 2251799818928129 },
	{ .p = 2251799816568833 },
	{ .p = 2251799815520257 } };

static std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
	{ .p = 2305843009220116481 },
	{ .p = 2305843009221820417 },
	{ .p = 2305843009224179713 },
	{ .p = 2305843009225228289 },
	{ .p = 2305843009227980801 },
	{ .p = 2305843009229160449 },
	{ .p = 2305843009229946881 },
	{ .p = 2305843009231650817 },
	{ .p = 2305843009235189761 },
	{ .p = 2305843009240301569 },
	{ .p = 2305843009242923009 },
	{ .p = 2305843009244889089 },
	{ .p = 2305843009245413377 },
	{ .p = 2305843009247641601 } };

void CudaEngine::negateExact(FIDESlib::CKKS::Ciphertext& ct_gpu) const {
	// Exact negation: multiply each limb by q_i - 1 (== -1 mod q_i). Unlike multScalar(-1.0) this
	// leaves the noise degree and scaling factor untouched, matching OpenFHE's EvalNegate.
	std::vector<uint64_t> negOne(ct_gpu.getLevel() + 1);
	for (size_t i = 0; i < negOne.size(); ++i)
		negOne[i] = (*context_)->prime[i].p - 1;
	// OpenFHE's EvalNegate negates EVERY component, so a degree-2 ciphertext's tail is negated too.
	ct_gpu.c0.multScalar(negOne);
	ct_gpu.c1.multScalar(negOne);
	if (ct_gpu.c2.has_value())
		ct_gpu.c2->multScalar(negOne);
}

void CudaEngine::requireRelinKey(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct, const char* op) const {
	auto& context		   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	const std::string& tag = deviceCt(ct)->keyID;
	const auto& keyMap	   = context->GetAllEvalMultKeys();
	if (keyMap.find(tag) == keyMap.end()) {
		OPENFHE_THROW(std::string(op) + ": no EvalMultKey is registered for key tag \"" + tag + "\"; call EvalMultKeyGen before LoadContext");
	}
}

Ciphertext<DCRTPoly> CudaEngine::evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	negateExact(*deviceCt(result));
	return result;
}

void CudaEngine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(ct);
	negateExact(*deviceCt(ct));
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->add(*ct2_gpu);
	return result;
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->add(*ct2_gpu);
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->addPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->addScalar(scalar);
	return result;
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadPlaintext(pt);
	auto res_gpu = deviceCt(ct1);
	auto pt_gpu	 = devicePt(pt);
	res_gpu->addPt(*pt_gpu);
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->addScalar(scalar);
}

Ciphertext<DCRTPoly> CudaEngine::evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
	}

	// Null entries are skipped, mirroring the CPU implementation.
	size_t first = 0;
	while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
		++first;
	}
	if (first == ciphertexts.size()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector has no non-null entries");
	}
	size_t second = first + 1;
	while (second < ciphertexts.size() && ciphertexts[second] == nullptr) {
		++second;
	}

	for (const auto& ct : ciphertexts) {
		if (ct != nullptr) {
			ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
		}
	}

	if (second == ciphertexts.size()) {
		// Single non-null entry: independent copy, like the CPU implementation's clone.
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertexts[first]);
	}

	// Left fold, same association as the CPU EvalAddMany: one fresh result, then in-place
	// accumulation. The old pairwise tree associated differently and was not bit-compatible.
	Ciphertext<DCRTPoly> result = ctx.EvalAdd(ciphertexts[first], ciphertexts[second]);
	for (size_t i = second + 1; i < ciphertexts.size(); ++i) {
		if (ciphertexts[i] != nullptr) {
			ctx.EvalAddInPlace(result, ciphertexts[i]);
		}
	}

	return result;
}

void CudaEngine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	for (const auto& ct : ciphertexts) {
		if (ct != nullptr) {
			ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
		}
	}

	// Serial in-place fold into slot 0 (null entries are skipped), matching the CPU
	// EvalAddManyInPlace association.
	size_t first = 0;
	while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
		++first;
	}
	if (first == ciphertexts.size()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector has no non-null entries");
	}
	for (size_t i = first + 1; i < ciphertexts.size(); ++i) {
		if (ciphertexts[i] != nullptr) {
			ctx.EvalAddInPlace(ciphertexts[first], ciphertexts[i]);
		}
	}
	if (first != 0) {
		ciphertexts[0] = ciphertexts[first];
	}
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->sub(*ct2_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->subPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->multScalar(-1.0);
	res_gpu->addPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->addScalar(-scalar);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	// scalar − ct is EvalAdd(EvalNegate(ct), scalar) (OpenFHE cryptocontext.h:1835-1837): negate
	// once, then add. The second multScalar(-1.0) this replaces undid the first one, so the whole
	// sequence computed ct − scalar; it also bumped the noise degree twice and, on a depth-2
	// operand, spent a level on the multScalar precheck's rescale.
	negateExact(*res_gpu);
	res_gpu->addScalar(scalar);
	return result;
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->sub(*ct2_gpu);
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->addScalar(-scalar);
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	// Same double-negate defect as the out-of-place overload above.
	negateExact(*res_gpu);
	res_gpu->addScalar(scalar);
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->mult(*ct2_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadPlaintext(pt);
	// The result's device payload is allocated fresh (same aux-poly path cloneCiphertextBackend
	// uses to construct) and written by the multiply, instead of being deep-copied from ct1 and
	// then overwritten byte for byte. Metadata and host-shadow semantics are unchanged: multPt's
	// out-of-place form merges the same metadata, and makeResultShell reproduces the host half of
	// the copy constructor.
	Ciphertext<DCRTPoly> result = makeResultShell(*ct1);
	auto ct1_gpu				= deviceCt(ct1);
	auto pt_gpu					= devicePt(pt);
	auto res_gpu				= std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	res_gpu->multPt(*ct1_gpu, *pt_gpu);
	result->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(res_gpu));
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	res_gpu->multScalar(scalar);
	return result;
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadPlaintext(pt);
	auto res_gpu = deviceCt(ct1);
	auto pt_gpu	 = devicePt(pt);
	res_gpu->multPt(*pt_gpu);
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->multScalar(scalar);
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(ct2);
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->mult(*ct2_gpu);
}

Ciphertext<DCRTPoly> CudaEngine::evalMultNoRelin(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	// Deliberately no relin-key guard: OpenFHE's EvalMultNoRelin is a pure tensor product.
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->multNoRelin(*ct2_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::relinearize(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	requireRelinKey(ctx, *ct, "Relinearize");
	// OpenFHE's Relinearize is Clone() + RelinearizeInPlace, so a degree-1 input still yields an
	// INDEPENDENT ciphertext; the CiphertextImpl copy deep-clones the device payload.
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	deviceCt(result)->relinearize();
	return result;
}

void CudaEngine::relinearizeInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(ct);
	requireRelinKey(ctx, *ct, "RelinearizeInPlace");
	deviceCt(ct)->relinearize();
}

Ciphertext<DCRTPoly> CudaEngine::evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->square();
	return result;
}

void CudaEngine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	ct_gpu->square();
}

Ciphertext<DCRTPoly> CudaEngine::evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	res_gpu->rotate(index);
	return result;
}

void CudaEngine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	ctx.LoadCiphertext(ciphertext);
	auto ct_gpu = deviceCt(ciphertext);
	ct_gpu->rotate(index);
}

Ciphertext<DCRTPoly>
CudaEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto ct_gpu					= deviceCt(ct);
	res_gpu->copy(*ct_gpu);

	// WITH a handle this is a real hoisted rotation: the ModUp was paid once, at
	// evalFastRotationPrecompute, and every index since has reused it. WITHOUT one -- a caller from
	// before this engine had a precompute, or one that deliberately passed nullptr -- the behaviour is
	// exactly what it was: a full key switch per call.
	if (auto* pre = fastRotationPrecompute(precomp)) {
		requireMatchingPrecompute(*pre, *ct_gpu, "EvalFastRotation");
		res_gpu->rotate_precomputed(*ct_gpu, pre->modup, (int)index, /*ext=*/false);
	} else {
		res_gpu->rotate((int)index, true);
	}

	return result;
}

Ciphertext<DCRTPoly>
CudaEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto ct_gpu					= deviceCt(ct);
	// ct_gpu->rotate((int)index, false);
	res_gpu->copy(*ct_gpu);

	// Same as evalFastRotation above, minus the ModDown: `digits` IS the precompute in this
	// signature, and rotate's moddown flag is rotate_precomputed's `ext` inverted.
	if (auto* pre = fastRotationPrecompute(digits)) {
		requireMatchingPrecompute(*pre, *ct_gpu, "EvalFastRotationExt");
		res_gpu->rotate_precomputed(*ct_gpu, pre->modup, (int)index, /*ext=*/true);
	} else {
		res_gpu->rotate((int)index, false);
	}

	return result;
}

std::vector<Ciphertext<DCRTPoly>> CudaEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const uint32_t m,
  const std::shared_ptr<void>& precomp) {
	std::vector<Ciphertext<DCRTPoly>> results;

	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	auto ct_gpu = deviceCt(ct);

	// Create result ciphertexts.
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = deviceCt(result);
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, false);
	return results;
}

std::vector<Ciphertext<DCRTPoly>>
CudaEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {
	std::vector<Ciphertext<DCRTPoly>> results;

	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	auto ct_gpu = deviceCt(ct);

	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = deviceCt(result);
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, true);
	return results;
}

// ---------------------------------------------------------------------------
// EvalRotateMany. One index, B independent sources.
//
// The batched key switch is Ciphertext::rotate_many (src/CKKS/Ciphertext.cpp); everything here is
// eligibility and chunking. The api contract is UNCONDITIONAL byte-equality with EvalRotate, so
// every case the batched kernel does not cover falls back to Engine::evalRotateMany -- the
// evalRotate loop, which is the definition of the answer:
//   * B == 1: nothing to share.
//   * mixed levels / slot counts / key tags / ModUp states: the kernel reads ONE limb layout and
//     ONE digit count for the whole batch (LimbPartitionBatch.cu takes them off out[0] and in[1]).
//   * index 0: the identity, and there is no key for it.
//   * multi-GPU, or hoistRotateFused off: those are the paths rotate()/rotate_hoisted() take
//     without the fused kernel this batches; they are not reached in this deployment and a silent
//     divergence there would be worse than the serial cost.
// FIDESLIB_ROTATE_MANY_BATCH caps the batch width (device memory: 2*B ciphertexts plus B digit
// decompositions live at once); 0 turns the batched path off entirely and is the kill switch if
// the GPU gate finds anything wrong with it.
// ---------------------------------------------------------------------------
std::vector<Ciphertext<DCRTPoly>> CudaEngine::evalRotateMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& cts, const int32_t index) {
	std::vector<Ciphertext<DCRTPoly>> results;
	const size_t B = cts.size();
	if (B == 0)
		return results;

	// Residency first: eligibility reads device-side level and key metadata.
	for (const auto& ct : cts)
		ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	const size_t maxBatch = rotateManyMaxBatch();
	auto first			  = deviceCt(cts[0]);

	bool eligible = maxBatch > 0 && B > 1 && FIDESlib::CKKS::hoistRotateFused && first->cc.GPUid.size() == 1 && first->normalyzeIndex(static_cast<int>(index)) != 0;
	for (size_t i = 0; eligible && i < B; ++i) {
		auto g = deviceCt(cts[i]);
		eligible = g->c0.getLevel() == first->c0.getLevel() && g->c1.getLevel() == first->c1.getLevel() && g->slots == first->slots && g->keyID == first->keyID &&
				   g->c0.isModUp() == first->c0.isModUp() && g->c1.isModUp() == first->c1.isModUp();
	}
	if (!eligible)
		return Engine::evalRotateMany(ctx, cts, index);

	results.reserve(B);
	for (size_t off = 0; off < B; off += maxBatch) {
		const size_t n = std::min(maxBatch, B - off);
		// The shared_ptrs are held for the duration of the call: rotate_many takes raw pointers,
		// and the device payloads must outlive it.
		std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> src_gpu, dst_gpu;
		std::vector<FIDESlib::CKKS::Ciphertext*> srcs, dsts;
		src_gpu.reserve(n);
		dst_gpu.reserve(n);
		srcs.reserve(n);
		dsts.reserve(n);
		for (size_t i = 0; i < n; ++i) {
			// Same clone evalRotate makes: a distinct device ciphertext at the source's level,
			// which is exactly the destination precondition rotate_many states.
			Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*cts[off + i]);
			src_gpu.push_back(deviceCt(cts[off + i]));
			dst_gpu.push_back(deviceCt(result));
			srcs.push_back(src_gpu.back().get());
			dsts.push_back(dst_gpu.back().get());
			results.push_back(std::move(result));
		}
		FIDESlib::CKKS::Ciphertext::rotate_many(srcs, dsts, static_cast<int>(index));
	}
	return results;
}

Ciphertext<DCRTPoly> CudaEngine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
	return result;
}

void CudaEngine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
}

Ciphertext<DCRTPoly> CudaEngine::rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	res_gpu->rescale();
	return result;
}

void CudaEngine::rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) {
	ctx.LoadCiphertext(ciphertext);
	auto res_gpu = deviceCt(ciphertext);
	res_gpu->rescale();
}

Ciphertext<DCRTPoly> CudaEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);

	// Accumulate rewrites slots as it folds; restore the input value so the result keeps the
	// caller's packing, matching what the CPU path leaves behind.
	const int inputSlots = res_gpu->slots;
	FIDESlib::CKKS::Accumulate(*res_gpu, ACCUMULATE_SUM_RADIX, stride, slots);
	res_gpu->slots = inputSlots;

	return result;
}

void CudaEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);

	const int inputSlots = res_gpu->slots;
	FIDESlib::CKKS::Accumulate(*res_gpu, ACCUMULATE_SUM_RADIX, stride, slots);
	res_gpu->slots = inputSlots;
}

void CudaEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);
	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots, start);
}

BootstrapSetupPolicy CudaEngine::bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const {
	// The GPU path forwards the caller's precompute/firstboot and the mod-eval levels unchanged
	// (matching the previous 7-arg EvalBootstrapSetup(..., precompute, btsfirstboot, modall) call).
	return BootstrapSetupPolicy{ precompute, btsfirstboot, modEvalLevels };
}

void CudaEngine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	if (isContextLoaded()) {
		OPENFHE_THROW("Context is already loaded");
	}
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);

	ctx.slots_bootstrap.push_back(slots);
	FIDESlib::CKKS::GenBootstrapKeys(skImpl, slots, ctx.keyDist == fideslib::SPARSE_ENCAPSULATED);
}

// THE HOST-SCHEME LOCK. Serializes every piece of lbcrypto HOST work this engine performs, and is
// TAKEN ONLY in the concurrent mode (FIDESlib::ConcurrentOps; the default mode keeps the unlocked
// path it has always had).
//
// WHY. A device backend's ops do not touch the host scheme -- that is the point of it. Two of
// them do, because they marshal:
//
//   * the host-to-device direction, loadPlaintext / loadCiphertext (GetRawPlainText /
//     GetRawCipherText). ALREADY serialized, by g_deviceLoadLock below.
//   * the device-to-host direction, refreshHostShadow (GetOpenFHECipherText), and the host glue of
//     the TWO-ITERATION bootstrap, which reads the ciphertext back and runs its mod-reduce,
//     integer scale-up, error subtraction and recombination through lbcrypto so the scale/level
//     bookkeeping is the reference's own. NOT serialized until now.
//
// Both directions run the same machinery: DCRTPoly SetFormat, i.e. lbcrypto's NTT. OpenFHE builds
// its twiddle/root tables LAZILY into process-wide static maps keyed by modulus
// (ChineseRemainderTransformFTT), and inserts into them from whichever thread reaches an
// unprepared modulus first. Several host threads there at once is a tree-corrupting race, not a
// lost cache entry -- and it corrupts the VALUES a ciphertext marshals as, silently.
//
// That asymmetry is exactly what an application sees as "concurrent matmuls are fine, concurrent
// bootstraps corrupt": a matmul's only host work is the load direction, which was locked; a
// two-iteration bootstrap is the one op that comes BACK to the host mid-circuit.
//
// RECURSIVE, deliberately: the Meta-BTS glue calls refreshHostShadow, which takes this same lock.
// The alternative -- an unlocked inner helper -- is one more way for a future caller to reach the
// host scheme unguarded, and the lock is uncontended in the only mode that takes it at all.
//
// The lock is DROPPED around the two nested single-iteration bootstraps, which is where all the
// time is: those are pure GPU work and still overlap across threads. What serializes is only the
// cheap host arithmetic between them.
static std::recursive_mutex g_hostSchemeLock;

Ciphertext<DCRTPoly>
CudaEngine::evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	return evalBootstrapInternal(ctx, ciphertext, numIterations, precision, prescaled, /*levelsToDrop=*/0);
}

// META-BTS ON THE DEVICE (the default since 2026-10-04; FIDESLIB_METABTS_DEVICE=0 is the kill switch, see MetaBtsDevice).
//
// The same ten steps as the host composition below (ckksrns-fhe.cpp EvalBootstrap,
// numIterations == 2), step for step, with every glue op on the device ciphertexts: the exact
// integer scale-up is multIntScalar (the op the bootstrap's own correction factor uses), the
// error subtraction is Ciphertext::sub, whose FLEXIBLEAUTO adjustment is the fork's transcription
// of AdjustLevelsAndDepthInPlace, and the final scale-down is multScalar, which is EvalMult by a
// constant. Nothing is read back: the host path's three refreshHostShadow readbacks (each a
// Ciphertext::store with two cudaDeviceSynchronize and a full device-to-host copy), the lbcrypto
// arithmetic on the issuing thread, and the re-upload of the host result on the next op are gone.
// The input is left intact (its device payload is copied before the first bootstrap), which the
// reference also guarantees.
Ciphertext<DCRTPoly>
CudaEngine::evalBootstrapMetaDevice(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t precision, bool prescaled, int levelsToDrop) {
	const bool trace = std::getenv("FIDESLIB_METABTS_TRACE") != nullptr;
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	auto in_gpu			   = deviceCt(ciphertext);
	const auto technique   = in_gpu->cc.rescaleTechnique;
	const uint64_t pow2	   = uint64_t{ 1 } << precision;
	const int inLevel	   = in_gpu->getLevel(); // FIDESlib level = towers - 1 (reversed against lbcrypto)

	// Step 3: the initial bootstrap, on a copy (the input is step 2's operand).
	auto b1 = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	b1->copy(*in_gpu);
	if (trace) fprintf(stderr, "[metabts] device glue: boot1 (input towers %d)\n", inLevel + 1);
	FIDESlib::CKKS::Bootstrap(*b1, b1->slots, prescaled, levelsToDrop);
	if (b1->NoiseLevel == 2)
		b1->rescale(); // ModReduceInternalInPlace when the noise degree is 2
	if (trace) fprintf(stderr, "[metabts] device glue: boot1 back (towers %d, nsd %d)\n", b1->getLevel() + 1, b1->NoiseLevel);

	if (b1->getLevel() <= inLevel) {
		// Bootstrapping gains no towers here -- the reference returns the input.
		Ciphertext<DCRTPoly> out = makeResultShell(*ciphertext);
		auto same				 = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
		same->copy(*in_gpu);
		out->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(same));
		return out;
	}

	// Step 4: scale the bootstrapped ciphertext up by 2^precision (exact; scale metadata unchanged).
	FIDESlib::CKKS::multIntScalar(*b1, pow2);

	// Step 2: scale the ORIGINAL input up by 2^precision.
	auto up = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	up->copy(*in_gpu);
	if (technique == FIDESlib::CKKS::FIXEDMANUAL && up->NoiseLevel > 1)
		up->rescale();
	FIDESlib::CKKS::multIntScalar(*up, pow2);

	// Steps 5-7: the bootstrapping error. Under FIXEDAUTO the reference drops the bootstrapped
	// ciphertext to the input's towers first; under the FLEXIBLE techniques the subtraction's own
	// level-and-scale adjustment brings it down (AdjustLevelsAndDepthInPlace).
	auto err = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	err->copy(*b1);
	if (technique == FIDESlib::CKKS::FIXEDAUTO)
		err->dropToLevel(inLevel);
	err->sub(*up);

	// Step 8: bootstrap the error.
	if (trace) fprintf(stderr, "[metabts] device glue: boot2 (err towers %d)\n", err->getLevel() + 1);
	FIDESlib::CKKS::Bootstrap(*err, err->slots, /*prescaled=*/false, levelsToDrop);
	if (err->NoiseLevel == 2)
		err->rescale();

	// Steps 9-10: subtract the bootstrapped error, scale back down.
	if (trace) fprintf(stderr, "[metabts] device glue: recombine (ct1 towers %d | ct2 towers %d)\n", b1->getLevel() + 1, err->getLevel() + 1);
	b1->sub(*err);
	b1->multScalar(1.0 / static_cast<double>(pow2));

	Ciphertext<DCRTPoly> out = makeResultShell(*ciphertext);
	out->device				 = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(b1));
	return out;
}

Ciphertext<DCRTPoly>
CudaEngine::evalBootstrapInternal(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled, int levelsToDrop) {
	if (numIterations > 1) {
		// Meta-BTS (iterative bootstrapping): previously numIterations/precision
		// were silently DROPPED here (BITCOMPAT "fails silently" class), so
		// precision-critical callers got a single-iteration refresh. Compose the
		// two-iteration algorithm exactly as OpenFHE's reference
		// (ckksrns-fhe.cpp EvalBootstrap, numIterations == 2): the two heavy
		// bootstraps run on the GPU; the glue arithmetic (integer scale-up,
		// error subtraction, recombination) runs on the HOST via lbcrypto so the
		// scale/level bookkeeping is the reference's own. Glue ops are cheap
		// (no key material); cost = two GPU bootstraps + ~4 host<->device
		// ciphertext transfers.
		if (numIterations != 2)
			OPENFHE_THROW("CUDA backend: CKKS bootstrapping supports 1 or 2 iterations.");
		if (FIDESlib::MetaBtsDevice())
			return evalBootstrapMetaDevice(ctx, ciphertext, precision, prescaled, levelsToDrop);
		auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
		const auto cryptoParams =
			std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
		const auto st					 = cryptoParams->GetScalingTechnique();
		const uint32_t powerOfTwoModulus = 1u << precision;
		const uint32_t compositeDegree	 = cryptoParams->GetCompositeDegree();
		const uint32_t L0				 = cryptoParams->GetElementParams()->GetParams().size();

		// See g_hostSchemeLock: held across every host-glue segment below, dropped around the two
		// nested GPU bootstraps. `defer_lock` + the ConcurrentOps test keeps the default mode on
		// exactly the code it had.
		std::unique_lock<std::recursive_mutex> host_lk(g_hostSchemeLock, std::defer_lock);
		const bool lock_host = FIDESlib::ConcurrentOps();
		if (lock_host)
			host_lk.lock();

		// Host view of the input (readback if device-resident).
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] input readback\n");
		refreshHostShadow(ctx, *ciphertext);
		auto ctIn			 = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->host);
		const auto initSizeQ = ctIn->GetElements()[0].GetNumOfElements();

		// Step 3 (reference numbering): initial GPU bootstrap.
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] boot1 (input level %zu, towers %zu)\n", (size_t)ctIn->GetLevel(), (size_t)initSizeQ);
		if (lock_host)
			host_lk.unlock();
		Ciphertext<DCRTPoly> boot1 = evalBootstrapInternal(ctx, ciphertext, 1, 0, prescaled, levelsToDrop);
		if (lock_host)
			host_lk.lock();
		refreshHostShadow(ctx, *boot1);
		auto ct1 = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(boot1->host);
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] boot1 back: level %zu towers %zu nsd %zu sf %.6g\n", (size_t)ct1->GetLevel(), (size_t)ct1->GetElements()[0].GetNumOfElements(), (size_t)ct1->GetNoiseScaleDeg(), ct1->GetScalingFactor());
		if (ct1->GetNoiseScaleDeg() > 1)
			context->GetScheme()->ModReduceInternalInPlace(ct1, compositeDegree);
		// Step 4: scale up by 2^precision (exact integer multiply, scale metadata unchanged).
		context->GetScheme()->MultByIntegerInPlace(ct1, powerOfTwoModulus);

		const auto bootstrappingSizeQ = ct1->GetElements()[0].GetNumOfElements();
		if (bootstrappingSizeQ <= initSizeQ) {
			// Bootstrapping gains no towers here — reference returns the input.
			Ciphertext<DCRTPoly> out = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
			out->host				 = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ctIn->Clone());
			out->device				 = std::any{};
			return out;
		}

		// Step 2: scale the ORIGINAL input up by 2^precision.
		auto ctScaledUp = ctIn->Clone();
		if (st == lbcrypto::FIXEDMANUAL && ctScaledUp->GetNoiseScaleDeg() > 1)
			context->GetScheme()->ModReduceInternalInPlace(ctScaledUp, ctScaledUp->GetNoiseScaleDeg() - 1);
		context->GetScheme()->MultByIntegerInPlace(ctScaledUp, powerOfTwoModulus);

		// Step 5: bring the bootstrapped ciphertext down to the input's modulus.
		auto ctBootstrappedScaledDown = ct1->Clone();
		if (st == lbcrypto::FIXEDAUTO) {
			for (auto& cv : ctBootstrappedScaledDown->GetElements())
				cv.DropLastElements(bootstrappingSizeQ - initSizeQ);
			ctBootstrappedScaledDown->SetLevel(L0 - ctBootstrappedScaledDown->GetElements()[0].GetNumOfElements());
		}

		// Steps 6-7: the bootstrapping error (mod-down to q implicit).
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] sub: down towers %zu nsd %zu | scaledUp towers %zu nsd %zu\n", (size_t)ctBootstrappedScaledDown->GetElements()[0].GetNumOfElements(), (size_t)ctBootstrappedScaledDown->GetNoiseScaleDeg(), (size_t)ctScaledUp->GetElements()[0].GetNumOfElements(), (size_t)ctScaledUp->GetNoiseScaleDeg());
		auto ctBootstrappingError = context->EvalSub(ctBootstrappedScaledDown, ctScaledUp);

		// Step 8: GPU-bootstrap the error.
		Ciphertext<DCRTPoly> errWrap = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
		errWrap->host				 = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(ctBootstrappingError));
		errWrap->device				 = std::any{};
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] boot2 (err towers %zu)\n", (size_t)std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(errWrap->host)->GetElements()[0].GetNumOfElements());
		if (lock_host)
			host_lk.unlock();
		Ciphertext<DCRTPoly> boot2	 = evalBootstrapInternal(ctx, errWrap, 1, 0, false, levelsToDrop);
		if (lock_host)
			host_lk.lock();
		refreshHostShadow(ctx, *boot2);
		auto ct2 = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(boot2->host);
		if (ct2->GetNoiseScaleDeg() > 1)
			context->GetScheme()->ModReduceInternalInPlace(ct2, compositeDegree);

		// Steps 9-10: subtract the bootstrapped error, scale back down.
		if (std::getenv("FIDESLIB_METABTS_TRACE")) fprintf(stderr, "[metabts] recombine: ct1 towers %zu | ct2 towers %zu\n", (size_t)ct1->GetElements()[0].GetNumOfElements(), (size_t)ct2->GetElements()[0].GetNumOfElements());
		auto finalCiphertext = context->EvalSub(ct1, ct2);
		context->EvalMultInPlace(finalCiphertext, 1.0 / powerOfTwoModulus);

		Ciphertext<DCRTPoly> out = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
		out->host				 = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(finalCiphertext));
		out->device				 = std::any{};
		return out;
	}
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled, levelsToDrop);
	return result;
}

void CudaEngine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	if (numIterations > 1) {
		// Route through the Meta-BTS composition above (same fix).
		Ciphertext<DCRTPoly> out = evalBootstrap(ctx, ciphertext, numIterations, precision, prescaled);
		ciphertext->host		 = out->host;
		ciphertext->device		 = out->device; // the device result (default); empty under the host glue (FIDESLIB_METABTS_DEVICE=0)
		return;
	}
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	auto res_gpu = deviceCt(ciphertext);
	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
}

// ---------------------------------------------------------------------------------------
// VARIABLE-OUTPUT-LEVEL BOOTSTRAP (opt-in; EvalBootstrap is untouched).
//
// WHAT IS CHEAPER AND WHY. Every op inside the refresh is priced per RNS limb, and the
// mod-raise is what sets how many limbs the refresh runs on: it reinterprets tower 0 onto the
// top of the chain, and CoeffsToSlots, EvalMod and SlotsToCoeffs then walk back down from
// there. Stopping the raise `levelsToDrop` towers lower is exact -- broadcastLimb0 replicates
// the same residue into however many towers were grown, and the EvalMod interval K and
// Chebyshev degree do not depend on the height -- and it removes those limbs from every one
// of those steps. The result comes back `levelsToDrop` levels deeper, which is the point: the
// caller asked for a level, not for everything the circuit could give it.
//
// WHAT IT COSTS. One extra set of transform diagonals per distinct output level, encoded at
// the reduced levels (LinearTransform requires the diagonals to sit at the ciphertext's own
// level). No extra key material: the BSGS rotation indices do not depend on the level.
// ---------------------------------------------------------------------------------------
uint32_t CudaEngine::bootstrapOutputLevel(CryptoContextImpl<DCRTPoly>& ctx, int slots) {
	auto& context  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	const auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE);
	if (!fhe)
		OPENFHE_THROW("EvalBootstrapToLevel: context has no CKKS bootstrap scheme");
	const auto it = fhe->m_bootPrecomMap.find(slots);
	if (it == fhe->m_bootPrecomMap.end())
		OPENFHE_THROW("EvalBootstrapToLevel: no bootstrap precomputation for " + std::to_string(slots) + " slots");
	const auto& precom = it->second;
	if (precom->m_paramsEnc.lvlb == 1 && precom->m_paramsDec.lvlb == 1)
		OPENFHE_THROW("EvalBootstrapToLevel: not implemented for a {1,1} level budget");
	if (precom->BTSlotsEncoding)
		OPENFHE_THROW("EvalBootstrapToLevel: the slots-encoding bootstrap order is not what the "
					  "device Bootstrap runs; this context's setup does not match the engine");

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	const uint32_t sizeQ	= static_cast<uint32_t>(cryptoParams->GetElementParams()->GetParams().size());
	const uint32_t sizeP	= static_cast<uint32_t>(cryptoParams->GetParamsP()->GetParams().size());
	const uint32_t cd		= cryptoParams->GetCompositeDegree();
	const uint32_t lvlb		= precom->m_paramsDec.lvlb;

	// Widest SlotsToCoeffs diagonal == the chain the transform's first step runs on; its own
	// level budget then says where it ends, and SlotsToCoeffs is last (ckksrns-fhe.cpp,
	// EvalCoeffsToSlotsPrecompute's towersToDrop / paramsVector).
	uint32_t widest = 0;
	for (const auto& step : precom->m_U0PreFFT)
		for (const auto& pt : step) {
			const uint32_t total = static_cast<uint32_t>(pt->template GetElement<lbcrypto::DCRTPoly>().GetAllElements().size());
			if (total > sizeP)
				widest = std::max(widest, total - sizeP);
		}
	if (widest < cd * lvlb)
		OPENFHE_THROW("EvalBootstrapToLevel: the SlotsToCoeffs set is shallower than its own level budget");
	return sizeQ - (widest - cd * lvlb);
}

void CudaEngine::ensureReducedBootPrecomputation(CryptoContextImpl<DCRTPoly>& ctx, int slots, int levelsToDrop) {
	if (levelsToDrop <= 0 || (*context_)->HasBootPrecomputation(slots, levelsToDrop))
		return;
	// Host-scheme work (OpenFHE encodes the diagonals), so it takes the same lock the
	// readback and the Meta-BTS glue take. See g_hostSchemeLock.
	std::unique_lock<std::recursive_mutex> host_lk(g_hostSchemeLock, std::defer_lock);
	if (FIDESlib::ConcurrentOps())
		host_lk.lock();
	if ((*context_)->HasBootPrecomputation(slots, levelsToDrop))
		return;
	// Default: views of the full-height set plus two scale folds, no new diagonal storage.
	// FIDESLIB_BOOT_SHARED_PRECOMP=0: the re-encoded copy this installed before 2026-10-03.
	if (FIDESlib::BootSharedPrecomp()) {
		FIDESlib::CKKS::AddBootstrapPrecomputationShared(slots, levelsToDrop, *context_);
		return;
	}
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	FIDESlib::CKKS::AddBootstrapPrecomputationReduced(context, slots, levelsToDrop, *context_);
}

Ciphertext<DCRTPoly>
CudaEngine::evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	const int slots			  = deviceCt(const_cast<Ciphertext<DCRTPoly>&>(ciphertext))->slots;
	const uint32_t defaultOut = bootstrapOutputLevel(ctx, slots);
	if (outputLevel <= defaultOut) {
		// Nothing to shorten: the bootstrap cannot return more levels than its circuit leaves.
		return evalBootstrap(ctx, ciphertext, numIterations, precision, prescaled);
	}
	const int levelsToDrop = static_cast<int>(outputLevel - defaultOut);
	ensureReducedBootPrecomputation(ctx, slots, levelsToDrop);
	Ciphertext<DCRTPoly> out = evalBootstrapInternal(ctx, ciphertext, numIterations, precision, prescaled, levelsToDrop);
	if (ctx.CiphertextLevel(*out) > outputLevel)
		OPENFHE_THROW("EvalBootstrapToLevel: the shortened refresh returned level " + std::to_string(ctx.CiphertextLevel(*out)) +
					  ", deeper than the requested " + std::to_string(outputLevel));
	// Any residue -- e.g. a Meta-BTS composition whose glue lands one level shallower than the
	// single-iteration circuit -- is spent exactly, so the contract holds to the level.
	dropToLevel(ctx, out, outputLevel);
	return out;
}

void CudaEngine::evalBootstrapToLevelInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	Ciphertext<DCRTPoly> out = evalBootstrapToLevel(ctx, ciphertext, outputLevel, numIterations, precision, prescaled);
	ciphertext->host		 = out->host;
	ciphertext->device		 = out->device;
}

void CudaEngine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	// On CUDA there is no record/replay step to run first, so the readback is exactly the shadow
	// refresh.
	refreshHostShadow(ctx, *ct);
}

void CudaEngine::refreshHostShadow(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		// Already resident on the host; nothing to read back.
		return;
	}
	// THE READBACK IS HOST-SCHEME WORK: GetOpenFHECipherText below builds DCRTPolys and puts them
	// in EVALUATION format, i.e. it runs lbcrypto's NTT and reaches the lazily-built static
	// transform tables. The host-to-device direction has been serialized since the concurrent mode
	// landed (g_deviceLoadLock); this is the same marshal in the other direction and needs the
	// same treatment. See g_hostSchemeLock. Recursive, because the two-iteration bootstrap calls
	// this from inside its own locked glue.
	std::unique_lock<std::recursive_mutex> host_lk(g_hostSchemeLock, std::defer_lock);
	if (FIDESlib::ConcurrentOps())
		host_lk.lock();
	ct.EnsureLazyHostCopy();
	auto& ct_cpu = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct.host);

	auto ct_gpu = deviceCt(ct);
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct_gpu->store(raw_ct);

	// Grow the host shadow if the device result carries more RNS limbs than it currently holds
	// (e.g. after a level-raising op like ModRaise). GetOpenFHECipherText overwrites the limb values
	// and trims down to the device limb count, so a zero ciphertext at the full level is a valid
	// container — built directly from the crypto parameters, no secret key needed.
	size_t gpu_levels = raw_ct.numRes;
	// The device result is degree 1 (c0, c1) or degree 2 (c0, c1, c2); the host shell must carry
	// exactly that many components before GetOpenFHECipherText writes into it.
	const size_t gpu_components = raw_ct.sub_2.empty() ? 2u : 3u;
	// Check the element count BEFORE indexing GetElements()[0]: an upload-dropped or
	// never-populated host shell has zero elements, and indexing it first is out of bounds.
	bool needsFresh	  = ct_cpu->GetElements().empty();
	size_t cpu_levels = 0;
	if (!needsFresh) {
		cpu_levels = ct_cpu->GetElements()[0].GetAllElements().size();
		needsFresh = cpu_levels < gpu_levels;
	}
	if (needsFresh) {
		auto& context		  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
		const auto elemParams = context->GetCryptoParameters()->GetElementParams();
		lbcrypto::DCRTPoly zero(elemParams, Format::EVALUATION, true);
		auto fresh = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_cpu);
		fresh->SetElements(std::vector<lbcrypto::DCRTPoly>(gpu_components, zero));
		// A full-tower shell is level 0 by definition. GetOpenFHECipherText adjusts
		// the level RELATIVELY (old level + old towers - device towers), so a shell
		// carrying the pre-grow ciphertext's level yields an impossible level
		// (observed: level 33 in a depth-25 context for a 13-tower bootstrap
		// readback: 20 + 26 - 13). Decrypt-only readbacks never noticed (decrypt
		// ignores the level field); leveled host ops on readbacks throw in
		// DropLastElements. Reset so the relative adjustment lands on L0 - towers.
		fresh->SetLevel(0);
		ct_cpu = fresh;
	} else if (ct_cpu->GetElements().size() != gpu_components) {
		// Reshape the existing shell in either direction. Cloning the last component keeps the
		// params, format and tower structure that GetOpenFHECipherText expects; its values are
		// overwritten from the device data below.
		auto elements = ct_cpu->GetElements();
		while (elements.size() < gpu_components)
			elements.push_back(elements.back());
		elements.resize(gpu_components);
		ct_cpu->SetElements(std::move(elements));
	}

	// Overwrite the host ciphertext with the device data.
	FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
}

void CudaEngine::linearTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool ext) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);

	// Hand the kernel exactly rowSize diagonals: it asserts every pointer in the vector is non-null
	// (a padded tail would trip that), and it never reads past rowSize anyway. The facade has
	// already checked size >= rowSize and rejected nulls in that range.
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(static_cast<size_t>(rowSize));
	for (int k = 0; k < rowSize; ++k) {
		const auto& pt = diagonals[static_cast<size_t>(k)];
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		// FIDESlib::CKKS::LinearTransform picks the extended path by INSPECTION: it takes it only
		// if every diagonal's device polynomial is mod-up (LinearTransform.cu:104-113), and
		// silently runs the slow one-ModDown-per-baby-rotation path otherwise. A silent fallback
		// is the failure mode this flag exists to prevent, so check it here. A plaintext lands
		// mod-up because its host encoding carried the P moduli and RNSPoly::loadConstant saw
		// them (RNSPoly.cpp:933-936) — i.e. because it came from MakeCKKSPackedPlaintextExtended
		// against this backend. If it did not, the P limbs were never uploaded.
		if (ext && !pt_gpu->c0.isModUp()) {
			OPENFHE_THROW("LinearTransformInPlace(ext=true): diagonal " + std::to_string(k) +
			  " is not in the extended Q||P basis on the device; build it with MakeCKKSPackedPlaintextExtended");
		}
		pts_gpu.push_back(pt_gpu.get());
	}

	FIDESlib::CKKS::LinearTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, offset);
}

std::vector<Ciphertext<DCRTPoly>> CudaEngine::linearTransformMany(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  int rowSize,
  int bStep,
  const std::vector<std::vector<Plaintext>>& diagonalSets,
  int stride,
  int offset,
  bool ext) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	// A working copy of the source: FIDESlib::CKKS::LinearTransformMany rescales a noise-level-2
	// input in place (once, for the whole batch) exactly as the single transform does, and the
	// caller's ciphertext is shared — q/k/v read one x. This clone is what the loop reference pays
	// per transform; here it is paid once.
	Ciphertext<DCRTPoly> src = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto src_gpu			 = deviceCt(src);

	// Result shells with a FRESH device ciphertext each: LinearTransformMany finishes every
	// transform with Ciphertext::copy, which sizes its destination (RNSPoly::copy drops/grows to the
	// source level) and then overwrites every limb. Cloning `ct` per result would deep-copy ~29 MB
	// on device per transform only to throw it away — the same trade makeResultShell exists for.
	std::vector<Ciphertext<DCRTPoly>> results;
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	results.reserve(diagonalSets.size());
	results_gpu.reserve(diagonalSets.size());
	for (size_t t = 0; t < diagonalSets.size(); ++t) {
		Ciphertext<DCRTPoly> shell = makeResultShell(*ct);
		auto res_gpu			   = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
		results_gpu.push_back(res_gpu.get());
		shell->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(res_gpu));
		results.push_back(std::move(shell));
	}

	// Same per-diagonal validation as the single transform, applied to every set: a silent fallback
	// to the one-ModDown-per-baby-rotation path is exactly what `ext` exists to prevent, and here it
	// would be a WHOLE-BATCH fallback — one launch takes one basis. The device Plaintext objects
	// stay owned by the value Plaintexts (their `device` slot), which the caller holds.
	std::vector<std::vector<FIDESlib::CKKS::Plaintext*>> pts_gpu(diagonalSets.size());
	for (size_t t = 0; t < diagonalSets.size(); ++t) {
		pts_gpu[t].reserve(static_cast<size_t>(rowSize));
		for (int k = 0; k < rowSize; ++k) {
			const auto& pt = diagonalSets[t][static_cast<size_t>(k)];
			ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
			auto pt_gpu = devicePt(pt);
			if (ext && !pt_gpu->c0.isModUp()) {
				OPENFHE_THROW("LinearTransformMany(ext=true): diagonal " + std::to_string(k) + " of set " + std::to_string(t) +
				  " is not in the extended Q||P basis on the device; build it with MakeCKKSPackedPlaintextExtended");
			}
			pts_gpu[t].push_back(pt_gpu.get());
		}
	}

	FIDESlib::CKKS::LinearTransformMany(*src_gpu, results_gpu, rowSize, bStep, pts_gpu, stride, offset);
	return results;
}

// ---- Prepared linear transforms ----

namespace {
// 64-bit FNV-1a, the same fold Engine::plaintextIdentity uses on the host side.
constexpr uint64_t kPrepFnvOffset = 1469598103934665603ull;
constexpr uint64_t kPrepFnvPrime  = 1099511628211ull;

inline uint64_t PrepFold(uint64_t h, uint64_t v) {
	for (int b = 0; b < 8; ++b) {
		h ^= static_cast<uint64_t>((v >> (8 * b)) & 0xffu);
		h *= kPrepFnvPrime;
	}
	return h;
}

// The device Plaintext objects behind a set of value Plaintexts, loaded and ext-checked exactly as
// linearTransformInPlace / linearTransformMany check them. Shared by the prepared entry points so
// there is ONE place that decides what "these diagonals, on this device, in this basis" means.
std::vector<FIDESlib::CKKS::Plaintext*>
deviceDiagonals(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Plaintext>& diagonals, int rowSize, bool ext, const char* what, int set) {
	const std::string where = set < 0 ? std::string{} : (" of set " + std::to_string(set));
	std::vector<FIDESlib::CKKS::Plaintext*> out;
	out.reserve(static_cast<size_t>(rowSize));
	for (int k = 0; k < rowSize; ++k) {
		const auto& pt = diagonals[static_cast<size_t>(k)];
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		if (ext && !pt_gpu->c0.isModUp()) {
			OPENFHE_THROW(std::string{ what } + "(ext=true): diagonal " + std::to_string(k) + where +
			  " is not in the extended Q||P basis on the device; build it with MakeCKKSPackedPlaintextExtended");
		}
		out.push_back(pt_gpu.get());
	}
	return out;
}

// The resident table a prepared handle carries. A handle this engine prepared always has one;
// reaching the throw means a handle prepared on another backend was handed to this one, which is a
// mistake to report rather than a slow path to take.
const FIDESlib::CKKS::PreparedLTTable* deviceTable(const PreparedLinearTransformImpl& prepared, const char* what) {
	if (!prepared.device.has_value())
		OPENFHE_THROW(std::string{ what } + ": the prepared transform carries no device table; it was prepared on another backend");
	return std::any_cast<const std::shared_ptr<FIDESlib::CKKS::PreparedLTTable>&>(prepared.device).get();
}
} // namespace

// THE DEVICE IDENTITY, and the whole staleness rule on this backend.
//
// A prepared table holds the ADDRESSES of each diagonal's device limb-pointer arrays. Those
// addresses are what goes stale, so those addresses are what is fingerprinted — together with the
// RNS shape that decides how many of them the kernel will read. Anything that re-encodes, re-levels,
// mod-ups, mod-downs, releases or reloads a diagonal moves at least one of them, and the facade's
// RequireFresh then throws before a single pointer reaches a kernel.
//
// Host-side reads only. limbptr.data / SPECIALlimbptr.data are host-resident members that HOLD
// device addresses, and getLevel()/isModUp() read device-side host metadata (the same fact that
// makes fhe/server's thousands of GetLevel calls free — design/host-serialization-scope.md,
// "Refuted, so nobody re-chases them"). No device traffic and no synchronisation, which it must not
// have: this runs once per diagonal on every prepared call.
uint64_t CudaEngine::plaintextIdentity(CryptoContextImpl<DCRTPoly>& ctx, const Plaintext& pt) {
	uint64_t h = kPrepFnvOffset;
	h		   = PrepFold(h, reinterpret_cast<uint64_t>(pt.get()));
	if (!pt)
		return h;
	h = PrepFold(h, pt->extended ? 0x9e37u : 0x1u);
	h = PrepFold(h, pt->device_only ? 0x85ebu : 0x2u);
	if (!pt->device.has_value()) {
		// Not resident: fall back to the host fold, so that a diagonal which later gains a device
		// payload fingerprints differently from the same diagonal before it did.
		return PrepFold(Engine::plaintextIdentity(ctx, pt), 0xfeedfaceull);
	}
	auto pt_gpu		 = devicePt(pt);
	const auto& poly = pt_gpu->c0;
	h				 = PrepFold(h, static_cast<uint64_t>(poly.getLevel()));
	h				 = PrepFold(h, poly.isModUp() ? 0x5bf0u : 0x4u);
	h				 = PrepFold(h, static_cast<uint64_t>(poly.GPU.size()));
	for (const auto& part : poly.GPU) {
		h = PrepFold(h, reinterpret_cast<uint64_t>(part.limbptr.data));
		h = PrepFold(h, reinterpret_cast<uint64_t>(part.SPECIALlimbptr.data));
		h = PrepFold(h, static_cast<uint64_t>(part.limb.size()));
		h = PrepFold(h, static_cast<uint64_t>(part.SPECIALlimb.size()));
	}
	return h;
}

PreparedLinearTransform
CudaEngine::prepareLinearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<std::vector<Plaintext>>& diagonalSets, const LinearTransformShape& shape) {
	// The diagonals must be ON the device before their pointers can be captured, so this is also
	// where a caller's diagonals get loaded — once, here, instead of on every call.
	std::vector<std::vector<FIDESlib::CKKS::Plaintext*>> pts_gpu;
	pts_gpu.reserve(diagonalSets.size());
	for (size_t t = 0; t < diagonalSets.size(); ++t) {
		pts_gpu.push_back(deviceDiagonals(
		  ctx, diagonalSets[t], shape.rowSize, shape.ext, "PrepareLinearTransform", diagonalSets.size() == 1 ? -1 : static_cast<int>(t)));
	}

	// The base handle first: it captures the plaintexts (so nothing below can be freed underneath
	// the table) and their identities as the device now sees them (so a later call can tell whether
	// it may still use it). Order matters — the identity must be read AFTER the load above, or
	// every first call would look stale.
	PreparedLinearTransform handle = Engine::prepareLinearTransform(ctx, diagonalSets, shape);

	// `stride` here is the table's pt_reuse_stride, which is 1 for every transform this library
	// issues (LinearTransform / LinearTransformMany both pass 1); the SHAPE's stride is the
	// ciphertext rotation step and is a different quantity that the table does not index by.
	handle->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::PreparedLTTable>>(
	  std::make_shared<FIDESlib::CKKS::PreparedLTTable>(**context_, pts_gpu, shape.rowSize, shape.bStep, /*stride=*/1, shape.ext));
	return handle;
}

void CudaEngine::linearTransformInPlacePrepared(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) {
	const auto& s = prepared.Shape();
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	// The same device Plaintext list the unprepared call builds: the transform still needs the
	// plaintext OBJECTS for its stream ordering and its ext inspection. What the table replaces is
	// the per-call assembly and pageable upload of their POINTERS, nothing else — which is why this
	// is the same call with one more argument and not a second implementation of the transform.
	auto pts_gpu = deviceDiagonals(ctx, prepared.Diagonals(), s.rowSize, s.ext, "LinearTransformInPlace", -1);
	FIDESlib::CKKS::LinearTransform(*ct_gpu, s.rowSize, s.bStep, pts_gpu, s.stride, s.offset, deviceTable(prepared, "LinearTransformInPlace"));
}

std::vector<Ciphertext<DCRTPoly>>
CudaEngine::linearTransformManyPrepared(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) {
	const auto& s	   = prepared.Shape();
	const auto& sets   = prepared.DiagonalSets();
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	// Everything below the diagonal handling is character for character the unprepared
	// linearTransformMany: the working copy of the shared source, the result shells with a fresh
	// device ciphertext each, and the one batched call. The handle covers the WHOLE batch precisely
	// so that this stays one call — a table per transform would have forced the batch apart and
	// traded the shared hoisted ModUp for the upload this arm saves.
	Ciphertext<DCRTPoly> src = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto src_gpu			 = deviceCt(src);

	std::vector<Ciphertext<DCRTPoly>> results;
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	results.reserve(sets.size());
	results_gpu.reserve(sets.size());
	for (size_t t = 0; t < sets.size(); ++t) {
		Ciphertext<DCRTPoly> shell = makeResultShell(*ct);
		auto res_gpu			   = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
		results_gpu.push_back(res_gpu.get());
		shell->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(res_gpu));
		results.push_back(std::move(shell));
	}

	std::vector<std::vector<FIDESlib::CKKS::Plaintext*>> pts_gpu;
	pts_gpu.reserve(sets.size());
	for (size_t t = 0; t < sets.size(); ++t)
		pts_gpu.push_back(deviceDiagonals(ctx, sets[t], s.rowSize, s.ext, "LinearTransformMany", static_cast<int>(t)));

	FIDESlib::CKKS::LinearTransformMany(
	  *src_gpu, results_gpu, s.rowSize, s.bStep, pts_gpu, s.stride, s.offset, deviceTable(prepared, "LinearTransformMany"));
	return results;
}

bool CudaEngine::supportsExtendedPlaintexts() const {
	return true;
}

void CudaEngine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		pts_gpu.push_back(pt_gpu.get());
	}

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::ConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, indexes, gStep);
}

void CudaEngine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		pts_gpu.push_back(pt_gpu.get());
	}

	// Load mask
	ctx.LoadPlaintext(mask);
	auto mask_gpu = devicePt(mask);

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::SpecialConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, *mask_gpu, stride, maskRotationStride, indexes, gStep);
}

void CudaEngine::loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) {
	if (isContextLoaded())
		return;

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	FIDESlib::CKKS::Parameters params{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

	// Determine the boot configuration based on the secret key distribution.
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	FIDESlib::BOOT_CONFIG bootConfig;
	switch (ctx.keyDist) {
	case fideslib::UNIFORM_TERNARY: bootConfig = FIDESlib::UNIFORM; break;
	case fideslib::SPARSE_TERNARY: bootConfig = FIDESlib::SPARSE; break;
	case fideslib::SPARSE_ENCAPSULATED: bootConfig = FIDESlib::ENCAPS; break;
	default: bootConfig = FIDESlib::UNIFORM; break;
	}

	// Opt-in (ctx.modReduction is zero-valued unless SetModReductionConfig was called): the device
	// evaluates the same truncated series the host reserved levels for in EvalBootstrapSetup.
	FIDESlib::CKKS::RawParams rawParams = FIDESlib::CKKS::GetRawParams(context, bootConfig, ctx.modReduction.chebyshevDegree);
	params								= params.adaptTo(rawParams);
	FIDESlib::CKKS::Context c			= FIDESlib::CKKS::GenCryptoContextGPU(params, devices_);

	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);

	// Multiplicative key switching key.
	auto& keyMap = context->GetAllEvalMultKeys(); // lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(pkImpl->GetKeyTag()) != keyMap.end()) {
		auto raw_eval_ksk = FIDESlib::CKKS::GetEvalKeySwitchKey(pkImpl);
		FIDESlib::CKKS::KeySwitchingKey eval_ksk(c);
		eval_ksk.Initialize(raw_eval_ksk);
		c->AddEvalKey(std::move(eval_ksk));
	}
	// Rotational key switching keys.
	for (const auto& step : ctx.rotation_indexes) {
		auto raw_rot_ksk = FIDESlib::CKKS::GetRotationKeySwitchKey(pkImpl, step);
		FIDESlib::CKKS::KeySwitchingKey rot_ksk(c);
		rot_ksk.Initialize(raw_rot_ksk);
		c->AddRotationKey(step, std::move(rot_ksk));
	}
	// Bootstrapping keys.
	for (const auto& slot : ctx.slots_bootstrap) {
		FIDESlib::CKKS::AddBootstrapPrecomputation(pkImpl, slot, c);
	}

	context_ = std::make_unique<FIDESlib::CKKS::Context>(std::move(c));
}

bool CudaEngine::supportsDeviceEncode() const {
	return true;
}

void CudaEngine::encodeDevice(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const DeviceEncodedCoefficients& enc) {
	// Reachable from an encode/upload WORKER THREAD, not only from the thread that issues ops.
	if (pt->device.has_value()) {
		OPENFHE_THROW("encodeDevice: this plaintext is already device-resident");
	}
	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	if (enc.moduli.empty()) {
		OPENFHE_THROW("encodeDevice: no Q moduli supplied");
	}
	if (enc.noiseScaleDeg != 1) {
		// noiseScaleDeg > 1 multiplies the encoded polynomial by llround(scalingFactor)^(deg-1) in
		// CRT, which under the FLEXIBLE* techniques is a double-rounded, level-dependent constant.
		// Reproducing that on this path would mean reproducing OpenFHE's scaling-factor table
		// arithmetic exactly; nothing in this library needs it, so refuse instead of approximating.
		OPENFHE_THROW("encodeDevice: only noiseScaleDeg == 1 is supported (got " + std::to_string(enc.noiseScaleDeg) + "); use MakeCKKSPackedPlaintext");
	}

	auto& context_gpu = *context_;

	// Level metadata is carried on the RNSPoly by the modulus count, exactly as in the load path:
	// RNSPoly::loadCoefficients grows/drops to moduli.size()-1.
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu);
	gpu_pt->loadCoefficients(enc.coefficients, enc.moduli, enc.bigBound, enc.scalingFactor, static_cast<int>(enc.noiseScaleDeg), static_cast<int>(enc.slots));

	pt->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(std::move(gpu_pt));
}

// FIDESLIB_DEVICE_IFFT=1 (opt-in): the slot values go up and the device runs the transform too.
bool CudaEngine::supportsDeviceSlotsEncode() const {
	return FIDESlib::DeviceIfft();
}

void CudaEngine::encodeDeviceSlotsMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const DeviceEncodedSlots& enc) {
	if (pts.size() != enc.values.size()) {
		OPENFHE_THROW("encodeDeviceSlotsMany: " + std::to_string(pts.size()) + " plaintexts but " + std::to_string(enc.values.size()) + " value vectors");
	}
	if (pts.empty()) {
		return;
	}
	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	if (enc.moduli.empty()) {
		OPENFHE_THROW("encodeDeviceSlotsMany: no Q moduli supplied");
	}
	if (enc.noiseScaleDeg != 1) {
		OPENFHE_THROW("encodeDeviceSlotsMany: only noiseScaleDeg == 1 is supported (got " + std::to_string(enc.noiseScaleDeg) + "); use MakeCKKSPackedPlaintext");
	}
	for (size_t i = 0; i < pts.size(); ++i) {
		if (!pts[i]) {
			OPENFHE_THROW("encodeDeviceSlotsMany: null plaintext at " + std::to_string(i));
		}
		if (pts[i]->device.has_value()) {
			OPENFHE_THROW("encodeDeviceSlotsMany: plaintext " + std::to_string(i) + " is already device-resident");
		}
		if (enc.values[i] == nullptr || enc.values[i]->size() != enc.slots) {
			OPENFHE_THROW("encodeDeviceSlotsMany: value vector " + std::to_string(i) + " does not hold exactly " + std::to_string(enc.slots) + " slots");
		}
	}
	auto& context_gpu = *context_;
	std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> gpu_pts;
	gpu_pts.reserve(pts.size());
	std::vector<FIDESlib::CKKS::Plaintext*> raw_pts;
	raw_pts.reserve(pts.size());
	for (size_t i = 0; i < pts.size(); ++i) {
		gpu_pts.push_back(std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu));
		raw_pts.push_back(gpu_pts.back().get());
	}
	FIDESlib::CKKS::Plaintext::loadSlotsBatch(raw_pts, enc.values, static_cast<int>(enc.slots), enc.scalingFactor, enc.moduli, enc.bigBound, enc.scalingFactor, static_cast<int>(enc.noiseScaleDeg));
	for (size_t i = 0; i < pts.size(); ++i) {
		pts[i]->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(std::move(gpu_pts[i]));
	}
}

bool CudaEngine::supportsDeviceEncodeMany() const {
	return true;
}

void CudaEngine::encodeDeviceMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const std::vector<DeviceEncodedCoefficients>& encs) {
	// Reachable from an encode/upload WORKER THREAD, not only from the thread that issues ops.
	if (pts.size() != encs.size()) {
		OPENFHE_THROW("encodeDeviceMany: " + std::to_string(pts.size()) + " plaintexts but " + std::to_string(encs.size()) + " coefficient sets");
	}
	if (pts.empty()) {
		return;
	}
	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	// Every precondition encodeDevice enforces, enforced here per element BEFORE anything is built,
	// plus the one the batch adds: a single shared modulus chain. Checking first is what makes the
	// call all-or-nothing -- the caller's fallback is to encode the whole group the other way, so a
	// batch that threw halfway would leave it re-encoding over half-built plaintexts.
	const DeviceEncodedCoefficients& head = encs.front();
	if (head.moduli.empty()) {
		OPENFHE_THROW("encodeDeviceMany: no Q moduli supplied");
	}
	for (size_t i = 0; i < pts.size(); ++i) {
		const DeviceEncodedCoefficients& enc = encs[i];
		if (!pts[i]) {
			OPENFHE_THROW("encodeDeviceMany: null plaintext at " + std::to_string(i));
		}
		if (pts[i]->device.has_value()) {
			OPENFHE_THROW("encodeDeviceMany: plaintext " + std::to_string(i) + " is already device-resident");
		}
		if (enc.noiseScaleDeg != 1) {
			// Same refusal, and the same reason, as encodeDevice: noiseScaleDeg > 1 multiplies the
			// encoded polynomial by llround(scalingFactor)^(deg-1) in CRT, a double-rounded,
			// level-dependent constant this path does not reproduce.
			OPENFHE_THROW("encodeDeviceMany: only noiseScaleDeg == 1 is supported (got " + std::to_string(enc.noiseScaleDeg) + "); use MakeCKKSPackedPlaintext");
		}
		// One level per batch. The device path indexes one prime table for the whole batch and
		// stamps one scale onto every plaintext, so a mixed-level batch would be silently wrong
		// rather than slow. The facade resolves a single `level` argument for the call, so this is
		// a guard against a future caller, not a restriction on today's.
		if (enc.moduli != head.moduli || enc.level != head.level || enc.scalingFactor != head.scalingFactor || enc.slots != head.slots || enc.bigBound != head.bigBound) {
			OPENFHE_THROW("encodeDeviceMany: element " + std::to_string(i) + " does not share the batch's level/modulus chain");
		}
	}

	auto& context_gpu = *context_;

	// Build the device plaintexts first and publish them into pts only once the encode has
	// succeeded, so a throw out of loadCoefficientsBatch leaves every pts[i]->device empty.
	std::vector<std::shared_ptr<FIDESlib::CKKS::Plaintext>> gpu_pts;
	gpu_pts.reserve(pts.size());
	std::vector<FIDESlib::CKKS::Plaintext*> raw_pts;
	raw_pts.reserve(pts.size());
	std::vector<const std::vector<uint64_t>*> coeffs;
	coeffs.reserve(pts.size());
	for (size_t i = 0; i < pts.size(); ++i) {
		gpu_pts.push_back(std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu));
		raw_pts.push_back(gpu_pts.back().get());
		coeffs.push_back(&encs[i].coefficients);
	}

	// Level metadata is carried on the RNSPoly by the modulus count, exactly as in encodeDevice.
	FIDESlib::CKKS::Plaintext::loadCoefficientsBatch(
	  raw_pts, coeffs, head.moduli, head.bigBound, head.scalingFactor, static_cast<int>(head.noiseScaleDeg), static_cast<int>(head.slots));

	for (size_t i = 0; i < pts.size(); ++i) {
		pts[i]->device = std::make_any<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(std::move(gpu_pts[i]));
	}
}

// Guards the COLD half of loadPlaintext/loadCiphertext -- the first-touch marshal that PUBLISHES a
// device payload into a host value's `std::any` -- and is TAKEN ONLY in the concurrent mode
// (FIDESlib::ConcurrentOps; the default mode keeps the unlocked path this has always had).
// Two host threads can reach the same value -- an
// application that issues independent branches of its circuit concurrently has them read the same
// source ciphertext, and a shared plaintext operand is read by every branch that multiplies by it
// -- and without this both would build a device copy and both would assign that `std::any`, which
// is a torn write and a double publish, not a lost cache hit. The hot path (already resident) never
// reaches the lock; the cold path is a host-to-device marshal, so one uncontended acquire in front
// of it costs nothing measurable.
static std::mutex g_deviceLoadLock;

void CudaEngine::loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) {
	if (pt->device.has_value())
		return;

	std::unique_lock<std::mutex> load_lk(g_deviceLoadLock, std::defer_lock);
	if (FIDESlib::ConcurrentOps()) {
		load_lk.lock();
		if (pt->device.has_value())
			return;
	}

	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								  = *context_;
	auto& context									  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	const auto& ptImpl								  = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
	FIDESlib::CKKS::RawPlainText raw_pt				  = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu, raw_pt);
	pt->device										  = std::make_any<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(std::move(gpu_pt));
}

void CudaEngine::loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	if (ct->device.has_value())
		return;

	std::unique_lock<std::mutex> load_lk(g_deviceLoadLock, std::defer_lock);
	if (FIDESlib::ConcurrentOps()) {
		load_lk.lock();
		if (ct->device.has_value())
			return;
	}

	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								   = *context_;
	auto& context									   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	auto& ctImpl									   = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
	FIDESlib::CKKS::RawCipherText raw_ct			   = FIDESlib::CKKS::GetRawCipherText(context, ctImpl);
	std::shared_ptr<FIDESlib::CKKS::Ciphertext> gpu_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
	ct->device										   = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(gpu_ct));

	// Opt-in: harnesses that never read an uploaded input's host residues can drop them here.
	// Results sharing this shell are rebuilt from device residues by refreshHostShadow.
	if (dropInputHostAfterUpload_) {
		ctImpl->SetElements(std::vector<lbcrypto::DCRTPoly>{});
	}
}

std::shared_ptr<void> CudaEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	// The ModUp of c1 -- the shared half of every key switch of THIS ciphertext. See
	// FastRotationPrecompute above for what the handle owns and why it is not the context scratch.
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	auto ct_gpu = deviceCt(ct);
	FIDESlib::CKKS::SetCurrentContext(*context_);
	return std::make_shared<FastRotationPrecompute>(**context_, *ct_gpu);
}

// ---- Ciphertext backend hooks ----

std::any CudaEngine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& src) {
	// Not device-resident → nothing to clone (the host value is copied by the value type itself).
	if (!src.device.has_value()) {
		return std::any{};
	}
	// Deep-copy the device payload into a fresh device ciphertext.
	auto src_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(src.device);
	auto new_ct	 = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	new_ct->copy(*src_gpu);
	return std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(new_ct));
}

size_t CudaEngine::ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetLevelHost();
	}
	// Depth is reversed in FIDESlib, so level = maxDepth - deviceLevel.
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return ctx.multiplicative_depth - ct_gpu->getLevel();
}

size_t CudaEngine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetNoiseScaleDegHost();
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return ct_gpu->NoiseLevel;
}

double CudaEngine::ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetScalingFactorHost();
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return ct_gpu->NoiseFactor;
}

size_t CudaEngine::ciphertextSlots(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetSlotsHost();
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return static_cast<size_t>(ct_gpu->slots);
}

size_t CudaEngine::ciphertextNumElements(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	// Pure read of the device metadata: no readback, no synchronization. Without this override the
	// facade would fall back to the host shadow, which reads 2 for a device-resident degree-2
	// ciphertext and would make the api's RequireDegree1 guard vacuous on CUDA.
	if (!ct.device.has_value()) {
		return ct.GetNumElementsHost();
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return static_cast<size_t>(ct_gpu->numElements());
}

void CudaEngine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	if (!ct.device.has_value()) {
		ct.SetSlotsHost(slots);
		return;
	}
	auto ct_gpu	  = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	ct_gpu->slots = static_cast<int>(slots);
}

void CudaEngine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) {
	if (!ct.device.has_value()) {
		ct.SetLevelHost(level);
		return;
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	ct_gpu->dropToLevel(ctx.multiplicative_depth - level);
}

// The public face of ContextData::releaseParkedScratch, which is where the whole argument lives:
// what is parked, why the two steps are in that order, and the precondition that no other thread
// is issuing ops. This adds two things and nothing else.
//
// FIRST, the null-context guard. An application can reach its pass boundary before it has ever
// loaded a context (a run that refuses the first request, say), and there is nothing parked then
// either. `ran` stays false, for the same reason it does in the default mode: nothing was
// attempted.
//
// SECOND, the type conversion. `ContextData::ScratchRelease` is a private (src/) CUDA type and
// `ScratchReleaseStats` is the public one an application and a CPU-only build can see; see the
// note on the public struct for why they are not the same type. Field for field, nothing else.
ScratchReleaseStats CudaEngine::releaseParkedScratch(CryptoContextImpl<DCRTPoly>&) {
	if (!context_) {
		return {};
	}
	const FIDESlib::CKKS::ContextData::ScratchRelease r = (*context_)->releaseParkedScratch();
	ScratchReleaseStats out;
	out.polys_released = r.polys_released;
	out.blocks_moved   = r.blocks_moved;
	out.slots_drained  = r.slots_drained;
	out.ran			   = r.ran;
	return out;
}

// ---- Context backend state ----

bool CudaEngine::isContextLoaded() const {
	return context_ != nullptr;
}

void CudaEngine::synchronize() const {
	if (!context_) {
		return;
	}
	for (const auto& device : devices_) {
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		CudaCheckErrorModNoSync;
	}
}

void CudaEngine::teardown() {
	if (!context_) {
		return;
	}
	FIDESlib::CKKS::DeregisterCryptoContextGPU(*context_);
	context_.reset();
}

void CudaEngine::setDevices(const std::vector<int>& devices) {
	devices_ = devices;
}

std::vector<int> CudaEngine::devices() const {
	return devices_;
}

CudaEngine::~CudaEngine() {
	teardown();
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_CUDA
