#include "engine/Engine.hpp"

#include <openfhe.h>

#include <algorithm>
#include <string>

namespace fideslib {

// Default implementations: every operation throws until a backend overrides it, so a backend under
// construction compiles (and reports a precise error at runtime) instead of having to stub the whole
// interface up front. teardown() is the one exception — see below.

void Engine::notImplemented(const char* op) const {
	OPENFHE_THROW(std::string(op) + " is not implemented by the " + name() + " backend");
}

// ---- Negation ----
Ciphertext<DCRTPoly> Engine::evalNegate(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalNegate");
}

void Engine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalNegateInPlace");
}

// ---- Addition ----
Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalAdd");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalAddInPlace");
}

Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalAdd");
}

Ciphertext<DCRTPoly> Engine::evalAdd(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalAdd");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalAddInPlace");
}

void Engine::evalAddInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalAddInPlace");
}

Ciphertext<DCRTPoly> Engine::evalAddMany(CryptoContextImpl<DCRTPoly>&, const std::vector<Ciphertext<DCRTPoly>>&) {
	notImplemented("evalAddMany");
}

void Engine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>&, std::vector<Ciphertext<DCRTPoly>>&) {
	notImplemented("evalAddManyInPlace");
}

// ---- Subtraction ----
Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, Plaintext&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalSub");
}

Ciphertext<DCRTPoly> Engine::evalSub(CryptoContextImpl<DCRTPoly>&, double, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSub");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSubInPlace");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalSubInPlace");
}

void Engine::evalSubInPlace(CryptoContextImpl<DCRTPoly>&, double, Ciphertext<DCRTPoly>&) {
	notImplemented("evalSubInPlace");
}

// ---- Multiplication ----
Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalMult");
}

Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalMult");
}

Ciphertext<DCRTPoly> Engine::evalMult(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalMult");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Plaintext&) {
	notImplemented("evalMultInPlace");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, double) {
	notImplemented("evalMultInPlace");
}

void Engine::evalMultInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalMultInPlace");
}

Ciphertext<DCRTPoly> Engine::evalMultNoRelin(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalMultNoRelin");
}

Ciphertext<DCRTPoly> Engine::relinearize(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("relinearize");
}

void Engine::relinearizeInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("relinearizeInPlace");
}

// ---- Square ----
Ciphertext<DCRTPoly> Engine::evalSquare(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalSquare");
}

void Engine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("evalSquareInPlace");
}

// ---- Rotation ----
Ciphertext<DCRTPoly> Engine::evalRotate(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t) {
	notImplemented("evalRotate");
}

void Engine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int32_t) {
	notImplemented("evalRotateInPlace");
}

std::shared_ptr<void> Engine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("evalFastRotationPrecompute");
}

Ciphertext<DCRTPoly> Engine::evalFastRotation(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const int32_t, const uint32_t, const std::shared_ptr<void>&) {
	notImplemented("evalFastRotation");
}

Ciphertext<DCRTPoly> Engine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const int32_t, const std::shared_ptr<void>&, bool) {
	notImplemented("evalFastRotationExt");
}

std::vector<Ciphertext<DCRTPoly>>
Engine::evalFastRotation(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const std::vector<int32_t>&, const uint32_t, const std::shared_ptr<void>&) {
	notImplemented("evalFastRotation");
}

std::vector<Ciphertext<DCRTPoly>>
Engine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, const std::vector<int32_t>&, const std::shared_ptr<void>&, bool) {
	notImplemented("evalFastRotationExt");
}

// THE REFERENCE for EvalRotateMany: rotate one at a time. Byte-identical to EvalRotate by
// construction — it IS evalRotate — which is what makes it the oracle the batched CUDA override is
// tested against (test/ApiTests.cpp EvalRotateMany*, test/ApiParityTest.cpp EvalRotateMany).
std::vector<Ciphertext<DCRTPoly>> Engine::evalRotateMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& cts, const int32_t index) {
	std::vector<Ciphertext<DCRTPoly>> out;
	out.reserve(cts.size());
	for (const auto& ct : cts)
		out.push_back(evalRotate(ctx, ct, index));
	return out;
}

// ---- Chebyshev series ----
Ciphertext<DCRTPoly> Engine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, std::vector<double>&, double, double) {
	notImplemented("evalChebyshevSeries");
}

void Engine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, std::vector<double>&, double, double) {
	notImplemented("evalChebyshevSeriesInPlace");
}

// ---- Rescale ----
Ciphertext<DCRTPoly> Engine::rescale(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	notImplemented("rescale");
}

void Engine::rescaleInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("rescaleInPlace");
}

// ---- Accumulate (sum reduction) ----
Ciphertext<DCRTPoly> Engine::accumulateSum(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int, int) {
	notImplemented("accumulateSum");
}

void Engine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int) {
	notImplemented("accumulateSumInPlace");
}

void Engine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, int) {
	notImplemented("accumulateSumInPlace");
}

// ---- Bootstrapping ----
BootstrapSetupPolicy Engine::bootstrapSetupPolicy(bool, bool, int32_t) const {
	notImplemented("bootstrapSetupPolicy");
}

void Engine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>&, const PrivateKey<DCRTPoly>&, uint32_t) {
	notImplemented("evalBootstrapKeyGen");
}

Ciphertext<DCRTPoly> Engine::evalBootstrap(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&, uint32_t, uint32_t, bool) {
	notImplemented("evalBootstrap");
}

void Engine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, uint32_t, uint32_t, bool) {
	notImplemented("evalBootstrapInPlace");
}

// VARIABLE-OUTPUT-LEVEL BOOTSTRAP (opt-in). The reference path: refresh the ciphertext the
// way the backend always does, then spend the levels the caller said it does not need. That
// is correct everywhere and cheaper nowhere -- it exists so the API has one meaning on every
// backend, and so a backend that CAN shorten its bootstrap only has to override the part it
// makes cheaper. See CudaEngine::evalBootstrapToLevel.
Ciphertext<DCRTPoly>
Engine::evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	Ciphertext<DCRTPoly> out = evalBootstrap(ctx, ciphertext, numIterations, precision, prescaled);
	dropToLevel(ctx, out, outputLevel);
	return out;
}

void Engine::evalBootstrapToLevelInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	evalBootstrapInPlace(ctx, ciphertext, numIterations, precision, prescaled);
	dropToLevel(ctx, ciphertext, outputLevel);
}

void Engine::dropToLevel(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, uint32_t level) {
	// Multiply by one, then rescale: the multiply is exact (the constant encodes to the
	// scaling factor with no rounding), and the rescale is the only operation that actually
	// removes a tower under an AUTO scaling technique. One pair per level, because a rescale
	// drops exactly one.
	while (ctx.CiphertextLevel(*ct) < level) {
		const size_t before = ctx.CiphertextLevel(*ct);
		ctx.EvalMultInPlace(ct, 1.0);
		// FIXEDAUTO rescales inside the multiply; the AUTO-with-lazy-rescale techniques leave
		// the ciphertext at noise scale degree 2 and need the rescale spelled out. Asking for
		// it only when the multiply did not already move the level covers both.
		if (ctx.CiphertextLevel(*ct) == before)
			ctx.RescaleInPlace(ct);
		if (ctx.CiphertextLevel(*ct) <= before)
			OPENFHE_THROW("FIDESlib: level drop made no progress at level " + std::to_string(before) +
						  " (the scaling technique does not rescale in place here)");
	}
}

// ---- Linear transform (fused BSGS matrix-vector product) ----
// CPU and CUDA override this; haze falls through to here.
void Engine::linearTransformInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, const std::vector<Plaintext>&, int, int, bool) {
	notImplemented("linearTransformInPlace");
}

// The reference for LinearTransformMany on every backend, and the CPU engine's actual
// implementation: N independent single transforms, each on its own clone of the shared source.
// There is nothing to share on a host backend — OpenFHE's EvalFastRotationPrecompute is per-call
// and its rotations key-switch straight back down to Q — so the loop is not a fallback, it IS the
// semantics the batched CUDA kernel has to reproduce exactly. Backends that do not implement the
// single transform (haze) throw from inside the first iteration, as they should.
std::vector<Ciphertext<DCRTPoly>> Engine::linearTransformMany(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  int rowSize,
  int bStep,
  const std::vector<std::vector<Plaintext>>& diagonalSets,
  int stride,
  int offset,
  bool ext) {
	std::vector<Ciphertext<DCRTPoly>> out;
	out.reserve(diagonalSets.size());
	for (const auto& diagonals : diagonalSets) {
		// Clone per transform: linearTransformInPlace mutates what it is handed, and the source is
		// shared by construction here.
		Ciphertext<DCRTPoly> y = ct->Clone();
		linearTransformInPlace(ctx, y, rowSize, bStep, diagonals, stride, offset, ext);
		out.push_back(std::move(y));
	}
	return out;
}

// ---- Prepared linear transforms ----

namespace {
// 64-bit FNV-1a over raw bytes. Used only to fold a plaintext's identity into one word; it is not
// a security primitive and nothing depends on it being collision-free against an adversary — only
// on it changing when the thing it describes changes.
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime  = 1099511628211ull;

inline uint64_t FoldWord(uint64_t h, uint64_t v) {
	for (int b = 0; b < 8; ++b) {
		h ^= static_cast<uint64_t>((v >> (8 * b)) & 0xffu);
		h *= kFnvPrime;
	}
	return h;
}
} // namespace

// THE REFERENCE IDENTITY, and the CPU engine's actual one.
//
// It answers one question: are the buffers a prepared table points at still the buffers it was
// built from? On a host backend there is no device table, so what stands in for it is the HOST
// ENCODING — and the fold below covers every way that encoding can change under a handle that
// holds the same PlaintextImpl:
//
//   * the object identity itself (a different plaintext is trivially a different fingerprint),
//   * the RNS shape: tower count and every tower's modulus and length — which is what a re-encode
//     at another level or another ring changes,
//   * the leading coefficients of every tower — which is what a re-encode of different VALUES at
//     the same level changes,
//   * the flags and the backend residency slot: `extended`, `device_only`, and whether `device`
//     has been populated (loading a plaintext to a device gives it buffers it did not have).
//
// It is O(towers) host reads and never touches the device, because it runs once per diagonal on
// every prepared call. A plaintext whose host value was RELEASED (`host` emptied, which is the
// state a backend free leaves behind) folds to a different value than the same plaintext folded
// while it held one, so a released diagonal is caught by the same check as a re-encoded one.
uint64_t Engine::plaintextIdentity(CryptoContextImpl<DCRTPoly>&, const Plaintext& pt) {
	uint64_t h = kFnvOffset;
	h		   = FoldWord(h, reinterpret_cast<uint64_t>(pt.get()));
	if (!pt)
		return h;
	h = FoldWord(h, pt->extended ? 0x9e37u : 0x1u);
	h = FoldWord(h, pt->device_only ? 0x85ebu : 0x2u);
	h = FoldWord(h, pt->device.has_value() ? 0xc2b2u : 0x3u);
	if (!pt->host.has_value())
		return FoldWord(h, 0xdeadbeefull);
	const auto& impl	  = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
	const auto& towers	  = impl->GetElement<lbcrypto::DCRTPoly>().GetAllElements();
	h					  = FoldWord(h, static_cast<uint64_t>(towers.size()));
	h					  = FoldWord(h, static_cast<uint64_t>(impl->GetLevel()));
	h					  = FoldWord(h, static_cast<uint64_t>(impl->GetSlots()));
	h					  = FoldWord(h, static_cast<uint64_t>(impl->GetNoiseScaleDeg()));
	for (const auto& t : towers) {
		const auto& v = t.GetValues();
		h			  = FoldWord(h, t.GetModulus().ConvertToInt());
		h			  = FoldWord(h, static_cast<uint64_t>(v.GetLength()));
		// A fixed, small prefix: enough to catch a re-encode of different values, cheap enough to
		// run on every call. The RNS shape above already catches every structural change.
		const size_t probe = std::min<size_t>(v.GetLength(), 8);
		for (size_t i = 0; i < probe; ++i)
			h = FoldWord(h, v[i].ConvertToInt());
	}
	return h;
}

// THE REFERENCE PREPARE, and the CPU/haze engines' actual one: capture the diagonals (by shared
// handle, so they cannot be freed underneath the result) and their identities, and nothing else.
// There is no host-side analogue of the device pointer table this exists to hoist, so `device`
// stays empty and the prepared calls below forward straight to the vector path.
PreparedLinearTransform
Engine::prepareLinearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<std::vector<Plaintext>>& diagonalSets, const LinearTransformShape& shape) {
	std::vector<std::vector<Plaintext>> kept;
	std::vector<std::vector<uint64_t>> identity;
	kept.reserve(diagonalSets.size());
	identity.reserve(diagonalSets.size());
	for (const auto& set : diagonalSets) {
		// Exactly rowSize diagonals per set: the transform never reads past rowSize, and the null
		// padding of the last giant step is the transform's to build, not the caller's.
		kept.emplace_back(set.begin(), set.begin() + shape.rowSize);
		std::vector<uint64_t> ids;
		ids.reserve(kept.back().size());
		for (const auto& pt : kept.back())
			ids.push_back(plaintextIdentity(ctx, pt));
		identity.push_back(std::move(ids));
	}
	return std::make_shared<PreparedLinearTransformImpl>(shape, std::move(kept), std::move(identity));
}

// The prepared calls ARE the vector calls on any backend that did not override prepare: same
// entry point, same diagonals, same arguments. Byte-identical output is structural here.
void Engine::linearTransformInPlacePrepared(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) {
	const auto& s = prepared.Shape();
	linearTransformInPlace(ctx, ct, s.rowSize, s.bStep, prepared.Diagonals(), s.stride, s.offset, s.ext);
}

std::vector<Ciphertext<DCRTPoly>>
Engine::linearTransformManyPrepared(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared) {
	const auto& s = prepared.Shape();
	return linearTransformMany(ctx, ct, s.rowSize, s.bStep, prepared.DiagonalSets(), s.stride, s.offset, s.ext);
}

// Extended (Q||P) plaintexts are a device-basis concept; a backend has to opt in.
bool Engine::supportsExtendedPlaintexts() const {
	return false;
}

// ---- Convolution transform (BSGS linear transform) ----
void Engine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, const std::vector<Plaintext>&, const std::vector<int>&, int, int) {
	notImplemented("convolutionTransformInPlace");
}

void Engine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&, int, int, const std::vector<Plaintext>&, Plaintext&, const std::vector<int>&, int, int, int) {
	notImplemented("specialConvolutionTransformInPlace");
}

// ---- Host readback ----
void Engine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("recoverHostCiphertext");
}

// No-op default: CPU and CUDA have no record/replay concept, so MarkOutput is a safe no-op on
// those backends. The haze backend overrides this to register ciphertexts for the final
// hazeTagOutput+hazeFlush batch in materialize(). The no-op default (rather than a throw) lets
// callers freely call MarkOutput regardless of backend.
void Engine::markOutput(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
}

// Device-side encode is a transfer optimisation, so a backend has to opt in — same reasoning as
// supportsExtendedPlaintexts above.
bool Engine::supportsDeviceEncode() const {
	return false;
}

// A throw, not a no-op: unlike markOutput this is not semantically neutral, and reaching it means a
// caller ignored supportsDeviceEncode and would otherwise get an empty plaintext.
void Engine::encodeDevice(CryptoContextImpl<DCRTPoly>&, Plaintext&, const DeviceEncodedCoefficients&) {
	notImplemented("encodeDevice");
}

// Narrower than supportsDeviceEncode on purpose: a backend opts into the batched entry point
// separately, and the facade loops the single call for everyone who does not.
bool Engine::supportsDeviceEncodeMany() const {
	return false;
}

// Same reasoning as encodeDevice: reaching this means a caller ignored supportsDeviceEncodeMany.
void Engine::encodeDeviceMany(CryptoContextImpl<DCRTPoly>&, std::vector<Plaintext>&, const std::vector<DeviceEncodedCoefficients>&) {
	notImplemented("encodeDeviceMany");
}

// The device transform is a CUDA arm; every other backend keeps the host transform.
bool Engine::supportsDeviceSlotsEncode() const {
	return false;
}

void Engine::encodeDeviceSlotsMany(CryptoContextImpl<DCRTPoly>&, std::vector<Plaintext>&, const DeviceEncodedSlots&) {
	notImplemented("encodeDeviceSlotsMany");
}

// A NO-OP, not a throw, and that is the whole contract: a backend with no device memory pool has
// nothing parked, so "released nothing" is the correct and complete answer rather than an
// unimplemented operation. It is what makes the call safe for an application to make
// unconditionally at its pass boundary, which is what keeps the CPU gates building.
ScratchReleaseStats Engine::releaseParkedScratch(CryptoContextImpl<DCRTPoly>&) {
	return {};
}

// ---- Device residency ----
void Engine::loadContext(CryptoContextImpl<DCRTPoly>&, const PublicKey<DCRTPoly>&) {
	notImplemented("loadContext");
}

void Engine::loadPlaintext(CryptoContextImpl<DCRTPoly>&, Plaintext&) {
	notImplemented("loadPlaintext");
}

void Engine::loadCiphertext(CryptoContextImpl<DCRTPoly>&, Ciphertext<DCRTPoly>&) {
	notImplemented("loadCiphertext");
}

// ---- Ciphertext backend hooks ----
std::any Engine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("cloneCiphertextBackend");
}

size_t Engine::ciphertextLevel(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextLevel");
}

size_t Engine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextNoiseScaleDeg");
}

double Engine::ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextScalingFactor");
}

size_t Engine::ciphertextSlots(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>&) {
	notImplemented("ciphertextSlots");
}

size_t Engine::ciphertextNumElements(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	return ct.GetNumElementsHost();
}

void Engine::refreshHostShadow(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&) {
	notImplemented("refreshHostShadow");
}

void Engine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&, size_t) {
	notImplemented("setCiphertextSlots");
}

void Engine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>&, size_t) {
	notImplemented("setCiphertextLevel");
}

// ---- Context backend state ----
bool Engine::isContextLoaded() const {
	notImplemented("isContextLoaded");
}

void Engine::synchronize() const {
	notImplemented("synchronize");
}

// The context destructor calls teardown() unconditionally, so the default must succeed (a backend with
// no device state simply has nothing to tear down) — a throw here would terminate the program.
void Engine::teardown() {
}

void Engine::setDevices(const std::vector<int>&) {
	notImplemented("setDevices");
}

std::vector<int> Engine::devices() const {
	notImplemented("devices");
}

} // namespace fideslib
