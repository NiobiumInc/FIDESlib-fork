#ifndef API_ENGINE_HPP
#define API_ENGINE_HPP

#include "Ciphertext.hpp"
#include "ConcurrentOps.hpp"
#include "Plaintext.hpp"
#include "PreparedLinearTransform.hpp"
#include "PublicKey.hpp"
#include "engine/Backend.hpp"

#include <any>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fideslib {

template <typename T> class CryptoContextImpl;

/// @brief Per-backend arguments for the host-side OpenFHE EvalBootstrapSetup call, which the facade
/// performs. The engine only supplies this policy; it does no bootstrap-setup work itself.
struct BootstrapSetupPolicy {
	bool precompute;
	bool btSlotsEncoding;
	int32_t modEvalLevels;
};

/// @brief The host half of a device encode: everything
/// CryptoContextImpl::MakeCKKSPackedPlaintextDevice computes before the backend takes over.
///
/// The split point is deliberate. Everything up to and including the rounding is floating-point
/// work whose result must match OpenFHE bit for bit, and it costs one pass over N values; the work
/// that follows is L passes of modular reduction plus L forward NTTs, and is pure integer
/// arithmetic. So the host keeps the part that is cheap and precision-critical, and the device
/// takes the part that is expensive and exactly reproducible. The wire between them is one vector
/// of N words instead of the L*N words an encoded plaintext occupies.
struct DeviceEncodedCoefficients {
	/// @brief Exactly ringDim biased coefficients: OpenFHE's `temp` array, already scattered to
	/// full ring length so that a sparse encoding's `gap` stride and zero fill are baked in.
	/// A signed coefficient c is carried as `c < 0 ? bigBound + c : c`.
	std::vector<uint64_t> coefficients;
	/// @brief The Q primes of the target level, in tower order. Q basis only.
	std::vector<uint64_t> moduli;
	/// @brief OpenFHE's Max64BitValue() — the bias constant `coefficients` was built with. Carried
	/// explicitly so that a host/device disagreement is a parameter error rather than silent noise.
	uint64_t bigBound = 0;
	/// @brief Plaintext metadata, identical to what CKKSPackedEncoding would have recorded.
	double scalingFactor   = 0.0;
	uint32_t noiseScaleDeg = 1;
	uint32_t level		   = 0;
	uint32_t slots		   = 0;
};

/// @brief The host half of a device encode that stops BEFORE the transform: the slot values a batch
/// of plaintexts was asked to carry, plus the same parameters DeviceEncodedCoefficients records. A
/// backend that answers true to supportsDeviceSlotsEncode runs FFTSpecialInv, the scale, the rounding
/// and the bias on the device and then reduces and NTTs as encodeDeviceMany does.
struct DeviceEncodedSlots {
	/// @brief One pointer per plaintext, each to exactly `slots` reals (the caller keeps them alive
	/// for the duration of the call). Imaginary parts are zero: REAL contexts only.
	std::vector<const std::vector<double>*> values;
	std::vector<uint64_t> moduli; ///< the Q primes of the target level, in tower order
	uint64_t bigBound	   = 0;	  ///< Max64BitValue(), the bias constant the device applies
	double scalingFactor   = 0.0;
	uint32_t noiseScaleDeg = 1;
	uint32_t level		   = 0;
	uint32_t slots		   = 0;
};

/// @brief One implementation per backend. Stateless: each method receives the
/// owning context. A backend's method bodies live entirely in its own
/// translation unit (engine/cpu for the CPU/OpenFHE backend, engine/cuda for the
/// CUDA backend), so CPU and CUDA code are separated by file rather than
/// interleaved per function.
class Engine {
  protected:
	/// @brief Backing for the default implementations: throws "<op> is not implemented by the
	/// <name()> backend".
	[[noreturn]] void notImplemented(const char* op) const;

  public:
	// ---- Lifecycle ----
	virtual ~Engine()				 = default;
	virtual const char* name() const = 0;
	/// @brief Which backend this engine is (for explicit serialization; never inferred from devices).
	virtual Backend backend() const = 0;

	// ---- Negation ----
	virtual Ciphertext<DCRTPoly> evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual void evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Addition ----
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual Ciphertext<DCRTPoly> evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	virtual void evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	// ---- Subtraction ----
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1);

	// ---- Multiplication ----
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	/// @brief Tensor product with NO key-switch: a degree-2 (3-poly) result. Level/depth adjust the
	/// same way as evalMult; the caller is responsible for relinearizing before any op that requires
	/// a degree-1 ciphertext.
	virtual Ciphertext<DCRTPoly> evalMultNoRelin(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	/// @brief Key-switch a degree-2 ciphertext's c2 term back into (c0, c1). Metadata is unchanged;
	/// a no-op on an already degree-1 input.
	virtual Ciphertext<DCRTPoly> relinearize(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	/// @brief In-place form of relinearize().
	virtual void relinearizeInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Square ----
	virtual Ciphertext<DCRTPoly> evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual void evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Rotation ----
	virtual Ciphertext<DCRTPoly> evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	virtual void evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	virtual std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual Ciphertext<DCRTPoly>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp);
	virtual Ciphertext<DCRTPoly>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	virtual std::vector<Ciphertext<DCRTPoly>>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp);
	virtual std::vector<Ciphertext<DCRTPoly>>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	/// @brief One index, many independent ciphertexts. See CryptoContextImpl::EvalRotateMany for
	/// the contract; this is its backend hook.
	///
	/// The base implementation is the REFERENCE and is correct on every backend: a loop of
	/// evalRotate, which is what the CPU and haze engines take as-is. A backend overrides it only
	/// to share the rotation key's digit-NTTs and the key-switch inner product across the batch;
	/// the result must be byte-identical either way.
	virtual std::vector<Ciphertext<DCRTPoly>> evalRotateMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& cts, int32_t index);

	// ---- Chebyshev series ----
	virtual Ciphertext<DCRTPoly> evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	virtual void evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);

	// ---- Rescale ----
	virtual Ciphertext<DCRTPoly> rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext);
	virtual void rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext);

	// ---- Accumulate (sum reduction) ----
	virtual Ciphertext<DCRTPoly> accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride);
	virtual void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride);
	virtual void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	// ---- Bootstrapping ----
	/// @brief Per-backend arguments for the facade's host EvalBootstrapSetup call (the setup itself is a
	/// host computation and lives on the facade, not here).
	virtual BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const;
	virtual void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);
	virtual Ciphertext<DCRTPoly>
	evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled);
	virtual void evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled);

	/// @brief Bootstrap and return the result at a CHOSEN output level. See
	/// CryptoContextImpl::EvalBootstrapToLevel for the contract; this is its backend hook.
	///
	/// The base implementation is the REFERENCE and is correct on every backend: bootstrap
	/// as usual, then spend the levels the caller does not want with exact multiply-by-one +
	/// rescale steps. A backend overrides it only when it can make the bootstrap ITSELF
	/// cheaper at the reduced height; the result must be the same either way.
	virtual Ciphertext<DCRTPoly>
	evalBootstrapToLevel(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled);
	virtual void evalBootstrapToLevelInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled);

	/// @brief Drop @p ct to @p level exactly (multiply by one, then rescale, once per level).
	/// Shared by evalBootstrapToLevel's reference path and any backend that finishes a
	/// shortened bootstrap with a residual drop. A ciphertext already at or below @p level is
	/// left alone.
	void dropToLevel(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, uint32_t level);

	// ---- Linear transform (fused BSGS matrix-vector product) ----
	// Diagonal layout, level and key requirements are documented once, on
	// CryptoContextImpl::LinearTransformInPlace; the engines implement that contract.
	// `ext`: the caller built the diagonals with MakeCKKSPackedPlaintextExtended and wants the
	// extended-basis fast path. Backends without an extended plaintext basis ignore it (the math
	// is identical); backends with one must fail loudly rather than silently fall back.
	virtual void
	linearTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool ext);

	/// @brief Several independent transforms of ONE source ciphertext. See
	/// CryptoContextImpl::LinearTransformMany for the contract.
	///
	/// The base implementation is the REFERENCE: a loop of linearTransformInPlace over a clone of
	/// the source per diagonal set, which is what every backend without a batched kernel wants
	/// (the CPU engine takes it as-is). A backend overrides it only to share work across the
	/// transforms; the result must be identical either way.
	virtual std::vector<Ciphertext<DCRTPoly>> linearTransformMany(CryptoContextImpl<DCRTPoly>& ctx,
	  const Ciphertext<DCRTPoly>& ct,
	  int rowSize,
	  int bStep,
	  const std::vector<std::vector<Plaintext>>& diagonalSets,
	  int stride,
	  int offset,
	  bool ext);

	// ---- Prepared linear transforms ----
	// See CryptoContextImpl::PrepareLinearTransform for the contract. The three hooks below have
	// REFERENCE implementations here that are correct on every backend and are what the CPU and
	// haze engines use verbatim: prepare captures the diagonals and their identities and nothing
	// else, and each prepared call forwards to the corresponding vector entry point with exactly
	// the diagonals the handle holds. So on a backend that does not override them the prepared
	// path is not a second implementation of the transform — it IS the vector path, which is why
	// "identical result" is a structural fact there and not a test result.

	/// @brief Build the handle over one or more diagonal sets (one = LinearTransformInPlace's
	/// form, N = a LinearTransformMany batch). A backend overrides this to attach resident backend
	/// state (the CUDA engine: the device pointer table) to `handle->device`.
	virtual PreparedLinearTransform
	prepareLinearTransform(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<std::vector<Plaintext>>& diagonalSets, const LinearTransformShape& shape);

	/// @brief Cheap fingerprint of a plaintext's CURRENT identity; see
	/// CryptoContextImpl::PlaintextIdentity. Host-side reads only — it runs once per diagonal on
	/// every prepared call, so it must never touch the device or synchronise.
	virtual uint64_t plaintextIdentity(CryptoContextImpl<DCRTPoly>& ctx, const Plaintext& pt);

	/// @brief linearTransformInPlace against a prepared handle. The facade has already checked the
	/// handle is non-null and fresh.
	virtual void linearTransformInPlacePrepared(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared);

	/// @brief linearTransformMany against ONE prepared batch handle. The facade has already
	/// checked it is non-null and fresh.
	virtual std::vector<Ciphertext<DCRTPoly>>
	linearTransformManyPrepared(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransformImpl& prepared);

	/// @brief Whether this backend has a device-side extended (mod-up, Q||P) plaintext basis.
	///
	/// False by default, and false for every host-arithmetic backend: "extended" is a property of
	/// the RNS basis a device keeps its limbs in, not of the CKKS math. Only backends that answer
	/// true get an extended encoding out of MakeCKKSPackedPlaintextExtended; the others get the
	/// ordinary Q-basis encoding, which computes exactly the same thing.
	virtual bool supportsExtendedPlaintexts() const;

	// ---- Convolution transform (BSGS linear transform) ----
	virtual void convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  const std::vector<int>& indexes,
	  int stride,
	  int rowSize);
	virtual void specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride,
	  int maskRotationStride,
	  int rowSize);

	// ---- Host readback ----
	// Refresh ct->host with the backend's current value, syncing it back from the device if resident, so a
	// caller can recover the underlying lbcrypto::Ciphertext without decrypting. This does NOT evict —
	// ct->device stays resident for further ops. CPU: no-op (already on host). CUDA: store() +
	// GetOpenFHECipherText, growing the host container to the device limb count when needed (e.g. after a
	// level-raising ModRaise).
	virtual void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	/// @brief Declare ct as a program output for record/replay backends (haze). No-op by default.
	/// CPU and CUDA inherit this no-op default unchanged; haze overrides it to register the
	/// ciphertext in its outputs_ list so materialize() can hazeTagOutput + hazeFlush all declared
	/// outputs in one batch. A no-op default (rather than a throw) is the correct choice here:
	/// MarkOutput is semantically safe on every backend and callers should not need to guard it.
	virtual void markOutput(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	/// @brief Whether this backend can build a plaintext's limbs on the device from a coefficient
	/// vector, instead of receiving limbs the host already encoded.
	///
	/// False by default and false for every host-arithmetic backend, for the same reason
	/// supportsExtendedPlaintexts is: the saving being claimed is transfer volume into a device, so
	/// a backend with no transfer has nothing to opt into. Callers query this to decide whether to
	/// route through MakeCKKSPackedPlaintextDevice; backends answering false hand back an ordinary
	/// MakeCKKSPackedPlaintext, which computes exactly the same plaintext.
	virtual bool supportsDeviceEncode() const;

	/// @brief Finish a device encode: reduce `enc.coefficients` against each of `enc.moduli` and
	/// NTT, leaving the result in `pt`'s backend slot.
	///
	/// Only ever called on a backend that answered true to supportsDeviceEncode. Preconditions
	/// (coefficient count, Q-basis moduli, bias constant) are the backend's to enforce, loudly —
	/// nothing downstream re-derives the limbs, so a violated precondition would otherwise surface
	/// only as a plaintext that decrypts to noise.
	virtual void encodeDevice(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const DeviceEncodedCoefficients& enc);

	/// @brief Whether this backend can device-encode a WHOLE BATCH of plaintexts in one call.
	///
	/// A strictly narrower claim than supportsDeviceEncode: a backend may build limbs on the device
	/// one plaintext at a time and still have nothing to gain from a batched entry point. False by
	/// default, and answering false costs a caller nothing — the facade's Many call then loops the
	/// single call, which produces exactly the same plaintexts.
	virtual bool supportsDeviceEncodeMany() const;

	/// @brief Finish a batch of device encodes: the encodeDevice contract, once per element.
	///
	/// `pts` and `encs` are parallel and non-empty, every `enc` shares one modulus chain (the caller
	/// resolves one level for the batch), and every `pt` is fresh and not yet device-resident. On
	/// return every `pt` holds exactly the backend value encodeDevice would have given it from the
	/// same `enc` — the batching is a call-count optimisation, never a numerical one.
	///
	/// Only ever called on a backend that answered true to supportsDeviceEncodeMany. ALL-OR-NOTHING:
	/// a throw must leave no plaintext half-built, because the caller's fallback is to encode the
	/// whole group again.
	virtual void encodeDeviceMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const std::vector<DeviceEncodedCoefficients>& encs);

	/// @brief Whether this backend runs the CKKS inverse special FFT on the device for a batch of
	/// plaintexts (DeviceEncodedSlots). False by default; CUDA answers FIDESlib::DeviceIfft().
	virtual bool supportsDeviceSlotsEncode() const;

	/// @brief Transform, scale, round, bias, reduce and NTT a whole batch on the device from its slot
	/// values. Only ever called on a backend that answered true to supportsDeviceSlotsEncode.
	/// ALL-OR-NOTHING like encodeDeviceMany: a throw leaves every plaintext's device slot empty.
	virtual void encodeDeviceSlotsMany(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Plaintext>& pts, const DeviceEncodedSlots& enc);

	/// @brief Hand back everything the concurrent mode has parked, at an application pass boundary.
	///
	/// Returns zeros with `ran == false` by default, which is the right answer for every backend
	/// with no device memory pool — there is nothing parked and nothing to release. Only the CUDA
	/// backend overrides it, and only when FIDESLIB_CONCURRENT_OPS is on; see
	/// FIDESlib::CKKS::ContextData::releaseParkedScratch for what is released and why.
	///
	/// PRECONDITION the caller owns: no other thread is issuing ops on this context. That is what
	/// a pass boundary means here, and no backend can check it.
	virtual ScratchReleaseStats releaseParkedScratch(CryptoContextImpl<DCRTPoly>& ctx);

	// ---- Device residency (CUDA-only; the CPU backend implements these as no-ops) ----
	virtual void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey);
	virtual void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt);
	virtual void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Ciphertext backend hooks ----
	// The neutral value type holds only its canonical host value plus an opaque `device` slot; all
	// backend manipulation of that slot happens here. The CPU backend reads the host value; the CUDA
	// backend reads/writes ct.device. The value types reach these through thin facade methods.
	// Clone the backend-resident payload of `src` into a fresh `device` slot (empty any == not resident).
	virtual std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src);
	virtual size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	virtual size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	virtual double ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	virtual size_t ciphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	/// @brief OpenFHE NumberCiphertextElements (2 normally, 3 after EvalMultNoRelin). The default reads
	/// the host shadow directly; this must NEVER trigger readback or a record/replay flush.
	virtual size_t ciphertextNumElements(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	/// @brief Sync ct's host shadow with the backend's current value, without declaring an output
	/// or triggering a record/replay flush. This is the value-type-level counterpart of
	/// recoverHostCiphertext, which additionally owns the haze MarkOutput/materialize step and so
	/// needs the owning shared_ptr. CPU: no-op. CUDA: pure D2H. haze: pure D2H, and a hard error if
	/// the value has not been flushed yet — GetElements() must never force a program flush.
	virtual void refreshHostShadow(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct);
	virtual void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots);
	virtual void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level);

	// ---- Context backend state (no ctx argument; each context owns its engine) ----
	virtual bool isContextLoaded() const;
	virtual void synchronize() const;
	/// @brief Default is a no-op, not a throw: the context destructor calls this unconditionally.
	virtual void teardown();
	virtual void setDevices(const std::vector<int>& devices);
	virtual std::vector<int> devices() const;
};

} // namespace fideslib

#endif
