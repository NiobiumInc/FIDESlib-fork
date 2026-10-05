#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "ConcurrentOps.hpp"
#include "Definitions.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PreparedLinearTransform.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "key/evalkey-fwd.h"

namespace fideslib {

class Engine;

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);
	/// @brief Devices this context is configured to use (empty on the CPU backend).
	std::vector<int> GetCudaDevices() const;

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	/// @brief Select the CUDA devices this context loads onto (CUDA backend only).
	/// Must be called before LoadContext; forwarded to the engine.
	void SetCudaDevices(const std::vector<int>& devices);

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load.
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Configure the approximate modular reduction (EvalMod) stage of bootstrapping.
	/// Opt-in; the default-constructed config reproduces the library's stock behaviour. Must be
	/// called before EvalBootstrapSetup and before LoadContext, because the setup reserves the
	/// stage's level budget and LoadContext builds the device's coefficient table from it.
	void SetModReductionConfig(const ModReductionConfig& cfg);
	/// @brief The configuration in force for the mod-reduction stage.
	const ModReductionConfig& GetModReductionConfig() const { return modReduction; }
	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget = { 5, 4 },
	  std::vector<uint32_t> dim1									 = { 0, 0 },
	  uint32_t slots												 = 0,
	  uint32_t correctionFactor										 = 0,
	  bool precompute												 = true,
	  bool btsfirstboot												 = false);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Key accessors ----
	// Deliberate lbcrypto passthroughs: the facade owns the static eval-key map definitions
	// (CryptoContext.cpp:32-34), so there is no fideslib-typed key-map wrapper to return here
	// (precedent: Ciphertext::GetElements() also returns a raw lbcrypto type). An unknown keyTag
	// throws, matching OpenFHE's own behavior. The returned references are invalidated by this
	// context's destructor, which calls ClearEvalMultKeys/ClearEvalAutomorphismKeys.
	static const std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& GetEvalMultKeyVector(const std::string& keyTag);
	static std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& GetEvalAutomorphismKeyMap(const std::string& keyTag);
	/// @brief Vector-index -> automorphism-index mapping (lbcrypto passthrough), keyed on this
	/// context's cyclotomic order.
	uint32_t FindAutomorphismIndex(uint32_t index) const;

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);

	/// @brief Encode a CKKS plaintext in the EXTENDED (mod-up, Q||P) RNS basis, for use as a
	/// LinearTransformInPlace diagonal with `ext = true`.
	///
	/// WHAT IT IS FOR. The fused BSGS transform normally key-switches each baby rotation all the
	/// way back down to Q before multiplying it by a diagonal: one ModDown per baby step. If every
	/// diagonal already carries the special primes, the whole giant step — baby rotations and the
	/// MAC accumulation — stays in Q||P and pays ONE ModDown at the end of it instead. That is the
	/// only thing this encoder buys, and the transform is the only place it is valid.
	///
	/// WHAT CHANGES vs MakeCKKSPackedPlaintext. Only the RNS basis the same encoded polynomial is
	/// represented in: the moduli become Q_level || P (all of the context's special primes) rather
	/// than Q_level. Slots, values, scaling factor, noise scale degree and `level` are unchanged,
	/// and so is the scale of the transform's result — the trailing ModDown divides the P back out.
	/// It follows that the DIAGONAL LAYOUT CONTRACT below is untouched: same index-to-rotation map,
	/// same pre-rotation, same level rule. An application that prefetches and caches diagonals only
	/// swaps which encoder it calls; the packing code does not move.
	///
	/// COSTS. Each diagonal grows from `level+1` to `level+1+K` limbs on the device, K being the
	/// number of special primes, so a prefetched diagonal set gets proportionally bigger in device
	/// memory. The encode itself also does K more NTTs.
	///
	/// BACKENDS. Extended is a device-basis concept. On a backend without one (CPU, haze) this
	/// returns exactly what MakeCKKSPackedPlaintext returns — the ordinary Q-basis encoding, which
	/// computes the same result — while still marking the plaintext as extended so `ext = true`
	/// stays a valid, checkable request on every backend.
	///
	/// RESTRICTION. The result is only valid as a LinearTransformInPlace diagonal. Do not pass it
	/// to EvalMult/EvalAdd/Encrypt: on a device backend the extra P limbs participate in those ops
	/// and nothing performs the compensating ModDown.
	///
	/// @param value          Values to pack.
	/// @param noiseScaleDeg  Degree of the scaling factor, as for MakeCKKSPackedPlaintext.
	/// @param level          Level to encode at, in OpenFHE's sense (number of dropped towers).
	///                       Must satisfy the transform's level rule, same as an ordinary diagonal.
	/// @param slots          Number of packed slots; 0 means the context's batch size.
	Plaintext MakeCKKSPackedPlaintextExtended(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, uint32_t slots = 0);
	Plaintext MakeCKKSPackedPlaintextExtended(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, uint32_t slots = 0);

	/// @brief AT A PASS BOUNDARY, hand back everything the concurrent mode has parked.
	///
	/// Under FIDESLIB_CONCURRENT_OPS a Ciphertext draws its auxiliary polynomials from, and returns
	/// them to, the CALLING THREAD'S scratch slot. An application that creates intermediates on
	/// worker threads and destroys them on the thread that merges their results therefore parks
	/// polynomials on one slot's list while the workers' lists starve and keep constructing new
	/// ones, and a parked polynomial is LIVE device memory the pool can never reclaim. This gives
	/// them back, and with them every pool lane's freed blocks, of every size class, which
	/// otherwise stay private to the lane that freed them.
	///
	/// SAFE TO CALL UNCONDITIONALLY. In the default (single-issuing-thread) mode, on a CPU-only
	/// build and on any backend without a device memory pool it returns immediately with
	/// `ScratchReleaseStats::ran == false` and touches nothing — no synchronize, no lock, no
	/// counter. It never changes what any ciphertext or plaintext holds, so it is invisible to the
	/// arithmetic in every mode.
	///
	/// WHERE TO CALL IT, and this is a PRECONDITION the caller owns: at a point where NO OTHER
	/// THREAD IS ISSUING OPS on this context — a server that has published its response and is
	/// about to poll for the next request, a harness between passes. It destroys polynomials that
	/// belong to other threads' slots and it synchronizes the device, so calling it mid-pass would
	/// both race those threads and stall them. It is not a general-purpose "free some memory" call.
	ScratchReleaseStats ReleaseParkedScratch();

	/// @brief Whether this context's backend can encode a plaintext on the device.
	///
	/// Callers use this to decide whether MakeCKKSPackedPlaintextDevice is worth routing to; it is
	/// never required, because that function falls back to the ordinary encoding when it is false.
	bool SupportsDeviceEncode() const;

	/// @brief Encode a plaintext by uploading its COEFFICIENTS and building the limbs on the device.
	///
	/// Computes the same plaintext as MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr,
	/// slots) — bit for bit, not approximately — but splits the work differently. The host runs
	/// OpenFHE's own inverse special FFT (DiscreteFourierTransform::FFTSpecialInv, the transform
	/// CKKSPackedEncoding::Encode calls) and the scale-and-round loop that follows it, then hands
	/// the backend the resulting integer coefficient vector; the backend does the per-tower CRT
	/// reduction and the forward NTT. Host cost falls by the L modular-reduction passes and the L
	/// forward NTTs; transfer falls from L*N words to N.
	///
	/// WHY THE SPLIT IS HERE, and not further down: everything before the rounding is
	/// double-precision work whose result must match OpenFHE exactly, and `std::llround`'s
	/// half-away-from-zero rule, the `logc`/approxFactor magnitude probe and the FLEXIBLE*
	/// scaling-factor tables are all easier to get exactly right — and cheaper, being one pass over
	/// N values — on the host. Everything after it is integer arithmetic that reproduces exactly on
	/// any device, and is L times larger. Moving the boundary either way costs correctness or
	/// costs the saving.
	///
	/// Backends that answer false to SupportsDeviceEncode fall back to MakeCKKSPackedPlaintext,
	/// which has identical semantics, so the call is safe to make unconditionally. On a backend
	/// that answers true the returned plaintext is DEVICE-ONLY: its host slot is empty, because
	/// materialising it would mean doing on the host precisely the work this call exists to avoid.
	/// Device-only plaintexts are valid wherever only the device value is read — notably as
	/// LinearTransformInPlace diagonals — and throw loudly anywhere the host value is needed.
	///
	/// MEMORY TRADE: a device-encoded plaintext is built by transforming on the device, so it
	/// carries the NTT scratch that a host-encoded one does not, and occupies TWICE the device
	/// memory of the same plaintext loaded through LoadPlaintext. An application that caps how
	/// many plaintexts it keeps device-resident must halve that cap when this arm is on. The
	/// transfer saving (a factor of the live limb count) is unaffected.
	///
	/// @param value          Values to pack. Real only: the complex overload is deliberately absent,
	///                       since every device-encode caller in this library packs real weights.
	/// @param noiseScaleDeg  Must be 1 on a device-encoding backend; see the throw in
	///                       CudaEngine::encodeDevice for why > 1 is refused rather than approximated.
	/// @param level          Level to encode at, in OpenFHE's sense (number of dropped towers).
	/// @param slots          Number of packed slots; 0 means the context's batch size.
	Plaintext MakeCKKSPackedPlaintextDevice(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, uint32_t slots = 0);

	/// @brief Whether this context's backend has a BATCHED device encode worth routing to.
	///
	/// Narrower than SupportsDeviceEncode: a backend can encode on the device one plaintext at a
	/// time and still gain nothing from a batch. Never required — MakeCKKSPackedPlaintextDeviceMany
	/// loops the single call when this is false, and returns the same plaintexts.
	bool SupportsDeviceEncodeMany() const;

	/// @brief MakeCKKSPackedPlaintextDevice for a whole group of value vectors at ONE level.
	///
	/// Element for element, `out[i]` is bit-identical to
	/// `MakeCKKSPackedPlaintextDevice(values[i], noiseScaleDeg, level, slots)`. That is the entire
	/// semantic content of this call: it exists only to change how often the host talks to the
	/// backend, never what the backend computes.
	///
	/// WHAT IS ACTUALLY SHARED. The host half is NOT shared — every value vector still gets its own
	/// inverse special FFT and its own scale-and-round pass, producing exactly the same integers as
	/// the single call, because that is the half whose bit-exactness against OpenFHE is the whole
	/// contract. What the backend may share is everything downstream of it: on the CUDA backend the
	/// batch travels in ONE host-to-device copy out of one pinned staging buffer instead of N
	/// pageable ones, and the per-tower CRT reduction and the forward NTT run as one launch over
	/// (batch x limbs x N) instead of once per plaintext. On a backend that answers false to
	/// SupportsDeviceEncodeMany this is a plain loop over the single call, so applications can turn
	/// the arm on unconditionally and the CPU gates stay a behavioural no-op.
	///
	/// ONE LEVEL PER BATCH, by signature: `level` and `slots` are scalars. That matches the caller
	/// this exists for — a BSGS giant step is a group of diagonals at one level — and it is what
	/// lets the backend index one modulus chain for the whole batch.
	///
	/// LIFETIMES ARE INDEPENDENT. Each returned plaintext owns its own device storage and can be
	/// released on its own; nothing in the batch keeps a sibling's memory alive.
	///
	/// Every per-plaintext property of MakeCKKSPackedPlaintextDevice carries over unchanged: the
	/// results are DEVICE-ONLY on a device-encoding backend (empty host slot, loud throw on a host
	/// read), noiseScaleDeg must be 1 there, and the device memory trade is the same.
	///
	/// @param values         One value vector per plaintext. Must not be empty, and no element may be.
	/// @param noiseScaleDeg  As MakeCKKSPackedPlaintextDevice.
	/// @param level          The level EVERY plaintext of the batch is encoded at.
	/// @param slots          Number of packed slots; 0 means the context's batch size.
	std::vector<Plaintext> MakeCKKSPackedPlaintextDeviceMany(const std::vector<std::vector<double>>& values, size_t noiseScaleDeg = 1, uint32_t level = 0, uint32_t slots = 0);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);
	// Refresh ct's host shadow (ct->host) with the backend's current value, syncing it back from the
	// device if resident (without evicting — the device copy stays resident), so the underlying
	// lbcrypto::Ciphertext can be recovered without decrypting. Used by the api-vs-OpenFHE parity
	// tests; the readback itself is backend-specific (see Engine).
	void RecoverHostCiphertext(Ciphertext<DCRTPoly>& ct);

	/// @brief Declare ct as a program output for record/replay backends (e.g. haze); no-op on
	/// CPU/CUDA. Must be called before the first Decrypt/RecoverHostCiphertext of any result when
	/// multiple outputs need to be read back — so all outputs are registered before the single
	/// hazeFlush that materializes them. On CPU/CUDA this is a no-op and may be safely called in
	/// all backends without a runtime cost.
	void MarkOutput(Ciphertext<DCRTPoly>& ct);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	/// @brief Tensor product with NO key-switch: a degree-2 (3-poly) result. Must be relinearized
	/// before any op that requires a degree-1 ciphertext. Performs the same level/depth
	/// adjustment as EvalMult (OpenFHE semantics) before the tensor product; there is no
	/// unadjusted "EvalMultNoRelinNoCheck" variant, deliberately.
	Ciphertext<DCRTPoly> EvalMultNoRelin(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	/// @brief Key-switch a degree-2 ciphertext back down to degree-1; a no-op on a degree-1 input.
	Ciphertext<DCRTPoly> Relinearize(const Ciphertext<DCRTPoly>& ct);
	/// @brief In-place form of Relinearize().
	void RelinearizeInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	/// @brief ONE rotation index applied to MANY INDEPENDENT ciphertexts.
	///
	/// Returns `r` with `r.size() == cts.size()` and, for every i,
	///
	///     r[i]  ==  EvalRotate(cts[i], index)
	///
	/// BYTE-IDENTICAL, not merely numerically equal, on whichever backend is active and under
	/// either secret distribution. `cts` itself is not modified.
	///
	/// WHY: this is the TRANSPOSE of the hoisting contract. EvalFastRotationPrecompute /
	/// EvalFastRotation share the digit decomposition of ONE source across MANY indices; the
	/// doubling reduction ladders of an attention score walk have the opposite shape — every step
	/// rotates a FRESHLY produced ciphertext, so nothing hoists, but the B independent bodies of
	/// one K/V page all take the SAME index at the same step. What those share is not the source
	/// but the KEY: one rotation key's digit-NTTs, one automorphism, B different inputs. A backend
	/// with a batched key switch loads the key once and issues one inner-product launch for the
	/// whole batch instead of B of them. Backends without one run the reference loop and get the
	/// same answer at the same cost as calling EvalRotate B times.
	///
	/// PRECONDITIONS: every entry non-null and degree 1 (as EvalRotate requires), and a rotation
	/// key for `index` generated by EvalRotateKeyGen. Entries may sit at different levels — a
	/// backend whose batched kernel needs one common level falls back to the loop for such a batch,
	/// since the contract above is unconditional.
	///
	/// COST: one launch needs every input and every output live at once, so a batched backend holds
	/// 2*B ciphertexts plus B digit decompositions where the serial loop holds 2. A caller near its
	/// device-memory budget batches narrower; the answer does not change with the batch width.
	std::vector<Ciphertext<DCRTPoly>> EvalRotateMany(const std::vector<Ciphertext<DCRTPoly>>& cts, int32_t index);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);

	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);

	/// @brief Bootstrap to a CHOSEN output level, instead of to the one the bootstrap circuit
	///        happens to leave.
	///
	/// EvalBootstrap returns a ciphertext at the level its own circuit ends on — the level
	/// budget plus the EvalMod depth below the top of the chain — whatever the caller is
	/// going to do with it. A caller that needs only a few more multiplications before its
	/// next refresh can say so here, and the backend is free to run the whole refresh on the
	/// shorter modulus chain that implies, which costs proportionally fewer limb operations
	/// in CoeffsToSlots, EvalMod and SlotsToCoeffs.
	///
	/// CONTRACT. The result is at exactly max(@p outputLevel, the level EvalBootstrap would
	/// have returned) — a bootstrap cannot hand back more levels than its circuit leaves, so
	/// a request for a shallower level (0 included) is the same as calling EvalBootstrap.
	/// Levels are OpenFHE's: the number of consumed towers, as Ciphertext::GetLevel() reports
	/// it, so a LARGER outputLevel means FEWER usable levels and fewer limbs.
	///
	/// PRECISION is the bootstrap's own, unchanged: the mod-raise is a tower-0 reinterpret
	/// that is exact at any chain height, and the EvalMod interval and Chebyshev degree do
	/// not depend on the height either. Backends that cannot shorten the raise fall back to
	/// bootstrapping and then spending the levels, which is the same ciphertext by a slower
	/// route.
	///
	/// Default behaviour is untouched: this is a separate entry point, and EvalBootstrap
	/// keeps the exact code path and output level it always had.
	///
	/// @param outputLevel  The level to return the refreshed ciphertext at.
	Ciphertext<DCRTPoly> EvalBootstrapToLevel(const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapToLevelInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);


	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	/// @brief Fused baby-step/giant-step matrix-vector product: the plaintext-diagonal linear
	/// transform, in place.
	///
	/// Computes, over the `slots` packed slots of `ct`:
	///
	///     ct <- sum_{k = 0}^{rowSize-1}  Rot(ct, k*stride + offset)  *  D_k
	///
	/// where `D_k` is the k-th LOGICAL diagonal — the coefficient vector that multiplies the
	/// ciphertext rotated by `k*stride + offset`. `Rot(x, r)` is the CKKS left rotation (slot i of
	/// the result is slot i+r of the input, cyclic over the packed slots), i.e. EvalRotate's sign.
	///
	/// For a plain n-by-n matrix-vector product M*v with rowSize = n, stride = 1, offset = 0, that
	/// makes D_k the ordinary k-th diagonal: D_k[i] = M[i][(i + k) mod n].
	///
	/// DIAGONAL LAYOUT CONTRACT (the caller pre-rotates; this is not optional):
	///   - `diagonals` is indexed by k in [0, rowSize). Index k <-> ciphertext rotation
	///     `k*stride + offset`. `diagonals.size()` must be >= rowSize; entries at index >= rowSize
	///     are never read. Pass exactly rowSize entries — the CUDA kernel asserts every element of
	///     the vector it is handed is non-null, so a padded tail must still hold real plaintexts.
	///   - Every entry in [0, rowSize) must be a non-null Plaintext; a null one throws here rather
	///     than tripping a device-side assert. (Null slots ARE used inside the transform: the last
	///     giant step is null-padded internally out to bStep*gStep. That padding is built by this
	///     call, not supplied by the caller.)
	///   - `diagonals[k]` must encode `Rot(D_k, r_k)` with
	///     `r_k = -bStep * (k / bStep) * stride - offset`, i.e. exactly
	///     `GetLinearTransformPlaintextRotationIndices(rowSize, bStep, stride, offset)[k]`.
	///     The fold rotates the running sum by `bStep*stride` once per giant step and by `offset`
	///     at the very end, so each diagonal has to be counter-rotated by however much rotation it
	///     will see after it is multiplied in.
	///   - Rotation keys: the caller must have run EvalRotateKeyGen for every index in
	///     `GetLinearTransformRotationIndices(bStep, stride, offset)` =
	///     {i*stride : i in [1, bStep]} plus `offset` when non-zero.
	///   - Levels: all diagonals must be encoded at the level the ciphertext is at when the
	///     transform starts multiplying — that is `ct`'s level, plus one if `ct` arrives at noise
	///     scale degree 2, since the transform rescales such an input first (CUDA asserts this
	///     equality). Encoding every diagonal at the same level is required; they are all consumed
	///     at one level.
	///   - `bStep` must be >= 1; the giant-step count is `gStep = ceil(rowSize / bStep)`. Nothing
	///     requires bStep to divide rowSize.
	///
	/// RESULT METADATA: the ciphertext keeps its level and comes back at noise scale degree 2
	/// (un-rescaled), which is what the CUDA kernel leaves behind; both backends agree. Under an
	/// AUTO scaling technique the next operation absorbs it; under FIXEDMANUAL call RescaleInPlace.
	///
	/// @param ext  Assert that the diagonals were built by MakeCKKSPackedPlaintextExtended, so the
	///   transform may keep the baby rotations and the MAC accumulation in the extended Q||P basis
	///   and pay ONE ModDown per giant step instead of one per baby rotation. This is a
	///   REQUIREMENT, not a mode switch: the fused kernel decides by inspecting the diagonals, and
	///   would otherwise quietly take the slow path when they are not extended. So passing true
	///   with an ordinary diagonal throws — here if the plaintext was never asked to be extended,
	///   and in the CUDA engine if it was asked but did not reach the device mod-up. Everything
	///   about the layout contract above is unchanged by `ext`; only the RNS basis of the
	///   diagonals differs. Backends with no extended plaintext basis (CPU, haze) still validate
	///   the flag and then ignore it — their result is identical either way.
	///
	/// Backends: CPU (reference composition over OpenFHE rotations/mults) and CUDA (one hoisted
	/// rotation + one fused MAC kernel). Not implemented on haze.
	void LinearTransformInPlace(Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride = 1, int offset = 0, bool ext = false);

	/// @brief Several INDEPENDENT linear transforms of ONE source ciphertext.
	///
	/// Returns `r` with `r.size() == diagonalSets.size()` and
	///
	///     r[t] = sum_{k = 0}^{rowSize-1}  Rot(ct, k*stride + offset)  *  diagonalSets[t][k]
	///
	/// i.e. exactly what `rowSize`/`bStep`/`stride`/`offset`/`ext` LinearTransformInPlace calls on
	/// `diagonalSets.size()` clones of `ct` would produce, one per set. Every rule of the
	/// LinearTransformInPlace contract above — diagonal layout, pre-rotation, encode level, rotation
	/// keys, `ext`, result metadata — applies unchanged to each set, and the shared shape parameters
	/// are what make one call possible: the sets differ only in their diagonal VALUES.
	///
	/// `ct` itself is not modified.
	///
	/// WHY: the transforms share a source, so they share its baby-step rotations. A backend with a
	/// batched kernel does the hoisted key-switch precompute and its bStep rotations ONCE for the
	/// whole batch instead of once per transform, and issues one MAC launch instead of N. That is
	/// the q/k/v (one x) and gate/up (one x2) shape of a transformer block. Backends without one
	/// run the reference loop and get the same answer at the same cost as calling N times.
	///
	/// COST: one launch needs everything live at once. A batched backend holds
	/// `diagonalSets.size() * gStep` partial-result ciphertexts where the single transform holds
	/// `gStep`, and the caller is holding `diagonalSets.size() * rowSize` plaintexts where it would
	/// otherwise hold `rowSize`. Both scale with the batch width, so batching is not free in device
	/// memory even though it is free in rotations — a caller near its budget should batch narrower.
	std::vector<Ciphertext<DCRTPoly>> LinearTransformMany(const Ciphertext<DCRTPoly>& ct,
	  int rowSize,
	  int bStep,
	  const std::vector<std::vector<Plaintext>>& diagonalSets,
	  int stride = 1,
	  int offset = 0,
	  bool ext	 = false);

	// ---- Prepared linear transforms ----

	/// @brief Freeze (shape + diagonals) into a handle that OWNS the backend's per-call setup.
	///
	/// Same contract as LinearTransformInPlace in every respect — diagonal layout, pre-rotation,
	/// encode level, rotation keys, `ext` — and the arguments mean exactly what they mean there.
	/// What this adds is RESIDENCE: the work a backend redoes on every call purely because the
	/// diagonals arrive as a fresh vector each time is done ONCE, here, and reused by every
	/// prepared call. On the CUDA backend that is the device pointer table the batched MAC kernel
	/// reads; on host backends there is nothing to hoist and the handle is a thin wrapper whose
	/// calls forward to the vector path (identical result, identical cost).
	///
	/// THE HANDLE KEEPS THE DIAGONALS ALIVE (it holds their shared handles) and records what each
	/// one looked like when it was prepared. If a diagonal is later re-encoded or released, every
	/// prepared call THROWS naming it. It never falls back to the vector path and never reads the
	/// old pointers: a stale entry in that table is a wrong answer, not a slowdown.
	///
	/// WHEN TO USE IT: when the same diagonals serve many transforms — a resident weight bank. A
	/// one-shot transform should call LinearTransformInPlace; preparing costs one device
	/// allocation and one upload and would never be amortised.
	///
	/// @param diagonals  At least `rowSize` non-null encoded diagonals; entries in [0, rowSize)
	///   are what the handle captures. @param ext  As LinearTransformInPlace: a REQUIREMENT, not a
	///   hint, checked here and again on the device.
	PreparedLinearTransform
	PrepareLinearTransform(const std::vector<Plaintext>& diagonals, int rowSize, int bStep, int stride = 1, int offset = 0, bool ext = false);

	/// @brief PrepareLinearTransform for a whole BATCH: one handle covering every set of a
	/// LinearTransformMany call.
	///
	/// The batch is the unit on purpose. LinearTransformMany's win is that the hoisted key-switch
	/// precompute, its baby rotations and the MAC launch are shared across the transforms, and a
	/// backend can only keep that if one resident table covers the whole batch — a handle per
	/// transform would force the batch apart and trade one optimisation for another. So the handle
	/// is prepared for the group of sets that will go out together, and
	/// `LinearTransformMany(ct, prepared)` takes exactly one of them.
	///
	/// Every rule of PrepareLinearTransform applies to every set: same shape for all of them (that
	/// is what makes one call legal), diagonals held alive, identities recorded, stale means throw.
	PreparedLinearTransform PrepareLinearTransformMany(const std::vector<std::vector<Plaintext>>& diagonalSets,
	  int rowSize,
	  int bStep,
	  int stride = 1,
	  int offset = 0,
	  bool ext	 = false);

	/// @brief LinearTransformInPlace against a prepared handle. Identical result, by construction.
	/// The handle must carry exactly one diagonal set (PrepareLinearTransform). Throws if it is
	/// null, is a batch handle, or any of its diagonals has moved since it was prepared.
	void LinearTransformInPlace(Ciphertext<DCRTPoly>& ct, const PreparedLinearTransform& prepared);

	/// @brief LinearTransformMany against ONE prepared batch handle. Identical result.
	///
	/// Returns `prepared->Sets()` ciphertexts, in set order — exactly what the vector overload
	/// returns for the same diagonals. Throws if @p prepared is null or any of its diagonals has
	/// moved since it was prepared.
	std::vector<Ciphertext<DCRTPoly>> LinearTransformMany(const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransform& prepared);

	/// @brief A cheap, backend-defined fingerprint of what @p pt currently IS.
	///
	/// The staleness seam of PrepareLinearTransform: equal fingerprints mean the buffers a prepared
	/// table points at are the ones it was built from. On a device backend it folds the diagonal's
	/// device limb-pointer-array addresses and RNS basis; on a host backend it folds the encoding
	/// itself. Host-side reads only — no device traffic, no synchronisation — because it runs once
	/// per diagonal on every prepared call.
	uint64_t PlaintextIdentity(const Plaintext& pt);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	/// @brief Canonical host value (lbcrypto::CryptoContext); the backend-neutral context.
	std::any host;
	/// @brief Backend engine this context dispatches operations to (CPU or CUDA). Owns all backend state.
	std::unique_ptr<Engine> engine_;
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Bootstrap slots available.
	std::vector<uint32_t> slots_bootstrap;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;
	/// @brief Opt-in overrides for the bootstrap's approximate modular reduction stage. Zero-valued
	/// by default, which is the stock derivation from `keyDist` alone.
	ModReductionConfig modReduction{};

	// ---- Ciphertext backend hooks ----
	// Thin pass-throughs the value type uses to reach its backend without naming any device type.
	std::any CloneCiphertextBackend(const CiphertextImpl<DCRTPoly>& src);
	size_t CiphertextLevel(const CiphertextImpl<DCRTPoly>& ct);
	size_t CiphertextNoiseScaleDeg(const CiphertextImpl<DCRTPoly>& ct);
	double CiphertextScalingFactor(const CiphertextImpl<DCRTPoly>& ct);
	size_t CiphertextSlots(const CiphertextImpl<DCRTPoly>& ct);
	size_t CiphertextNumElements(const CiphertextImpl<DCRTPoly>& ct);
	void RefreshCiphertextHost(CiphertextImpl<DCRTPoly>& ct);
	void SetCiphertextSlots(CiphertextImpl<DCRTPoly>& ct, size_t slots);
	void SetCiphertextLevel(CiphertextImpl<DCRTPoly>& ct, size_t level);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);

	/// @brief Ciphertext rotation indices LinearTransformInPlace needs keys for: feed these to
	/// EvalRotateKeyGen (together with whatever else the circuit rotates by) before calling it.
	static std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride = 1, int offset = 0);

	/// @brief Rotation to apply to each logical diagonal before encoding it, indexed by the same k
	/// as the `diagonals` vector. See the LinearTransformInPlace layout contract.
	static std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride = 1, int offset = 0);
};

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP