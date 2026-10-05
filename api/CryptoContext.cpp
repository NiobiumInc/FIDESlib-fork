#include "CryptoContext.hpp"

#include "HostMath.hpp"

#include "Definitions.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "ciphertext-fwd.h"
#include "cryptocontext-fwd.h"
#include "engine/Engine.hpp"
#include "engine/EngineCommon.hpp"
#include "lattice/hal/lat-backend.h"

#include <algorithm>
#include <any>
#include <mutex>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <openfhe.h>
// The host half of the device encode reuses OpenFHE's own encoding pieces rather than restating
// them: FFTSpecialInv is the transform CKKSPackedEncoding::Encode calls, and Max64BitValue /
// is64BitOverflow / MAX_BITS_IN_WORD are the exact constants it rounds and range-checks against.
#include <constants-defs.h>
#include <math/dftransform.h>
#include <utils/utilities.h>
// Serialization headers - required for cereal type registration.
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-cryptoparameters.h>
#include <scheme/ckksrns/ckksrns-fhe.h> // FHECKKSRNS::MakeAuxPlaintext, for the extended (Q||P) encoding
#include <scheme/ckksrns/ckksrns-ser.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

template <> std::map<std::string, std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
template <>
std::map<std::string, std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalAutomorphismKeyMap;

namespace fideslib {

static std::unordered_map<PKESchemeFeature, lbcrypto::PKESchemeFeature> PKESchemeFeatureMap = {
	{ PKESchemeFeature::PKE, lbcrypto::PKE },
	{ PKESchemeFeature::KEYSWITCH, lbcrypto::KEYSWITCH },
	{ PKESchemeFeature::PRE, lbcrypto::PRE },
	{ PKESchemeFeature::LEVELEDSHE, lbcrypto::LEVELEDSHE },
	{ PKESchemeFeature::ADVANCEDSHE, lbcrypto::ADVANCEDSHE },
	{ PKESchemeFeature::MULTIPARTY, lbcrypto::MULTIPARTY },
	{ PKESchemeFeature::FHE, lbcrypto::FHE },
	{ PKESchemeFeature::SCHEMESWITCH, lbcrypto::SCHEMESWITCH },
};

CryptoContextImpl<DCRTPoly>::~CryptoContextImpl() {
	// Tear down backend state (e.g. GPU context) before clearing the OpenFHE eval keys.
	// engine_ is null on a moved-from context.
	if (engine_) {
		engine_->teardown();
	}
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();
}

// ---- Enable features ----

void CryptoContextImpl<DCRTPoly>::Enable(PKESchemeFeature feature) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	context->Enable(PKESchemeFeatureMap[feature]);
}

void CryptoContextImpl<DCRTPoly>::Enable(uint32_t featureMask) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	context->Enable(featureMask);
}

// ---- Getters ----

uint32_t CryptoContextImpl<DCRTPoly>::GetCyclotomicOrder() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	return context->GetCyclotomicOrder();
}

uint32_t CryptoContextImpl<DCRTPoly>::GetRingDimension() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	return context->GetRingDimension();
}

double CryptoContextImpl<DCRTPoly>::GetPreScaleFactor(uint32_t /*slots*/) {
	// Pure host computation (a single scalar derived from the crypto parameters + the bootstrap
	// correction factor); it touches no device, so it lives here rather than on the engine. This
	// reproduces the GPU GetPreScaleFactor exactly, reading the OpenFHE host parameters in place of
	// that function's device-mirrored copies (see engine-host-audit.md). `slots` is unused:
	// OpenFHE keeps a single m_correctionFactor (the last EvalBootstrapSetup wins), which is what the
	// device copy mirrors — callers set up bootstrap for one slot count.
	auto& context			 = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto cryptoParams	 = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	const auto& elementParam = cryptoParams->GetElementParams()->GetParams();
	const size_t L			 = elementParam.size() - 1;

	const double qDouble = elementParam[0]->GetModulus().ConvertToDouble(); // q0 (FIDESlib prime[0])
	const double powP	 = std::pow(2.0, static_cast<double>(cryptoParams->GetPlaintextModulus()));
	const int32_t deg	 = static_cast<int32_t>(std::round(std::log2(qDouble / powP)));

	const uint32_t correctionFactor = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE)->GetCKKSBootCorrectionFactor();
	const uint32_t correction		= correctionFactor - deg;

	const auto st = cryptoParams->GetScalingTechnique();
	if (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) {
		const uint32_t lvl		= (st == lbcrypto::FLEXIBLEAUTOEXT) ? 1U : 0U;
		const double targetSF	= cryptoParams->GetScalingFactorReal(lvl);		   // FIDESlib SFReal[L-lvl]
		const double sourceSF	= cryptoParams->GetScalingFactorReal(L - 1);	   // FIDESlib SFReal[1]
		const double modToDrop	= elementParam[1]->GetModulus().ConvertToDouble(); // FIDESlib prime[1]
		double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
		adjustmentFactor *= std::pow(2.0, -1.0 * static_cast<double>(correction));
		return adjustmentFactor;
	}
	return std::pow(2.0, -1.0 * static_cast<double>(correction));
}

// ---- Setters ----

void CryptoContextImpl<DCRTPoly>::SetAutoLoadPlaintexts(bool autoload) {
	this->auto_load_plaintexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetAutoLoadCiphertexts(bool autoload) {
	this->auto_load_ciphertexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetCudaDevices(const std::vector<int>& devices) {
	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("SetCudaDevices must be called before LoadContext");
	}
	engine_->setDevices(devices);
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetCudaDevices() const {
	return engine_->devices();
}

// ---- Load to devices ----

void CryptoContextImpl<DCRTPoly>::LoadContext(const PublicKey<DCRTPoly>& publicKey) {
	engine_->loadContext(*this, publicKey);
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt) {
	engine_->loadPlaintext(*this, pt);
}

void CryptoContextImpl<DCRTPoly>::LoadCiphertext(Ciphertext<DCRTPoly>& ct) {
	engine_->loadCiphertext(*this, ct);
}

// ---- Key Generation ----

namespace {

// SPARSE_ENCAPSULATED SECURITY TRIPWIRE.
//
// stock KeyGenInternal (base-pke.cpp) draws the MAIN secret at Hamming weight 192 for
// SPARSE_ENCAPSULATED — the ~123-bit sparse instance encapsulation exists to avoid. We keep the
// main key UNIFORM by forcing UNIFORM_TERNARY around the draw (see KeyGen below). This guard PROVES
// that worked: it counts the nonzero coefficients of the main secret and refuses to hand back a key
// that is not uniform while a real (>=128-bit) security level is claimed. A uniform-ternary secret
// has ~2N/3 nonzero coefficients; the h=192 landmine has exactly 192, which is far below N/4 for
// every ring this library uses. This makes the landmine impossible to reintroduce silently.
void AssertMainSecretUniform(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk) {
	const auto cryptoParams =
	  std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(sk->GetCryptoParameters());
	const bool securityClaimed =
	  cryptoParams && cryptoParams->GetStdLevel() != lbcrypto::HEStd_NotSet;

	lbcrypto::DCRTPoly s = sk->GetPrivateElement();
	s.SetFormat(Format::COEFFICIENT);
	const auto& poly0	 = s.GetElementAtIndex(0);
	const size_t N		 = poly0.GetLength();
	size_t weight		 = 0;
	const lbcrypto::NativeInteger zero(0);
	for (size_t i = 0; i < N; ++i) {
		if (poly0[i] != zero) ++weight;
	}

	if (securityClaimed && weight < N / 4) {
		OPENFHE_THROW(
		  std::string("SPARSE_ENCAPSULATED SECURITY GUARD FIRED: the main secret key has Hamming "
					  "weight ")
		  + std::to_string(weight) + " out of ring dimension " + std::to_string(N)
		  + " — this is the h=192 sparse instance from stock KeyGenInternal, NOT the uniform main "
			"key encapsulation requires (expected ~2N/3 = ~"
		  + std::to_string(2 * N / 3)
		  + "). A >=128-bit security level was claimed, so this is a silent ~123-bit downgrade. "
			"Refusing to return the key. (The uniform-forcing around KeyGen must have been skipped "
			"— e.g. FIDESLIB_ENCAPS_FORCE_SPARSE_MAIN_KEY was set, which is a test-only override.)");
	}
}

} // namespace

KeyPair<DCRTPoly> CryptoContextImpl<DCRTPoly>::KeyGen() {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);

	// SPARSE_ENCAPSULATED: keep the MAIN key UNIFORM (see CCParams.cpp / AssertMainSecretUniform).
	// Temporarily force the host params to UNIFORM_TERNARY around the secret draw so stock's
	// KeyGenInternal takes its uniform branch instead of the h=192 sparse branch, then restore
	// SPARSE_ENCAPSULATED for the bootstrap machinery (EvalBootstrapKeyGen keys off it to emit the
	// 2N-2 / 2N-4 switching keys and EvalBootstrap keys off it to run the KeySwitchSparse dance).
	const bool encaps = (this->keyDist == SPARSE_ENCAPSULATED);
	// TEST-ONLY override: leaving this env var set skips the uniform-forcing, so the draw hits the
	// h=192 landmine and the guard below fires. Used by the encaps security probe to prove the
	// tripwire works. NEVER set it in a real deployment.
	const bool forceSparseForTest =
	  encaps && std::getenv("FIDESLIB_ENCAPS_FORCE_SPARSE_MAIN_KEY") != nullptr;

	std::shared_ptr<lbcrypto::CryptoParametersCKKSRNS> cryptoParams;
	if (encaps) {
		cryptoParams =
		  std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	}
	if (encaps && cryptoParams && !forceSparseForTest) {
		cryptoParams->SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	}

	auto keys = context->KeyGen();

	if (encaps && cryptoParams) {
		cryptoParams->SetSecretKeyDist(lbcrypto::SPARSE_ENCAPSULATED);
		AssertMainSecretUniform(keys.secretKey);
	}

	KeyPair<DCRTPoly> keypair;
	keypair.publicKey = std::make_shared<PublicKeyImpl<DCRTPoly>>();
	keypair.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();

	keypair.publicKey->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(keys.publicKey);
	keypair.secretKey->pimpl = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(keys.secretKey);

	return keypair;
}

void CryptoContextImpl<DCRTPoly>::EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk) {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalMultKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalMultKeyGen(skImpl);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps) {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalRotateKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalRotateKeyGen(skImpl, steps);
	this->rotation_indexes.insert(this->rotation_indexes.end(), steps.begin(), steps.end());
}

// ---- Bootstrapping ----

void CryptoContextImpl<DCRTPoly>::SetModReductionConfig(const ModReductionConfig& cfg) {
	// The device context caches on the coefficient table (Parameters::operator<), and the host
	// reservation is baked into the bootstrap precomputation, so neither may already exist.
	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("SetModReductionConfig must be called before LoadContext");
	}
	modReduction = cfg;
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor, bool precompute, bool btsfirstboot) {
	// Bootstrap setup is a host (OpenFHE) computation; the facade performs it and the engine only
	// supplies the per-backend argument policy.
	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalBootstrapSetup must be called before LoadContext");
	}
	auto& context		   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	int32_t modall		   = bootstrapModEvalLevels(this->keyDist, this->modReduction);
	BootstrapSetupPolicy p = engine_->bootstrapSetupPolicy(precompute, btsfirstboot, modall);
	context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, p.precompute, p.btSlotsEncoding, p.modEvalLevels);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	engine_->evalBootstrapKeyGen(*this, secretKey, slots);
}

// ---- Serialization ----

bool CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(std::ostream& ser, const fideslib::SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case fideslib::SerType::BINARY: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::SerializeEvalMultKey(ser, lbcrypto::SerType::BINARY, keyTag); break;
	case fideslib::SerType::JSON: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::SerializeEvalMultKey(ser, lbcrypto::SerType::JSON, keyTag); break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case SerType::BINARY: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::SerializeEvalAutomorphismKey(ser, lbcrypto::SerType::BINARY, keyTag); break;
	case SerType::JSON: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::SerializeEvalAutomorphismKey(ser, lbcrypto::SerType::JSON, keyTag); break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Key accessors ----

const std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(const std::string& keyTag) {
	return lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalMultKeyVector(keyTag);
}

std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(const std::string& keyTag) {
	return lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMap(keyTag);
}

uint32_t CryptoContextImpl<DCRTPoly>::FindAutomorphismIndex(uint32_t index) const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	return context->FindAutomorphismIndex(index);
}

// ---- Deserialization ----

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("DeserializeEvalMultKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::DeserializeEvalMultKey(ser, lbcrypto::SerType::BINARY); break;
	case SerType::JSON: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::DeserializeEvalMultKey(ser, lbcrypto::SerType::JSON); break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("DeserializeEvalAutomorphismKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::DeserializeEvalAutomorphismKey(ser, lbcrypto::SerType::BINARY); break;
	case SerType::JSON: res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::DeserializeEvalAutomorphismKey(ser, lbcrypto::SerType::JSON); break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Encoding ----

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
  size_t noiseScaleDeg,
  uint32_t level,
  const std::shared_ptr<void> params,
  uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->host		= std::make_any<lbcrypto::Plaintext>(pt);

	if (!this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->host		= std::make_any<lbcrypto::Plaintext>(pt);

	if (!this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

namespace {

/// Element params for the extended basis at `level`: the surviving Q towers followed by EVERY
/// special prime P. Both halves matter to the device: RNSPoly::loadConstant (RNSPoly.cpp:909-946)
/// splits the incoming limbs by matching moduli — a prefix that equals the context's Q primes in
/// order, then a tail it matches against the special primes one by one. A tail prime the device
/// does not know is dropped silently while the polynomial is still marked mod-up, which would
/// leave a special limb uninitialized, so the tail must be exactly the context's P set.
///
/// This is the same construction OpenFHE uses for its own extended-basis linear-transform
/// plaintexts (FHECKKSRNS::EvalLinearTransformPrecompute, ckksrns-fhe.cpp:1379-1425) — the ones
/// FIDESlib already consumes as bootstrap precomputations and runs the ext path on.
std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>> ExtendedElementParams(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, uint32_t level) {

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	if (!cryptoParams) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextExtended: context is not CKKS-RNS");
	}
	const auto paramsP = cryptoParams->GetParamsP();
	if (!paramsP || paramsP->GetParams().empty()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextExtended: context has no special (P) primes, so there is no extended basis");
	}

	lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> elementParams = *(cryptoParams->GetElementParams());
	if (level >= elementParams.GetParams().size()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextExtended: level " + std::to_string(level) + " drops every tower (only " + std::to_string(elementParams.GetParams().size()) +
		  " available)");
	}
	for (uint32_t i = 0; i < level; ++i)
		elementParams.PopLastParam();

	const auto& q = elementParams.GetParams();
	const auto& p = paramsP->GetParams();
	std::vector<lbcrypto::NativeInteger> moduli(q.size() + p.size());
	std::vector<lbcrypto::NativeInteger> roots(q.size() + p.size());
	for (size_t i = 0; i < q.size(); ++i) {
		moduli[i] = q[i]->GetModulus();
		roots[i]  = q[i]->GetRootOfUnity();
	}
	for (size_t i = 0; i < p.size(); ++i) {
		moduli[q.size() + i] = p[i]->GetModulus();
		roots[q.size() + i]	 = p[i]->GetRootOfUnity();
	}

	return std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>(context->GetCyclotomicOrder(), moduli, roots);
}

/// The host half of a device encode: OpenFHE's CKKSPackedEncoding::Encode
/// (src/pke/lib/encoding/ckkspackedencoding.cpp) run up to — but not including — FitToNativeVector,
/// which is the step that expands one coefficient vector into L towers and the only step the device
/// takes over. Every line below is a deliberate transcription of that function, and the comments
/// mark the places where getting it "obviously right" would in fact be wrong:
///
///   * the imaginary parts are zeroed BEFORE the transform, not after — the CKKSPackedEncoding
///     constructor does it, so a REAL plaintext's iFFT input is purely real;
///   * FFTSpecialInv normalises by vals.size(), i.e. by `slots`, NOT by N and NOT by m/4. Those
///     coincide only for a fully packed plaintext, so passing a padded vector would silently change
///     the answer for sparse packings. We resize to exactly `slots` first, as Encode does;
///   * the scaling factor is applied AFTER the transform;
///   * rounding is std::llround (half away from zero), applied independently to re and im;
///   * the bias constant is Max64BitValue() = 2^63 - 2^9 - 1, not 2^63;
///   * reals land in the first half of the `2*slots` coefficient positions and imags in the second,
///     scattered with stride gap = N/(2*slots) — two contiguous blocks, not an interleave.
///
/// Two of Encode's branches are refused rather than reproduced, because both would require
/// reproducing OpenFHE's own double arithmetic (not merely its integer arithmetic) to stay
/// bit-exact, and neither is reachable from this library's callers: the approxFactor rescue for
/// coefficients wider than MAX_BITS_IN_WORD, and noiseScaleDeg > 1.
DeviceEncodedCoefficients EncodeCoefficientsForDevice(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(context->GetCryptoParameters());
	if (!cryptoParams) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: context is not CKKS-RNS");
	}
	if (value.empty()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: cannot encode an empty value vector");
	}

	// ---- level -> element params (MakeCKKSPackedPlaintextInternal, cryptocontext.h:372-416) ----
	const size_t numModuli = cryptoParams->GetElementParams()->GetParams().size();
	if (level >= numModuli) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: level " + std::to_string(level) + " drops every tower (only " + std::to_string(numModuli) + " available)");
	}
	lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> elemParams = *(cryptoParams->GetElementParams());
	for (uint32_t i = 0; i < level; ++i) {
		elemParams.PopLastParam();
	}
	const uint32_t ringDim = elemParams.GetRingDimension();

	// ---- scaling factor (cryptocontext.h:393-402) ----
	// FLEXIBLEAUTOEXT at level 0 uses the BIG factor and forces noiseScaleDeg to 1; every other
	// case reads GetScalingFactorReal(level), which is level-dependent under the FLEXIBLE* and
	// COMPOSITE* techniques and a flat 2^p under FIXEDMANUAL/FIXEDAUTO.
	double scFact = 0.0;
	if (cryptoParams->GetScalingTechnique() == lbcrypto::FLEXIBLEAUTOEXT && level == 0) {
		scFact		  = cryptoParams->GetScalingFactorRealBig(level);
		noiseScaleDeg = 1;
	} else {
		scFact = cryptoParams->GetScalingFactorReal(level);
	}
	if (noiseScaleDeg != 1) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: only noiseScaleDeg == 1 is supported (got " + std::to_string(noiseScaleDeg) +
		  "); a higher degree multiplies the encoding by llround(scalingFactor)^(deg-1) in CRT, which is not reproduced here");
	}

	// ---- slots (CKKSPackedEncoding::GetDefaultSlotSize, ckkspackedencoding.h:254-266) ----
	if (slots == 0) {
		const uint32_t batch = context->GetEncodingParams()->GetBatchSize();
		slots				 = (batch == 0) ? (ringDim >> 1) : batch;
	}
	if ((slots & (slots - 1)) != 0) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: the number of slots should be a power of two");
	}
	if (slots > (ringDim >> 1)) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: the number of slots cannot be larger than half of ring dimension");
	}
	if (slots < value.size()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: the number of slots [" + std::to_string(slots) + "] is less than the size of data [" + std::to_string(value.size()) + "]");
	}

	// ---- inverse special FFT (ckkspackedencoding.cpp:128-132) ----
	// Imaginary parts are already zero: this overload takes reals, which is exactly the state the
	// CKKSPackedEncoding constructor forces for CKKSDataType::REAL. A COMPLEX-typed context would
	// need the imaginary halves carried through, so refuse rather than silently drop them.
	if (context->GetCKKSDataType() != lbcrypto::REAL) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: only CKKSDataType::REAL contexts are supported");
	}
	std::vector<std::complex<double>> inverse(value.size());
	std::transform(value.begin(), value.end(), inverse.begin(), [](double d) { return std::complex<double>(d); });
	inverse.resize(slots);
	// FFTSpecialInv lazily builds its twiddle tables per cyclotomic order and
	// THROWS if Initialize was not called first — a data race when prefetch
	// workers hit the first encode concurrently (observed: recursive terminate
	// on the GPU runner). Initialize once, before any worker can race it.
	{
		static std::once_flag dft_init;
		std::call_once(dft_init, [&] { lbcrypto::DiscreteFourierTransform::Initialize(ringDim * 2, ringDim / 2); });
	}
	lbcrypto::DiscreteFourierTransform::FFTSpecialInv(inverse, ringDim * 2);

	// ---- scale + magnitude probe (ckkspackedencoding.cpp:190-214) ----
	int32_t logc = std::numeric_limits<int32_t>::min();
	for (uint32_t i = 0; i < slots; ++i) {
		inverse[i] *= scFact;
		if (inverse[i].real() != 0.) {
			const auto logci = static_cast<int32_t>(std::ceil(std::log2(std::abs(inverse[i].real()))));
			if (logc < logci)
				logc = logci;
		}
		if (inverse[i].imag() != 0.) {
			const auto logci = static_cast<int32_t>(std::ceil(std::log2(std::abs(inverse[i].imag()))));
			if (logc < logci)
				logc = logci;
		}
	}
	logc = (logc == std::numeric_limits<int32_t>::min()) ? 0 : logc;
	if (logc < 0) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: scaling factor too small");
	}
	if (logc > static_cast<int32_t>(lbcrypto::MAX_BITS_IN_WORD)) {
		// Encode's approxFactor path would divide the coefficients down here and multiply the CRT
		// polynomial back up afterwards. Refused rather than reproduced: it is unreachable at this
		// library's scaling factors, and a wrong reproduction would be invisible.
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: coefficient magnitude needs OpenFHE's approxFactor path (logc " + std::to_string(logc) + " > " +
		  std::to_string(static_cast<int32_t>(lbcrypto::MAX_BITS_IN_WORD)) + "); use MakeCKKSPackedPlaintext");
	}

	// ---- round, bias, scatter (ckkspackedencoding.cpp:219-284 + FitToNativeVector's `gap`) ----
	const int64_t maxBitValue = lbcrypto::Max64BitValue();
	const uint32_t gap		  = ringDim / (2 * slots);

	DeviceEncodedCoefficients enc;
	enc.coefficients.assign(ringDim, 0); // untouched positions stay 0, as NativeVector's zero-init does
	for (uint32_t i = 0; i < slots; ++i) {
		const double dre = inverse[i].real();
		const double dim = inverse[i].imag();
		if (lbcrypto::is64BitOverflow(dre) || lbcrypto::is64BitOverflow(dim)) {
			OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: overflow, try to decrease scaling factor");
		}
		const int64_t re = std::llround(dre);
		const int64_t im = std::llround(dim);

		enc.coefficients[gap * i]			= static_cast<uint64_t>((re < 0) ? maxBitValue + re : re);
		enc.coefficients[gap * (i + slots)] = static_cast<uint64_t>((im < 0) ? maxBitValue + im : im);
	}

	const auto& towers = elemParams.GetParams();
	enc.moduli.reserve(towers.size());
	for (const auto& t : towers) {
		enc.moduli.push_back(t->GetModulus().ConvertToInt());
	}

	enc.bigBound = static_cast<uint64_t>(maxBitValue);
	// Encode ends with scalingFactor = pow(scFact, noiseScaleDeg); noiseScaleDeg is 1 here.
	enc.scalingFactor = scFact;
	enc.noiseScaleDeg = 1;
	enc.level		  = level;
	enc.slots		  = slots;
	return enc;
}

/// The parameter half of EncodeCoefficientsForDevice, for a batch whose TRANSFORM runs on the
/// device (FIDESLIB_DEVICE_IFFT): the same level -> element params, scaling factor, slot and
/// data-type checks, with the same refusals, and no transform. Returns false (take the host path)
/// when a value vector is not exactly `slots` long: Encode pads shorter vectors with zeros before
/// the transform, and the device path uploads the vectors as they are.
bool SlotsForDevice(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, const std::vector<std::vector<double>>& values, size_t noiseScaleDeg, uint32_t level, uint32_t slots,
  DeviceEncodedSlots& enc) {
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(context->GetCryptoParameters());
	if (!cryptoParams) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: context is not CKKS-RNS");
	}
	const size_t numModuli = cryptoParams->GetElementParams()->GetParams().size();
	if (level >= numModuli) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: level " + std::to_string(level) + " drops every tower (only " + std::to_string(numModuli) + " available)");
	}
	lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> elemParams = *(cryptoParams->GetElementParams());
	for (uint32_t i = 0; i < level; ++i) {
		elemParams.PopLastParam();
	}
	const uint32_t ringDim = elemParams.GetRingDimension();
	double scFact		   = 0.0;
	if (cryptoParams->GetScalingTechnique() == lbcrypto::FLEXIBLEAUTOEXT && level == 0) {
		scFact		  = cryptoParams->GetScalingFactorRealBig(level);
		noiseScaleDeg = 1;
	} else {
		scFact = cryptoParams->GetScalingFactorReal(level);
	}
	if (noiseScaleDeg != 1) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: only noiseScaleDeg == 1 is supported (got " + std::to_string(noiseScaleDeg) + ")");
	}
	if (slots == 0) {
		const uint32_t batch = context->GetEncodingParams()->GetBatchSize();
		slots				 = (batch == 0) ? (ringDim >> 1) : batch;
	}
	if ((slots & (slots - 1)) != 0) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: the number of slots should be a power of two");
	}
	if (slots > (ringDim >> 1)) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: the number of slots cannot be larger than half of ring dimension");
	}
	if (context->GetCKKSDataType() != lbcrypto::REAL) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: only CKKSDataType::REAL contexts are supported");
	}
	for (const std::vector<double>& value : values) {
		if (slots < value.size()) {
			OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: the number of slots [" + std::to_string(slots) + "] is less than the size of data [" + std::to_string(value.size()) + "]");
		}
		if (value.size() != slots) {
			return false; // Encode would zero-pad; the host path does that, the device path does not
		}
	}
	enc.values.clear();
	enc.values.reserve(values.size());
	for (const std::vector<double>& value : values) {
		enc.values.push_back(&value);
	}
	const auto& towers = elemParams.GetParams();
	enc.moduli.clear();
	enc.moduli.reserve(towers.size());
	for (const auto& t : towers) {
		enc.moduli.push_back(t->GetModulus().ConvertToInt());
	}
	enc.bigBound	  = static_cast<uint64_t>(lbcrypto::Max64BitValue());
	enc.scalingFactor = scFact;
	enc.noiseScaleDeg = 1;
	enc.level		  = level;
	enc.slots		  = slots;
	return true;
}
} // namespace

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintextExtended(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {

	if (value.empty()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextExtended: cannot encode an empty value vector");
	}

	// Backends with no extended device basis get the ordinary encoding: same values, same scale,
	// same level, and the same result out of the transform. Only the `extended` mark carries over,
	// so the caller's `ext = true` remains a valid request everywhere.
	if (!engine_->supportsExtendedPlaintexts()) {
		Plaintext plaintext	 = this->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);
		plaintext->extended	 = true;
		return plaintext;
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);

	// MakeAuxPlaintext indexes its own encode loop by `slots` and does not resolve 0 the way the
	// CKKSPackedEncoding constructor does, so resolve it here to the same default.
	if (slots == 0) {
		const uint32_t batch = context->GetEncodingParams()->GetBatchSize();
		slots				 = (batch == 0) ? (context->GetRingDimension() >> 1) : batch;
	}

	auto pt = lbcrypto::FHECKKSRNS::MakeAuxPlaintext(*context, ExtendedElementParams(context, level), value, noiseScaleDeg, level, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->host		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->extended = true;

	if (!this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintextExtended(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {

	std::vector<std::complex<double>> complexValue(value.size());
	std::transform(value.begin(), value.end(), complexValue.begin(), [](double d) { return std::complex<double>(d); });
	return this->MakeCKKSPackedPlaintextExtended(complexValue, noiseScaleDeg, level, slots);
}

bool CryptoContextImpl<DCRTPoly>::SupportsDeviceEncode() const {
	return engine_->supportsDeviceEncode();
}

// A straight forward to the backend, like SupportsDeviceEncode above: the facade has no parked
// state of its own, and every decision (concurrent mode or not, context loaded or not) belongs to
// the backend that owns the scratch. The base Engine answers with zeros, so this is a no-op on
// every backend but CUDA.
ScratchReleaseStats CryptoContextImpl<DCRTPoly>::ReleaseParkedScratch() {
	return engine_->releaseParkedScratch(*this);
}

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintextDevice(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {

	if (value.empty()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDevice: cannot encode an empty value vector");
	}

	// A backend with no device has nothing to save; hand back the ordinary encoding, which computes
	// exactly the same plaintext. This is what makes the call safe to make unconditionally, and it
	// is why the CPU gates stay a behavioural no-op when an application turns this arm on.
	if (!engine_->supportsDeviceEncode()) {
		return this->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);

	const DeviceEncodedCoefficients enc = EncodeCoefficientsForDevice(context, value, noiseScaleDeg, level, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	// host stays EMPTY: materialising it means running the encode this call exists to skip. The
	// flag is what turns any host-side read into a clear error instead of a bad_any_cast.
	plaintext->device_only = true;

	engine_->encodeDevice(*this, plaintext, enc);

	return plaintext;
}

bool CryptoContextImpl<DCRTPoly>::SupportsDeviceEncodeMany() const {
	return engine_->supportsDeviceEncodeMany();
}

std::vector<Plaintext> CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintextDeviceMany(const std::vector<std::vector<double>>& values, size_t noiseScaleDeg, uint32_t level,
  uint32_t slots) {

	if (values.empty()) {
		OPENFHE_THROW("MakeCKKSPackedPlaintextDeviceMany: cannot encode an empty batch");
	}

	// A backend with no batched encode gets the LOOP — which is not a degraded path but the
	// definition of the call: the results are the same objects the caller would have built one at a
	// time, and MakeCKKSPackedPlaintextDevice itself falls back to MakeCKKSPackedPlaintext on a
	// backend with no device at all. That chain is what makes this safe to call unconditionally and
	// what keeps the CPU gates a behavioural no-op when an application turns the arm on.
	if (!engine_->supportsDeviceEncodeMany()) {
		std::vector<Plaintext> plaintexts;
		plaintexts.reserve(values.size());
		for (const std::vector<double>& value : values) {
			plaintexts.push_back(this->MakeCKKSPackedPlaintextDevice(value, noiseScaleDeg, level, slots));
		}
		return plaintexts;
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);

	// FIDESLIB_DEVICE_IFFT (opt-in): the transform runs on the device too. The parameter checks are
	// the host path's; a batch the device path cannot take as-is (a vector shorter than `slots`)
	// falls through to the host transform below and produces the same plaintexts.
	if (engine_->supportsDeviceSlotsEncode()) {
		DeviceEncodedSlots enc;
		if (SlotsForDevice(context, values, noiseScaleDeg, level, slots, enc)) {
			std::vector<Plaintext> plaintexts;
			plaintexts.reserve(values.size());
			for (size_t i = 0; i < values.size(); ++i) {
				Plaintext plaintext	  = std::make_shared<PlaintextImpl>(this->self_reference.lock());
				plaintext->device_only = true;
				plaintexts.push_back(std::move(plaintext));
			}
			engine_->encodeDeviceSlotsMany(*this, plaintexts, enc);
			return plaintexts;
		}
	}

	// The host half, per plaintext, unchanged and unshared: the same EncodeCoefficientsForDevice the
	// single call runs, so the integers handed to the backend are the same integers, bit for bit.
	// Nothing here is amortised across the batch on purpose — this is the precision-critical half.
	std::vector<DeviceEncodedCoefficients> encs;
	encs.reserve(values.size());
	for (const std::vector<double>& value : values) {
		encs.push_back(EncodeCoefficientsForDevice(context, value, noiseScaleDeg, level, slots));
	}

	std::vector<Plaintext> plaintexts;
	plaintexts.reserve(values.size());
	for (size_t i = 0; i < values.size(); ++i) {
		Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
		// host stays EMPTY, for the same reason as the single call.
		plaintext->device_only = true;
		plaintexts.push_back(std::move(plaintext));
	}

	// All-or-nothing: the backend contract is that a throw here leaves every plaintext's device slot
	// empty, so a caller that falls back re-encodes a clean group rather than half a built one.
	engine_->encodeDeviceMany(*this, plaintexts, encs);

	return plaintexts;
}

// ---- Encryption ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->host);

	auto ct							= context->Encrypt(pkImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->host				= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (!this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt) {
	return Encrypt(pt, pk);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->host);

	auto ct							= context->Encrypt(skImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->host				= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (!this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt) {
	return Encrypt(pt, sk);
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt) {
	if (pt == nullptr) {
		OPENFHE_THROW("Plaintext pointer is null");
	}
	// The only backend-specific step is the readback (device->host); it is its own engine hook
	// (no-op on CPU). The decrypt itself is a host OpenFHE computation, so it lives here.
	engine_->recoverHostCiphertext(*this, ct);

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& ct_host = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext ptImpl;
	auto res = context->Decrypt(skImpl, ct_host, &ptImpl);

	if (pt->get() != nullptr) {
		(*pt)->host = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		// Drop any stale backend payload a reused output plaintext may still hold (empty `any` ==
		// not resident; RAII frees a device copy). Clearing the value type's own slot — no backend
		// dispatch, and no inspection of device-residency state in this neutral facade.
		(*pt)->device.reset();
	} else {
		*pt			= std::make_shared<PlaintextImpl>();
		(*pt)->host = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
	}

	DecryptResult result{};
	result.isValid		 = res.isValid;
	result.messageLength = res.messageLength;
	return result;
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt) {
	return Decrypt(ct, sk, pt);
}

void CryptoContextImpl<DCRTPoly>::RecoverHostCiphertext(Ciphertext<DCRTPoly>& ct) {
	engine_->recoverHostCiphertext(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::MarkOutput(Ciphertext<DCRTPoly>& ct) {
	engine_->markOutput(*this, ct);
}

// ---- Operations ----

// Most ops below require a degree-1 (2-poly) operand; EvalMultNoRelin is the one op that
// produces a degree-2 result, and Relinearize is the only way back down. Mirrors OpenFHE's own
// "ciphertext should be relinearized before" wording.
static void RequireDegree1(const Ciphertext<DCRTPoly>& ct, const char* op) {
	if (ct && ct->GetNumElements() != 2) {
		OPENFHE_THROW(std::string(op) + ": ciphertext should be relinearized before (has " + std::to_string(ct->GetNumElements()) + " elements; call Relinearize)");
	}
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalNegate(const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalNegate(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalNegateInPlace(Ciphertext<DCRTPoly>& ct) {
	engine_->evalNegateInPlace(*this, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	return engine_->evalAdd(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return engine_->evalAdd(*this, ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar) {
	return engine_->evalAdd(*this, ct, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	engine_->evalAddInPlace(*this, ct1, ct2);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	engine_->evalAddInPlace(*this, ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalAddInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalAdd(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalAddInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	return engine_->evalAddMany(*this, ciphertexts);
}

void CryptoContextImpl<DCRTPoly>::EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	engine_->evalAddManyInPlace(*this, ciphertexts);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	return engine_->evalSub(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return engine_->evalSub(*this, ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalSub(*this, pt, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar) {
	return engine_->evalSub(*this, ct, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalSub(*this, scalar, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	engine_->evalSubInPlace(*this, ct1, ct2);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalSubInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	engine_->evalSubInPlace(*this, scalar, ct1);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalSub(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalSub(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalSub(pt, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalSubInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	RequireDegree1(ct1, "EvalMult");
	RequireDegree1(ct2, "EvalMult");
	return engine_->evalMult(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	return engine_->evalMult(*this, ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar) {
	return engine_->evalMult(*this, ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	engine_->evalMultInPlace(*this, ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalMultInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalMultInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalMult(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalMult(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	RequireDegree1(ct1, "EvalMultMutableInPlace");
	RequireDegree1(ct2, "EvalMultMutableInPlace");
	engine_->evalMultInPlace(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultNoRelin(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	RequireDegree1(ct1, "EvalMultNoRelin");
	RequireDegree1(ct2, "EvalMultNoRelin");
	return engine_->evalMultNoRelin(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Relinearize(const Ciphertext<DCRTPoly>& ct) {
	return engine_->relinearize(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::RelinearizeInPlace(Ciphertext<DCRTPoly>& ct) {
	engine_->relinearizeInPlace(*this, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquare(const Ciphertext<DCRTPoly>& ct) {
	RequireDegree1(ct, "EvalSquare");
	return engine_->evalSquare(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSquareInPlace(Ciphertext<DCRTPoly>& ct) {
	RequireDegree1(ct, "EvalSquareInPlace");
	engine_->evalSquareInPlace(*this, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquareMutable(Ciphertext<DCRTPoly>& ct) {
	return EvalSquare(ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	RequireDegree1(ciphertext, "EvalRotate");
	return engine_->evalRotate(*this, ciphertext, index);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	RequireDegree1(ciphertext, "EvalRotateInPlace");
	engine_->evalRotateInPlace(*this, ciphertext, index);
}

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct) {
	RequireDegree1(ct, "EvalFastRotationPrecompute");
	return engine_->evalFastRotationPrecompute(*this, ct);
}

Ciphertext<DCRTPoly>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
	RequireDegree1(ct, "EvalFastRotation");
	return engine_->evalFastRotation(*this, ct, index, m, precomp);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	RequireDegree1(ct, "EvalFastRotationExt");
	return engine_->evalFastRotationExt(*this, ct, index, digits, addFirst);
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) {
	RequireDegree1(ct, "EvalFastRotation");
	return engine_->evalFastRotation(*this, ct, indices, m, precomp);
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {
	RequireDegree1(ct, "EvalFastRotationExt");
	return engine_->evalFastRotationExt(*this, ct, indices, digits, addFirst);
}

std::vector<Ciphertext<DCRTPoly>> CryptoContextImpl<DCRTPoly>::EvalRotateMany(const std::vector<Ciphertext<DCRTPoly>>& cts, const int32_t index) {
	// An empty batch is a no-op, not an error: a caller that filtered its bodies down to none
	// should not have to special-case the call.
	if (cts.empty())
		return {};
	// Validate the WHOLE batch before any of it runs — one call, one launch on a batched backend,
	// so a bad entry must not leave the batch half-computed. Same reasoning as LinearTransformMany.
	for (size_t i = 0; i < cts.size(); ++i) {
		if (!cts[i]) {
			OPENFHE_THROW("EvalRotateMany: input " + std::to_string(i) + " is null");
		}
		RequireDegree1(cts[i], "EvalRotateMany");
	}
	return engine_->evalRotateMany(*this, cts, index);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	RequireDegree1(ct, "EvalChebyshevSeries");
	return engine_->evalChebyshevSeries(*this, ct, coeffs, a, b);
}

void CryptoContextImpl<DCRTPoly>::EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	RequireDegree1(ct, "EvalChebyshevSeriesInPlace");
	engine_->evalChebyshevSeriesInPlace(*this, ct, coeffs, a, b);
}

std::vector<double> CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree) {
	return fideslib::get_chebyshev_coefficients(func, a, b, degree);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Rescale(const Ciphertext<DCRTPoly>& ciphertext) {
	return engine_->rescale(*this, ciphertext);
}

void CryptoContextImpl<DCRTPoly>::RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext) {
	engine_->rescaleInPlace(*this, ciphertext);
}

void CryptoContextImpl<DCRTPoly>::SetLevel(Ciphertext<DCRTPoly>& ct, size_t level) {
	ct->SetLevel(level);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	RequireDegree1(ciphertext, "EvalBootstrap");
	return engine_->evalBootstrap(*this, ciphertext, numIterations, precision, prescaled);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	RequireDegree1(ciphertext, "EvalBootstrapInPlace");
	engine_->evalBootstrapInPlace(*this, ciphertext, numIterations, precision, prescaled);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrapToLevel(const Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	RequireDegree1(ciphertext, "EvalBootstrapToLevel");
	return engine_->evalBootstrapToLevel(*this, ciphertext, outputLevel, numIterations, precision, prescaled);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapToLevelInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t outputLevel, uint32_t numIterations, uint32_t precision, bool prescaled) {
	RequireDegree1(ciphertext, "EvalBootstrapToLevelInPlace");
	engine_->evalBootstrapToLevelInPlace(*this, ciphertext, outputLevel, numIterations, precision, prescaled);
}


Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	RequireDegree1(ct, "AccumulateSum");
	return engine_->accumulateSum(*this, ct, slots, stride);
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	RequireDegree1(ct, "AccumulateSumInPlace");
	engine_->accumulateSumInPlace(*this, ct, slots, stride);
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	RequireDegree1(ct, "AccumulateSumInPlace");
	engine_->accumulateSumInPlace(*this, ct, slots, stride, start);
}

namespace {
// The shape and diagonal checks LinearTransformInPlace and LinearTransformMany both owe the caller.
// `what` names the entry point in the message, and `set` names which diagonal set failed (-1 for the
// single-transform call, which has only one).
void ValidateLinearTransformDiagonals(const std::string& what, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, bool ext, int set) {
	const std::string where = set < 0 ? std::string{} : (" of set " + std::to_string(set));
	if (rowSize <= 0 || bStep <= 0) {
		OPENFHE_THROW(what + ": rowSize and bStep must be positive (got rowSize=" + std::to_string(rowSize) + ", bStep=" + std::to_string(bStep) + ")");
	}
	if (diagonals.size() < static_cast<size_t>(rowSize)) {
		OPENFHE_THROW(what + ": need at least rowSize diagonals" + where + " (got " + std::to_string(diagonals.size()) + " for rowSize=" + std::to_string(rowSize) + ")");
	}
	// Reject a null diagonal here rather than letting it reach the CUDA kernel's assert. The
	// null-padding of the last giant step is built inside the transform, not passed in.
	for (int k = 0; k < rowSize; ++k) {
		if (!diagonals[static_cast<size_t>(k)]) {
			OPENFHE_THROW(what + ": diagonal " + std::to_string(k) + where + " is null; every entry in [0, rowSize) must be an encoded plaintext");
		}
		// ext is a requirement, not a hint: the fused kernel picks its path by inspecting the
		// diagonals and quietly takes the slow one when any is not extended. Catch the mistake at
		// the facade, on every backend, so a caller who forgot the extended encoder finds out on
		// CPU too and not only when the GPU turns out slow. (The device-side counterpart of this
		// check lives in CudaEngine::linearTransformInPlace / linearTransformMany.)
		if (ext && !diagonals[static_cast<size_t>(k)]->extended) {
			OPENFHE_THROW(what + "(ext=true): diagonal " + std::to_string(k) + where + " was not built by MakeCKKSPackedPlaintextExtended");
		}
	}
}
} // namespace

void CryptoContextImpl<DCRTPoly>::LinearTransformInPlace(Ciphertext<DCRTPoly>& ct, int rowSize, int bStep, const std::vector<Plaintext>& diagonals, int stride, int offset, bool ext) {
	RequireDegree1(ct, "LinearTransformInPlace");
	ValidateLinearTransformDiagonals("LinearTransformInPlace", rowSize, bStep, diagonals, ext, -1);
	engine_->linearTransformInPlace(*this, ct, rowSize, bStep, diagonals, stride, offset, ext);
}

std::vector<Ciphertext<DCRTPoly>> CryptoContextImpl<DCRTPoly>::LinearTransformMany(const Ciphertext<DCRTPoly>& ct,
  int rowSize,
  int bStep,
  const std::vector<std::vector<Plaintext>>& diagonalSets,
  int stride,
  int offset,
  bool ext) {
	RequireDegree1(ct, "LinearTransformMany");
	if (diagonalSets.empty()) {
		OPENFHE_THROW("LinearTransformMany: needs at least one diagonal set");
	}
	// Every set is validated BEFORE any of them runs: one call, one launch, so a bad set must not
	// leave the batch half-computed.
	for (size_t t = 0; t < diagonalSets.size(); ++t)
		ValidateLinearTransformDiagonals("LinearTransformMany", rowSize, bStep, diagonalSets[t], ext, static_cast<int>(t));
	return engine_->linearTransformMany(*this, ct, rowSize, bStep, diagonalSets, stride, offset, ext);
}

// ---- Prepared linear transforms ----

PreparedLinearTransform
CryptoContextImpl<DCRTPoly>::PrepareLinearTransform(const std::vector<Plaintext>& diagonals, int rowSize, int bStep, int stride, int offset, bool ext) {
	// The same validation the immediate calls do, at the same place in the call — so a caller who
	// switches to the prepared form gets the same diagnostics at prepare time rather than later.
	ValidateLinearTransformDiagonals("PrepareLinearTransform", rowSize, bStep, diagonals, ext, -1);
	const LinearTransformShape shape{ rowSize, bStep, stride, offset, ext };
	return engine_->prepareLinearTransform(*this, { diagonals }, shape);
}

PreparedLinearTransform CryptoContextImpl<DCRTPoly>::PrepareLinearTransformMany(const std::vector<std::vector<Plaintext>>& diagonalSets,
  int rowSize,
  int bStep,
  int stride,
  int offset,
  bool ext) {
	if (diagonalSets.empty())
		OPENFHE_THROW("PrepareLinearTransformMany: needs at least one diagonal set");
	for (size_t t = 0; t < diagonalSets.size(); ++t)
		ValidateLinearTransformDiagonals("PrepareLinearTransformMany", rowSize, bStep, diagonalSets[t], ext, static_cast<int>(t));
	const LinearTransformShape shape{ rowSize, bStep, stride, offset, ext };
	return engine_->prepareLinearTransform(*this, diagonalSets, shape);
}

void CryptoContextImpl<DCRTPoly>::LinearTransformInPlace(Ciphertext<DCRTPoly>& ct, const PreparedLinearTransform& prepared) {
	RequireDegree1(ct, "LinearTransformInPlace");
	if (!prepared)
		OPENFHE_THROW("LinearTransformInPlace: prepared transform is null");
	if (prepared->Sets() != 1) {
		OPENFHE_THROW("LinearTransformInPlace: this prepared transform carries " + std::to_string(prepared->Sets()) +
		  " diagonal sets; it is a batch handle and belongs to LinearTransformMany");
	}
	prepared->RequireFresh(*this, "LinearTransformInPlace");
	engine_->linearTransformInPlacePrepared(*this, ct, *prepared);
}

std::vector<Ciphertext<DCRTPoly>> CryptoContextImpl<DCRTPoly>::LinearTransformMany(const Ciphertext<DCRTPoly>& ct, const PreparedLinearTransform& prepared) {
	RequireDegree1(ct, "LinearTransformMany");
	if (!prepared)
		OPENFHE_THROW("LinearTransformMany: prepared transform is null");
	// Every set is validated BEFORE any of them runs, exactly as the vector overload validates
	// every set up front: one call, one launch, so a bad set must not leave the batch
	// half-computed.
	prepared->RequireFresh(*this, "LinearTransformMany");
	return engine_->linearTransformManyPrepared(*this, ct, *prepared);
}

uint64_t CryptoContextImpl<DCRTPoly>::PlaintextIdentity(const Plaintext& pt) {
	return engine_->plaintextIdentity(*this, pt);
}

void CryptoContextImpl<DCRTPoly>::ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	RequireDegree1(ct, "ConvolutionTransformInPlace");
	engine_->convolutionTransformInPlace(*this, ct, gStep, bStep, pts, indexes, stride, rowSize);
}

void CryptoContextImpl<DCRTPoly>::SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	RequireDegree1(ct, "SpecialConvolutionTransformInPlace");
	engine_->specialConvolutionTransformInPlace(*this, ct, gStep, bStep, pts, mask, indexes, stride, maskRotationStride, rowSize);
}

// ---- Ciphertext backend hooks ----

std::any CryptoContextImpl<DCRTPoly>::CloneCiphertextBackend(const CiphertextImpl<DCRTPoly>& src) {
	return engine_->cloneCiphertextBackend(*this, src);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextLevel(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextLevel(*this, ct);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextNoiseScaleDeg(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextNoiseScaleDeg(*this, ct);
}

double CryptoContextImpl<DCRTPoly>::CiphertextScalingFactor(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextScalingFactor(*this, ct);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextSlots(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextSlots(*this, ct);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextNumElements(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextNumElements(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::RefreshCiphertextHost(CiphertextImpl<DCRTPoly>& ct) {
	engine_->refreshHostShadow(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::SetCiphertextSlots(CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	engine_->setCiphertextSlots(*this, ct, slots);
}

void CryptoContextImpl<DCRTPoly>::SetCiphertextLevel(CiphertextImpl<DCRTPoly>& ct, size_t level) {
	engine_->setCiphertextLevel(*this, ct, level);
}

void CryptoContextImpl<DCRTPoly>::Synchronize() const {
	engine_->synchronize();
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	return fideslib::GetConvolutionTransformRotationIndices(rowSize, bStep, stride, gStep);
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetLinearTransformRotationIndices(int bStep, int stride, int offset) {
	return fideslib::GetLinearTransformRotationIndices(bStep, stride, offset);
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride, int offset) {
	return fideslib::GetLinearTransformPlaintextRotationIndices(rowSize, bStep, stride, offset);
}

} // namespace fideslib
