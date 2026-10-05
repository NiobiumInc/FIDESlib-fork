#include "CCParams.hpp"
#include "lattice/constants-lattice.h"

#include <openfhe.h>

#include <cassert>
#include <iostream>
#include <string>

namespace fideslib {

CCParams<CryptoContextCKKSRNS>::CCParams() {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
	this->host = std::make_any<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>>(std::move(params));
}

// ---- CKKS Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetMultiplicativeDepth(uint32_t depth) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetMultiplicativeDepth(depth);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetScalingModSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetBatchSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetBatchSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetRingDim(uint32_t dim) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetRingDim(dim);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingTechnique(ScalingTechnique tech) {
	auto& params	   = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto scale_openfhe = static_cast<lbcrypto::ScalingTechnique>(tech);
	assert((int)scale_openfhe == (int)tech);
	params.SetScalingTechnique(scale_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetNumLargeDigits(uint32_t numDigits) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetNumLargeDigits(numDigits);
}

void CCParams<CryptoContextCKKSRNS>::SetFirstModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetFirstModSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetDigitSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetDigitSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetKeySwitchTechnique(KeySwitchTechnique tech) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto ks_openfhe = static_cast<lbcrypto::KeySwitchTechnique>(tech);
	assert((int)ks_openfhe == (int)tech);
	params.SetKeySwitchTechnique(ks_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetSecretKeyDist(SecretKeyDist dist) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);

	// In-context encapsulation: SPARSE_ENCAPSULATED now maps onto the STOCK lbcrypto
	// SPARSE_ENCAPSULATED enum, not UNIFORM_TERNARY. That is what makes stock's own
	// EvalBootstrapKeyGen emit the 2N-2 / 2N-4 switching keys (in-context, the 2N-4 being the
	// two-prime (q0,p) sparse-switch key) and makes stock EvalBootstrap run the in-context
	// KeySwitchSparse dance instead of a plain uniform bootstrap. See ckksrns-fhe.cpp.
	//
	// ***SECURITY LANDMINE*** — read before touching this. stock KeyGenInternal
	// (src/pke/lib/schemebase/base-pke.cpp) puts SPARSE_ENCAPSULATED in the SAME case as
	// SPARSE_TERNARY and draws the MAIN secret at Hamming weight 192 — the ~123-bit sparse
	// instance this entire workstream exists to AVOID. Encapsulation's whole point is a UNIFORM
	// main key with the sparse h=32 key confined to the mod-raise. The main key is therefore kept
	// uniform in CryptoContextImpl<DCRTPoly>::KeyGen(), which temporarily forces UNIFORM_TERNARY
	// around the secret draw and then asserts (loudly, when a real security level is claimed) that
	// the resulting main secret is uniform, not h=192. Do NOT map SPARSE_ENCAPSULATED here without
	// that keygen-time guard in place — it would be a silent security regression, not a refactor.
	switch (dist) {
	case SecretKeyDist::SPARSE_TERNARY:
		params.SetSecretKeyDist(lbcrypto::SPARSE_TERNARY);
		break;
	case SecretKeyDist::SPARSE_ENCAPSULATED:
		params.SetSecretKeyDist(lbcrypto::SPARSE_ENCAPSULATED);
		break;
	default:
		params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
		break;
	}
	keyDist = dist;
}

void CCParams<CryptoContextCKKSRNS>::SetSecurityLevel(SecurityLevel level) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto sl_openfhe = static_cast<lbcrypto::SecurityLevel>(level);
	assert((int)sl_openfhe == (int)level);
	params.SetSecurityLevel(sl_openfhe);
}

// ---- Getters ----

SecretKeyDist CCParams<CryptoContextCKKSRNS>::GetSecretKeyDist() const {
	auto& params	 = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto skd_openfhe = params.GetSecretKeyDist();
	return static_cast<SecretKeyDist>(skd_openfhe);
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetMultiplicativeDepth() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	return params.GetMultiplicativeDepth();
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetBatchSize() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	return params.GetBatchSize();
}

// ---- Backend Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetBackend(Backend backend) {
	if (!IsBackendAvailable(backend)) {
		// Build a descriptive name for the unavailable backend.
		const char* name = "unknown";
		switch (backend) {
		case Backend::CPU:  name = "CPU";  break;
		case Backend::CUDA: name = "CUDA"; break;
		case Backend::HAZE: name = "haze (FHETCH)"; break;
		}
		OPENFHE_THROW(std::string(name) + " backend requested but not available (FIDESlib was not built with support for this backend)");
	}
	this->backend = backend;
}

void CCParams<CryptoContextCKKSRNS>::SetPlaintextAutoload(bool autoload) {
	this->plaintextAutoload = autoload;
}

void CCParams<CryptoContextCKKSRNS>::SetCiphertextAutoload(bool autoload) {
	this->ciphertextAutoload = autoload;
}

void CCParams<CryptoContextCKKSRNS>::SetReducedNoise(bool enable) {
	this->reducedNoise = enable;
}

} // namespace fideslib