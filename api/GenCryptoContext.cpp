#include "GenCryptoContext.hpp"
#include "Interop.hpp"

#include "engine/Backend.hpp"
#include "engine/Engine.hpp"

#include <openfhe.h>

#include <shared_mutex>
#include <stdexcept>
#include <vector>

namespace fideslib {

CryptoContext<DCRTPoly> GenCryptoContext(CCParams<CryptoContextCKKSRNS>& params) {
	if (!params.reducedNoise.has_value()) {
		throw std::runtime_error("reducedNoise is unset: call CCParams::SetReducedNoise before GenCryptoContext");
	}

	auto& impl_params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(params.host);
	auto cc			  = lbcrypto::GenCryptoContext(impl_params);

	if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
		std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.clear();

	CryptoContextImpl<DCRTPoly> context;
	context.host				  = std::make_any<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc);
	context.engine_				  = MakeEngine(params.backend, *params.reducedNoise);
	context.auto_load_plaintexts  = params.plaintextAutoload;
	context.auto_load_ciphertexts = params.ciphertextAutoload;
	context.multiplicative_depth  = impl_params.GetMultiplicativeDepth();
	context.keyDist				  = params.keyDist;
	auto ptr					  = std::make_shared<CryptoContextImpl<DCRTPoly>>(std::move(context));
	ptr->self_reference			  = std::weak_ptr<CryptoContextImpl<DCRTPoly>>(ptr);

	return ptr;
}

CryptoContext<DCRTPoly> ImportCryptoContext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                            Backend backend, uint32_t multiplicativeDepth,
                                            bool reducedNoise) {
	if (!cc) {
		OPENFHE_THROW("ImportCryptoContext: null lbcrypto CryptoContext");
	}
	// Built IN PLACE, deliberately: ~CryptoContextImpl() calls ClearEvalMultKeys() /
	// ClearEvalAutomorphismKeys() unconditionally -- the engine_ null-check guards only the
	// teardown, not the clears. Constructing a stack object and moving it (the way
	// GenCryptoContext does) therefore destroys a moved-from husk whose destructor WIPES
	// OpenFHE's process-global key maps. GenCryptoContext survives that because callers
	// register keys after it; an imported context is the opposite order -- a server has
	// already called EvalMultKeyGen -- and the keys would be silently deleted, surfacing
	// later as "Call EvalMultKeyGen() to have EvalMultKey available for ID [...]" for a tag
	// that demonstrably IS registered.
	auto ptr					= std::make_shared<CryptoContextImpl<DCRTPoly>>();
	ptr->host					= std::make_any<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc);
	ptr->engine_				= MakeEngine(backend, reducedNoise);
	ptr->auto_load_plaintexts	= false;
	ptr->auto_load_ciphertexts	= true;
	ptr->multiplicative_depth	= multiplicativeDepth;
	// keyDist is left at its default: it only feeds bootstrap key generation, and an imported
	// context is for evaluating a program whose keys the client already generated.
	ptr->self_reference = std::weak_ptr<CryptoContextImpl<DCRTPoly>>(ptr);
	return ptr;
}

Ciphertext<DCRTPoly> ImportCiphertext(const CryptoContext<DCRTPoly>& cc,
                                      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct) {
	if (!cc) {
		OPENFHE_THROW("ImportCiphertext: null CryptoContext");
	}
	if (!ct) {
		OPENFHE_THROW("ImportCiphertext: null lbcrypto Ciphertext");
	}
	auto impl  = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc));
	impl->host = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
	return impl;
}

PublicKey<DCRTPoly> ImportPublicKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk) {
	if (!pk) {
		OPENFHE_THROW("ImportPublicKey: null lbcrypto PublicKey");
	}
	auto impl	= std::make_shared<PublicKeyImpl<DCRTPoly>>();
	impl->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(pk);
	return impl;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ExportCiphertext(const Ciphertext<DCRTPoly>& ct) {
	if (!ct) {
		OPENFHE_THROW("ExportCiphertext: null Ciphertext");
	}
	return std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
}

} // namespace fideslib