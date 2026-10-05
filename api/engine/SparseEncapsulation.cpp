//
// Created by carlosad on 14/09/25 as src/CKKS/openfhe-interface/ParameterSwitch.cu.
// Moved here (api/, always compiled) on 2026-09-02 so a CPU-only build can
// generate the SPARSE_ENCAPSULATED switching keys — see SparseEncapsulation.h.
// The two moved functions are byte-for-byte the same code as before; only the
// translation unit changed, so the key material and the order of the RNG draws
// are unchanged.
//

#include "engine/SparseEncapsulation.h"

#include <scheme/ckksrns/ckksrns-fhe.h> // lbcrypto::FHECKKSRNS::KeySwitchGenSparse

#include <map>

namespace FIDESlib {
namespace CKKS {

// AddSparseEncapsulationKeys now transcribes stock FHECKKSRNS::EvalBootstrapKeyGen's
// SPARSE_ENCAPSULATED block (ckksrns-fhe.cpp:340-350) rather than building a second CryptoContext.
// The old dual-context helpers (createSwitchableContextBasedOnContext / createContextSwitchingKeys)
// and their two switching-key shapes are gone: the in-context GPU bootstrap consumes stock's keys —
// the M-4 GHS single-digit (q0,p) sparse-switch key and the M-2 ordinary hybrid key — and this is
// exactly what OpenFheEngine::evalBootstrapKeyGen already gets by calling stock EvalBootstrapKeyGen
// directly, so both backends now emit identical key material.
//
// KEY-TAG NOTE (matches stock, but see the GPU-side override): stock does NOT set a tag on the
// discarded sparse secret skNew, so the M-4 key's own tag is skNew's default, not secretKey's. The
// CPU path looks the keys up by MAP INDEX (2N-4 / 2N-2), so the tag is irrelevant there; the GPU
// path re-keys them to the public key's tag in AddBootstrapKeys.
void AddSparseEncapsulationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& secretKey, int hamming_weight) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = secretKey->GetCryptoContext();
	auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(secretKey->GetCryptoParameters());
	auto algo         = cc->GetScheme();
	const uint32_t M  = cc->GetCyclotomicOrder();

	// Stock ckksrns-fhe.cpp:342-344 — fresh sparse mod-raise secret at Hamming weight 32.
	lbcrypto::DCRTPoly::TugType tug;
	auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cc);
	skNew->SetPrivateElement(lbcrypto::DCRTPoly(tug, cryptoParams->GetElementParams(), Format::EVALUATION, hamming_weight));

	auto evalKeys = std::make_shared<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>();
	// Stock ckksrns-fhe.cpp:348-349.
	(*evalKeys)[M - 4] = lbcrypto::FHECKKSRNS::KeySwitchGenSparse(secretKey, skNew); // dense -> sparse (GHS (q0,p))
	(*evalKeys)[M - 2] = algo->KeySwitchGen(skNew, secretKey);                       // sparse -> dense (hybrid)

	// Merges: only the two reserved indices that are not already present are added, so any
	// rotation/conjugation keys already published survive untouched.
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, secretKey->GetKeyTag());
}

} // namespace CKKS
} // namespace FIDESlib
