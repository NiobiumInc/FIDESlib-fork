//
// SPARSE_ENCAPSULATED switching-key generation — BACKEND INDEPENDENT.
//
// This is host-only OpenFHE code (no CUDA, no device state). It used to live in
// src/CKKS/openfhe-interface/ParameterSwitch.cu, which is compiled ONLY when
// FIDESLIB_ENABLE_CUDA is ON, so a CPU-only build had no way to produce the
// encapsulation key material at all. That made SPARSE_ENCAPSULATED unusable on
// the split client/server topology every deployment actually uses: the client
// key-gens on a CPU backend and the server (CUDA) never holds the secret key,
// so NEITHER side could generate the a->b / b->a switching keys the GPU
// bootstrap consumes.
//
// Moving it to api/ (always compiled) fixes that. ParameterSwitch.cuh is now a
// shim over this header so the CUDA sources and tests keep their include paths
// and there is exactly ONE implementation — the key material is bit-identical
// to what the CUDA path generated before, because the code and the order of the
// RNG draws are unchanged.
//
// NOTE ON THE HEADER NAME: this is `.h`, not `.hpp`, deliberately. The api's
// CMake globs `${API_DIR}/*.hpp` into the INSTALLED public header set, and the
// public headers are (mostly) OpenFHE-free by design (BITCOMPAT O7). This
// header names lbcrypto types, so it stays internal.
//
#ifndef FIDESLIB_API_SPARSE_ENCAPSULATION_H
#define FIDESLIB_API_SPARSE_ENCAPSULATION_H

#include <openfhe.h>

#include <memory>
#include <utility>

namespace FIDESlib {
namespace CKKS {

// createSwitchableContextBasedOnContext / createContextSwitchingKeys removed — the in-context
// SPARSE_ENCAPSULATED design builds no second CryptoContext and no dual-context switching keys.
// AddSparseEncapsulationKeys now emits stock's two switching keys directly (see the .cpp).

/**
 * Generate the SPARSE_ENCAPSULATED switching keys for `secretKey` and publish
 * them in the process-global lbcrypto EvalAutomorphism key map, at the two
 * indices OpenFHE reserves for encapsulation:
 *
 *     2N - 4   dense main secret -> sparse mod-raise secret  (GHS (q0,p) sparse-switch key,
 *              KeySwitchGenSparse; consumed by keySwitchSparse BEFORE the raise)
 *     2N - 2   sparse mod-raise secret -> dense main secret  (ordinary hybrid key, KeySwitchGen;
 *              consumed by the generic keySwitch at the raised level, AFTER the raise)
 *
 * Those indices are EVEN, so they cannot collide with any rotation key (all
 * automorphism indices are 5^k mod 2N, hence odd) nor with the conjugation key
 * at 2N-1. Stock OpenFHE reserves the same two slots for the same purpose
 * (ckksrns-fhe.cpp, FHECKKSRNS::EvalBootstrapKeyGen).
 *
 * WHY THE AUTOMORPHISM MAP AND NOT SOMEWHERE PRIVATE: it is what makes the
 * split topology work. The map is what
 * CryptoContextImpl::SerializeEvalAutomorphismKey writes, so a client that
 * calls this ships the encapsulation keys to the server in the SAME stream that
 * already carries the rotation and conjugation keys — no new relay object, no
 * new format. On the far side FIDESlib::CKKS::AddBootstrapKeys reads exactly
 * these two indices back out (src/CKKS/openfhe-interface/RawCiphertext.cu).
 *
 * These are PUBLIC evaluation keys: each is a key-switching key from one secret
 * to another, i.e. an encryption of a secret under a secret, which is the same
 * class of material as a rotation or relinearization key and is published as a
 * matter of course. The secret itself (`secretKey`, and the freshly drawn
 * sparse secret, which is discarded here) never leaves this function.
 *
 * Merging is safe: InsertEvalAutomorphismKey only adds indices that are not
 * already present, so calling this after EvalBootstrapKeyGen leaves the
 * rotation and conjugation keys untouched.
 *
 * @param secretKey        the main secret key (any distribution).
 * @param hamming_weight   Hamming weight of the mod-raise secret (32, matching
 *                         both OpenFHE's SPARSE_ENCAPSULATED and the FIDESlib
 *                         GPU bootstrap's K = 16 Chebyshev table).
 */
void AddSparseEncapsulationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& secretKey, int hamming_weight = 32);

} // namespace CKKS
} // namespace FIDESlib

#endif // FIDESLIB_API_SPARSE_ENCAPSULATION_H
