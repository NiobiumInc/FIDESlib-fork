//
// The lbcrypto views of the api objects and the exact-equality macros the api-vs-OpenFHE parity suites share. Each suite
// compares the active backend's ciphertext with the one OpenFHE computes from the same inputs on the context's host shadow.
//
#ifndef FIDESLIB_TEST_PARITYHELPERS_HPP
#define FIDESLIB_TEST_PARITYHELPERS_HPP

#include <any>

#include <gtest/gtest.h>
#include <openfhe.h>

#include "fideslib.hpp"

// Exact ciphertext equality: scaling metadata + every RNS limb of both polynomials. Mirrors the
// ASSERT_EQ_CIPHERTEXT in test/ParametrizedTest.cuh, minus the per-limb debug printing.
#define ASSERT_EQ_CIPHERTEXT(ct1, ct2)                                                                                                                   \
	do {                                                                                                                                                 \
		ASSERT_EQ((ct1)->GetNoiseScaleDeg(), (ct2)->GetNoiseScaleDeg());                                                                                 \
		ASSERT_EQ((ct1)->GetScalingFactor(), (ct2)->GetScalingFactor());                                                                                 \
		ASSERT_EQ((ct1)->GetEncodingType(), (ct2)->GetEncodingType());                                                                                   \
		ASSERT_EQ((ct1)->GetElements().size(), (ct2)->GetElements().size());                                                                             \
		for (size_t j = 0; j < (ct1)->GetElements().size(); ++j) {                                                                                       \
			ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().size(), (ct2)->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < (ct1)->GetElements().at(j).GetAllElements().size(); ++i) {                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                     \
				  (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues(), (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                            \
		}                                                                                                                                                \
	} while (0)

// ASSERT_EQ_CIPHERTEXT, plus pinning the expected component count up front (e.g. degree-2 vs.
// degree-3 ciphertexts after a multiplication that has not been relinearized).
#define ASSERT_EQ_CIPHERTEXT_N(ct1, ct2, n)                            \
	do {                                                               \
		ASSERT_EQ((ct1)->GetElements().size(), static_cast<size_t>(n)); \
		ASSERT_EQ_CIPHERTEXT(ct1, ct2);                                 \
	} while (0)

// Component-count and per-limb value equality only, no NoiseScaleDeg/ScalingFactor/EncodingType
// asserts. For ops where a documented metadata divergence exists (Rescale family) but limb
// bit-exactness must still be pinned.
#define ASSERT_EQ_CIPHERTEXT_LIMBS(ct1, ct2)                                                                                                             \
	do {                                                                                                                                                 \
		ASSERT_EQ((ct1)->GetElements().size(), (ct2)->GetElements().size());                                                                             \
		for (size_t j = 0; j < (ct1)->GetElements().size(); ++j) {                                                                                       \
			ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().size(), (ct2)->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < (ct1)->GetElements().at(j).GetAllElements().size(); ++i) {                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                     \
				  (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues(), (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                            \
		}                                                                                                                                                \
	} while (0)

// The OpenFHE context the api wraps (its CPU shadow). It holds the same keys the api generated, so
// it is the parity oracle.
inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& LbCc(fideslib::CryptoContext<fideslib::DCRTPoly>& cc) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->host);
}

// The lbcrypto ciphertext behind a freshly-encrypted api ciphertext (host-resident before any op).
// Only valid on encrypt outputs — a mid-chain api ciphertext's shadow is stale on the CUDA backend.
inline lbcrypto::Ciphertext<lbcrypto::DCRTPoly> LbCt(fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

// Recover the lbcrypto ciphertext of an api result, syncing it back from the device first (the device
// copy stays resident, so the ciphertext can still be used on-device afterward).
inline lbcrypto::Ciphertext<lbcrypto::DCRTPoly> HostCt(fideslib::CryptoContext<fideslib::DCRTPoly>& cc, fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {
	cc->RecoverHostCiphertext(ct);
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

#endif
