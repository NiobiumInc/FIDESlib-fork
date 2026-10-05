#ifndef API_INTEROP_HPP
#define API_INTEROP_HPP

// Bridge between OpenFHE objects and the fideslib facade.
//
// This is the ONLY facade header that includes OpenFHE. The rest deliberately do not --
// the canonical host values are held as std::any precisely so the public interface stays
// OpenFHE-free. A consumer that already owns lbcrypto objects (a server that deserialized
// them from a client) includes this one and pays for the dependency knowingly.

#include "Ciphertext.hpp"
#include "CryptoContext.hpp"
#include "PublicKey.hpp"
#include "engine/Backend.hpp"

// Bare <openfhe.h> is how the rest of this codebase includes OpenFHE; the build sets the
// include path up for it. A path-qualified spelling bypasses that setup and OpenFHE's own
// headers then fail to resolve each other (ReadOnlyCiphertext undeclared, etc.).
#include <openfhe.h>

namespace fideslib {

/// @brief Wrap an EXISTING lbcrypto CryptoContext so the engine can run on it.
///
/// GenCryptoContext builds a context from parameters, which suits a process that owns the
/// whole pipeline. A server does not: it receives a serialized context from a client and
/// deserializes it with OpenFHE. This is the way in for that case -- the same shape
/// GenCryptoContext produces, but around a context that already exists.
///
/// multiplicativeDepth is taken explicitly rather than recovered from the context, because
/// the depth a caller asked for is not reliably readable back off an lbcrypto context.
/// @param cc                  Deserialized lbcrypto context. Must not be null.
/// @param backend             Which engine to run on.
/// @param multiplicativeDepth Depth the context was generated with.
/// @param reducedNoise        haze-only: centered (ReducedNoise) FBC.
CryptoContext<DCRTPoly> ImportCryptoContext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                            Backend backend, uint32_t multiplicativeDepth,
                                            bool reducedNoise = false);

/// @brief Wrap an EXISTING lbcrypto Ciphertext (e.g. one a server deserialized).
///
/// The engine already treats ct->host as the canonical value and derives the device copy
/// from it lazily (CudaEngine::loadCiphertext does GetRawCipherText(ctx.host, ct->host)),
/// so populating host is the intended seam rather than a side door.
Ciphertext<DCRTPoly> ImportCiphertext(const CryptoContext<DCRTPoly>& cc,
                                      const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct);

/// @brief Wrap an EXISTING lbcrypto PublicKey.
///
/// CryptoContext::LoadContext needs one, but only as a handle: the engine reads its
/// GetKeyTag() to find the eval-mult key in OpenFHE's process-global key map (see
/// CudaEngine::loadContext). A server that deserialized the client's public key can
/// therefore hand it straight over.
PublicKey<DCRTPoly> ImportPublicKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk);

/// @brief Recover the lbcrypto ciphertext from a fideslib one.
/// @note Call CryptoContext::RecoverHostCiphertext(ct) first when the value was computed on
///       a device -- otherwise this returns the stale host shadow rather than the result.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ExportCiphertext(const Ciphertext<DCRTPoly>& ct);

} // namespace fideslib

#endif // API_INTEROP_HPP
