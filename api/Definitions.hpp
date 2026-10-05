#ifndef API_DEFINITIONS_HPP
#define API_DEFINITIONS_HPP

#include <cinttypes>
#include <memory>

namespace fideslib {

/// @brief Phantom type for the CKKS-RNS scheme.
typedef uint32_t CryptoContextCKKSRNS;

/// @brief Phantom type for the DCRTPoly representation.
typedef uint32_t DCRTPoly;

/// @brief Ciphertext representation for the CKKS-RNS scheme.
/// @tparam T Underlying representation type.
template <typename T> class CiphertextImpl;

/// @brief Shared pointer alias for CiphertextImpl.
/// @tparam T Underlying representation type.
template <typename T> using Ciphertext = std::shared_ptr<CiphertextImpl<T>>;

/// @brief Class for managing a cryptographic context.
/// @tparam T Underlying representation type.
template <typename T> class CryptoContextImpl;

/// @brief Shared pointer alias for CryptoContextImpl.
/// @tparam T Underlying representation type.
template <typename T> using CryptoContext = std::shared_ptr<CryptoContextImpl<T>>;

/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl;

/// @brief Shared pointer alias for PlaintextImpl.
using Plaintext = std::shared_ptr<PlaintextImpl>;

/// @brief Enumeration of supported PKE scheme features.
enum PKESchemeFeature {
	PKE			 = 0x01,
	KEYSWITCH	 = 0x02,
	PRE			 = 0x04,
	LEVELEDSHE	 = 0x08,
	ADVANCEDSHE	 = 0x10,
	MULTIPARTY	 = 0x20,
	FHE			 = 0x40,
	SCHEMESWITCH = 0x80,
};

/// @brief Result structure for decryption operations.
struct DecryptResult {
	bool isValid;
	uint32_t messageLength;
};

/// @brief Enumeration of supported scaling techniques.
enum ScalingTechnique {
	FIXEDMANUAL = 0,
	FIXEDAUTO,
	FLEXIBLEAUTO,
	FLEXIBLEAUTOEXT,
};

/// @brief Enumeration of supported key switching techniques.
enum KeySwitchTechnique {
	INVALID_KS_TECH = 0,
	// BV,
	HYBRID = 2,
};

/// @brief Enumeration of supported secret key distributions.
enum SecretKeyDist {
	GAUSSIAN			= 0,
	UNIFORM_TERNARY		= 1,
	SPARSE_TERNARY		= 2,
	SPARSE_ENCAPSULATED = 3,
};

/// @brief Optional overrides for the approximate modular reduction (EvalMod) stage of CKKS
/// bootstrapping. Both fields are opt-in: left at 0 the library behaves exactly as before, deriving
/// the stage from the coefficient table of the context's secret-key distribution.
///
/// `chebyshevDegree` truncates that table to the given degree before it is evaluated. The
/// truncation error is bounded by the sum of the dropped Chebyshev coefficients, and the depth the
/// stage consumes drops with the Paterson-Stockmeyer degree bin (OpenFHE `GetDepthByDegree`), so
/// this is the dial that trades approximation accuracy for bootstrap depth.
///
/// HOW TO COST A TRUNCATION. The dropped-coefficient bound is an error on the value the series
/// produces INSIDE EvalMod, and that is not the unit the bootstrap's output precision is quoted in.
/// Two conversions stand between them, and both are large:
///   * the `r` double-angle squarings amplify it (empirically ~8.2x for r = 3, since the value being
///     squared is O((2*pi)^-2^(j-r)), not O(1));
///   * a full-scale message is the value |m| * 2^-correctionFactor inside EvalMod -- the bootstrap
///     divides the message by 2^correction before the mod raise and multiplies the EvalMod output
///     by post * corFactor = 2^correctionFactor on the way out -- so an error added inside EvalMod
///     reaches the output message scaled by 2^correctionFactor (8 at ring 2^16 / 2^15 slots,
///     OpenFHE ckksrns-fhe.cpp correction-factor fit; up to 14 by its own clamp).
/// So a 2^-25 dropped-coefficient bound is ~2^-14 of output message error, not ~2^-25. Costing the
/// dial against the boot's output precision without both factors understates it by ~2^11.
///
/// WHICH DEGREES ARE EXERCISED. The evaluator takes its Paterson-Stockmeyer split from
/// `lbcrypto::Degree(coefficients)`, so a truncation can move the split as well as the depth. The
/// shipped tables select k = 5, m = 3 (encapsulated, degree 32), k = 7, m = 3 (sparse, degree 44)
/// and k = 6, m = 4 (uniform, degree 88). On the encapsulated table every cut that actually saves a
/// level (degree <= 27) selects k = 4, m = 3 -- a split no shipped table reaches. Treat a cut that
/// changes the split as an untested evaluator path and measure it against a cut that does not
/// (degrees 28..31 keep k = 5, m = 3) before attributing any loss to the approximation.
///
/// `levels` overrides the level budget reserved for the stage, which the facade forwards to
/// OpenFHE's `EvalBootstrapSetup(..., modevallevels)`; it moves the level the slots-to-coefficients
/// plaintexts are precomputed at, and hence the level a bootstrap returns at, without changing the
/// evaluated circuit. Left at 0 it is derived from the (possibly truncated) table as before.
///
/// The number of double-angle iterations is deliberately NOT exposed. The scalar folded in at each
/// iteration is a function of the iteration COUNT alone (`-(2*pi)^(-2^(j-r))`,
/// `applyDoubleAngleIterations` in src/CKKS/ApproxModEval.cu), but the COMPOSITION it forms is only
/// a sine approximation for the table it was fitted with, so a different count requires a different
/// table rather than a flag.
///
/// The Hamming weight of the sparse mod-raise secret is likewise NOT here, and cannot be: it is
/// fixed at 32 inside OpenFHE's own key generation (the `DCRTPoly(tug, ..., 32)` in
/// `FHECKKSRNS::EvalBootstrapKeyGen`), and the overflow bound K it pairs with is baked into the
/// coefficients-to-slots precomputation at setup. Moving it needs a hunk in
/// `deps/fideslib-ref-1.5.1.6.patch` at the key-generation site and at every `K_SPARSE_ENCAPSULATED`
/// / `g_coefficientsSparseEncapsulated` selection site, not a field here; `SPARSE_TERNARY` is the
/// only weight-192 bootstrap the library can run today, and it is not encapsulated.
struct ModReductionConfig {
	/// @brief Truncate the Chebyshev series to this degree. 0 = use the full table.
	uint32_t chebyshevDegree = 0;
	/// @brief Levels reserved for the stage at bootstrap setup. 0 = derive from the table.
	uint32_t levels = 0;
};

/// @brief Enumeration of supported security levels.
enum SecurityLevel {
	HEStd_128_classic,
	HEStd_192_classic,
	HEStd_256_classic,
	HEStd_128_quantum,
	HEStd_192_quantum,
	HEStd_256_quantum,
	HEStd_NotSet,
};

} // namespace fideslib

#endif
