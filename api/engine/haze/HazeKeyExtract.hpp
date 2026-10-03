#ifndef API_HAZEKEYEXTRACT_HPP
#define API_HAZEKEYEXTRACT_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include <openfhe.h>

#include <cstdint>
#include <string>
#include <vector>

namespace fideslib::hazebk {

/// @brief Raw uint64 limb vectors for one OpenFHE HYBRID keyswitch key (relin or rotation).
///
/// Layout — HYBRID partition scheme:
///   numPartQ = a_limbs.size()  (== b_limbs.size())
///   alpha    = ceil(|Q| / numPartQ)
///   Partition i covers q_base[i*alpha .. min((i+1)*alpha, |Q|))
///
/// Each digit (part) carries a full Q||P tower set:
///   a_limbs[part][tower][coeff]   tower in [0, |Q|+|P|)
///   b_limbs[part][tower][coeff]   tower in [0, |Q|+|P|)
///
/// Towers are in OpenFHE EVALUATION (NTT) form.  q_base and p_base hold the
/// per-tower moduli in the same order OpenFHE returns them from GetParams().
///
/// This matches FIDESlib's RawKeySwitchKey convention and the layout documented
/// in haze test/openfhe_key_extract.hpp.
struct HybridKeyswitchLimbs {
	/// [digit/part][tower][coeff]; towers cover Q then P, in EVALUATION form.
	std::vector<std::vector<std::vector<uint64_t>>> a_limbs;
	std::vector<std::vector<std::vector<uint64_t>>> b_limbs;

	std::vector<uint64_t> q_base;
	std::vector<uint64_t> p_base;
};

/// @brief Extract the relin (EvalMult) key registered under keyTag.
///
/// Calls GetEvalMultKeyVector(keyTag) to retrieve the key; pk and sk of a keypair
/// share the same tag, so passing the public key's tag is sufficient.
///
/// @throws (std::runtime_error) with a descriptive message on any failure:
///   - no EvalMult key registered for the tag
///   - keyswitch technique is not HYBRID
///   - internal shape mismatch (tower count, ring dimension)
HybridKeyswitchLimbs extractEvalMultKeyLimbs(
	const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
	const std::string& keyTag);

/// @brief Extract the automorphism key for autoIndex registered under keyTag.
///
/// Calls GetEvalAutomorphismKeyMapPtr(keyTag) and looks up autoIndex in the map.
/// pk and sk of a keypair share the same tag.
///
/// @throws (std::runtime_error) with a descriptive message on any failure:
///   - no automorphism key map registered for the tag
///   - autoIndex not present in the map
///   - keyswitch technique is not HYBRID
///   - internal shape mismatch (tower count, ring dimension)
HybridKeyswitchLimbs extractAutomorphismKeyLimbs(
	const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
	const std::string& keyTag,
	uint32_t autoIndex);

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
#endif // API_HAZEKEYEXTRACT_HPP
