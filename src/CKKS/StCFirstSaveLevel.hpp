#ifndef FIDESLIB_CKKS_STCFIRSTSAVELEVEL_HPP
#define FIDESLIB_CKKS_STCFIRSTSAVELEVEL_HPP

// Host-side reading of the StC-first level-saving correction (OpenFHE PR #1374), shared by the CUDA and haze engines so both
// take the applied correction from OpenFHE's own selector instead of re-deriving it.

#include <openfhe.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace FIDESlib::CKKS {

// A device check on a scaling factor must agree with the oracle's.
constexpr double kStCFirstScalingFactorTolerance = lbcrypto::FHECKKSRNS::STC_FIRST_SCALING_FACTOR_TOLERANCE;

// Returns the exponent OpenFHE's EvalBootstrapStCFirst applies for `slots` right now, so the oracle's refusals (a degree above the factor, a
// factor raised after setup) are raised at the oracle's moment. A device encodes SlotsToCoeffs when its context loads, so a later
// re-encoding that changed the folded exponent from `loadedFolded` cannot take effect there and is refused.
inline uint32_t ReadStCFirstAppliedCorrection(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& host, uint32_t slots, uint32_t loadedFolded) {
	const auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(host->GetScheme()->m_FHE);
	if (!fhe)
		throw std::invalid_argument("StC-first save level: the context has no CKKS bootstrapping scheme");
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(host->GetCryptoParameters());
	if (!cryptoParams)
		throw std::invalid_argument("StC-first save level: the context is not CKKS");
	const auto correction = fhe->GetStCFirstCorrection(slots, cryptoParams);
	if (correction.folded != loadedFolded)
		throw std::invalid_argument("StC-first save level: the SlotsToCoeffs matrices for " + std::to_string(slots) +
		  " slots were re-encoded after the context reached the device (folded " + std::to_string(correction.folded) + " now, " + std::to_string(loadedFolded) +
		  " when loaded), so the context must be loaded again");
	return correction.applied;
}

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_STCFIRSTSAVELEVEL_HPP
