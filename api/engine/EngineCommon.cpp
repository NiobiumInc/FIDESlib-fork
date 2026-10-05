#include "engine/EngineCommon.hpp"

#include <openfhe.h>

#include <vector>

namespace fideslib {

int32_t bootstrapModEvalLevels(SecretKeyDist keyDist, const ModReductionConfig& cfg) {
	std::vector<double> coeffchebyshev;
	int doubleAngleIts = 3;
	if (keyDist == SPARSE_ENCAPSULATED) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (keyDist == SPARSE_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (keyDist == UNIFORM_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
	} else {
		OPENFHE_THROW("Unsupported key distribution");
	}
	// Opt-in truncation: the evaluated series is the same table cut to `chebyshevDegree`, so the
	// reserved budget must be looked up for the cut degree, not the full one. GetRawParams performs
	// the matching cut on the device side; keeping both derivations here and there driven by the
	// same field is what keeps the reservation and the evaluated circuit in step.
	if (cfg.chebyshevDegree != 0 && cfg.chebyshevDegree + 1 < coeffchebyshev.size()) {
		coeffchebyshev.resize(cfg.chebyshevDegree + 1);
	}
	if (cfg.levels != 0) {
		return static_cast<int32_t>(cfg.levels);
	}
	return static_cast<int>(lbcrypto::GetMultiplicativeDepthByCoeffVector(coeffchebyshev, false)) + doubleAngleIts;
}

} // namespace fideslib
