#ifndef API_ENGINE_COMMON_HPP
#define API_ENGINE_COMMON_HPP

#include <cstdint>

#include "config_core.h" // PARTIAL_SUM_RADIX

#include "Definitions.hpp"

namespace fideslib {

// Radix of the rotation-fold accumulation behind AccumulateSum. Taken from OpenFHE's compile-time
// PARTIAL_SUM_RADIX so every engine folds in the same order as the oracle: the association is
// bit-significant, so a hardcoded radix here would silently break CPU/GPU parity whenever OpenFHE
// is configured with a different CKKS_PARTIAL_SUM_RADIX.
constexpr int ACCUMULATE_SUM_RADIX = PARTIAL_SUM_RADIX;

// Number of mod-evaluation levels the bootstrap mod-reduction Chebyshev approximation consumes for
// the given secret-key distribution. Both engines pass this to OpenFHE's EvalBootstrapSetup so the
// CPU and CUDA paths configure mod-evaluation identically. Defined in EngineCommon.cpp.
//
// `cfg` is opt-in (see ModReductionConfig): a non-zero chebyshevDegree truncates the distribution's
// coefficient table before the Paterson-Stockmeyer depth is looked up, and a non-zero levels
// replaces the result outright. Default-constructed, this is the stock computation.
int32_t bootstrapModEvalLevels(SecretKeyDist keyDist, const ModReductionConfig& cfg = {});

} // namespace fideslib

#endif // API_ENGINE_COMMON_HPP
