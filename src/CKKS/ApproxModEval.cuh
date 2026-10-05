//
// Created by carlosad on 12/11/24.
//

#ifndef GPUCKKS_APPROXMODEVAL_CUH
#define GPUCKKS_APPROXMODEVAL_CUH

#include "CKKS/forwardDefs.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {
/** OpenFHE: for degree 5 or less uses naïve implementation, on FIDESlib, its always Patterson Stockmayer
 *  I suggest to only use range [-1, 1]
 * */
void evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound = -1.0, double upper_bound = 1.0);

/** @param outputScale  A real factor the result comes out multiplied by, folded into the
 *  Chebyshev coefficients and the double-angle constants so it costs no level: scaling the series
 *  by c = outputScale^(1/2^r) and iteration j's constant by c^(2^j) turns each 2y^2 + s into
 *  c^(2^j) (2y^2 + s), so after r iterations the output is outputScale x EvalMod, exactly in the
 *  reals. 1.0 (the default) runs the stock coefficients and constants untouched. Used by a
 *  shared-storage bootstrap set (BootstrapPrecomputation::stc_fold). */
void approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post, double outputScale = 1.0);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

/** @param outputScale  As for approxModReduction. */
void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post, double outputScale = 1.0);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_APPROXMODEVAL_CUH