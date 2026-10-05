//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
class KeySwitchingKey;
// GPU transcription of FHECKKSRNS::KeySwitchSparse (the 2N-4 GHS sparse switch).
void keySwitchSparse(Ciphertext& ctxt, const KeySwitchingKey& ek);
void BootstrapCPUraise(Ciphertext& ctxt,
  const int slots,
  std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
  const bool prescaled);
/**
 * @brief Refresh a ciphertext in place (mod-raise, CoeffsToSlots, EvalMod, SlotsToCoeffs).
 *
 * @pre ctxt.getLevel() >= ctxt.NoiseLevel (levels count RNS limbs minus one): a NoiseLevel-1 input needs at least
 *      two limbs (level >= 1) and a NoiseLevel-2 input (pending rescale) at least three (level >= 2), because the
 *      modulus raise rescales once (twice for NoiseLevel 2) before raising. With `prescaled == true` the input must
 *      instead satisfy level == NoiseLevel - 1. Violations throw std::invalid_argument.
 * @param ctxt      Ciphertext to bootstrap; on return it is at the post-bootstrap level with NoiseLevel 1.
 * @param slots     Number of slots the bootstrap precomputation was generated for (>= ctxt.slots).
 * @param prescaled Whether the caller already applied the modulus-raise prescaling.
 * @param levelsToDrop  OPT-IN, 0 = the library's default behaviour. How many Q towers BELOW
 *   the chain top the mod-raise stops at, so the whole bootstrap circuit runs on that many
 *   fewer limbs and the result comes back that many levels deeper. It is the caller's choice
 *   of output level expressed as a shortening of the raise, and the value is sound for any
 *   0 <= levelsToDrop <= (chain towers - the bootstrap's own level consumption - 1): the
 *   raise is a tower-0 reinterpret, which is exact at any chain height, and the EvalMod
 *   interval K and Chebyshev degree do not depend on it.
 *
 *   A non-zero value REQUIRES the matching reduced-level precomputation to be installed
 *   (AddBootstrapPrecomputationReduced with the same slots and levelsToDrop): the
 *   CoeffsToSlots/SlotsToCoeffs diagonals have to be encoded at the levels the ciphertext
 *   meets them at, and the default set is encoded for the full-height raise.
 */
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false, const int levelsToDrop = 0);
double GetPreScaleFactor(Context& cc, int slots);
/// @param levelsToDrop Stop the raise that many Q towers below the chain top (0 = the top,
///                     the default). See Bootstrap.
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false, bool sparse_encaps = false, const int levelsToDrop = 0);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAP_CUH
