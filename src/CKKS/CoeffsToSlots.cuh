//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_COEFFSTOSLOTS_CUH
#define GPUCKKS_COEFFSTOSLOTS_CUH
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {
/// @param levelsToDrop Which bootstrap precomputation to read the diagonals from: 0 is the
///                     full-height set, a positive value the set encoded that many towers
///                     lower (Context::GetBootPrecomputation(slots, levelsToDrop)). The
///                     diagonals must sit at the ciphertext's own level, so a bootstrap that
///                     raised to a shorter chain has to name its set here.
void EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode, int levelsToDrop = 0);

/// @copydoc EvalLinearTransform
void EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode, int levelsToDrop = 0);
} // namespace FIDESlib::CKKS
#endif // GPUCKKS_COEFFSTOSLOTS_CUH
