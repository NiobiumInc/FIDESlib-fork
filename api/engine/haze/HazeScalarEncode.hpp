#ifndef API_HAZESCALARENCODE_HPP
#define API_HAZESCALARENCODE_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include <cstdint>
#include <vector>

#include <openfhe.h> // lbcrypto::ScalingTechnique (an enum nested in lbcrypto; not forward-declarable)

namespace fideslib::hazebk {

/// @brief Borrowed views of per-context double-arithmetic caches held by the engine,
/// all in OpenFHE (level) orientation.
///
/// Array orientations (L == |qBase| − 1):
///   sfReal[lvl]           == CryptoParametersCKKSRNS::GetScalingFactorReal(lvl),    lvl = 0..L
///   sfRealBig[lvl]        == CryptoParametersCKKSRNS::GetScalingFactorRealBig(lvl), lvl = 0..L−1
///   modReduceFactor[i]    == CryptoParametersCKKSRNS::GetModReduceFactor(i),        tower index i = 0..L
///   qBase[i]              == q_i (the i-th RNS prime), tower index i = 0..L
///
/// These are the OPPOSITE orientation from FIDESlib's param.ScalingFactorReal (which is
/// FLIPPED: param.ScalingFactorReal[k] == GetScalingFactorReal(L − k)).
/// modReduceFactor and qBase are NOT flipped in either representation.
struct ScalarEncodeParams {
	/// Q moduli in tower order (qBase[0] == q_0, ..., qBase[L] == q_L).
	const std::vector<uint64_t>& qBase;

	/// Scaling factors in OpenFHE level order: sfReal[lvl] == GetScalingFactorReal(lvl).
	const std::vector<double>& sfReal;

	/// Big scaling factors in OpenFHE level order: sfRealBig[lvl] == GetScalingFactorRealBig(lvl).
	const std::vector<double>& sfRealBig;

	/// Per-tower ModReduce factor: modReduceFactor[i] == GetModReduceFactor(i) == double(q_i)
	/// for FLEXIBLEAUTO/FLEXIBLEAUTOEXT; equals approxSF for FIXEDAUTO/FIXEDMANUAL.
	const std::vector<double>& modReduceFactor;

	/// Scaling technique of the context (used to select the FLEXIBLEAUTOEXT branch in
	/// elemForEvalAddOrSub).
	lbcrypto::ScalingTechnique scalingTech;
};

/// @brief Per-limb CRT encoding of `operand` (a double scalar) for a ct×double operation
/// at `towers` active RNS limbs.
///
/// Mirrors OpenFHE LeveledSHECKKSRNS::GetElementForEvalMult (NATIVEINT==64 path) and the
/// FIDESlib CUDA member function ContextData::ElemForEvalMult.
///
/// @param p          Engine caches in OpenFHE orientation.
/// @param towers     Number of active RNS limbs (== FIDESlib level + 1).
/// @param operand    The double scalar to encode.
/// @param towersIn   When non-zero, selects the bootstrap prescale fast-path: the input
///                   ciphertext has `towersIn` towers (a different level from `towers`),
///                   so the scaling factor is computed as
///                       scFactorOut * ModReduceFactor[towers−1] / scFactorIn
///                   This corresponds to Context.cu's `level_in != -1 && level_in != level`
///                   branch.  Pass 0 (default) for the normal case (level_in == -1).
///
/// @return Vector of length `towers`; result[i] is the encoding of operand reduced mod q_i.
std::vector<uint64_t> elemForEvalMult(
	const ScalarEncodeParams& p,
	size_t                    towers,
	double                    operand,
	size_t                    towersIn = 0);

/// @brief Per-limb CRT encoding of `operand` (a double scalar) for a ct±double operation
/// at `towers` active RNS limbs and the given noiseScaleDeg.
///
/// Mirrors OpenFHE LeveledSHECKKSRNS::GetElementForEvalAddOrSub (NATIVEINT==64 path) and
/// the FIDESlib CUDA member function ContextData::ElemForEvalAddOrSub.
///
/// Caller is responsible for handling negative operands: encode std::fabs(operand) and
/// flip each limb to q_i − result[i] before passing it to hazeSubScalarMrp / hazeAddScalarMrp.
///
/// @param p             Engine caches in OpenFHE orientation.
/// @param towers        Number of active RNS limbs.
/// @param operand       The double scalar to encode (caller passes |operand|, not operand).
/// @param noiseScaleDeg OpenFHE m_noiseScaleDeg of the target ciphertext.
///
/// @return Vector of length `towers`; result[i] is the encoding of operand reduced mod q_i.
std::vector<uint64_t> elemForEvalAddOrSub(
	const ScalarEncodeParams& p,
	size_t                    towers,
	double                    operand,
	size_t                    noiseScaleDeg);

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
#endif // API_HAZESCALARENCODE_HPP
