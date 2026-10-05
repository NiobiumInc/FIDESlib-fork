#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeScalarEncode.hpp"

#include <openfhe.h>  // lbcrypto::CKKSPackedEncoding, lbcrypto::LargeScalingFactorConstants,
                      // lbcrypto::DCRTPoly::Integer, lbcrypto::ScalingTechnique, uint32_t

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fideslib::hazebk {

// ---------------------------------------------------------------------------
// Index-orientation mapping summary
// ---------------------------------------------------------------------------
// Let Q = p.qBase.size() (total number of RNS primes), L = Q − 1.
// FIDESlib "level" is zero-based from the *bottom* of the modulus chain:
//   numTowers (FIDESlib)  = fideslib_level + 1  = towers   (parameter below)
//   OpenFHE level         = L − fideslib_level  = Q − towers
//
// Scaling-factor arrays:
//   param.ScalingFactorReal[k]     == GetScalingFactorReal(L − k)
//     → GetScalingFactorReal at towers t  == p.sfReal[Q − t]
//     → param.ScalingFactorReal[fideslib_level]  = sfReal[Q − towers]
//     → param.ScalingFactorReal[fideslib_level − 1]  (one fewer tower = one higher OpenFHE level)
//                                                = sfReal[Q − (towers − 1)]
//
//   param.ScalingFactorRealBig[k]  == GetScalingFactorRealBig(L − k)
//     → GetScalingFactorRealBig at towers t  == p.sfRealBig[Q − t]
//     → When towers == Q (full chain, FLEXIBLEAUTOEXT):
//         param.ScalingFactorRealBig[L] == sfRealBig[0]  ==  sfRealBig[Q − Q]  ✓
//
//   param.ModReduceFactor[i]  (NOT flipped)  ==  p.modReduceFactor[i]  (tower index)
//     → param.ModReduceFactor[fideslib_level]  ==  p.modReduceFactor[towers − 1]
//
//   prime[i].p  ==  p.qBase[i]  (tower index, same order both sides)
// ---------------------------------------------------------------------------

std::vector<uint64_t> elemForEvalMult(
	const ScalarEncodeParams& p,
	size_t                    towers,
	double                    operand,
	size_t                    towersIn)
{
	// Q = total number of primes; OpenFHE level for `towers` active limbs = Q − towers.
	const size_t Q = p.qBase.size();

	// Build the moduli vector for the active limbs.
	const uint32_t numTowers = static_cast<uint32_t>(towers);
	std::vector<lbcrypto::DCRTPoly::Integer> moduli(numTowers);
	for (uint32_t i = 0; i < numTowers; i++) {
		moduli[i] = p.qBase[i];
	}

	// Determine the effective scaling factor.
	//   Normal path (towersIn == 0, mirrors level_in == -1 or level_in == level):
	//     scFactor = GetScalingFactorReal(Q − towers)  ==  sfReal[Q − towers]
	//   Bootstrap prescale path (towersIn != 0 && towersIn != towers,
	//     mirrors level_in != -1 && level_in != level):
	//     scFactor = scFactorOut * rescalingFactor / scFactorIn
	double scFactor;
	if (towersIn == 0 || towersIn == towers) {
		// Normal case: level_in == -1 (or == level, same result).
		scFactor = p.sfReal[Q - towers];
	} else {
		// Bootstrap prescale: input ciphertext is at a different level (towersIn towers).
		// Mirrors Context.cu lines 361-369:
		//   assert(level > 0)
		//   scFactorIn      = param.ScalingFactorReal[level_in]      == sfReal[Q − towersIn]
		//   scFactorOut     = param.ScalingFactorReal[level - 1]      == sfReal[Q − (towers−1)]
		//   rescalingFactor = param.ModReduceFactor[level]            == modReduceFactor[towers−1]
		assert(towers > 0);
		double scFactorIn      = p.sfReal[Q - towersIn];
		double scFactorOut     = p.sfReal[Q - (towers - 1)];
		double rescalingFactor = p.modReduceFactor[towers - 1];
		scFactor               = scFactorOut * rescalingFactor / scFactorIn;

		// Translated from Context.cu:368-369:
		//   assert(|sfReal[Q−(towers−1)] * modReduceFactor[towers−1]
		//          − sfReal[Q−towers] * sfReal[Q−towers]| < 1e-9)
		//   assert(|scFactorIn * scFactor / rescalingFactor − scFactorOut| < 1e-9)
		assert(std::abs(p.sfReal[Q - (towers - 1)] * rescalingFactor
		                - p.sfReal[Q - towers] * p.sfReal[Q - towers]) < 1e-9);
		assert(std::abs(scFactorIn * scFactor / rescalingFactor - scFactorOut) < 1e-9);
	}

	// The body below is a verbatim port of Context.cu:372-428 / OpenFHE GetElementForEvalMult
	// (NATIVEINT==64 / HAVE_INT128 path) with no algorithmic changes.

	typedef __int128 DoubleInteger;
	const int32_t MAX_BITS_IN_WORD_LOCAL = 125;  // deliberately differs from MAX_BITS_IN_WORD

	// Compute logApprox: how many bits we must trim to stay within MAX_BITS_IN_WORD_LOCAL.
	int32_t logApprox   = 0;
	const double res    = std::fabs(operand * scFactor);
	if (res > 0) {
		int32_t logSF    = static_cast<int32_t>(std::ceil(std::log2(res)));
		int32_t logValid = (logSF <= MAX_BITS_IN_WORD_LOCAL) ? logSF : MAX_BITS_IN_WORD_LOCAL;
		logApprox        = logSF - logValid;
	}
	double approxFactor = std::pow(2, logApprox);

	DoubleInteger large     = static_cast<DoubleInteger>(operand / approxFactor * scFactor + 0.5);
	DoubleInteger large_abs = (large < 0 ? -large : large);
	DoubleInteger bound     = static_cast<uint64_t>(1) << 63;

	std::vector<lbcrypto::DCRTPoly::Integer> factors(numTowers);

	// CUDA uses `large_abs > bound` (Context.cu:390): at large_abs == 2^63 it takes the int64
	// else-branch (wrapping to INT64_MIN), unlike OpenFHE's `>=` which keeps the true __int128 value.
	if (large_abs > bound) {
		for (uint32_t i = 0; i < numTowers; i++) {
			DoubleInteger reduced = large % static_cast<__int128>(moduli[i].ConvertToInt());
			factors[i] = (reduced < 0)
			             ? static_cast<uint64_t>(reduced + static_cast<__int128>(moduli[i].ConvertToInt()))
			             : static_cast<uint64_t>(reduced);
		}
	} else {
		int64_t scConstant = static_cast<int64_t>(large);
		for (uint32_t i = 0; i < numTowers; i++) {
			int64_t reduced = scConstant % static_cast<int64_t>(moduli[i].ConvertToInt());
			factors[i]      = (reduced < 0)
			                  ? static_cast<uint64_t>(reduced + static_cast<int64_t>(moduli[i].ConvertToInt()))
			                  : static_cast<uint64_t>(reduced);
		}
	}

	// Scale back up by approxFactor via CRT multiplications (split into MAX_LOG_STEP chunks).
	if (logApprox > 0) {
		int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
		                  ? logApprox
		                  : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
		lbcrypto::DCRTPoly::Integer intStep = static_cast<uint64_t>(1) << logStep;
		std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(numTowers, intStep);
		logApprox -= logStep;

		while (logApprox > 0) {
			int32_t logStep2 = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
			                   ? logApprox
			                   : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
			lbcrypto::DCRTPoly::Integer intStep2 = static_cast<uint64_t>(1) << logStep2;
			std::vector<lbcrypto::DCRTPoly::Integer> crtSF(numTowers, intStep2);
			crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF, moduli);
			logApprox -= logStep2;
		}
		factors = lbcrypto::CKKSPackedEncoding::CRTMult(factors, crtApprox, moduli);
	}

	// Extract uint64_t results and final mod reduction (Context.cu:422-426).
	std::vector<uint64_t> result(numTowers);
	for (uint32_t i = 0; i < numTowers; ++i) {
		result[i] = factors[i].ConvertToInt() % p.qBase[i];
	}

	return result;
}

// ---------------------------------------------------------------------------

std::vector<uint64_t> elemForEvalAddOrSub(
	const ScalarEncodeParams& p,
	size_t                    towers,
	double                    operand,
	size_t                    noiseScaleDeg)
{
	const size_t Q = p.qBase.size();

	const uint32_t sizeQl = static_cast<uint32_t>(towers);
	std::vector<lbcrypto::DCRTPoly::Integer> moduli(sizeQl);
	for (uint32_t i = 0; i < sizeQl; i++) {
		moduli[i] = p.qBase[i];
	}

	// Select the scaling factor.
	//
	// OpenFHE condition (NATIVEINT==64 branch, line 275-280):
	//   FLEXIBLEAUTOEXT && ciphertext->GetLevel() == 0
	//     → GetScalingFactorRealBig(level==0) = sfRealBig[0]
	//
	// FIDESlib condition (Context.cu line 448):
	//   rescaleTechnique == FLEXIBLEAUTOEXT && level == L
	//   where L = Q − 1  →  towers == Q  (full chain)
	//   → param.ScalingFactorRealBig.at(L) = sfRealBig[Q − towers] = sfRealBig[0]  ✓
	//
	// Generic formula for sfRealBig: sfRealBig[Q − towers].
	// When towers == Q (the FLEXIBLEAUTOEXT case) this equals sfRealBig[0], which is
	// identical to GetScalingFactorRealBig(OpenFHE level 0).  The same formula works for
	// all other (towers < Q) cases where the non-Big sfReal path is taken.
	double scFactor = 0.0;
	if (p.scalingTech == lbcrypto::FLEXIBLEAUTOEXT && towers == Q) {
		// Full-chain FLEXIBLEAUTOEXT: use the "big" scaling factor.
		// sfRealBig[Q − towers] == sfRealBig[0] == GetScalingFactorRealBig(OpenFHE level 0).
		scFactor = p.sfRealBig[Q - towers];
	} else {
		// Normal case: sfReal[Q − towers] == GetScalingFactorReal(OpenFHE level Q − towers).
		scFactor = p.sfReal[Q - towers];
	}

	// The body below is a verbatim port of Context.cu:454-524 / OpenFHE GetElementForEvalAddOrSub
	// (NATIVEINT==64 path, non-COMPOSITESCALING branch) with no algorithmic changes.

	int32_t logApprox  = 0;
	const double res   = std::fabs(operand * scFactor);
	if (res > 0) {
		int32_t logSF    = static_cast<int32_t>(std::ceil(std::log2(res)));
		int32_t logValid = (logSF <= lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD)
		                   ? logSF
		                   : lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD;
		logApprox        = logSF - logValid;
	}
	double approxFactor = std::pow(2, logApprox);

	lbcrypto::DCRTPoly::Integer scConstant =
	    static_cast<uint64_t>(operand * scFactor / approxFactor + 0.5);
	std::vector<lbcrypto::DCRTPoly::Integer> crtConstant(sizeQl, scConstant);

	// Scale back up by approxFactor via CRT multiplications.
	if (logApprox > 0) {
		int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
		                  ? logApprox
		                  : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
		lbcrypto::DCRTPoly::Integer intStep = static_cast<uint64_t>(1) << logStep;
		std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(sizeQl, intStep);
		logApprox -= logStep;

		while (logApprox > 0) {
			int32_t logStep2 = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
			                   ? logApprox
			                   : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
			lbcrypto::DCRTPoly::Integer intStep2 = static_cast<uint64_t>(1) << logStep2;
			std::vector<lbcrypto::DCRTPoly::Integer> crtSF2(sizeQl, intStep2);
			crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF2, moduli);
			logApprox -= logStep2;
		}
		crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtApprox, moduli);
	}

	// In FLEXIBLEAUTOEXT mode at the full chain (towers == Q), skip the noiseScaleDeg
	// multiplications and return early (Context.cu:485-501 / OpenFHE line 340-342).
	if (p.scalingTech == lbcrypto::FLEXIBLEAUTOEXT && towers == Q) {
		// Use uint128_t for the final mod reduction to handle large NativeInteger values.
		using uint128_t = unsigned __int128;
		std::vector<uint128_t> tmp(sizeQl);
		for (uint32_t i = 0; i < sizeQl; ++i) {
			tmp[i] = crtConstant[i].ConvertToInt<uint128_t>();
		}
		for (uint32_t i = 0; i < sizeQl; ++i) {
			tmp[i] = tmp[i] % static_cast<uint128_t>(p.qBase[i]);
		}
		std::vector<uint64_t> result(sizeQl);
		for (uint32_t i = 0; i < sizeQl; ++i) {
			result[i] = static_cast<uint64_t>(tmp[i]);
		}
		return result;
	}

	// Normal path: multiply crtConstant by intScFactor a total of (noiseScaleDeg − 1) times.
	lbcrypto::DCRTPoly::Integer intScFactor = static_cast<uint64_t>(scFactor + 0.5);
	std::vector<lbcrypto::DCRTPoly::Integer> crtScFactor(sizeQl, intScFactor);

	for (uint32_t i = 1; i < static_cast<uint32_t>(noiseScaleDeg); i++) {
		crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtScFactor, moduli);
	}

	// Final mod reduction via uint128_t (Context.cu:510-524).
	using uint128_t = unsigned __int128;
	std::vector<uint128_t> result_128(sizeQl);
	for (uint32_t i = 0; i < sizeQl; ++i) {
		result_128[i] = crtConstant[i].ConvertToInt<uint128_t>();
	}
	for (uint32_t i = 0; i < sizeQl; ++i) {
		result_128[i] = result_128[i] % static_cast<uint128_t>(p.qBase[i]);
	}

	std::vector<uint64_t> result(sizeQl);
	for (uint32_t i = 0; i < sizeQl; ++i) {
		result[i] = static_cast<uint64_t>(result_128[i]);
	}

	return result;
}

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
