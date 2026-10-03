#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeKeyExtract.hpp"

// <openfhe.h> is included via HazeKeyExtract.hpp.  Bring in the remaining
// standard headers used in this translation unit.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Note: no haze headers are included here.  This file is pure host OpenFHE code;
// it accesses keyswitch keys via public OpenFHE accessors only.

namespace fideslib::hazebk {

namespace detail {

/// @brief Extract per-tower uint64 limbs from a DCRTPoly (already in EVALUATION form).
///
/// Iterates over each RNS tower with GetElementAtIndex, reads the NativePoly values,
/// and converts them to uint64_t via ConvertToInt.  Builds into a local vector and
/// moves into @p out only when all towers pass the dimension check.
///
/// @return true on success; false if tower count or per-tower length mismatches.
static bool extractDcrtpolyLimbs(
	const lbcrypto::DCRTPoly& poly,
	std::size_t expectedTowers,
	std::size_t ringDim,
	std::vector<std::vector<uint64_t>>& out)
{
	if (poly.GetNumOfElements() != expectedTowers)
		return false;

	std::vector<std::vector<uint64_t>> tmp(expectedTowers, std::vector<uint64_t>(ringDim));
	for (std::size_t t = 0; t < expectedTowers; ++t) {
		const auto& np   = poly.GetElementAtIndex(static_cast<uint32_t>(t));
		const auto& vals = np.GetValues();
		if (vals.GetLength() != ringDim)
			return false;
		for (std::size_t i = 0; i < ringDim; ++i) {
			tmp[t][i] = vals[i].template ConvertToInt<uint64_t>();
		}
	}
	out = std::move(tmp);
	return true;
}

/// @brief Walk an EvalKey (relin or rotation) into a HybridKeyswitchLimbs.
///
/// Validates HYBRID technique and shape, then extracts Q||P towers for each digit
/// of both A and B vectors.  @p out is overwritten only on full success.
///
/// Identical extraction math to haze test/openfhe_key_extract.hpp
/// detail::extract_keyswitch_key_into, ported to throw via OPENFHE_THROW instead
/// of returning hazeError_t.
///
/// @throws (OPENFHE_THROW) on invalid crypto parameters, non-HYBRID technique,
///         or tower/dimension mismatch.
static void extractKeyswitchKeyInto(
	const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
	const std::shared_ptr<lbcrypto::EvalKeyImpl<lbcrypto::DCRTPoly>>& evalKey,
	HybridKeyswitchLimbs& out)
{
	const auto cryptoParams =
		std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(cc->GetCryptoParameters());
	if (!cryptoParams)
		OPENFHE_THROW("haze backend: failed to cast CryptoParameters to CryptoParametersRNS");

	if (cryptoParams->GetKeySwitchTechnique() != lbcrypto::HYBRID)
		OPENFHE_THROW("haze backend: keyswitch technique is not HYBRID");

	if (!evalKey)
		OPENFHE_THROW("haze backend: EvalKey pointer is null");

	const auto elementParams = cryptoParams->GetElementParams();
	const auto paramsP       = cryptoParams->GetParamsP();
	if (!elementParams || !paramsP)
		OPENFHE_THROW("haze backend: element params or P params are null");

	const std::size_t ringDim = elementParams->GetRingDimension();
	if (ringDim == 0)
		OPENFHE_THROW("haze backend: ring dimension is zero");

	const std::uint32_t numPartQ = cryptoParams->GetNumPartQ();
	if (numPartQ == 0)
		OPENFHE_THROW("haze backend: numPartQ is zero");

	const auto& aVec = evalKey->GetAVector();
	const auto& bVec = evalKey->GetBVector();
	if (aVec.size() != numPartQ || bVec.size() != numPartQ)
		OPENFHE_THROW("haze backend: A/B vector size does not match numPartQ");

	// Build into a local; move into out only on full success.
	HybridKeyswitchLimbs tmp;

	const auto& qParams = elementParams->GetParams();
	const auto& pParams = paramsP->GetParams();
	tmp.q_base.reserve(qParams.size());
	for (const auto& p : qParams)
		tmp.q_base.push_back(p->GetModulus().template ConvertToInt<uint64_t>());
	tmp.p_base.reserve(pParams.size());
	for (const auto& p : pParams)
		tmp.p_base.push_back(p->GetModulus().template ConvertToInt<uint64_t>());

	// Each per-digit poly must span all Q+P towers (HYBRID packs Q||P into every digit).
	const std::size_t qpTowers = tmp.q_base.size() + tmp.p_base.size();

	tmp.a_limbs.resize(numPartQ);
	tmp.b_limbs.resize(numPartQ);
	for (std::uint32_t part = 0; part < numPartQ; ++part) {
		if (!extractDcrtpolyLimbs(aVec[part], qpTowers, ringDim, tmp.a_limbs[part]))
			OPENFHE_THROW("haze backend: A[" + std::to_string(part) +
				"] tower count or ring dimension mismatch (expected qpTowers=" +
				std::to_string(qpTowers) + ", ringDim=" + std::to_string(ringDim) + ")");
		if (!extractDcrtpolyLimbs(bVec[part], qpTowers, ringDim, tmp.b_limbs[part]))
			OPENFHE_THROW("haze backend: B[" + std::to_string(part) +
				"] tower count or ring dimension mismatch (expected qpTowers=" +
				std::to_string(qpTowers) + ", ringDim=" + std::to_string(ringDim) + ")");
	}

	out = std::move(tmp);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

HybridKeyswitchLimbs extractEvalMultKeyLimbs(
	const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
	const std::string& keyTag)
{
	if (!cc)
		OPENFHE_THROW("haze backend: CryptoContext is null in extractEvalMultKeyLimbs");

	// GetEvalMultKeyVector throws when no key is registered for the tag; we let that
	// exception propagate as-is after wrapping it with a descriptive message.
	const std::vector<std::shared_ptr<lbcrypto::EvalKeyImpl<lbcrypto::DCRTPoly>>>* evalKeys = nullptr;
	try {
		evalKeys = &lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalMultKeyVector(keyTag);
	} catch (...) {
		OPENFHE_THROW("haze backend: no EvalMult key registered for tag \"" + keyTag + "\"");
	}

	if (!evalKeys || evalKeys->empty())
		OPENFHE_THROW("haze backend: no EvalMult key registered for tag \"" + keyTag + "\"");

	HybridKeyswitchLimbs result;
	detail::extractKeyswitchKeyInto(cc, (*evalKeys)[0], result);
	return result;
}

HybridKeyswitchLimbs extractAutomorphismKeyLimbs(
	const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
	const std::string& keyTag,
	uint32_t autoIndex)
{
	if (!cc)
		OPENFHE_THROW("haze backend: CryptoContext is null in extractAutomorphismKeyLimbs");

	std::shared_ptr<std::map<uint32_t, std::shared_ptr<lbcrypto::EvalKeyImpl<lbcrypto::DCRTPoly>>>> keyMapPtr;
	try {
		keyMapPtr =
			lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMapPtr(keyTag);
	} catch (...) {
		OPENFHE_THROW("haze backend: no automorphism key map registered for tag \"" + keyTag + "\"");
	}

	if (!keyMapPtr)
		OPENFHE_THROW("haze backend: no automorphism key map registered for tag \"" + keyTag + "\"");

	const auto it = keyMapPtr->find(autoIndex);
	if (it == keyMapPtr->end())
		OPENFHE_THROW("haze backend: automorphism index " + std::to_string(autoIndex) +
			" not found in key map for tag \"" + keyTag + "\"");

	HybridKeyswitchLimbs result;
	detail::extractKeyswitchKeyInto(cc, it->second, result);
	return result;
}

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
