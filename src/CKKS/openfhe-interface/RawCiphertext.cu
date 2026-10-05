//
// Created by carlosad on 24/04/24.
//
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/ParameterSwitch.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "Math.cuh"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
using namespace lbcrypto;

/**
 * Converts a vector of polynomial limbs to a single flattened array
 */
std::vector<std::vector<uint64_t>> FIDESlib::CKKS::GetRawArray(const std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>>& polys) {
	// total size is r * N
	int numRes		= polys.size();
	int numElements = (polys[0].GetValues() /*.m_values*/).GetLength();

	std::vector<std::vector<uint64_t>> flattened(numRes, std::vector<uint64_t>(numElements));

	// Fill the array. `flattened` is the only intermediate and it is the return value: GetValues()
	// hands back a const reference (hoisted out of the inner loop), and the NativeInteger ->
	// uint64_t conversion is done straight into the destination.
	for (int r = 0; r < numRes; ++r) {
		const auto& values = polys[r].GetValues() /*.m_values*/;
		auto& out		   = flattened[r];
		for (int i = 0; i < numElements; i++) {
			out[i] = values[i].ConvertToInt();
		}
	}
	return flattened;
};

/**
 * Gets the moduli from a vector of polynomial limbs and returns a single array
 */
static std::vector<uint64_t> GetModuli(const std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>>& polys) {
	int numRes = polys.size();
	std::vector<uint64_t> moduli(numRes);
	for (int r = 0; r < numRes; r++) {
		moduli[r] = polys[r].GetModulus().ConvertToInt();
	}
	return moduli;
};

/**
 * Converts a ciphertext from openFHE into the RawCiphertext format
 */
FIDESlib::CKKS::RawCipherText FIDESlib::CKKS::GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Ciphertext<DCRTPoly> ct, int REV) {
	RawCipherText result; //{ .cc = cc };
	// result.originalCipherText=ct;
	const size_t numComponents = ct->GetElements().size();
	if (numComponents != 2 && numComponents != 3) {
		throw std::runtime_error("unsupported ciphertext component count: " + std::to_string(numComponents));
	}
	result.numRes = ct->GetElements()[0].GetAllElements().size();
	result.N	  = ((ct->GetElements()[0].GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
	result.sub_0  = GetRawArray(ct->GetElements()[0].GetAllElements());
	result.sub_1  = GetRawArray(ct->GetElements()[1].GetAllElements());
	if (numComponents == 3) {
		result.sub_2 = GetRawArray(ct->GetElements()[2].GetAllElements());
	}

	// We read (hopefully) in eval form, and OpenFHE should be bit_reversed, so REVERSE == 0.
	// Changed the REV variable to an argument to the function.
	// purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
	if (!REV) {
		for (auto& i : result.sub_0)
			bit_reverse_vector(i);
		for (auto& i : result.sub_1)
			bit_reverse_vector(i);
		for (auto& i : result.sub_2)
			bit_reverse_vector(i);
	}
	result.moduli = GetModuli(ct->GetElements()[0].GetAllElements());
	result.format = ct->GetElements()[0].GetFormat();

	result.Noise	  = ct->GetScalingFactor(); // m_scalingFactor;
	result.NoiseLevel = ct->GetNoiseScaleDeg(); // m_noiseScaleDeg;
	result.keyid	  = ct->GetKeyTag();		// keyTag;
	result.slots	  = ct->GetSlots();

	return result;
};

/**
 * Converts a ciphertext from the RawCiphertext format back to the OpenFHE ciphertext format*/
void FIDESlib::CKKS::GetOpenFHECipherText(lbcrypto::Ciphertext<DCRTPoly> result, RawCipherText raw, int REV) {

	int size = result->GetElements()[0].GetAllElements().size();
	if (size < raw.numRes) {
		raw.numRes = size;
		raw.sub_0.resize(size);
		raw.sub_1.resize(size);
		if (!raw.sub_2.empty())
			raw.sub_2.resize(size);
	}
	assert(result->GetElements()[0].GetAllElements().size() >= raw.numRes);
	// Changed the REV variable to an argument to the function.
	// purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
	if (!REV) {
		for (auto& i : raw.sub_0)
			bit_reverse_vector(i);
		for (auto& i : raw.sub_1)
			bit_reverse_vector(i);
		for (auto& i : raw.sub_2)
			bit_reverse_vector(i);
	}
	const bool degree2 = !raw.sub_2.empty();
	DCRTPoly sub_0	   = result->GetElements().at(0);
	DCRTPoly sub_1	   = result->GetElements().at(1);
	// A degree-2 result needs a third container: reuse the existing one when the target ciphertext
	// already has three components, otherwise clone component 1 (same params, format and tower
	// structure) and overwrite its values below.
	DCRTPoly sub_2 = degree2 ? (result->GetElements().size() >= 3 ? result->GetElements().at(2) : result->GetElements().at(1)) : DCRTPoly();
	auto& dcrt_0   = sub_0.GetAllElements();
	auto& dcrt_1   = sub_1.GetAllElements();
	result->SetLevel(result->GetLevel() + result->GetElements().at(0).GetNumOfElements() - raw.numRes);
	dcrt_0.resize(raw.numRes);
	dcrt_1.resize(raw.numRes);
	for (int r = 0; r < raw.numRes; r++) {
		for (int i = 0; i < raw.N; i++) {
			(*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
			(*dcrt_1.at(r).m_values).at(i).SetValue(raw.sub_1.at(r).at(i));
		}
	}
	if (degree2) {
		auto& dcrt_2 = sub_2.GetAllElements();
		dcrt_2.resize(raw.numRes);
		for (int r = 0; r < raw.numRes; r++) {
			for (int i = 0; i < raw.N; i++) {
				(*dcrt_2.at(r).m_values).at(i).SetValue(raw.sub_2.at(r).at(i));
			}
		}
	}

	// sub_0.m_vectors=dcrt_0;
	// sub_1.m_vectors=dcrt_1;
	for (size_t i = sub_0.GetParams()->GetParams() /*m_params->m_params*/.size(); i > sub_0.GetAllElements() /*.m_vectors*/.size(); --i) {
		DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams() /*.m_params*/);
		newP->PopLastParam();
		sub_0.m_params.reset(newP);
	}

	for (size_t i = sub_1.GetParams()->GetParams().size(); i > sub_1.GetAllElements().size(); --i) {
		DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_1.GetParams());
		newP->PopLastParam();
		sub_1.m_params.reset(newP);
	}

	if (degree2) {
		for (size_t i = sub_2.GetParams()->GetParams().size(); i > sub_2.GetAllElements().size(); --i) {
			DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_2.GetParams());
			newP->PopLastParam();
			sub_2.m_params.reset(newP);
		}
	}

	std::vector<lbcrypto::DCRTPoly> ct_new = degree2 ? std::vector<lbcrypto::DCRTPoly>{ sub_0, sub_1, sub_2 } : std::vector<lbcrypto::DCRTPoly>{ sub_0, sub_1 };
	result->SetElements(ct_new);

	result->SetScalingFactor(raw.Noise);	  // Getm_scalingFactor*/ = raw.Noise;
	result->SetNoiseScaleDeg(raw.NoiseLevel); // /*m_noiseScaleDeg*/ = raw.NoiseLevel;
	result->SetKeyTag(raw.keyid);
	result->SetSlots(raw.slots);
}

void FIDESlib::CKKS::GetOpenFHEPlaintext(lbcrypto::Plaintext result, RawPlainText raw, int REV) {

	assert(result->GetElement<DCRTPoly>().GetAllElements().size() >= raw.numRes);
	// Changed the REV variable to an argument to the function.
	// purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
	if (!REV) {
		for (auto& i : raw.sub_0)
			bit_reverse_vector(i);
	}
	DCRTPoly sub_0 = result->GetElement<DCRTPoly>();
	auto& dcrt_0   = sub_0.GetAllElements();
	result->SetLevel(result->GetLevel() + result->GetElement<DCRTPoly>().GetNumOfElements() - raw.numRes);
	dcrt_0.resize(raw.numRes);
	for (int r = 0; r < raw.numRes; r++) {
		for (int i = 0; i < raw.N; i++) {
			(*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
		}
	}

	// sub_0.m_vectors=dcrt_0;
	// sub_1.m_vectors=dcrt_1;
	for (size_t i = sub_0.GetParams()->GetParams().size(); i > sub_0.GetAllElements().size(); --i) {
		DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams());
		newP->PopLastParam();
		sub_0.m_params.reset(newP);
	}

	result->GetElement<DCRTPoly>() /*encodedVectorDCRT*/ = sub_0;
	result->SetScalingFactor(raw.Noise);	  // scalingFactor = raw.Noise;
	result->SetNoiseScaleDeg(raw.NoiseLevel); // noiseScaleDeg = raw.NoiseLevel;
	result->SetSlots(raw.slots);
	/*
	std::cout << result << std::endl;
bool ok = result->Decode();
	std::cout << ok << " " << result << std::endl;
	*/
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Plaintext pt) {
	RawPlainText result; //{.cc = cc};
	result.originalPlainText = pt;
	result.numRes			 = pt->GetElement<DCRTPoly>().GetAllElements().size();
	result.N				 = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
	result.sub_0			 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
	result.moduli			 = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());

	result.format = pt->GetElement<DCRTPoly>().GetFormat();

	if constexpr (REVERSE) {
		for (auto& i : result.sub_0)
			bit_reverse_vector(i);
	}

	result.Noise	  = pt->GetScalingFactor();
	result.NoiseLevel = pt->GetNoiseScaleDeg();
	result.slots	  = pt->GetSlots();

	return result;
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, ReadOnlyPlaintext pt) {
	RawPlainText result; //{.cc = cc};
	// result.originalPlainText = pt;
	result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
	result.N	  = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
	result.sub_0  = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
	result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());

	result.format = pt->GetElement<DCRTPoly>().GetFormat();

	if constexpr (REVERSE) {
		for (auto& i : result.sub_0)
			bit_reverse_vector(i);
	}

	result.Noise	  = pt->GetScalingFactor();
	result.NoiseLevel = pt->GetNoiseScaleDeg();
	result.slots	  = pt->GetSlots();

	return result;
}

FIDESlib::CKKS::RawParams FIDESlib::CKKS::GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, FIDESlib::BOOT_CONFIG boot_conf, uint32_t chebyshevDegreeCap) {
	RawParams result;
	result.N	= cc->GetRingDimension();
	result.logN = std::bit_width((uint32_t)result.N) - 1;
	// cc->GetCryptoParameters()
	result.L = cc->GetCryptoParameters()->GetElementParams()->GetParams().size() - 1;
	// result.L = cc->params->m_params->m_params.size() - 1;
	const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
	result.scalingTechnique = cryptoParams->GetScalingTechnique();
	// result.qbit = cc->params->m_params->m_params->;
	// auto aux = cc->GetCryptoParameters()->GetParamsPK()->GetParamPartition();

	for (auto& i : /*cc->params->m_params->m_params*/
	  cc->GetCryptoParameters()->GetElementParams()->GetParams()) {
		result.moduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
		result.root_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
		result.cyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
	}

	// intnat::ChineseRemainderTransformFTTNat<intnat::NativeVector>::m_rootOfUnityReverseTableByModulus
	for (size_t i = 0; i < result.moduli.size(); ++i) {
		using namespace intnat;
		//  NumberTheoreticTransformNat<NativeVector>().PreCompute
		using FFT	   = ChineseRemainderTransformFTTNat<NativeVector>;
		auto mapSearch = FFT::m_rootOfUnityReverseTableByModulus.find(result.moduli[i]);
		if (mapSearch == FFT::m_rootOfUnityReverseTableByModulus.end() || mapSearch->second.GetLength() != (size_t)result.N /*CycloOrderHf*/) {
			FFT().PreCompute(result.root_of_unity[i], result.N << 1, result.moduli[i]);
		}

		if (mapSearch == FFT::m_rootOfUnityReverseTableByModulus.end() || mapSearch->second.GetLength() != (size_t)result.N /*CycloOrderHf*/) {
			assert("OpenFHE has not generated the NTT tables we want yet :(" == nullptr);
		}

		{
			int size = FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].GetLength();

			for (int k = 0; k < size; ++k) {
				result.psi[i].push_back(FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].at(k).ConvertToInt<uint64_t>() /*.m_value*/);
				result.psi_inv[i].push_back(FFT::m_rootOfUnityInverseReverseTableByModulus[result.moduli[i]].at(k).ConvertToInt<uint64_t>() /*m_value*/);
			}
			result.N_inv.push_back(FFT::m_cycloOrderInverseTableByModulus.at(result.moduli[i]).at(result.logN).ConvertToInt<uint64_t>() /*.m_value*/);
		}
	}

	//    intnat::NumberTheoreticTransformNat<intnat::NativeVector>().
	//    mubintvec<ubint<unsigned long>>;

	result.ModReduceFactor.resize(result.L + 1);
	for (size_t i = 0; i < result.ModReduceFactor.size(); ++i) {
		result.ModReduceFactor[/*result.L - */ i] = cryptoParams->GetModReduceFactor(i);
	}
	result.ScalingFactorReal.resize(result.L + 1);
	for (size_t i = 0; i < result.ScalingFactorReal.size(); ++i) {
		result.ScalingFactorReal[result.L - i] = cryptoParams->GetScalingFactorReal(i);
	}

	result.ScalingFactorRealBig.resize(result.L + 1);
	for (size_t i = 0; i < result.ScalingFactorRealBig.size(); ++i) {
		result.ScalingFactorRealBig[result.L - i] = cryptoParams->GetScalingFactorRealBig(i);
	}

	{
		auto& src  = cryptoParams->m_QlQlInvModqlDivqlModq;
		auto& dest = result.m_QlQlInvModqlDivqlModq;
		dest.resize(src.size());
		for (size_t i = 0; i < src.size(); ++i) {
			dest[i].resize(src[i].size());
			for (size_t j = 0; j < src[i].size(); ++j) {
				dest[i][j] = src[i][j].ConvertToInt<uint64_t>();
			}
		}

		if (MODRAISE_WITH_P0) {
			size_t k_ = cryptoParams->GetElementParams()->GetParams().size();
			std::vector<NativeInteger> moduliQ(k_ + 1);
			std::vector<NativeInteger> rootsQ(k_ + 1);
			for (size_t i = 0; i < k_; i++) {
				moduliQ[i] = cc->GetElementParams()->GetParams()[i]->GetModulus();
				rootsQ[i]  = cc->GetElementParams()->GetParams()[i]->GetRootOfUnity();
			}
			moduliQ[k_] = cryptoParams->GetParamsP()->GetParams().at(0)->GetModulus();
			rootsQ[k_]	= cryptoParams->GetParamsP()->GetParams().at(0)->GetRootOfUnity();

			dest.resize(dest.size() + 1);
			dest[k_ - 1].resize(k_);

			const int sizeQ = static_cast<int>(k_);
			for (int k = 1; k <= sizeQ; ++k) {

				lbcrypto::BigInteger modulusQ = cryptoParams->GetElementParams()->GetModulus();

				int curr_l = sizeQ;
				while (curr_l > (k - (sizeQ == k && lbcrypto::FLEXIBLEAUTOEXT == cryptoParams->GetScalingTechnique()))) {
					// divide modulus q by the small last limb
					modulusQ = modulusQ / lbcrypto::BigInteger(moduliQ.at(curr_l - 1));
					// k -= 1;
					curr_l--;
				}
				/*
				m_QlQlInvModqlDivqlModq.resize(sizeQ - 1);
				m_QlQlInvModqlDivqlModqPrecon.resize(sizeQ - 1);
				m_qlInvModq.resize(sizeQ - 1);
				m_qlInvModqPrecon.resize(sizeQ - 1);
				for (size_t k = 0; k < sizeQ - 1; k++) {
					size_t l = sizeQ - (k + 1);
					modulusQ = modulusQ / BigInteger(moduliQ[l]);
					m_QlQlInvModqlDivqlModq[k].resize(l);
					m_QlQlInvModqlDivqlModqPrecon[k].resize(l);
					m_qlInvModq[k].resize(l);
					m_qlInvModqPrecon[k].resize(l);
					BigInteger QlInvModql = modulusQ.ModInverse(moduliQ[l]);
					BigInteger result     = (QlInvModql * modulusQ) / BigInteger(moduliQ[l]);
					for (uint32_t i = 0; i < l; i++) {
						m_QlQlInvModqlDivqlModq[k][i]       = result.Mod(moduliQ[i]).ConvertToInt();
						m_QlQlInvModqlDivqlModqPrecon[k][i] = m_QlQlInvModqlDivqlModq[k][i].PrepModMulConst(moduliQ[i]);
						m_qlInvModq[k][i]                   = moduliQ[l].ModInverse(moduliQ[i]);
						m_qlInvModqPrecon[k][i]             = m_qlInvModq[k][i].PrepModMulConst(moduliQ[i]);
					}
				}
				*/
				lbcrypto::NativeInteger moduliP0 = moduliQ[k];
				lbcrypto::BigInteger QlInvModp0	 = modulusQ.ModInverse(moduliP0);
				lbcrypto::BigInteger result		 = (QlInvModp0 * modulusQ) / lbcrypto::BigInteger(moduliP0);
				for (int i = 0; i < k; i++) {
					if (k < sizeQ) {
						uint64_t res = result.Mod(moduliQ[i]).ConvertToInt<uint64_t>();
						assert(dest[cryptoParams->GetElementParams()->GetParams().size() - k - 1][i] == res);
					} else {
						dest[k - 1][i] = result.Mod(moduliQ[i]).ConvertToInt<uint64_t>();
					}
					// hG_.QlQlInvModqlDivqlModq[q.size()][i] = result.Mod(moduliQ[i]).ConvertToInt();
				}
			}
		}

		if (MODRAISE_WITH_P0) {
			dest.resize(dest.size() + 1);
			size_t k = cryptoParams->GetElementParams()->GetParams().size();
			dest[k - 1].resize(k);
			lbcrypto::BigInteger modulusQ = cryptoParams->GetElementParams()->GetModulus();
			std::vector<NativeInteger> moduliQ(k);
			std::vector<NativeInteger> rootsQ(k);

			for (size_t i = 0; i < k; i++) {
				moduliQ[i] = cc->GetElementParams()->GetParams()[i]->GetModulus();
				rootsQ[i]  = cc->GetElementParams()->GetParams()[i]->GetRootOfUnity();
			}
			if (lbcrypto::FLEXIBLEAUTOEXT == cryptoParams->GetScalingTechnique()) {
				// divide modulus q by the small last limb
				modulusQ = modulusQ / lbcrypto::BigInteger(moduliQ.back());
				// k -= 1;
			}
			/*
			m_QlQlInvModqlDivqlModq.resize(sizeQ - 1);
			m_QlQlInvModqlDivqlModqPrecon.resize(sizeQ - 1);
			m_qlInvModq.resize(sizeQ - 1);
			m_qlInvModqPrecon.resize(sizeQ - 1);
			for (size_t k = 0; k < sizeQ - 1; k++) {
				size_t l = sizeQ - (k + 1);
				modulusQ = modulusQ / BigInteger(moduliQ[l]);
				m_QlQlInvModqlDivqlModq[k].resize(l);
				m_QlQlInvModqlDivqlModqPrecon[k].resize(l);
				m_qlInvModq[k].resize(l);
				m_qlInvModqPrecon[k].resize(l);
				BigInteger QlInvModql = modulusQ.ModInverse(moduliQ[l]);
				BigInteger result     = (QlInvModql * modulusQ) / BigInteger(moduliQ[l]);
				for (uint32_t i = 0; i < l; i++) {
					m_QlQlInvModqlDivqlModq[k][i]       = result.Mod(moduliQ[i]).ConvertToInt();
					m_QlQlInvModqlDivqlModqPrecon[k][i] = m_QlQlInvModqlDivqlModq[k][i].PrepModMulConst(moduliQ[i]);
					m_qlInvModq[k][i]                   = moduliQ[l].ModInverse(moduliQ[i]);
					m_qlInvModqPrecon[k][i]             = m_qlInvModq[k][i].PrepModMulConst(moduliQ[i]);
				}
			}
			*/
			lbcrypto::NativeInteger moduliP0 = cryptoParams->GetParamsP()->GetParams().at(0)->GetModulus();
			lbcrypto::BigInteger QlInvModp0	 = modulusQ.ModInverse(moduliP0);
			lbcrypto::BigInteger result		 = (QlInvModp0 * modulusQ) / lbcrypto::BigInteger(moduliP0);
			for (uint32_t i = 0; i < k; i++) {
				assert(dest[k - 1][i] == result.Mod(moduliQ[i]).ConvertToInt<uint64_t>());
				//  dest[k - 1][i] = result.Mod(moduliQ[i]).ConvertToInt<uint64_t>();
				//  hG_.QlQlInvModqlDivqlModq[q.size()][i] = result.Mod(moduliQ[i]).ConvertToInt();
			}
		}
	}

	/// Key Switching precomputations !!!
	result.dnum = cryptoParams->GetNumPartQ();
	result.K	= cryptoParams->GetParamsP()->GetParams().size();
	assert(cryptoParams->GetNumPartQ() == cryptoParams->GetNumberOfQPartitions());
	cryptoParams->GetNumPerPartQ();

	{
		auto& src = cryptoParams->GetParamsP()->m_params;
		for (auto& i : src) {
			result.SPECIALmoduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
			result.SPECIALroot_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
			result.SPECIALcyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
		}
	}

	{
		auto& src = cryptoParams->m_paramsPartQ;
		for (auto& i : src) {
			result.PARTITIONmoduli.emplace_back();
			for (auto& j : i->GetParams()) {
				result.PARTITIONmoduli.back().push_back(j->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
			}
		}
	}

	{
		auto& src  = cryptoParams->GetPHatInvModp();
		auto& dest = result.PHatInvModp;
		dest.resize(src.size());
		for (size_t i = 0; i < src.size(); ++i) {
			dest[i] = src[i].ConvertToInt<uint64_t>();
		}
	}

	{
		auto& src  = cryptoParams->GetPInvModq();
		auto& dest = result.PInvModq;
		dest.resize(src.size());
		for (size_t i = 0; i < src.size(); ++i) {
			dest[i] = src[i].ConvertToInt<uint64_t>(); // m_value;
		}
	}

	{
		auto& src  = cryptoParams->GetPHatModq();
		auto& dest = result.PHatModq;
		dest.resize(src.size());
		for (size_t i = 0; i < src.size(); ++i) {
			dest[i].resize(src[i].size());
			for (size_t j = 0; j < src[i].size(); ++j) {
				dest[i][j] = src[i][j].ConvertToInt<uint64_t>(); // m_value;
			}
		}
	}

	{
		auto& dest = result.PartQlHatInvModq;
		auto& src  = cryptoParams->m_PartQlHatInvModq;
		dest.resize(src.size());
		for (size_t k = 0; k < dest.size(); ++k) {
			dest[k].resize(src[k].size());
			for (size_t i = 0; i < dest[k].size(); ++i) {
				dest[k][i].resize(src[k][i].size());
				for (size_t j = 0; j < src[k][i].size(); ++j) {
					dest[k][i][j] = src[k][i][j].ConvertToInt<uint64_t>(); // m_value;
				}
			}
		}
	}

	{
		auto& dest = result.PartQlHatModp;
		auto& src  = cryptoParams->m_PartQlHatModp;
		dest.resize(result.dnum);
		dest.resize(src.size());

		for (size_t k = 0; k < dest.size(); ++k) {
			dest[k].resize(src[k].size());
			for (size_t i = 0; i < dest[k].size(); ++i) {
				dest[k][i].resize(src[k][i].size());
				for (size_t j = 0; j < src[k][i].size(); ++j) {
					dest[k][i][j].resize(src[k][i][j].size());
					for (size_t l = 0; l < src[k][i][j].size(); ++l) {
						dest[k][i][j][l] = src[k][i][j][l].ConvertToInt<uint64_t>(); // m_value;
					}
				}
			}
		}
	}

	if (cc->GetScheme()->m_FHE) {
		if (boot_conf == FIDESlib::ENCAPS) {
			// lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated
			result.coefficientsCheby = { 0.24554573401685137,
				-0.047919064883347899,
				0.28388702040840819,
				-0.029944538735513584,
				0.35576522619036460,
				0.015106561885073030,
				0.29532946674499999,
				0.071203602333739374,
				-0.10347347339668074,
				0.044997590512555294,
				-0.42750712431925747,
				-0.090342129729094875,
				0.36762876269324946,
				0.049318066039335348,
				-0.14535986272411980,
				-0.015106938483063579,
				0.035951935499240355,
				0.0031036582188686437,
				-0.0062644606607068463,
				-0.00046609430477154916,
				0.00082128798852385086,
				0.000053910533892372678,
				-0.000084551549768927401,
				-4.9773801787288514e-6,
				7.0466620439083618e-6,
				3.7659807574103204e-7,
				-4.8648510153626034e-7,
				-2.3830267651437146e-8,
				2.8329709716159918e-8,
				1.2817720050334158e-9,
				-1.4122220430105397e-9,
				-5.9306213139085216e-11,
				6.3298928388417848e-11 }; // degree 32
			// in-context rework step 5: k = 1.0 for SPARSE_ENCAPSULATED (stock ckksrns-fhe.cpp:878 —
			// "do not divide by k as we already did it during precomputation"). The 1/K
			// is baked into the CtS precomputation, so constantEvalMult = pre*(1/(1.0*N))
			// matches stock's EvalMultInPlace(raised, pre*(1.0/(k*N))). Correct only now
			// that the host CCParams say SPARSE_ENCAPSULATED (in-context rework step 1): while they said
			// UNIFORM_TERNARY the precomputation did NOT bake 1/K in and the GPU applied it.
			result.bootK			 = 1.0;
			result.sparse_encaps	 = true;
			result.doubleAngleIts	 = lbcrypto::FHECKKSRNS::R_SPARSE;
		} else if (boot_conf == FIDESlib::SPARSE) {
			result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
			// k = K_SPARSE;
			result.bootK		  = cryptoParams->GetSecretKeyDist() == lbcrypto::SPARSE_TERNARY ?
					   1.0 :
					   std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_SPARSE; // do not divide by k as we already did it during precomputation
			result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
			auto dist			  = cryptoParams->GetSecretKeyDist();
			result.sparse_encaps  = dist == lbcrypto::SPARSE_TERNARY ? false : true;
			// } else if (cryptoParams->GetSecretKeyDist() == SPARSE_ENCAPSULATED) {    // Switch to this with OpenFHE v1.4, remove the flag
		} else if (boot_conf == FIDESlib::UNIFORM) {
			// result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
			result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
			result.bootK			 = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_UNIFORM; // lbcrypto::FHECKKSRNS::K_UNIFORM;
			result.doubleAngleIts	 = lbcrypto::FHECKKSRNS::R_UNIFORM;
			result.sparse_encaps	 = false;
		} else if (boot_conf == FIDESlib::UNIFORM_2) {
			result.coefficientsCheby = { 2.207266599864877165693144e-01,
				-2.682587999577537660883531e-03,
				2.381211781853223574678680e-01,
				-2.217484225267288364819018e-03,
				2.812572129228093631425622e-01,
				-1.118768081838177096479225e-03,
				3.175289620003503565648373e-01,
				7.418465825669621170612711e-04,
				2.838568337485863901648031e-01,
				2.959589426714506928128845e-03,
				1.111409007132102000348084e-01,
				4.045004272966680990142319e-03,
				-1.773752281024201515879923e-01,
				1.966283967074625767951224e-03,
				-3.431231212979015121611326e-01,
				-2.725087963359526001261290e-03,
				-7.807165506705357471695095e-02,
				-3.945018814199019625832410e-03,
				3.567953191665136358778909e-01,
				2.327088764749175604090725e-03,
				7.009711220121370156554974e-02,
				3.696241718778929454675142e-03,
				-4.332158158181031448741294e-01,
				-5.611596576812074611828596e-03,
				4.036825197782618612762917e-01,
				3.850187461178859998217616e-03,
				-2.204550282997065902002021e-01,
				-1.747584499380709470439665e-03,
				8.550195814336092325902428e-02,
				5.904770390752328004455030e-04,
				-2.553292477957162798229973e-02,
				-1.575954197449093332605158e-04,
				6.145509744931569942605343e-03,
				3.446140101881852877904744e-05,
				-1.228526976412591372248007e-03,
				-6.331581547383084787063417e-06,
				2.084131601810473930890683e-04,
				9.958114491488687767422640e-07,
				-3.049845562371355744382857e-05,
				-1.360239966314108502055247e-07,
				3.899969756237415622062044e-06,
				1.632622019731083661639916e-08,
				-4.404121172058021798864386e-07,
				-1.738463098910276156704066e-09,
				4.430993120182116282036096e-08,
				1.655643173042383465668248e-10,
				-4.003300390835467618282280e-09,
				-1.412477955902056943370032e-11,
				3.507103856919978252042301e-10 };
			result.bootK			 = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_UNIFORM; // lbcrypto::FHECKKSRNS::K_UNIFORM;
			result.doubleAngleIts	 = 7;
			result.sparse_encaps	 = false;
		}

		// Opt-in truncation of the mod-reduction series. evalChebyshevSeries derives its
		// Paterson-Stockmeyer split from lbcrypto::Degree(coefficients), so dropping the tail is
		// all that is needed to make the stage shallower; the error it adds is bounded by the sum
		// of the dropped coefficients, since |T_i| <= 1 on the [-1, 1] the bootstrap evaluates on.
		// Left at 0 (the default) nothing is dropped and the table is exactly the stock one.
		if (chebyshevDegreeCap != 0 && chebyshevDegreeCap + 1 < result.coefficientsCheby.size()) {
			result.coefficientsCheby.resize(chebyshevDegreeCap + 1);
		}

		;
	}

	result.p = cryptoParams->GetPlaintextModulus();

	return result;
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetKeySwitchKey(std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> ek) {

	std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
	std::vector<std::vector<std::vector<uint64_t>>> b;
	std::string keytag;

	// for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
	for (/*auto a_raw = ek.get()->m_rKey;*/ auto& i : { ek->GetAVector(), ek->GetBVector() }) {
		std::vector<std::vector<std::vector<uint64_t>>> a_inner;
		std::vector<std::vector<uint64_t>> a_inner_moduli;
		for (auto& j : i) {
			auto v = GetRawArray(j.GetAllElements() /*.m_vectors*/);
			a_inner_moduli.emplace_back();
			auto& a_aux = a_inner_moduli.back();
			for (auto& p : j.GetParams()->GetParams() /*m_params->m_params*/) {
				a_aux.push_back(p->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
			}
			a_inner.push_back(v);
		}
		a.push_back(a_inner);
		a_moduli.push_back(a_inner_moduli);
	}
	keytag = ek->GetKeyTag();

	return RawKeySwitchKey(std::move(a_moduli), std::move(a), std::move(b), std::move(keytag));
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {

	// const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

	auto& keyMap = keys.publicKey->GetCryptoContext()->GetAllEvalMultKeys();
	// lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
		const std::vector<EvalKey<DCRTPoly>>& key = keyMap[keys.secretKey->GetKeyTag()];
		const auto ek							  = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));
		return GetKeySwitchKey(ek);
	} else {
		assert("EvalKey is not present !!!" == nullptr);
	}

	return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
	auto& keyMap								   = cc->GetAllEvalMultKeys();
	// lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
		const std::vector<EvalKey<DCRTPoly>>& key = keyMap[publicKey->GetKeyTag()];
		const auto ek							  = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));

		return GetKeySwitchKey(ek);
	} else {
		assert("EvalKey is not present !!!" == nullptr);
	}

	return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(const KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
	auto& keyMap = cc->GetAllEvalAutomorphismKeys();

	if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
		auto& keyMap2 = keyMap[keys.secretKey->GetKeyTag()];
		uint32_t x	  = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);
		if (keyMap2->find(x) == keyMap2->end()) {
			cc->EvalAtIndexKeyGen(keys.secretKey, { index });
		}
		{
			const auto& key = keyMap2->at(x);
			assert(key != nullptr);
			const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

			return GetKeySwitchKey(ek);
		}
	} else {
		assert("RotKey is not present !!!" == nullptr);
		std::cout << "RotKey is not present !!!" << std::endl;
	}
	return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int index) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
	auto& keyMap								   = cc->GetAllEvalAutomorphismKeys();

	if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
		auto& keyMap2 = keyMap[publicKey->GetKeyTag()];
		uint32_t x	  = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);

		{
			const auto& key = keyMap2->at(x);
			assert(key != nullptr);
			const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

			return GetKeySwitchKey(ek);
		}
	} else {
		assert("RotKey is not present !!!" == nullptr);
		std::cout << "RotKey is not present !!!" << std::endl;
	}
	return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetConjugateKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
	auto& keyMap2								   = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

	if (keyMap2.find(2 * cc->GetRingDimension() - 1) != keyMap2.end()) {
		const auto& key = keyMap2.at(2 * cc->GetRingDimension() - 1);
		assert(key != nullptr);
		const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);
		// std::cout << std::endl << "Clave " << ek->GetKeyTag() << "\n";
		return GetKeySwitchKey(ek);
	} else {
		assert("RotKey is not present for rotation !!!" == nullptr);
		std::cout << "RotKey is not present for conjugation!!!" << std::endl;
	}
	return RawKeySwitchKey{};
}

#include "CKKS/BootstrapPrecomputation.cuh"

#include "CKKS/KeySwitchingKey.cuh"

#include "CKKS/LimbPartition.cuh"

#include "CKKS/RNSPoly.cuh"

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
FIDESlib::CKKS::GenRotationKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
	return GenRotationKeys(keys.secretKey, indexes);
}

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
FIDESlib::CKKS::GenRotationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
	std::set<int> indexes2(indexes.begin(), indexes.end());
	std::vector<int> indexes3;
	for (int i : indexes2) {
		if (i) {
			indexes3.emplace_back(i);
		}
	}
	auto evalKeys = cc->GetScheme()->EvalAtIndexKeyGen(keys, indexes3);
	CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag());
	return evalKeys;
	// std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
	//     ->cc->EvalAtIndexKeyGen(keys.secretKey, indexes3);
}

void FIDESlib::CKKS::AddRotationKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
	std::set<int> indexes2(indexes.begin(), indexes.end());
	std::vector<int> indexes3;
	for (int i : indexes2) {
		if (i && !GPUcc->HasRotationKey(i, publicKey->GetKeyTag())) {
			indexes3.emplace_back(i);
		}
	}
	for (int i : indexes3) {
		auto clave_rotacion = FIDESlib::CKKS::GetRotationKeySwitchKey(publicKey, i);
		// std::cout << "Load rotation key " << i << std::endl;
		FIDESlib::CKKS::KeySwitchingKey clave_rotacion_gpu(GPUcc);
		clave_rotacion_gpu.Initialize(clave_rotacion);
		GPUcc->AddRotationKey(i, std::move(clave_rotacion_gpu));
	}
}

void FIDESlib::CKKS::GenAndAddRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  const KeyPair<lbcrypto::DCRTPoly>& keys,
  FIDESlib::CKKS::Context& GPUcc,
  std::vector<int> indexes) {

	GenRotationKeys(keys, indexes);
	AddRotationKeys(keys.publicKey, GPUcc, indexes);
}

constexpr bool remove_extension		= false;
[[maybe_unused]] constexpr bool MAKE_CTS_LT_FRIENDLY = true;
[[maybe_unused]] constexpr bool MAKE_STC_LT_FRIENDLY = true;

std::vector<int> FIDESlib::CKKS::GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::BootstrapPrecomputation* result_) {
	// ContextData& GPUcc = *GPUcc_;
	std::vector<int> indexes;
	BootstrapPrecomputation result;
	auto precom = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

	uint32_t M = cc->GetCyclotomicOrder();
	uint32_t N = cc->GetRingDimension();

	int slots_red = std::min((int)M / 4, slots * 2);

	if (precom->m_paramsEnc.lvlb /* [CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1 && precom->m_paramsDec.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1) {

		result.LT.slots = slots;
		result.LT.bStep = (precom->m_paramsEnc.g /*m_dim1*/ == 0) ? ceil(sqrt(slots)) : precom->m_paramsEnc.g /*m_dim1*/;

		for (int i = 1; i < result.LT.bStep; ++i) {
			indexes.push_back(ReduceRotation(i, slots_red));
		}

#if AFFINE_LT
		indexes.push_back(ReduceRotation(result.LT.bStep, slots_red));
#else
		for (int i = result.LT.bStep; i < result.LT.slots; i += result.LT.bStep) {
			indexes.push_back(i);
		}
#endif
	} else {
		{ // CoeffToSlots metadata

			int32_t levelBudget		= precom->m_paramsEnc.lvlb;			   // [CKKS_BOOT_PARAMS::LEVEL_BUDGET];
			int32_t layersCollapse	= precom->m_paramsEnc.layersCollapse;  //[CKKS_BOOT_PARAMS::LAYERS_COLL];
			int32_t remCollapse		= precom->m_paramsEnc.remCollapse;	   // [CKKS_BOOT_PARAMS::LAYERS_REM];
			int32_t numRotations	= precom->m_paramsEnc.numRotations;	   // [CKKS_BOOT_PARAMS::NUM_ROTATIONS];
			int32_t b				= precom->m_paramsEnc.b;			   // [CKKS_BOOT_PARAMS::BABY_STEP];
			int32_t g				= precom->m_paramsEnc.g;			   //[CKKS_BOOT_PARAMS::GIANT_STEP];
			int32_t numRotationsRem = precom->m_paramsEnc.numRotationsRem; //[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
			int32_t bRem			= precom->m_paramsEnc.bRem;			   // [CKKS_BOOT_PARAMS::BABY_STEP_REM];
			int32_t gRem			= precom->m_paramsEnc.gRem;			   //[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

			int32_t stop	= -1;
			int32_t flagRem = 0;

			auto algo = cc->GetScheme();

			if (remCollapse != 0) {
				stop	= 0;
				flagRem = 1;
			}

			// precompute the inner and outer rotations
			{
				result.CtS.resize(levelBudget);
				for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
					if (flagRem == 1 && i == 0) {
						// remainder corresponds to index 0 in encoding and to last index in decoding
						result.CtS[i].bStep = gRem;
						result.CtS[i].gStep = bRem;
						result.CtS[i].slots = numRotationsRem;
						result.CtS[i].rotIn.resize(gRem);
						result.CtS[i].rotOut.resize(bRem);
					} else {
						result.CtS[i].bStep = g;
						result.CtS[i].gStep = b;
						result.CtS[i].slots = numRotations;
						result.CtS[i].rotIn.resize(g);
						result.CtS[i].rotOut.resize(b);
					}
				}

				for (int32_t s = levelBudget - 1; s > stop; s--) {
					for (int32_t j = 0; j < g; j++) {
						result.CtS[s].rotIn[j] = ReduceRotation((j) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots_red);
					}

					for (int32_t i = 0; i < b; i++) {
						result.CtS[s].rotOut[i] = ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots_red);
					}
				}

				if (flagRem) {
					for (int32_t j = 0; j < gRem; j++) {
						result.CtS[stop].rotIn[j] = ReduceRotation((j), slots_red);
					}

					for (int32_t i = 0; i < bRem; i++) {
						result.CtS[stop].rotOut[i] = ReduceRotation((gRem * i), slots_red);
					}
				}
			}

			// std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
		}

		{ // SlotToCoeff metadata
			uint32_t M = cc->GetCyclotomicOrder();
			uint32_t N = cc->GetRingDimension();

			int32_t levelBudget		= precom->m_paramsDec.lvlb;			   // [CKKS_BOOT_PARAMS::LEVEL_BUDGET];
			int32_t layersCollapse	= precom->m_paramsDec.layersCollapse;  //[CKKS_BOOT_PARAMS::LAYERS_COLL];
			int32_t remCollapse		= precom->m_paramsDec.remCollapse;	   // [CKKS_BOOT_PARAMS::LAYERS_REM];
			int32_t numRotations	= precom->m_paramsDec.numRotations;	   // [CKKS_BOOT_PARAMS::NUM_ROTATIONS];
			int32_t b				= precom->m_paramsDec.b;			   // [CKKS_BOOT_PARAMS::BABY_STEP];
			int32_t g				= precom->m_paramsDec.g;			   //[CKKS_BOOT_PARAMS::GIANT_STEP];
			int32_t numRotationsRem = precom->m_paramsDec.numRotationsRem; //[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
			int32_t bRem			= precom->m_paramsDec.bRem;			   // [CKKS_BOOT_PARAMS::BABY_STEP_REM];
			int32_t gRem			= precom->m_paramsDec.gRem;			   //[CKKS_BOOT_PARAMS::GIANT_STEP_REM];
			auto algo				= cc->GetScheme();

			int32_t flagRem = 0;

			if (remCollapse != 0) {
				flagRem = 1;
			}

			// precompute the inner and outer rotations
			{
				result.StC.resize(levelBudget);
				for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {

					if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
						// remainder corresponds to index 0 in encoding and to last index in decoding
						result.StC[i].bStep = gRem;
						result.StC[i].gStep = bRem;
						result.StC[i].slots = numRotationsRem;
						result.StC[i].rotIn.resize(gRem);
						result.StC.at(i).rotOut.resize(bRem);
					} else {
						result.StC[i].bStep = g;
						result.StC[i].gStep = b;
						result.StC[i].slots = numRotations;
						result.StC[i].rotIn.resize(g);
						result.StC.at(i).rotOut.resize(b);
					}
				}

				for (int32_t s = 0; s < levelBudget - flagRem; s++) {
					for (int32_t j = 0; j < g; j++) {
						result.StC.at(s).rotIn.at(j) = ReduceRotation((j) * (1 << (s * layersCollapse)), slots_red);
					}

					for (int32_t i = 0; i < b; i++) {
						result.StC.at(s).rotOut.at(i) = ReduceRotation((g * i) * (1 << (s * layersCollapse)), slots_red);
					}
				}

				if (flagRem) {
					int32_t s = levelBudget - flagRem;
					for (int32_t j = 0; j < gRem; j++) {
						result.StC.at(s).rotIn.at(j) = ReduceRotation((j) * (1 << (s * layersCollapse)), slots_red);
					}

					for (int32_t i = 0; i < bRem; i++) {
						result.StC.at(s).rotOut.at(i) = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), slots_red);
					}
				}
			}

			// std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
		}

		std::reverse(result.CtS.begin(), result.CtS.end());

		{
			int acc_offset = 0;
			for (size_t i = 0; i < result.CtS.size(); i++) {
				acc_offset += (result.CtS[i].slots / 2) * (result.CtS[i].bStep > 1 ? result.CtS[i].rotIn[1] : (result.CtS[i].gStep > 1 ? result.CtS[i].rotOut[1] : 0));
			}
			indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));

			auto& j = result.CtS.back().rotOut[0];
			j		= ReduceRotation(-acc_offset, slots_red);
		}

		{
			int acc_offset = 0;
			for (size_t i = 0; i < result.StC.size(); i++) {
				acc_offset += (result.StC[i].slots / 2) * (result.StC[i].bStep > 1 ? result.StC[i].rotIn[1] : (result.StC[i].gStep > 1 ? result.StC[i].rotOut[1] : 0));
			}
			indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));

			auto& j = result.StC.back().rotOut[0];
			j		= ReduceRotation(-acc_offset, slots_red);
		}

		for (auto& v : { &result.CtS, &result.StC }) {
			for (auto& i : *v) {
				for (auto& j : i.rotIn) {
					indexes.push_back(ReduceRotation(j, slots_red));
				}
#if AFFINE_LT
				// We do not include rotOut[0], it is later set to 0 but the last that is set to acc_offset
				for (auto& j : { /*i.rotOut[0],*/ i.rotOut.size() > 1 ? i.rotOut[1] /*- i.rotOut[0]*/ : 0 }) {
					indexes.push_back(ReduceRotation(j, slots_red));
				}
#else
				for (auto& j : i.rotOut) {
					if (j && !GPUcc.HasRotationKey(j)) {
						indexes.push_back(j);
					}
				}
#endif
			}
		}
	}

	/*
	int slots_transform = std::min((int)slots * 2, (int)cc->GetCyclotomicOrder() / 4);
	for (auto& i : indexes) {
		auto j_ = i % slots_transform;
		if (j_ < 0)
			j_ += slots_transform;
		if (j_ > slots_transform / 2)
			j_ += cc->GetCyclotomicOrder() / 4 - slots_transform;
		i = j_;
	}
	*/

	if (cc->GetRingDimension() / 2 != static_cast<uint32_t>(slots)) {
		int bStep                  = PARTIAL_SUM_RADIX;
		result.accumulate_bStep    = bStep;
		std::vector<int> rotations = GetAccumulateRotationIndices(bStep, slots, cc->GetRingDimension() / 2 / slots);
		for (auto idx : rotations) {
			indexes.push_back(idx);
		}
	}

	if (result_)
		*result_ = std::move(result);
	return indexes;
}

void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, bool SSE) {
	GenBootstrapKeys(keys.secretKey, slots, SSE);
}

void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, int slots, bool SSE) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
	std::vector<int> indexes					   = GetBootstrapIndexes(cc, slots, nullptr);
	cc->EvalMultKeyGen(keys);

	auto evalKeys = GenRotationKeys(keys, GetBootstrapIndexes(cc, slots, nullptr));
	auto conjKey  = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ConjugateKeyGen(keys);

	(*evalKeys)[cc->GetCyclotomicOrder() - 1] = conjKey;

	CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag()); // Reinsert all keys to add the particular conj key

	if (SSE) {
		// The encapsulation switching keys (2N-2 / 2N-4) are now generated by
		// the backend-independent helper in api/engine/SparseEncapsulation.cpp,
		// which publishes them into the same automorphism map — so a CPU-only
		// client reaches exactly this key material without a CUDA build. Same
		// code, same call order, therefore the same draws off the RNG as when
		// this block was inline here.
		CKKS::AddSparseEncapsulationKeys(keys, 32);
	}
}

void FIDESlib::CKKS::AddBootstrapKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_) {
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc	= publicKey->GetCryptoContext();
	FIDESlib::CKKS::BootstrapPrecomputation& result = GPUcc_->GetBootPrecomputation(slots);

	ContextData& GPUcc		 = *GPUcc_;
	std::vector<int> indexes = GetBootstrapIndexes(cc, slots, nullptr);

	// std::cout << "Add eval key" << std::endl;
	{
		KeySwitchingKey ksk(GPUcc_);
		RawKeySwitchKey rksk = GetEvalKeySwitchKey(publicKey);
		ksk.Initialize(rksk);
		GPUcc.AddEvalKey(std::move(ksk));
	}

	// std::cout << "Add conjugate key" << std::endl;
	{
		KeySwitchingKey ksk(GPUcc_);
		RawKeySwitchKey rksk = GetConjugateKeySwitchKey(publicKey);
		ksk.Initialize(rksk);
		GPUcc.AddRotationKey(GPUcc.N * 2 - 1, std::move(ksk));
	}
	// std::cout << "Add rotation keys" << std::endl;

	AddRotationKeys(publicKey, GPUcc_, indexes);

	if (GPUcc.param.raw->sparse_encaps) {
		// In-context ENCAPS: both switching keys stock's EvalBootstrapKeyGen emits load
		// into the MAIN context (no second "switchable" GPU context). Stored as rotation keys
		// at the two reserved even indices (2N-2 / 2N-4), reachable from Bootstrap.cu via
		// GetRotationKey — the same slots the automorphism map reserves them at.
		auto& evalKeys = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

		// The stored rotation-key lookup is by KeyHash: Bootstrap.cu does
		// GetRotationKey(2N-*, ctxt.keyID). Both keys MUST be keyed by the public key's tag.
		// The 2N-4 key's own tag is the DISCARDED sparse mod-raise secret skNew's tag (stock
		// KeySwitchGenSparse sets ek tag = newPrivateKey tag), which is NOT the ciphertext's
		// key tag — so override keyID to publicKey's tag before storing.
		const KeyHash bootKeyTag = publicKey->GetKeyTag();

		// 2N-2: ordinary hybrid key (sparse mod-raise secret -> dense main secret), at raised
		// params — stock algo->KeySwitchGen(skNew, privateKey). Standard hybrid Initialize.
		{
			std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
			  std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - 2]);
			assert(res != nullptr);
			FIDESlib::CKKS::KeySwitchingKey ksk_dense(GPUcc_);
			FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(res);
			ksk_dense.Initialize(rawKskEval);
			ksk_dense.keyID = bootKeyTag;
			GPUcc.AddRotationKey(2 * GPUcc.N - 2, std::move(ksk_dense));
		}

		// 2N-4: GHS single-digit two-prime (q0,p) sparse-switch key (dense main -> sparse
		// mod-raise) — stock FHECKKSRNS::KeySwitchGenSparse. Sparse Initialize (2-limb load).
		{
			std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
			  std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - 4]);
			assert(res != nullptr);
			FIDESlib::CKKS::KeySwitchingKey ksk_sparse(GPUcc_);
			FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(res);
			ksk_sparse.InitializeSparse(rawKskEval2);
			ksk_sparse.keyID = bootKeyTag;
			GPUcc.AddRotationKey(2 * GPUcc.N - 4, std::move(ksk_sparse));
		}
	}

	std::cout << "Rotation keys loaded: " << GPUcc.precom.keys.begin()->second.rot_keys.size() << " ~ "
			  << 2 * ((long long)GPUcc.precom.keys.begin()->second.rot_keys.size() * GPUcc.dnum * (GPUcc.L + GPUcc.K + 1) * GPUcc.N * 8 / (1 << 20)) << "MB"
			  << std::endl;
}

void FIDESlib::CKKS::AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::Context& GPUcc_, FIDESlib::CKKS::BootstrapPrecomputation& result) {
	ContextData& GPUcc = *GPUcc_;
	auto precom		   = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

	if (precom->m_paramsEnc.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1 && precom->m_paramsDec.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1) {

		if (!GPUcc.HasBootPrecomputation(slots)) {
			if constexpr (1) { // extended limbs computation
				auto auxA	 = precom->m_U0hatTPre;
				auto auxInvA = precom->m_U0Pre;

				result.LT.A.clear();
				for (uint32_t i = 0; i < auxA.size(); ++i) {
					RawPlainText raw = GetRawPlainText(cc, auxA.at(i));
					result.LT.A.emplace_back(GPUcc_, raw);
					if constexpr (remove_extension)
						result.LT.A.back().c0.freeSpecialLimbs();
				}

				result.LT.invA.clear();
				for (uint32_t i = 0; i < auxInvA.size(); ++i) {
					RawPlainText raw = GetRawPlainText(cc, auxInvA.at(i));
					result.LT.invA.emplace_back(GPUcc_, raw);
					if constexpr (remove_extension)
						result.LT.invA.back().c0.freeSpecialLimbs();
				}
			}
		}

	} else {

		if (!GPUcc.HasBootPrecomputation(slots)) {
			uint32_t M = cc->GetCyclotomicOrder();
			auto& A	   = precom->m_U0hatTPreFFT;
			auto& invA = precom->m_U0PreFFT;

			for (uint32_t i = 0; i < A.size(); ++i) {
				for (uint32_t j = 0; j < A.at(A.size() - 1 - i).size(); ++j) {
					RawPlainText raw = GetRawPlainText(cc, A.at(A.size() - 1 - i).at(j));
					result.CtS.at(i).A.emplace_back(GPUcc_, raw);
					if constexpr (remove_extension)
						result.CtS.at(i).A.back().c0.freeSpecialLimbs();
				}
			}

			for (uint32_t i = 0; i < invA.size(); ++i) {
				for (uint32_t j = 0; j < invA.at(i).size(); ++j) {
					RawPlainText raw = GetRawPlainText(cc, invA.at(i).at(j));
					result.StC.at(i).A.emplace_back(GPUcc_, raw);
					if constexpr (remove_extension)
						result.StC.at(i).A.back().c0.freeSpecialLimbs();
				}
			}
		}
	}
}


namespace {
/// Q-basis tower count of a precomputed transform diagonal. The bootstrap diagonals are
/// stored in the extended basis Q_l || P, so the P tail comes off before the count means a
/// level (ckksrns-fhe.cpp EvalCoeffsToSlotsPrecompute builds them that way).
uint32_t BootDiagonalQTowers(const lbcrypto::ReadOnlyPlaintext& pt, uint32_t sizeP) {
	const uint32_t total = static_cast<uint32_t>(pt->GetElement<lbcrypto::DCRTPoly>().GetAllElements().size());
	if (total <= sizeP)
		OPENFHE_THROW("FIDESlib: bootstrap diagonal carries no Q towers");
	return total - sizeP;
}

/// The `L` argument OpenFHE encoded a collapsed-FFT transform set with, recovered from the
/// set itself. EvalCoeffsToSlotsPrecompute keeps `L + compositeDegree*lvlb` towers for the
/// first step it evaluates and one composite level fewer per step after that, so the widest
/// diagonal in the set fixes L without re-deriving EvalBootstrapPrecompute's lEnc / lDec
/// arithmetic (which also depends on the mod-eval depth the setup was given).
uint32_t BootTransformL(const std::vector<std::vector<lbcrypto::ReadOnlyPlaintext>>& set, uint32_t lvlb, uint32_t compositeDegree, uint32_t sizeP) {
	uint32_t widest = 0;
	for (const auto& step : set)
		for (const auto& pt : step)
			widest = std::max(widest, BootDiagonalQTowers(pt, sizeP));
	if (widest < compositeDegree * lvlb)
		OPENFHE_THROW("FIDESlib: bootstrap transform set is shallower than its own level budget");
	return widest - compositeDegree * lvlb;
}
} // namespace


void FIDESlib::CKKS::AddBootstrapPrecomputationReduced(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int slots, int levelsToDrop, FIDESlib::CKKS::Context& GPUcc_) {
	if (levelsToDrop == 0)
		return; // the default set; AddBootstrapPrecomputation owns it
	if (levelsToDrop < 0)
		OPENFHE_THROW("FIDESlib: AddBootstrapPrecomputationReduced needs a non-negative levelsToDrop");

	ContextData& GPUcc = *GPUcc_;
	// GetRawPlainText and GetBootstrapIndexes take the context by non-const reference/value.
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> ccRef = cc;
	if (GPUcc.HasBootPrecomputation(slots, levelsToDrop))
		return;
	if (!GPUcc.HasBootPrecomputation(slots))
		OPENFHE_THROW("FIDESlib: install the full-height bootstrap precomputation before a reduced-level one");

	const auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	if (!fhe)
		OPENFHE_THROW("FIDESlib: context has no CKKS bootstrap scheme");
	const auto it = fhe->m_bootPrecomMap.find(slots);
	if (it == fhe->m_bootPrecomMap.end())
		OPENFHE_THROW("FIDESlib: no bootstrap precomputation for " + std::to_string(slots) + " slots (run EvalBootstrapSetup first)");
	const auto& precom = it->second;

	if (precom->m_paramsEnc.lvlb == 1 && precom->m_paramsDec.lvlb == 1)
		OPENFHE_THROW("FIDESlib: a reduced-level bootstrap precomputation is only implemented for a "
					  "collapsed-FFT level budget; this context was set up with {1,1}");
	if (precom->BTSlotsEncoding)
		OPENFHE_THROW("FIDESlib: a reduced-level bootstrap precomputation is only implemented for the "
					  "classic bootstrap order (raise, CoeffsToSlots, EvalMod, SlotsToCoeffs) -- the one "
					  "the device Bootstrap runs. This context was set up with slots encoding, whose "
					  "SlotsToCoeffs runs on the INPUT and cannot be moved with the raise.");

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
	const uint32_t sizeP	= static_cast<uint32_t>(cryptoParams->GetParamsP()->GetParams().size());
	const uint32_t cd		= cryptoParams->GetCompositeDegree();

	const uint32_t lEnc = BootTransformL(precom->m_U0hatTPreFFT, precom->m_paramsEnc.lvlb, cd, sizeP);
	const uint32_t lDec = BootTransformL(precom->m_U0PreFFT, precom->m_paramsDec.lvlb, cd, sizeP);
	if (static_cast<uint32_t>(levelsToDrop) >= lDec)
		OPENFHE_THROW("FIDESlib: levelsToDrop " + std::to_string(levelsToDrop) + " leaves the bootstrap no output levels (SlotsToCoeffs ends at " + std::to_string(lDec) + " towers)");

	// The transform arguments OpenFHE's own EvalBootstrapPrecompute passes, reproduced so the
	// reduced set differs from the default one in the encoding LEVEL and in nothing else
	// (ckksrns-fhe.cpp, FHECKKSRNS::EvalBootstrapPrecompute).
	const uint32_t m	 = 4 * static_cast<uint32_t>(slots);
	const uint32_t mmask = m - 1;

	std::vector<uint32_t> rotGroup(slots);
	{
		uint32_t fivePows = 1;
		for (int i = 0; i < slots; ++i) {
			rotGroup[i] = fivePows;
			fivePows *= 5;
			fivePows &= mmask;
		}
	}
	std::vector<std::complex<double>> ksiPows(m + 1);
	{
		const double ak = 2 * M_PI / m;
		for (uint32_t j = 0; j < m; ++j)
			ksiPows[j] = std::complex<double>(std::cos(ak * j), std::sin(ak * j));
		ksiPows[m] = ksiPows[0];
	}

	double k = 1.0;
	switch (cryptoParams->GetSecretKeyDist()) {
		case lbcrypto::UNIFORM_TERNARY:
			k = 1.0;
			break;
		case lbcrypto::SPARSE_TERNARY:
			k = lbcrypto::FHECKKSRNS::K_SPARSE;
			break;
		case lbcrypto::SPARSE_ENCAPSULATED:
			k = lbcrypto::FHECKKSRNS::K_SPARSE_ENCAPSULATED;
			break;
		default:
			OPENFHE_THROW("FIDESlib: unsupported SecretKeyDist for a reduced-level bootstrap precomputation");
	}
	double qDouble = 1.0;
	for (uint32_t j = 0; j < cd; ++j)
		qDouble *= cryptoParams->GetElementParams()->GetParams()[j]->GetModulus().ConvertToDouble();
	const double factor	  = std::ldexp(1.0, static_cast<int>(std::round(std::log2(qDouble))));
	const double pre	  = (cd > 1) ? 1.0 : qDouble / factor;
	const double scaleEnc = pre / k;
	const double scaleDec = (cd > 1) ? qDouble / cryptoParams->GetScalingFactorReal(0) : 1.0 / pre;
	const bool flagPack	  = !(cc->GetCKKSDataType() == lbcrypto::REAL || !precom->BTSlotsEncoding);

	auto A	  = fhe->EvalCoeffsToSlotsPrecompute(*cc, ksiPows, rotGroup, false, scaleEnc, lEnc - levelsToDrop, flagPack);
	auto invA = fhe->EvalSlotsToCoeffsPrecompute(*cc, ksiPows, rotGroup, false, scaleDec, lDec - levelsToDrop, flagPack);

	// Everything but the diagonals is level-independent, so the reduced set inherits it from
	// the full-height one: the BSGS geometry and rotation indices (same rotation keys, no new
	// key material), the correction factor, and the encapsulation flag.
	FIDESlib::CKKS::BootstrapPrecomputation result;
	GetBootstrapIndexes(ccRef, slots, &result);
	{
		BootstrapPrecomputation& full = GPUcc_->GetBootPrecomputation(slots);
		result.accumulate_bStep		  = full.accumulate_bStep;
		result.correctionFactor		  = full.correctionFactor;
		result.sparse_encaps		  = full.sparse_encaps;
	}
	if (result.CtS.size() != A.size() || result.StC.size() != invA.size())
		OPENFHE_THROW("FIDESlib: reduced-level bootstrap transform has an unexpected step count");

	for (uint32_t i = 0; i < A.size(); ++i)
		for (uint32_t j = 0; j < A.at(A.size() - 1 - i).size(); ++j) {
			RawPlainText raw = GetRawPlainText(ccRef, A.at(A.size() - 1 - i).at(j));
			result.CtS.at(i).A.emplace_back(GPUcc_, raw);
		}
	for (uint32_t i = 0; i < invA.size(); ++i)
		for (uint32_t j = 0; j < invA.at(i).size(); ++j) {
			RawPlainText raw = GetRawPlainText(ccRef, invA.at(i).at(j));
			result.StC.at(i).A.emplace_back(GPUcc_, raw);
		}

	GPUcc.AddBootPrecomputation(slots, levelsToDrop, std::move(result));
}

void FIDESlib::CKKS::AddBootstrapPrecomputationShared(int slots, int levelsToDrop, FIDESlib::CKKS::Context& GPUcc_) {
	if (levelsToDrop == 0)
		return; // the default set; AddBootstrapPrecomputation owns it
	if (levelsToDrop < 0)
		OPENFHE_THROW("FIDESlib: AddBootstrapPrecomputationShared needs a non-negative levelsToDrop");

	ContextData& GPUcc = *GPUcc_;
	if (GPUcc.HasBootPrecomputation(slots, levelsToDrop))
		return;
	if (!GPUcc.HasBootPrecomputation(slots))
		OPENFHE_THROW("FIDESlib: install the full-height bootstrap precomputation before a shared view of it");
	const BootstrapPrecomputation& full = GPUcc.GetBootPrecomputation(slots);
	if (full.CtS.empty() || full.StC.empty())
		OPENFHE_THROW("FIDESlib: a shared bootstrap precomputation is only implemented for a collapsed-FFT "
					  "level budget; this context was set up with {1,1}");

	// WHY A VIEW IS VALID, AND WHAT IT IS OFF BY.
	//
	// OpenFHE encodes step s of a set at the level the ciphertext meets it at, with integer
	// coefficients round(a x Delta(level)) (MakeAuxPlaintext, scFact = GetScalingFactorReal), in
	// the prefix q_0..q_level of the chain plus P. A set `levelsToDrop` = d lower encodes the same
	// transform values a at level - d over the shorter prefix. The primes are the same prefix, so
	// limbs 0..level-d of the full-height plaintext are a valid plaintext at level - d -- but of
	// the integer polynomial round(a x Delta(level)), not round(a x Delta(level - d)). Under
	// FLEXIBLEAUTO Delta depends on the chain position, so the two differ by
	//     rho_s = Delta(level_s) / Delta(level_s - d)       (FIDESlib indices: ScalingFactorReal[])
	// which is 1 +- ~2^-10; under the FIXED techniques Delta is one number and rho_s == 1.
	//
	// DIRECTION. A plaintext with integers m and NoiseFactor F decodes to m / F. The view keeps
	// the source's integers, m = a x Delta(level_s), and must carry the CANONICAL NoiseFactor of
	// the level it is read at, F = Delta(level_s - d) (FIDESlib asserts canonical scales on every
	// rescale and encodes multScalar constants against ScalingFactorReal[level], so a ciphertext
	// may never carry anything else). It therefore decodes to a x rho_s: one step multiplies the
	// ciphertext's value by rho_s more than the canonical set would, and a transform of n steps
	// by R = prod_s rho_s. Both transforms are linear, so the bootstrap cancels R by multiplying
	// the transform's INPUT by 1/R:
	//     cts_fold = 1 / prod_{CtS steps} rho_s   -> constantEvalMult, before CoeffsToSlots
	//     stc_fold = 1 / prod_{StC steps} rho_s   -> EvalMod's output, i.e. SlotsToCoeffs' input
	// (BootstrapPrecomputation::cts_fold / stc_fold; Bootstrap and approxModReduction apply them).
	//
	// Every source plaintext is checked to be exactly that canonical encode -- level shared by its
	// step, NoiseFactor == ScalingFactorReal[level], noise degree 1 -- because the ratio above is
	// only right for one; anything else throws here rather than producing a quietly mis-scaled set.
	BootstrapPrecomputation result;
	result.LT.slots			= full.LT.slots;
	result.LT.bStep			= full.LT.bStep;
	result.accumulate_bStep = full.accumulate_bStep;
	result.correctionFactor = full.correctionFactor;
	result.sparse_encaps	= full.sparse_encaps;
	result.shared_view		= true;

	auto viewSteps = [&](const std::vector<BootstrapPrecomputation::LTstep>& from, std::vector<BootstrapPrecomputation::LTstep>& into) {
		double R = 1.0;
		into.resize(from.size());
		for (size_t i = 0; i < from.size(); ++i) {
			const BootstrapPrecomputation::LTstep& f = from[i];
			BootstrapPrecomputation::LTstep& t		 = into[i];
			t.slots									 = f.slots;
			t.bStep									 = f.bStep;
			t.gStep									 = f.gStep;
			t.rotIn									 = f.rotIn;
			t.rotOut								 = f.rotOut;
			if (f.A.empty())
				OPENFHE_THROW("FIDESlib: the full-height bootstrap set has an empty transform step");

			const int srcLevel	= f.A.front().c0.getLevel();
			const int viewLevel = srcLevel - levelsToDrop;
			// The narrowest step must keep a level to rescale into, which is the same bound
			// AddBootstrapPrecomputationReduced states as levelsToDrop < lDec.
			if (viewLevel < 1)
				OPENFHE_THROW("FIDESlib: levelsToDrop " + std::to_string(levelsToDrop) + " leaves the bootstrap no output levels (a transform step sits at level " +
				  std::to_string(srcLevel) + ")");
			const double srcScale  = GPUcc.param.ScalingFactorReal.at(srcLevel);
			const double viewScale = GPUcc.param.ScalingFactorReal.at(viewLevel);

			t.A.reserve(f.A.size());
			for (const Plaintext& pt : f.A) {
				if (pt.c0.getLevel() != srcLevel || pt.NoiseFactor != srcScale || pt.NoiseLevel != 1)
					OPENFHE_THROW("FIDESlib: a full-height bootstrap diagonal is not the canonical encode at its level; "
								  "a shared view's scale fold would be wrong for it");
				t.A.emplace_back(GPUcc_);
				Plaintext& v = t.A.back();
				v.c0.aliasPrefixOf(pt.c0, viewLevel);
				v.NoiseFactor = viewScale;
				v.NoiseLevel  = pt.NoiseLevel;
				v.slots		  = pt.slots;
			}
			R *= srcScale / viewScale;
		}
		return R;
	};
	result.cts_fold = 1.0 / viewSteps(full.CtS, result.CtS);
	result.stc_fold = 1.0 / viewSteps(full.StC, result.StC);

	GPUcc.AddBootPrecomputation(slots, levelsToDrop, std::move(result));
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_) {
	ContextData& GPUcc							   = *GPUcc_;
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();

	if (!GPUcc.HasBootPrecomputation(slots)) {
		FIDESlib::CKKS::BootstrapPrecomputation result_;

		FIDESlib::CKKS::BootstrapPrecomputation& result = GPUcc.HasBootPrecomputation(slots) ? GPUcc_->GetBootPrecomputation(slots) : result_;

		std::vector<int> indexes = GetBootstrapIndexes(cc, slots, &result);

		AddBootstrapPlaintexts(cc, slots, GPUcc_, result);

		result.correctionFactor = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_correctionFactor;

		if (GPUcc.param.raw->sparse_encaps) {
			result.sparse_encaps = true;
		}

		GPUcc.AddBootPrecomputation(slots, std::move(result));
	}

	AddBootstrapKeys(publicKey, slots, GPUcc_);
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, const KeyPair<lbcrypto::DCRTPoly>& keys, int slots, FIDESlib::CKKS::Context& GPUcc_) {

	GenBootstrapKeys(keys, slots, GPUcc_->param.raw->sparse_encaps);

	AddBootstrapPrecomputation(keys.publicKey, slots, GPUcc_);
}
