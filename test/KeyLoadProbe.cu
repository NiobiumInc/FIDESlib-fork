// PROBE (not for merge): what loading key-switching keys costs the calling thread.
#include <openfhe.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

namespace FIDESlib::Testing {

class KeyLoadProbe : public GeneralParametrizedTest {};

TEST_P(KeyLoadProbe, LoadRotationKeys) {
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	const std::vector<int> rot_index{ 1, 2, 3, 4, 5, 6, 7, 8 };
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, rot_index);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	FIDESlib::CKKS::Context GPUcc_		= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc	= *GPUcc_;

	std::vector<FIDESlib::CKKS::RawKeySwitchKey> raws;
	for (int idx : rot_index)
		raws.push_back(FIDESlib::CKKS::GetRotationKeySwitchKey(keys, idx, cc));

	double best_host = 1e300, best_total = 1e300;
	for (int rep = 0; rep < 5; ++rep) {
		cudaDeviceSynchronize();
		const auto t0 = std::chrono::steady_clock::now();
		for (size_t k = 0; k < rot_index.size(); ++k) {
			FIDESlib::CKKS::KeySwitchingKey ksk(GPUcc_);
			ksk.Initialize(raws[k]);
			GPUcc.AddRotationKey(rot_index[k], std::move(ksk));
		}
		const auto t1 = std::chrono::steady_clock::now();
		cudaDeviceSynchronize();
		const auto t2 = std::chrono::steady_clock::now();
		best_host = std::min(best_host, std::chrono::duration<double, std::milli>(t1 - t0).count());
		best_total = std::min(best_total, std::chrono::duration<double, std::milli>(t2 - t0).count());
	}
	const char* env = std::getenv("FIDESLIB_PROBE_POOLED_PT");
	std::cout << "[bench] keyload pooled=" << ((env && env[0] == '1') ? 1 : 0) << " keys=" << rot_index.size() << " host_ms=" << best_host << " total_ms=" << best_total
			  << std::endl;
	CKKS::DeregisterAllContexts();
}

INSTANTIATE_TEST_SUITE_P(KeyLoadProbes, KeyLoadProbe, testing::Values(tparams64_13_1_flex, tparams64_13_2_fix));

} // namespace FIDESlib::Testing
