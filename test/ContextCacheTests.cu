#include "ParametrizedTest.cuh"

#include <gtest/gtest.h>
#include <openfhe.h>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::Testing {

// The GPU context cache is keyed on FIDESlib::CKKS::Parameters. Two contexts differing only in
// bootstrap configuration share every ring/moduli field, so if Parameters::operator< ignores the
// boot config they compare equal and GenCryptoContextGPU serves the first one for both -  a
// sparse-key context would silently run the degree-88 uniform Chebyshev table with K_UNIFORM.
// Holding both alive at once is what makes them collide; ~CudaEngine's deregistration hides the
// bug whenever contexts are strictly sequential.
TEST(ContextCacheBootConfig, SparseAndUniformDoNotAlias) {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(25);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(8);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetRingDim(1 << 12);
	parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);

	auto cc = lbcrypto::GenCryptoContext(parameters);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	FIDESlib::CKKS::RawParams rawUniform = FIDESlib::CKKS::GetRawParams(cc, FIDESlib::UNIFORM);
	FIDESlib::CKKS::RawParams rawSparse	 = FIDESlib::CKKS::GetRawParams(cc, FIDESlib::SPARSE);

	// Pin the host-side tables first, so a failure below is unambiguously the cache and not
	// GetRawParams. g_coefficientsUniform is degree 88 (89 coefficients), g_coefficientsSparse
	// is degree 44 (45).
	ASSERT_EQ(rawUniform.coefficientsCheby.size(), 89u);
	ASSERT_EQ(rawSparse.coefficientsCheby.size(), 45u);

	FIDESlib::CKKS::Parameters base{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

	FIDESlib::CKKS::Context gpuUniform = FIDESlib::CKKS::GenCryptoContextGPU(base.adaptTo(rawUniform), devices);
	FIDESlib::CKKS::Context gpuSparse  = FIDESlib::CKKS::GenCryptoContextGPU(base.adaptTo(rawSparse), devices);

	EXPECT_EQ(gpuUniform->GetCoeffsChebyshev().size(), 89u) << "uniform context did not get g_coefficientsUniform";
	EXPECT_EQ(gpuSparse->GetCoeffsChebyshev().size(), 45u) << "sparse context was served the cached uniform one: Parameters::operator< ignores the boot config";
	EXPECT_EQ(gpuUniform->GetDoubleAngleIts(), 6);
	EXPECT_EQ(gpuSparse->GetDoubleAngleIts(), 3);

	FIDESlib::CKKS::DeregisterAllContexts();
}

} // namespace FIDESlib::Testing
