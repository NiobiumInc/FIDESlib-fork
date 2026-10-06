// BOOTSTRAP PRECISION, CPU vs GPU, at a fixed parameter point (N = 2^16, depth 25, scale 55,
// first modulus 56, budget {3,2}, dnum 6, FLEXIBLEAUTO, SPARSE_TERNARY, 2^15 slots).
//
// WHY. A one-iteration refresh on this point measures 18.6 bits on the CPU (OpenFHE) and 11.5-15
// bits on the GPU, with the same secret and the same relative levels; the two-iteration (Meta-BTS)
// refresh hides the loss (29-32 bits on the GPU). ApiParityBootstrapTest compares the two backends
// at 1e-2, which a 4-7 bit loss passes. These tests measure against the PLAINTEXT, in bits.
//
//   OneIterationCpuVsGpu  one input ciphertext, refreshed by the host OpenFHE context (the oracle)
//                         and by the device (natural output and EvalBootstrapToLevel targets);
//                         prints both precisions. REPORTS today; the bound is asserted by the fix.
//   GpuStageNoise         the device refresh of a ZERO message and of a signal, decrypted at every
//                         ChecksumProbe point through the test-only stage hook. From EvalMod on, a
//                         zero message must decrypt to zero, so rms(zero run) / rms(signal run) is
//                         the stage's noise-to-signal ratio; before EvalMod the zero run holds
//                         integers (the mod-raise overflow I/K), so the ratio is taken on the
//                         distance to that lattice. The stage where the ratio jumps adds the loss.
//
// Run with FIDESLIB_TEST_BACKEND=cuda; skipped otherwise. Test code: holds the secret key.

#include <gtest/gtest.h>
#include <openfhe.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "CKKS/Checksum.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "fideslib.hpp"

using namespace fideslib;

namespace {

bool UseCuda() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	return b != nullptr && std::string(b) == "cuda" && IsBackendAvailable(Backend::CUDA);
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& LbCc(CryptoContext<DCRTPoly>& cc) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->host);
}

class BootstrapPrecision : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth	= 25;
	static constexpr uint32_t kScaleMod = 55;
	static constexpr uint32_t kFirstMod = 56;
	static constexpr uint32_t kRingDim	= 1u << 16;
	static constexpr uint32_t kSlots	= 1u << 15;
	static constexpr uint32_t kDnum		= 6;
	const std::vector<uint32_t> kBudget = { 3, 2 };

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	// The deepest input the host bootstrap accepts (its guard: L0 - (budget[1] + 2)).
	uint32_t inLevel() const {
		return kDepth - (kBudget[1] + 2);
	}

	void SetUp() override {
		if (!UseCuda())
			GTEST_SKIP() << "set FIDESLIB_TEST_BACKEND=cuda: this compares the device refresh with the host oracle";
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(kScaleMod);
		params.SetFirstModSize(kFirstMod);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FLEXIBLEAUTO);
		params.SetKeySwitchTechnique(HYBRID);
		params.SetNumLargeDigits(kDnum);
		params.SetSecretKeyDist(SPARSE_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		params.SetBackend(Backend::CUDA);
		params.SetReducedNoise(LinkedOpenFheReducedNoise()); // matches the oracle's variant
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		cc->EvalBootstrapSetup(kBudget, { 0, 0 }, kSlots);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		cc->LoadContext(keys.publicKey);
	}

	const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& LbSk() {
		return std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(keys.secretKey->pimpl);
	}

	std::vector<std::complex<double>> DecryptHost(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct, size_t n) {
		lbcrypto::Plaintext pt;
		LbCc(cc)->Decrypt(LbSk(), ct, &pt);
		pt->SetLength(n);
		return pt->GetCKKSPackedValue();
	}

	// A device ciphertext as a host one, the CudaEngine::refreshHostShadow way: a full-tower zero
	// shell from the host parameters, overwritten with the device limbs and metadata.
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> DeviceToHost(const FIDESlib::CKKS::Ciphertext& dev) {
		FIDESlib::CKKS::Ciphertext tmp(dev.cc_);
		tmp.copy(dev);
		FIDESlib::CKKS::RawCipherText raw;
		tmp.store(raw);
		auto& lbcc			  = LbCc(cc);
		const auto elemParams = lbcc->GetCryptoParameters()->GetElementParams();
		lbcrypto::DCRTPoly zero(elemParams, Format::EVALUATION, true);
		auto shell = lbcc->Encrypt(LbSk(), lbcc->MakeCKKSPackedPlaintext(std::vector<double>(8, 0.0), 1, 0, nullptr, kSlots));
		shell->SetElements(std::vector<lbcrypto::DCRTPoly>(raw.sub_2.empty() ? 2u : 3u, zero));
		shell->SetLevel(0);
		FIDESlib::CKKS::GetOpenFHECipherText(shell, raw);
		return shell;
	}

	static double Bits(double err) {
		return err > 0 ? -std::log2(err) : 99.0;
	}
};

double MaxErr(const std::vector<std::complex<double>>& got, const std::vector<double>& want) {
	double e = 0.0;
	for (size_t i = 0; i < want.size(); ++i)
		e = std::max(e, std::abs(got[i].real() - want[i]));
	return e;
}

} // namespace

TEST_F(BootstrapPrecision, OneIterationCpuVsGpu) {
	std::mt19937_64 rng(0);
	std::uniform_real_distribution<double> u(-1.0, 1.0);
	std::vector<double> x(kSlots);
	for (auto& v : x)
		v = u(rng);
	auto pt	  = cc->MakeCKKSPackedPlaintext(x, 1, inLevel(), nullptr, kSlots);
	auto ct	  = cc->Encrypt(keys.publicKey, pt);
	auto lbCt = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);

	// The oracle: the host OpenFHE context, same keys, same input ciphertext.
	auto cpu		 = LbCc(cc)->EvalBootstrap(lbCt, 1, 0);
	const double ecp = MaxErr(DecryptHost(cpu, kSlots), x);
	std::printf("[bootprec] CPU oracle   iters=1 in %u -> out %zu: abs err %.3e (%.1f bits)\n", inLevel(), cpu->GetLevel(), ecp, Bits(ecp));

	auto readback = [&](Ciphertext<DCRTPoly>& r) {
		cc->RecoverHostCiphertext(r);
		return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(r->host);
	};
	auto gpu		 = cc->EvalBootstrap(ct, 1, 0);
	const double egp = MaxErr(DecryptHost(readback(gpu), kSlots), x);
	std::printf("[bootprec] GPU natural  iters=1 in %u -> out %zu: abs err %.3e (%.1f bits)\n", inLevel(), gpu->GetLevel(), egp, Bits(egp));
	for (uint32_t L : { 15u, 16u, 17u, 18u, 19u }) {
		try {
			auto t			= cc->EvalBootstrapToLevel(ct, L, 1, 0);
			const double et = MaxErr(DecryptHost(readback(t), kSlots), x);
			std::printf("[bootprec] GPU L=%-2u     iters=1 in %u -> out %zu: abs err %.3e (%.1f bits)\n", L, inLevel(), t->GetLevel(), et, Bits(et));
		} catch (const std::exception& e) {
			std::printf("[bootprec] GPU L=%-2u     FAILED: %s\n", L, e.what());
		}
	}
	std::printf("[bootprec] GAP natural: GPU %.1f bits vs CPU %.1f bits (%.1f bits short)\n", Bits(egp), Bits(ecp), Bits(ecp) - Bits(egp));
	std::fflush(stdout);
	// Sanity only (the refresh produced the message at all); the precision bound is the fix's assertion.
	EXPECT_LT(egp, 1e-2);
	EXPECT_LT(ecp, 1e-2);
}

TEST_F(BootstrapPrecision, GpuStageNoise) {
	const auto K = static_cast<double>(lbcrypto::FHECKKSRNS::K_SPARSE);
	std::mt19937_64 rng(1);
	std::uniform_real_distribution<double> u(-1.0, 1.0);
	std::vector<double> sig(kSlots), zero(kSlots, 0.0);
	for (auto& v : sig)
		v = u(rng);

	// Per tag, in probe order: the decrypted slot vector of the run in progress.
	struct Stage {
		std::string tag;
		std::vector<std::complex<double>> v;
		size_t towers;
		int noiseLevel;
	};
	auto capture = [&](const std::vector<double>& msg) {
		std::vector<Stage> out;
		FIDESlib::CKKS::SetStageHook([&](const char* tag, const FIDESlib::CKKS::Ciphertext& dev) {
			Stage s{ tag, {}, 0, dev.NoiseLevel };
			// One try per datum: a stage that cannot be read is named, the others still land.
			try {
				auto h	 = DeviceToHost(dev);
				s.towers = h->GetElements()[0].GetNumOfElements();
				s.v		 = DecryptHost(h, kSlots);
			} catch (const std::exception& e) {
				std::printf("[stage] %-14s UNREADABLE: %s\n", tag, e.what());
			}
			out.push_back(std::move(s));
		});
		auto pt = cc->MakeCKKSPackedPlaintext(msg, 1, inLevel(), nullptr, kSlots);
		auto ct = cc->Encrypt(keys.publicKey, pt);
		auto r	= cc->EvalBootstrap(ct, 1, 0);
		FIDESlib::CKKS::SetStageHook(nullptr);
		return out;
	};
	const auto Z = capture(zero);
	const auto S = capture(sig);
	ASSERT_EQ(Z.size(), S.size()) << "the two runs took different probe paths";

	auto rms = [](const std::vector<std::complex<double>>& v, auto f) {
		double a = 0.0;
		for (const auto& c : v)
			a += f(c) * f(c);
		return v.empty() ? 0.0 : std::sqrt(a / v.size());
	};
	auto frac = [](double y) { return y - std::round(y); };
	bool after_evalmod = false;
	std::printf("[stage] %-14s %6s %3s  %-10s %-10s  %s\n", "tag", "towers", "nl", "zero rms", "sig rms", "noise/signal");
	for (size_t i = 0; i < Z.size(); ++i) {
		const auto& z = Z[i];
		const auto& s = S[i];
		if (z.tag == "post-evalmod" || z.tag == "em-out")
			after_evalmod = true;
		if (z.v.empty() || s.v.empty())
			continue;
		const bool pre = !after_evalmod && z.tag != "in" && z.tag.rfind("em-", 0) != 0;
		if (pre) {
			// Before EvalMod: real and imaginary parts carry (I + c/q0)/K-type values; pick the
			// lattice unit (K/2, K, 2K) on which the zero run sits closest to integers.
			double best = 1e300, unit = K;
			for (double m : { 0.5 * K, K, 2.0 * K }) {
				const double d = rms(z.v, [&](const std::complex<double>& c) { return std::hypot(frac(m * c.real()), frac(m * c.imag())); });
				if (d < best) {
					best = d;
					unit = m;
				}
			}
			const double sigf = rms(s.v, [&](const std::complex<double>& c) { return std::hypot(frac(unit * c.real()), frac(unit * c.imag())); });
			std::printf("[stage] %-14s %6zu %3d  %.3e  %.3e  %.3e (2^%.1f)  [lattice unit %.0f]\n", z.tag.c_str(), z.towers, z.noiseLevel, best, sigf,
			  best / sigf, std::log2(best / sigf), unit);
		} else {
			const double zr = rms(z.v, [](const std::complex<double>& c) { return std::abs(c); });
			const double sr = rms(s.v, [](const std::complex<double>& c) { return std::abs(c); });
			const bool meaningful = after_evalmod || z.tag == "in";
			std::printf("[stage] %-14s %6zu %3d  %.3e  %.3e  %s\n", z.tag.c_str(), z.towers, z.noiseLevel, zr, sr,
			  meaningful ? (std::to_string(zr / sr) + " (2^" + std::to_string(std::log2(zr / sr)) + ")").c_str() : "(inside EvalMod: not a noise ratio)");
		}
	}
	std::fflush(stdout);
	EXPECT_FALSE(Z.empty()) << "the stage hook never fired: the bootstrap took no ChecksumProbe path";
}
