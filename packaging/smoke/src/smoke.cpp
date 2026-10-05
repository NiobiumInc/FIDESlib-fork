// Redistributable-package smoke test: encrypt two numbers, add them under
// encryption, decrypt, and check the result.
//
// This exists to prove the shipped tarball is self-contained. packaging/test-package.sh
// builds it with plain -I/-L/-l flags against the installed package -- no cmake, no
// find_package(CONFIG) -- since a consumer with no CMake at all still has to be able
// to link this. See CMakeLists.txt alongside this file for the equivalent CMake-based
// build, kept as a convenience for consumers who do use it.
//
// The backend comes from FIDESLIB_BACKEND (cpu | haze). On haze the default replay
// target is the in-process simulator, which needs no external replay binary;
// FIDESLIB_HAZE_TARGET selects FUNC_SIM or an FPGA device instead.

#include <fideslib.hpp>
#include "BackendEnv.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

// Tolerance for a single CKKS add at 50-bit scaling. The operation is exact in the
// plaintext domain; this only absorbs encode/decode rounding.
constexpr double kTolerance = 1e-6;

// Hardware targets use ring_dim=65536. Harmless but much slower on the
// simulators, so it is opt-in rather than the default.
bool wantsLargeRing() {
	const char* t = std::getenv("FIDESLIB_HAZE_TARGET");
	return t != nullptr && std::string(t).rfind("FPGA", 0) == 0;
}

} // namespace

int main() {
	using namespace fideslib;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(1);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(8);
	if (wantsLargeRing()) {
		parameters.SetRingDim(65536);
	}
	parameters.SetBackend(BackendFromEnv());
	// Centered FBC, for bit-exact parity with a WITH_REDUCED_NOISE OpenFHE oracle.
	parameters.SetReducedNoise(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	// Uploads context and keys to the backend. Required before any encrypt or
	// evaluation call on a device backend; a no-op on CPU.
	cc->LoadContext(keys.publicKey);

	const std::vector<double> a = {2.5};
	const std::vector<double> b = {4.25};
	const double expected = a[0] + b[0];

	Plaintext ptA = cc->MakeCKKSPackedPlaintext(a);
	Plaintext ptB = cc->MakeCKKSPackedPlaintext(b);

	auto ctA	= cc->Encrypt(keys.publicKey, ptA);
	auto ctB	= cc->Encrypt(keys.publicKey, ptB);
	auto ctSum	= cc->EvalAdd(ctA, ctB);

	// haze records one program per context and executes it at the first decryption,
	// so every value read back must be declared before that point. No-op elsewhere.
	cc->MarkOutput(ctSum);

	Plaintext result;
	cc->Decrypt(keys.secretKey, ctSum, &result);
	result->SetLength(1);

	const double got = result->GetRealPackedValue()[0];
	const double err = std::fabs(got - expected);

	std::cout << "backend=" << (std::getenv("FIDESLIB_BACKEND") ? std::getenv("FIDESLIB_BACKEND") : "(default)")
			  << " target=" << (std::getenv("FIDESLIB_HAZE_TARGET") ? std::getenv("FIDESLIB_HAZE_TARGET") : "local")
			  << ": " << a[0] << " + " << b[0] << " = " << got
			  << " (expected " << expected << ", error " << err << ")" << std::endl;

	if (!(err < kTolerance)) {
		std::cerr << "FAIL: error " << err << " exceeds tolerance " << kTolerance << std::endl;
		return 1;
	}
	std::cout << "PASS" << std::endl;
	return 0;
}
