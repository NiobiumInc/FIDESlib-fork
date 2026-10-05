// Minimal haze-backend example: encrypt two numbers, add them, read the result.
//
// The backend comes from FIDESLIB_BACKEND (cpu | cuda | haze) and the replay
// target from FIDESLIB_HAZE_TARGET (unset = in-process simulator). The target
// also selects the data format: point it at a Montgomery device (func_sim_hw,
// an FPGA device id) to run the hardware datapath. The program does not change.
//
// Four lines differ from the same program written against plain OpenFHE, marked
// [D1]..[D4] below: two replace an OpenFHE line, two are additions. Everything
// else, SetRingDim included, is unchanged OpenFHE API. Lines marked [O1]..[O3]
// are FIDESlib-only but optional in this program; see "Differences from plain
// OpenFHE" in the haze documentation for what each one buys.

#include <fideslib.hpp> // [D1] replaces #include "openfhe.h"
#include "BackendEnv.hpp" // [O1] only for BackendFromEnv; a literal Backend value needs no include

#include <iostream>

using namespace fideslib; // [D2] replaces using namespace lbcrypto

int main() {
	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(1);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(8);
	// Hardware targets use ring_dim=65536; these parameters would otherwise
	// pick 16384. Harmless on the simulators, just slower.
	parameters.SetRingDim(65536);
	parameters.SetBackend(BackendFromEnv()); // [D3] added; OpenFHE has no backend to select
	parameters.SetReducedNoise(true); // [O2] centered FBC, for bit-exact parity with a WITH_REDUCED_NOISE OpenFHE

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	// Uploads the context and keys to the backend. Required before any encrypt
	// or evaluation call on a device backend.
	cc->LoadContext(keys.publicKey); // [D4] added; haze aborts without it, no-op on CPU

	std::vector<double> a = { 2.5 };
	std::vector<double> b = { 4.25 };

	// Encrypt takes Plaintext&, so the encodings need names.
	Plaintext ptA = cc->MakeCKKSPackedPlaintext(a);
	Plaintext ptB = cc->MakeCKKSPackedPlaintext(b);

	auto ctA = cc->Encrypt(keys.publicKey, ptA);
	auto ctB = cc->Encrypt(keys.publicKey, ptB);

	auto ctSum = cc->EvalAdd(ctA, ctB);

	// Declare every ciphertext that will be read back. haze records one program
	// per context and executes it once at the first decryption, so outputs must
	// be declared before that point. No-op on the CPU/CUDA backends.
	cc->MarkOutput(ctSum); // [O3] not required for a single output, which its own decrypt covers

	Plaintext result;
	cc->Decrypt(keys.secretKey, ctSum, &result);
	result->SetLength(1);

	std::cout << a[0] << " + " << b[0] << " = " << result->GetRealPackedValue()[0] << std::endl;
	return 0;
}
