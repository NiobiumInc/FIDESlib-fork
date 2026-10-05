//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cinttypes>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {

class KeySwitchingKey {
	static constexpr const char* loc{ "KeySwitchingKey" };
	CudaNvtxRange my_range;

  public:
	KeyHash keyID;
	Context& cc;
	RNSPoly a;
	RNSPoly b;
	// std::vector<RNSPoly> mgpu_a;
	// std::vector<RNSPoly> mgpu_b;

	explicit KeySwitchingKey(Context& cc);

	void Initialize(RawKeySwitchKey& rkk);

	/**
	 * Load a SPARSE_ENCAPSULATED 2N-4 GHS single-digit key. Unlike Initialize
	 * (a dnum-digit hybrid key), this key's A/B vectors are each ONE (q0,p) DCRTPoly
	 * — a single digit over two primes — so a and b are loaded as plain 2-limb
	 * polynomials (q0 as a Q limb, p as a special limb) via RNSPoly::load, with no
	 * decomp/digit structure. Consumed only by keySwitchSparse, never by the generic
	 * hybrid keySwitch. See FHECKKSRNS::KeySwitchGenSparse (ckksrns-fhe.cpp:3758).
	 */
	void InitializeSparse(RawKeySwitchKey& rkk);
};

} // namespace CKKS
} // namespace FIDESlib

#endif // GPUCKKS_KEYSWITCHINGKEY_CUH
