
#ifndef __RAW_CIPHER_TEXT__
#define __RAW_CIPHER_TEXT__

#include "CKKS/forwardDefs.cuh"
#include <cinttypes>
#include <vector>
// #include "CKKS/BootstrapPrecomputation.cuh"
#include <openfhe.h>

namespace FIDESlib::CKKS {

constexpr bool REVERSE = false;

/*
 * The rawCipherText Class contains the basic information needed to hold a ciphertext, with limb data that can be transferred between the GPU and main memory
 * This is stored in RNS/DCRT format.
 * A ciphertext of degree 1 carries two components (sub_0, sub_1) and leaves sub_2 empty; a degree-2
 * ciphertext (the un-relinearized product of two ciphertexts) carries all three.
 */
struct RawCipherText {
	// lbcrypto::CryptoContext<lbcrypto::DCRTPoly> & cc; // Original CryptoContext object from OpenFHE;
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> originalCipherText; // Original Ciphertext object from OpenFHE;
	std::vector<std::vector<uint64_t>> sub_0;
	// uint64_t* sub_0; // pointer to sub-ciphertext 0
	std::vector<std::vector<uint64_t>> sub_1; // sub-ciphertext 1
	std::vector<std::vector<uint64_t>> sub_2; // sub-ciphertext 2, empty for a degree-1 ciphertext
	std::vector<uint64_t> moduli;			  // moduli for each limb
	int numRes;								  // number of residues of ciphertext, length of moduli array and first dimension of sub-ciphertexts
	int N;									  // length of each polynomial
	Format format;							  // current format of ciphertext, either coefficient or evaluation
	double Noise;
	int NoiseLevel;
	int slots;
	std::string keyid;
	// GPUCKKS::Event e;
};

struct RawPlainText {
	//  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> & cc; // Original CryptoContext object from OpenFHE;
	lbcrypto::Plaintext originalPlainText; // Original Ciphertext object from OpenFHE;
	std::vector<std::vector<uint64_t>> sub_0;
	std::vector<uint64_t> moduli; // moduli for each limb
	int numRes;					  // number of residues of ciphertext, length of moduli array and first dimension of sub-ciphertexts
	int N;						  // length of each polynomial
	Format format;				  // current format of ciphertext, either coefficient or evaluation
	double Noise;
	int slots;
	int NoiseLevel;
};

struct RawParams {
	int N;
	int L;
	int K;
	int logN;
	lbcrypto::ScalingTechnique scalingTechnique;
	std::vector<uint64_t> moduli;
	std::vector<uint64_t> root_of_unity;
	std::vector<uint64_t> cyclotomic_order;
	std::vector<uint64_t> SPECIALmoduli;
	std::vector<uint64_t> SPECIALroot_of_unity;
	std::vector<uint64_t> SPECIALcyclotomic_order;
	std::map<int, std::vector<uint64_t>> psi;
	std::map<int, std::vector<uint64_t>> psi_inv;
	std::vector<uint64_t> N_inv;
	std::vector<double> ModReduceFactor;
	std::vector<std::vector<uint64_t>> m_QlQlInvModqlDivqlModq;

	int dnum;
	std::vector<std::vector<uint64_t>> PARTITIONmoduli;
	std::vector<std::vector<std::vector<uint64_t>>> PartQlHatInvModq;
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> PartQlHatModp;
	std::vector<uint64_t> PHatInvModp;
	std::vector<std::vector<uint64_t>> PHatModq;
	std::vector<uint64_t> PInvModq;

	std::vector<double> ScalingFactorReal;
	std::vector<double> ScalingFactorRealBig;

	/** Bootstrapping */
	std::vector<double> coefficientsCheby;
	int doubleAngleIts{ 0 };
	uint32_t bootK;
	uint32_t correctionFactor;
	bool sparse_encaps{ false };
	int p;
};

struct RawKeySwitchKey {
	std::vector<std::vector<std::vector<uint64_t>>> r_key_moduli;
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> r_key;
	std::vector<std::vector<std::vector<uint64_t>>> dcrt_keys;
	std::string keyid;
};

/// Read-only: takes the limb vector by reference. Every call site passes the result of
/// DCRTPoly::GetAllElements(), which is a `const std::vector<PolyType>&`, so a by-value parameter
/// deep-copied every limb of the polynomial just to read it back out.
std::vector<std::vector<uint64_t>> GetRawArray(const std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>>& polys);

RawCipherText GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct, int REV = 1);

void GetOpenFHECipherText(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result, RawCipherText raw, int REV = 1);

RawPlainText GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Plaintext pt);
RawPlainText GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::ReadOnlyPlaintext pt);

void GetOpenFHEPlaintext(lbcrypto::Plaintext result, RawPlainText raw, int REV = 1);

// RawParams GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
// `chebyshevDegreeCap` is opt-in: 0 keeps the full coefficient table for `boot_conf` (stock
// behaviour), any smaller degree truncates the series that approxModReduction evaluates, which
// lowers the Paterson-Stockmeyer depth the mod-reduction stage consumes. The caller is responsible
// for reserving a matching level budget at bootstrap setup (see fideslib::bootstrapModEvalLevels).
RawParams GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, FIDESlib::BOOT_CONFIG boot_conf = UNIFORM, uint32_t chebyshevDegreeCap = 0);

RawKeySwitchKey GetKeySwitchKey(std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> ksk);

RawKeySwitchKey GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys);

RawKeySwitchKey GetEvalKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey);

RawKeySwitchKey GetRotationKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);

RawKeySwitchKey GetRotationKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int index);

RawKeySwitchKey GetConjugateKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey);

void GenAndAddRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes);

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> GenRotationKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes);

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> GenRotationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes);

void AddRotationKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes);

/** Used in AddBootstrapPrecomputation */
void AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::Context& GPUcc_, FIDESlib::CKKS::BootstrapPrecomputation& result);

/** Used in GenBootstrapKeys and AddBootstrapKeys */
std::vector<int> GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::BootstrapPrecomputation* result_);

/** Server-side bootstrap setup */
void AddBootstrapPrecomputation(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_);


/**
 * @brief Install a bootstrap precomputation whose CoeffsToSlots / SlotsToCoeffs diagonals are
 *        encoded @p levelsToDrop towers below the chain top, so Bootstrap(..., levelsToDrop)
 *        can stop the mod-raise there.
 *
 * Opt-in and additive: the full-height precomputation the ordinary bootstrap reads is left
 * exactly as it is, and a context that never calls this behaves as before. The diagonals are
 * ENCODED at the reduced levels by OpenFHE's own EvalCoeffsToSlotsPrecompute /
 * EvalSlotsToCoeffsPrecompute (the `L` argument those take exists for precisely this), not
 * derived from the full-height ones, so the transform's precision at the reduced height is
 * the precision it has at the full height.
 *
 * Costs device memory: one extra diagonal set per (slots, levelsToDrop), smaller than the
 * full-height set by @p levelsToDrop limbs per plaintext. Calling it twice with the same
 * arguments is a no-op.
 *
 * Supported for a level budget other than {1,1} (the collapsed-FFT transforms). A {1,1}
 * budget uses the single-step linear transform and is rejected rather than silently ignored.
 *
 * @param levelsToDrop  0 is the default set and returns without doing anything.
 */
void AddBootstrapPrecomputationReduced(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int slots, int levelsToDrop, FIDESlib::CKKS::Context& GPUcc_);

/**
 * @brief The same reduced-level set as AddBootstrapPrecomputationReduced, built WITHOUT new
 *        diagonal storage: every plaintext is a non-owning view of the full-height set's first
 *        (limbs - @p levelsToDrop) limbs and its special limbs, and the set carries the two real
 *        scale folds (BootstrapPrecomputation::cts_fold / stc_fold) that cancel the difference
 *        between the full-height integers and a canonical encode at the lower levels.
 *
 * Device memory: one pointer table per view plaintext (MAXP x 8 x (4 + 4 dnum) bytes, 8 KiB at
 * dnum 3), no limb data. Installed under the same (slots, levelsToDrop) key, so Bootstrap and the
 * transforms read it through GetBootPrecomputation unchanged. The full-height set must already be
 * installed and must outlive it (clearBootPrecomputation drops a slot count's sets together).
 * Calling it twice with the same arguments, or after AddBootstrapPrecomputationReduced for the
 * same key, is a no-op. A diagonal that is not the canonical encode at its level throws.
 *
 * @param levelsToDrop  0 is the default set and returns without doing anything.
 */
void AddBootstrapPrecomputationShared(int slots, int levelsToDrop, FIDESlib::CKKS::Context& GPUcc_);

/** DEPRECATED: split into GenBootstrapKeys and AddBootstrapPrecomputation overload with public key*/
void AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, FIDESlib::CKKS::Context& GPUcc);

/**
 * Used in AddBootstrapPrecomputation
 */
void AddBootstrapKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_);

/**
 * Client-side bootstrap setup: Keys for FIDESlib Bootstrap do not match OpenFHE's, be careful if you intend to run Bootstrap on CPU
 *
 * The code can be used to implement SSE and regular Boot on a regular non-FIDESlib OpenFHE installation with server-side FIDESlib.
 */
void GenBootstrapKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, bool SSE);
void GenBootstrapKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, int slots, bool SSE = false);
} // namespace FIDESlib::CKKS

#endif //__RAW_CIPHER_TEXT__