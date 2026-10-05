//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include "Plaintext.cuh"
#include <vector>

namespace FIDESlib::CKKS {

class BootstrapPrecomputation {
  public:
	struct {
		int slots = -1;
		int bStep = -1;
		std::vector<Plaintext> A;
		std::vector<Plaintext> invA;
	} LT;

	struct LTstep {
		int slots = -1;
		int bStep = -1;
		int gStep = -1;
		std::vector<Plaintext> A;
		std::vector<int> rotIn;
		std::vector<int> rotOut;
	};

	std::vector<LTstep> StC;
	std::vector<LTstep> CtS;
	int accumulate_bStep = 4;
	uint32_t correctionFactor;
	// SPARSE_ENCAPSULATED is now an in-context dance. The two switching keys live in
	// the MAIN context's rotation-key map at 2N-2 / 2N-4 (see AddBootstrapKeys); there is no
	// second "switchable" GPU context, so the old `std::weak_ptr<ContextData> sparse_context`
	// member is gone.
	bool sparse_encaps{ false };
	/// @brief How many Q towers below the chain top this set's transform plaintexts are
	/// encoded at. 0 is the full-height set every ordinary bootstrap uses. A non-zero value
	/// is what makes a shortened mod-raise legal: the CoeffsToSlots/SlotsToCoeffs diagonals
	/// must be encoded at exactly the levels the ciphertext meets them at
	/// (LinearTransform asserts the equality), so a bootstrap that raises to a lower
	/// modulus needs its own set. See AddBootstrapPrecomputationReduced.
	int levels_to_drop{ 0 };
	/// @brief Real factors this set's diagonals need folded into the bootstrap, both 1.0 for a set
	/// whose diagonals are encoded at their own levels (every set but a shared view).
	///
	/// `cts_fold` multiplies the constant applied to the raised ciphertext before CoeffsToSlots
	/// (`constantEvalMult` in Bootstrap); `stc_fold` is the factor EvalMod's output is delivered
	/// at, folded into the Chebyshev coefficients and double-angle constants (approxModReduction's
	/// `outputScale`). Both transforms are linear, so a factor in front of one is a factor on its
	/// output; each is the reciprocal of the product, over that transform's steps, of how far the
	/// step's integers sit from a canonical encode at the level they are read at. See
	/// AddBootstrapPrecomputationShared for where the ratios come from. At 1.0 Bootstrap takes
	/// neither branch.
	double cts_fold{ 1.0 };
	double stc_fold{ 1.0 };
	/// @brief True for a set whose diagonals are non-owning limb-prefix views of the full-height
	/// set's (AddBootstrapPrecomputationShared). It holds no limb memory of its own and is only
	/// valid while the full-height set of the same slot count is installed;
	/// ContextData::clearBootPrecomputation removes every height of a slot count together.
	bool shared_view{ false };
};

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
