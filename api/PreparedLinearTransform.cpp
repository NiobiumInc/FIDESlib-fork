#include "PreparedLinearTransform.hpp"

#include <openfhe.h>

#include "CryptoContext.hpp"
#include "Plaintext.hpp"

namespace fideslib {

std::string LinearTransformShape::str() const {
	return "rowSize=" + std::to_string(rowSize) + " bStep=" + std::to_string(bStep) + " stride=" + std::to_string(stride) + " offset=" + std::to_string(offset) +
	  " ext=" + (ext ? "true" : "false");
}

PreparedLinearTransformImpl::PreparedLinearTransformImpl(const LinearTransformShape& shape,
  std::vector<std::vector<Plaintext>> sets,
  std::vector<std::vector<uint64_t>> identity)
: shape_(shape), sets_(std::move(sets)), identity_(std::move(identity)) {
	if (sets_.empty())
		OPENFHE_THROW("PreparedLinearTransform: needs at least one diagonal set");
	if (identity_.size() != sets_.size())
		OPENFHE_THROW("PreparedLinearTransform: identity table does not match the diagonal set count");
	for (size_t t = 0; t < sets_.size(); ++t) {
		if (identity_[t].size() != sets_[t].size())
			OPENFHE_THROW("PreparedLinearTransform: identity table does not match the diagonal count of set " + std::to_string(t));
	}
}

const std::vector<Plaintext>& PreparedLinearTransformImpl::Diagonals() const {
	if (sets_.size() != 1) {
		OPENFHE_THROW("PreparedLinearTransform: this handle carries " + std::to_string(sets_.size()) +
		  " diagonal sets; it is a batch handle and belongs to LinearTransformMany, not LinearTransformInPlace");
	}
	return sets_[0];
}

void PreparedLinearTransformImpl::RequireFresh(CryptoContextImpl<DCRTPoly>& ctx, const char* what) const {
	for (size_t t = 0; t < sets_.size(); ++t) {
		const std::string where = sets_.size() == 1 ? std::string{} : (" of set " + std::to_string(t));
		for (size_t k = 0; k < sets_[t].size(); ++k) {
			if (!sets_[t][k]) {
				// Unreachable through PrepareLinearTransform* (they reject nulls), so reaching it
				// means the handle itself was corrupted. Still a throw, never a launch.
				OPENFHE_THROW(std::string{ what } + ": prepared diagonal " + std::to_string(k) + where + " is null");
			}
			const uint64_t now = ctx.PlaintextIdentity(sets_[t][k]);
			if (now != identity_[t][k]) {
				OPENFHE_THROW(std::string{ what } + ": diagonal " + std::to_string(k) + where +
				  " has been re-encoded or released since this transform was prepared (identity " + std::to_string(identity_[t][k]) + " -> " +
				  std::to_string(now) +
				  "). A prepared transform points at the device buffers of the diagonals it was built from; "
				  "rebuild the handle with PrepareLinearTransform. This is a throw and not a fallback because "
				  "reading the old pointers would be a wrong answer, not a slowdown.");
			}
		}
	}
}

} // namespace fideslib
