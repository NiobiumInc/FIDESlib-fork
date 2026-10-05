#include "HostMath.hpp"

#include <openfhe.h> // OPENFHE_THROW

#include <algorithm>
#include <cmath>

namespace fideslib {

// Moved verbatim from src/PolyApprox.cu (was in the GPU CKKS namespace).
std::vector<double> get_chebyshev_coefficients(const std::function<double(double)>& func, const double a, const double b, const uint32_t degree) {
	if (!degree) {
		OPENFHE_THROW("The degree of approximation can not be zero");
	}

	const size_t coeffTotal{ degree + 1 };
	const double bMinusA = 0.5 * (b - a);
	const double bPlusA	 = 0.5 * (b + a);
	const double PiByDeg = M_PI / static_cast<double>(coeffTotal);
	std::vector<double> functionPoints(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i)
		functionPoints[i] = func(std::cos(PiByDeg * (i + 0.5)) * bMinusA + bPlusA);

	const double multFactor = 2.0 / static_cast<double>(coeffTotal);
	std::vector<double> coefficients(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i) {
		for (size_t j = 0; j < coeffTotal; ++j)
			coefficients[i] += functionPoints[j] * std::cos(PiByDeg * i * (j + 0.5));
		coefficients[i] *= multFactor;
	}
	return coefficients;
}

// Moved verbatim from src/CKKS/LinearTransform.cu (was in the GPU CKKS namespace).
std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	std::vector<int> res;
	// Internal block size for DotProductPtInternal
	constexpr uint32_t INTERNAL_GSTEP = 8;

	// Intra-block rotations: stride * k for k in [1, INTERNAL_GSTEP]
	// We need rotations up to the max block size used, which is min(gStep, INTERNAL_GSTEP)
	// But to be safe and simple, we can generate up to INTERNAL_GSTEP if gStep >= INTERNAL_GSTEP
	uint32_t maxIntra = std::min(gStep, INTERNAL_GSTEP);
	for (uint32_t k = 1; k < maxIntra; ++k) {
		res.push_back(stride * k);
	}

	// Inter-block rotations: stride * INTERNAL_GSTEP * k for k in [1, blockCount - 1]
	uint32_t blockCount = (gStep + INTERNAL_GSTEP - 1) / INTERNAL_GSTEP;
	if (blockCount > 1) {
		int baseRotation = INTERNAL_GSTEP * stride;
		for (uint32_t k = 1; k < blockCount; ++k) {
			res.push_back(baseRotation * k);
		}
	}

	return res;
}

// Moved from src/CKKS/LinearTransform.cu (FIDESlib::CKKS::GetLinearTransformRotationIndices), with
// the ceil(double) giant-step count replaced by the equivalent integer form.
//
// The transform rotates the ciphertext by i*stride for i in [0, bStep) (all from ONE hoisted ModUp),
// then folds the gStep partial sums with a rotation by bStep*stride per step, then applies `offset`
// once at the end. That is exactly {i*stride : i in [1, bStep]} plus `offset` — index 0 is the
// identity and needs no key.
std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride, int offset) {
	std::vector<int> res(static_cast<size_t>(bStep) + (offset != 0));
	for (int i = 1; i <= bStep; ++i)
		res[static_cast<size_t>(i) - 1] = i * stride;
	if (offset != 0)
		res[static_cast<size_t>(bStep)] = offset;
	return res;
}

// Moved from src/CKKS/LinearTransform.cu (FIDESlib::CKKS::GetLinearTransformPlaintextRotationIndices).
//
// Diagonal k is consumed inside giant step j = k / bStep, i.e. AFTER the fold has already rotated the
// running sum by j*bStep*stride and the final `offset`. The caller therefore has to pre-rotate the
// diagonal by the inverse of that, so the encoded plaintext lines up with the ciphertext slot the
// kernel multiplies it into. res[k] is the rotation to apply to logical diagonal k before encoding.
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride, int offset) {
	std::vector<int> res(static_cast<size_t>(rowSize));
	const int gStep = (rowSize + bStep - 1) / bStep;

	for (int j = 0; j < gStep; j++) {
		for (int i = 0; i < bStep; ++i) {
			if (i + j * bStep < rowSize)
				res[static_cast<size_t>(i + j * bStep)] = -bStep * j * stride - offset;
		}
	}
	return res;
}

} // namespace fideslib
