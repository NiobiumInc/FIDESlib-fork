#ifndef API_HOSTMATH_HPP
#define API_HOSTMATH_HPP

#include <cstdint>
#include <functional>
#include <vector>

namespace fideslib {

// Pure-host helpers that used to live in CUDA translation units (src/PolyApprox and
// src/CKKS/LinearTransform). They contain no device code, so they are compiled in every
// build and the facade's static wrappers forward to them.

/// @brief Discrete Chebyshev approximation coefficients of `func` on [a, b].
std::vector<double> get_chebyshev_coefficients(const std::function<double(double)>& func, double a, double b, uint32_t degree);

/// @brief BSGS rotation indices required by the convolution transform.
std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);

/// @brief Ciphertext rotation indices required by the (fused BSGS) linear transform.
std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride, int offset);

/// @brief Per-diagonal plaintext pre-rotation the linear transform's BSGS layout expects,
/// indexed the same way as the diagonal vector itself.
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride, int offset);

} // namespace fideslib

#endif // API_HOSTMATH_HPP
