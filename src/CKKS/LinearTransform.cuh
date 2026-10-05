//
// Created by carlosad on 7/05/25.
//

#ifndef FIDESLIB_LINEARTRANSFORM_CUH
#define FIDESLIB_LINEARTRANSFORM_CUH

#include "forwardDefs.cuh"
#include <vector>

namespace FIDESlib::CKKS {

/// @param prepared  Optional, default none: a resident device diagonal table built over
///   exactly these diagonals at exactly this shape (see CKKS/PreparedLT.cuh). It replaces the
///   per-call host build and pageable upload of the diagonal third of the MAC kernel's pointer
///   table; everything else about the call, the numerics included, is unchanged. A table whose
///   shape does not match this call throws rather than being indexed as if it did.
void LinearTransform(Ciphertext& ctxt,
  int rowSize,
  int bStep,
  const std::vector<Plaintext*>& pts,
  int stride						= 1,
  int offset						= 0,
  const PreparedLTTable* prepared	= nullptr);

/// Several INDEPENDENT linear transforms of ONE source ciphertext, in one launch.
///
/// `out[t]` receives the transform of `ctxt` by diagonal set `pts[t]`; every set obeys exactly the
/// contract of the single-transform LinearTransform above (same rowSize / bStep / stride / offset,
/// so the same rotation keys and the same plaintext pre-rotation). `ctxt` is the shared source and
/// is rescaled in place if it arrives at noise level 2, as the single-transform call does.
///
/// What the batching buys: the hoisted key-switch precompute and its bStep baby rotations are done
/// ONCE for all out.size() transforms instead of once each, and the MAC runs as a single batched
/// kernel launch. The per-transform partial sums and the backwards Horner fold are NOT shared.
///
/// WHAT IT COSTS, in device memory. One launch needs everything live at once, so this call holds
/// `out.size() * gStep` partial-result ciphertexts (extended, during the MAC) against the single
/// transform's `gStep`, plus the caller's `out.size() * rowSize` diagonal plaintexts against
/// `rowSize`. Both scale with the batch width, so a caller near its device budget should batch
/// narrower rather than assume one call is free.
///
/// @param prepared  Optional, default none: as LinearTransform above, for the whole batch.
void LinearTransformMany(Ciphertext& ctxt,
  const std::vector<Ciphertext*>& out,
  int rowSize,
  int bStep,
  const std::vector<std::vector<Plaintext*>>& pts,
  int stride					  = 1,
  int offset					  = 0,
  const PreparedLTTable* prepared = nullptr);

template <CiphertextPtr ptrT, PlaintextPtr ptrU>
void LinearTransform(CiphertextBatch<ptrT>& ctxt, int rowSize, int bStep, const PlaintextBatch<ptrU>& pts, int stride = 1, int offset = 0);

std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride = 1, int offset = 0);
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride = 1, int offset = 0);

// ConvolutionTransform: Like LinearTransform but with INVERTED order (rotate then sum, forward loop)
void ConvolutionTransform(Ciphertext& ctxt, int rowSize, int bStep, const std::vector<Plaintext*>& pts, int stride, const std::vector<int>& indexes, uint32_t gStep);

// SpecialConvolutionTransform: Like ConvolutionTransform but with special masking logic
// After each gStep's bStep sum: 3 rotations with additions + mask multiplication before accumulation
void SpecialConvolutionTransform(Ciphertext& ctxt,
  int rowSize,
  int bStep,
  const std::vector<Plaintext*>& pts,
  Plaintext& mask,
  int stride,
  int maskRotationStride,
  const std::vector<int>& indexes,
  uint32_t gStep);

void LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt2,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3);
/*
	void LinearTransformPt(FIDESlib::CKKS::Plaintext& ptxt, FIDESlib::CKKS::Context& cc, int rowSize, int bStep,
										std::vector<Plaintext*> pts, int stride, int offset);
*/
void LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  FIDESlib::CKKS::Plaintext& ptxt,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3);

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_LINEARTRANSFORM_CUH