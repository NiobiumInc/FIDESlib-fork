//
// Created by carlosad on 4/04/24.
//

#ifndef FIDESLIB_NTT_CUH
#define FIDESLIB_NTT_CUH

#include "CKKS/forwardDefs.cuh"
#include "ConstantsGPU.cuh"
#include <cinttypes>

namespace FIDESlib {

struct FusedIterationsParams {
	struct __align__(128) AtomicCounter {
		uint32_t n = 0;
		uint32_t pad[(128 - sizeof(uint32_t)) / sizeof(uint32_t)];
	};

	AtomicCounter counters[MAXP];

	struct Conf {
		dim3 grid;
		dim3 block;
	};

	Conf first;
	Conf second;
};

/* Utility function, no real use other than testing. */
template <typename T> __global__ void Bit_Reverse(T* dat, uint32_t N);

/* Get pointer to kernel, needed for explicit Cuda Graph construction. */
void* get_NTT_reference(bool second);

// ------------------------------------- INTT ----------------------------------------
/** Kernel fusions */
enum INTT_MODE { INTT_NONE, INTT_MULT_AND_SAVE, INTT_MULT_AND_ACC, INTT_ROTATE_AND_SAVE, INTT_SQUARE_AND_SAVE };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
__global__ void INTT_(const Global::Globals* Globals,
  T* __restrict__ dat,
  const int __grid_constant__ primeid,
  T* __restrict__ res,
  const T* __restrict__ dat2	= nullptr,
  T* __restrict__ res0			= nullptr,
  T* __restrict__ res1			= nullptr,
  const T* __restrict__ kska	= nullptr,
  const T* __restrict__ kskb	= nullptr,
  T* __restrict__ c0			= nullptr,
  const T* __restrict__ c0tilde = nullptr);

template <bool second, ALGO algo, INTT_MODE mode>
__global__ void INTT_(const Global::Globals* Globals,
  void** __restrict__ dat,
  const int __grid_constant__ primeid_init,
  void** __restrict__ res,
  void** __restrict__ dat2	  = nullptr,
  void** __restrict__ res0	  = nullptr,
  void** __restrict__ res1	  = nullptr,
  void** __restrict__ kska	  = nullptr,
  void** __restrict__ kskb	  = nullptr,
  void** __restrict__ c0	  = nullptr,
  void** __restrict__ c0tilde = nullptr);

// ------------------------------------- NTT ----------------------------------------
/** Kernel fusions */
enum NTT_MODE { NTT_NONE, NTT_RESCALE, NTT_MULTPT, NTT_MODDOWN, NTT_KSK_DOT, NTT_KSK_DOT_ACC };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
__global__ void NTT_(const Global::Globals* Globals,
  T* __restrict__ dat,
  const int __grid_constant__ primeid,
  T* __restrict__ res,
  const T* __restrict__ pt					  = nullptr,
  const int __grid_constant__ primeid_rescale = -1,
  T* __restrict__ res2						  = nullptr,
  const T* __restrict__ kskb				  = nullptr);

template <bool second, ALGO algo, NTT_MODE mode>
__global__ void NTT_(const Global::Globals* Globals,
  void** __restrict__ dat,
  const int __grid_constant__ primeid_init,
  void** __restrict__ res,
  void** __restrict__ pt					  = nullptr,
  const int __grid_constant__ primeid_rescale = -1,
  void** __restrict__ res2					  = nullptr,
  void** __restrict__ kskb					  = nullptr);

/// @brief Forward NTT over a BATCH of polynomials that share one modulus chain, in ONE launch.
///
/// Same transform as the void** NTT_ above, with one extra grid dimension. blockIdx.y still selects
/// the LIMB (and therefore the prime, exactly as there: primeid_flattened[primeid_init + y], so the
/// caller passes the same PARTITION(id, 0) it passes to ApplyNTT), and blockIdx.z selects which
/// polynomial of the batch, so the pointer tables are flat and indexed `blockIdx.z * limbs +
/// blockIdx.y`. The per-block work is byte for byte the work NTT_ does -- this calls the same
/// NTT__ device function with the same arguments, and NTT__ reads only blockIdx.x/gridDim.x -- so
/// the result is bit-identical to running NTT_ once per polynomial.
///
/// NTT_NONE only: the fused modes all carry a second operand whose table would need the same
/// re-indexing, and nothing batches them. Instantiated for ALGO_SHOUP alone, which is what
/// RNSPoly::NTT uses.
///
/// @param dat    Flat table of `batch * limbs` source pointers (first stage: the limbs; second: the
///               NTT scratch).
/// @param res    Flat table of `batch * limbs` destination pointers, same indexing.
/// @param limbs  Limbs per polynomial on this partition == gridDim.y.
template <bool second, ALGO algo>
__global__ void NTTBatch_(const Global::Globals* Globals,
  void** __restrict__ dat,
  const int __grid_constant__ primeid_init,
  void** __restrict__ res,
  const int __grid_constant__ limbs);

// ------------------------------------- 1D NTT version ----------------------------------------

template <typename T, int WARP_SIZE = 32>
__global__ void
NTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N, const int __grid_constant__ primeid, const int __grid_constant__ logN);

template <typename T, int WARP_SIZE = 32>
__global__ void
INTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N, const int __grid_constant__ primeid, const T N_inv, const int __grid_constant__ logN);
} // namespace FIDESlib

#endif // FIDESLIB_NTT_CUH
