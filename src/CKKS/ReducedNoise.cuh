#ifndef FIDESLIB_CKKS_REDUCEDNOISE_CUH
#define FIDESLIB_CKKS_REDUCEDNOISE_CUH

#include "ConstantsGPU.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace FIDESlib::CKKS {

// OpenFHE's ReducedNoise SwitchModulus reads a residue above q/2 as v - q. Counting those lets the
// fused sum absorb the shift as one -count * Q_src term per target.
__device__ __forceinline__ uint32_t lifted_by_reduced_noise(const uint64_t v, const int primeid) {
	return v > (C_.primes[primeid] >> 1);
}

// Counts lifted residues across buff[0*blockDim.x+tid .. (n-1)*blockDim.x+tid] (tid = threadIdx.x),
// source primeid = primeid_base + i for index i: the single-residue-per-thread layout ModDown2 and
// DecompAndModUpConv share.
__device__ __forceinline__ uint32_t count_lifted(const uint64_t* buff, int n, int primeid_base) {
	uint32_t lifted = 0;
	for (int i = 0; i < n; ++i) {
		lifted += lifted_by_reduced_noise(buff[i * blockDim.x + threadIdx.x], primeid_base + i);
	}
	return lifted;
}

// Same, but primeid for index i comes from primeid_of[i] instead of an affine base: the digit
// decomposition kernels' source set.
__device__ __forceinline__ uint32_t count_lifted(const uint64_t* buff, int n, const int* primeid_of) {
	uint32_t lifted = 0;
	for (int i = 0; i < n; ++i) {
		lifted += lifted_by_reduced_noise(buff[i * blockDim.x + threadIdx.x], primeid_of[i]);
	}
	return lifted;
}

// Two independent lift counts (x/y lanes) in one pass, for the ulonglong2-paired kernels (ModDown3,
// DecompAndModUpConv_spec2); buff is the same raw uint64_t* the caller already has, reinterpreted
// here.
struct LiftedPair {
	uint32_t x;
	uint32_t y;
};

__device__ __forceinline__ LiftedPair count_lifted_pair(const uint64_t* buff, int n, int primeid_base) {
	const ulonglong2* pairs = (const ulonglong2*)buff;
	LiftedPair lifted{ 0, 0 };
	for (int i = 0; i < n; ++i) {
		const ulonglong2 data = pairs[i * blockDim.x + threadIdx.x];
		lifted.x += lifted_by_reduced_noise(data.x, primeid_base + i);
		lifted.y += lifted_by_reduced_noise(data.y, primeid_base + i);
	}
	return lifted;
}

__device__ __forceinline__ LiftedPair count_lifted_pair(const uint64_t* buff, int n, const int* primeid_of) {
	const ulonglong2* pairs = (const ulonglong2*)buff;
	LiftedPair lifted{ 0, 0 };
	for (int i = 0; i < n; ++i) {
		const ulonglong2 data = pairs[i * blockDim.x + threadIdx.x];
		lifted.x += lifted_by_reduced_noise(data.x, primeid_of[i]);
		lifted.y += lifted_by_reduced_noise(data.y, primeid_of[i]);
	}
	return lifted;
}

} // namespace FIDESlib::CKKS

namespace FIDESlib {

// -Q mod p_t for the lowest s+1 primes of digit d, the source set DecompAndModUpConv sees at that
// level. DECOMPmeta lists each digit's primes in ascending order, so a prefix of it is that set.
// Defined in ReducedNoiseConstants.cu; called once from SetupConstants.
void fillDecompAndModUpNegQ(Constants& host_constants,
  Global& host_global,
  const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
  const std::vector<std::vector<int>>& digitGPUid,
  const std::vector<int>& GPUid);

} // namespace FIDESlib

#endif // FIDESLIB_CKKS_REDUCEDNOISE_CUH
