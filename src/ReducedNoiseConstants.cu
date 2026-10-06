#include "CKKS/ReducedNoise.cuh"
#include "CudaUtils.cuh"
#include "Math.cuh"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace FIDESlib {

void fillDecompAndModUpNegQ(Constants& host_constants,
  Global& host_global,
  const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
  const std::vector<std::vector<int>>& digitGPUid,
  const std::vector<int>& GPUid) {
	for (size_t i = 0; i < digitGPUid.size(); ++i) {
		for (size_t j = 0; j < digitGPUid.at(i).size(); ++j) {
			const int d		  = digitGPUid.at(i).at(j);
			const auto& limbs = DECOMPmeta.at(i).at(j);
			for (size_t s = 1; s < limbs.size(); ++s) {
				if (limbs[s - 1].id >= limbs[s].id) {
					throw std::runtime_error("reduced-noise negQ fill: DECOMPmeta digit " + std::to_string(d) + " is not ascending by primeid");
				}
			}
			for (int t = 0; t < hC_.L + hC_.K; ++t) {
				uint64_t Qmod = 1;
				for (size_t s = 0; s < limbs.size(); ++s) {
					Qmod							 = modprod(Qmod, hC_.primes[limbs[s].id], hC_.primes[t]);
					hG_.DecompAndModUp_negQ[d][s][t] = (hC_.primes[t] - Qmod) % hC_.primes[t];
				}
			}
		}
	}

	constexpr int bytes = sizeof(Global::DecompAndModUp_negQ);
	for (uint32_t i = 0; i < GPUid.size(); ++i) {
		cudaSetDevice(GPUid[i]);
		cudaMemcpy(((char*)hG_.globals[i]) + offsetof(Global::Globals, DecompAndModUp_negQ), hG_.DecompAndModUp_negQ, bytes, cudaMemcpyHostToDevice);
		CudaCheckErrorMod;
	}
}

} // namespace FIDESlib
