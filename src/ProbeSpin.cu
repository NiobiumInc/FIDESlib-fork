// PROBE (not for merge): see ProbeSpin.cuh.
#include "ProbeSpin.cuh"

#include <cstdlib>

namespace FIDESlib {
namespace {
__global__ void probe_spin(const long long cycles) {
	const long long t0 = clock64();
	while (clock64() - t0 < cycles) {
	}
}
} // namespace

long long ProbeUploadDelayCycles() {
	static const long long cycles = [] {
		const char* env = std::getenv("FIDESLIB_PROBE_UPLOAD_DELAY_US");
		return (env != nullptr && env[0] != '\0') ? std::atoll(env) * 2000LL : 0LL; // about 2000 cycles per microsecond
	}();
	return cycles;
}

void ProbeScramble(cudaStream_t stream, void* data, size_t bytes) {
	const long long cycles = ProbeUploadDelayCycles();
	if (cycles <= 0 || data == nullptr || bytes == 0)
		return;
	void* saved = nullptr;
	cudaMallocAsync(&saved, bytes, stream);
	cudaMemcpyAsync(saved, data, bytes, cudaMemcpyDeviceToDevice, stream);
	cudaMemsetAsync(data, 0, bytes, stream);
	probe_spin<<<1, 1, 0, stream>>>(cycles);
	cudaMemcpyAsync(data, saved, bytes, cudaMemcpyDeviceToDevice, stream);
	cudaFreeAsync(saved, stream);
}
} // namespace FIDESlib
