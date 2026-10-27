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

long long EnvCycles(const char* name) {
	const char* env = std::getenv(name);
	return (env != nullptr && env[0] != '\0') ? std::atoll(env) * 2000LL : 0LL; // about 2000 cycles per microsecond
}
} // namespace

long long ProbeUploadDelayCycles() {
	static const long long cycles = EnvCycles("FIDESLIB_PROBE_UPLOAD_DELAY_US");
	return cycles;
}

long long ProbeKeyDelayCycles() {
	static const long long cycles = EnvCycles("FIDESLIB_PROBE_KEY_DELAY_US");
	return cycles;
}

long long ProbeKeyBDelayCycles() {
	static const long long cycles = [] {
		const char* env = std::getenv("FIDESLIB_PROBE_KEY_B_DELAY_US");
		return (env != nullptr && env[0] != '\0') ? EnvCycles("FIDESLIB_PROBE_KEY_B_DELAY_US") : ProbeKeyDelayCycles();
	}();
	return cycles;
}

bool ProbeWar() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PROBE_WAR");
		return env != nullptr && env[0] == '1';
	}();
	return enabled;
}

void ProbeSpin(cudaStream_t stream, long long cycles) {
	if (cycles > 0)
		probe_spin<<<1, 1, 0, stream>>>(cycles);
}

void ProbeScramble(cudaStream_t stream, void* data, size_t bytes, long long cycles) {
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
