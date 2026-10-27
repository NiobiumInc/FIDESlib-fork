// PROBE (not for merge): see ProbeSpin.cuh.
#include "ProbeSpin.cuh"

#include <cstdlib>

namespace FIDESlib {
namespace {
__global__ void probe_upload_delay_spin(const long long cycles) {
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

void ProbeUploadDelay(cudaStream_t stream) {
	if (const long long cycles = ProbeUploadDelayCycles(); cycles > 0)
		probe_upload_delay_spin<<<1, 1, 0, stream>>>(cycles);
}
} // namespace FIDESlib
