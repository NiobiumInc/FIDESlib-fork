// PROBE (not for merge): see ProbeSpin.cuh.
#include "ProbeSpin.cuh"

#include <chrono>
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

std::chrono::steady_clock::time_point& ProbeLastScrambleTime() {
	static std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	return t;
}

void ProbeScramble(cudaStream_t stream, const std::vector<std::pair<void*, size_t>>& buffers, long long cycles) {
	if (cycles <= 0 || buffers.empty())
		return;
	ProbeLastScrambleTime() = std::chrono::steady_clock::now();
	// Every buffer saved and zeroed first, then ONE spin, then every buffer restored: a spin per
	// buffer would serialize on the stream and multiply the window by the limb count.
	std::vector<void*> saved(buffers.size(), nullptr);
	for (size_t i = 0; i < buffers.size(); ++i) {
		const auto& [data, bytes] = buffers[i];
		if (data == nullptr || bytes == 0)
			continue;
		cudaMallocAsync(&saved[i], bytes, stream);
		cudaMemcpyAsync(saved[i], data, bytes, cudaMemcpyDeviceToDevice, stream);
		cudaMemsetAsync(data, 0, bytes, stream);
	}
	probe_spin<<<1, 1, 0, stream>>>(cycles);
	for (size_t i = 0; i < buffers.size(); ++i) {
		if (saved[i] == nullptr)
			continue;
		cudaMemcpyAsync(buffers[i].first, saved[i], buffers[i].second, cudaMemcpyDeviceToDevice, stream);
		cudaFreeAsync(saved[i], stream);
	}
}
} // namespace FIDESlib
