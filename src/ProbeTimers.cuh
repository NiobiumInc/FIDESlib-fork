// PROBE (not for merge): wall-time accumulators for the phases of building a device plaintext.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

namespace FIDESlib::ProbeTimers {

enum Id { kPtLoad, kGrow, kGenerate, kGPUmalloc, kGPUfree, kUploadLoop, kMemcpyCall, kCount };

inline std::atomic<int64_t> ns[kCount];
inline std::atomic<int64_t> calls[kCount];

inline void Reset() {
	for (int i = 0; i < kCount; ++i) {
		ns[i].store(0);
		calls[i].store(0);
	}
}

struct Scope {
	int id;
	std::chrono::steady_clock::time_point t0;
	explicit Scope(int i) : id(i), t0(std::chrono::steady_clock::now()) {}
	~Scope() {
		ns[id].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
		calls[id].fetch_add(1, std::memory_order_relaxed);
	}
};

} // namespace FIDESlib::ProbeTimers
