// PROBE (not for merge): wall-time accumulators for the phases of building a device plaintext.
#pragma once
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace FIDESlib::ProbeTimers {

enum Id { kPtLoad, kGrow, kGenerate, kGPUmalloc, kGPUfree, kUploadLoop, kMemcpyCall, kCount };

inline std::atomic<int64_t> ns[kCount];
inline std::atomic<int64_t> calls[kCount];

// Per-call record of the host-to-device copies of a plaintext's limbs.
inline std::mutex copy_mutex;
inline std::vector<double> copy_us;
inline std::map<void*, int> copy_streams; // stream -> calls
inline int copy_stream_flags_nonblocking = 0, copy_stream_flags_blocking = 0, copy_stream_legacy = 0;
inline std::map<int, int> copy_dst_types; // cudaMemoryType of the destination -> samples
inline int copy_samples = 0;

inline void RecordCopy(void* stream, double us) {
	std::lock_guard<std::mutex> lk(copy_mutex);
	copy_us.push_back(us);
	++copy_streams[stream];
}
inline void RecordStreamKind(bool legacy, bool nonblocking) {
	std::lock_guard<std::mutex> lk(copy_mutex);
	if (legacy)
		++copy_stream_legacy;
	else if (nonblocking)
		++copy_stream_flags_nonblocking;
	else
		++copy_stream_flags_blocking;
}
inline void RecordDstType(int type) {
	std::lock_guard<std::mutex> lk(copy_mutex);
	++copy_dst_types[type];
	++copy_samples;
}

inline std::string CopyStatsLine() {
	std::lock_guard<std::mutex> lk(copy_mutex);
	std::vector<double> v = copy_us;
	std::sort(v.begin(), v.end());
	auto pct = [&](double p) { return v.empty() ? 0.0 : v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; };
	double total = 0, slow = 0;
	for (double x : v) {
		total += x;
		if (x > 100.0)
			slow += x;
	}
	size_t nslow = std::count_if(v.begin(), v.end(), [](double x) { return x > 100.0; });
	std::ostringstream o;
	o << "copy_calls=" << v.size() << " distinct_streams=" << copy_streams.size() << " p50_us=" << pct(0.5) << " p90_us=" << pct(0.9) << " p99_us=" << pct(0.99)
	  << " max_us=" << (v.empty() ? 0.0 : v.back()) << " calls_over_100us=" << nslow << " time_in_those_ms=" << slow / 1000.0 << " total_ms=" << total / 1000.0
	  << " streams_legacy=" << copy_stream_legacy << " streams_nonblocking=" << copy_stream_flags_nonblocking << " streams_blocking=" << copy_stream_flags_blocking
	  << " dst_types(sampled)=";
	for (auto& [t, n] : copy_dst_types)
		o << t << ":" << n << ",";
	return o.str();
}

inline std::atomic<bool> pooled_plaintext_streams{ false };

inline void Reset() {
	{
		std::lock_guard<std::mutex> lk(copy_mutex);
		copy_us.clear();
		copy_streams.clear();
		copy_dst_types.clear();
		copy_stream_legacy = copy_stream_flags_nonblocking = copy_stream_flags_blocking = 0;
		copy_samples = 0;
	}
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
