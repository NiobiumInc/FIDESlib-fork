//
// DIAGNOSTIC (FIDESLIB_CKSUM=1, off by default): asynchronous per-lane ciphertext checksums.
//
// The staged concurrent-ops case compares each lane's OUTPUT with the serial run; when the
// bootstrap stage mismatches, that says nothing about WHICH sub-step first produced a wrong
// coefficient. ChecksumProbe() enqueues one small reduction kernel on the ciphertext's own
// partition stream (so it sees exactly the state every later kernel on that stream sees, and
// adds no host synchronisation that would de-phase the lanes) and stores a 64-bit digest of c0
// and of c1 into a per-thread device buffer. ChecksumFlush() -- called by the lane at its end,
// where it synchronises anyway before storing -- copies the digests back and prints one line
// per probe carrying the thread's label (set by the test: stage / mode / rep / lane), so the
// serial and concurrent digests can be lined up probe by probe after the run.
//
#include "CKKS/Checksum.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace FIDESlib::CKKS {

namespace {
constexpr int kMaxProbes = 4096;

struct ChecksumState {
	unsigned long long* d = nullptr; // 2 * kMaxProbes digests (c0, c1) on the device
	std::vector<std::string> tags;
	std::string label;
	bool failed = false;
};

thread_local ChecksumState g_ck;
std::mutex g_ck_print;

__global__ void cksum_kernel_(void** limbs, const int nlimbs, const int N, unsigned long long* out) {
	const long long total = static_cast<long long>(nlimbs) * N;
	unsigned long long acc = 0;
	for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
		 i += static_cast<long long>(gridDim.x) * blockDim.x) {
		const unsigned long long x = reinterpret_cast<const unsigned long long*>(limbs[i / N])[i % N];
		// Position-sensitive: a permutation of the same coefficients changes the digest.
		acc += (x ^ (x >> 31)) * (0x9e3779b97f4a7c15ULL + 2ULL * static_cast<unsigned long long>(i));
	}
	atomicAdd(out, acc);
}

// One RNSPoly's digest into out (single-GPU partition 0; multi-GPU partitions are skipped).
void probePoly(const RNSPoly& poly, unsigned long long* out) {
	const LimbPartition& p = poly.GPU.at(0);
	const int nl		   = static_cast<int>(const_cast<LimbPartition&>(p).getLimbSize(poly.getLevel())); // getLimbSize is not const
	cudaStream_t st		   = p.getS().ptr();
	cudaMemsetAsync(out, 0, sizeof(unsigned long long), st);
	if (nl <= 0)
		return;
	for (int i = 0; i < nl && i < static_cast<int>(p.limb.size()); ++i)
		if (p.limb[i].index() != FIDESlib::TYPE::U64)
			return; // 32-bit limbs: leave the digest at 0 rather than misread them
	if (nl > static_cast<int>(p.limb.size()))
		return; // level above the allocated limbs: nothing sound to read
	cksum_kernel_<<<64, 256, 0, st>>>(p.limbptr.data, nl, p.cc.N, out);
}
} // namespace

bool ChecksumEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("FIDESLIB_CKSUM");
		return v != nullptr && *v != '\0' && *v != '0';
	}();
	return on;
}

void ChecksumLabel(const std::string& label) {
	if (!ChecksumEnabled())
		return;
	g_ck.label = label;
}

namespace {
StageHook g_stage_hook;   // test-only; see Checksum.cuh
} // namespace

void SetStageHook(StageHook hook) {
	g_stage_hook = std::move(hook);
}

void ChecksumProbe(const Ciphertext& ct, const char* tag) {
	if (g_stage_hook)
		g_stage_hook(tag, ct);
	if (!ChecksumEnabled() || g_ck.failed)
		return;
	if (g_ck.d == nullptr) {
		if (cudaMalloc(&g_ck.d, sizeof(unsigned long long) * 2 * kMaxProbes) != cudaSuccess) {
			g_ck.failed = true;
			cudaGetLastError();
			return;
		}
	}
	const int idx = static_cast<int>(g_ck.tags.size());
	if (idx >= kMaxProbes)
		return;
	g_ck.tags.emplace_back(tag);
	probePoly(ct.c0, g_ck.d + 2 * idx);
	probePoly(ct.c1, g_ck.d + 2 * idx + 1);
}

void ChecksumFlush() {
	if (!ChecksumEnabled())
		return;
	if (g_ck.tags.empty())
		return;
	const int n = static_cast<int>(g_ck.tags.size());
	std::vector<unsigned long long> h(2 * static_cast<size_t>(n), 0);
	cudaDeviceSynchronize();
	if (g_ck.d != nullptr)
		cudaMemcpy(h.data(), g_ck.d, sizeof(unsigned long long) * h.size(), cudaMemcpyDeviceToHost);
	{
		std::lock_guard<std::mutex> lk(g_ck_print);
		for (int i = 0; i < n; ++i)
			std::printf("[cksum] %s probe=%d tag=%s c0=%016llx c1=%016llx\n", g_ck.label.c_str(), i, g_ck.tags[i].c_str(), h[2 * i], h[2 * i + 1]);
		std::fflush(stdout);
	}
	g_ck.tags.clear();
}

} // namespace FIDESlib::CKKS
