//
// Created by carlosad on 25/03/24.
//

#include "CudaUtils.cuh"
#include <atomic>
#include <deque>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unistd.h>

#include <dlfcn.h>
#include <exception>
#include <unordered_map>

#include "TableOwner.cuh"

#include "nvtx3/nvtx3.hpp"
#include <iostream>
#include <ostream>

// #include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

#include <cuda_runtime.h>

#include "TableOwnerFences.cuh" // DEBUG BRANCH ONLY: after every other header

namespace FIDESlib {

extern std::vector<cudaDeviceProp> GPUprop;

/// See CudaUtils.cuh. Default: one issuing thread per CryptoContext (the library's supported mode);
/// set to 1 to let several host threads issue ops on one context concurrently. Read once, into a
/// function-local static, so the mode cannot flip mid-run: every lock that is skipped is skipped
/// for the whole process, and every slot handed out stays valid.
bool ConcurrentOps() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_CONCURRENT_OPS");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled;
}

/// See CudaUtils.cuh (ConcurrentOpsDiag). Parsed once; the mask is 0 unless the concurrent mode
/// is on AND the variable names at least one primitive.
static unsigned ConcurrentOpsDiagMask() {
	static const unsigned mask = [] {
		const char* env = std::getenv("FIDESLIB_CONCURRENT_OPS_DIAG");
		if (!ConcurrentOps() || env == nullptr || *env == '\0')
			return 0u;
		unsigned m = 0;
		const std::string spec(env);
		size_t i = 0;
		while (i <= spec.size()) {
			const size_t j		  = spec.find(',', i);
			const std::string tok = spec.substr(i, j == std::string::npos ? std::string::npos : j - i);
			if (tok == "fence")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kFence);
			else if (tok == "modup")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kModup);
			else if (tok == "ksk")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kKsk);
			else if (tok == "moddown")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kModdown);
			else if (tok == "rescale")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kRescale);
			else if (tok == "mult")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kMult);
			else if (tok == "multpt")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kMultPt);
			else if (tok == "auxpoly")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kAuxPoly);
			else if (tok == "ntt")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kNtt);
			else if (tok == "load")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kLoad);
			else if (tok == "store")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kStore);
			else if (tok == "ctload")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kCtLoad);
			else if (tok == "boot")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kBoot);
			else if (tok == "lt")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kLt);
			else if (tok == "modraise")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kModRaise);
			else if (tok == "evalmod")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kEvalMod);
			else if (tok == "accum")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kAccum);
			else if (tok == "keyswitch")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kKeySwitch);
			else if (tok == "monomial")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kMonomial);
			else if (tok == "conj")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kConj);
			else if (tok == "freesync")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kFreeSync);
			else if (tok == "takesync")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kTakeSync);
			else if (tok == "noreuse")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kNoReuse);
			else if (tok == "privstreams")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kPrivStreams);
			else if (tok == "lanepool")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kLanePool);
			else if (tok == "churn")
				m |= static_cast<unsigned>(ConcurrentOpsDiag::kChurn);
			else if (tok == "all")
				m = ~0u & ~static_cast<unsigned>(ConcurrentOpsDiag::kFence) & ~static_cast<unsigned>(ConcurrentOpsDiag::kFreeSync) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kTakeSync) & ~static_cast<unsigned>(ConcurrentOpsDiag::kNoReuse) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kPrivStreams) & ~static_cast<unsigned>(ConcurrentOpsDiag::kLanePool) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kChurn) & ~static_cast<unsigned>(ConcurrentOpsDiag::kModupEntry) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kModupDigits) & ~static_cast<unsigned>(ConcurrentOpsDiag::kModupDs2) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kModupExit) & ~static_cast<unsigned>(ConcurrentOpsDiag::kModupCopy) &
					~static_cast<unsigned>(ConcurrentOpsDiag::kModupAlloc);
			else if (!tok.empty())
				throw std::runtime_error("FIDESLIB_CONCURRENT_OPS_DIAG: unknown primitive \"" + tok +
										 "\"; the names are fence, modup, ksk, moddown, rescale, mult, multpt, auxpoly, ntt, "
										 "load, store, ctload, boot, lt, modraise, evalmod, accum, keyswitch, monomial, conj, all (= every lock but fence), "
										 "and the diagnostics freesync, takesync, noreuse, privstreams, lanepool, churn (not locks; never part of all)");
			if (j == std::string::npos)
				break;
			i = j + 1;
		}
		return m;
	}();
	return mask;
}

bool ConcurrentOpsDiagOn(ConcurrentOpsDiag which) {
	return (ConcurrentOpsDiagMask() & static_cast<unsigned>(which)) != 0;
}

static std::recursive_mutex& concurrentOpsDiagLock() {
	static std::recursive_mutex* m = new std::recursive_mutex(); // never destroyed: outlives every static-destruction-order caller
	return *m;
}

// ---- DIAG STALL (see CudaUtils.cuh DiagStall) ---------------------------------------------------
namespace {
thread_local ConcurrentOpsDiag g_diag_kind_stack[64];
thread_local int g_diag_kind_depth = 0;

int StallMicros() {
	static const int us = [] {
		const char* env = std::getenv("FIDESLIB_STALL_US");
		return (env && *env) ? std::atoi(env) : 0;
	}();
	return us;
}
// Parses a comma list of ConcurrentOpsDiag kind names (the same names FIDESLIB_STALL_IN takes) into a
// bit mask; unset or empty = every kind.
unsigned ParseKindMaskEnv(const char* var) {
	{
		const char* env = std::getenv(var);
		if (env == nullptr || *env == '\0')
			return ~0u;
		unsigned m = 0;
		std::string spec(env);
		size_t i = 0;
		struct Name {
			const char* n;
			ConcurrentOpsDiag k;
		};
		static const Name names[] = {
			{ "fence", ConcurrentOpsDiag::kFence },		  { "modup", ConcurrentOpsDiag::kModup },	  { "ksk", ConcurrentOpsDiag::kKsk },
			{ "moddown", ConcurrentOpsDiag::kModdown },	  { "rescale", ConcurrentOpsDiag::kRescale }, { "mult", ConcurrentOpsDiag::kMult },
			{ "multpt", ConcurrentOpsDiag::kMultPt },	  { "auxpoly", ConcurrentOpsDiag::kAuxPoly }, { "ntt", ConcurrentOpsDiag::kNtt },
			{ "load", ConcurrentOpsDiag::kLoad },		  { "store", ConcurrentOpsDiag::kStore },	  { "ctload", ConcurrentOpsDiag::kCtLoad },
			{ "boot", ConcurrentOpsDiag::kBoot },		  { "lt", ConcurrentOpsDiag::kLt },			  { "modraise", ConcurrentOpsDiag::kModRaise },
			{ "evalmod", ConcurrentOpsDiag::kEvalMod },	  { "accum", ConcurrentOpsDiag::kAccum },	  { "keyswitch", ConcurrentOpsDiag::kKeySwitch },
			{ "monomial", ConcurrentOpsDiag::kMonomial }, { "conj", ConcurrentOpsDiag::kConj },
			// Sub-kinds of modup (stall bisect): each names ONE fence site inside modupMGPU / modupInto.
			{ "modup_entry", ConcurrentOpsDiag::kModupEntry },	{ "modup_digits", ConcurrentOpsDiag::kModupDigits },
			{ "modup_ds2", ConcurrentOpsDiag::kModupDs2 },		{ "modup_exit", ConcurrentOpsDiag::kModupExit },
			{ "modup_copy", ConcurrentOpsDiag::kModupCopy },		{ "modup_alloc", ConcurrentOpsDiag::kModupAlloc },
		};
		while (i <= spec.size()) {
			const size_t j		  = spec.find(',', i);
			const std::string tok = spec.substr(i, j == std::string::npos ? std::string::npos : j - i);
			if (tok == "all")
				m = ~0u;
			else
				for (const Name& nm : names)
					if (tok == nm.n)
						m |= static_cast<unsigned>(nm.k);
			if (j == std::string::npos)
				break;
			i = j + 1;
		}
		return m;
	}
	return ~0u; // not reached: the block above always returns
}
unsigned StallKindMask() {
	static const unsigned mask = ParseKindMaskEnv("FIDESLIB_STALL_IN");
	return mask;
}
// FIDESLIB_DEVSYNC_IN (diagnostic): cudaDeviceSynchronize() on ENTRY to every
// ConcurrentOpsDiagScope of the listed kinds. Unset = off (unlike the stall mask, "all" must be
// explicit). A world-stop at a scope entry says whether the kernels that scope launches race with
// work that was already in flight anywhere on the device (closed by the sync) or with work issued
// afterwards by another thread (not closed by it).
unsigned DevSyncKindMask() {
	static const unsigned mask = [] {
		const char* env = std::getenv("FIDESLIB_DEVSYNC_IN");
		if (env == nullptr || *env == '\0')
			return 0u;
		return ParseKindMaskEnv("FIDESLIB_DEVSYNC_IN");
	}();
	return mask;
}
__global__ void diag_stall_kernel_(long long cycles) {
	const long long t0 = clock64();
	while (clock64() - t0 < cycles) {
	}
}
} // namespace

void DiagStall(cudaStream_t stream) {
	const int us = StallMicros();
	if (us <= 0 || !ConcurrentOps() || stream == nullptr)
		return;
	const unsigned mask = StallKindMask();
	if (mask != ~0u) {
		bool inside = false;
		for (int d = 0; d < g_diag_kind_depth; ++d)
			if (mask & static_cast<unsigned>(g_diag_kind_stack[d]))
				inside = true;
		if (!inside)
			return;
	}
	diag_stall_kernel_<<<1, 1, 0, stream>>>(static_cast<long long>(us) * 1500ll); // ~1.5 GHz SM clock
}

ConcurrentOpsDiagScope::ConcurrentOpsDiagScope(ConcurrentOpsDiag which) : lk_(concurrentOpsDiagLock(), std::defer_lock) {
	if (ConcurrentOps() && (DevSyncKindMask() & static_cast<unsigned>(which)) != 0)
		cudaDeviceSynchronize(); // DIAG (FIDESLIB_DEVSYNC_IN): world-stop at this scope's entry
	if (ConcurrentOpsDiagOn(which))
		lk_.lock();
	if (g_diag_kind_depth < 64) {
		g_diag_kind_stack[g_diag_kind_depth++] = which;
		pushed_								   = true;
	}
}

ConcurrentOpsDiagScope::~ConcurrentOpsDiagScope() {
	if (pushed_ && g_diag_kind_depth > 0)
		--g_diag_kind_depth;
}

/// See CudaUtils.cuh. DIAGNOSTIC ONLY, and only meaningful with the concurrent mode on.
bool ConcurrentOpsSync() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_CONCURRENT_OPS_SYNC");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled && ConcurrentOps();
}

void ConcurrentOpsSyncPoint() {
	if (!ConcurrentOpsSync())
		return;
	cudaDeviceSynchronize();
}

int ScratchSlotCap() {
	static const int cap = [] {
		const char* v = std::getenv("FIDESLIB_SCRATCH_SLOTS");
		const int n	  = (v && *v) ? std::atoi(v) : 32;
		return n < 1 ? 1 : n;
	}();
	return cap;
}

bool PersistChurn() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PERSIST_CHURN");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled && ConcurrentOps();
}

bool WsumSrcSync() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_WSUM_SRCSYNC");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled && ConcurrentOps();
}

bool PersistOpTables() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PERSIST_TABLES");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled && ConcurrentOps();
}

bool BootSharedPrecomp() {
	// DEFAULT ON since 2026-10-03: the view sets take no extra device memory against several GB per
	// re-encoded set, their outputs equal the re-encoded path's to three digits, and the run is faster
	// with every output level served. FIDESLIB_BOOT_SHARED_PRECOMP=0 restores the re-encoded copies
	// (the kill switch); any other value, or unset, is the views.
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_BOOT_SHARED_PRECOMP");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
}

bool MetaBtsDevice() {
	// DEFAULT ON since 2026-10-04: measured against the host glue on the same card, the device
	// composition halves the time per refresh, keeps the error well within tolerance and leaves peak
	// memory unchanged. FIDESLIB_METABTS_DEVICE=0 restores the host glue (the kill switch, the previous
	// code path); any other value, or unset, composes on the device.
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_METABTS_DEVICE");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
}

bool DeviceIfft() {
	// DEFAULT ON since 2026-10-04: together with the encode scratch it shortens a device-encode-heavy run by about 13%, with the error well within tolerance and peak memory unchanged. FIDESLIB_DEVICE_IFFT=0 is the kill switch (the previous code path); any other value, or unset, is on.
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_DEVICE_IFFT");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
}

bool DeviceIfftCheck() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_DEVICE_IFFT_CHECK");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
}

bool EncodeScratch() {
	// DEFAULT ON since 2026-10-04: about 5% faster on its own, about 13% with the device transform; limbs bit-identical. FIDESLIB_ENCODE_SCRATCH=0 is the kill switch (the previous code path); any other value, or unset, is on.
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_ENCODE_SCRATCH");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
}

bool PlaintextArena() {
	// OPT-IN until measured (fork rule: a byte-identical default until the GPU run says otherwise).
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PT_ARENA");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled;
}

bool DeviceChecks() {
#ifndef NDEBUG
	return true;
#else
	// OPT-OUT until measured (fork rule: the shipped behaviour stays the default until the GPU run says
	// otherwise): unset or any non-zero value runs the queries; 0 skips them.
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_DEVICE_CHECKS");
		if (env == nullptr || env[0] == '\0')
			return true;
		return std::atoi(env) != 0;
	}();
	return enabled;
#endif
}

namespace {
/// One (device, slot, class) ring of persistent op-table buffers. See OpTableBuffer.
struct OpTableRing {
	static constexpr int kDepth = 16;
	void* buf[kDepth]			= {};
	size_t cap[kDepth]			= {};
	int next					= 0;
};

std::mutex& op_table_mu() {
	static std::mutex mu;
	return mu;
}

/// Never destroyed, for the same reason the slot registry is not: a thread may still be inside an
/// op during static destruction.
std::map<std::tuple<int, int, int>, OpTableRing>& op_table_rings() {
	static auto* m = new std::map<std::tuple<int, int, int>, OpTableRing>();
	return *m;
}
} // namespace

void* OpTableBuffer(int device, size_t bytes, int which) {
	// One ring per (device, issuing slot, table class), so two threads never touch one ring and a
	// thread's own consecutive ops walk kDepth distinct buffers before reusing one. The depth is
	// what makes this sound WITHOUT a fence: the buffer an op writes is not handed back to that
	// same thread until kDepth ops later, by which time its kernel has long retired. It is still a
	// DIAGNOSTIC and not a mode to ship -- nothing here proves kDepth is enough for every caller.
	std::lock_guard<std::mutex> lk(op_table_mu());
	OpTableRing& r = op_table_rings()[{ device, ScratchSlot(), which }];
	const int i	   = r.next;
	r.next		   = (r.next + 1) % OpTableRing::kDepth;
	if (r.cap[i] < bytes) {
		if (r.buf[i] != nullptr)
			cudaFree(r.buf[i]);
		// cudaMalloc, NOT cudaMallocAsync: the whole point of this instrument is to take these
		// tables OUT of the stream-ordered allocator's pool, which is the class the earlier pool
		// exclusions never covered (that was the library's own pool, not the driver's).
		const cudaError_t e = cudaMalloc(&r.buf[i], bytes);
		if (e != cudaSuccess)
			throw std::runtime_error(std::string("OpTableBuffer: cudaMalloc failed: ") + cudaGetErrorString(e));
		r.cap[i] = bytes;
	}
	return r.buf[i];
}

// ---- DEBUG BRANCH ONLY: stream-ordered allocation log for TableOwner.cuh ----
namespace {
struct MemEvent {
	unsigned long long seq;
	void* ptr;
	size_t bytes;
	cudaStream_t stream;
	void* site;
	int tid;
	char op;		 ///< 'A' allocation, 'F' free, 'U' table upload, 'K' kernel launch reading a table,
				 ///< 'R' event record (ptr = the event), 'W' stream wait on an event (ptr = the event)
	int upload_slot; ///< 'U': its entry in the upload ring, else -1
	unsigned long long enter; ///< the log position when the call started (MemLogEnter), 0 if not stamped
};
constexpr size_t kMemLogSize = size_t{ 1 } << 23; // records and waits are most of the events
MemEvent* mem_log() {
	static auto* v = new MemEvent[kMemLogSize]();
	return v;
}
std::atomic<unsigned long long> mem_seq{ 0 };
std::mutex& mem_size_mu() {
	static std::mutex mu;
	return mu;
}
std::unordered_map<void*, size_t>& mem_sizes() {
	static auto* m = new std::unordered_map<void*, size_t>();
	return *m;
}
int MemLogTid() {
	static std::atomic<int> next{ 0 };
	thread_local const int tid = next.fetch_add(1);
	return tid;
}
/// The first bytes of every table upload, so the report can name the upload that wrote a value.
struct UploadCopy {
	unsigned long long seq;
	unsigned long long words[64];
};
constexpr size_t kUploadRing = size_t{ 1 } << 16;
UploadCopy* upload_ring() {
	static auto* v = new UploadCopy[kUploadRing]();
	return v;
}
std::atomic<unsigned long long> upload_seq{ 0 };

unsigned long long MemLogPut(const char op, void* p, const size_t bytes, const cudaStream_t s, void* site, const int upload_slot = -1,
  const unsigned long long enter = 0) {
	const unsigned long long seq = mem_seq.fetch_add(1) + 1;
	MemEvent& e					 = mem_log()[seq % kMemLogSize];
	e.seq						 = 0;
	e.ptr						 = p;
	e.bytes						 = bytes;
	e.stream					 = s;
	e.site						 = site;
	e.tid						 = MemLogTid();
	e.op						 = op;
	e.upload_slot				 = upload_slot;
	e.enter						 = enter;
	e.seq						 = seq;
	return seq;
}

// ---- FREE MARKS (TableOwner.cuh MemLogFreeMark) ----
constexpr unsigned long long kMarkSlots = 1ull << 22; ///< one word per allocation, indexed by its generation
constexpr int kMarkOld					= 8;		  ///< previous owners checked per allocation
constexpr unsigned kMarkKept			= 8;		  ///< violations kept in detail
struct FreeViolation {
	unsigned long long gen;		///< the new allocation; written last, so non-zero means the entry is whole
	unsigned long long old_gen; ///< the previous owner whose free had not been reached
	unsigned long long seen;	///< what its mark slot held
};
struct FreeMarkReport {
	unsigned int count;
	unsigned int pad;
	FreeViolation v[kMarkKept];
};
struct FreeCheckArgs {
	unsigned long long old[kMarkOld];
	unsigned long long gen;
	int n;
};

__global__ void FreeMarkKernel(unsigned long long* marks, const unsigned long long gen) {
	__stcg(&marks[gen % kMarkSlots], gen);
}

__global__ void FreeCheckKernel(const unsigned long long* marks, const FreeCheckArgs a, FreeMarkReport* r) {
	for (int i = 0; i < a.n; ++i) {
		const unsigned long long m = __ldcg(&marks[a.old[i] % kMarkSlots]);
		if (m >= a.old[i])
			continue;
		const unsigned int k = atomicAdd(&r->count, 1u);
		if (k < kMarkKept) {
			atomicExch(&r->v[k].old_gen, a.old[i]);
			atomicExch(&r->v[k].seen, m);
			atomicExch(&r->v[k].gen, a.gen);
		}
	}
}

/// Spins until *flag is non-zero or `ns` have passed, whichever comes first (flag may be null).
__global__ void FreeMarkSpin(const volatile unsigned int* flag, const unsigned long long ns) {
	unsigned long long start, now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do {
		if (flag != nullptr && *flag != 0)
			return;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	} while (now - start < ns);
}

__global__ void FreeMarkSetFlag(unsigned int* flag) {
	atomicExch(flag, 1u);
	__threadfence_system();
}

/// Where each generation lived, so a violation can be matched to its log entries.
struct GenInfo {
	unsigned long long gen, a_seq, f_seq;
	cudaStream_t a_stream, f_stream;
	void* ptr;
	size_t bytes;
};
constexpr size_t kGenRing = size_t{ 1 } << 20;
struct LiveBlock {
	unsigned long long gen;
	size_t bytes;
	bool marked; ///< MemLogFreeMark ran for it: only then may a later owner check its mark
};
struct FreedBlock {
	unsigned long long gen;
	size_t bytes;
	cudaStream_t stream;
};
struct FreeMarkState {
	bool on	  = false;
	int device = -1;
	unsigned long long* marks = nullptr; ///< device memory, kMarkSlots words
	FreeMarkReport* host	  = nullptr; ///< mapped pinned
	FreeMarkReport* dev		  = nullptr;
	GenInfo* gens			  = nullptr;
	std::mutex mu; ///< live, freed, max_freed, gens
	std::unordered_map<void*, LiveBlock> live;
	std::map<unsigned long long, FreedBlock> freed; ///< by start address: freed and not handed out again yet
	size_t max_freed = 0;
	std::atomic<unsigned long long> next_gen{ 16 }; ///< 1-15 are the self-test's
	std::atomic<unsigned long long> checks{ 0 }, cross{ 0 };
	std::mutex print_mu;
	unsigned int printed = 0;
};
FreeMarkState& FM() {
	static auto* st = new FreeMarkState();
	return *st;
}
void FreeMarkPoll();
void FreeMarkInit();
bool FreeMarkReady(const cudaStream_t) {
	FreeMarkInit();
	FreeMarkState& st = FM();
	if (!st.on)
		return false;
	int dev = -1;
	return cudaGetDevice(&dev) == cudaSuccess && dev == st.device;
}
std::mutex& table_report_mu() {
	static std::mutex mu;
	return mu;
}
unsigned long long bad_word_addr = 0, bad_value = 0; ///< set by TableOwnerReport before printing
void PrintMemEvent(const MemEvent& e, const unsigned long long now, const unsigned long long x, const char* what) {
	Dl_info info{};
	const char* obj	   = "?";
	unsigned long off  = 0;
	if (e.site != nullptr && dladdr(e.site, &info) != 0 && info.dli_fname != nullptr) {
		obj = info.dli_fname;
		off = static_cast<unsigned long>(static_cast<const char*>(e.site) - static_cast<const char*>(info.dli_fbase));
	}
	const auto p = reinterpret_cast<unsigned long long>(e.ptr);
	char carried[96] = "";
	if (e.op == 'U' && e.upload_slot >= 0 && bad_word_addr >= p) {
		const UploadCopy& u			  = upload_ring()[static_cast<size_t>(e.upload_slot)];
		const unsigned long long word = (bad_word_addr - p) / 8;
		if (u.seq == e.seq && word < 64 && (bad_word_addr - p) % 8 == 0)
			std::snprintf(carried, sizeof(carried), " carried=0x%llx%s", u.words[word], u.words[word] == bad_value ? " MATCHES-BAD-VALUE" : "");
		else
			std::snprintf(carried, sizeof(carried), " carried=?");
	}
	std::printf("[tableowner] %s %c seq=%llu (%llu events ago) ptr=0x%llx bytes=%zu offset_of_bad=%lld stream=%p tid=%d site=%s+0x%lx%s\n", what, e.op,
	  e.seq, now - e.seq, p, e.bytes, static_cast<long long>(x - p), static_cast<void*>(e.stream), e.tid, obj, off, carried);
}
} // namespace

unsigned long long MemLogEnter() {
	return mem_seq.load();
}

void MemLogAlloc(void* p, const size_t bytes, const cudaStream_t s, void* site, const unsigned long long enter) {
	if (p == nullptr)
		return;
	{
		std::lock_guard<std::mutex> lk(mem_size_mu());
		mem_sizes()[p] = bytes;
	}
	const unsigned long long seq = MemLogPut('A', p, bytes, s, site, -1, enter);
	if (!FreeMarkReady(s))
		return;
	FreeMarkState& st			 = FM();
	const unsigned long long gen = st.next_gen.fetch_add(1);
	FreeCheckArgs a{};
	a.gen	   = gen;
	bool cross = false;
	{
		std::lock_guard<std::mutex> lk(st.mu);
		// Every block freed and not handed out since that overlaps this one: the allocator took them.
		const auto lo = reinterpret_cast<unsigned long long>(p), hi = lo + bytes;
		auto it		  = st.freed.lower_bound(hi);
		for (int walked = 0; it != st.freed.begin() && walked < 256; ++walked) {
			--it;
			if (it->first + it->second.bytes <= lo) {
				if (it->first + st.max_freed <= lo)
					break;
				continue;
			}
			if (a.n < kMarkOld)
				a.old[a.n++] = it->second.gen;
			cross = cross || it->second.stream != s;
			it	  = st.freed.erase(it);
		}
		st.live[p]				 = { gen, bytes, false };
		st.gens[gen % kGenRing] = { gen, seq, 0, s, nullptr, p, bytes };
	}
	if (a.n > 0) {
		FreeCheckKernel<<<1, 1, 0, s>>>(st.marks, a, st.dev);
		st.checks.fetch_add(1);
		if (cross)
			st.cross.fetch_add(1);
	}
	FreeMarkPoll();
}

void MemLogFreeMark(void* p, const cudaStream_t s) {
	if (p == nullptr || !FreeMarkReady(s))
		return;
	FreeMarkState& st = FM();
	unsigned long long gen = 0;
	{
		std::lock_guard<std::mutex> lk(st.mu);
		const auto it = st.live.find(p);
		if (it == st.live.end())
			return;
		it->second.marked = true;
		gen				  = it->second.gen;
	}
	FreeMarkKernel<<<1, 1, 0, s>>>(st.marks, gen);
}

void MemLogFree(void* p, const cudaStream_t s, void* site, const unsigned long long enter) {
	if (p == nullptr)
		return;
	size_t bytes = 0;
	{
		std::lock_guard<std::mutex> lk(mem_size_mu());
		const auto it = mem_sizes().find(p);
		if (it != mem_sizes().end()) {
			bytes = it->second;
			mem_sizes().erase(it);
		}
	}
	const unsigned long long seq = MemLogPut('F', p, bytes, s, site, -1, enter);
	FreeMarkState& st			 = FM();
	if (!st.on)
		return;
	std::lock_guard<std::mutex> lk(st.mu);
	const auto it = st.live.find(p);
	if (it == st.live.end())
		return;
	const LiveBlock b = it->second;
	st.live.erase(it);
	if (!b.marked)
		return; // no mark was launched before this free: a later check could only report it falsely
	st.freed[reinterpret_cast<unsigned long long>(p)] = { b.gen, b.bytes, s };
	st.max_freed									  = std::max(st.max_freed, b.bytes);
	GenInfo& g										  = st.gens[b.gen % kGenRing];
	if (g.gen == b.gen) {
		g.f_seq	   = seq;
		g.f_stream = s;
	}
}

void MemLogUpload(void* dst, const void* src, const size_t bytes, const cudaStream_t s, void* site, const unsigned long long enter) {
	if (dst == nullptr || src == nullptr || bytes == 0)
		return;
	const unsigned long long useq = upload_seq.fetch_add(1);
	const int slot				  = static_cast<int>(useq % kUploadRing);
	UploadCopy& u				  = upload_ring()[static_cast<size_t>(slot)];
	u.seq						  = 0;
	std::memcpy(u.words, src, bytes < sizeof(u.words) ? bytes : sizeof(u.words));
	u.seq = MemLogPut('U', dst, bytes, s, site, slot, enter);
	// The ring entry and the event carry the same seq; a reader that sees them differ skips it.
	mem_log()[u.seq % kMemLogSize].upload_slot = slot;
}

__attribute__((noinline)) void MemLogLaunch(void* table, const cudaStream_t s, void*) {
	MemLogPut('K', table, 8, s, __builtin_return_address(0));
}

void MemLogFence(const char op, cudaEvent_t e, const cudaStream_t s, void* site, const unsigned long long enter) {
	MemLogPut(op, static_cast<void*>(e), 0, s, site, -1, enter);
}

namespace {
/// The ring event with this seq, or nullptr once the ring has overwritten it.
const MemEvent* MemLogAt(const unsigned long long q) {
	const MemEvent& e = mem_log()[q % kMemLogSize];
	return e.seq == q ? &e : nullptr;
}

/// Whether the log orders the allocation `al` (stream Y) after the free `fr` (stream X), printed as one
/// line, with the records and waits on X and Y in between when it does not.
///
/// The allocator may hand a block freed on stream X to an allocation on stream Y only once Y is ordered
/// after the free. With FIDESLIB_DEBUG_NO_XSTREAM_REUSE=1 the only way it may know that is a chain of
/// event records and waits (or an op on the legacy default stream) issued after the free; with =2 it
/// may not reuse across streams at all.
///
/// The log is written next to the calls, not atomically with them: A, F, U, R and W are logged after
/// their call returns and carry the log position at which the call started (`enter`). An entry a
/// provably came before an entry b in the driver when a.seq <= b.enter: a's call had returned before
/// b's started. A chain is PROVEN when every link (the free before the record, each record before its
/// wait, the last wait before the allocation) is; otherwise two calls on it overlapped and the log
/// cannot say which of them the driver saw first.
void ReuseStep(const char* kind, const int step, const MemEvent& fr, const MemEvent& al, const unsigned long long lo) {
	const cudaStream_t X = fr.stream, Y = al.stream;
	char head[256];
	std::snprintf(head, sizeof(head), "[tableowner] REUSE %s %d: free seq=%llu tid=%d stream=%p -> allocation seq=%llu tid=%d stream=%p:", kind, step, fr.seq,
	  fr.tid, static_cast<void*>(X), al.seq, al.tid, static_cast<void*>(Y));
	if (X == Y) {
		std::printf("%s SAME STREAM\n", head);
		return;
	}
	// Streams ordered after the free: the seq at which each became so, and whether provably.
	struct Reach {
		unsigned long long seq;
		bool proven;
	};
	std::unordered_map<cudaStream_t, Reach> reach{ { X, { fr.seq, fr.enter != 0 } } };
	// Events whose latest record sits on a stream already ordered after the free.
	struct Cover {
		unsigned long long seq;
		cudaStream_t stream;
		bool proven;
	};
	std::unordered_map<cudaEvent_t, Cover> covering;
	std::vector<std::string> edges;
	unsigned long long legacy = 0;
	const auto after		  = [](const unsigned long long a_seq, const MemEvent& b) { return b.enter != 0 && a_seq <= b.enter; };
	for (unsigned long long q = fr.seq + 1; q < al.seq; ++q) {
		const MemEvent* e = MemLogAt(q);
		if (e == nullptr)
			continue;
		const auto ev = static_cast<cudaEvent_t>(e->ptr);
		if (e->stream == nullptr && legacy == 0)
			legacy = q;
		if (e->op == 'R') {
			const auto r = reach.find(e->stream);
			if (r != reach.end())
				covering[ev] = { q, e->stream, r->second.proven && after(r->second.seq, *e) };
			else
				covering.erase(ev);
		} else if (e->op == 'W') {
			const auto c = covering.find(ev);
			if (c == covering.end())
				continue;
			const bool proven = c->second.proven && after(c->second.seq, *e);
			const auto r	  = reach.find(e->stream);
			if (r == reach.end() || (!r->second.proven && proven)) {
				reach[e->stream] = { q, proven };
				char line[288];
				std::snprintf(line, sizeof(line), "[tableowner] REUSE   edge seq=%llu tid=%d stream=%p waits on event=%p recorded at seq=%llu on stream=%p%s", q,
				  e->tid, static_cast<void*>(e->stream), e->ptr, c->second.seq, static_cast<void*>(c->second.stream), proven ? "" : " (link not proven)");
				edges.emplace_back(line);
			}
		}
	}
	// An event Y waited on BEFORE it was re-recorded on a stream past the free: a dependency only if
	// the allocator read the event's state at allocation time instead of at the wait.
	std::unordered_map<cudaEvent_t, unsigned long long> waited;
	const unsigned long long back = al.seq > 400000 ? al.seq - 400000 : 1;
	for (unsigned long long q = al.seq - 1; q >= back && q >= lo && q > 0; --q) {
		const MemEvent* e = MemLogAt(q);
		if (e != nullptr && e->op == 'W' && e->stream == Y)
			waited.emplace(static_cast<cudaEvent_t>(e->ptr), q);
	}
	int stale = 0;
	for (const auto& [ev, rec] : covering) {
		const auto w = waited.find(ev);
		if (w != waited.end() && w->second < rec.seq)
			++stale;
	}
	if (reach.count(Y) != 0) {
		const bool proven = reach[Y].proven && after(reach[Y].seq, al);
		std::printf("%s ordered by %zu record/wait edges from seq=%llu, chain %s, %d stale-event candidates\n", head, edges.size(), reach[Y].seq,
		  proven ? "PROVEN by call order" : "NOT PROVEN (calls on it overlapped)", stale);
		// The edges that lead to Y, newest first: each one's recording stream must have been reached earlier.
		for (auto it = edges.rbegin(); it != edges.rend() && it - edges.rbegin() < 8; ++it)
			std::printf("%s\n", it->c_str());
		return;
	}
	if (legacy != 0) {
		std::printf("%s ordered only by an op on the legacy default stream at seq=%llu, %d stale-event candidates\n", head, legacy, stale);
		return;
	}
	std::printf("%s NOT ORDERED by any logged record, wait or legacy-stream op; %d stale-event candidates\n", head, stale);
	int shown = 0;
	for (unsigned long long q = fr.seq + 1; q < al.seq && shown < 16; ++q) {
		const MemEvent* e = MemLogAt(q);
		if (e != nullptr && (e->op == 'R' || e->op == 'W') && (e->stream == X || e->stream == Y)) {
			std::printf("[tableowner] REUSE   between: %c seq=%llu tid=%d stream=%p event=%p\n", e->op, q, e->tid, static_cast<void*>(e->stream), e->ptr);
			++shown;
		}
	}
	shown = 0;
	for (const auto& [ev, rec] : covering) {
		const auto w = waited.find(ev);
		if (w != waited.end() && w->second < rec.seq && shown++ < 6)
			std::printf("[tableowner] REUSE   stale: event=%p waited on by stream %p at seq=%llu, re-recorded at seq=%llu on stream=%p past the free\n",
			  static_cast<void*>(ev), static_cast<void*>(Y), w->second, rec.seq, static_cast<void*>(rec.stream));
	}
}

/// Follows the memory that held the bad value from the upload that wrote it through every later
/// hand-off to a new owner (free, then the next allocation covering it), and says for each whether the
/// log orders the new owner after the free. The kernel that read the value ran after some owner's
/// upload that came later in host order, so at least one of these hand-offs did not hold.
void ReuseReport(const unsigned long long x, const unsigned long long now, const unsigned long long lo) {
	const char* mode = std::getenv("FIDESLIB_DEBUG_NO_XSTREAM_REUSE");
	std::printf("[tableowner] REUSE pool setting FIDESLIB_DEBUG_NO_XSTREAM_REUSE=%s\n", mode != nullptr ? mode : "unset (all three reuse policies on)");
	// The newest upload into this memory that carried the bad value.
	const MemEvent* up = nullptr;
	for (unsigned long long q = now; q >= lo && q > 0 && up == nullptr; --q) {
		const MemEvent* e = MemLogAt(q);
		if (e == nullptr || e->op != 'U' || e->upload_slot < 0)
			continue;
		const auto p = reinterpret_cast<unsigned long long>(e->ptr);
		if (!(p <= bad_word_addr && bad_word_addr < p + e->bytes) || (bad_word_addr - p) % 8 != 0 || (bad_word_addr - p) / 8 >= 64)
			continue;
		const UploadCopy& u = upload_ring()[static_cast<size_t>(e->upload_slot)];
		if (u.seq == e->seq && u.words[(bad_word_addr - p) / 8] == bad_value)
			up = e;
	}
	if (up == nullptr) {
		std::printf("[tableowner] REUSE no logged upload carried the bad value\n");
		return;
	}
	std::printf("[tableowner] REUSE the bad value came from the upload at seq=%llu tid=%d stream=%p\n", up->seq, up->tid, static_cast<void*>(up->stream));
	const auto covers = [&](const MemEvent& e) {
		const auto p = reinterpret_cast<unsigned long long>(e.ptr);
		return p <= x && x < p + (e.bytes > 0 ? e.bytes : 1);
	};
	const MemEvent* fr = nullptr;
	int step = 0;
	for (unsigned long long q = up->seq + 1; q <= now && step < 12; ++q) {
		const MemEvent* e = MemLogAt(q);
		if (e == nullptr || !covers(*e))
			continue;
		if (fr == nullptr && e->op == 'F')
			fr = e;
		else if (fr != nullptr && e->op == 'A') {
			ReuseStep("step", ++step, *fr, *e, lo);
			fr = nullptr;
		}
	}
	if (step == 0)
		std::printf("[tableowner] REUSE %s\n", fr == nullptr ? "the uploading owner never freed the memory in the log" : "no allocation took the memory after the free");

	// The reader may instead be an earlier owner whose kernel ran late, after the uploading owner's
	// upload: the last logged launch on this table before that upload. Every hand-off between that
	// owner's free and the uploading owner's allocation had to order the upload after the launch.
	const MemEvent* k = nullptr;
	for (unsigned long long q = up->seq - 1; q >= lo && q > 0 && k == nullptr; --q) {
		const MemEvent* e = MemLogAt(q);
		if (e != nullptr && e->op == 'K' && reinterpret_cast<unsigned long long>(e->ptr) == x)
			k = e;
	}
	if (k == nullptr) {
		std::printf("[tableowner] REUSE late-reader: no logged launch on this table before that upload\n");
		return;
	}
	PrintMemEvent(*k, now, x, "REUSE late-reader candidate");
	fr	 = nullptr;
	step = 0;
	for (unsigned long long q = k->seq + 1; q < up->seq && step < 12; ++q) {
		const MemEvent* e = MemLogAt(q);
		if (e == nullptr || !covers(*e))
			continue;
		if (fr == nullptr && e->op == 'F')
			fr = e;
		else if (fr != nullptr && e->op == 'A') {
			ReuseStep("late-reader step", ++step, *fr, *e, lo);
			fr = nullptr;
		}
	}
	if (step == 0)
		std::printf("[tableowner] REUSE late-reader: no hand-off between that launch and the upload\n");
}

/// Prints the violations the device recorded and the host has not printed yet, with the records and
/// waits between the previous owner's free and the new allocation.
void FreeMarkPoll() {
	FreeMarkState& st = FM();
	if (!st.on)
		return;
	const unsigned int count = *static_cast<volatile unsigned int*>(&st.host->count);
	if (count <= st.printed || st.printed >= kMarkKept)
		return;
	std::lock_guard<std::mutex> lk(st.print_mu);
	const unsigned long long now = mem_seq.load();
	const unsigned long long lo	 = now > kMemLogSize ? now - kMemLogSize + 1 : 1;
	while (st.printed < count && st.printed < kMarkKept) {
		const volatile FreeViolation& v = st.host->v[st.printed];
		const unsigned long long gen	= v.gen;
		if (gen == 0)
			break; // the device is still writing this entry
		const unsigned long long old = v.old_gen, seen = v.seen;
		++st.printed;
		GenInfo o{}, n{};
		{
			std::lock_guard<std::mutex> g(st.mu);
			o = st.gens[old % kGenRing];
			n = st.gens[gen % kGenRing];
		}
		std::printf("[freemark] VIOLATION %u: an allocation ran on the device before a previous owner of its memory reached its free. New: gen %llu A seq=%llu "
					"stream=%p %zu bytes at %p. Previous: gen %llu A seq=%llu F seq=%llu stream=%p %zu bytes at %p. Mark slot held %llu.\n",
		  st.printed, gen, n.gen == gen ? n.a_seq : 0ull, static_cast<void*>(n.a_stream), n.bytes, n.ptr, old, o.gen == old ? o.a_seq : 0ull,
		  o.gen == old ? o.f_seq : 0ull, static_cast<void*>(o.f_stream), o.bytes, o.ptr, seen);
		const MemEvent* fr = (o.gen == old && o.f_seq != 0) ? MemLogAt(o.f_seq) : nullptr;
		const MemEvent* al = n.gen == gen ? MemLogAt(n.a_seq) : nullptr;
		if (fr != nullptr && al != nullptr && fr->seq >= lo) {
			PrintMemEvent(*fr, now, reinterpret_cast<unsigned long long>(fr->ptr), "FREEMARK-FREE");
			PrintMemEvent(*al, now, reinterpret_cast<unsigned long long>(al->ptr), "FREEMARK-ALLOC");
			ReuseStep("step", static_cast<int>(st.printed), *fr, *al, lo);
		} else {
			std::printf("[freemark]   the free or the allocation is no longer in the log\n");
		}
	}
	std::fflush(stdout);
}

/// Once per process, at the first logged allocation: the marks, the report, and a self-test that must
/// see a violation where nothing orders the check after the mark, and none where an event wait does.
void FreeMarkInit() {
	static std::once_flag once;
	std::call_once(once, [] {
		FreeMarkState& st = FM();
		const char* env	  = std::getenv("FIDESLIB_DEBUG_FREEMARK");
		if (env != nullptr && env[0] == '0') {
			std::printf("[freemark] OFF (FIDESLIB_DEBUG_FREEMARK=0)\n");
			std::fflush(stdout);
			return;
		}
		void* h = nullptr;
		void* d = nullptr;
		if (cudaGetDevice(&st.device) != cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&st.marks), kMarkSlots * 8) != cudaSuccess ||
			cudaMemset(st.marks, 0, kMarkSlots * 8) != cudaSuccess || cudaHostAlloc(&h, sizeof(FreeMarkReport), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
			cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
			std::printf("[freemark] could not set up: %s\n", cudaGetErrorString(cudaGetLastError()));
			std::fflush(stdout);
			return;
		}
		std::memset(h, 0, sizeof(FreeMarkReport));
		st.host = static_cast<FreeMarkReport*>(h);
		st.dev	= static_cast<FreeMarkReport*>(d);
		st.gens = new GenInfo[kGenRing]();

		// Self-test on two private non-blocking streams (no other thread's legacy-stream op can order
		// them). Check 2 runs first, then sets a flag that the spin ahead of mark 1 waits for: nothing
		// orders the check after the mark, and the check must report it. Mark 3 is followed by an
		// event the checking stream waits on: check 4 must not.
		cudaStream_t a = nullptr, b = nullptr;
		cudaEvent_t e  = nullptr;
		unsigned int* flag = nullptr;
		cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
		cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
		cudaMalloc(reinterpret_cast<void**>(&flag), sizeof(unsigned int));
		cudaMemset(flag, 0, sizeof(unsigned int));
		// Lazy loading (the default since CUDA 12.2) loads a kernel at its first launch, and the load
		// may synchronize the context: a first launch of the check would wait for the mark before it.
		// Launch every kernel once, with no effect, before the test.
		FreeMarkSpin<<<1, 1, 0, a>>>(nullptr, 0);
		FreeMarkKernel<<<1, 1, 0, a>>>(st.marks, 0);
		FreeCheckKernel<<<1, 1, 0, a>>>(st.marks, FreeCheckArgs{}, st.dev);
		FreeMarkSetFlag<<<1, 1, 0, a>>>(flag);
		cudaMemsetAsync(flag, 0, sizeof(unsigned int), a);
		const cudaError_t warm = cudaStreamSynchronize(a);
		FreeCheckArgs c{};
		c.n = 1;
		FreeMarkSpin<<<1, 1, 0, a>>>(flag, 2'000'000'000);
		FreeMarkKernel<<<1, 1, 0, a>>>(st.marks, 1);
		c.old[0] = 1;
		c.gen	 = 2;
		FreeCheckKernel<<<1, 1, 0, b>>>(st.marks, c, st.dev);
		FreeMarkSetFlag<<<1, 1, 0, b>>>(flag);
		FreeMarkSpin<<<1, 1, 0, a>>>(nullptr, 20'000'000);
		FreeMarkKernel<<<1, 1, 0, a>>>(st.marks, 3);
		cudaEventRecord(e, a);
		cudaStreamWaitEvent(b, e, 0);
		c.old[0] = 3;
		c.gen	 = 4;
		FreeCheckKernel<<<1, 1, 0, b>>>(st.marks, c, st.dev);
		const cudaError_t r = warm != cudaSuccess ? warm : cudaDeviceSynchronize();
		const bool ok		= r == cudaSuccess && st.host->count == 1 && st.host->v[0].gen == 2 && st.host->v[0].old_gen == 1;
		std::printf("[freemark] self-test %s: unordered check reported %u violation(s), first gen %llu old %llu (want 1, 2, 1)%s%s\n", ok ? "PASSED" : "FAILED",
		  st.host->count, st.host->v[0].gen, st.host->v[0].old_gen, r == cudaSuccess ? "" : " ", r == cudaSuccess ? "" : cudaGetErrorString(r));
		cudaFree(flag);
		cudaEventDestroy(e);
		cudaStreamDestroy(a);
		cudaStreamDestroy(b);
		std::memset(h, 0, sizeof(FreeMarkReport));
		if (!ok) {
			std::fflush(stdout);
			return;
		}
		st.on = true;
		std::printf("[freemark] ON: every logged free marks itself on its stream; every logged allocation checks, on its own stream, the marks of the "
					"blocks it took over\n");
		std::fflush(stdout);
		std::atexit(FreeMarkSummary);
	});
}
} // namespace

void FreeMarkSummary() {
	FreeMarkState& st = FM();
	if (!st.on)
		return;
	FreeMarkPoll();
	std::printf("[freemark] checks=%llu (cross-stream %llu) violations=%u\n", st.checks.load(), st.cross.load(),
	  *static_cast<volatile unsigned int*>(&st.host->count));
	std::fflush(stdout);
}

void TableOwnerReport() {
	std::lock_guard<std::mutex> lk(table_report_mu());
	static bool done = false;
	if (done)
		return;
	done				   = true;
	const TableReport* r   = TableReportHost();
	if (r == nullptr || r->hit == 0) {
		std::printf("[tableowner] no bad table entry was recorded\n");
		FreeMarkSummary();
		std::fflush(stdout);
		return;
	}
	static const char* const names[] = { "?", "eval_linear_w_sum_", "fusedDotKSK_2_", "dotProductLtBatchedPt2___" };
	const int k						 = (r->kernel >= 1 && r->kernel <= 3) ? r->kernel : 0;
	std::printf("[tableowner] BAD ENTRY kernel=%s level=%d index=%d table=0x%llx value=0x%llx array=0x%llx block=(%d,%d,%d)\n", names[k], r->level,
	  r->index, r->table, r->value, r->array, r->bx, r->by, r->bz);
	for (int i = 0; i < r->nwords; i += 4) {
		std::printf("[tableowner] words[%2d..]:", i);
		for (int j = i; j < i + 4 && j < r->nwords; ++j)
			std::printf(" 0x%016llx", r->words[j]);
		std::printf("\n");
	}
	// The memory that held the bad value: the table itself (level 1) or the array it points to.
	const unsigned long long x	 = r->level == 1 ? r->table : r->array;
	bad_word_addr				 = x + static_cast<unsigned long long>(r->index) * 8;
	bad_value					 = r->value;
	const unsigned long long now = mem_seq.load();
	const unsigned long long lo	 = now > kMemLogSize ? now - kMemLogSize + 1 : 1;
	int covering = 0, near = 0;
	for (unsigned long long q = now; q >= lo && q > 0; --q) {
		const MemEvent& e = mem_log()[q % kMemLogSize];
		if (e.seq != q)
			continue;
		const auto p	 = reinterpret_cast<unsigned long long>(e.ptr);
		const size_t len = e.bytes > 0 ? e.bytes : 1;
		if (p <= x && x < p + len) {
			if (covering < 30)
				PrintMemEvent(e, now, x, "COVERS");
			++covering;
		} else if (near < 4 && ((p + len <= x && x - (p + len) < 65536) || (p > x && p - x < 65536))) {
			PrintMemEvent(e, now, x, "NEAR");
			++near;
		}
	}
	std::printf("[tableowner] %d logged events cover 0x%llx (%llu events in the log)\n", covering, x, now - lo + 1);
	ReuseReport(x, now, lo);
	FreeMarkSummary();
	std::fflush(stdout);
}

// ---- Per-slot device memory pools (DIAGNOSTIC / candidate fix, FIDESLIB_SLOT_MEMPOOL=1) ----
//
// See CudaUtils.cuh SlotMemPool() for what this discriminates. The invariant it buys is one line:
// a device block allocated by one lane is never handed to another lane, because the two lanes draw
// from different cudaMemPool_t objects. Slot 0 is left on the device default pool, so with the flag
// unset (or outside the concurrent mode) nothing below is ever reached and OpMallocAsync is the
// plain cudaMallocAsync it replaced.

namespace {
std::atomic<unsigned long long> slot_pool_allocs{ 0 };	  ///< Tables that came from a slot pool.
std::atomic<unsigned long long> slot_pool_fallbacks{ 0 }; ///< Tables that fell back to the default pool.

/// One cudaMemPool_t per (device, slot), flat: device * ScratchSlotCap() + slot.
///
/// NO LOCK, and none is needed: the vector is built once (function-local static initialization is
/// thread-safe) and never resized, and an entry for slot N is written and read only by the ONE
/// thread that owns slot N -- ScratchSlot() hands each thread its own, opScratch(N) runs on that
/// thread and OpMallocAsync runs on that thread. Distinct elements of a vector are distinct memory
/// locations, so two lanes creating their pools at the same time is not a race. The only all-slots
/// writer is DestroySlotMemPools, which carries destroyOpScratch's teardown precondition.
///
/// Never destroyed as a container, for the same reason the slot registry and the op-table rings are
/// not: a thread may still be inside an op during static destruction.
std::vector<cudaMemPool_t>& slot_mem_pools() {
	static auto* v = new std::vector<cudaMemPool_t>(static_cast<size_t>(MAXG) * static_cast<size_t>(ScratchSlotCap()), nullptr);
	return *v;
}

size_t slot_mem_pool_index(const int device, const int slot) {
	return static_cast<size_t>(device) * static_cast<size_t>(ScratchSlotCap()) + static_cast<size_t>(slot);
}

void SlotMemPoolReport() {
	const unsigned long long from_pool = slot_pool_allocs.load(std::memory_order_relaxed);
	const unsigned long long fallback  = slot_pool_fallbacks.load(std::memory_order_relaxed);
	// The fallback count is the one number that makes a NULL result readable: a run where every
	// table still came from the device default pool has not tested anything.
	printf("[slotmempool] per-op tables: %llu from a slot pool, %llu from the device default pool\n", from_pool, fallback);
	fflush(stdout);
}
} // namespace

bool SlotMemPool() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_SLOT_MEMPOOL");
		const bool on  = env != nullptr && env[0] != '\0' && std::atoi(env) != 0 && ConcurrentOps();
		if (on) {
			printf("[slotmempool] ON: per-op device tables come from a cudaMemPool_t private to the issuing slot (slot 0 keeps the device default pool)\n");
			fflush(stdout);
			std::atexit(SlotMemPoolReport);
		}
		return on;
	}();
	return enabled;
}

cudaMemPool_t SlotMemPoolFor(const int device, const int slot) {
	if (!SlotMemPool())
		return nullptr;
	// Slot 0 IS the device default pool: leaving it alone is what makes the default mode
	// byte-identical, and in the concurrent mode it keeps one lane on the path the control measures.
	if (slot <= 0 || slot >= ScratchSlotCap() || device < 0 || device >= MAXG)
		return nullptr;

	std::vector<cudaMemPool_t>& pools = slot_mem_pools();
	const size_t i					  = slot_mem_pool_index(device, slot);
	if (pools[i] != nullptr)
		return pools[i];

	// Created on first ask rather than for every slot up front: a slot that never issues an op
	// costs one null pointer. ContextData::opScratch calls this eagerly for each of its devices, so
	// the creation normally happens when the slot's scratch is built and not inside a hot op; the
	// lazy path here covers a lane whose first op allocates a table before it touches any scratch.
	cudaMemPoolProps props{};
	props.allocType		= cudaMemAllocationTypePinned;
	props.handleTypes	= cudaMemHandleTypeNone;
	props.location.type = cudaMemLocationTypeDevice;
	props.location.id	= device;

	cudaMemPool_t pool	= nullptr;
	const cudaError_t e = cudaMemPoolCreate(&pool, &props);
	if (e != cudaSuccess)
		throw std::runtime_error(std::string("SlotMemPoolFor: cudaMemPoolCreate failed: ") + cudaGetErrorString(e));
	// The same release threshold the context sets on the device default pool (Context.cu), so the
	// two pools differ in WHO can draw from them and in nothing else -- in particular a slot pool
	// keeps its reserve across ops instead of returning it to the driver at every free.
	uint64_t threshold = UINT64_MAX;
	cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

	pools[i] = pool;
	return pool;
}

void DestroySlotMemPools() {
	if (!SlotMemPool())
		return;
	for (cudaMemPool_t& p : slot_mem_pools()) {
		if (p == nullptr)
			continue;
		// cudaMemPoolDestroy returns immediately if allocations are still outstanding and releases
		// the pool once the last one is freed, so this is safe even against the one table this fork
		// knowingly leaks (modup_ksk_moddown_mgpu's digits).
		cudaMemPoolDestroy(p);
		p = nullptr;
	}
}

void* OpMallocAsync(const size_t bytes, const cudaStream_t s) {
	static std::once_flag table_check_once;
	std::call_once(table_check_once, [] {
		TableCheckInit();
		std::set_terminate([] {
			TableOwnerReport();
			if (const std::exception_ptr e = std::current_exception()) {
				try {
					std::rethrow_exception(e);
				} catch (const std::exception& x) {
					std::fprintf(stderr, "terminate called after throwing: %s\n", x.what());
				} catch (...) {
				}
			}
			std::abort();
		});
	});
	if (SlotMemPool()) {
		const int slot = ScratchSlot();
		if (slot > 0) {
			// The op path always runs under cudaSetDevice(partition.device), so the current device
			// is the stream's device and therefore the device whose pool this slot must draw from.
			int device = 0;
			if (cudaGetDevice(&device) == cudaSuccess) {
				const cudaMemPool_t pool = SlotMemPoolFor(device, slot);
				if (pool != nullptr) {
					void* p = nullptr;
					const unsigned long long enter = MemLogEnter();
					const cudaError_t e = cudaMallocFromPoolAsync(&p, bytes, pool, s);
					if (e != cudaSuccess)
						throw std::runtime_error(std::string("OpMallocAsync: cudaMallocFromPoolAsync failed: ") + cudaGetErrorString(e));
					MemLogAlloc(p, bytes, s, __builtin_return_address(0), enter);
					slot_pool_allocs.fetch_add(1, std::memory_order_relaxed);
					return p;
				}
			}
		}
		slot_pool_fallbacks.fetch_add(1, std::memory_order_relaxed);
	}
	// Default mode and slot 0: byte-identical to the cudaMallocAsync this replaced, down to not
	// checking the status (the call sites' CudaCheckError macros do that where they always did).
	void* p = nullptr;
	const unsigned long long enter = MemLogEnter();
	cudaMallocAsync(&p, bytes, s);
	MemLogAlloc(p, bytes, s, __builtin_return_address(0), enter);
	return p;
}

void OpFreeAsync(void* p, const cudaStream_t s) {
	// Unconditional by design -- see CudaUtils.cuh OpFreeAsync.
	const unsigned long long enter = MemLogEnter();
	MemLogFreeMark(p, s);
	cudaFreeAsync(p, s);
	MemLogFree(p, s, __builtin_return_address(0), enter);
}

void TableTraceReport();

bool TableTrace() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_TABLE_TRACE");
		const bool on  = env != nullptr && env[0] != '\0' && std::atoi(env) != 0 && ConcurrentOps();
		if (on) {
			// The report MUST be unmissable: an earlier run of this instrument over
			// 16 reps printed nothing, because the reporter had no caller. A banner at
			// enable time says the trace is ON (so silence later means zero events, not a dead
			// instrument), and atexit prints the counters at a normal exit. A CUDA failure that
			// nothing catches ends in std::terminate, which skips atexit: the live heartbeat lines
			// are what is left in that case.
			printf("[tabletrace] ON: per-op device tables traced (alloc/free/hot-reuse); summary prints at exit\n");
			fflush(stdout);
			std::atexit(TableTraceReport);
		}
		return on;
	}();
	return enabled;
}

namespace {
/// What the driver's stream-ordered allocator did with one per-op table address.
struct TableOwner {
	int slot = -1;
	cudaEvent_t freed{};   ///< recorded on the freeing stream AFTER cudaFreeAsync
};

std::mutex& table_trace_mu() {
	static std::mutex mu;
	return mu;
}
std::map<void*, TableOwner>& table_owners() {
	static auto* m = new std::map<void*, TableOwner>();
	return *m;
}
std::atomic<unsigned long long> tt_allocs{ 0 }, tt_reuse{ 0 }, tt_hot_same{ 0 }, tt_hot_cross{ 0 };
std::atomic<int> tt_printed{ 0 };
} // namespace

void TableTraceAlloc(void* p, const char* site) {
	if (p == nullptr)
		return;
	const unsigned long long n = tt_allocs.fetch_add(1) + 1;
	if ((n & 0xFFFu) == 0) {   // every 4096 tables: a heartbeat with the live counters
		printf("[tabletrace] allocs=%llu reuse=%llu hot_same_slot=%llu HOT_CROSS_LANE=%llu (live)\n",
		  n, tt_reuse.load(), tt_hot_same.load(), tt_hot_cross.load());
		fflush(stdout);
	}
	const int me = ScratchSlot();
	std::lock_guard<std::mutex> lk(table_trace_mu());
	auto it = table_owners().find(p);
	if (it == table_owners().end())
		return;
	// This address was handed back out. The question the instrument exists to answer is
	// whether the PREVIOUS owner's free has actually been REACHED on its stream yet: if the
	// event is still pending, the driver has recycled a block whose releasing stream has not
	// got to the release, and any kernel of that owner still reading the table is reading a
	// block this caller is about to overwrite.
	tt_reuse.fetch_add(1);
	const bool hot = (cudaEventQuery(it->second.freed) != cudaSuccess);
	if (hot) {
		if (it->second.slot == me) {
			tt_hot_same.fetch_add(1);
		} else {
			tt_hot_cross.fetch_add(1);
			if (tt_printed.fetch_add(1) < 64)
				printf("[tabletrace] HOT CROSS-LANE REUSE  ptr=%p  freed-by-slot=%d  taken-by-slot=%d  site=%s\n",
				  p, it->second.slot, me, site ? site : "?");
		}
	}
	cudaEventDestroy(it->second.freed);
	table_owners().erase(it);
}

void TableTraceFree(void* p, cudaStream_t s) {
	if (p == nullptr)
		return;
	cudaEvent_t ev{};
	if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
		return;
	cudaEventRecord(ev, s);
	std::lock_guard<std::mutex> lk(table_trace_mu());
	auto& o = table_owners()[p];
	if (o.freed != cudaEvent_t{})
		cudaEventDestroy(o.freed);
	o.slot	= ScratchSlot();
	o.freed = ev;
}

void TableTraceReport() {
	if (!TableTrace())
		return;
	printf("[tabletrace] allocs=%llu reuse=%llu hot_same_slot=%llu HOT_CROSS_LANE=%llu (final)\n",
	  tt_allocs.load(), tt_reuse.load(), tt_hot_same.load(), tt_hot_cross.load());
	fflush(stdout);
}

bool SlotGraphCache() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_SLOT_GRAPH_CACHE");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled && ConcurrentOps();
}


namespace {
/// Slot bookkeeping. A slot is bound to a thread for as long as that thread lives and is RETURNED
/// when it exits, so the cap bounds the number of threads issuing ops AT ONCE, not the number of
/// threads the application has ever created. That distinction is the whole design: an application
/// that fans out with a fresh std::thread per batch of work creates unboundedly many threads over
/// a run while never having more than a handful alive, and a slot that is never given back would
/// exhaust any cap within seconds of such a run.
///
/// Reusing a slot after its thread exits is safe for the same reason reusing the context's own
/// scratch between two consecutive ops is: the scratch polynomials carry their OWN streams, so the
/// next user's kernels are ordered behind the previous user's on those streams. No host sync is
/// needed and none is done.
struct SlotRegistry {
	std::mutex mu;
	std::vector<int> freed; // returned slots, reused LIFO
	int next = 0;
};

/// Never destroyed: a thread can exit during static destruction, and its slot release must not
/// touch a mutex that has already been torn down.
SlotRegistry& slotRegistry() {
	static SlotRegistry* reg = new SlotRegistry();
	return *reg;
}

/// Holds this thread's slot and gives it back at thread exit.
struct SlotHolder {
	int slot = -1;
	~SlotHolder() {
		if (slot < 0)
			return;
		SlotRegistry& reg = slotRegistry();
		std::lock_guard<std::mutex> lk(reg.mu);
		reg.freed.push_back(slot);
	}
};
} // namespace

int ScratchSlot() {
	// Slots are handed out in order of first REQUEST, not thread creation: a thread that never
	// issues an op that needs scratch (an encode-only worker, for instance) never takes one and
	// never pays for one -- the plaintext encode and load paths request none (Plaintext::load ->
	// RNSPoly::loadConstant, Plaintext::loadCoefficients(Batch) -> RNSPoly::loadCoefficients, none
	// of which reaches a scratch accessor). They are handed BACK at thread exit, so the cap is a
	// bound on concurrent issuers, not on threads ever created. The registry is process-wide
	// rather than per context because a slot index is just a name for "this thread's set" -- a
	// context materializes only the slots it is asked for.
	// Default mode: ONE slot, which is the context's own scratch members -- the code path this
	// library has always taken.
	if (!ConcurrentOps())
		return 0;
	thread_local SlotHolder holder;
	if (holder.slot < 0) {
		const int cap	  = ScratchSlotCap();
		SlotRegistry& reg = slotRegistry();
		std::lock_guard<std::mutex> lk(reg.mu);
		if (!reg.freed.empty()) {
			// An exited thread's set, reused rather than a new allocation.
			holder.slot = reg.freed.back();
			reg.freed.pop_back();
		} else if (reg.next < cap) {
			holder.slot = reg.next++;
		} else {
			// THROW, do not fall back. The concurrent mode promises every issuing thread its OWN
			// scratch; handing the overflow thread slot 0 would hand it the scratch another thread
			// is already using, which is the exact corruption the mode exists to remove -- and it
			// would do so silently, in the middle of a run. This counts threads issuing ops AT
			// ONCE (slots come back when a thread exits), so hitting it means the application
			// really does have more concurrent issuers than the cap allows.
			throw std::runtime_error("FIDESlib: " + std::to_string(cap + 1) +
									 " threads are issuing ops on one context at once but "
									 "FIDESLIB_SCRATCH_SLOTS is " +
									 std::to_string(cap) +
									 "; raise FIDESLIB_SCRATCH_SLOTS to at least the number of threads that issue "
									 "ops concurrently, or issue from fewer threads.");
		}
	}
	return holder.slot;
}

namespace {
/// Lane bookkeeping for the device-memory pool's handshake streams. Same shape as SlotRegistry
/// above, and DELIBERATELY SEPARATE from it: a thread that only ALLOCATES device memory (an
/// encode/upload worker building device plaintexts, say) must not consume an op-scratch slot,
/// because a slot is a promise of ~150 MiB of scratch polynomials and the application sizes
/// FIDESLIB_SCRATCH_SLOTS for the threads that ISSUE OPS.
///
/// The other difference is the overflow rule. Running out of op scratch throws, because sharing
/// scratch would corrupt intermediates. Running out of POOL LANES cannot corrupt anything: a lane
/// only names which handshake stream a freed block is fenced into, so an overflowing thread falls
/// back to lane 0 and simply gets the conservative, shared-stream fence the default mode always
/// used. It is slower, never wrong, so it is a silent fallback and not an error.
struct LaneRegistry {
	std::mutex mu;
	std::vector<int> freed;
	int next = 1; // lane 0 is the pool's own stream and belongs to nobody in particular
};

LaneRegistry& laneRegistry() {
	static LaneRegistry* reg = new LaneRegistry();
	return *reg;
}

struct LaneHolder {
	int lane = -1;
	~LaneHolder() {
		if (lane <= 0)
			return;
		LaneRegistry& reg = laneRegistry();
		std::lock_guard<std::mutex> lk(reg.mu);
		reg.freed.push_back(lane);
	}
};
} // namespace

/// This thread's device-memory-pool lane. Always 0 in the default mode.
int MemPoolLane() {
	if (!ConcurrentOps())
		return 0;
	thread_local LaneHolder holder;
	if (holder.lane < 0) {
		// Sized off the same knob as the op scratch, so there is one number to raise; a lane
		// itself costs one CUDA stream handle and one event, not device memory.
		const int cap	  = ScratchSlotCap();
		LaneRegistry& reg = laneRegistry();
		std::lock_guard<std::mutex> lk(reg.mu);
		if (!reg.freed.empty()) {
			holder.lane = reg.freed.back();
			reg.freed.pop_back();
		} else if (reg.next < cap) {
			holder.lane = reg.next++;
		} else {
			holder.lane = 0; // conservative shared fence; see LaneRegistry.
		}
	}
	return holder.lane;
}

struct my_domain {
	static constexpr char const* name{ "FIDESlib" };
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

std::map<std::string, std::pair<std::unique_ptr<nvtx3::unique_range_in<my_domain>>, int>> lifetimes_map;

/// Guards `lifetimes_map`. The LIFETIME category keeps a refcount per message in a std::map, and
/// the objects that push/pop it are ordinary value types -- CKKS::Plaintext and CKKS::Ciphertext
/// each hold a `CudaNvtxRange my_range` member (Plaintext.cuh:24), so *constructing or destroying
/// one* mutates this map. An application that builds device plaintexts on worker threads (encode /
/// upload pipelining) therefore inserts into the same std::map from several threads at once, which
/// is a tree-corrupting data race, not a lost counter. The FUNCTION category needs no lock: NVTX
/// push/pop is per-thread by construction.
std::mutex lifetimes_lock;

/// @brief Whether the LIFETIME NVTX bookkeeping runs: `FIDESLIB_NVTX_LIFETIME=1`, default off.
///
/// `lifetimes_map` exists only so an NVTX profiler timeline shows a "N x Plaintext" range per live
/// message; no code in the library ever reads it. Paying for it costs `lifetimes_lock` plus a map
/// lookup and two std::strings on *every* construction and destruction of a Plaintext, Ciphertext,
/// KeySwitchingKey or Context -- each holds a `CudaNvtxRange my_range` of this category. With 24
/// prefetch worker threads building ~306k device plaintexts per pass alongside the consumer
/// thread's own ciphertexts, all 25 threads serialize on that single mutex. So it is opt-in.
///
/// The environment is read once, on first use, into a function-local static: the flag cannot flip
/// mid-run, so a Start that was skipped is always matched by a skipped End and the refcounts stay
/// balanced; the cost when off is one perfectly-predicted branch.
bool NvtxLifetimeEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_NVTX_LIFETIME");
		return env != nullptr && env[0] == '1' && env[1] == '\0';
	}();
	return enabled;
}

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {

	if (cat == FUNCTION) {
		using namespace nvtx3;
		int size = msg.size();
		const event_attributes attr{ msg,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ val },
		                             category{ static_cast<unsigned int>(cat) } };

		nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	} else if (cat == LIFETIME) {
		if (!NvtxLifetimeEnabled())
			return;

		using namespace nvtx3;
		std::lock_guard<std::mutex> guard(lifetimes_lock);
		int size      = msg.size();
		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i + 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i + 1 },
		                             category{ static_cast<unsigned int>(cat) } };
		i = i + 1;
		if (!r) {
			r = std::make_unique<unique_range_in<my_domain>>(attr);
		} else {
			*r = unique_range_in<my_domain>(attr);
		}
	}
	// nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
	if (cat == FUNCTION) {
		nvtxDomainRangePop(D);
	} else if (cat == LIFETIME) {
		if (!NvtxLifetimeEnabled())
			return;

		using namespace nvtx3;
		std::lock_guard<std::mutex> guard(lifetimes_lock);
		int size = msg.size();

		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i - 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i - 1 },
		                             category{ static_cast<unsigned int>(cat) } };

		i = i - 1;
		if (i <= 0) {
			if (r) {
				r.reset();
			}
		} else {
			*r = unique_range_in<my_domain>(attr);
		}

		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	}
}

int getNumDevices() {
	int d;
	cudaGetDeviceCount(&d);
	return d;
};

void CudaHostSync() {
	cudaDeviceSynchronize();
}

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
	cudaGraph_t graph;
	if constexpr (capture) {
		cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
		CudaCheckErrorModNoSync;
	}
	run();
	if constexpr (capture) {
		cudaStreamEndCapture(s.ptr(), &graph);
		if (!exec) {
			cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
			// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
			CudaCheckErrorModNoSync;
		} else {
			if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
				CudaCheckErrorModNoSync;
				// only instantiate a new graph if update fails
				cudaGraphExecDestroy(exec);
				cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
				// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
				CudaCheckErrorModNoSync;
			}
		}
		cudaGraphDestroy(graph);
		cudaGraphLaunch(exec, s.ptr());
	}
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
	void Stream::wait(const Event &ev) const {
		cudaStreamWaitEvent(ptr, ev.ptr());
	}
*/
void Stream::capture_begin() {
	CudaCheckErrorMod;
	std::cout << "Hello capture" << std::endl;
	cudaStreamCaptureStatus cap;
	cudaStreamIsCapturing(ptr(), &cap);

	CudaCheckErrorMod;
	if (cap == cudaStreamCaptureStatusNone) {
		std::cout << "None" << std::endl;
		cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
	} else if (cap == cudaStreamCaptureStatusActive) {
		std::cout << "Fail: activo" << std::endl;
	} else if (cap == cudaStreamCaptureStatusInvalidated) {
		std::cout << "Fail: invalidado" << std::endl;
	} else {
		std::cout << "Fail" << std::endl;
	}
	CudaCheckErrorMod;
}

void Stream::capture_end() {
	cudaGraph_t graph;
	cudaStreamEndCapture(ptr(), &graph);
	CudaCheckErrorMod;
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
	CudaCheckErrorMod;
	cudaGraphDestroy(graph);
	CudaCheckErrorMod;
	cudaGraphLaunch(graphExec, 0);
	cudaGraphExecDestroy(graphExec);

	cudaStreamSynchronize(0);
}

/// Caller holds `fence_lock`. The default mode returns on the first line: `ev`, no allocation, no
/// map, no growth of `lane_ev` -- the same event this function's callers always recorded.
cudaEvent_t Stream::fenceEvent() {
	if (!ConcurrentOps())
		return ev;
	const int lane_i = MemPoolLane();
	// MemPoolLane() is >= 0 by construction (it returns 0 as its overflow fallback), but a
	// negative index here would be a silent out-of-bounds, so fall back to the shared event
	// rather than trust it.
	if (lane_i < 0)
		return ev;
	const size_t lane = static_cast<size_t>(lane_i);
	if (lane_ev.size() <= lane)
		lane_ev.resize(lane + 1, nullptr);
	if (lane_ev[lane] == nullptr) {
		cudaEvent_t e = nullptr;
		if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess || e == nullptr) {
			// No event for this lane: fall back to the shared one, which is still correct --
			// every record and wait of it happens under `fence_lock` -- just serialized.
			cudaGetLastError();
			return ev;
		}
		lane_ev[lane] = e;
	}
	return lane_ev[lane];
}

void Stream::record(bool external) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kFence);
	//if (ptr_ == 0)
	//	return;
	CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
	// cudaEventDestroy(ev);
	// cudaEventCreate(&ev, cudaEventDisableTiming);
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	std::unique_lock<std::mutex> fg(fence_lock, std::defer_lock);
	if (ConcurrentOps())
		fg.lock();
	// DEFAULT MODE: fenceEvent() is `ev`, so this is the same call it always was. An `external`
	// record stays on `ev`: the only consumer of an external record is the graph it was captured
	// in, which names `ev` and knows nothing about lanes.
	const cudaEvent_t e = external ? ev : fenceEvent();
	cudaEventRecordWithFlags(e, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
	updated = true;
	// PROVENANCE. A record with no wait: whoever waits on `e` later is naming THIS point of this
	// stream, so the dump needs the pair (stream, event) to match a later wait against.
	PoolTraceFenceEv(PoolTraceFence::kRecord, ptr_, nullptr, e, /*early_out=*/false, /*lock_held=*/fg.owns_lock());
#endif
}

cudaEvent_t Stream::record_and_wait(cudaStream_t consumer) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kFence);
	CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	std::unique_lock<std::mutex> fg(fence_lock, std::defer_lock);
	if (ConcurrentOps())
		fg.lock();
	const cudaEvent_t e = fenceEvent();
	cudaEventRecordWithFlags(e, ptr_, cudaEventRecordDefault);
	updated = true;
	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(consumer, e);
	CudaCheckErrorModNoSync;
	// PROVENANCE. This is the pool hand-off itself: `self` is the PREVIOUS owner's lane stream,
	// `other` the taker's consumer stream, `ev` the fence the taker got. GPUmalloc copies the
	// returned event into the block's own ring, so the two dumps name the same fence.
	PoolTraceFenceEv(PoolTraceFence::kRecordAndWait, ptr_, consumer, e, /*early_out=*/false, /*lock_held=*/fg.owns_lock());
	return e;
#else
	(void)consumer;
	return nullptr;
#endif
}

/// DEFAULT MODE ONLY, and currently WITHOUT CALLERS. It reads `s.ev` with no lock and without
/// recording, so it depends on both things the concurrent mode gives up: that `updated` proves
/// where `ev` sits, and that `ev` is not a resource another thread is also recording. A
/// concurrent-mode caller must use `wait(Stream&)`, which records under both fence locks and on
/// the calling lane's own event.
void Stream::wait_recorded(const Stream& s) {
	assert(!ConcurrentOps() && "wait_recorded is not safe under FIDESLIB_CONCURRENT_OPS; use wait(Stream&)");
	if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
		return;
#if !DISABLE_STREAMS
	assert(s.updated);
	cudaStreamWaitEvent(ptr_, s.ev);
	this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kFence);

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	//
	// THE EARLY-OUT. `ptr_ == s.ptr_` says "we are already the same stream, and a stream is
	// ordered against itself". In the DEFAULT mode that reading is right, and this branch is kept
	// exactly as it was: one thread hands out Stream objects, so two of them carrying the same
	// `cudaStream_t` really are the same launch queue used twice.
	//
	// In the CONCURRENT mode it is not the same statement. `Stream::init` (below, ~:660) hands
	// every Stream a stream out of a 37-entry per-device pool, and a four-lane bootstrap asks for
	// a couple of hundred, so `ptr_ == s.ptr_` is overwhelmingly ALIASING -- two unrelated
	// logical streams, owned by two threads that share no lock, that happened to collide in the
	// pool. What the caller asked for is a dependency on `s`; what the early-out substitutes is
	// the FIFO order of a queue the OTHER thread is also enqueueing into. It is also the one
	// branch of this function that leaves `s` with no record at all, so a consumer that reaches
	// `s` through some other object (its `updated` flag, or the pool handshake) sees a stale one.
	// Compare identity instead: `this == &s` is the only case in which the fence is provably
	// nothing (and the only case that would self-deadlock the scoped_lock below).
	const bool same_stream = ConcurrentOps() ? (this == &s) : (ptr_ == s.ptr_);
	if (/*ptr_ == 0 ||*/ s.ptr_ == 0 || same_stream) {
		// PROVENANCE. An early out is a fence that was ASKED FOR and NOT ISSUED, so it belongs in
		// the ring exactly as much as one that was: a dropped dependency shows up here as a
		// consumer that took this branch against a producer it did not in fact alias.
		PoolTraceFenceEv(PoolTraceFence::kWaitStreamObj, ptr_, s.ptr_, nullptr, /*early_out=*/true, /*lock_held=*/false);
		updated = false;
		return;
	}
	//assert(ptr_ != nullptr);
	//assert(s.ptr_ != nullptr);
	assert(s.ev != nullptr);
	if (!ConcurrentOps()) {
		// DEFAULT MODE: untouched, instruction for instruction.
		if (!s.updated) {
			assert(!external); // Has to be recorded in the origin graph
			CudaCheckErrorModNoSync;
			cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
			s.updated = true;
			CudaCheckErrorModNoSync;
		}
		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
		this->updated = false;
		CudaCheckErrorModNoSync;
		// Unreachable while tracing is on (PoolTraceEnabled() implies ConcurrentOps()); kept so
		// that the ring describes every branch of this function rather than most of them.
		PoolTraceFenceEv(PoolTraceFence::kWaitStreamObj, ptr_, s.ptr_, s.ev, /*early_out=*/false, /*lock_held=*/false);
		return;
	}
	// CONCURRENT MODE. A fence touches BOTH Stream objects, and EITHER of them can be the one
	// that is shared between threads:
	//
	//   * `s` is shared when several threads READ one value -- two concurrent branches of a
	//     circuit multiplying by the same precomputed plaintext, say. That is the direction the
	//     first version of this lock covered.
	//   * `this` is shared when a reader BACK-FENCES the value it just read, which is what a copy
	//     out of a shared source does: LimbPartition::copyLimb fences the destination on the
	//     source (LimbPartition.cu:855), launches the copy, and then makes the SOURCE's stream
	//     wait on the destination's (LimbPartition.cu:870) so nothing overwrites the source
	//     underneath the copy. Three threads cloning one ciphertext therefore run
	//     `src.wait(dst_i)` on ONE `src` at once -- and the old code took only `s.fence_lock`,
	//     i.e. each thread took its OWN destination's lock and none of them serialized against
	//     the others or against a concurrent `dst.wait(src)` holding `src.fence_lock`.
	//
	// Two consequences, both removed here:
	//   1. `this->updated = false` was stored outside any lock on `this`, racing the
	//      `s.updated = true` a fence in the other direction performs under `s.fence_lock`. The
	//      loser leaves `updated` TRUE over a record taken before work that was enqueued after
	//      it, and the next fence on that stream binds cudaStreamWaitEvent to the stale record --
	//      a DROPPED DEPENDENCY, i.e. a consumer kernel that may read a buffer its producer has
	//      not written yet. Both stores now happen under both locks.
	//   2. `updated` is no longer trusted at all here: `Stream::ptr()` clears it on the
	//      kernel-launch path WITHOUT the lock (it cannot take one -- it is every launch), so
	//      "updated == true" is not a sound proof that `ev` sits at the stream's tail once more
	//      than one thread launches onto a stream. So record unconditionally. An extra
	//      cudaEventRecord per fence is a few microseconds of driver time and it retires the
	//      whole stale-record class; cudaStreamWaitEvent captures the event's state at CALL time,
	//      so another thread re-recording afterwards cannot weaken a fence already issued.
	//
	// Both locks are taken with std::scoped_lock, which orders them consistently, so two threads
	// fencing in opposite directions on the same pair cannot deadlock.
	//
	// THE EVENT IS THIS LANE'S, not `s`'s only one: see Stream::lane_ev. Two lanes fencing on one
	// shared producer therefore no longer record and wait on a single `cudaEvent_t`, which is
	// what made the correctness of every such fence rest on nothing but this lock -- and what
	// made the pool's split `record(); cudaStreamWaitEvent(x, ev);` pair (GPUmalloc) a hole,
	// since that pair holds a DIFFERENT mutex. An `external` wait still names `s.ev`: its record
	// came from the origin graph, which knows only that event.
	{
		std::scoped_lock<std::mutex, std::mutex> fg(fence_lock, s.fence_lock);
		assert(!external || s.updated); // external records have to come from the origin graph
		const cudaEvent_t e = external ? s.ev : s.fenceEvent();
		if (!external) {
			CudaCheckErrorModNoSync;
			cudaEventRecordWithFlags(e, s.ptr_, cudaEventRecordDefault);
			s.updated = true;
			CudaCheckErrorModNoSync;
		}
		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(ptr_, e, external ? cudaEventWaitExternal : cudaEventWaitDefault);
		this->updated = false;
		// PROVENANCE, INSIDE the critical section: the ring's order then matches the order the
		// driver saw, which is the whole question when two lanes fence on one producer.
		PoolTraceFenceEv(PoolTraceFence::kWaitStreamObj, ptr_, s.ptr_, e, /*early_out=*/false, /*lock_held=*/true);
		DiagStall(ptr_);
	}
#endif
	CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kFence);

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	// Same reading of `s == ptr_` as in wait(Stream&) above: in the concurrent mode it is an
	// aliasing collision out of the 37-entry pool, not a proof that the two are one launch queue,
	// and the caller (GPUfree's `mine.wait(stream)`) is fencing a pool handshake stream on a
	// compute stream that belongs to a different logical object. Issuing the fence anyway is a
	// record plus a wait on the same stream -- ordering-wise a no-op, and it leaves this stream's
	// lane event where the next `record_and_wait` expects it.
	if (s == 0 || (s == ptr_ && !ConcurrentOps())) {
		PoolTraceFenceEv(PoolTraceFence::kWaitRaw, ptr_, s, nullptr, /*early_out=*/true, /*lock_held=*/false);
		updated = false;
		return;
	}
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	CudaCheckErrorModNoSync;
	{
		std::unique_lock<std::mutex> fg(fence_lock, std::defer_lock);
		if (ConcurrentOps())
			fg.lock();
		const cudaEvent_t e = fenceEvent();
		cudaEventRecordWithFlags(e, s, cudaEventRecordDefault);
		updated = false;
		CudaCheckErrorModNoSync;

		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(ptr_, e, cudaEventWaitDefault);
		// PROVENANCE. This is GPUfree's half of the hand-off: `other` is the RELEASING stream the
		// event was recorded on, `self` the freeing lane's handshake stream that now waits on it.
		PoolTraceFenceEv(PoolTraceFence::kWaitRaw, ptr_, s, e, /*early_out=*/false, /*lock_held=*/fg.owns_lock());
		DiagStall(ptr_);
	}
#endif
	CudaCheckErrorModNoSync;
}

int low  = -1;
int high = -1;

constexpr int POOL_SIZE = 37;
// DIAGNOSTIC (bigpool): FIDESLIB_STREAM_POOL_SIZE=N (N <= kPoolSizeMax) widens the SHARED pool so two
// live Stream objects rarely hold one handle, with no per-object create/destroy. Default 37, as always.
constexpr int kPoolSizeMax = 4096;
static int PoolSizeEffective() {
	static const int n = [] {
		const char* env = std::getenv("FIDESLIB_STREAM_POOL_SIZE");
		const int v		= (env && *env) ? std::atoi(env) : POOL_SIZE;
		return v < 1 ? POOL_SIZE : (v > kPoolSizeMax ? kPoolSizeMax : v);
	}();
	return n;
}
cudaStream_t stream_pool[MAXG][kPoolSizeMax];

bool initPool() {
	int devs;
	cudaGetDeviceCount(&devs);
	for (int j = 0; j < devs; j++) {
		cudaSetDevice(j);
		for (int i = 0; i < PoolSizeEffective(); ++i) {
			cudaStreamCreateWithFlags(&stream_pool[j][i], 0);
		}
	}
	return true;
}

// DIAGNOSTIC (privstreams bisect): FIDESLIB_PRIVSTREAMS_PRIO="50,100" restricts the private-handle
// treatment to Stream::init calls made with those priority arguments -- 50 is every polynomial
// partition stream (LimbPartition::initStream), 100 the per-slot digit and top-limb scratch streams,
// 0 the pool handshake streams, digitStream2 and the context's per-limb records. Unset = every class.
static bool PrivStreamsPrioAllowed(const int priority) {
	static const std::vector<int> allowed = [] {
		std::vector<int> v;
		const char* env = std::getenv("FIDESLIB_PRIVSTREAMS_PRIO");
		if (env == nullptr || *env == '\0')
			return v;
		std::string spec(env);
		size_t i = 0;
		while (i <= spec.size()) {
			const size_t j = spec.find(',', i);
			const std::string tok = spec.substr(i, j == std::string::npos ? std::string::npos : j - i);
			if (!tok.empty())
				v.push_back(std::atoi(tok.c_str()));
			if (j == std::string::npos)
				break;
			i = j + 1;
		}
		return v;
	}();
	if (allowed.empty())
		return true;
	for (const int p : allowed)
		if (p == priority)
			return true;
	return false;
}

// DIAGNOSTIC (widepool): a second shared pool of kPoolSizeMax handles for the init-priority classes named
// in FIDESLIB_WIDEPOOL_PRIO (comma list; unset = nobody). Created on first use.
static bool WidePoolPrioAllowed(const int priority) {
	static const std::vector<int> allowed = [] {
		std::vector<int> v;
		const char* env = std::getenv("FIDESLIB_WIDEPOOL_PRIO");
		if (env == nullptr || *env == '\0')
			return v;
		std::string spec(env);
		size_t i = 0;
		while (i <= spec.size()) {
			const size_t j = spec.find(',', i);
			const std::string tok = spec.substr(i, j == std::string::npos ? std::string::npos : j - i);
			if (!tok.empty())
				v.push_back(std::atoi(tok.c_str()));
			if (j == std::string::npos)
				break;
			i = j + 1;
		}
		return v;
	}();
	for (const int p : allowed)
		if (p == priority)
			return true;
	return false;
}
cudaStream_t wide_pool[MAXG][kPoolSizeMax];
std::atomic<unsigned> wide_pool_idx[MAXG];
std::mutex wide_pool_lock;
static cudaStream_t widePoolStream(const int dev) {
	{
		std::lock_guard<std::mutex> lk(wide_pool_lock);
		if (wide_pool[dev][0] == nullptr) {
			for (int i = 0; i < kPoolSizeMax; ++i)
				cudaStreamCreateWithFlags(&wide_pool[dev][i], 0);
		}
	}
	return wide_pool[dev][wide_pool_idx[dev].fetch_add(1, std::memory_order_relaxed) % kPoolSizeMax];
}

// DIAGNOSTIC (lanepool): one POOL_SIZE-handle pool per (device, lane), created on first use by the
// lane's own thread. Lanes past kLanePoolLanes fall back to the shared pool. See kLanePool.
constexpr int kLanePoolLanes = 64;
cudaStream_t lane_pool[MAXG][kLanePoolLanes][POOL_SIZE];
std::atomic<unsigned> lane_pool_idx[MAXG][kLanePoolLanes];
std::mutex lane_pool_lock;
static cudaStream_t lanePoolStream(const int dev, const int lane) {
	if (lane < 0 || lane >= kLanePoolLanes)
		return nullptr;
	{
		std::lock_guard<std::mutex> lk(lane_pool_lock);
		if (lane_pool[dev][lane][0] == nullptr) {
			for (int i = 0; i < POOL_SIZE; ++i)
				cudaStreamCreateWithFlags(&lane_pool[dev][lane][i], 0);
		}
	}
	return lane_pool[dev][lane][lane_pool_idx[dev][lane].fetch_add(1, std::memory_order_relaxed) % POOL_SIZE];
}

#define USEPOOL true
#if USEPOOL
// The pool is created on first use instead of from a static initializer. A static
// initializer issues the first CUDA runtime calls of the process before main(); if one
// of them fails (e.g. cudaGetDeviceCount on a host with a degraded GPU) the runtime
// stays in the error state, the pool remains empty and Stream::init crashes later.
bool poolInitialized() {
	static const bool initialized = initPool();
	return initialized;
}
#else
bool initialized = false;
#endif

void Stream::init(int priority) {
	// Round-robin hand into the per-device stream pool. ATOMIC because Stream::init runs from
	// whichever thread constructs an RNSPoly: every LimbPartition takes a stream here
	// (LimbPartition.cu:70-81), so an application that builds device plaintexts on worker threads
	// reaches this counter concurrently. A torn increment would hand two partitions the same pool
	// stream (legal but serializing) and, worse, is a plain data race on a non-atomic int.
	static std::atomic<unsigned> assigner_idx[MAXD] = {};
	if (ptr_) {
		// free[ptr]++;
		cudaEventDestroy(ev);
		// cudaStreamDestroy(ptr_);
		// The lane events go with the stream they were recorded on. Empty in the default mode,
		// where `lane_ev` is never grown, so this loop does not run there.
		for (cudaEvent_t& e : lane_ev) {
			if (e != nullptr) {
				cudaEventDestroy(e);
				e = nullptr;
			}
		}
		lane_ev.clear();
		if (owned_)
			cudaStreamDestroy(ptr_);
		owned_ = false;
		ptr_   = nullptr;
		ev	   = nullptr;
	}

#if !DISABLE_STREAMS
#if USEPOOL
	poolInitialized();
#endif
	// Function-local static: C++11 guarantees the initializer runs exactly once, even under
	// concurrent first calls. The bare `if (high == -1)` this replaces was a read-modify-write of
	// two file-scope ints from every constructing thread.
	static const bool priority_range_queried = [] {
		cudaDeviceGetStreamPriorityRange(&low, &high);
		return true;
	}();
	(void)priority_range_queried;

	// int prio = low + priority * ((high - low - 1)) / 100;

#if USEPOOL
	int dev;
	cudaGetDevice(&dev);
	// DIAGNOSTIC (privstreams): a stream of this object's own, blocking like the pool's, instead
	// of one of the POOL_SIZE shared handles. See ConcurrentOpsDiag::kPrivStreams.
	if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kPrivStreams) && PrivStreamsPrioAllowed(priority)) {
		cudaStreamCreateWithPriority(&ptr_, 0, priority);
		owned_ = true;
	} else if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kLanePool) && (ptr_ = lanePoolStream(dev, MemPoolLane())) != nullptr) {
		// per-lane pool handle assigned above; pooled, so not owned
	} else if (WidePoolPrioAllowed(priority)) {
		// DIAGNOSTIC (widepool): FIDESLIB_WIDEPOOL_PRIO="50" draws THIS class from a separate
		// kPoolSizeMax-handle pool while every other class keeps the POOL_SIZE shared one. Earlier,
		// bigpool (every class wide) was 8/8 clean but 3.2x slower in the application; this asks
		// whether the partition streams alone need distinct handles, and what that alone costs.
		ptr_ = widePoolStream(dev);
	} else {
		ptr_ = stream_pool[dev][assigner_idx[dev].fetch_add(1, std::memory_order_relaxed) % PoolSizeEffective()];
	}
	// DIAGNOSTIC (churn): the API churn of privstreams without its handle change. See kChurn.
	if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kChurn)) {
		cudaStream_t throwaway = nullptr;
		cudaStreamCreateWithPriority(&throwaway, 0, priority);
		cudaStreamDestroy(throwaway);
	}
#else
	cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, priority);
	// cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
#endif

	cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
#else
	ptr_ = nullptr;
	ev   = nullptr;
#endif
	// free[ptr] = 0;
}

void Stream::initDefault() {
	ptr_    = 0;
	ev      = nullptr;
	updated = true;
}

// std::map<void *, int> free;

Stream::~Stream() {
	if (ptr_) {
		if (owned_)
			cudaStreamDestroy(ptr_); // privstreams only; pooled handles belong to the pool
		else if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kChurn)) {
			cudaStream_t throwaway = nullptr;
			cudaStreamCreateWithFlags(&throwaway, 0);
			cudaStreamDestroy(throwaway);
		}
		owned_ = false;
		ptr_   = nullptr;
	}
	if (ev) {
		cudaEventDestroy(ev);
		ev = nullptr;
	}
	// Empty in the default mode: `lane_ev` is only ever grown by fenceEvent() under
	// ConcurrentOps(), so this loop does not run in a default run.
	for (cudaEvent_t& e : lane_ev) {
		if (e != nullptr) {
			cudaEventDestroy(e);
			e = nullptr;
		}
	}
	lane_ev.clear();
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept
	: ptr_(s.ptr_), owned_(s.owned_), lane_ev(std::move(s.lane_ev)), ev(s.ev) {
	// fence_lock is deliberately NOT moved: a Stream is only ever moved while nobody is fencing on
	// it (vector growth at construction time), and std::mutex is not movable. The lane events ARE
	// moved: they belong to `ptr_`, which is moving, and leaving them behind would destroy them
	// in the source's destructor while the moved-to Stream still names them.
	updated.store(s.updated.load(std::memory_order_relaxed), std::memory_order_relaxed);
	s.lane_ev.clear();
	s.ptr_	 = nullptr;
	s.owned_ = false;
	s.ev	 = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
	if (GPUprop.empty()) {
		int count = 0;
		cudaGetDeviceCount(&count);
		for (int i = 0; i < count; ++i) {
			GPUprop.emplace_back();
			cudaGetDeviceProperties(&GPUprop.back(), i);

			std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
				<< ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
				<< ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
				<< " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

				<< "-bit" << std::endl;
		}
	}
}

// ============================================================================================
// PER-STREAM POOLED FREE LISTS
//
// THE PROBLEM. The legacy pool below (still present, still the fallback, see
// FIDESLIB_POOL_PER_STREAM) has ONE free-list map and ONE private Stream+event per device. Every
// pooled malloc records that event on the pool stream and makes the CALLER's stream wait on it;
// every pooled free records it on the caller's stream and makes the pool stream wait. So each
// pooled operation is a lock plus two driver calls, and -- worse -- every stream ends up ordered
// behind every other stream's most recent pool operation, whether or not the two ever touch the
// same block. An application that builds device plaintexts on 24 worker threads while a consumer
// thread builds ciphertexts does ~20 of these per plaintext, which makes the pool a process-wide
// serialization point rather than an allocator.
//
// THE SHAPE OF THE FIX. A pooled block does not need a global ordering; it needs exactly one
// thing: *not to be handed to a consumer that could run before the previous owner's work on THAT
// BLOCK has finished*. Keeping the free list PER STREAM discharges that for the common case at
// zero cost, because a stream is ordered against itself:
//
//   (1) SAME STREAM. A block freed by stream S goes on S's list and is preferentially handed back
//       to S. Every kernel S issued that touches the block was issued before the free, hence
//       before the malloc that hands it back, hence before anything S issues afterwards. Stream
//       order alone is the guarantee -- no event, no lock beyond S's own, no cross-stream edge.
//       This is the same argument every stream-ordered allocator (cudaMallocAsync included) runs on.
//
//   (2) CROSS-STREAM HANDOFF. When S's own list is empty we take a block from another stream T's
//       list rather than growing the pool. Only here is an event needed, and it is a PER-BLOCK
//       edge, not a global one: we record T's own event on T's stream and make S wait on it.
//       The record happens at HANDOFF time, not at free time, which is deliberate -- T's stream is
//       monotonic, so its "now" is at or after the free, and ordering S behind more of T than
//       strictly necessary is conservative, never unsafe. Recording lazily is what keeps the free
//       path at ZERO driver calls.
//
//   (3) FRESH MEMORY. A refill allocates its slab with cudaMallocAsync ON THE REQUESTING STREAM,
//       so that stream may use it immediately with no event at all. Any other stream can only
//       reach those blocks through (2), whose record is necessarily after the allocation.
//
// A block is in exactly one list at a time: it is removed under its owner's lock before a handoff
// and inserted under exactly one lock on free, so two streams can never hold the same block.
//
// WHAT ORDERING IS GIVEN UP. Only the part nothing depended on. The legacy event ordered EVERY
// malloc behind EVERY free on the device; the obligation it was discharging is the per-block one
// above, which (1)-(3) discharge per block instead. Nothing in this library reads a pooled block
// through a stream it did not get the block from.
//
// MEMORY FOOTPRINT. Per-stream lists would strand memory if a stream could refill while another
// stream sat on free blocks of the same size -- which is exactly why the handoff is tried BEFORE
// the refill and scans every registered stream. The pool therefore grows only when no free block
// of that size class exists anywhere on the device, which is the same condition the legacy pool
// grows under, so the steady-state ceiling is unchanged. (Neither pool ever returns memory to the
// driver; that is pre-existing and out of scope here.) The scan is the rebalance: there is no
// separate cap, and a cap without one would only convert stranding into failure.
//
// A refill additionally takes a per-device lock and RE-CHECKS, so two threads that both find the
// device empty cannot both allocate a 1 GB slab for the same class -- the one invariant the single
// global lock used to provide for free.

namespace {

/// @brief One device's free lists for ONE stream. Never destroyed: streams come from a
/// process-lifetime pool (initPool below), and tearing an event down at exit runs straight into the
/// cudaErrorCudartUnloading problem this file already documents.
struct StreamPool {
	std::mutex lock;						  ///< Guards `by_size` and `ev`'s creation.
	cudaStream_t stream = nullptr;
	int device			= -1;
	/// Handoff event, created on first use and re-recorded on every handoff. Re-recording is safe
	/// for a waiter that has already issued its cudaStreamWaitEvent: that call captures the event's
	/// state at the time of the call.
	cudaEvent_t ev = nullptr;
	std::map<int, std::vector<void*>> by_size;
};

/// Open-addressed, insert-only registry of the streams this device has pooled for. Sized well above
/// the 37 streams initPool creates, so the probe is short and the table never fills in practice; if
/// it ever did, PoolFor returns nullptr and the caller falls back to the legacy path rather than
/// failing.
constexpr size_t kPoolStreamSlots = 256;

struct StreamPoolSlot {
	std::atomic<cudaStream_t> key{ nullptr }; ///< Published LAST, with release, after `pool`.
	StreamPool* pool = nullptr;
};

StreamPoolSlot pool_slots[MAXG][kPoolStreamSlots];
/// Dense list of the same pools, for the handoff scan. Append-only; `pool_n` publishes the length.
StreamPool* pool_list[MAXG][kPoolStreamSlots];
std::atomic<size_t> pool_n[MAXG];
std::mutex pool_registry_lock[MAXG];
/// Serializes SLAB REFILLS per device (not ordinary allocation): see the footprint note above.
std::mutex pool_refill_lock[MAXG];

/// @brief Whether the per-stream pool is in use. Read once.
///
/// DEFAULT OFF: unset or `0` is the legacy single-lock pool, byte for byte the behaviour of the
/// parity tip; `FIDESLIB_POOL_PER_STREAM=1` opts in. The arm is the deviation and the base is the
/// control, so an A/B run whose experimental env is dropped measures the baseline rather than
/// silently measuring the arm twice.
///
/// PRECEDENCE. This flag is only ever consulted on the NON-concurrent branch of GPUmalloc /
/// GPUfree. FIDESLIB_CONCURRENT_OPS takes its own branch first and returns from it, so with
/// CONCURRENT_OPS on the per-stream pool is inert no matter what this is set to: the two schemes
/// are alternatives for the same obligation, not layers. Setting both looks like a composition and
/// is not one, which would put a per-stream timing of record on a run that never entered the
/// per-stream path (RULES 12/14), so say so once on stderr rather than let it pass.
bool PoolPerStreamEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_POOL_PER_STREAM");
		const bool on	= env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
		if (on && ConcurrentOps()) {
			std::cerr << "FIDESLIB_POOL_PER_STREAM=1 is IGNORED while FIDESLIB_CONCURRENT_OPS is set: "
					  << "the concurrent per-lane pool handles this device's recycling and the per-stream "
					  << "pool is never reached. Unset one of the two." << std::endl;
			return false;
		}
		return on;
	}();
	return enabled;
}

size_t PoolSlotHash(cudaStream_t s) {
	// Streams are pointers out of a small driver-side table; the low bits carry no entropy.
	return (reinterpret_cast<uintptr_t>(s) >> 5) % kPoolStreamSlots;
}

/// @brief This device's pool for `stream`, creating it on first use. nullptr only if the table is
/// full, which makes the caller take the legacy path.
StreamPool* PoolFor(int id, cudaStream_t stream) {
	// The DEFAULT stream is not registrable: nullptr is this table's empty marker, and a legacy
	// caller that reaches the pool with no stream is rare enough that handing it to the legacy path
	// costs nothing. The two pools then simply partition the blocks between them -- a block is
	// freed into exactly one of them and can only be handed out again by that one, so each keeps
	// its own ordering argument intact.
	if (stream == nullptr || id < 0 || id >= MAXG)
		return nullptr;

	const size_t h = PoolSlotHash(stream);
	// Lock-free read path. Insertion only ever fills the FIRST empty slot of a key's probe
	// sequence and nothing is ever removed, so stopping at an empty slot cannot skip an existing
	// entry; a concurrent insert is picked up by the locked re-scan below.
	for (size_t k = 0; k < kPoolStreamSlots; ++k) {
		StreamPoolSlot& slot   = pool_slots[id][(h + k) % kPoolStreamSlots];
		const cudaStream_t key = slot.key.load(std::memory_order_acquire);
		if (key == stream)
			return slot.pool; // published before `key`, so this read is safe
		if (key == nullptr)
			break;
	}

	std::lock_guard<std::mutex> guard(pool_registry_lock[id]);
	for (size_t k = 0; k < kPoolStreamSlots; ++k) {
		StreamPoolSlot& slot   = pool_slots[id][(h + k) % kPoolStreamSlots];
		const cudaStream_t key = slot.key.load(std::memory_order_relaxed);
		if (key == stream)
			return slot.pool;
		if (key == nullptr) {
			auto* p	  = new StreamPool(); // process-lifetime, deliberately never deleted
			p->stream = stream;
			p->device = id;
			slot.pool = p;
			const size_t n		= pool_n[id].load(std::memory_order_relaxed);
			pool_list[id][n]	= p;
			pool_n[id].store(n + 1, std::memory_order_release);
			slot.key.store(stream, std::memory_order_release);
			return p;
		}
	}
	return nullptr;
}

/// @brief GPUmalloc's pooled path, per-stream. Returns nullptr if it cannot serve the request, in
/// which case the caller falls through to the legacy path.
///
/// @param MBs slab size in MiB for a refill -- passed in rather than recomputed so the refill
///            granularity stays exactly what the legacy pool uses for that size class.
void* PoolTakeOwn(StreamPool* mine, int bytes) {
	std::lock_guard<std::mutex> guard(mine->lock);
	auto it = mine->by_size.find(bytes);
	if (it == mine->by_size.end() || it->second.empty())
		return nullptr;
	void* ptr = it->second.back();
	it->second.pop_back();
	return ptr;
}

/// @brief Case (2): take a block off another stream's list and order `stream` behind that stream.
void* PoolTakeHandoff(int id, int bytes, cudaStream_t stream, StreamPool* mine) {
	const size_t n = pool_n[id].load(std::memory_order_acquire);
	for (size_t i = 0; i < n; ++i) {
		StreamPool* other = pool_list[id][i];
		if (other == nullptr || other == mine)
			continue;
		std::lock_guard<std::mutex> guard(other->lock);
		auto it = other->by_size.find(bytes);
		if (it == other->by_size.end() || it->second.empty())
			continue;
		if (other->ev == nullptr) {
			cudaEvent_t ev = nullptr;
			if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess || ev == nullptr) {
				cudaGetLastError();
				continue; // no event, no ordering, no handoff -- leave the block where it is
			}
			other->ev = ev;
		}
		void* ptr = it->second.back();
		// Record and wait INSIDE the donor's lock. `other->stream` is monotonic, so this record is
		// at or after the free that put the block on that list -- which is the whole correctness
		// argument for the handoff, and why recording here rather than at free time is conservative
		// rather than unsafe. A concurrent re-record by another taker can only move the point
		// LATER, which is also conservative.
		cudaEventRecord(other->ev, other->stream);
		cudaStreamWaitEvent(stream, other->ev);
		it->second.pop_back();
		return ptr;
	}
	return nullptr;
}

void* PoolMallocPerStream(int id, int bytes, cudaStream_t stream, uint64_t MBs) {
	StreamPool* mine = PoolFor(id, stream);
	if (mine == nullptr)
		return nullptr;

	// (1) OUR OWN list: no driver call, no cross-stream edge, one uncontended lock.
	if (void* ptr = PoolTakeOwn(mine, bytes))
		return ptr;

	// (2) HANDOFF from another stream, with a per-block ordering edge. Tried before any refill so
	// the pool rebalances instead of growing.
	if (void* ptr = PoolTakeHandoff(id, bytes, stream, mine))
		return ptr;

	// (3) REFILL. One at a time per device, and BOTH lookups repeated under that lock: two threads
	// that each saw the device empty must not both allocate a slab for the same class, and the
	// second one usually finds what the first just produced.
	std::lock_guard<std::mutex> refill(pool_refill_lock[id]);
	if (void* ptr = PoolTakeOwn(mine, bytes))
		return ptr;
	if (void* ptr = PoolTakeHandoff(id, bytes, stream, mine))
		return ptr;

	// Allocated ON THE REQUESTING STREAM, so this stream may use the blocks with no event at all;
	// every other stream reaches them only through the handoff above.
	void* base = nullptr;
	cudaMallocAsync(&base, MBs * 1024 * 1024, stream);
	MemLogAlloc(base, MBs * 1024 * 1024, stream, __builtin_return_address(0));
	CudaCheckErrorModNoSync;
	if (base == nullptr)
		return nullptr;

	std::lock_guard<std::mutex> guard(mine->lock);
	std::vector<void*>& free_limb = mine->by_size[bytes];
	for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes) {
		free_limb.emplace_back(((char*)base) + i);
	}
	void* ptr = free_limb.back();
	free_limb.pop_back();
	return ptr;
}

/// @brief GPUfree's pooled path, per-stream: ZERO driver calls. Returns false only if the stream
/// registry is full, in which case the caller falls through to the legacy path.
bool PoolFreePerStream(int id, void* ptr, int bytes, cudaStream_t stream) {
	StreamPool* mine = PoolFor(id, stream);
	if (mine == nullptr)
		return false;
	std::lock_guard<std::mutex> guard(mine->lock);
	mine->by_size[bytes].emplace_back(ptr);
	return true;
}

} // namespace

// ============================================================================================

// Guards BOTH `size_to_memory[id]` and the pool's private stream `s[id]`.
//
// It used to be taken only around the vector push/pop, which left three races open the moment a
// second host thread allocated or freed device memory (which is what an encode/upload pipeline in
// the application does -- every device plaintext it builds allocates limbs here, and every one it
// evicts frees them):
//   1. `size_to_memory[id][bytes]` is a std::map::operator[]. Two threads reaching an absent size
//      class insert into the same red-black tree at once -- corruption, not a lost entry -- and
//      even an existing key cannot be looked up while another thread rebalances the tree.
//   2. `free_limb.empty()` was tested outside the lock against emplace_back inside it.
//   3. `s[id]` is ONE shared Stream object with ONE shared cudaEvent. The malloc path records that
//      event and then makes the caller's stream wait on it; the free path records it on the
//      caller's stream and makes the pool stream wait. Interleaved from two threads, a thread can
//      end up waiting on the *other* thread's record -- i.e. on a point in time that says nothing
//      about its own transfer -- which is how pooled memory gets handed out while its previous
//      owner's kernels are still reading it.
// The critical section is a vector pop plus two stream-ordering calls; the slab cudaMallocAsync
// inside it runs once per size class per 1 GB, so serializing it costs nothing measurable.
std::mutex mempool_lock[MAXG];

std::map<int, std::vector<void*>> size_to_memory[MAXG];

// ---- Concurrent mode: one HANDSHAKE STREAM per issuing thread ----------------------------------
//
// WHY. The pool recycles device memory, so a block must not be handed to its next owner while the
// previous owner's kernels are still reading it. The handshake that guarantees that is:
// GPUfree makes the POOL stream wait on the freeing stream, and GPUmalloc makes the NEW owner's
// stream wait on the pool stream. With ONE pool stream per device that is correct but maximally
// conservative -- the pool stream accumulates a wait on EVERY freeing stream, so every allocation
// inherits a dependency on every thread's most recent free.
//
// With one issuing thread that costs nothing: the chain it builds is that thread's own already
// ordered work. With several, it is a false dependency between INDEPENDENT chains, and it is
// paid per limb: it welds the threads' streams into one device order and stalls each thread's
// pipeline behind the others'. (Measured in the application that motivated the concurrent mode:
// four issuing threads made its matmul phase 3x SLOWER than one, with the issuing threads 97%
// blocked on the device.)
//
// So in the concurrent mode each allocating thread gets its OWN handshake stream (a POOL LANE,
// see MemPoolLane), and every pooled block remembers WHICH lane freed it. An allocation then
// fences on exactly one stream -- the block's previous owner -- instead of on all of them. The
// FREE LISTS stay shared, so the pool's device-memory footprint is unchanged (per-slot free lists
// would carve a fresh slab per thread per size class).
//
// Lane 0's handshake stream IS `s[id]`, and in the DEFAULT mode this whole structure is never
// touched: the code below takes the same branch, the same object and the same driver calls it
// took before.
/// Lane tag for a block that owes its next owner NO FENCE AT ALL. Only drainFreedLanes sets it,
/// and only on a block it has just made quiescent with a full device synchronize: after that sync
/// there is nothing in flight anywhere on the device that could still be reading the block, so
/// there is no stream for the next owner to wait on.
///
/// WHY IT IS NOT SIMPLY THE TAKER'S LANE. A reclaimed block tagged with the taker would make every
/// OTHER lane that later pops it record on, and wait for, the taker's handshake stream -- a
/// dependency on an unrelated thread's device order, which is exactly the false cross-lane
/// dependency the lane pool was built to remove (the one that made the application's matmul phase
/// 3x slower on four threads). The sentinel keeps the drain's benefit without paying that back.
///
/// It is never an index: it does not name a lane, so it must not reach mtPoolStream, freed_mt, or
/// the per-lane rows of MemPoolStatsSnapshot. The only place it can live is the FRESH tier, and
/// GPUfree re-tags the block with the freeing lane the moment it is released again -- a block that
/// has been reused since the drain has a live previous owner and must fence normally.
constexpr int kNoFenceLane = -1;

struct PooledBlock {
	void* ptr;
	/// Pool lane whose handshake stream the next owner must fence on: the lane that freed the
	/// block, or the lane that cut its chunk if it has never been handed out. `kNoFenceLane` if
	/// the block was reclaimed by a drain and is already quiescent.
	int lane;
};
/// Per LANE, not per device: in the concurrent mode a thread reuses only the blocks IT freed. A
/// freed block's last reader may be a kernel on a stream other than the freeing one (a fused kernel
/// reading it as an operand from its own output's stream), and the release handshake fences only
/// the freeing stream; within one thread the reuse comes later in that thread's device order (the
/// default-mode invariant), across threads the new owner's first write raced the old owner's last
/// read. Lists are indexed [device][lane]; lane 0 is the shared fallback.
/// Two tiers, so that footprint is shared but reuse is per thread:
///   * size_to_memory_mt[device][bytes] -- FRESH blocks: cut from a chunk, never handed out yet.
///     Any lane may take one (a fresh block has no prior reader). Chunks stay shared across lanes;
///     a per-lane chunk pool exhausted the device ('out of memory').
///   * freed_mt[device][lane][bytes] -- blocks FREED by that lane. Only that lane reuses them: a
///     freed block's last reader may be a kernel on a stream other than the freeing one, and the
///     release handshake fences only the freeing stream; within one thread the reuse comes later
///     in that thread's own device order (the default-mode invariant), across threads it did not
///     (two concurrent bootstraps mismatch, one at a time is clean).
std::map<int, std::vector<PooledBlock>> size_to_memory_mt[MAXG];
std::vector<std::map<int, std::vector<PooledBlock>>> freed_mt[MAXG];
std::vector<std::unique_ptr<Stream>> mt_pool_stream[MAXG];

FIDESlib::Stream s[MAXG];

/// The handshake stream of pool lane `lane` on device `id`. Lane 0 is the pool's own `s[id]`, so
/// a single-threaded run keeps exactly the stream it always had. Caller holds `mempool_lock[id]`.
static Stream& mtPoolStream(const int id, const int lane) {
	if (lane == 0) {
		if (s[id].ptr() == nullptr)
			s[id].init();
		return s[id];
	}
	std::vector<std::unique_ptr<Stream>>& v = mt_pool_stream[id];
	if (v.size() <= static_cast<size_t>(lane))
		v.resize(static_cast<size_t>(lane) + 1);
	if (v[lane] == nullptr) {
		v[lane] = std::make_unique<Stream>();
		v[lane]->init();
	}
	return *v[lane];
}

// ---- Pool counters and the drain ---------------------------------------------------------------
//
// Every counter below is read and written ONLY with `mempool_lock[id]` held, which is also what
// guards the lists they describe. Plain integers are therefore correct in both modes, and the
// default mode keeps the promise made in CudaUtils.cuh: no extra allocation, no lock, no atomic on
// a path it did not already have one on.
//
// They cover the two lock-guarded tiers (the concurrent lane pool and the legacy shared list). The
// The per-stream pool cuts its own slabs and is not counted here; it cannot be reached in the
// concurrent mode, because GPUmalloc takes the lane branch and returns from it first.
static std::map<int, unsigned long long> pool_chunks_cut[MAXG];
static std::map<int, unsigned long long> pool_drains[MAXG];
static std::map<int, unsigned long long> pool_blocks_reclaimed[MAXG];
static std::map<int, unsigned long long> pool_drain_skips[MAXG];
/// The PASS BOUNDARY release (MemPoolReleaseAllLanes), kept in its own counters rather than folded
/// into the drain's: a drain says the pool ran dry mid-pass, a boundary release says only that the
/// application reached a quiescent point, and a reader who cannot tell them apart cannot use either.
/// Per class for the blocks (so a report says WHICH class was stranded), per device for the calls.
static std::map<int, unsigned long long> pool_boundary_blocks[MAXG];
static unsigned long long pool_boundary_releases[MAXG] = { 0 };
static unsigned long long pool_boundary_skips[MAXG]	   = { 0 };

/// One line per NEW CHUNK, for FIDESLIB_POOL_STATS=1 only. The teardown report says what the pool
/// ENDED at; a run that dies before teardown (which is exactly the run worth investigating) says
/// nothing at all, and a run that grows says only its total. This says WHEN each chunk was cut and
/// what the drain had managed by then, so the log carries the growth curve rather than a single
/// end state, and it is bounded to one line per GiB claimed -- a few dozen lines in the runs that
/// motivated it, not a line per allocation.
///
/// Caller holds `mempool_lock[id]`: the counters below are read raw, and `find` rather than
/// `operator[]` so that printing cannot itself create a size-class row that the run never had.
static void logChunkCut(const int id, const int bytes, const uint64_t MBs, const int lane) {
	if (!MemPoolStatsEnabled())
		return;
	const auto at = [](const std::map<int, unsigned long long>& m, const int k) -> unsigned long long {
		const auto it = m.find(k);
		return it == m.end() ? 0ull : it->second;
	};
	std::fprintf(stderr,
				 "mempool dev%d class %d B: NEW CHUNK #%llu (%llu MiB) lane %d -- drains %llu reclaimed %llu "
				 "drain-skips %llu\n",
				 id, bytes, at(pool_chunks_cut[id], bytes), static_cast<unsigned long long>(MBs), lane,
				 at(pool_drains[id], bytes), at(pool_blocks_reclaimed[id], bytes), at(pool_drain_skips[id], bytes));
}

/// Reclaim every lane's freed blocks of size class `bytes` into the shared FRESH tier `fresh`.
/// Returns how many blocks moved. Caller holds `mempool_lock[id]` and has found BOTH the taker's
/// own freed list and `fresh` empty.
///
/// WHY THIS EXISTS. A freed block goes onto the FREEING lane's list and only that lane takes it
/// back, so in a producer/consumer application blocks flow one way and never return: the prefetch
/// workers device-encode plaintexts on their lanes and the consumer thread frees them on its. The
/// producers' fresh tier then empties forever, and before this drain the only answer to an empty
/// fresh tier was another chunk. Measured: ~113k limb blocks (~58 GiB) per transformer block, two
/// blocks filling a 96 GB card (`out of memory` at the CudaCheck after the
/// cudaMallocAsync below).
///
/// WHY A FULL DEVICE SYNCHRONIZE. A block in the fresh tier owes its taker exactly one thing: the
/// cudaMallocAsync that cut its chunk must have completed, which is what the taker's fence on the
/// cutting lane's stream gives it. Moving a FREED block into that tier means making the same true
/// of it, and the release handshake does not: GPUfree fences only the FREEING stream, while a
/// block's last reader may be a kernel on another stream entirely (a fused kernel reading it as an
/// operand from its own output's stream). That hole is the whole reason freed blocks are
/// lane-private, so recording an event on each lane's handshake stream and waiting on all of them
/// would inherit it exactly -- it fences the frees, not the reads. `cudaDeviceSynchronize` is the
/// only fence that covers every stream on the device, including the ones the pool has never heard
/// of, so it is the one that makes a reclaimed block genuinely quiescent and genuinely fresh.
///
/// COST. One device drain, paid ONLY at an exhaustion point -- where the alternative is a
/// chunk-sized cudaMallocAsync, itself a synchronizing call once the device is under pressure. A
/// drain returns every block of the class that is not live at that instant, so the next drain of
/// that class is about that many allocations away; at steady state it is rare and the chunk count
/// stops growing.
///
/// WHAT THE DEVICE GUARD IS FOR. `cudaDeviceSynchronize` waits on the CALLING THREAD'S CURRENT
/// DEVICE and on nothing else, so the fence above is only the fence this function claims when both
/// of these hold:
///   1. The calling thread's current device IS `id`. Otherwise the sync drains some other card
///      while blocks on `id` stay in flight, and the drain would hand out live memory.
///   2. No kernel on ANOTHER card can be reading a block on `id`. The context enables peer access
///      between every pair of devices it owns (Context.cu ~:940 and ~:946), and a P2P read is
///      issued on the PEER's stream -- a stream on a device this sync does not cover. There is no
///      device-wide fence that closes that, so the only sound answer is not to drain.
/// There is no `taker` parameter any more: reclaimed blocks are tagged kNoFenceLane rather than
/// with the taker's lane, so the caller's lane is no longer an input to this function.
///
/// Condition 2 is tested as `cudaGetDeviceCount() > 1` -- a deliberately CONSERVATIVE stand-in.
/// The precise question is whether the CONTEXT is using more than one card (its `GPUid` vector),
/// but that lives in CKKS::ContextData and this file is below it in the layering and cannot see
/// it. A second visible card that the context never touches therefore also disables the drain.
/// That costs nothing correctness-wise and little in practice: the concurrent mode is explicitly
/// single-device to begin with (Context.cu:463-464 -- "the NCCL registration of the context set is
/// for the multi-GPU peer copies, which the concurrent mode does not extend to"), so the
/// configuration this rules out is one the concurrent mode does not support anyway.
///
/// SKIPPING IS SAFE, not a failure: it returns 0, the caller finds `fresh` still empty and cuts a
/// chunk, which is exactly the behaviour that shipped before this branch. The skips are counted
/// (`drain_skips`) and reported on the FIDESLIB_POOL_STATS line, so a run that is growing because
/// the guard turned the drain off says so instead of looking like a drain that does not work.
static size_t drainFreedLanes(const int id, const int bytes, std::vector<PooledBlock>& fresh) {
	int current	= -1;
	int devices = 0;
	const cudaError_t got_dev	= cudaGetDevice(&current);
	const cudaError_t got_count = cudaGetDeviceCount(&devices);
	if (got_dev != cudaSuccess || got_count != cudaSuccess || current != id || devices > 1) {
		++pool_drain_skips[id][bytes];
		return 0;
	}

	cudaDeviceSynchronize();
	CudaCheckErrorModNoSync;
	size_t moved = 0;
	// Indexed rather than ranged so the SOURCE LANE of each reclaimed block has a name: a drain
	// is one of the interleavings the provenance tracer exists to distinguish, and "moved off
	// lane 2" is the fact that distinguishes it.
	for (size_t li = 0; li < freed_mt[id].size(); ++li) {
		std::map<int, std::vector<PooledBlock>>& lane_lists = freed_mt[id][li];
		const auto it										= lane_lists.find(bytes);
		if (it == lane_lists.end())
			continue;
		for (const PooledBlock& b : it->second) {
			// Quiescent after the sync above: nothing anywhere on this device is still reading it,
			// and the guard has ruled out a reader on another one. So the block goes into the
			// fresh tier owing its next owner NO fence -- see kNoFenceLane for why tagging it with
			// a real lane would put back the cross-lane dependency the lane pool exists to avoid.
			fresh.emplace_back(PooledBlock{ b.ptr, kNoFenceLane });
			if (PoolTraceEnabled())
				PoolTraceBlock(PoolTraceKind::kDrainMove, b.ptr, bytes, static_cast<int>(li), nullptr, nullptr, nullptr);
			++moved;
		}
		it->second.clear();
	}
	++pool_drains[id][bytes];
	pool_blocks_reclaimed[id][bytes] += moved;
	return moved;
}

/// See the declaration in CudaUtils.cuh for what this is for and why the fence is the one it is.
/// This is drainFreedLanes' argument applied to EVERY size class at once, at a point the CALLER
/// says is quiescent instead of at an exhaustion point.
///
/// It takes `mempool_lock[id]` itself, unlike drainFreedLanes, whose caller already holds it.
/// Every caller of this one is outside the pool (the pass boundary), so there is no lock to
/// inherit and none of the recursion hazard MemPoolFailureReport had to work around.
///
/// THE GUARD IS DUPLICATED, not shared with drainFreedLanes, because the two differ in what they
/// do when it declines: the drain is inside a chunk cut and simply returns 0 into a caller that
/// then cuts, while this one has a counter of its own to bump and nothing to fall through to.
/// Factoring the four driver calls out would save four lines and cost the reader the one thing
/// that matters here, which is that BOTH sites test it.
size_t MemPoolReleaseAllLanes(const int id) {
	// The default mode has no lane pool: `freed_mt` is never written, `size_to_memory_mt` is never
	// read, and GPUmalloc/GPUfree take the legacy branch. Returning here is what makes the whole
	// feature invisible to a run that has not asked for the concurrent mode -- no lock, no
	// synchronize, no counter.
	if (!ConcurrentOps())
		return 0;
	if (id < 0 || id >= MAXG)
		return 0;

	int current = -1;
	int devices = 0;
	const cudaError_t got_dev	= cudaGetDevice(&current);
	const cudaError_t got_count = cudaGetDeviceCount(&devices);
	if (got_dev != cudaSuccess || got_count != cudaSuccess || current != id || devices > 1) {
		std::lock_guard<std::mutex> guard(mempool_lock[id]);
		++pool_boundary_skips[id];
		return 0;
	}

	// BEFORE the lock. The synchronize is the expensive part and it needs no pool state; holding
	// `mempool_lock` across it would block every other thread's allocate and free for the whole
	// fence, and the caller's precondition says no such thread is doing useful work anyway -- but
	// a prefetch worker shutting down, say, has no reason to be stalled by our bookkeeping.
	//
	// Nothing between this fence and the moves below launches device work: the only calls are host
	// map/vector operations under the lock. So every block the loop finds is still quiescent when
	// it is tagged kNoFenceLane, which is exactly the claim that tag makes.
	cudaDeviceSynchronize();
	CudaCheckErrorModNoSync;

	size_t moved = 0;
	{
		std::lock_guard<std::mutex> guard(mempool_lock[id]);
		// Indexed for the same reason as drainFreedLanes above: the provenance ring records
		// which lane each block was taken off.
		for (size_t li = 0; li < freed_mt[id].size(); ++li) {
			std::map<int, std::vector<PooledBlock>>& lane_lists = freed_mt[id][li];
			for (std::pair<const int, std::vector<PooledBlock>>& per_class : lane_lists) {
				const int bytes					 = per_class.first;
				std::vector<PooledBlock>& blocks = per_class.second;
				if (blocks.empty())
					continue;
				std::vector<PooledBlock>& fresh = size_to_memory_mt[id][bytes];
				fresh.reserve(fresh.size() + blocks.size());
				for (const PooledBlock& b : blocks) {
					fresh.emplace_back(PooledBlock{ b.ptr, kNoFenceLane });
					if (PoolTraceEnabled())
						PoolTraceBlock(PoolTraceKind::kBoundaryMove, b.ptr, bytes, static_cast<int>(li), nullptr, nullptr, nullptr);
				}
				pool_boundary_blocks[id][bytes] += blocks.size();
				moved += blocks.size();
				blocks.clear();
			}
		}
		++pool_boundary_releases[id];
	}
	return moved;
}

// ---- NULL-STREAM GUARD FOR THE LANE POOL --------------------------------------------------
//
// See PoolNullStreamEvents() in the header for what a null stream does to a pooled hand-off and
// where one used to come from. This is the guard, and it lives ONLY inside the two concurrent
// branches below: the default mode never calls it, so it adds no atomic and no branch there.
//
// A null stream IS legal in the default mode -- Plaintext and KeySwitchingKey build their
// polynomials on the legacy default stream on purpose, and the legacy stream's implicit
// synchronization against every blocking stream is what orders them. It is never legal here.

/// Total null-stream arrivals at the lane pool. Only written under ConcurrentOps().
static std::atomic<unsigned long long> pool_null_stream_events{ 0 };

/// @brief Count a null stream at the lane pool and say so ONCE per process.
///
/// Debug builds abort on the assert in the caller; release builds print one line and carry on
/// with the (unfenced, and therefore possibly wrong) hand-off, which is what they did before this
/// existed -- the point of the line is that the condition stops being silent.
static void PoolNoteNullStream(const char* where) {
	const unsigned long long n = pool_null_stream_events.fetch_add(1, std::memory_order_relaxed);
	if (n == 0) {
		std::cerr << "FIDESlib: WARNING: " << where
				  << " was handed a NULL stream in the concurrent mode. The pool hand-off for that block is "
					 "UNFENCED: Stream::wait(cudaStream_t) early-outs on s == 0, so the block returns to the free "
					 "list owing nothing and its next owner may overwrite it while the previous owner's kernels "
					 "are still running. The owner of that memory has a Stream whose ptr_ is null -- a partition "
					 "built with def_stream, or a moved-from Stream still referenced by a Limb. Further "
					 "occurrences are counted, not printed: see FIDESlib::PoolNullStreamEvents()."
				  << std::endl;
	}
}

unsigned long long PoolNullStreamEvents() {
	return pool_null_stream_events.load(std::memory_order_relaxed);
}

#define MEMPOOL true
// void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
	void* ptr = nullptr;

	uint64_t MBs = 1024;

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
		MBs   = bytes / 1024;
	}
#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0 && ConcurrentOps()) {
		// CONCURRENT MODE: same pool, same slabs, but the handshake names the block's PREVIOUS
		// OWNER instead of "every thread that has ever freed". See the note on size_to_memory_mt.
		//
		// `stream` is the only thing that will order this allocation behind the block's previous
		// owner (`record_and_wait(stream)` below). A null one makes that fence a barrier on the
		// LEGACY default stream instead of a dependency on this caller's queue, and it means the
		// matching free will early out and record nothing at all. See PoolNullStreamEvents().
		assert(stream != nullptr);
		if (stream == nullptr)
			PoolNoteNullStream("GPUmalloc");
		const int lane = MemPoolLane();
		std::lock_guard<std::mutex> guard(mempool_lock[id]);

		if (freed_mt[id].size() <= static_cast<size_t>(lane))
			freed_mt[id].resize(static_cast<size_t>(lane) + 1);
		std::vector<PooledBlock>& own_freed = freed_mt[id][lane][bytes];
		std::vector<PooledBlock>& fresh		= size_to_memory_mt[id][bytes];
		Stream& mine						= mtPoolStream(id, lane);
		CudaCheckErrorModNoSync;
		PooledBlock block{ nullptr, lane };
		if (!own_freed.empty()) {
			block = own_freed.back();
			own_freed.pop_back();
		} else {
			// EXHAUSTION of both tiers of this size class. Take back what the OTHER lanes freed
			// BEFORE cutting anything: in a producer/consumer application that is where all the
			// memory is, and cutting instead is what filled the card in a GPU run. See
			// drainFreedLanes for the fence this costs and why it is the one that is needed.
			if (fresh.empty())
				drainFreedLanes(id, bytes, fresh);
			if (fresh.empty()) {
				uint64_t* base = nullptr;
				// The return value is CAPTURED but the sticky last-error is deliberately left
				// alone: CudaCheckErrorModNoSync below still sees it and still aborts exactly as
				// it did. All this buys is the chance to print the pool's own numbers first --
				// this is the site that died in a GPU run, and it died silently because
				// the abort path runs long before the teardown report.
				const cudaError_t chunk = cudaMallocAsync(&base, MBs * 1024 * 1024, mine.ptr());
				MemLogAlloc(base, MBs * 1024 * 1024, mine.ptr(), __builtin_return_address(0));
				if (chunk != cudaSuccess)
					MemPoolFailureReport(id, bytes, "GPUmalloc chunk cut, concurrent lane pool");
				CudaCheckErrorModNoSync;
				++pool_chunks_cut[id][bytes];
				logChunkCut(id, bytes, MBs, lane);
				for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes) {
					fresh.emplace_back(PooledBlock{ ((char*)base) + i, lane });
					// The block's FIRST appearance, and the only point at which the tracer
					// allocates: this is what creates its ring. `mine` is the stream the chunk's
					// cudaMallocAsync was issued on, which is the one a taker of a never-used
					// fresh block fences against.
					if (PoolTraceEnabled())
						PoolTraceBlock(PoolTraceKind::kCut, ((char*)base) + i, bytes, lane, nullptr, mine.raw(), nullptr);
				}
			}
			block = fresh.back();
			fresh.pop_back();
		}

		// ONE fence, on the stream that last released this block. Everything the previous owner
		// had in flight over it was fenced into that stream by GPUfree.
		//
		// Except for a block a drain reclaimed: that one was made quiescent by a full device
		// synchronize and carries kNoFenceLane, so there is no previous owner and no stream to
		// wait on. Fencing it anyway would mean recording on, and waiting for, whichever lane
		// happened to take it -- an invented dependency between two unrelated threads. Note this
		// also keeps kNoFenceLane out of mtPoolStream, which would index a vector with it.
		//
		// ONE CRITICAL SECTION, on the owner's own `fence_lock`. This used to be
		// `owner.record(); cudaStreamWaitEvent(stream, owner.ev);` -- a record and a wait on one
		// `cudaEvent_t` with the lock that guards that event held across the first statement and
		// dropped before the second, the pair standing instead on `mempool_lock[id]`. That is a
		// different mutex from the one Stream::record / Stream::wait take, so the pair was sound
		// only for as long as nothing outside this pool ever fenced on a handshake stream.
		// record_and_wait holds the right lock across both, and is also the only way to write the
		// pair now that the event is lane-private (Stream::lane_ev): `owner.ev` is no longer the
		// event a concurrent-mode record lands on.
		const Stream* owner_s = nullptr;
		cudaEvent_t fence_ev  = nullptr;
		if (block.lane != kNoFenceLane) {
			Stream& owner = mtPoolStream(id, block.lane);
			fence_ev	  = owner.record_and_wait(stream);
			owner_s		  = &owner;
			CudaCheckErrorModNoSync;
		}
		// PROVENANCE. The lane recorded is the TAKER's (that is what "last taken by lane N"
		// has to mean); the PREVIOUS owner is named by `owner_stream`, and by the FREE event
		// sitting earlier in this same ring with that lane on it. A null owner_stream and a null
		// event mean the block carried kNoFenceLane -- reclaimed by a drain, owing nothing.
		if (PoolTraceEnabled())
			PoolTraceBlock(PoolTraceKind::kTake, block.ptr, bytes, lane, stream, owner_s == nullptr ? nullptr : owner_s->raw(), fence_ev);
		// DIAGNOSTIC (takesync): the taker starts only after every kernel enqueued so far, on any
		// stream, has finished -- a superset of every fence the pool could have issued. See the enum.
		if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kTakeSync))
			cudaDeviceSynchronize();
		return block.ptr;
	}

	if (cache && (bytes & (bytes - 1)) == 0) {
		// Per-stream free lists. Same size classes, same slab granularity, same pooled/
		// non-pooled split -- only the list the block comes from and the ordering that makes it
		// safe are different. See the block comment above PoolFor. A null return means only that
		// the stream registry is full or the slab allocation failed, and the legacy path below
		// still handles it.
		if (PoolPerStreamEnabled()) {
			void* pooled = PoolMallocPerStream(id, bytes, stream, MBs);
			if (pooled != nullptr)
				return pooled;
		}

		// One critical section for the whole pooled path: the map lookup, the slab refill, the
		// pool-stream event handshake and the pop are all state shared across threads. See the
		// note on mempool_lock.
		std::lock_guard<std::mutex> guard(mempool_lock[id]);

		std::vector<void*>& free_limb = size_to_memory[id][bytes];

		if (s[id].ptr() == nullptr) {
			s[id].init();
		}
		CudaCheckErrorModNoSync;
		if (free_limb.empty()) {
			uint64_t* base;
			// Same capture-and-report as the concurrent branch above, for the same reason and with
			// the same non-effect on what happens next: the error stays sticky and the
			// CudaCheckErrorModNoSync after this loop still aborts on it.
			const cudaError_t chunk = cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());
			MemLogAlloc(base, MBs * 1024 * 1024, s[id].ptr(), __builtin_return_address(0));
			if (chunk != cudaSuccess)
				MemPoolFailureReport(id, bytes, "GPUmalloc chunk cut, default shared pool");
			++pool_chunks_cut[id][bytes];
			logChunkCut(id, bytes, MBs, 0);

			for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes) {
				free_limb.emplace_back(((char*)base) + i);
			}
		}
		CudaCheckErrorModNoSync;

		//if (stream != nullptr) {
		s[id].record();
		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(stream, s[id].ev);
		//}

		CudaCheckErrorModNoSync;
		ptr = free_limb.back();
		free_limb.pop_back();
		// ptr = free_limb.front();
		// free_limb.pop_front();
		// std::cout << "get " << ptr << std::endl;
		return ptr;
	}

#endif
	// std::cout << bytes << std::endl;
	if (0) {
		cudaMalloc(&ptr, bytes);
	} else if (1) {
		const unsigned long long enter = MemLogEnter();
		cudaMallocAsync(&ptr, bytes, 0);
		MemLogAlloc(ptr, bytes, 0, __builtin_return_address(0), enter);
	} else {
		if (size_to_memory[id][bytes].empty()) {
			cudaSetDevice(id);
			cudaMallocAsync(&ptr, bytes, stream);
		} else {
			mempool_lock[id].lock();
			if (size_to_memory[id][bytes].empty()) {
				mempool_lock[id].unlock();
				cudaSetDevice(id);
				cudaMallocAsync(&ptr, bytes, stream);
			} else {
				ptr = size_to_memory[id][bytes].back();
				size_to_memory[id][bytes].pop_back();
				mempool_lock[id].unlock();
			}
		}
	}

	return ptr;
}

struct pointerdata {
	void* pointer;
	int id;
	int bytes;
};

void CUDART_CB streamCallback(void* userData) {

	auto* p = reinterpret_cast<pointerdata*>(userData);

	mempool_lock[p->id].lock();
	size_to_memory[p->id][p->bytes].push_back(p->pointer);
	mempool_lock[p->id].unlock();
	delete p;
}

thread_local bool gpufree_presynced = false;

namespace {
/// DIAGNOSTIC (noreuse): the quarantine FIFO. Process-wide, under mempool_lock of the block's device
/// (every push and pop below happens inside GPUfree's lock), so no further lock.
struct QuarantinedBlock {
	void* ptr;
	int id;
	int bytes;
};
std::deque<QuarantinedBlock> pool_quarantine;
size_t pool_quarantine_bytes = 0;
constexpr size_t kPoolQuarantineBytes = 24ull << 30; // 24 GiB held back before the oldest block is recycled
} // namespace

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
	}

#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0 && ConcurrentOps()) {
		// CONCURRENT MODE: fence THIS thread's handshake stream on the releasing stream and
		// record which slot that was, so the block's next owner waits on this thread alone.
		//
		// `stream` is the RELEASING stream, and the whole handshake is `mine.wait(stream)` below.
		// A null one takes that call's `s == 0` early out (Stream::wait(cudaStream_t)), which
		// issues NOTHING: the block is pushed back carrying this lane's tag but no dependency, and
		// the next lane fences against an empty handshake stream. See PoolNullStreamEvents().
		assert(stream != nullptr);
		if (stream == nullptr)
			PoolNoteNullStream("GPUfree");
		const int lane = MemPoolLane();
		std::lock_guard<std::mutex> guard(mempool_lock[id]);

		if (freed_mt[id].size() <= static_cast<size_t>(lane))
			freed_mt[id].resize(static_cast<size_t>(lane) + 1);
		std::vector<PooledBlock>& free_limb = freed_mt[id][lane][bytes];
		Stream& mine						= mtPoolStream(id, lane);
		CudaCheckErrorModNoSync;
		// DIAGNOSTIC (freesync): make the block quiescent before it is listed, whatever stream its
		// last kernel ran on. Splits an under-fenced free from a shared-object race; see the enum.
		if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kFreeSync))
			cudaDeviceSynchronize();
		// DIAGNOSTIC (noreuse): hold the block back instead of listing it. Recycle the oldest only
		// past the cap, synchronized, into the shared fresh tier as fence-free. See the enum.
		if (ConcurrentOpsDiagOn(ConcurrentOpsDiag::kNoReuse)) {
			pool_quarantine.push_back(QuarantinedBlock{ ptr, id, bytes });
			pool_quarantine_bytes += static_cast<size_t>(bytes);
			while (pool_quarantine_bytes > kPoolQuarantineBytes && !pool_quarantine.empty()) {
				const QuarantinedBlock old = pool_quarantine.front();
				pool_quarantine.pop_front();
				pool_quarantine_bytes -= static_cast<size_t>(old.bytes);
				cudaDeviceSynchronize();
				if (old.id == id)
					size_to_memory_mt[id][old.bytes].emplace_back(PooledBlock{ old.ptr, kNoFenceLane });
				else
					pool_quarantine.push_back(old); // another device: keep holding it (single-GPU runs never get here)
			}
			return;
		}
		mine.wait(stream);
		CudaCheckErrorModNoSync;
		// RE-TAG, unconditionally, with the freeing lane. A block that arrived here carrying
		// kNoFenceLane (reclaimed by a drain, then handed out) has been used since that drain, so
		// it has a live previous owner again and its next owner must fence on this lane's
		// handshake stream like any other freed block. The sentinel describes one moment, not the
		// block.
		free_limb.emplace_back(PooledBlock{ ptr, lane });
		// PROVENANCE. `stream` is the RELEASING stream -- the only stream this free fences, which
		// is exactly the hole the race is suspected to sit in when the block's last READER ran on
		// some other stream. The event `mine.wait(stream)` used is the fence-ring entry
		// immediately before this one.
		if (PoolTraceEnabled())
			PoolTraceBlock(PoolTraceKind::kFree, ptr, bytes, lane, stream, mine.raw(), nullptr);
		return;
	}

	if (cache && (bytes & (bytes - 1)) == 0) {
		// The block goes back to the list of the stream that freed it, and no event is recorded
		// -- see (1) in the block comment above PoolFor for why stream order alone is sufficient
		// when the same stream takes it back, and (2) for the edge a cross-stream taker records.
		if (PoolPerStreamEnabled() && PoolFreePerStream(id, ptr, bytes, stream)) {
			return;
		}

		// Same single critical section as GPUmalloc: `s[id].wait(stream)` records the pool's ONE
		// shared event on the caller's stream, so it must not interleave with another thread's
		// record of the same event. See the note on mempool_lock.
		std::lock_guard<std::mutex> guard(mempool_lock[id]);

		std::vector<void*>& free_limb = size_to_memory[id][bytes];
		if (s[id].ptr() == nullptr) {
			s[id].init();
		}
		CudaCheckErrorModNoSync;
		// cudaDeviceSynchronize();
		//if (stream != nullptr) {
		if (!gpufree_presynced)   // redundant while ContextData::clearAuxilarPoly holds the device synchronized
			s[id].wait(stream);
		//}
		CudaCheckErrorModNoSync;
		free_limb.emplace_back(ptr);
		// std::cout << "free " << ptr << std::endl;
		return;
	}
#endif

	if (0) {
		cudaFree(ptr);
	} else if (1) {
		const unsigned long long enter = MemLogEnter();
		MemLogFreeMark(ptr, 0);
		cudaFreeAsync(ptr, 0);
		MemLogFree(ptr, 0, __builtin_return_address(0), enter);
	} else {
		auto* p    = new pointerdata;
		p->id      = id;
		p->bytes   = bytes;
		p->pointer = ptr;
		cudaLaunchHostFunc(stream, streamCallback, p);
	}
}

// ---- EXACT-SIZE POOL CLASS (the plaintext arena; LimbPartition::generateLimbArena) ----
//
// GPUmalloc's default shared pool, keyed by the EXACT byte count instead of a power of two: the same
// free list (size_to_memory[id][bytes]), the same lock, the same pool-stream fences, chunks cut as
// as many whole blocks as fit in 128 MB (at least one; see the chunk-size note in the body). The pow2 class cost
// 8 MB for 5.5 MB of limbs, and with many resident plaintexts the difference added GBs of peak memory;
// this keeps the arena's footprint at its payload. Serial pool only: the concurrent mode's lane pool
// is not involved, which is fine because a block taken here is returned ONLY through GPUfreeExact,
// so no other path ever sees this class. Not for sizes below 64 KB (GPUmalloc's small classes).
void* GPUmallocExact(const int id, const size_t bytes, const cudaStream_t stream) {
	if (bytes < 64 * 1024 || bytes % 4096 != 0 || bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
		throw std::invalid_argument("GPUmallocExact: bytes must be a multiple of 4096 in [64 KB, 2 GB)");
	const int key = static_cast<int>(bytes);
	std::lock_guard<std::mutex> guard(mempool_lock[id]);
	std::vector<void*>& free_limb = size_to_memory[id][key];
	if (s[id].ptr() == nullptr) {
		s[id].init();
	}
	CudaCheckErrorModNoSync;
	if (free_limb.empty()) {
		// 128 MB chunks, not GPUmalloc's 1 GB: device-encoded plaintexts arrive at several levels, so
		// the arena spreads over several exact classes (one per limb count), and each class's LAST chunk
		// is mostly empty. At 1 GB that over-allocation reached ~13 GB in testing (90 chunks
		// against the control's 77 with the same resident bytes); at 128 MB it is bounded by
		// 128 MB x classes.
		const size_t chunkTarget = 128ull << 20;
		const size_t nblocks	 = std::max<size_t>(1, chunkTarget / bytes);
		const size_t chunk		 = nblocks * bytes;
		uint64_t* base		 = nullptr;
		const cudaError_t rc = cudaMallocAsync(&base, chunk, s[id].ptr());
		MemLogAlloc(base, chunk, s[id].ptr(), __builtin_return_address(0));
		if (rc != cudaSuccess)
			MemPoolFailureReport(id, key, "GPUmallocExact chunk cut, default shared pool");
		++pool_chunks_cut[id][key];
		logChunkCut(id, key, chunk >> 20, 0);
		for (size_t i = 0; i < nblocks; ++i) {
			free_limb.emplace_back(((char*)base) + i * bytes);
		}
	}
	CudaCheckErrorModNoSync;
	s[id].record();
	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(stream, s[id].ev);
	CudaCheckErrorModNoSync;
	void* ptr = free_limb.back();
	free_limb.pop_back();
	return ptr;
}

void GPUfreeExact(void* ptr, const int id, const size_t bytes, const cudaStream_t stream) {
	const int key = static_cast<int>(bytes);
	std::lock_guard<std::mutex> guard(mempool_lock[id]);
	std::vector<void*>& free_limb = size_to_memory[id][key];
	if (s[id].ptr() == nullptr) {
		s[id].init();
	}
	CudaCheckErrorModNoSync;
	s[id].wait(stream);
	CudaCheckErrorModNoSync;
	free_limb.emplace_back(ptr);
}


bool MemPoolStatsEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_POOL_STATS");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled;
}

// ---- BLOCK PROVENANCE TRACER ------------------------------------------------------------------
//
// See CudaUtils.cuh for what this is for. Everything below is unreachable unless
// FIDESLIB_POOL_TRACE=1 AND FIDESLIB_CONCURRENT_OPS=1: every entry point tests PoolTraceEnabled()
// -- one cached-bool read -- and returns.

bool PoolTraceEnabled() {
	static const bool enabled = [] {
		// The tracer describes the LANE POOL, which only the concurrent mode has: in the default
		// mode there are no lanes, no per-lane freed lists and no cross-thread hand-off to trace,
		// so the variable alone must not be able to turn any of this on.
		if (!ConcurrentOps())
			return false;
		const char* env = std::getenv("FIDESLIB_POOL_TRACE");
		return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
	}();
	return enabled;
}

namespace {

/// Events kept per pooled block. Eight is what the failing shape needs: a block that came back
/// wrong was cut once and has been taken and freed a handful of times since the last drain, and
/// the interesting window is the LAST hand-off plus the one before it.
constexpr size_t kPoolTraceBlockEvents = 8;
/// Events kept in the global fence ring. 64Ki entries x 64 B is ~4 MiB, preallocated once.
constexpr size_t kPoolTraceFenceEvents = 1u << 16;

/// ONE sequence for both rings, so a block event and the fence event it caused are adjacent in a
/// single order and the two dumps can be read interleaved by `seq`.
std::atomic<unsigned long long> trace_seq{ 0 };

unsigned long long traceEpochNs() {
	static const unsigned long long t0 = static_cast<unsigned long long>(
	  std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
	return t0;
}

/// Coarse host timestamp: microseconds since the first traced event. Coarse on purpose -- it is
/// there to say "these two events are 3 us apart", not to time anything.
unsigned long long traceNowUs() {
	// The epoch FIRST: it is initialized on its own first call, so sampling `now` before it would
	// make the very first traced event sit before the epoch and underflow to ~1.8e19 us.
	const unsigned long long t0	 = traceEpochNs();
	const unsigned long long now = static_cast<unsigned long long>(
	  std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
	return now > t0 ? (now - t0) / 1000ull : 0ull;
}

/// A small dense id per host thread, assigned in first-touch order. std::thread::id prints as an
/// opaque value that differs run to run; "thr 2" is what a dump can be read with.
std::atomic<int> trace_thread_counter{ 0 };
int traceThreadId() {
	static thread_local const int id = trace_thread_counter.fetch_add(1, std::memory_order_relaxed);
	return id;
}

struct TraceBlockEvent {
	unsigned long long seq	= 0;
	unsigned long long t_us = 0;
	int thread				= -1;
	int lane				= -2;
	PoolTraceKind kind		= PoolTraceKind::kCut;
	const void* consumer	= nullptr; ///< taker's consumer stream / releasing stream / writing stream
	const void* owner		= nullptr; ///< pool handshake stream fenced on, if any
	const void* ev			= nullptr; ///< the fence event that carried it, if any
};

struct TraceBlockRing {
	TraceBlockEvent e[kPoolTraceBlockEvents];
	unsigned long long total = 0; ///< events ever appended; the i-th sits in slot i % N
	int bytes				 = 0;
	int last_taken_lane		 = -2; ///< -2 = never, otherwise a lane (or kNoFenceLane)
	int last_freed_lane		 = -2;
	int last_output_lane	 = -2;
	unsigned long long last_taken_seq  = 0;
	unsigned long long last_freed_seq  = 0;
	unsigned long long last_output_seq = 0;
	const void* last_output_stream	   = nullptr;
};

/// Guards the block map ONLY. Taken inside `mempool_lock[id]` by the pool hooks and on its own by
/// the output mark, and it takes no other lock itself, so it can never invert against the pool's.
std::mutex trace_block_lock;
/// ORDERED by address on purpose: a limb's data pointer can be an interior offset of the block
/// that backs it, and an ordered map answers "which block contains this address" with one
/// upper_bound. Nodes are inserted when a chunk is cut, i.e. during warm-up.
std::map<const void*, TraceBlockRing> trace_blocks;

struct TraceFenceEvent {
	/// Written LAST, with release, and read with acquire: a dump that runs while lanes are still
	/// issuing then either sees a fully written slot or skips it, rather than printing halves of
	/// two different events. In the case this instrument is built for the lanes are joined and
	/// the device is synchronized before anything dumps.
	std::atomic<unsigned long long> seq{ 0 };
	unsigned long long t_us = 0;
	int thread				= -1;
	int lane				= -2;
	PoolTraceFence kind		= PoolTraceFence::kRecord;
	const void* self		= nullptr;
	const void* other		= nullptr;
	const void* ev			= nullptr;
	bool early_out			= false;
	bool lock_held			= false;
};

/// Preallocated once, on the first traced fence, and never grown or freed: the ring is a fixed
/// window and the process exits with it. `new[]` rather than a vector because the entries hold an
/// atomic and are therefore neither copyable nor movable.
TraceFenceEvent* traceFenceRing() {
	static TraceFenceEvent* const ring = new TraceFenceEvent[kPoolTraceFenceEvents]();
	return ring;
}

const char* traceKindName(PoolTraceKind k) {
	switch (k) {
		case PoolTraceKind::kCut: return "CUT";
		case PoolTraceKind::kTake: return "TAKE";
		case PoolTraceKind::kFree: return "FREE";
		case PoolTraceKind::kDrainMove: return "DRAIN-MOVE";
		case PoolTraceKind::kBoundaryMove: return "BOUNDARY-MOVE";
		case PoolTraceKind::kOutput: return "OUTPUT";
	}
	return "?";
}

const char* traceFenceName(PoolTraceFence k) {
	switch (k) {
		case PoolTraceFence::kRecord: return "record";
		case PoolTraceFence::kWaitStreamObj: return "wait(Stream&)";
		case PoolTraceFence::kWaitRaw: return "wait(cudaStream_t)";
		case PoolTraceFence::kRecordAndWait: return "record_and_wait";
	}
	return "?";
}

/// "lane 3", or "lane none" for the drain's kNoFenceLane sentinel, or "lane -" for "never".
std::string traceLaneName(int lane) {
	if (lane == -2)
		return "-";
	if (lane == kNoFenceLane)
		return "none";
	return std::to_string(lane);
}

/// CALLER HOLDS `trace_block_lock`. Null if no block contains `p`.
TraceBlockRing* traceFindBlockLocked(const void* p) {
	const auto exact = trace_blocks.find(p);
	if (exact != trace_blocks.end())
		return &exact->second;
	auto it = trace_blocks.upper_bound(p); // first base strictly greater than p
	if (it == trace_blocks.begin())
		return nullptr;
	--it; // greatest base at or below p
	const char* base = static_cast<const char*>(it->first);
	if (static_cast<const char*>(p) < base + it->second.bytes)
		return &it->second;
	return nullptr;
}

} // namespace

void PoolTraceBlock(const PoolTraceKind kind, const void* ptr, const int bytes, const int lane, const void* consumer, const void* owner,
					const void* ev) {
	if (!PoolTraceEnabled() || ptr == nullptr)
		return;
	const unsigned long long seq = trace_seq.fetch_add(1, std::memory_order_relaxed);
	const unsigned long long t	 = traceNowUs();
	const int thread			 = traceThreadId();
	std::lock_guard<std::mutex> guard(trace_block_lock);
	// operator[] default-constructs the ring the first time this block is seen -- which is the
	// kCut that slices it out of a fresh chunk, i.e. warm-up. Steady state finds the node.
	TraceBlockRing& r = trace_blocks[ptr];
	if (bytes > 0)
		r.bytes = bytes;
	TraceBlockEvent& e = r.e[r.total % kPoolTraceBlockEvents];
	e.seq			   = seq;
	e.t_us			   = t;
	e.thread		   = thread;
	e.lane			   = lane;
	e.kind			   = kind;
	e.consumer		   = consumer;
	e.owner			   = owner;
	e.ev			   = ev;
	++r.total;
	if (kind == PoolTraceKind::kTake) {
		r.last_taken_lane = lane;
		r.last_taken_seq  = seq;
	} else if (kind == PoolTraceKind::kFree) {
		r.last_freed_lane = lane;
		r.last_freed_seq  = seq;
	} else if (kind == PoolTraceKind::kOutput) {
		r.last_output_lane	 = lane;
		r.last_output_seq	 = seq;
		r.last_output_stream = consumer;
	}
}

void PoolTraceFenceEv(const PoolTraceFence kind, const void* self, const void* other, const void* ev, const bool early_out,
					  const bool lock_held) {
	if (!PoolTraceEnabled())
		return;
	TraceFenceEvent* const ring	 = traceFenceRing();
	const unsigned long long seq = trace_seq.fetch_add(1, std::memory_order_relaxed);
	TraceFenceEvent& slot		 = ring[seq % kPoolTraceFenceEvents];
	slot.t_us					 = traceNowUs();
	slot.thread					 = traceThreadId();
	slot.lane					 = MemPoolLane();
	slot.kind					 = kind;
	slot.self					 = self;
	slot.other					 = other;
	slot.ev						 = ev;
	slot.early_out				 = early_out;
	slot.lock_held				 = lock_held;
	// LAST, and with release: everything above must be visible to a reader that sees this seq.
	// `seq + 1` is stored so that 0 stays the "never written" value for a slot the run never
	// reached; the printer subtracts it back out.
	slot.seq.store(seq + 1, std::memory_order_release);
}

void PoolTraceDumpFor(const void* device_ptr, std::ostream& os) {
	if (!PoolTraceEnabled()) {
		os << "[trace] tracing is off (set FIDESLIB_POOL_TRACE=1 with FIDESLIB_CONCURRENT_OPS=1)\n";
		return;
	}
	// Copy under the lock and print outside it: formatting into a stream can block, and this lock
	// is taken by every pooled allocate and free.
	TraceBlockRing r;
	const void* base = nullptr;
	{
		std::lock_guard<std::mutex> guard(trace_block_lock);
		const TraceBlockRing* found = traceFindBlockLocked(device_ptr);
		if (found == nullptr) {
			os << "[trace] block " << device_ptr << ": no provenance (never pooled, or cut before tracing began)\n";
			return;
		}
		r = *found;
		// Recover the base for the header line: find it the same way, by address order.
		auto it = trace_blocks.upper_bound(device_ptr);
		if (it != trace_blocks.begin()) {
			--it;
			base = it->first;
		}
		const auto exact = trace_blocks.find(device_ptr);
		if (exact != trace_blocks.end())
			base = exact->first;
	}
	const size_t kept  = r.total < kPoolTraceBlockEvents ? static_cast<size_t>(r.total) : kPoolTraceBlockEvents;
	const size_t first = static_cast<size_t>(r.total) - kept;
	os << "[trace] block " << base << " (" << r.bytes << " B, " << r.total << " events, last " << kept << ")";
	if (base != device_ptr)
		os << " contains " << device_ptr << " at +" << (static_cast<const char*>(device_ptr) - static_cast<const char*>(base));
	os << "\n";
	os << "[trace]   last taken by lane " << traceLaneName(r.last_taken_lane) << " at seq " << r.last_taken_seq << ", last freed by lane "
	   << traceLaneName(r.last_freed_lane) << " at seq " << r.last_freed_seq;
	if (r.last_output_lane != -2)
		os << ", last declared an OUTPUT of lane " << traceLaneName(r.last_output_lane) << " at seq " << r.last_output_seq << " on stream "
		   << r.last_output_stream;
	os << "\n";
	for (size_t i = first; i < static_cast<size_t>(r.total); ++i) {
		const TraceBlockEvent& e = r.e[i % kPoolTraceBlockEvents];
		os << "[trace]   seq " << e.seq << " t " << e.t_us << "us thr " << e.thread << " lane " << traceLaneName(e.lane) << " "
		   << traceKindName(e.kind) << " stream=" << e.consumer << " owner_stream=" << e.owner << " ev=" << e.ev << "\n";
	}
}

void PoolTraceDumpRecent(std::ostream& os, size_t n) {
	if (!PoolTraceEnabled()) {
		os << "[trace] tracing is off (set FIDESLIB_POOL_TRACE=1 with FIDESLIB_CONCURRENT_OPS=1)\n";
		return;
	}
	TraceFenceEvent* const ring	 = traceFenceRing();
	const unsigned long long end = trace_seq.load(std::memory_order_acquire);
	if (n > kPoolTraceFenceEvents)
		n = kPoolTraceFenceEvents;
	const unsigned long long begin = end > n ? end - n : 0ull;
	// The window is the last `n` SEQUENCE NUMBERS, which the two rings share: the fence events in
	// it are the subset printed below, and the gaps are this block's take / free / output events,
	// which PoolTraceDumpFor prints. Saying so is what keeps "I printed 2000" from reading as
	// "there were only 2000 fences".
	os << "[trace] fence events within the last " << (end - begin) << " sequence numbers (" << end << " traced events overall)\n";
	// `sq`, not `s`: `s` is the pool's array of per-device handshake streams at file scope, and a
	// loop variable that shadows it would read as one.
	for (unsigned long long sq = begin; sq < end; ++sq) {
		const TraceFenceEvent& slot		   = ring[sq % kPoolTraceFenceEvents];
		const unsigned long long committed = slot.seq.load(std::memory_order_acquire);
		// The slot holds `seq + 1` for the event that wrote it. Anything else means this sequence
		// number went to the BLOCK ring (the two share one counter) or the slot has been lapped.
		if (committed != sq + 1)
			continue;
		os << "[trace]   seq " << sq << " t " << slot.t_us << "us thr " << slot.thread << " lane " << traceLaneName(slot.lane) << " "
		   << traceFenceName(slot.kind) << " self=" << slot.self << " other=" << slot.other << " ev=" << slot.ev
		   << (slot.early_out ? " EARLY-OUT" : "") << (slot.lock_held ? " lock=held" : " lock=none") << "\n";
	}
}

std::string PoolTraceOwnersLine(const void* device_ptr) {
	if (!PoolTraceEnabled())
		return {};
	std::lock_guard<std::mutex> guard(trace_block_lock);
	const TraceBlockRing* r = traceFindBlockLocked(device_ptr);
	if (r == nullptr)
		return {};
	std::string out = "last taken by lane " + traceLaneName(r->last_taken_lane) + " (seq " + std::to_string(r->last_taken_seq) +
					  "), last freed by lane " + traceLaneName(r->last_freed_lane) + " (seq " + std::to_string(r->last_freed_seq) + ")";
	if (r->last_output_lane != -2)
		out += ", last written as an output by lane " + traceLaneName(r->last_output_lane) + " (seq " + std::to_string(r->last_output_seq) + ")";
	return out;
}

/// The body of MemPoolStatsSnapshot, WITHOUT the lock. Split out for exactly one caller besides
/// the snapshot itself: MemPoolFailureReport, which runs on the failing thread while that thread
/// already holds `mempool_lock[id]` (the chunk cut takes it for the whole allocation). That mutex
/// is not recursive, so a report that re-took it would hang the process instead of printing the
/// one set of numbers that explains the failure. Every other caller goes through the locking
/// wrapper below and this function's precondition -- the lock is held -- is unchanged from the
/// code it was lifted out of.
static MemPoolStats memPoolStatsSnapshotLocked(const int id) {
	MemPoolStats out;

	// Ordered by size class, so the report is stable run to run and two runs diff cleanly.
	std::map<int, MemPoolSizeClassStats> by_bytes;
	const auto row = [&by_bytes](const int b) -> MemPoolSizeClassStats& {
		MemPoolSizeClassStats& r = by_bytes[b];
		r.bytes					 = b;
		return r;
	};
	for (const auto& kv : pool_chunks_cut[id])
		row(kv.first).chunks_cut = kv.second;
	for (const auto& kv : pool_drains[id])
		row(kv.first).drains = kv.second;
	for (const auto& kv : pool_blocks_reclaimed[id])
		row(kv.first).blocks_reclaimed = kv.second;
	for (const auto& kv : pool_drain_skips[id])
		row(kv.first).drain_skips = kv.second;
	for (const auto& kv : pool_boundary_blocks[id])
		row(kv.first).boundary_blocks_moved = kv.second;
	for (const auto& kv : size_to_memory_mt[id]) {
		MemPoolSizeClassStats& r = row(kv.first);
		r.fresh_blocks			 = kv.second.size();
		// The sentinel is a property of a block IN THIS TIER, so it is counted here and nowhere
		// else. It is a subset of fresh_blocks, never a separate population, and in particular it
		// is never used as an index into lane_freed_blocks below.
		for (const PooledBlock& b : kv.second)
			if (b.lane == kNoFenceLane)
				++r.fresh_no_fence_blocks;
	}

	// Per-lane totals (summed over classes) AND per-class totals (summed over lanes): the first is
	// what the report prints, the second is what makes `fresh + lane_freed == chunks * blocks-per-
	// chunk` checkable for one size class. Every block here was pushed by GPUfree, which always
	// tags with a real lane, so no sentinel can appear on these lists.
	out.lane_freed_blocks.assign(freed_mt[id].size(), 0);
	for (size_t lane = 0; lane < freed_mt[id].size(); ++lane) {
		for (const auto& kv : freed_mt[id][lane]) {
			out.lane_freed_blocks[lane] += kv.second.size();
			row(kv.first).lane_freed_blocks += kv.second.size();
		}
	}

	out.classes.reserve(by_bytes.size());
	for (const auto& kv : by_bytes) {
		out.chunks_cut += kv.second.chunks_cut;
		out.drains += kv.second.drains;
		out.blocks_reclaimed += kv.second.blocks_reclaimed;
		out.drain_skips += kv.second.drain_skips;
		out.boundary_blocks_moved += kv.second.boundary_blocks_moved;
		out.classes.push_back(kv.second);
	}
	// Per DEVICE, not per class: one release moves every class, so there is no per-class call
	// count to sum. Read here rather than in the loop above so that a device whose lanes were
	// already empty still reports the calls that found nothing.
	out.boundary_releases = pool_boundary_releases[id];
	out.boundary_skips	  = pool_boundary_skips[id];
	return out;
}

MemPoolStats MemPoolStatsSnapshot(const int id) {
	std::lock_guard<std::mutex> guard(mempool_lock[id]);
	return memPoolStatsSnapshotLocked(id);
}

/// The summary line for an ALREADY TAKEN snapshot. Split out so that MemPoolStatsLine and
/// MemPoolStatsReport share one formatter and, more to the point, so that the report can format
/// its header from the same snapshot it formats its rows from: calling MemPoolStatsLine(id) from
/// inside the report would take `mempool_lock` a SECOND time, at a later instant, and print a
/// header describing a pool the rows below it no longer match.
static std::string memPoolStatsLineFrom(const MemPoolStats& st, const int id) {
	size_t fresh = 0;
	for (const MemPoolSizeClassStats& c : st.classes)
		fresh += c.fresh_blocks;
	size_t lane_held = 0;
	for (const size_t n : st.lane_freed_blocks)
		lane_held += n;
	char buf[320];
	std::snprintf(buf, sizeof(buf),
				  "mempool dev%d: chunks %llu drains %llu reclaimed %llu drain-skips %llu fresh %zu lane-held %zu "
				  "boundary-releases %llu boundary-moved %llu boundary-skips %llu",
				  id, st.chunks_cut, st.drains, st.blocks_reclaimed, st.drain_skips, fresh, lane_held,
				  st.boundary_releases, st.boundary_blocks_moved, st.boundary_skips);
	return std::string(buf);
}

std::string MemPoolStatsLine(const int id) {
	return memPoolStatsLineFrom(MemPoolStatsSnapshot(id), id);
}

/// The full report for an ALREADY TAKEN snapshot, for the same reason memPoolStatsLineFrom exists:
/// the failure path has to format a snapshot it took under a lock it still holds, and the normal
/// path has to format one it took and released. Both get the same text out of this.
static std::string memPoolStatsReportFrom(const MemPoolStats& st, const int id) {
	std::string out = memPoolStatsLineFrom(st, id) + "\n";
	char buf[320];
	for (const MemPoolSizeClassStats& c : st.classes) {
		std::snprintf(buf, sizeof(buf),
					  "  mempool dev%d class %d B: chunks %llu drains %llu reclaimed %llu drain-skips %llu fresh %zu "
					  "(no-fence %zu) lane-held %zu boundary-moved %llu\n",
					  id, c.bytes, c.chunks_cut, c.drains, c.blocks_reclaimed, c.drain_skips, c.fresh_blocks,
					  c.fresh_no_fence_blocks, c.lane_freed_blocks, c.boundary_blocks_moved);
		out += buf;
	}
	// Only the lanes that are holding something: the cap is 32 by default and most runs leave most
	// of them empty, so printing every index would bury the one lane that matters.
	std::string lanes;
	for (size_t lane = 0; lane < st.lane_freed_blocks.size(); ++lane) {
		if (st.lane_freed_blocks[lane] == 0)
			continue;
		std::snprintf(buf, sizeof(buf), "%s%zu:%zu", lanes.empty() ? "" : " ", lane, st.lane_freed_blocks[lane]);
		lanes += buf;
	}
	std::snprintf(buf, sizeof(buf), "  mempool dev%d freed lists: %s\n", id, lanes.empty() ? "all empty" : lanes.c_str());
	out += buf;
	return out;
}

std::string MemPoolStatsReport(const int id) {
	// ONE snapshot, one lock acquisition, one instant: the line, the per-class rows and the lane
	// lengths all come from `st`.
	return memPoolStatsReportFrom(MemPoolStatsSnapshot(id), id);
}

namespace {
/// The reporter SetPoolFailureReporter last stored. A plain function pointer, read without a lock
/// on the failure path on purpose: it is written once per Context construction and once per
/// destruction, it is only ever read by a thread that is about to abort, and a mutex here would be
/// one more thing that can deadlock in the one place that must not. `std::atomic` for the store's
/// visibility, `relaxed` because there is nothing to order it against -- the pointer either
/// arrives before the failure or it does not, and both are honest reports.
std::atomic<PoolFailureReporter> pool_failure_reporter{ nullptr };
} // namespace

CudaError::CudaError(const cudaError_t error, const char* const file, const int line)
	: std::runtime_error(std::string("Cuda failure ") + file + ":" + std::to_string(line) + ": '" + cudaGetErrorString(error) + "'"),
	  error_(error), file_(file), line_(line) {
}

void CudaFailure(const cudaError_t error, const char* const file, const int line, const bool with_backtrace) {
	TableOwnerReport(); // DEBUG BRANCH ONLY: TableOwner.cuh
	if (with_backtrace) {
		void* frames[10];
		const int n = backtrace(frames, 10);
		backtrace_symbols_fd(frames, n, STDERR_FILENO);
	}
	CudaError failure(error, file, line);
	std::printf("%s\n", failure.what());
	std::fflush(stdout);
	breakpoint();
	throw failure;
}

void SetPoolFailureReporter(const PoolFailureReporter fn) {
	pool_failure_reporter.store(fn, std::memory_order_relaxed);
}

void MemPoolFailureReport(const int id_locked, const int bytes, const char* const what) {
	// Everything here is best effort. The caller is about to report the failure through
	// CudaCheckError, which throws CudaError, so an exception out of this function would replace
	// that report with a less useful one, or with std::terminate in a destructor; that is what the
	// catch-all is for.
	try {
		std::string out = "\n==== FIDESlib POOL FAILURE ====\n";
		char buf[256];
		std::snprintf(buf, sizeof(buf), "  at: %s (device %d, size class %d B)\n",
					  what == nullptr ? "(unnamed)" : what, id_locked, bytes);
		out += buf;

		// getNumDevices() is the count the library initialised, not cudaGetDeviceCount(): the pool
		// arrays are indexed by the same id space, and a device the library never touched has
		// nothing to report anyway.
		const int devices = getNumDevices();
		for (int id = 0; id < devices && id < MAXG; ++id) {
			if (id == id_locked) {
				// The caller holds this one. See memPoolStatsSnapshotLocked.
				out += memPoolStatsReportFrom(memPoolStatsSnapshotLocked(id), id);
			} else {
				out += memPoolStatsReportFrom(MemPoolStatsSnapshot(id), id);
			}
		}

		if (const PoolFailureReporter fn = pool_failure_reporter.load(std::memory_order_relaxed))
			out += fn();
		else
			out += "  (no layer-above reporter registered; see SetPoolFailureReporter)\n";

		out += "==== END POOL FAILURE ====\n";
		std::fputs(out.c_str(), stderr);
		std::fflush(stderr);
	} catch (...) {
		std::fputs("\n==== FIDESlib POOL FAILURE (report itself threw) ====\n", stderr);
		std::fflush(stderr);
	}
}

int GetTargetThreads(int id) {
	return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}

// ---- Pinned host-to-device staging ring (opt-in: FIDESLIB_PINNED_STAGING=1) ----
//
// v1 of this arena held ONE slot per device and called cudaStreamSynchronize after every copy to
// make that slot reusable. It measured no win: the synchronize serialized the upload against
// everything else the stream had queued, so whatever pinned-source bandwidth it bought was handed
// straight back as lost overlap.
//
// v2 is a ring of N slots per device (FIDESLIB_PINNED_SLOTS, default 8). Each slot carries its own
// cudaEvent, recorded on the destination limb's stream immediately after that slot's
// cudaMemcpyAsync. Acquiring a slot waits on THAT SLOT'S EVENT -- not on the stream -- so the host
// only ever blocks when it has lapped the whole ring, i.e. when the DMA engine is genuinely the
// bottleneck. Everything the stream has queued behind the copy stays free to overlap.

namespace {
struct PinnedStagingSlot {
	void* ptr		  = nullptr;
	size_t bytes	  = 0;
	cudaEvent_t event = nullptr; ///< Completion of the last H2D copy issued out of this slot.
	bool busy		  = false;	 ///< Reserved by some thread right now (see PinnedStagingUpload).
};

struct PinnedStagingArena {
	std::mutex lock;
	std::condition_variable slot_freed;
	/// Allocated exactly once, on first use, and NEVER resized: callers hold a reference to a slot
	/// across an unlock, so the backing storage must not move. A unique_ptr array makes that
	/// invariant structural rather than a comment on a std::vector.
	std::unique_ptr<PinnedStagingSlot[]> slots;
	size_t n_slots = 0;
	size_t cursor  = 0; ///< Round-robin hand: the oldest slot is the one whose event is likeliest done.
	/// Largest request seen on this device. Slots are grown to it on acquire, so the ring settles at
	/// the largest limb the caller actually asks for (N * sizeof(T), 512 KB at logN = 16) instead of
	/// thrashing between sizes. Nothing here is hardcoded.
	size_t high_water = 0;
	/// Set once a pinned allocation (or event creation) has failed on this device; every later call
	/// then declines immediately instead of hammering an exhausted pinned pool.
	bool failed = false;
};

PinnedStagingArena pinned_staging[MAXG];
/// Separate ring for the gather path: its slots are batch-sized, and `high_water` is per arena, so
/// sharing one arena would grow every limb-sized slot to the batch size for the rest of the process.
PinnedStagingArena pinned_gather[MAXG];

/// @brief The slot machinery shared by both rings: reserve a slot, wait on ITS event, (re)allocate
/// it to the arena's high-water mark, let `fill` write the host bytes into it, issue ONE H2D and
/// record the slot's event behind that copy.
///
/// Extracted verbatim from the single-source version so the two rings cannot drift. `fill` is
/// called with the slot pointer and runs OUTSIDE the arena lock, like the memcpy it replaced.
template <typename Fill>
bool StageThroughArena(PinnedStagingArena& arena, size_t slots_wanted, void* dst, size_t bytes, cudaStream_t stream, Fill&& fill) {
	size_t idx = 0;
	size_t want = 0;

	// --- reserve a slot (short critical section: no CUDA call, no memcpy, happens under the lock) ---
	{
		std::unique_lock<std::mutex> guard(arena.lock);
		if (arena.failed)
			return false;
		if (arena.slots == nullptr) {
			arena.n_slots = slots_wanted;
			arena.slots	  = std::make_unique<PinnedStagingSlot[]>(arena.n_slots);
		}
		if (bytes > arena.high_water)
			arena.high_water = bytes;
		want = arena.high_water;

		// Round-robin from the cursor, skipping slots another thread is currently filling.
		for (;;) {
			size_t found = arena.n_slots;
			for (size_t k = 0; k < arena.n_slots; ++k) {
				const size_t c = (arena.cursor + k) % arena.n_slots;
				if (!arena.slots[c].busy) {
					found = c;
					break;
				}
			}
			if (found < arena.n_slots) {
				idx = found;
				break;
			}
			// More concurrent uploaders than slots: wait for one to hand its slot back rather than
			// spinning on the ring.
			arena.slot_freed.wait(guard);
			if (arena.failed)
				return false;
		}
		arena.slots[idx].busy = true;
		arena.cursor		  = (idx + 1) % arena.n_slots;
	}

	// From here to the release below, `slot` is exclusively ours: `busy` keeps every other thread
	// off it, and the array never moves. So the wait, the (re)allocation and the copies all run
	// OUTSIDE the arena lock -- one slow uploader never blocks another from filling a different slot.
	PinnedStagingSlot& slot = arena.slots[idx];

	// The only thing that can still be reading this slot is the H2D copy we issued out of it last
	// time round the ring. Wait on that copy specifically, not on the stream: everything the stream
	// has queued behind it stays free to overlap. A never-recorded event returns immediately.
	if (slot.event != nullptr)
		cudaEventSynchronize(slot.event);

	bool ok = true;
	if (slot.bytes < want) {
		if (slot.ptr != nullptr) {
			// Safe: the event wait above guarantees no transfer is still reading these pages.
			cudaFreeHost(slot.ptr);
			slot.ptr   = nullptr;
			slot.bytes = 0;
		}
		void* ptr = nullptr;
		// Portable: the buffer must be usable no matter which device context is current when a
		// later caller reaches it (the same flag RNSPoly's timeline semaphores use).
		if (cudaHostAlloc(&ptr, want, cudaHostAllocPortable) != cudaSuccess || ptr == nullptr) {
			cudaGetLastError();
			ok = false;
		} else {
			slot.ptr   = ptr;
			slot.bytes = want;
		}
	}
	if (ok && slot.event == nullptr) {
		// DisableTiming only, matching the events LimbPartitionMGPU already creates.
		cudaEvent_t ev = nullptr;
		if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess || ev == nullptr) {
			cudaGetLastError();
			ok = false;
		} else {
			slot.event = ev;
		}
	}

	if (ok) {
		fill(slot.ptr);
		cudaMemcpyAsync(dst, slot.ptr, bytes, cudaMemcpyHostToDevice, stream);
		// Record on the DESTINATION stream, right behind its own copy: this event is what the next
		// taker of this slot waits on.
		cudaEventRecord(slot.event, stream);
	}

	{
		std::lock_guard<std::mutex> guard(arena.lock);
		slot.busy = false;
		if (!ok)
			arena.failed = true; // sticky: from now on everyone takes the pageable path
		// notify_all, not notify_one: on failure every waiter has to wake up and see `failed`.
		arena.slot_freed.notify_all();
	}
	return ok;
}

/// @brief The guard every ring entry point shares: the arena table is indexed by CUDA device
/// ordinal, and cudaEventRecord needs the event and the stream on the same device.
bool StagingDeviceUsable(int device) {
	if (device < 0 || device >= MAXG)
		return false;
	int current = -1;
	if (cudaGetDevice(&current) != cudaSuccess || current != device) {
		cudaGetLastError(); // do not leave the failure for an unrelated check to trip over
		return false;
	}
	return true;
}
} // namespace

bool PinnedStagingEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PINNED_STAGING");
		const bool on	= env != nullptr && env[0] == '1' && env[1] == '\0';
		if (on) { // DEBUG BRANCH ONLY: proves the variant took the pinned path
			printf("[pinnedstaging] ON: host-to-device uploads go through %zu pinned slots per device\n", PinnedStagingSlots());
			fflush(stdout);
		}
		return on;
	}();
	return enabled;
}

size_t PinnedStagingSlots() {
	static const size_t slots = [] {
		constexpr size_t kDefault = 8;
		constexpr size_t kMax	  = 64;
		const char* env			  = std::getenv("FIDESLIB_PINNED_SLOTS");
		if (env == nullptr || env[0] == '\0')
			return kDefault;
		char* end		   = nullptr;
		unsigned long long n = std::strtoull(env, &end, 10);
		if (end == env || *end != '\0' || n == 0)
			return kDefault;
		return static_cast<size_t>(n < kMax ? n : kMax);
	}();
	return slots;
}

bool PinnedMGPU() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PINNED_MGPU");
		const bool on  = env != nullptr && env[0] == '1' && env[1] == '\0';
		if (on) {
			printf("[pinnedmgpu] ON: the fused key switch's digit-table uploads take the pinned staging arena\n");
			fflush(stdout);
		}
		return on;
	}();
	return enabled;
}

/// The arena path with NO env gate of its own: every caller applies one first. Kept as one function
/// so UploadH2DMGPU cannot drift from PinnedStagingUpload in what it considers a valid upload.
static bool StageUploadUngated(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream) {
	if (bytes == 0 || src == nullptr || dst == nullptr)
		return false;
	// Every path into Limb<T>::load runs under cudaSetDevice(v.device) (RNSPoly::load / loadConstant
	// set it per limb; LimbPartition::loadDecompDigit sets it once at entry), and every op in
	// LimbPartitionMGPU.cu sets it at function entry, so the device check normally passes -- it is
	// here so that a future caller that forgets quietly gets the pageable path instead of an
	// event/stream device mismatch.
	if (!StagingDeviceUsable(device))
		return false;

	return StageThroughArena(pinned_staging[device], PinnedStagingSlots(), dst, bytes, stream, [&](void* slot) { std::memcpy(slot, src, bytes); });
}

/// DEBUG BRANCH ONLY: the first upload that took the pageable path although pinned staging is on.
static void PinnedStagingFellBack(const size_t bytes) {
	static std::atomic<bool> said{ false };
	if (PinnedStagingEnabled() && !said.exchange(true)) {
		printf("[pinnedstaging] FELL BACK to a pageable upload (%zu bytes)\n", bytes);
		fflush(stdout);
	}
}

__attribute__((noinline)) void UploadH2D(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream) {
	const unsigned long long enter = MemLogEnter(); // DEBUG BRANCH ONLY
	if (!PinnedStagingUpload(dst, src, bytes, device, stream)) {
		PinnedStagingFellBack(bytes);
		cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
	}
	MemLogUpload(dst, src, bytes, stream, __builtin_return_address(0), enter); // DEBUG BRANCH ONLY
}

__attribute__((noinline)) void UploadH2DMGPU(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream) {
	const unsigned long long enter = MemLogEnter(); // DEBUG BRANCH ONLY
	// EITHER variable: FIDESLIB_PINNED_STAGING for the sweep 0f72c59 meant to be whole, and
	// FIDESLIB_PINNED_MGPU for the three sites it missed, on their own. See CudaUtils.cuh.
	if (!(PinnedStagingEnabled() || PinnedMGPU()) || !StageUploadUngated(dst, src, bytes, device, stream)) {
		PinnedStagingFellBack(bytes);
		cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
	}
	MemLogUpload(dst, src, bytes, stream, __builtin_return_address(0), enter); // DEBUG BRANCH ONLY
}

bool PinnedStagingUpload(void* dst, const void* src, size_t bytes, int device, cudaStream_t stream) {
	if (!PinnedStagingEnabled())
		return false;
	return StageUploadUngated(dst, src, bytes, device, stream);
}

bool PinnedGatherEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("FIDESLIB_PINNED_GATHER");
		// Default ON: the only caller is the batched device encode, which is itself opt-in at the
		// application, so no path that runs today reaches this ring. "0" is the kill switch.
		return !(env != nullptr && env[0] == '0' && env[1] == '\0');
	}();
	return enabled;
}

size_t PinnedGatherSlots() {
	static const size_t slots = [] {
		constexpr size_t kDefault = 8;
		constexpr size_t kMax	  = 64;
		const char* env			  = std::getenv("FIDESLIB_PINNED_GATHER_SLOTS");
		if (env == nullptr || env[0] == '\0')
			return kDefault;
		char* end			 = nullptr;
		unsigned long long n = std::strtoull(env, &end, 10);
		if (end == env || *end != '\0' || n == 0)
			return kDefault;
		return static_cast<size_t>(n < kMax ? n : kMax);
	}();
	return slots;
}

bool PinnedStagingUploadGather(void* dst, const void* const* srcs, size_t n, size_t bytes_each, int device, cudaStream_t stream) {
	if (!PinnedGatherEnabled() || n == 0 || bytes_each == 0 || srcs == nullptr || dst == nullptr)
		return false;
	if (!StagingDeviceUsable(device))
		return false;
	for (size_t i = 0; i < n; ++i) {
		if (srcs[i] == nullptr)
			return false;
	}

	const size_t total = n * bytes_each;
	return StageThroughArena(pinned_gather[device], PinnedGatherSlots(), dst, total, stream, [&](void* slot) {
		char* out = static_cast<char*>(slot);
		for (size_t i = 0; i < n; ++i) {
			std::memcpy(out + i * bytes_each, srcs[i], bytes_each);
		}
	});
}

// The slots and their events are deliberately never freed: they are process-lifetime buffers, and
// calling cudaFreeHost / cudaEventDestroy from a static destructor runs straight into the
// cudaErrorCudartUnloading problem this file already documents.

} // namespace FIDESlib

// ---- DEBUG BRANCH ONLY: the wrappers of TableOwnerFences.cuh. Logged after the call, stamped with the
// log position before it (see ReuseStep). ----
__attribute__((noinline)) cudaError_t TableOwnerEventRecord(cudaEvent_t e, cudaStream_t s) {
	const unsigned long long enter = FIDESlib::MemLogEnter();
	const cudaError_t r			   = (cudaEventRecord)(e, s);
	FIDESlib::MemLogFence('R', e, s, __builtin_return_address(0), enter);
	return r;
}

__attribute__((noinline)) cudaError_t TableOwnerEventRecordWithFlags(cudaEvent_t e, cudaStream_t s, unsigned int flags) {
	const unsigned long long enter = FIDESlib::MemLogEnter();
	const cudaError_t r			   = (cudaEventRecordWithFlags)(e, s, flags);
	FIDESlib::MemLogFence('R', e, s, __builtin_return_address(0), enter);
	return r;
}

__attribute__((noinline)) cudaError_t TableOwnerStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags) {
	const unsigned long long enter = FIDESlib::MemLogEnter();
	const cudaError_t r			   = (cudaStreamWaitEvent)(s, e, flags);
	FIDESlib::MemLogFence('W', e, s, __builtin_return_address(0), enter);
	return r;
}
