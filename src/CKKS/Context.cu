//
// Created by carlosad on 2/05/24.
//
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CudaUtils.cuh"
#include "CKKS/Ciphertext.cuh"
#include <atomic>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include "CKKS/Context.cuh"
#include <mutex>
#include <source_location>

#include "../parallel_for.hpp"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

namespace FIDESlib {
extern thread_local bool gpufree_presynced;   // defined in CudaUtils.cu next to GPUfree
}

namespace FIDESlib::CKKS {

std::atomic_uint64_t next_uid = 0;
constexpr bool SPLIT_SPECIAL  = true;

std::map<Parameters, std::shared_ptr<ContextData>> map_param_context;
Context currentContext;
/// Guards `currentContext`. SetCurrentContext sits at the top of the value-type entry points
/// (Plaintext::load, Ciphertext::load, ...), so once an application drives any of those from more
/// than one host thread -- e.g. building device plaintexts on encode/upload workers -- this
/// shared_ptr is read concurrently, and written by whichever thread first installs a context.
/// Uncontended, this is a few nanoseconds against calls that issue CUDA work.
std::mutex currentContextLock;

// std::map<std::pair<Parameters, Parameters>, std::shared_ptr<std::map<KeyHash, KeySwitchingKey>>> map_param_switch;
std::vector<std::pair<std::pair<Parameters, Parameters>, std::unique_ptr<std::map<KeyHash, KeySwitchingKey>>>> map_param_switch;

/* Communicate internally if ContextData created succesfully, on unsuccessful creation,
 * its param field contains a "normalized" version for caching */
bool OK = false;

namespace {
/// The context the device-memory pool's FAILURE PATH asks about its auxiliary-polynomial lists.
///
/// The pool (CudaUtils) is below CKKS in this library's layering and cannot name a ContextData, so
/// it takes a plain `std::string(*)()` and this is the state that function needs. Last constructed
/// context wins, and ~ContextData clears it; two live contexts at once is a debugging
/// configuration, not a deployed one, and a pointer with last-writer-wins is the smallest thing
/// that cannot itself fail on the path it is reporting from. Never dereferenced except from
/// MemPoolFailureReport, i.e. only on a thread that is about to exit.
std::atomic<ContextData*> pool_failure_context{ nullptr };

std::string poolFailureAuxReport() {
	ContextData* const cc = pool_failure_context.load(std::memory_order_relaxed);
	if (cc == nullptr)
		return "  (auxpoly: no context registered)\n";
	return cc->auxPolyStatsReportTry();
}
} // namespace

ContextData::ContextData(const Parameters& param_, const std::vector<int>& devs, const int secBits)
: my_range(loc, LIFETIME), param((CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), param_)), precom(), logN(param.logN), N(1 << logN),
  rescaleTechnique(translateRescalingTechnique(param.scalingTechnique)), L(param.L), logQ(computeLogQ(L, param.primes)), batch(param.batch), GPUid(devs),
  dnum((validateDnum(GPUid, param.dnum) /*, param.dnum*/)), GPUdigits(generateGPUdigits(dnum, GPUid)), prime((param.primes.resize(L + 1), param.primes)),
  meta{ generateMeta(GPUid, dnum, GPUdigits, prime, param) }, logQ_d(computeLogQ_d(dnum, meta, prime)), K(computeK(logQ_d, param.Sprimes, param)),
  logP(computeLogQ(K - 1, param.Sprimes)), specialPrime((param.Sprimes.resize(K), param.Sprimes)), specialMeta(generateSpecialMeta(meta, specialPrime, L + 1, GPUid)),
  splitSpecialMeta(generateSplitSpecialMeta(specialMeta.at(0), GPUid)), decompMeta(generateDecompMeta(meta, GPUdigits, GPUid, L)),
  digitMeta(generateDigitMeta(meta, splitSpecialMeta, specialMeta.at(0), GPUdigits, GPUid)), gatherMeta(generateGatherMeta(meta, L)),
  limbGPUid(generateLimbGPUid(meta, L, splitSpecialMeta, K)), digitGPUid(generateDigitGPUid(meta, L, dnum)), GPUrank(GPUid.size())
// top_limb(devs.size())
{
#ifndef NCCL
	if (GPUid.size() > 1) {
		std::cerr << "MGPU requested but no NCCL linked, aborting" << std::endl;
		exit(-1);
	}
#endif

	if (map_param_context.contains(param)) {
		OK = false;
		return;
	}

	// auto& constants = precom.constants;
	// auto& globals = precom.globals;
	auto [constants, globals] = SetupConstants<Parameters>(prime, meta, specialPrime, specialMeta.at(0), decompMeta, digitMeta, GPUdigits, GPUid, N, param);

	precom.constants = constants;
	precom.globals   = std::move(globals);

	// One slot per thread that may issue ops concurrently. Sized once, before any op can run, and
	// never resized, so indexing it later needs no lock; every entry starts null and is
	// materialized by its own thread on first use (ContextData::opScratch). Slot 0 is this
	// object's own scratch members, so entry 0 stays null forever -- and in the default mode there
	// is only entry 0 and nothing here is ever allocated.
	op_scratch.resize(ConcurrentOps() ? ScratchSlotCap() : 1);

	// From here on, a device allocation that fails anywhere in the process can print THIS context's
	// auxiliary-polynomial lists next to the pool's own numbers. Registered after op_scratch is
	// sized, because that is what poolFailureAuxReport walks. Costs one relaxed store per context
	// construction and nothing at all on any successful allocation.
	pool_failure_context.store(this, std::memory_order_relaxed);
	SetPoolFailureReporter(&poolFailureAuxReport);

	PrepareNCCLCommunication();

	// CheckBitSecurity();
	int bits = 0;
	for (auto& j : { prime, specialPrime })
		for (auto& i : j)
			bits += i.bits;

	for (int dev : GPUid) {
		cudaSetDevice(dev);
		cudaMemPool_t mp;
		cudaDeviceGetDefaultMemPool(&mp, dev);
		uint64_t threshold = UINT64_MAX; // 5l * 1024l * 1024l * 1024l;  // One Gigabyte of memory
		cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &threshold);
		CudaCheckErrorModNoSync;
	}

	OK = true;
	CudaNvtxStop();
}

std::vector<dim3>
ContextData::generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L, const std::vector<std::vector<LimbRecord>>& SPECIALmeta, const int K) {
	std::vector<dim3> res(L + 1 + K, 0);
	for (int i = 0; i < static_cast<int>(meta.size()); ++i) {
		for (size_t j = 0; j < meta.at(i).size(); ++j) {
			res.at(meta[i][j].id) = { static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0 };
		}
	}

	for (int i = 0; i < static_cast<int>(SPECIALmeta.size()); ++i) {
		for (size_t j = 0; j < SPECIALmeta.at(i).size(); ++j) {
			res.at(SPECIALmeta[i][j].id) = { static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0 };
		}
	}
	return res;
}

std::vector<std::vector<std::vector<LimbRecord>>> ContextData::generateDigitMeta(const std::vector<std::vector<LimbRecord>>& meta,
  const std::vector<std::vector<LimbRecord>>& splitSpecialMeta,
  const std::vector<LimbRecord>& specialMeta,
  const std::vector<std::vector<int>>& digitGPUid,
  const std::vector<int>& GPUid) {
	std::vector<std::vector<std::vector<LimbRecord>>> digitMeta(meta.size());

	for (size_t i = 0; i < digitGPUid.size(); ++i) {
		cudaSetDevice(GPUid[i]);
		for (int d : digitGPUid.at(i)) {
			digitMeta[i].emplace_back();

			if constexpr (SPLIT_SPECIAL) {
				for (auto& l : splitSpecialMeta.at(i)) {
					digitMeta[i].back().emplace_back(LimbRecord{ .id = l.id, .type = l.type, .digit = l.digit });
					digitMeta[i].back().back().stream.init();
				}
			} else {
				for (auto& l : specialMeta) {
					digitMeta[i].back().emplace_back(LimbRecord{ .id = l.id, .type = l.type, .digit = l.digit });
					digitMeta[i].back().back().stream.init();
				}
			}

			for (auto& l : meta.at(i)) {
				if (l.digit != d) {
					digitMeta[i].back().emplace_back(LimbRecord{ .id = l.id, .type = l.type, .digit = l.digit });
					digitMeta[i].back().back().stream.init();
				}
			}

			/*
			std::sort(
				digitMeta[i].back().begin() + specialMeta.size(), digitMeta[i].back().end(),
				[](LimbRecord& a, LimbRecord& b) { return a.digit < b.digit || (a.digit == b.digit && a.id < b.id); });
			*/
		}
	}
	return digitMeta;
}

std::vector<std::vector<std::vector<LimbRecord>>>
ContextData::generateDecompMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                const std::vector<std::vector<int>> digitGPUid,
                                const std::vector<int>& GPUid,
                                int L) {
	std::vector<std::vector<std::vector<LimbRecord>>> decompMeta(meta.size());

	for (size_t i = 0; i < digitGPUid.size(); ++i) {
		cudaSetDevice(GPUid[i]);
		for (int d : digitGPUid.at(i)) {
			decompMeta[i].emplace_back();

			for (int primeid = 0; primeid <= L; ++primeid) {
				for (auto& m : meta) {
					for (auto& l : m) {
						if (l.id == primeid && l.digit == d) {
							decompMeta[i].back().push_back(LimbRecord{ .id = l.id, .type = l.type, .digit = l.digit });
							decompMeta[i].back().back().stream.init();
						}
					}
				}
			}
		}
	}

	return decompMeta;
}

bool ContextData::isValidPrimeId(const int i) const {
	return (i >= 0 && i < L + 1 + K);
}

int ContextData::computeLogQ(const int L, std::vector<PrimeRecord>& primes) {
	int res = 0;
	assert(L <= (int)primes.size());
	for (int i = 0; i <= L; ++i) {
		res += (primes[i].bits == -1) ? (primes[i].bits = (int)std::bit_width(primes[i].p)) : primes[i].bits;
	}
	return res;
}

// A dnum ABOVE MAXD SILENTLY CORRUPTS THE STACK, so refuse it here.
//
// dnum is the number of hybrid key-switching digits, and it is the first index
// into the [MAXD][MAXP] arrays of `Constants` -- primeid_digit_from,
// primeid_digit_to, pos_in_digit, num_primeid_digit_from/to -- written via
// digitGPUid, whose values run 0..dnum-1 (generateDigitGPUid). `Constants
// host_constants` is a STACK LOCAL in SetupConstants (ConstantsGPU.cu), so
// dnum > MAXD writes past those rows and over whatever follows on the stack.
//
// Measured on an RTX PRO 6000 at N=2^16, L=26, with MAXD = 8:
//   dnum 6, 7  PASS
//   dnum 9     one row over -> adjacent member corrupted -> the device reads a
//              garbage primeid and faults: "Invalid __global__ read" in
//              Scalar_mult_, 24 bytes past the scalar table
//   dnum 13    five rows over -> past the struct -> "*** stack smashing
//              detected ***"
//
// This function previously returned `dnum` unchecked, so neither failure named
// its cause. Raising MAXD raises the ceiling; it does not remove the need for
// the check.
const int& ContextData::validateDnum(const std::vector<int>& GPUid, const int& dnum) {
	if (dnum < 1 || dnum > MAXD) {
		throw std::invalid_argument(
		  "FIDESlib: dnum = " + std::to_string(dnum) + " is out of range; it indexes the [MAXD][MAXP] digit arrays of Constants and MAXD = " +
		  std::to_string(MAXD) + ". Choose dnum in [1, " + std::to_string(MAXD) + "], or raise MAXD in src/ConstantsGPU.cuh (bounded by the 64 KB constant-memory budget -- see the static_assert there).");
	}
	return dnum;
}

int findDigitOnParam(const Parameters& param, uint64_t modulus) {
	for (size_t i = 0; i < param.raw->PARTITIONmoduli.size(); ++i) {
		for (uint64_t j : param.raw->PARTITIONmoduli.at(i)) {
			if (modulus == j)
				return i;
		}
	}
	return -1;
}

std::vector<std::vector<LimbRecord>>
ContextData::generateMeta(const std::vector<int>& GPUid,
                          const int dnum,
                          const std::vector<std::vector<int>> digitGPUid,
                          const std::vector<PrimeRecord>& prime,
                          const Parameters& param) {
	int devs = GPUid.size();
	std::vector<std::vector<LimbRecord>> meta(devs);

	// for (int i = 0; i < devs; ++i) {
	//  cudaSetDevice(GPUid.at(i));
	//  meta.at(i).resize((prime.size() + devs - i - 1) / devs);
	// }

	if constexpr (0) {
		int threshhold1 = (prime.size() / 2 + devs - 1) / devs;
		int threshhold2 = threshhold1 * devs;
		int dev         = 0;
		for (int i = 0; i < (int)prime.size(); ++i) {
			int digit_ = !param.raw ? i % dnum : findDigitOnParam(param, prime.at(i).p);

			if (i < threshhold1) {
			} else if (i < threshhold2) {
				dev = (dev + 1) % devs;
				if (dev == 0)
					dev = (dev + 1) % devs;
			} else {
				dev = (dev + 1) % devs;
			}
			/*{
			int dev = -1;
			for (size_t j = 0; j < digitGPUid.size(); ++j) {
				for (auto& k : digitGPUid.at(j))
					if (k == digit)
						dev = j;
			}
		}*/

			cudaSetDevice(GPUid[dev]);

			meta[dev].push_back(LimbRecord{ .id = i, .type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64)), .digit = digit_ });
			meta[dev].back().stream.init();
			// std::cout << "i: " << i << " gpu:" << dev << std::endl;
		}
	} else {
		for (int i = 0; i < (int)prime.size(); ++i) {
			int digit_ = !param.raw ? i % dnum : findDigitOnParam(param, prime.at(i).p);

			int dev = i % GPUid.size();
			/*{
			int dev = -1;
			for (size_t j = 0; j < digitGPUid.size(); ++j) {
				for (auto& k : digitGPUid.at(j))
					if (k == digit)
						dev = j;
			}
		}*/
			cudaSetDevice(GPUid[dev]);

			meta[dev].push_back(LimbRecord{ .id = i, .type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64)), .digit = digit_ });
			meta[dev].back().stream.init();
		}
	}

	return meta;
}

std::vector<int> ContextData::computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta, const std::vector<PrimeRecord>& prime) {
	std::vector<int> logQ_d(dnum, 0);

	for (auto& i : meta)
		for (auto& j : i)
			logQ_d.at(j.digit) += prime.at(j.id).bits;

	return logQ_d;
}

const int& ContextData::computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, Parameters& param) {

	size_t res  = 0;
	int logMaxD = *std::max_element(logQ_d.begin(), logQ_d.end());
	int bits    = 0;
	for (; bits < logMaxD && res < Sprimes.size(); ++res) {
		bits += (Sprimes.at(res).bits <= 0) ? (Sprimes.at(res).bits = (int)std::bit_width(Sprimes.at(res).p)) - 1 : Sprimes.at(res).bits - 1;
	}

	if (param.K != -1) {
		return param.K;
	}

	assert(bits >= logMaxD);
	return param.K = res;
}

std::vector<std::vector<LimbRecord>>
ContextData::generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                 const std::vector<PrimeRecord>& specialPrime,
                                 const int ID0,
                                 const std::vector<int>& GPUid) {
	std::vector<std::vector<LimbRecord>> specialMeta(GPUid.size());

	for (size_t d = 0; d < GPUid.size(); ++d) {
		specialMeta.at(d).resize(specialPrime.size());
		cudaSetDevice(GPUid[d]);
		for (int i = 0; i < (int)specialPrime.size(); ++i) {
			specialMeta.at(d).at(i).id   = ID0 + i;
			specialMeta.at(d).at(i).type = (specialPrime[i].type ? *(specialPrime[i].type) : (specialPrime[i].bits <= 30 ? U32 : U64));
			specialMeta.at(d).at(i).stream.init();
		}
	}

	return specialMeta;
}

std::vector<LimbRecord> ContextData::generateGatherMeta(const std::vector<std::vector<LimbRecord>>& meta, int L) {
	std::vector<LimbRecord> gatherMeta(L + 1);

	for (int i = 0; i <= L; ++i) {
		for (size_t j = 0; j < meta.size(); ++j) {
			for (size_t k = 0; k < meta.at(j).size(); ++k) {
				if (meta[j][k].id == i) {
					gatherMeta.at(i).id    = i;
					gatherMeta.at(i).digit = meta[j][k].digit;
					gatherMeta.at(i).type  = meta[j][k].type;
				}
			}
		}
	}

	return gatherMeta;
}

std::vector<std::vector<int>> ContextData::generateGPUdigits(const int dnum, const std::vector<int>& devs) {
	std::vector<std::vector<int>> res(devs.size());
	for (int d = 0; d < dnum; ++d) {
		for (uint32_t gpu = 0; gpu < devs.size(); ++gpu) {
			res[gpu].push_back(d);
		}
	}
	return res;
}

ContextData::OpScratch& ContextData::opScratch(const int slot) {
	// op_scratch was sized to ScratchSlotCap() in the constructor and is never resized, so this is
	// a plain indexed read. A slot index names ONE thread (ScratchSlot hands each thread its own,
	// and threads past the cap fall back to slot 0, which lives in the members below and never
	// reaches here), so entry `slot` is read and written by that thread alone -- no lock, and no
	// lock on the hot path either.
	assert(slot > 0 && static_cast<size_t>(slot) < op_scratch.size());
	if (op_scratch[slot] == nullptr) {
		op_scratch[slot] = std::make_unique<OpScratch>();
		// DIAG (R4 candidate C1, FIDESLIB_SLOT_MEMPOOL): this slot's private driver memory pool, one
		// per device this context uses, built with the rest of the slot's scratch so the creation is
		// not paid inside a hot op. A NO-OP unless the flag is set, and slot 0 never reaches here --
		// it keeps the device default pool, which is what makes the default mode byte-identical.
		// See CudaUtils.cuh SlotMemPool().
		for (const int dev : GPUid)
			FIDESlib::SlotMemPoolFor(dev, slot);
	}
	return *op_scratch[slot];
}

RNSPoly& ContextData::getKeySwitchAux() {
	const int slot				= ScratchSlot();
	std::unique_ptr<RNSPoly>& p = slot == 0 ? key_switch_aux : opScratch(slot).key_switch_aux;
	if (p == nullptr)
		p = std::make_unique<RNSPoly>(*this, L, false);

	p->generateDecompAndDigit(false);
	p->generateSpecialLimbs(false, false);
	return *p;
}

RNSPoly& ContextData::getKeySwitchAux2() {
	const int slot				= ScratchSlot();
	std::unique_ptr<RNSPoly>& p = slot == 0 ? key_switch_aux2 : opScratch(slot).key_switch_aux2;
	if (p == nullptr)
		p = std::make_unique<RNSPoly>(*this, L, false);
	p->generateDecompAndDigit(false);
	p->generateSpecialLimbs(false, false);
	return *p;
}

RNSPoly& ContextData::getModdownAux(const int num) {
	const int slot					   = ScratchSlot();
	const size_t i					   = num % moddown_aux.size();
	std::array<std::unique_ptr<RNSPoly>, 2>& set = slot == 0 ? moddown_aux : opScratch(slot).moddown_aux;
	if (set[i] == nullptr)
		set[i] = std::make_unique<RNSPoly>(*this, L, false);
	set[i]->generateSpecialLimbs(false, true);
	return *set[i];
}

std::map<int, RNSPoly>& ContextData::getMonomialCache() {
	const int slot = ScratchSlot();
	return slot == 0 ? precom.monomialCache : opScratch(slot).monomialCache;
}

/// WHY MULTI-GPU KEEPS THE CONTEXT STREAMS. The MGPU key-switch runs one OMP thread PER DEVICE
/// (RNSPoly.cpp: `#pragma omp parallel num_threads(GPU.size())` around modup_ksk_moddown_mgpu /
/// modupMGPU), and those threads fence ACROSS devices: device `id`'s thread waits on device `i`'s
/// stream (`getDigitStream(d, i)`, `getDigitStream2(0, i)`). Each OMP worker is a different host
/// thread and so holds a different slot, so a per-slot stream there would have the waiter fence on
/// a stream nobody launched on -- a dropped dependency, which is worse than the sharing this split
/// removes. With more than one GPU the whole op therefore stays on the context's streams, exactly
/// as today. With one GPU (every cross-device loop body is then unreachable) the op is issued
/// entirely by the calling thread, so its slot names one consistent set of streams from the first
/// launch to the last fence.
bool ContextData::perSlotDigitStreams() const {
	return GPUid.size() == 1;
}

Stream& ContextData::getDigitStream(const int d, const int id) {
	const int slot = ScratchSlot();
	if (slot == 0 || !perSlotDigitStreams())
		return digitStream.at(d).at(id);
	OpScratch& sc = opScratch(slot);
	if (sc.digitStream.empty()) {
		int prev = -1;
		cudaGetDevice(&prev);
		sc.digitStream.resize(dnum);
		for (int i = 0; i < dnum; ++i) {
			sc.digitStream[i].resize(GPUid.size());
			for (size_t j = 0; j < GPUid.size(); ++j) {
				cudaSetDevice(GPUid[j]);
				sc.digitStream[i][j].init(100);
			}
		}
		if (prev >= 0)
			cudaSetDevice(prev);
	}
	return sc.digitStream.at(d).at(id);
}

/// The twin of the above for `digitStream2`, the stream the MGPU key switch LAUNCHES on
/// (`fusedDotKSK_2_`, the post-mod-up limb NTTs, and the whole of moddownMGPU). Same shape, same
/// laziness, and the same `init()` -- DEFAULT priority, matching the context's own digitStream2
/// (the constructor calls `.init()` there and `.init(100)` for digitStream), so a slot's streams
/// schedule against each other exactly as the context's do.
Stream& ContextData::getDigitStream2(const int d, const int id) {
	const int slot = ScratchSlot();
	if (slot == 0 || !perSlotDigitStreams())
		return digitStream2.at(d).at(id);
	OpScratch& sc = opScratch(slot);
	if (sc.digitStream2.empty()) {
		int prev = -1;
		cudaGetDevice(&prev);
		sc.digitStream2.resize(dnum);
		for (int i = 0; i < dnum; ++i) {
			sc.digitStream2[i].resize(GPUid.size());
			for (size_t j = 0; j < GPUid.size(); ++j) {
				cudaSetDevice(GPUid[j]);
				sc.digitStream2[i][j].init();
			}
		}
		if (prev >= 0)
			cudaSetDevice(prev);
	}
	return sc.digitStream2.at(d).at(id);
}

void ContextData::ensureTopLimb(OpScratch& sc) {
	if (!sc.top_limb_stream.empty())
		return;
	// Mirror the constructor's layout exactly (see the top_limb_* setup in the constructor):
	// 2 x GPUid.size() buffers and pointer tables per set, entry i on device i % GPUid.size(),
	// streams initialised once per device. Plain cudaMalloc: a slot's scratch is single-device
	// scratch for one issuing thread (the NCCL registration of the context set is for the
	// multi-GPU peer copies, which the concurrent mode does not extend to).
	int prev = -1;
	cudaGetDevice(&prev);
	const size_t G = GPUid.size();
	const size_t n = 2 * G;
	sc.top_limb_stream.resize(n);
	sc.top_limb_stream2.resize(n);
	sc.top_limb_buffer.resize(n, nullptr);
	sc.top_limb_buffer2.resize(n, nullptr);
	sc.top_limbptr.reserve(n);
	sc.top_limbptr2.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		const size_t g = i % G;
		cudaSetDevice(GPUid[g]);
		if (i < G) {
			sc.top_limb_stream[g].init(100);
			sc.top_limb_stream2[g].init(100);
		}
		cudaMalloc((void**)&sc.top_limb_buffer[i], sizeof(uint64_t) * N);
		cudaMalloc((void**)&sc.top_limb_buffer2[i], sizeof(uint64_t) * N);
		sc.top_limbptr.emplace_back(sc.top_limb_stream[g], 1, GPUid[g], (void**)&sc.top_limb_buffer[i]);
		sc.top_limbptr2.emplace_back(sc.top_limb_stream2[g], 1, GPUid[g], (void**)&sc.top_limb_buffer2[i]);
	}
	if (prev >= 0)
		cudaSetDevice(prev);
}

Stream& ContextData::getTopLimbStream(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limb_stream.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limb_stream.at(id);
}

VectorGPU<void*>& ContextData::getTopLimbPtr(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limbptr.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limbptr.at(id);
}

Stream& ContextData::getTopLimbStream2(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limb_stream2.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limb_stream2.at(id);
}

uint64_t* ContextData::getTopLimbBuffer(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limb_buffer.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limb_buffer.at(id);
}

uint64_t* ContextData::getTopLimbBuffer2(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limb_buffer2.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limb_buffer2.at(id);
}

VectorGPU<void*>& ContextData::getTopLimbPtr2(const int id) {
	const int slot = ScratchSlot();
	if (slot == 0)
		return top_limbptr2.at(id);
	OpScratch& sc = opScratch(slot);
	ensureTopLimb(sc);
	return sc.top_limbptr2.at(id);
}

void ContextData::destroyOpScratch() {
	std::lock_guard<std::mutex> lk(op_scratch_lock);
	for (auto& up : op_scratch) {
		if (up == nullptr)
			continue;
		OpScratch& sc = *up;
		sc.key_switch_aux.reset(nullptr);
		sc.key_switch_aux2.reset(nullptr);
		for (auto& i : sc.moddown_aux)
			i.reset(nullptr);
		sc.monomialCache.clear();
		sc.auxPoly.clear();
		for (size_t i = 0; i < sc.top_limbptr.size(); ++i) {
			const size_t g = i % GPUid.size();
			cudaSetDevice(GPUid[g]);
			sc.top_limbptr[i].free(sc.top_limb_stream[g]);
			cudaFree(sc.top_limb_buffer[i]);
		}
		for (size_t i = 0; i < sc.top_limbptr2.size(); ++i) {
			const size_t g = i % GPUid.size();
			cudaSetDevice(GPUid[g]);
			sc.top_limbptr2[i].free(sc.top_limb_stream2[g]);
			cudaFree(sc.top_limb_buffer2[i]);
		}
		sc.top_limbptr.clear();
		sc.top_limbptr2.clear();
		sc.top_limb_buffer.clear();
		sc.top_limb_buffer2.clear();
		sc.digitStream.clear();
		sc.digitStream2.clear();
		sc.top_limb_stream.clear();
		sc.top_limb_stream2.clear();
		up.reset(nullptr);
	}
	// DIAG (FIDESLIB_SLOT_MEMPOOL): the slots' private memory pools go with the slots. Idempotent
	// and a no-op with the flag unset. The registry is process-wide rather than per-context, so an
	// application that tore ONE context down while another still issued ops on the same slots would
	// take its pools away -- not a case this library has (a context owns the slot registry's whole
	// lifetime in every run that sets the flag), and the flag is a diagnostic.
	FIDESlib::DestroySlotMemPools();
}

std::vector<uint64_t> ContextData::ElemForEvalMult(int level, const double operand, int level_in) {

	uint32_t numTowers = level + 1;
	std::vector<lbcrypto::DCRTPoly::Integer> moduli(numTowers);
	for (uint32_t i = 0; i < numTowers; i++) {
		if (i < prime.size()) {
			moduli[i] = prime[i].p;
		} else {
			moduli[i] = specialPrime[i - prime.size()].p;
		}
	}

	double scFactor;
	if (level_in == -1 || level_in == level) {
		if (static_cast<size_t>(level) < param.ScalingFactorReal.size()) {
			scFactor = param.ScalingFactorReal[level];
		} else {
			scFactor = moduli.back().ConvertToDouble();
		}
	} else {
		/** Lets handle scale changes more efficiently!*/
		assert(level > 0);
		double scFactorIn      = param.ScalingFactorReal[level_in];
		double scFactorOut     = param.ScalingFactorReal[level - 1];
		double rescalingFactor = param.ModReduceFactor[level];
		scFactor               = scFactorOut * rescalingFactor / scFactorIn;

		assert(abs(param.ScalingFactorReal[level - 1] * rescalingFactor - param.ScalingFactorReal[level] * param.ScalingFactorReal[level]) < 1e-9);
		assert(abs(scFactorIn * scFactor / rescalingFactor - scFactorOut) < 1e-9);
	}

	typedef int128_t DoubleInteger;
	int32_t MAX_BITS_IN_WORD_LOCAL = 125;

	int32_t logApprox = 0;
	const double res  = std::fabs(operand * scFactor);
	if (res > 0) {
		int32_t logSF    = static_cast<int32_t>(std::ceil(std::log2(res)));
		int32_t logValid = (logSF <= MAX_BITS_IN_WORD_LOCAL) ? logSF : MAX_BITS_IN_WORD_LOCAL;
		logApprox        = logSF - logValid;
	}
	double approxFactor = pow(2, logApprox);

	DoubleInteger large     = static_cast<DoubleInteger>(operand / approxFactor * scFactor + 0.5);
	DoubleInteger large_abs = (large < 0 ? -large : large);
	DoubleInteger bound     = (uint64_t)1 << 63;

	std::vector<lbcrypto::DCRTPoly::Integer> factors(numTowers);

	if (large_abs > bound) {
		for (uint32_t i = 0; i < numTowers; i++) {
			DoubleInteger reduced = large % moduli[i].ConvertToInt();

			factors[i] = (reduced < 0) ? static_cast<uint64_t>(reduced + moduli[i].ConvertToInt()) : static_cast<uint64_t>(reduced);
		}
	} else {
		int64_t scConstant = static_cast<int64_t>(large);
		for (uint32_t i = 0; i < numTowers; i++) {
			int64_t reduced = scConstant % static_cast<int64_t>(moduli[i].ConvertToInt());

			factors[i] = (reduced < 0) ? reduced + moduli[i].ConvertToInt() : reduced;
		}
	}

	// Scale back up by approxFactor within the CRT multiplications.
	if (logApprox > 0) {
		int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP) ? logApprox : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
		lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
		std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(numTowers, intStep);
		logApprox -= logStep;

		while (logApprox > 0) {
			int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP) ? logApprox : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
			lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
			std::vector<lbcrypto::DCRTPoly::Integer> crtSF(numTowers, intStep);
			crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF, moduli);
			logApprox -= logStep;
		}
		factors = lbcrypto::CKKSPackedEncoding::CRTMult(factors, crtApprox, moduli);
	}

	std::vector<uint64_t> result(numTowers);
	for (uint32_t i = 0; i < result.size(); ++i) {
		result[i] = factors[i].ConvertToInt();
		if (i < prime.size()) {
			result[i] = result[i] % prime[i].p;
		} else {
			result[i] = result[i] % specialPrime[i - prime.size()].p;
		}
	}

	return result;
}

std::ostream& operator<<(std::ostream& o, const uint128_t& x) {
	if (x == std::numeric_limits<uint128_t>::min())
		return o << "0";
	if (x < 10)
		return o << (char)(x + '0');
	return o << x / 10 << (char)(x % 10 + '0');
}

std::vector<uint64_t> ContextData::ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg) {
	uint32_t sizeQl = level + 1;
	std::vector<lbcrypto::DCRTPoly::Integer> moduli(sizeQl);
	for (uint32_t i = 0; i < sizeQl; i++) {
		moduli[i] = prime[i].p;
	}

	// double scFactor = param.ScalingFactorReal.at(level);
	double scFactor = 0;
	if (this->rescaleTechnique == FLEXIBLEAUTOEXT && level == L) {
		scFactor = param.ScalingFactorRealBig.at(level); // cryptoParams->GetScalingFactorRealBig(ciphertext->GetLevel());
	} else {
		scFactor = param.ScalingFactorReal.at(level); // cryptoParams->GetScalingFactorReal(ciphertext->GetLevel());
	}

	int32_t logApprox = 0;
	const double res  = std::fabs(operand * scFactor);
	if (res > 0) {
		int32_t logSF    = static_cast<int32_t>(std::ceil(std::log2(res)));
		int32_t logValid = (logSF <= lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD) ? logSF : lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD;
		logApprox        = logSF - logValid;
	}
	double approxFactor = pow(2, logApprox);

	lbcrypto::DCRTPoly::Integer scConstant = static_cast<uint64_t>(operand * scFactor / approxFactor + 0.5);
	std::vector<lbcrypto::DCRTPoly::Integer> crtConstant(sizeQl, scConstant);

	// Scale back up by approxFactor within the CRT multiplications.
	if (logApprox > 0) {
		int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP) ? logApprox : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
		lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
		std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(sizeQl, intStep);
		logApprox -= logStep;

		while (logApprox > 0) {
			int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP) ? logApprox : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
			lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
			std::vector<lbcrypto::DCRTPoly::Integer> crtSF(sizeQl, intStep);
			crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF, moduli);
			logApprox -= logStep;
		}
		crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtApprox, moduli);
	}

	// In FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
	// so we return the value before taking the depth into account.
	if (this->rescaleTechnique == FLEXIBLEAUTOEXT && level == L) {
		std::vector<uint128_t> result(sizeQl);
		for (uint32_t i = 0; i < result.size(); ++i) {
			result[i] = crtConstant[i].ConvertToInt<uint128_t>();
		}

		for (uint32_t i = 0; i < result.size(); ++i) {
			result[i] = result[i] % prime[i].p;
		}

		std::vector<uint64_t> result2(crtConstant.size());
		for (uint32_t i = 0; i < result.size(); ++i) {
			result2[i] = result[i];
		}

		return result2;
	}

	lbcrypto::DCRTPoly::Integer intScFactor = static_cast<uint64_t>(scFactor + 0.5);
	std::vector<lbcrypto::DCRTPoly::Integer> crtScFactor(sizeQl, intScFactor);

	for (uint32_t i = 1; i < static_cast<uint32_t>(noise_deg); i++) {
		crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtScFactor, moduli);
	}

	std::vector<uint128_t> result(sizeQl);
	for (uint32_t i = 0; i < result.size(); ++i) {
		result[i] = crtConstant[i].ConvertToInt<uint128_t>();
	}

	for (uint32_t i = 0; i < result.size(); ++i) {
		result[i] = result[i] % prime[i].p;
	}

	std::vector<uint64_t> result2(crtConstant.size());
	for (uint32_t i = 0; i < result.size(); ++i) {
		result2[i] = result[i];
	}

	return result2;
}

std::vector<double>& ContextData::GetCoeffsChebyshev() {
	assert(param.raw);
	return param.raw->coefficientsCheby;
}

int ContextData::GetDoubleAngleIts() {
	assert(param.raw);
	return param.raw ? param.raw->doubleAngleIts : 3;
}

int ContextData::GetBootK() {
	assert(param.raw);
	return param.raw ? param.raw->bootK : 1;
}

bool ContextData::HasBootPrecomputation(int slots) {

	return HasBootPrecomputation(slots, 0);
}

bool ContextData::HasBootPrecomputation(int slots, int levelsToDrop) {

	return precom.boot.contains({ slots, levelsToDrop });
}

BootstrapPrecomputation& ContextData::GetBootPrecomputation(int slots) {
	return GetBootPrecomputation(slots, 0);
}

BootstrapPrecomputation& ContextData::GetBootPrecomputation(int slots, int levelsToDrop) {
	// .at(), not operator[]: every bootstrap reads this map and several may be in flight at once
	// (FIDESLIB_CONCURRENT_OPS), and operator[] is a potential INSERTION -- a tree mutation on a
	// path that must be a pure lookup. The precomputation is installed once, before any op, so a
	// missing key is a programming error and throwing names it instead of silently default-
	// constructing an empty precomputation behind an assert that is a no-op in Release.
	const auto it = precom.boot.find({ slots, levelsToDrop });
	if (it == precom.boot.end())
		throw std::runtime_error("FIDESlib: no bootstrap precomputation for " + std::to_string(slots) + " slots" +
		  (levelsToDrop ? " at " + std::to_string(levelsToDrop) + " levels below the chain top" : ""));
	return it->second;
}

KeySwitchingKey& ContextData::GetRotationKey(int index, const KeyHash& keyID) {

	if (!precom.keys.at(keyID).rot_keys.contains(index)) {
		throw std::runtime_error("Rotation index " + std::to_string(index) + " not found");
	}

	return precom.keys.at(keyID).rot_keys.at(index);
}

KeySwitchingKey& ContextData::GetRotationKey(int index, const KeyHash& keyID, int slots, int& actual_index) {
	if (index != 2 * N - 1) {
		// Handle conjugate key independently
		index = index % (N / 2);
		if (index < 0)
			index += this->N / 2;
		// if (index > slots / 2)
		//	index += N / 2 - slots;

		if (!precom.keys.at(keyID).rot_keys.contains(index)) {
			if (slots != -1 && slots != N / 2) {
				// std::cout << "Looking for alternative key modulo-slot compatible to " << index << std::endl;

				for (int i = 1; i < N / 2 / slots; ++i) {
					int index_ = (index + i * slots) % (N / 2);
					if (precom.keys.at(keyID).rot_keys.contains(index_)) {
						actual_index = index_;
						return precom.keys.at(keyID).rot_keys.at(index_);
					}
				}
			}
			std::cout << "Rotation index " << index << "/ " << index - slots << "not found." << std::endl;
			throw std::runtime_error("Rotation index" + std::to_string(index) + " not found");
		}
	}

	actual_index = index;
	return precom.keys.at(keyID).rot_keys.at(index);
}

void ContextData::AddRotationKey(int index, KeySwitchingKey&& ksk) {
	// index = index % (cc.N / 2);
	while (index < 0)
		index += this->N / 2;
	if (!precom.keys.contains(ksk.keyID))
		precom.keys[ksk.keyID] = Precomputations::KeyPrecomputations{};
	precom.keys.at(ksk.keyID).rot_keys.emplace(index, std::move(ksk));
}

bool ContextData::HasRotationKey(int index, const KeyHash& keyID) {
	// index = index % (cc.N / 2);
	while (index < 0)
		index += this->N / 2;
	if (!precom.keys.contains(keyID))
		precom.keys[keyID] = Precomputations::KeyPrecomputations{};
	return precom.keys.at(keyID).rot_keys.contains(index);
}

void ContextData::AddEvalKey(KeySwitchingKey&& ksk) {
	if (!precom.keys.contains(ksk.keyID))
		precom.keys[ksk.keyID] = Precomputations::KeyPrecomputations{};
	std::unique_ptr<KeySwitchingKey> key       = std::make_unique<KeySwitchingKey>(std::move(ksk));
	std::unique_ptr<KeySwitchingKey>& dest_key = precom.keys.at(key->keyID).eval_key;
	dest_key                                   = std::move(key);
}

bool ContextData::HasEvalKey(const KeyHash& keyID) {
	return precom.keys.contains(keyID) && precom.keys.at(keyID).eval_key != nullptr;
}

KeySwitchingKey& ContextData::GetEvalKey(const KeyHash& keyID) {
	assert(precom.keys.contains(keyID));
	assert(precom.keys[keyID].eval_key);
	return *precom.keys.at(keyID).eval_key;
}

void ContextData::AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) {
	AddBootPrecomputation(slots, 0, std::move(precomp));
}

void ContextData::AddBootPrecomputation(int slots, int levelsToDrop, BootstrapPrecomputation&& precomp) {
	precomp.levels_to_drop = levelsToDrop;
	if (precomp.shared_view) {
		// A view set allocates no limb memory (AddBootstrapPrecomputationShared): the count it
		// would have loaded is printed as views, and the two folds it carries instead.
		const size_t views = precomp.StC.size() * precomp.StC.at(0).A.size() + precomp.CtS.size() * precomp.CtS.at(0).A.size();
		std::ostringstream folds;
		folds.precision(17);
		folds << "CtS " << precomp.cts_fold << " StC " << precomp.stc_fold;
		std::cout << "Adding bootstrap precomputation to GPU for " << slots << " slots, " << levelsToDrop
				  << " levels below the chain top, shared with the full-height set.\n"
				  << "Plaintexts loaded: 0 ~ 0MB, " << views << " views, scale folds " << folds.str() << "\n";
	} else {
		std::cout << "Adding bootstrap precomputation to GPU for " << slots << " slots"
				  << (levelsToDrop ? ", " + std::to_string(levelsToDrop) + " levels below the chain top" : "") << ".\n"

				  << "Plaintexts loaded: "
				  << (precomp.CtS.size() == 0 ? (precomp.LT.A.size() + precomp.LT.invA.size()) :
												(precomp.StC.size() * precomp.StC.at(0).A.size() + precomp.CtS.size() * precomp.CtS.at(0).A.size()))
				  << " ~ "
				  << (precomp.CtS.size() == 0 ?
						 (precomp.LT.A.size() * (precomp.LT.A.at(0).c0.getLevel() + precomp.LT.A.at(0).c0.isModUp() * specialMeta[0].size()) +
						   precomp.LT.invA.size() * (precomp.LT.invA.at(0).c0.getLevel() + precomp.LT.invA.at(0).c0.isModUp() * specialMeta[0].size())) :
						 (precomp.StC.size() * precomp.StC.at(0).A.size() *
							 (1 + precomp.StC.at(0).A.at(0).c0.getLevel() + precomp.StC.at(0).A.at(0).c0.isModUp() * specialMeta[0].size()) +
						   precomp.CtS.size() * precomp.CtS.at(0).A.size() *
							 (1 + precomp.CtS.at(0).A.at(0).c0.getLevel() + precomp.CtS.at(0).A.at(0).c0.isModUp() * specialMeta[0].size()))) *
			N * 8 / (1 << 20)
			<< "MB\n";
	}

	precom.boot.emplace(std::pair{ slots, levelsToDrop }, std::move(precomp));
}

FIDESlib::CKKS::RESCALE_TECHNIQUE ContextData::translateRescalingTechnique(lbcrypto::ScalingTechnique technique) {
	return technique == lbcrypto::ScalingTechnique::FIXEDAUTO  ? FIDESlib::CKKS::FIXEDAUTO :
	  technique == lbcrypto::ScalingTechnique::FIXEDMANUAL	   ? FIDESlib::CKKS::FIXEDMANUAL :
	  technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT ? FIDESlib::CKKS::FLEXIBLEAUTOEXT :
	  technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTO	   ? FIDESlib::CKKS::FLEXIBLEAUTO :
																 FIDESlib::CKKS::NO_RESCALE;
}

void ContextData::PrepareNCCLCommunication() {

	if (GPUid.size() > 1) {

		std::set<int> ids;
		for (int i : GPUid)
			ids.insert(i);
		int num_ranks = ids.size();

		bool p2p = true;
		for (auto& i : ids) {
			cudaSetDevice(i);
			for (auto& j : ids) {
				if (i != j) {
					int canAccessPeer;
					cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
					if (canAccessPeer)
						cudaDeviceEnablePeerAccess(j, 0);
					else
						p2p = false;

					cudaDeviceCanAccessPeer(&canAccessPeer, j, i);
					if (canAccessPeer)
						cudaDeviceEnablePeerAccess(j, 0);
					else
						p2p = false;
				}
			}
		}
		this->canP2P = p2p;
		std::cout << "GPU P2P? " << this->canP2P << std::endl;

		/*
		if (GPUid.size() > 1) {
			if (this->canP2P)
				std::cout << "P2P (Nvlink?) detected" << std::endl;
			else
				std::cout << "NO P2P" << std::endl;
		}*/
#ifdef NCCL
		NCCLCHECK(ncclGetUniqueId(&communicatorID));
		GPUrank.resize(GPUid.size());
		ncclGroupStart();
		for (uint32_t i = 0; i < GPUid.size(); i++) {
			cudaSetDevice(GPUid[i]);
			if (precom.dev_to_communicator[GPUid[i]] == nullptr) {
				NCCLCHECK(ncclCommInitRank(GPUrank.data() + i, num_ranks, communicatorID, i));
				precom.dev_to_communicator[GPUid[i]] = GPUrank.data() + i;
			} else {
				GPUrank[i] = *precom.dev_to_communicator[GPUid[i]];
			}
		}
		ncclGroupEnd();
#endif
		cudaDeviceSynchronize();
	}

	top_limb_stream.resize(2 * GPUid.size());
	top_limb_stream2.resize(2 * GPUid.size());
	top_limb_buffer.resize(2 * GPUid.size());
	top_limb_buffer2.resize(2 * GPUid.size());
	top_limb_buffer_handle.resize(2 * GPUid.size());
	top_limb_buffer2_handle.resize(2 * GPUid.size());
	for (size_t i = 0; i < 2 * GPUid.size(); ++i) {
		int g = i % GPUid.size();
		cudaSetDevice(GPUid[g]);

		top_limb_stream[g].init(100);
		top_limb_stream2[g].init(100);

		if (GPUid.size() > 1) {
#ifdef NCCL
			NCCLCHECK(ncclMemAlloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N));
			NCCLCHECK(ncclCommRegister(GPUrank[g], top_limb_buffer[i], sizeof(uint64_t) * N, &top_limb_buffer_handle[i]));
			NCCLCHECK(ncclMemAlloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N));
			NCCLCHECK(ncclCommRegister(GPUrank[g], top_limb_buffer2[i], sizeof(uint64_t) * N, &top_limb_buffer2_handle[i]));
#else
			cudaMalloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N);
			cudaMalloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N);
#endif
		} else {
			cudaMalloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N);
			cudaMalloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N);
		}
		// cudaDeviceSynchronize();
		top_limbptr.emplace_back(top_limb_stream[g], 1, GPUid[g], (void**)&top_limb_buffer[i]);
		top_limbptr2.emplace_back(top_limb_stream2[g], 1, GPUid[g], (void**)&top_limb_buffer2[i]);
		gatherStream.resize(GPUid.size());

		for (size_t i = 0; i < GPUid.size(); ++i) {
			gatherStream[i].resize(GPUid.size());
			for (size_t j = 0; j < GPUid.size(); ++j) {
				cudaSetDevice(GPUid[j]);
				gatherStream[i][j].init(100);
			}
		}

		/* for (int i = 0; i < dnum; ++i) {
			key_switch_digits.emplace_back(*this, L, true);
			key_switch_digits.back().generateSpecialLimbs();
		}*/

		CudaCheckErrorModNoSync;
	}

	digitStream.resize(dnum);
	digitStreamForMemcpyPeer.resize(dnum);
	for (int i = 0; i < dnum; ++i) {
		digitStream[i].resize(GPUid.size());
		digitStreamForMemcpyPeer[i].resize(GPUid.size());
		for (size_t j = 0; j < GPUid.size(); ++j) {
			cudaSetDevice(GPUid[j]);
			digitStream[i][j].init(100);
			digitStreamForMemcpyPeer[i][j].resize(GPUid.size());
			for (size_t k = 0; k < GPUid.size(); ++k) {
				digitStreamForMemcpyPeer[i][j][k].init(100);
			}
		}
	}

	digitStream2.resize(dnum);
	for (int i = 0; i < dnum; ++i) {
		digitStream2[i].resize(GPUid.size());
		for (size_t j = 0; j < GPUid.size(); ++j) {
			cudaSetDevice(GPUid[j]);
			digitStream2[i][j].init();
		}
	}
}

const std::vector<int> ContextData::generateDigitGPUid(std::vector<std::vector<LimbRecord>>& meta, const int L, const int dnum) {
	std::vector<int> res(dnum);
	for (size_t i = 0; i < meta.size(); ++i) {
		for (auto& j : meta[i]) {
			res[j.digit] = i;
		}
	}
	return res;
}

std::vector<std::vector<LimbRecord>> ContextData::generateSplitSpecialMeta(std::vector<LimbRecord>& specialMeta, const std::vector<int> GPUid) {
	std::vector<std::vector<LimbRecord>> res(GPUid.size());

	int init = 0;
	for (uint32_t i = 0; i < GPUid.size(); ++i) {
		cudaSetDevice(GPUid[i]);
		int num = (specialMeta.size() - init) / (GPUid.size() - i);
		for (int j = init; j < init + num; ++j) {
			res[i].emplace_back(LimbRecord{ .id = specialMeta[j].id, .type = specialMeta[j].type, .digit = specialMeta[j].digit });
			res[i].back().stream.init();
		}
		init += num;
	}
	return res;
}

ContextData::~ContextData() {
	// FIRST: stop the pool's failure path from reaching an object that is being torn down. Every
	// free and clear below can itself reach the pool, so de-registering here is what makes a
	// failure during teardown report the pool alone instead of walking lists that are mid-destruction.
	if (pool_failure_context.load(std::memory_order_relaxed) == this) {
		SetPoolFailureReporter(nullptr);
		pool_failure_context.store(nullptr, std::memory_order_relaxed);
	}

	// BEFORE anything is released: destroyOpScratch() empties the slots' auxiliary-polynomial
	// lists and the clears below empty the context's own, so a report taken later would say every
	// list was empty no matter what the run did. Off unless FIDESLIB_POOL_STATS=1 (MemPoolStats-
	// Enabled), so a library that is not being investigated prints nothing; the counters
	// themselves are kept either way and an application can read the pool line at any time
	// (fideslib::MemPoolStatsLine).
	if (MemPoolStatsEnabled()) {
		std::string report = MemPoolStatsReport(GPUid.empty() ? 0 : GPUid[0]);
		report += auxPolyStatsReport();
		std::fputs(report.c_str(), stderr);
	}
	for (uint32_t i = 0; i < GPUid.size(); ++i) {
		cudaSetDevice(GPUid[i]);
		CudaCheckErrorMod;
	}
	destroyOpScratch();
	key_switch_aux.reset(nullptr);
	//   CudaCheckErrorMod;
	key_switch_aux2.reset(nullptr);
	//   CudaCheckErrorMod;
	for (auto& i : moddown_aux) {
		i.reset(nullptr);
	}
	//   CudaCheckErrorMod;
	precom.auxPoly.clear();
	//   CudaCheckErrorMod;
	precom.monomialCache.clear();
	//   CudaCheckErrorMod;
	precom.boot.clear();
	//   CudaCheckErrorMod;
	for (size_t i = 0; i < top_limbptr.size(); ++i) {
		int g = i % GPUid.size();
		cudaSetDevice(GPUid[g]);
		if (GPUid.size() > 1) {
#ifdef NCCL
			if (top_limb_buffer_handle[i])
				NCCLCHECK(ncclCommDeregister(GPUrank[g], top_limb_buffer_handle[i]));
			if (top_limb_buffer2_handle[i])
				NCCLCHECK(ncclCommDeregister(GPUrank[g], top_limb_buffer2_handle[i]));
			NCCLCHECK(ncclMemFree(top_limb_buffer[i]));
			NCCLCHECK(ncclMemFree(top_limb_buffer2[i]));
#else

			cudaFree(top_limb_buffer[i]);
			cudaFree(top_limb_buffer2[i]);
#endif
		} else {
			cudaFree(top_limb_buffer[i]);
			cudaFree(top_limb_buffer2[i]);
		}
		top_limbptr[i].free(top_limb_stream[g]);
		top_limbptr2[i].free(top_limb_stream2[g]);
	}
	top_limbptr.clear();
	top_limbptr2.clear();
	CudaCheckErrorMod;
#ifdef NCCL
	NCCLCHECK(ncclGroupStart());
	for (auto rank : precom.dev_to_communicator) {
		if (rank.second) {
			cudaSetDevice(rank.first);
			NCCLCHECK(ncclCommFinalize(*rank.second));
			// CudaCheckErrorMod;
			NCCLCHECK(ncclCommDestroy(*rank.second));
			// CudaCheckErrorMod;
		}
	}
	NCCLCHECK(ncclGroupEnd());
#endif
	Ciphertext::clearOpRecord();
}

// precom.auxPoly is the free list every Ciphertext constructor takes two polynomials from and every
// destructor returns them to (Ciphertext.cpp:78, :94-98, :104, :113). One host thread issuing ops is
// one consumer of it; several are several, and a std::vector cannot be pushed and popped
// concurrently. The lock is taken ONLY in the concurrent mode (the default mode keeps the unlocked
// code path this has always had), and when it is taken it covers only the vector operation -- never the
// device work -- so it is a few tens of nanoseconds against ops that cost microseconds to issue.
/// The free list the calling thread recycles through: the context's own in the default mode and
/// for slot 0, the thread's own slot list otherwise (see OpScratch::auxPoly for why).
std::vector<RNSPoly>& ContextData::auxPolyList() {
	const int slot = ScratchSlot();
	return slot == 0 ? precom.auxPoly : opScratch(slot).auxPoly;
}

std::vector<RNSPoly>& ContextData::auxPolyListFor(const int slot) {
	if (slot <= 0)
		return auxPolyList();
	if (static_cast<size_t>(slot) >= op_scratch.size() || op_scratch[slot] == nullptr)
		return auxPolyList();
	return op_scratch[slot]->auxPoly;
}

bool ContextData::hasAuxilarPoly() const {
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	return const_cast<ContextData*>(this)->auxPolyList().empty();
}

RNSPoly ContextData::getAuxilarPoly() {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kAuxPoly);
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	std::vector<RNSPoly>& list = auxPolyList();
	if (list.empty()) {
		++aux_poly_created_;
		RNSPoly fresh(*this);
		fresh.aux_slot = ScratchSlot(); // the list this polynomial belongs to, for its return
		return fresh;
	} else {
		++aux_poly_reused_;
		RNSPoly res(std::move(list.back()));
		list.pop_back();
		return res;
	}
}

void ContextData::returnAuxilarPoly(RNSPoly&& c) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kAuxPoly);
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	++aux_poly_returned_;
	// Back to the slot that CREATED it (RNSPoly::aux_slot), not to the returning thread's list.
	// In the default mode both are slot 0. Under FIDESLIB_CONCURRENT_OPS this is what keeps the
	// per-slot lists at their own steady state instead of draining every worker's into slot 0.
	std::vector<RNSPoly>& own	= auxPolyList();
	std::vector<RNSPoly>& home = auxPolyListFor(c.aux_slot);
	if (&home != &own)
		++aux_poly_rehomed_;
	home.emplace_back(std::move(c));
}

/// See the declaration. `auxPolyList()` answers for the CALLING thread's slot, so this walks the
/// slots itself: the asymmetry it is meant to show is precisely that one slot's list grows while
/// another's stays empty, and a per-caller view could never say so.
///
/// TEARDOWN ONLY, and that is the whole safety argument. Its one caller is ~ContextData(), which
/// runs when the context is going away and therefore when no other thread may still be issuing
/// ops against it. This function must not be called from anywhere else.
///
/// It used to also take `op_scratch_lock` while walking `op_scratch`, which was MISLEADING: that
/// mutex is only taken by destroyOpScratch() and by nothing on the allocating side -- opScratch()
/// creates `op_scratch[slot]` with no lock at all, on the grounds that a slot belongs to exactly
/// one thread (see the comment there). So holding it here excluded nothing a live thread could do
/// and merely made the walk LOOK synchronized. It is dropped rather than propagated: the honest
/// statement is the teardown precondition above, and adding a real lock to opScratch()'s hot path
/// to make the pretence true would be a cost paid by every op for the benefit of a debug print.
///
/// `aux_poly_lock` IS kept. It genuinely guards the lists this reads (returnAuxilarPoly pushes to
/// them under it), it is already the lock the counters are written under, and taking it costs an
/// uncontended acquire once per context destruction.
std::string ContextData::auxPolyStatsReport() const {
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	return auxPolyStatsText(true);
}

/// See the declaration. The difference from auxPolyStatsReport() is entirely about the lock: this
/// one is called from a thread that is about to abort and may itself be inside `aux_poly_lock`, so
/// it takes what it can get and reports honestly about what it could not.
std::string ContextData::auxPolyStatsReportTry() const {
	if (!ConcurrentOps())
		return auxPolyStatsText(true);
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::try_to_lock);
	return auxPolyStatsText(lk.owns_lock());
}

std::string ContextData::auxPolyStatsText(const bool with_lists) const {
	char buf[256];
	// `released` is the pass-boundary release (releaseParkedScratch); it does NOT change
	// `outstanding`, which is polynomials currently held by a live Ciphertext -- a released
	// polynomial was parked, i.e. already returned, and is subtracted from neither side.
	std::snprintf(buf, sizeof(buf), "auxpoly: created %llu reused %llu returned %llu (rehomed %llu) released %llu outstanding %lld\n",
				  static_cast<unsigned long long>(aux_poly_created_), static_cast<unsigned long long>(aux_poly_reused_),
				  static_cast<unsigned long long>(aux_poly_returned_), static_cast<unsigned long long>(aux_poly_rehomed_),
				  static_cast<unsigned long long>(aux_poly_released_),
				  static_cast<long long>(aux_poly_created_ + aux_poly_reused_) - static_cast<long long>(aux_poly_returned_));
	std::string out = buf;

	if (!with_lists) {
		// The failure path could not take `aux_poly_lock`. The counters above were read without
		// it and could be torn; the lists cannot be walked at all, because returnAuxilarPoly is
		// free to be pushing to one right now.
		out += "  auxpoly slot lists: NOT READ (aux_poly_lock busy); counters above are unsynchronised\n";
		return out;
	}

	// Slot 0 IS `precom.auxPoly` (see auxPolyList), so it is named the same way the slots are.
	std::string lists;
	std::snprintf(buf, sizeof(buf), "0:%zu", precom.auxPoly.size());
	lists = buf;
	for (size_t slot = 1; slot < op_scratch.size(); ++slot) {
		if (op_scratch[slot] == nullptr)
			continue;
		std::snprintf(buf, sizeof(buf), " %zu:%zu", slot, op_scratch[slot]->auxPoly.size());
		lists += buf;
	}
	std::snprintf(buf, sizeof(buf), "  auxpoly slot lists: %s\n", lists.c_str());
	out += buf;
	return out;
}

/// See the declaration. `aux_poly_lock` is taken in the concurrent mode exactly as every other
/// reader of these lists takes it; `op_scratch` is walked under the same precondition
/// releaseParkedScratch states, which is the caller's to meet.
size_t ContextData::auxPolyParkedCount() const {
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	size_t n = precom.auxPoly.size();
	for (size_t slot = 1; slot < op_scratch.size(); ++slot)
		if (op_scratch[slot] != nullptr)
			n += op_scratch[slot]->auxPoly.size();
	return n;
}

/// See the declaration in Context.cuh for the drift this undoes, the order of the two steps and
/// the caller's precondition. The code here is the mechanism only.
///
/// WHY THE LISTS ARE SWAPPED OUT RATHER THAN CLEARED IN PLACE. Destroying an RNSPoly calls GPUfree,
/// which takes `mempool_lock`. Clearing under `aux_poly_lock` would hold BOTH -- an ordering that
/// does already exist (getAuxilarPoly holds aux_poly_lock and then allocates, which takes
/// mempool_lock, so aux -> pool is the established direction and this would not invert it), but
/// holding a lock across tens of thousands of frees is a stall with no purpose. So the vectors are
/// moved out under the lock, which is a pointer swap, and destroyed after it is released.
///
/// The counters are bumped inside the lock with the swap, so a report taken between the two halves
/// says the polynomials are gone -- which is true of the LISTS, which is what the counters
/// describe. Nothing reads `aux_poly_released_` to decide anything.
ContextData::ScratchRelease ContextData::releaseParkedScratch() {
	ScratchRelease out;
	// DEFAULT MODE: exactly the no-op promised in the header. Before any lock, any counter and
	// any driver call, so an application that calls this unconditionally at its pass boundary
	// leaves the single-threaded path byte-identical.
	if (!ConcurrentOps())
		return out;
	out.ran = true;

	// Captured FIRST, before anything can move it: ~LimbPartition does its own cudaSetDevice and
	// does not put it back, so reading the caller's device after the frees below would read a
	// destructor's choice rather than the caller's.
	int entry_device = -1;
	cudaGetDevice(&entry_device);

	// ---- 1. the parked auxiliary polynomials, every slot ----------------------------------
	std::vector<std::vector<RNSPoly>> taken;
	{
		std::lock_guard<std::mutex> lk(aux_poly_lock);
		// Slot 0 IS `precom.auxPoly` (auxPolyList), so it is drained the same way the rest are
		// and is not a special case beyond where the vector lives.
		if (!precom.auxPoly.empty()) {
			out.polys_released += precom.auxPoly.size();
			++out.slots_drained;
			taken.emplace_back(std::move(precom.auxPoly));
			precom.auxPoly.clear(); // a moved-from vector is valid but unspecified; make it empty.
		}
		for (size_t slot = 1; slot < op_scratch.size(); ++slot) {
			if (op_scratch[slot] == nullptr || op_scratch[slot]->auxPoly.empty())
				continue;
			out.polys_released += op_scratch[slot]->auxPoly.size();
			++out.slots_drained;
			taken.emplace_back(std::move(op_scratch[slot]->auxPoly));
			op_scratch[slot]->auxPoly.clear();
		}
		aux_poly_released_ += out.polys_released;
	}
	// The frees themselves, outside `aux_poly_lock`. Each RNSPoly's limbs go back through GPUfree
	// onto THIS thread's lane freed list, which step 2 then moves to the shared fresh tier.
	taken.clear();

	// ---- 2. the pool lanes, every size class ----------------------------------------------
	// MemPoolReleaseAllLanes synchronizes the CALLING THREAD'S CURRENT DEVICE and refuses to run
	// on any other, so the device has to be set per id and put back afterwards -- this is called
	// from the application's pass thread, whose current device is its own business.
	for (const int id : GPUid) {
		cudaSetDevice(id);
		out.blocks_moved += MemPoolReleaseAllLanes(id);
	}
	if (entry_device >= 0)
		cudaSetDevice(entry_device);
	CudaCheckErrorModNoSync;
	return out;
}

/// The CALLING THREAD'S slot only, and still without a caller in this tree. It is not what the
/// pass-boundary release needed: the drift is one slot hoarding what the OTHER slots' threads
/// created, and a function that can only reach the list of whoever calls it cannot see that. See
/// releaseParkedScratch, which walks every slot and is the one with a caller.
void ContextData::trimAuxilarPoly(size_t size) {
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	std::vector<RNSPoly>& list = auxPolyList();
	while (list.size() > size)
		list.pop_back();
	// precom.auxPoly.erase(precom.auxPoly.begin() + std::min(size, precom.auxPoly.size()), precom.auxPoly.end());
}

void ContextData::clearAuxilarPoly() {
	std::unique_lock<std::mutex> lk(aux_poly_lock, std::defer_lock);
	if (ConcurrentOps())
		lk.lock();
	// The pool can hold hundreds of RNSPolys with tens of pooled limb buffers each. Every pooled free records an event on
	// the freeing stream and makes the pool stream wait on it, so a clear costs thousands of event operations (nsys, H200,
	// N = 2^16, 104 polys: 4816 cudaEventRecord + 4816 cudaStreamWaitEvent, 12.6 ms of host time). After one device-wide
	// synchronize no buffer is in use any more, so those waits are redundant: skip them while the pool is being cleared.
	cudaDeviceSynchronize();
	FIDESlib::gpufree_presynced = true;
	precom.auxPoly.clear();
	FIDESlib::gpufree_presynced = false;
}

void ContextData::clearAutomorphismKeys(const KeyHash& KeyID) {
	for (auto& i : precom.keys) {
		if (KeyID.empty() || KeyID == i.first) {
			i.second.rot_keys.clear();
		}
	}
}

void ContextData::clearEvalMultKeys(const KeyHash& KeyID) {
	for (auto& i : precom.keys) {
		if (KeyID.empty() || KeyID == i.first) {
			i.second.eval_key.reset();
		}
	}
}

void ContextData::clearBootPrecomputation(const int slots) {
	if (slots == -1)
		precom.boot.clear();
	else {
		// Every height of that slot count goes, the reduced-level sets included: they are
		// derived from the same transforms and are meaningless without them.
		std::erase_if(precom.boot, [slots](const auto& e) { return e.first.first == slots; });
	}
}

void ContextData::clearParamSwitchKeys(const KeyHash& KeyID) {

	if (KeyID.empty())
		map_param_switch.clear();
	else {
		for (auto& i : map_param_switch) {
			if ((i.first.first == this->param || i.first.second == this->param)) {
				if (i.second) {
					if (i.second->contains(KeyID)) {
						i.second->erase(KeyID);
					}
				}
			}
		}
	}
}

Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs) {

	ContextData* data = new ContextData(param, devs);
	if (OK) {
		// Context cc();
		Context cc(data); //= std::make_shared<ContextData>(param, devs);
		// Context cc;

		map_param_context[data->param] = cc;

		// if (!currentContext)
		SetCurrentContext(cc);
	} else {
		SetCurrentContext(map_param_context[data->param]);
		delete data;
	}
	Context res = GetCurrentContext();
	return res;
}

void DeregisterCryptoContextGPU(const Parameters& param) {
	map_param_context.erase(param);
	std::lock_guard<std::mutex> guard(currentContextLock);
	if (currentContext && currentContext->param == param) {
		currentContext.reset();
	}
}

void DeregisterCryptoContextGPU(Context cc) {
	// SPARSE_ENCAPSULATED (and only it) leaves two pieces of state behind that this function used to
	// walk straight past, because neither lives in the ContextData being dropped -- both live in
	// namespace-scope containers in this file:
	//
	//   * the parameter-switching key PAIR, moved into map_param_switch by AddSecretSwitchingKey.
	//     Nothing ever erased it, so it was destroyed during static destruction.
	//   * the SECOND, "switchable" GPU context. AddBootstrapKeys builds it with
	//     GenCryptoContextGPU, which registers it in map_param_context; the only handle anyone keeps
	//     is BootstrapPrecomputation::sparse_context, a weak_ptr. So map_param_context held the last
	//     strong reference and nothing ever erased that either.
	//
	// THE CRASH the first of those caused, measured: bootstrap_probe --dist encaps on an RTX PRO
	// 6000 printed every bootstrap measurement and then died of SIGSEGV, nine times out of nine,
	// while --dist sparse and --dist uniform exited 0 every time. compute-sanitizer memcheck
	// reported "ERROR SUMMARY: 0 errors" alongside "process didn't terminate successfully", so
	// nothing on the device was wrong; the kernel's own record named a host-side WRITE fault at one
	// fixed code offset in four reproductions:
	//
	//   bootstrap_probe[8136]: segfault at 5cc0f3424b14 ip 00005cc0f26ce062 sp 00007ffde9b0e970
	//     error 6 in bootstrap_probe[5cc0f2516000+41d000]        (offset 0x1b8062; error 6 = user
	//                                                             write; the target is brk heap,
	//                                                             15.8 MB past a 4.3 MB mapping)
	//
	// i.e. a write through a dangling reference into freed heap, which is exactly what the ownership
	// graph predicts. A KeySwitchingKey owns two RNSPolys; an RNSPoly's LimbPartitions bind
	// `std::vector<LimbRecord>& meta` to cc.meta.at(id), and each Limb holds `Stream& stream`
	// pointing at a LimbRecord::stream that is stored INSIDE the ContextData. ~Limb calls
	// v.free(stream), and Stream::ptr() assigns `updated = false` -- so destroying a key WRITES into
	// its context. CudaEngine::teardown destroyed the main ContextData inside main and left the keys
	// in map_param_switch for static destruction, which put the write strictly after the free.
	//
	// So release both HERE, inside main, and in this order: keys first, contexts second, so every
	// key writes into a context that is still there. (Making KeySwitchingKey::cc owning would also
	// close the hole, and is the wrong fix: the rotation and eval keys live inside ContextData, so an
	// owning back-reference would cycle and leak every context in the process.)
	//
	// the whole dual-context teardown above is now dead. In-context ENCAPS keeps both
	// switching keys in the MAIN context's rotation-key map (torn down with the context), builds
	// no second "switchable" GPU context, and calls no AddSecretSwitchingKey — so there is
	// nothing extra to release. Left as documentation of the crash this used to guard against.
	DeregisterCryptoContextGPU(cc->param);
}

Context GetCurrentContext() {
	std::lock_guard<std::mutex> guard(currentContextLock);
	return currentContext;
}

__global__ void dummy() {
}

void SetCurrentContext(Context& cc) {
	// DIAGNOSTIC DRAIN (FIDESLIB_CONCURRENT_OPS_SYNC=1, and only with the concurrent mode on; see
	// CudaUtils.cuh ConcurrentOpsSync). Every CKKS op enters through here, so one drain at this
	// point means no thread issues an op while any other thread's kernels are still running --
	// which makes every stream order legal and so removes every missing-device-dependency defect,
	// leaving host-state defects exactly as they were. Placed BEFORE the lock so host threads
	// still interleave normally; a no-op (one predictable branch on a cached bool) when unset,
	// which is always in the default mode.
	ConcurrentOpsSyncPoint();
	std::lock_guard<std::mutex> guard(currentContextLock);
	// NOTE: when the context is already current this is a no-op, so it does NOT establish the
	// calling thread's CUDA device. That is deliberate and unchanged: the CUDA current device is
	// per-thread, and every path that needs one sets it explicitly (RNSPoly::load /
	// loadConstant per limb, LimbPartition per partition). Do not turn this into an
	// unconditional cudaSetDevice -- on the hot single-GPU path it would be a syscall per op.
	if (cc != currentContext) {
		CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
		const bool swapping_out = static_cast<bool>(currentContext);
		currentContext          = cc;
		if (currentContext) {

			parallel_for(0, currentContext->GPUid.size(), 1, [&](int i) {
				// for (size_t i = 0; i < currentContext->GPUid.size(); ++i) {
				cudaSetDevice(currentContext->GPUid[i]);
				// SWAPPING `constants` IS A DEVICE-WIDE BARRIER, NOT A COPY. `FIDESlib::constants`
				// (ConstantsGPU.cu) is ONE process-wide __constant__ object and it is where every
				// kernel in this library reads its moduli, roots, Shoup precomputations and the
				// u64/u32 type mask from (C_, TABLE32/TABLE64, ISU64). Overwriting it changes what
				// EVERY in-flight and subsequently-launched kernel computes against.
				//
				// The two synchronizations below used to be commented out, and the copy was
				// cudaMemcpyToSymbolAsync on stream 0 -- i.e. unordered against the per-limb
				// streams all real work runs on. Two races, both of them "kernel indexes a table
				// for a limb the other context does not have":
				//   * kernels of the OUTGOING context still executing when the copy lands read the
				//     INCOMING context's tables;
				//   * kernels of the INCOMING context launched on a limb stream before the stream-0
				//     copy completes read the OUTGOING context's tables.
				//
				// For one context in a process this whole block is dead code -- the `cc !=
				// currentContext` guard above never fires after startup -- which is why it never
				// hurt SPARSE_TERNARY or UNIFORM_TERNARY and why the races went unseen. But
				// SPARSE_ENCAPSULATED is the one configuration with TWO GPU contexts: the
				// switchable sparse context AddBootstrapKeys builds for the mod-raise dance. Its
				// bootstrap flips MAIN -> SPARSE -> MAIN mid-compute, and the sparse context has
				// ONE limb and ONE special prime against the main context's 26 and 6, so a
				// misread lands far outside the tables. Measured consequence (a
				// full application run at the default encaps secret): block 0 completes healthy
				// and then a kernel takes 'an illegal memory access was encountered', surfacing as
				// a sticky error at the next stream/event call (CudaUtils.cu:229 / :449) and
				// SIGSEGV. Block 0 survives because a cold pipeline serializes the copy against
				// its dependents; by block 1 the pipeline is warm and overlapped and the window
				// is open.
				//
				// So: drain the device before overwriting the symbol, and make the copy itself
				// blocking so nothing launched after this call can outrun it. Both costs are paid
				// ONLY on a real context switch -- never on the single-context hot path, where the
				// guard keeps this unreachable -- so the sparse and uniform paths are untouched,
				// bit-for-bit and cycle-for-cycle.
				if (swapping_out)
					cudaDeviceSynchronize();
				cudaMemcpyToSymbol(FIDESlib::constants, &(currentContext->precom.constants[i]), sizeof(FIDESlib::Constants), 0, cudaMemcpyHostToDevice);
			});
		}
	}
}

// Add/Has/GetSecretSwitchingKey removed. They keyed the dual-context switching keys off a
// (Parameters, Parameters) pair in the namespace-scope map_param_switch; in-context ENCAPS stores
// both keys in the MAIN context's rotation-key map instead (AddBootstrapKeys), so no cross-context
// key store is needed. map_param_switch / clearParamSwitchKeys are left in place (now always empty)
// to avoid touching unrelated declarations.

int32_t normalyzeIndex(int32_t index, int32_t slots, int32_t N) {
	index = index % slots;
	if (index < 0)
		index += slots;
	if (index > slots / 2)
		index += N / 2 - slots;
	return index;
}

void DeregisterAllContexts() {
	map_param_switch.clear();
	map_param_context.clear();
	std::lock_guard<std::mutex> guard(currentContextLock);
	currentContext.reset();
}

} // namespace FIDESlib::CKKS
