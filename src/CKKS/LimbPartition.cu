//
// Created by carlosad on 27/04/24.
//
#include <algorithm>
#include "CudaUtils.cuh"
#include <array>
#include <stdexcept>
#include <variant>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "Rotation.cuh"
#include "VectorGPU.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
// constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
// constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

LimbPartition::LimbPartition(LimbPartition&& l) noexcept
: cc(l.cc), uid(l.uid), level(l.level), id(l.id), device((cudaSetDevice(l.device), l.device)), rank(l.rank), s(std::move(l.s)), meta(l.meta),
  SPECIALmeta(l.SPECIALmeta), digitid(l.digitid), DECOMPmeta(l.DECOMPmeta), DIGITmeta(l.DIGITmeta), GATHERmeta(l.GATHERmeta), limb(std::move(l.limb)),
  SPECIALlimb(std::move(l.SPECIALlimb)), DECOMPlimb(std::move(l.DECOMPlimb)), DIGITlimb(std::move(l.DIGITlimb)), GATHERlimb(std::move(l.GATHERlimb)),
  bufferAUXptrs(l.bufferAUXptrs), limbptr(std::move(l.limbptr)), auxptr(std::move(l.auxptr)),

  SPECIALlimbptr(std::move(l.SPECIALlimbptr)), SPECIALauxptr(std::move(l.SPECIALauxptr)), DECOMPlimbptr(std::move(l.DECOMPlimbptr)),
  //      DECOMPauxptr(std::move(l.DECOMPlimbptr)),
  DIGITlimbptr(std::move(l.DIGITlimbptr)),
  //      DIGITauxptr(std::move(l.DIGITlimbptr)),
  GATHERptr(std::move(l.GATHERptr)), bufferDECOMPandDIGIT(l.bufferDECOMPandDIGIT), bufferSPECIAL(l.bufferSPECIAL), bufferLIMB(l.bufferLIMB),
  bufferGATHER(l.bufferGATHER), bufferDECOMPandDIGIT_handle(l.bufferDECOMPandDIGIT_handle), bufferGATHER_handle(l.bufferGATHER_handle),
  bufferDECOMPandDIGIT_bytes(l.bufferDECOMPandDIGIT_bytes), bufferSPECIAL_bytes(l.bufferSPECIAL_bytes), bufferLIMB_bytes(l.bufferLIMB_bytes), bufferLIMB_pooled(l.bufferLIMB_pooled),
  bufferGATHER_bytes(l.bufferGATHER_bytes) {
	// THE INVARIANT THIS MOVE RESTS ON: the source holds NO LIMBS YET.
	//
	// `s(std::move(l.s))` leaves `l.s.ptr_` null (CudaUtils.cu, Stream::Stream(Stream&&)), and a
	// `Limb` holds its stream as a `Stream&` (Limb.cuh:24) bound to the PARTITION's `s`
	// (USE_PARTITION_STREAM, ~:188-224 below). A reference cannot be re-seated, so any limb that
	// already existed in `l` would keep referring to the moved-from Stream and would from here on
	// launch on the legacy default stream and free with `GPUfree(..., 0)` -- the unfenced pool
	// hand-off described above `initStream`.
	//
	// Today that cannot happen, and this assert is what keeps it that way rather than a comment
	// hoping so. The only element-wise move of a `LimbPartition` is `std::vector<LimbPartition>`
	// reallocation inside `RNSPoly::RNSPoly` (RNSPoly.cpp:65-69), which emplaces one partition per
	// GPU BEFORE `grow()` creates any limb; every other move of a polynomial moves the VECTOR
	// (`GPU(std::move(src.GPU))`, RNSPoly.cpp:84), which steals the buffer and leaves each
	// LimbPartition -- and therefore each `s` -- at the address its limbs already point at.
	//
	// Asserted on THIS, not on `l`: the limb vectors were transferred by the initializer list
	// above, so a moved-from `l.limb` reads empty whatever it held, while a non-empty `limb` here
	// is exactly "the source had limbs" -- the case that would leave a dangling `Stream&`.
	// `DECOMPlimb` / `DIGITlimb` are sized to the digit count by the ordinary constructor and hold
	// one EMPTY vector per digit until `generateDecompAndDigit` runs, so the outer vector being
	// non-empty says nothing -- it is the inner ones that would carry limbs.
	assert(limb.empty() && SPECIALlimb.empty() && GATHERlimb.empty());
	assert(([&] {
		for (const auto& d : DECOMPlimb)
			if (!d.empty())
				return false;
		for (const auto& d : DIGITlimb)
			if (!d.empty())
				return false;
		return true;
	}()));
	l.bufferSPECIAL				  = nullptr;
	l.bufferLIMB				  = nullptr;
	l.bufferDECOMPandDIGIT		  = nullptr;
	l.bufferAUXptrs				  = nullptr;
	l.bufferGATHER				  = nullptr;
	l.bufferDECOMPandDIGIT_handle = nullptr;
	l.bufferGATHER_handle		  = nullptr;
	// The size follows the pointer out of the source, so a moved-from partition cannot free a
	// buffer it no longer owns with a size that says it does.
	l.bufferDECOMPandDIGIT_bytes = 0;
	l.bufferSPECIAL_bytes		 = 0;
	l.bufferLIMB_bytes			 = 0;
	l.bufferLIMB_pooled			 = false;
	l.bufferGATHER_bytes		 = 0;
}

std::vector<VectorGPU<void*>> LimbPartition::generateDecompLimbptr(void** buffer, const std::vector<std::vector<LimbRecord>>& DECOMPmeta, const int device, int offset) {
	std::vector<VectorGPU<void*>> result;
	for (auto& d : DECOMPmeta) {
		result.emplace_back(buffer, std::max(1ul, d.size()), device, offset);
		offset += MAXP;
	}
	return result;
}

void** CudaMallocAuxBuffer(Stream& stream, unsigned long size, int device) {
	void** malloc;
	CudaCheckErrorModNoSync;
	malloc = (void**)GPUmalloc(device, MAXP * sizeof(void*) * (4ul + 4 * std::max(size, 1ul)), stream.ptr(), false);
	// cudaMallocAsync(&malloc, MAXP * sizeof(void*) * (4ul + 4 * std::max(size, 1ul)),
	//                 stream.ptr());  // TODO: can reduce to 3
	CudaCheckErrorModNoSync;
	return malloc;
}

/// @brief The stream every Limb of this partition will launch on, fence with and free through.
///
/// `default_` is the `def_stream` flag of RNSPoly's constructor, and exactly two owners set it:
/// `Plaintext` (Plaintext.cu:53, :59 -- `c0(this->cc, -1, false, true)`) and `KeySwitchingKey`
/// (KeySwitchingKey.cu:60 -- `a(*cc, -1, false, true), b(*cc, -1, false, true)`). It used to mean
/// `Stream::initDefault()` unconditionally, i.e. `ptr_ = 0` (CudaUtils.cu, Stream::initDefault).
///
/// THAT NULL IS NOT PRIVATE TO THE PARTITION. `LimbPartition::generate` binds EVERY limb it
/// creates to the partition's stream (`USE_PARTITION_STREAM`, ~:188-224 below) and `Limb` holds it
/// as a `Stream&` (Limb.cuh:24), so every limb of a plaintext or of a key-switching key:
///   * LAUNCHES on `stream.ptr()`, which is 0 -- the legacy default stream;
///   * ALLOCATES with `GPUmalloc(..., stream.ptr())` (VectorGPU.cu:84) and FREES with
///     `GPUfree(..., stream.ptr())` (Limb.cu:20-22 -> VectorGPU.cu:54), both with 0.
///
/// DEFAULT MODE -- SOUND, AND UNCHANGED HERE. `initPool` creates every pooled stream with
/// `cudaStreamCreateWithFlags(..., 0)`, i.e. BLOCKING, so the legacy default stream's implicit
/// synchronization orders a null-stream launch after all work already issued on any of them and
/// before all work issued on them afterwards. With one issuing thread host issue order IS the
/// order, so the fences these limbs never ask for are supplied by the legacy stream, and that is
/// why this has never shown serially. `ConcurrentOps()` is false there, the branch below is the
/// branch it always took, and `initDefault()` is called with the same arguments as before: the
/// default path is behaviourally identical, instruction for instruction.
///
/// CONCURRENT MODE -- THE ROOT OF THE RACE, WHICH IS WHY THE FLAG STOPS MEANING THE NULL STREAM.
/// Two independent failures, both traced (FIDESLIB_POOL_TRACE):
///   * THE POOL HAND-OFF IS NEVER FENCED. `GPUfree(ptr, ..., 0)` reaches the lane pool and
///     `mine.wait(stream)` takes `Stream::wait(cudaStream_t)`'s `s == 0` early out, which issues
///     nothing. The block goes back on the free list tagged with the freeing lane but owing no
///     dependency; the next lane's `record_and_wait` then fences on a handshake stream that has
///     never been recorded on, and writes the block while the previous owner's kernels are still
///     reading it. Every one of the 96 traced pool FREE events carried stream = 0. The legacy
///     stream does not cover this: the early out issues NOTHING on it, and the readers and the
///     new writer are on two different blocking streams, which are not ordered against each other.
///   * THE CROSS-LANE FENCES ARE HALF-MISSING AND HALF-GLOBAL. `x.wait(pt.s)` early-outs on
///     `s.ptr_ == 0` and takes no dependency, while `pt.s.wait(x)` issues
///     `cudaStreamWaitEvent(nullptr, e)` -- a barrier on the legacy stream, i.e. on every blocking
///     stream on the device. That is the whole-lane serialization stage 8 measured when it tried
///     to fence stream 0 from inside the pool instead of removing it here.
/// So in the concurrent mode a `def_stream` partition takes an ordinary pooled stream, at the same
/// priority as every other partition, and the fences the surrounding code already asks for become
/// real ones. Nothing is locked, no atomic is added and nothing is synchronized to do it:
/// `ConcurrentOps()` is one cached bool.
Stream initStream(bool default_) {
	Stream s;
	if (default_ && !ConcurrentOps()) {
		s.initDefault();
	} else {
		s.init(50);
	}
	return s;
}

LimbPartition::LimbPartition(ContextData& cc, const uint64_t& uid, int* level, const int id, const bool def_stream)
: cc(cc), uid(uid), level(level), id(id), device((cudaSetDevice(cc.GPUid.at(id)), cc.GPUid.at(id))), rank(cc.GPUrank.at(id)), s(initStream(def_stream)),
  meta(cc.meta.at(id)), SPECIALmeta(cc.specialMeta.at(id)), digitid(cc.GPUdigits.at(id)), DECOMPmeta(cc.decompMeta.at(id)), DIGITmeta(cc.digitMeta.at(id)),
  GATHERmeta(cc.gatherMeta), DECOMPlimb(DECOMPmeta.size()), DIGITlimb(DIGITmeta.size()), bufferAUXptrs(CudaMallocAuxBuffer(s, cc.dnum, device)),
  /*
		limbptr(s, meta.size(), device),
		auxptr(s, meta.size(), device),
		SPECIALlimbptr(s, SPECIALmeta.size(), device),
		SPECIALauxptr(s, SPECIALmeta.size(), device),
		*/

  limbptr(bufferAUXptrs, std::max(1ul, meta.size()), device, 0), auxptr(bufferAUXptrs, std::max(1ul, meta.size()), device, MAXP),
  SPECIALlimbptr(bufferAUXptrs, std::max(1ul, SPECIALmeta.size()), device, 2 * MAXP),
  SPECIALauxptr(bufferAUXptrs, std::max(1ul, SPECIALmeta.size()), device, 3 * MAXP),

  DECOMPlimbptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, 4 * MAXP)),
  //      DECOMPauxptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
  DIGITlimbptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
  //      , DIGITauxptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + 3 * DECOMPmeta.size()) * MAXP))
  GATHERptr(bufferAUXptrs, std::max(1ul, GATHERmeta.size()), device, (4 + 2 * DECOMPmeta.size()) * MAXP) {
}

LimbPartition::~LimbPartition() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	cudaSetDevice(device);
	/*
		for (auto &i: limb) s.wait(STREAM(i));
		for (auto &i: SPECIALlimb) s.wait(STREAM(i));
		for (auto &i: DECOMPlimb) for (auto &j: i) s.wait(STREAM(j));
		for (auto &i: DIGITlimb) for (auto &j: i) s.wait(STREAM(j));
*/

	limbptr.free(s);
	auxptr.free(s);
	SPECIALlimbptr.free(s);
	SPECIALauxptr.free(s);
	GATHERptr.free(s);
	for (auto& d : DECOMPlimbptr)
		d.free(s);
	//    for (auto& d : DECOMPauxptr)
	//        d.free(s);
	for (auto& d : DIGITlimbptr)
		d.free(s);
	// for (auto& d : DIGITauxptr)
	//    d.free(s);

	// CudaCheckErrorMod;
	if (bufferDECOMPandDIGIT_handle) {
		cudaStreamSynchronize(s.ptr());
#ifdef NCCL
		if (bufferDECOMPandDIGIT_handle != (void*)-1)
			NCCLCHECK(ncclCommDeregister(rank, bufferDECOMPandDIGIT_handle));
		NCCLCHECK(ncclMemFree(bufferDECOMPandDIGIT));
#else
		assert(false);
#endif
	} else {
		if (bufferDECOMPandDIGIT) {
			assert(bufferDECOMPandDIGIT_bytes > 0);
			GPUfree(bufferDECOMPandDIGIT, id, (int)bufferDECOMPandDIGIT_bytes, s.ptr());
			bufferDECOMPandDIGIT	   = nullptr;
			bufferDECOMPandDIGIT_bytes = 0;
		}
		// cudaFreeAsync(bufferDECOMPandDIGIT, s.ptr());
	}
	// bufferSPECIAL IS NOT A POOL BLOCK. Its one allocation is the raw `cudaMalloc` in
	// generateSpecialLimb below, so it must not be handed to `GPUfree`: at its real size nothing
	// pools it and `GPUfree` ends in `cudaFreeAsync`, which is defined only for memory that came
	// from `cudaMallocAsync`. It goes back the way it came, after the stream its limbs were fenced
	// into has drained -- `cudaMalloc` was synchronous, and so is this.
	if (bufferSPECIAL) {
		cudaStreamSynchronize(s.ptr());
		cudaFree(bufferSPECIAL);
		bufferSPECIAL		= nullptr;
		bufferSPECIAL_bytes = 0;
	}
	if (bufferLIMB) {
		assert(bufferLIMB_bytes > 0);
		if (bufferLIMB_pooled)
			GPUfreeExact(bufferLIMB, id, bufferLIMB_bytes, s.ptr());
		else
			GPUfree(bufferLIMB, id, (int)bufferLIMB_bytes, s.ptr());
		bufferLIMB		 = nullptr;
		bufferLIMB_bytes = 0;
		bufferLIMB_pooled = false;
		// cudaFreeAsync(bufferLIMB, s.ptr());
	}
	if (bufferAUXptrs)
		GPUfree(bufferAUXptrs, id, MAXP * sizeof(void*) * (4ul + 4 * std::max(cc.dnum, 1)), s.ptr(), false);
	// cudaFreeAsync(bufferAUXptrs, s.ptr());
	if (bufferGATHER_handle) {
		cudaStreamSynchronize(s.ptr());

#ifdef NCCL
		if (bufferGATHER_handle != (void*)-1)
			NCCLCHECK(ncclCommDeregister(rank, bufferGATHER_handle));
		NCCLCHECK(ncclMemFree(bufferGATHER));
#else
		assert(false);
#endif
	} else {
		if (bufferGATHER) {
			assert(bufferGATHER_bytes > 0);
			GPUfree(bufferGATHER, id, (int)bufferGATHER_bytes, s.ptr());
			bufferGATHER	   = nullptr;
			bufferGATHER_bytes = 0;
		}
		// cudaFreeAsync(bufferGATHER, s.ptr());
	}
	limb.clear();
	SPECIALlimb.clear();
	DECOMPlimb.clear();
	DIGITlimb.clear();
}

Global::Globals* LimbPartition::getGlobals() {
	return cc.precom.globals->globals[id];
}

void LimbPartition::generate(std::vector<LimbRecord>& records,
  std::vector<LimbImpl>& limbs,
  VectorGPU<void*>& ptrs,
  int pos,
  VectorGPU<void*>* auxptrs,
  uint64_t* buffer,
  size_t offset,
  uint64_t* buffer_aux,
  size_t offset_aux,
  bool noptr) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	constexpr bool USE_PARTITION_STREAM = true;
	assert(pos < (int)records.size());
	cudaSetDevice(device);

	const int limbs_size = limbs.size();
	int size			 = std::max((int)(pos - limbs_size + 1), (int)0);
	std::vector<void*> cpu_ptr(size, nullptr);
	std::vector<void*> cpu_auxptr(size, nullptr);
	for (int i = limbs_size; i <= pos; ++i) {
		const LimbRecord& r = records.at(i);
		if (r.type == U32) {
			if (buffer && buffer_aux) {
				limbs.emplace_back(Limb<uint32_t>(
				  cc, (uint32_t*)buffer, 2 * offset, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, (uint32_t*)buffer_aux, 2 * offset_aux));
				offset += cc.N;
				offset_aux += cc.N;
			} else if (buffer) {
				limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
				offset += cc.N;
			} else
				limbs.emplace_back(Limb<uint32_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, auxptrs ? 1 : 0));
			cpu_ptr[i - limbs_size]	   = { &(std::get<U32>(limbs.back()).v.data)[0] };
			cpu_auxptr[i - limbs_size] = { &(std::get<U32>(limbs.back()).aux.data)[0] };
		}
		if (r.type == U64) {
			if (buffer && buffer_aux) {
				limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, buffer_aux, offset_aux));
				offset += cc.N;
				offset_aux += cc.N;
			} else if (buffer) {
				limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
				offset += cc.N;
			} else {
				if (auxptrs) {
					limbs.emplace_back(Limb<uint64_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, false));
				} else {
					limbs.emplace_back(Limb<uint64_t>(cc, id, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, true));
				}
			}

			cpu_ptr[i - limbs_size]	   = { &(std::get<U64>(limbs.back()).v.data)[0] };
			cpu_auxptr[i - limbs_size] = { &(std::get<U64>(limbs.back()).aux.data)[0] };
			// cudaFreeHost(aux);
		}
		CudaCheckErrorModNoSync;
		s.wait(USE_PARTITION_STREAM ? s : records.at(i).stream);
		CudaCheckErrorModNoSync;
	}
	if (size > 0) {
		if (!noptr) {
			UploadH2D(ptrs.data + limbs_size, cpu_ptr.data(), size * sizeof(void*), device, s.ptr());
			CudaCheckErrorModNoSync;
			if (auxptrs) {
				UploadH2D((*auxptrs).data + limbs_size, cpu_auxptr.data(), size * sizeof(void*), device, s.ptr());
			}
			CudaCheckErrorModNoSync;
		}
	}
	CudaCheckErrorModNoSync;
}

/*
void LimbPartition::generateLimb() {
	cudaSetDevice(device);
	generate(meta, limb, limbptr, (int)limb.size(), &auxptr);
}
*/

void LimbPartition::generateLimbToLevel(int new_level) {
	cudaSetDevice(device);
	int new_size = getLimbSize(new_level);
	if (static_cast<size_t>(new_size) > limb.size()) {
		generate(meta, limb, limbptr, new_size - 1, &auxptr);
	}
}

/*
void LimbPartition::generateAllDecompLimb(uint64_t* pInt, size_t offset) {
	cudaSetDevice(device);
	DECOMPlimb.resize(DECOMPmeta.size());
	for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
		generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)DECOMPmeta[i].size() - 1,
				 nullptr, pInt, offset, nullptr, 0);
		offset += cc.N * DECOMPmeta.at(i).size();
	}
}

*/

void LimbPartition::generateAllDigitLimb(uint64_t* pInt, size_t offset) {
	cudaSetDevice(device);
	DIGITlimb.resize(DIGITmeta.size());
	for (size_t i = 0; i < DIGITmeta.size(); ++i) {
		generate(DIGITmeta[i], DIGITlimb[i], DIGITlimbptr[i], (int)DIGITmeta[i].size() - 1, nullptr /*&DIGITauxptr[i]*/, pInt, offset, nullptr, 0);
		offset += cc.N * DIGITmeta.at(i).size();
	}
}

void LimbPartition::generateSpecialLimb(const bool zero_out, const bool for_communication) {
	cudaSetDevice(device);
	if ((for_communication && cc.GPUid.size() > 0 && bufferSPECIAL == nullptr && SPECIALmeta.size() > 0) ||
	  (!(for_communication && cc.GPUid.size() > 0) && SPECIALlimb.size() == 0 && SPECIALmeta.size() > 0)) {

		if ((for_communication && cc.GPUid.size() > 0)) {
			assert(SPECIALlimb.size() == 0);
			// RAW cudaMalloc, not the pool: the size is recorded so the free side can tell the two
			// apart and call cudaFree rather than pushing a non-pool pointer into the pool.
			bufferSPECIAL_bytes = std::max(1ul, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t));
			cudaMalloc(&bufferSPECIAL, bufferSPECIAL_bytes);
			generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, bufferSPECIAL, 0, bufferSPECIAL, cc.N * SPECIALmeta.size());
		} else {
			assert(bufferSPECIAL == nullptr);
			// bufferSPECIAL = (uint64_t*)GPUmalloc(device, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t), s.ptr());
			CudaCheckErrorModNoSync;
			// generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, nullptr, 0,
			//          nullptr, 0);
			// cudaMalloc(&bufferSPECIAL, std::max(1ul, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t)));
			generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, bufferSPECIAL, 0, bufferSPECIAL, cc.N * SPECIALmeta.size());

			// cudaMallocAsync(&bufferDECOMPandDIGIT, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t), s.ptr());
		}
	}
	if (zero_out) {
		if (bufferSPECIAL) {
			if (cc.N * SPECIALmeta.size() * sizeof(uint64_t) > 0)
				cudaMemsetAsync(bufferSPECIAL, 0, cc.N * SPECIALmeta.size() * sizeof(uint64_t), s.ptr());
		} else {
			for (auto& i : SPECIALlimb) {
				if (i.index() == U32) {
					cudaMemsetAsync(std::get<U32>(i).v.data, 0, cc.N * sizeof(uint32_t), STREAM(i).ptr());
				} else {
					cudaMemsetAsync(std::get<U64>(i).v.data, 0, cc.N * sizeof(uint64_t), STREAM(i).ptr());
				}
			}
		}
	}
}

template <ALGO algo, NTT_MODE mode>
void LimbPartition::ApplyNTT(int batch,
  LimbPartition::NTT_fusion_fields fields,
  std::vector<LimbImpl>& limb,
  VectorGPU<void*>& limbptr,
  VectorGPU<void*>& auxptr,
  ContextData& cc,
  const int primeid_init,
  const int limbsize) {
	constexpr int M = 4;

	const dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
	const dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
	const int bytesFirst	  = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
	const int bytesSecond	  = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
	const int size			  = (limbsize != -1 ? limbsize : limb.size()) - (mode == NTT_RESCALE || mode == NTT_MULTPT);

	for (int i = 0; i < size; i += batch) {
		uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

		NTT_<false, algo, mode><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr()>>>(getGlobals(),
		  (mode == NTT_RESCALE || mode == NTT_MULTPT) ? limbptr.data + size :
			(mode == NTT_MODDOWN)					  ? fields.op2->limbptr.data + i :
														limbptr.data + i,
		  primeid_init + i,
		  auxptr.data + i,
		  nullptr,
		  (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0,
		  nullptr,
		  nullptr);

		NTT_<true, algo, mode><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(i)).ptr()>>>(getGlobals(),
		  auxptr.data + i,
		  primeid_init + i,
		  limbptr.data + i,
		  mode == NTT_MULTPT ? fields.pt->limbptr.data + i : nullptr,
		  (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0,
		  nullptr,
		  nullptr);
	}
}

// ApplyNTT / ApplyINTT are called from other translation units (the single-limb coefficient-domain
// rebase helpers in Bootstrap.cu), so their instantiations have to actually land in the library
// archive. The NTT<algo, mode> / INTT<algo, mode> instantiations below are NOT enough: they only
// *implicitly* instantiate ApplyNTT/ApplyINTT, whose definitions live in this .cu (not in
// LimbPartition.cuh). An implicit instantiation has vague (COMDAT) linkage, so once it is inlined
// into its single in-file caller the out-of-line copy is dropped and no ApplyNTT/ApplyINTT symbol
// survives in libfideslib.a -- an external caller then fails to link. Explicit instantiation over
// the same (ALGO, MODE) grid as NTT/INTT forces the out-of-line copies to be emitted. Same reason
// Rescale.cu explicitly instantiates SwitchModulus<uint32_t/uint64_t> for its outside launchers.
#define YYY(algo, mode)                                                                                                                            \
	template void LimbPartition::ApplyNTT<algo, mode>(int batch, NTT_fusion_fields fields, std::vector<LimbImpl>& limb, VectorGPU<void*>& limbptr, \
	  VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init, const int limbsize);

#include "ntt_types.inc"
#undef YYY

template <ALGO algo, NTT_MODE mode> void LimbPartition::NTT(int batch, bool sync, NTT_fusion_fields fields) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kNtt);
	cudaSetDevice(device);
	int limbsize = getLimbSize(*level);

	if (batch >= 1) {
		if (limbsize > 0) {
			if (sync) {
				for (int i = 0; i < limbsize; i += batch) {
					STREAM(limb[i]).wait(s);
				}
			}
			ApplyNTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
			if (sync) {
				for (int i = 0; i < limbsize; i += batch) {
					s.wait(STREAM(limb[i]));
				}
			}
		}
		if (*level == cc.L + 1) {
			if (SPECIALmeta.size() > 0 && SPECIALmeta.at(0).id == cc.L + 1) {
				ApplyNTT<algo, mode>(batch, fields, SPECIALlimb, SPECIALlimbptr, SPECIALauxptr, cc, SPECIAL(id, 0), 1);
			}
		}
	} else {
		assert("Invalid NTT batch configuration!");
	}
}

#define YYY(algo, mode) template void LimbPartition::NTT<algo, mode>(int batch, bool sync, NTT_fusion_fields fields);

#include "ntt_types.inc"
#undef YYY

template <ALGO algo, INTT_MODE mode>
void LimbPartition::ApplyINTT(int batch,
  LimbPartition::INTT_fusion_fields fields,
  std::vector<LimbImpl>& limb,
  VectorGPU<void*>& limbptr,
  VectorGPU<void*>& auxptr,
  ContextData& cc,
  const int primeid_init,
  const int limbsize) {
	constexpr int M = 4;

	dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN - (cc.logN > 13 ? 0 : 0)) / 2 - 1)) };
	dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1 + (cc.logN > 13 ? 0 : 0)) / 2 - 1)) };
	int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
	int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

	for (int i = 0; i < limbsize; i += batch) {
		uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(limbsize - i));

		INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr()>>>(
		  getGlobals(), limbptr.data + i, primeid_init + i, auxptr.data + i);

		INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(i)).ptr()>>>(
		  getGlobals(), auxptr.data + i, primeid_init + i, limbptr.data + i);
	}
}

#define WWW(algo, mode)                                                                                                                              \
	template void LimbPartition::ApplyINTT<algo, mode>(int batch, INTT_fusion_fields fields, std::vector<LimbImpl>& limb, VectorGPU<void*>& limbptr, \
	  VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init, const int limbsize);

#include "ntt_types.inc"
#undef WWW

template <ALGO algo, INTT_MODE mode> void LimbPartition::INTT(int batch, bool sync, INTT_fusion_fields fields) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kNtt);
	cudaSetDevice(device);
	// TODO check level
	const int limbsize = getLimbSize(*level);
	if (batch >= 1) {
		if (limbsize > 0) {
			if (sync) {
				for (int i = 0; i < limbsize; i += batch) {
					STREAM(limb[i]).wait(s);
				}
			}
			ApplyINTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
			if (sync) {
				for (int i = 0; i < limbsize; i += batch) {
					s.wait(STREAM(limb[i]));
				}
			}
		}
	} else {
		assert("Invalid INTT batch configuration!");
	}
}

#define WWW(algo, mode) template void LimbPartition::INTT<algo, mode>(int batch, bool sync, INTT_fusion_fields fields);

#include "ntt_types.inc"
#undef WWW

void LimbPartition::add(const LimbPartition& p, const bool exta, const bool extb) {
	cudaSetDevice(device);
	const int limbsize = getLimbSize(*level);
	s.wait(p.getS());

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		if (exta == extb) {
			add_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
		} else if (exta) {
			add_scale_p_b_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
		} else if (extb) {
			add_scale_p_a_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
		}
	}
	if (exta || extb) {
		if (exta && !extb) {
			// DO NOTHING !!!
		} else if (extb && !exta) {
			int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
			int num_limbs = cc.splitSpecialMeta.at(id).size();
			for (size_t i = start; i < static_cast<size_t>(start + num_limbs); i += cc.batch) {
				STREAM(SPECIALlimb[i]).wait(s);
				uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);
				{
					copy_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(p.SPECIALlimbptr.data + i, SPECIALlimbptr.data + i);
				} // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
			}
			for (size_t i = start; i < static_cast<size_t>(start + num_limbs); i += cc.batch) {
				s.wait(STREAM(SPECIALlimb[i]));
			}
		} else if (exta && extb) {
			int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
			int num_limbs = cc.splitSpecialMeta.at(id).size();
			for (size_t i = start; i < static_cast<size_t>(start + num_limbs); i += cc.batch) {
				STREAM(SPECIALlimb[i]).wait(s);
				uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);
				{
					add_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(SPECIALlimbptr.data + i,
					  p.SPECIALlimbptr.data + i,
					  SPECIAL(id,
						i)); // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
				}
			}
			for (size_t i = start; i < static_cast<size_t>(start + num_limbs); i += cc.batch) {
				s.wait(STREAM(SPECIALlimb[i]));
			}
		}
	}
	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	p.getS().wait(s);
}

void LimbPartition::scaleByP() {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		scaleByP_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
}

void LimbPartition::sub(const LimbPartition& p) {
	cudaSetDevice(device);
	const int limbsize = getLimbSize(*level);
	s.wait(p.getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		sub_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	p.getS().wait(s);
}

void LimbPartition::multElement(const LimbPartition& p) {
	cudaSetDevice(device);

	int limbsize = getLimbSize(*level);
	assert(limbsize <= (int)p.limb.size());

	s.wait(p.getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		Mult_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	p.getS().wait(s);
}

void LimbPartition::multElement(const LimbPartition& partition1, const LimbPartition& partition2) {
	cudaSetDevice(device);
	int limbsize = getLimbSize(*level);
	assert(limbsize <= partition1.limb.size());
	assert(limbsize <= partition2.limb.size());

	s.wait(partition1.getS());
	s.wait(partition2.getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		Mult_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  (void**)limbptr.data + i, (void**)partition1.limbptr.data + i, (void**)partition2.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	partition1.getS().wait(s);
	partition2.getS().wait(s);
}

void LimbPartition::rescale() {
	const int limbsize = getLimbSize(*level);
	assert(cc.GPUid.size() > 1 || limbsize > 1);
	if (limbsize == 0)
		return;

	cudaSetDevice(device);

	LimbImpl& top = limb.at(limbsize - 1);

	int aux_size = 0;
	SWITCH_RET(top, aux.size, aux_size);
	if (aux_size == 0) {
		{
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			int start = 0;
			// Per-ISSUING-THREAD scratch (slot 0 is the context's own top_limb_*, i.e. exactly what
			// this read before): the top-limb INTT stages through ONE device buffer on ONE stream,
			// so two threads rescaling at once used to trample each other's coefficients.
			Stream& top_s			 = cc.getTopLimbStream(id);
			VectorGPU<void*>& top_p	 = cc.getTopLimbPtr(id);
			for (int i = limbsize - 1; i < limbsize; i += cc.batch) {
				top_s.wait(s);
				uint32_t num_limbs = 1;

				INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, top_s.ptr()>>>(
				  getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), top_p.data);

				// Stage 2 reads the scratch that stage 1 writes, and they sit on different streams:
				// the limb's stream must wait for the scratch stream HERE, before the launch. The
				// s.wait(top_s) after the loop orders only what FOLLOWS the loop.
				STREAM(limb.at(start + i)).wait(top_s);
				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), top_p.data, PARTITION(id, start + i), limbptr.data + start + i);
			}
			s.wait(top_s);
		}
	} else {
		STREAM(top).wait(s);
		SWITCH(top, INTT<ALGO_SHOUP>());
	}
	if (aux_size == 0) {
		auto& auxLimbs = cc.getModdownAux(0).GPU.at(id);
		s.wait(auxLimbs.getS());
		if (limbsize > 0) {
			constexpr ALGO algo = ALGO_SHOUP;
			constexpr int M		= 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			{
				NTT_<false, algo, NTT_RESCALE>
				  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), static_cast<unsigned int>(limbsize - 1) }, blockDimFirst, bytesFirst, s.ptr()>>>(
					getGlobals(), limbptr.data + limbsize - 1, PARTITION(id, 0), auxLimbs.limbptr.data, nullptr /*limbptr.data*/, PRIMEID(top));

				NTT_<true, algo, NTT_RESCALE>
				  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), static_cast<unsigned int>(limbsize - 1) }, blockDimSecond, bytesSecond, s.ptr()>>>(
					getGlobals(), auxLimbs.limbptr.data, PARTITION(id, 0), limbptr.data, nullptr, PRIMEID(top));
			}
		}
		auxLimbs.s.wait(s);
	} else {
		for (int32_t i = 0; i < limbsize - 1; i += cc.batch) {
			STREAM(limb[i]).wait(STREAM(top));
		}
		NTT<ALGO_SHOUP, NTT_RESCALE>(cc.batch, false, NTT_fusion_fields{});
		for (int32_t i = 0; i < limbsize - 1; i += cc.batch) {
			STREAM(top).wait(STREAM(limb[i]));
		}

		s.wait(STREAM(top));
	}
	// while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
	//     STREAM(limb.back()).wait(s);
	//     limb.pop_back();
	// }
}

void LimbPartition::multPt(const LimbPartition& p) {
	const int limbsize = getLimbSize(*level);
	// assert(SPECIALlimb.size() == 0 && p.SPECIALlimb.size() == 0);
	assert(limbsize <= p.limb.size());
	assert(limbsize > 1);
	cudaSetDevice(device);
	// The plaintext is an operand, as in multElement: wait for whatever its stream still has in
	// flight (its upload) before reading it, and below make its stream wait for this read before it
	// can be rewritten or freed. Without the first wait a plaintext loaded just before the call is
	// read half-written; without the second its blocks can return to the pool under the read.
	s.wait(p.getS());

	constexpr bool capture = false;
	// thread_local, not a bare function-local static: this is on the ct x pt path -- the hottest
	// call in the library for a matmul-shaped circuit -- and `exec_map[limbsize]` is a
	// std::map::operator[], i.e. a red-black tree INSERTION the first time a limb count is seen.
	// Two host threads issuing ops on one context (FIDESLIB_CONCURRENT_OPS) reach it at once and
	// corrupt the tree; a lookup concurrent with a rebalance is equally undefined. The map holds
	// one graph per limb count, so a copy per issuing thread costs a handful of entries and the
	// default mode is unaffected (one thread, one map).
	static thread_local std::map<int, cudaGraphExec_t> exec_map;

	{
		LimbImpl& top = limb.at(limbsize - 1);

		cudaGraphExec_t& exec = exec_map[limbsize];

		run_in_graph<capture>(exec, s, [&]() {
			STREAM(top).wait(s);
			SWITCH(top, mult(p.limb.at(limbsize - 1)));
			SWITCH(top, INTT<ALGO_SHOUP>());

			for (int32_t i = 0; i < limbsize - 1; i += cc.batch) {
				STREAM(limb.at(i)).wait(STREAM(top));
			}
			if (limbsize > 1)
				NTT<ALGO_SHOUP, NTT_MULTPT>(cc.batch, false, NTT_fusion_fields{ .pt = &p });
			for (int32_t i = 0; i < limbsize - 1; i += cc.batch) {
				STREAM(top).wait(STREAM(limb.at(i)));
			}

			s.wait(STREAM(top));
		});

		// while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
		//     STREAM(limb.back()).wait(s);
		//     limb.pop_back();
		// }
	}
	p.getS().wait(s);
}

void LimbPartition::modup(LimbPartition& aux_partition) {

	constexpr ALGO algo	 = ALGO_SHOUP;
	constexpr bool PRINT = false;
	// assert(SPECIALlimb.empty());
	cudaSetDevice(device);

	const int limbsize = *level + 1;
	generateAllDecompAndDigit(false);
	s.wait(aux_partition.getS());

	if constexpr (PRINT) {
		std::cout << "Before modup ";
		for (auto& i : limb) {
			SWITCH(i, printThisLimb(2));
		}
		std::cout << std::endl;
	}

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb.at(j).size();
		int size = std::min((int)DECOMPlimb.at(d).size(), limbsize - start);
		if (size <= 0)
			break;
		/*
		for (auto& l : DECOMPlimb[d]) {
			for (auto& p : limb) {
				if (PRIMEID(l) == PRIMEID(p)) {
					STREAM(l).wait(STREAM(p));
					SWITCH(l, INTT_from(p));
					s_d.wait(STREAM(l));
				}
			}
		}
		*/
		// Per-issuing-thread as well: this stream SCHEDULES the digit's mod-up, and two threads
		// sharing it would fence each other's decomposition kernels into one serial chain.
		Stream& s_d = cc.getDigitStream(d, id);
		s_d.wait(s);

		{
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			for (int i = 0; i < size; i += cc.batch) {
				STREAM(limb.at(start + i)).wait(s_d);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), auxptr.data + start + i);

				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), auxptr.data + start + i, PARTITION(id, start + i), DECOMPlimbptr[d].data + i);
			}
			for (int32_t i = 0; i < size; i += cc.batch) {
				s_d.wait(STREAM(limb.at(start + i)));
			}
		}

		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			std::cout << "After INTT ";
			for (auto& i : DECOMPlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}

		{
			dim3 blockSize{ 64, 2 };
			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
			DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s_d.ptr()>>>(DECOMPlimbptr[d].data, *level + 1, DIGITlimbptr[d].data, digitid[d], getGlobals());
		}
		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			std::cout << "After conv ";
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}

		const int digitsize = cc.precom.constants[id].num_primeid_digit_to[digitid.at(d)][*level];
		for (int32_t i = 0; i < digitsize; i += cc.batch) {
			STREAM(DIGITlimb.at(d).at(i)).wait(s_d);
		}
		ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, DIGITlimb.at(d), DIGITlimbptr.at(d), aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

		for (int32_t i = 0; i < digitsize; i += cc.batch) {
			s_d.wait(STREAM(DIGITlimb.at(d).at(i)));
		}

		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			std::cout << "After NTT ";
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}
	}
	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
		s.wait(cc.getDigitStream(d, id));
	}

	aux_partition.getS().wait(s);
}

void LimbPartition::freeSpecialLimbs() {
	cudaSetDevice(device);
	for (size_t i = 0; i < SPECIALlimb.size(); ++i) {
		STREAM(SPECIALlimb.at(i)).wait(s);
	}
	SPECIALlimb.clear();
	if (bufferSPECIAL != nullptr) {
		// Matching release for the raw `cudaMalloc` in generateSpecialLimb -- see the destructor.
		// The loop above has already fenced `s` behind every limb that viewed this buffer.
		assert(bufferSPECIAL_bytes > 0);
		cudaStreamSynchronize(s.ptr());
		cudaFree(bufferSPECIAL);
		// cudaFreeAsync(bufferSPECIAL, s.ptr());
		bufferSPECIAL		= nullptr;
		bufferSPECIAL_bytes = 0;
	}
}

void LimbPartition::copyLimb(const LimbPartition& partition) {
	cudaSetDevice(device);
	s.wait(partition.getS());
	int limbsize = getLimbSize(*level);
	assert(*level == *partition.level);
	// std::cout << "GPU: " << id << " copy " << limbsize << "limbs" << std::endl;
	if (limbsize > 0)
		copy_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)limbsize }, 128, 0, s.ptr()>>>(partition.limbptr.data, limbptr.data);
	/*
	for (size_t i = 0; i < partition.limb.size(); ++i) {
		STREAM(limb.at(i)).wait(s);
		SWITCH(limb.at(i), copyV(partition.limb.at(i)));
	}
	for (size_t i = 0; i < partition.limb.size(); ++i) {
		s.wait(STREAM(limb.at(i)));
	}
	*/
	partition.getS().wait(s);
}

void LimbPartition::copySpecialLimb(const LimbPartition& p) {
	cudaSetDevice(device);
	this->generateSpecialLimb(false, false);
	s.wait(p.getS());
	assert(*level == *p.level);
	int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
	int num_limbs = cc.splitSpecialMeta.at(id).size();
	for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
		STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]).wait(s);
		uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

		copy_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]).ptr()>>>(
		  p.SPECIALlimbptr.data + i - (SPECIALmeta.size() > p.SPECIALlimb.size()) * start, SPECIALlimbptr.data + i - (SPECIALmeta.size() > SPECIALlimb.size()) * start);
	}
	for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
		s.wait(STREAM(SPECIALlimb[i - (SPECIALmeta.size() > SPECIALlimb.size()) * start]));
	}
	p.getS().wait(s);
}

void LimbPartition::generateAllDecompAndDigit(bool iskey) {
	cudaSetDevice(device);
	if ((!(iskey || cc.GPUid.size() == 1) && bufferGATHER == nullptr) || ((iskey || cc.GPUid.size() == 1) && DECOMPlimb[0].size() == 0)) {
		int decomp_limbs = 0;
		for (auto& d : DECOMPmeta)
			decomp_limbs += d.size();
		int digit_limbs = 0;
		for (auto& d : DIGITmeta)
			digit_limbs += d.size();

		// size_t size = cc.N * (/*decomp_limbs +*/ digit_limbs);
		if (cc.GPUid.size() == 1 || iskey) {
			// bufferDECOMPandDIGIT = (uint64_t*)GPUmalloc(device, std::max(1ul, size) * sizeof(uint64_t), s.ptr());
			// cudaMallocAsync(&bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t), s.ptr());
		} else {

#ifdef NCCL
			/*
			cudaStreamSynchronize(s.ptr());
			NCCLCHECK(ncclMemAlloc((void**)&bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t)));
			NCCLCHECK(ncclCommRegister(rank, bufferDECOMPandDIGIT, std::max(1ul, size) * sizeof(uint64_t),
									   &bufferDECOMPandDIGIT_handle));
			if (bufferDECOMPandDIGIT_handle == nullptr)
				bufferDECOMPandDIGIT_handle = (void*)-1;
			cudaDeviceSynchronize();
			*/
#else
			assert(false);
#endif
		}
		// generateAllDecompLimb(bufferDECOMPandDIGIT, 0);
		generateGatherLimb(iskey);
		DECOMPlimb.resize(DECOMPmeta.size());
		for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
			for (size_t j = 0; j < DECOMPmeta.at(i).size(); ++j) {
				int pos = 0;
				for (size_t k = 0; k < cc.meta.size(); ++k) {
					for (size_t l = 0; l < cc.meta[k].size(); ++l) {
						if (cc.meta[k][l].id == DECOMPmeta[i][j].id) {
							generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)j, nullptr, bufferGATHER, pos * cc.N, nullptr, 0, true);
						}
						pos++;
					}
				}
			}
			std::vector<void*> cpu_ptr(DECOMPmeta.at(i).size(), nullptr);
			for (uint32_t j = 0; j < DECOMPlimb[i].size(); ++j) {
				cpu_ptr[j] = DECOMPlimb[i][j].index() == U32 ? (void*)std::get<U32>(DECOMPlimb[i][j]).v.data : (void*)std::get<U64>(DECOMPlimb[i][j]).v.data;
			}

			if (DECOMPmeta.at(i).size() * sizeof(void*) > 0)
				UploadH2D(DECOMPlimbptr[i].data, cpu_ptr.data(), DECOMPmeta.at(i).size() * sizeof(void*), device, s.ptr());
		}
		generateGatherLimb(iskey);
		generateAllDigitLimb(bufferDECOMPandDIGIT, 0 /*cc.N * decomp_limbs*/);
	}
}

void LimbPartition::mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2, const LimbPartition& partition3, const LimbPartition& partition4) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	assert(limbsize <= partition1.limb.size());
	assert(limbsize <= partition2.limb.size());
	assert(limbsize <= partition3.limb.size());
	assert(limbsize <= partition4.limb.size());

	s.wait(partition1.getS());
	s.wait(partition2.getS());
	s.wait(partition3.getS());
	s.wait(partition4.getS());

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		mult1AddMult23Add4_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i, partition3.limbptr.data + i, partition4.limbptr.data + i);
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	partition1.getS().wait(s);
	partition2.getS().wait(s);
	partition3.getS().wait(s);
	partition4.getS().wait(s);
}

void LimbPartition::multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1, const LimbPartition& in, const LimbPartition& aux) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	assert(limbsize <= c0.limb.size());
	assert(limbsize <= bc0.limb.size());
	assert(limbsize <= bc1.limb.size());
	assert(limbsize <= in.limb.size());
	assert(limbsize <= aux.limb.size());

	s.wait(c0.getS());
	s.wait(bc0.getS());
	s.wait(bc1.getS());
	s.wait(in.getS());
	s.wait(aux.getS());
	c0.getS().wait(aux.getS());

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		multnomoddownend_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  PARTITION(id, i), limbptr.data + i, c0.limbptr.data + i, bc0.limbptr.data + i, bc1.limbptr.data + i, in.limbptr.data + i, aux.limbptr.data + i);
	}
	this->copySpecialLimb(in);
	c0.copySpecialLimb(aux);
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	aux.getS().wait(c0.getS());
	aux.getS().wait(s);
	in.getS().wait(s);
	bc1.getS().wait(s);
	bc0.getS().wait(s);
	c0.getS().wait(s);
}

void LimbPartition::mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	assert(limbsize <= partition1.limb.size());
	assert(limbsize <= partition2.limb.size());

	s.wait(partition1.getS());
	s.wait(partition2.getS());

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		mult1Add2_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i);
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	partition1.getS().wait(s);
	partition2.getS().wait(s);
}

void LimbPartition::generateLimbSingleMalloc() {
	cudaSetDevice(device);
	const int limbsize = meta.size();

	assert(limbsize <= meta.size());
	if (bufferLIMB == nullptr) {
		assert(limb.size() == 0);

		bufferLIMB_bytes = cc.N * limbsize * 2 * sizeof(uint64_t);
		bufferLIMB		 = (uint64_t*)GPUmalloc(device, (int)bufferLIMB_bytes, s.ptr());
		// cudaMallocAsync(&bufferLIMB, std::max(1ul, cc.N * limbsize * 2 * sizeof(uint64_t)), s.ptr());
	}

	limb.clear();

	// generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, bufferLIMB, 0, bufferLIMB, cc.N * (limbsize));
	generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, nullptr, 0, nullptr, cc.N * (limbsize));
}

void LimbPartition::generateLimbArena(const int limbsize) {
	cudaSetDevice(device);
	if (!limb.empty() || bufferLIMB != nullptr)
		throw std::logic_error("FIDESlib: generateLimbArena needs a freshly constructed partition (no limbs, no buffer)");
	if (limbsize <= 0 || (size_t)limbsize > meta.size())
		throw std::logic_error("FIDESlib: generateLimbArena: limb count out of range");
	// The exact-size pool class (GPUmallocExact): the arena costs its payload, N * limbsize * 8 bytes,
	// not the next power of two (the pow2 class costs 8 MB per 5.5 MB arena, GBs of peak memory with
	// many resident plaintexts). Every limb record here is 64-bit (meta is the Q chain).
	bufferLIMB_bytes  = (size_t)cc.N * (size_t)limbsize * sizeof(uint64_t);
	bufferLIMB_pooled = true; // the destructor frees it through GPUfreeExact
	bufferLIMB		  = (uint64_t*)GPUmallocExact(id, bufferLIMB_bytes, s.ptr());
	CudaCheckErrorModNoSync;
	generate(meta, limb, limbptr, limbsize - 1, nullptr /*no aux*/, bufferLIMB, 0, nullptr, 0);
}

void LimbPartition::generateLimbConstant() {
	cudaSetDevice(device);

	const int limbsize = getLimbSize(*level);
	assert(limb.size() == 0);
	assert(limbsize <= meta.size());

	if (bufferLIMB == nullptr) {
		// bufferLIMB = (uint64_t*)GPUmalloc(device, std::max(1ul, cc.N * limbsize * sizeof(uint64_t)), s.ptr());
		// cudaMallocAsync(&bufferLIMB, std::max(1ul, cc.N * limbsize * sizeof(uint64_t)), s.ptr());
	} else {
		// NOT `cc.N * limbsize * 2 * sizeof(uint64_t)`: `limbsize` here is getLimbSize(*level),
		// while generateLimbSingleMalloc sized this buffer from meta.size(). Only the recorded
		// size is the one GPUmalloc was given.
		assert(bufferLIMB_bytes > 0);
		if (bufferLIMB_pooled)
			GPUfreeExact(bufferLIMB, id, bufferLIMB_bytes, s.ptr());
		else
			GPUfree(bufferLIMB, id, (int)bufferLIMB_bytes, s.ptr());
		bufferLIMB		 = nullptr;
		bufferLIMB_bytes = 0;
		bufferLIMB_pooled = false;
		// cudaFreeAsync(&bufferLIMB, s.ptr());
		// bufferLIMB = (uint64_t*)GPUmalloc(device, std::max(1ul, cc.N * limbsize * sizeof(uint64_t)), s.ptr());
		// cudaMallocAsync(&bufferLIMB, std::max(1ul, cc.N * limbsize * sizeof(uint64_t)), s.ptr());
	}

	// limb.clear();
	// generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, bufferLIMB, 0, nullptr, 0);
	generate(meta, limb, limbptr, (int)limbsize - 1, nullptr /*&auxptr*/, nullptr, 0, nullptr, 0);
}

void LimbPartition::aliasLimbsOf(const LimbPartition& src) {
	cudaSetDevice(device);
	if (src.device != device || src.id != id)
		throw std::logic_error("FIDESlib: a limb view must sit on the partition it aliases");
	if (!limb.empty() || !SPECIALlimb.empty() || bufferLIMB != nullptr || bufferSPECIAL != nullptr)
		throw std::logic_error("FIDESlib: a limb view must be built on a freshly constructed partition");
	const size_t qlimbs = getLimbSize(*level);
	if (qlimbs > src.limb.size())
		throw std::logic_error("FIDESlib: a limb view cannot hold more limbs than its source");

	// The source's limbs were written on its stream; the pointer table below is uploaded on ours.
	s.wait(src.getS());

	auto alias = [&](const LimbImpl& from, std::vector<LimbImpl>& into, std::vector<void*>& ptrs) {
		if (from.index() == U32) {
			const Limb<uint32_t>& l = std::get<U32>(from);
			into.emplace_back(Limb<uint32_t>(cc, l.v.data, 0, id, s, l.primeid, nullptr, 0));
			ptrs.push_back(std::get<U32>(into.back()).v.data);
		} else {
			const Limb<uint64_t>& l = std::get<U64>(from);
			into.emplace_back(Limb<uint64_t>(cc, l.v.data, 0, id, s, l.primeid, nullptr, 0));
			ptrs.push_back(std::get<U64>(into.back()).v.data);
		}
	};

	std::vector<void*> cpu_ptr;
	cpu_ptr.reserve(qlimbs);
	for (size_t i = 0; i < qlimbs; ++i)
		alias(src.limb[i], limb, cpu_ptr);
	if (!cpu_ptr.empty())
		UploadH2D(limbptr.data, cpu_ptr.data(), cpu_ptr.size() * sizeof(void*), device, s.ptr());

	std::vector<void*> cpu_special;
	cpu_special.reserve(src.SPECIALlimb.size());
	for (const auto& l : src.SPECIALlimb)
		alias(l, SPECIALlimb, cpu_special);
	if (!cpu_special.empty())
		UploadH2D(SPECIALlimbptr.data, cpu_special.data(), cpu_special.size() * sizeof(void*), device, s.ptr());
	CudaCheckErrorModNoSync;
}

void LimbPartition::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli) {
	cudaSetDevice(device);
	int limb_size = getLimbSize(*level);

	if (cc.GPUid.size() == 1) {

		for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
			for (auto& j : DECOMPlimb.at(i)) {
				for (size_t k = 0; k < data.at(i).size(); ++k) {
					if (cc.precom.constants[id].primes[PRIMEID(j)] == moduli.at(i).at(k)) {
						STREAM(j).wait(s);
						SWITCH(j, load(data.at(i).at(k)));
						k = data.at(i).size();
					}
				}
			}
		}
		std::vector<void*> cpu_ptr(MAXP, nullptr);
		for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
			for (auto& j : DECOMPlimb.at(i)) {
				for (size_t k = 0; k < meta.size(); ++k) {
					if (PRIMEID(j) == meta.at(k).id) {
						if (j.index() == U64) {
							cpu_ptr[k] = std::get<U64>(j).v.data;
						} else {
							cpu_ptr[k] = std::get<U32>(j).v.data;
						}
					}
				}
			}
		}
		UploadH2D(limbptr.data, cpu_ptr.data(), cpu_ptr.size() * sizeof(void*), device, s.ptr());
	} else {
		for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
			for (int32_t j = 0; j < limb_size; ++j) {
				if (meta[j].digit == static_cast<int32_t>(i)) {
					for (size_t k = 0; k < data.at(i).size(); ++k) {
						if (cc.precom.constants[id].primes[PRIMEID(limb[j])] == moduli.at(i).at(k)) {
							STREAM(limb[j]).wait(s);
							SWITCH(limb[j], load(data.at(i).at(k)));
							k = data.at(i).size();
						}
					}
				}
			}
		}
	}

	for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
		for (auto& j : DIGITlimb.at(i)) {
			for (size_t k = 0; k < data.at(i).size(); ++k) {
				if (cc.precom.constants[id].primes[PRIMEID(j)] == moduli.at(i).at(k)) {
					STREAM(j).wait(s);
					SWITCH(j, load(data.at(i).at(k)));
					k = data.at(i).size();
				}
			}
		}
	}
}

/** TODO: deprecate towards fused version */

void LimbPartition::dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace, const LimbPartition* limbsrc) {
	cudaSetDevice(device);
	constexpr bool PRINT = false;
	s.wait(src.getS());
	s.wait(ksk.getS());
	const int limbsize = *level + 1;
	assert(limbsize <= limb.size());
	assert(limbsize <= src.limb.size());

	if constexpr (0) {
		std::map<int, int> used;
		for (size_t i = 0; i < src.DIGITlimb.size(); ++i) {

			{
				int start = 0;
				for (int j = 0; j < i; ++j)
					start += src.DECOMPlimb[j].size();
				int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
				if (size <= 0)
					break;
			}

			for (size_t j = 0; j < ksk.DECOMPlimb.at(i).size(); ++j) {

				int primeid = PRIMEID(ksk.DECOMPlimb.at(i).at(j));

				for (size_t k = 0; k < limbsize; ++k) {
					auto& l = src.limb.at(k);
					if (PRIMEID(l) == primeid) {
						// STREAM(limb.at(k)).wait(s);
						// STREAM(limb.at(k)).wait(STREAM(ksk.DECOMPlimb.at(i).at(j)));
						// STREAM(limb.at(k)).wait(STREAM(l));

						if (!used[primeid]) {
							SWITCH(limb.at(k), mult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
							used[primeid]++;
							if constexpr (PRINT)
								std::cout << "Init " << primeid; //<< std::endl;
						} else {
							SWITCH(limb.at(k), addMult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
							if constexpr (PRINT)
								std::cout << "Acc " << primeid; // << std::endl;
						}
						if constexpr (PRINT)
							SWITCH(limb.at(k), printThisLimb(1));
						if constexpr (PRINT)
							SWITCH(l, printThisLimb(1));
					}
				}

				if constexpr (PRINT)
					SWITCH(ksk.DECOMPlimb.at(i).at(j), printThisLimb(1));
			}

			CudaCheckErrorModNoSync;
			for (size_t j = 0; j < src.DIGITlimb.at(i).size(); ++j) {
				int primeid = PRIMEID(src.DIGITlimb.at(i).at(j));

				if (primeid < cc.precom.constants[id].L) {
					for (auto& l : limb) {
						if (PRIMEID(l) == primeid) {
							// STREAM(l).wait(s);
							// STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
							// STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));

							if (!used[primeid]) {
								SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
								used[primeid]++;
								if constexpr (PRINT)
									std::cout << "Init2 " << primeid; // << std::endl;
							} else

							{
								SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
								if constexpr (PRINT)
									std::cout << "Acc2 " << primeid; // << std::endl;
							}

							if constexpr (PRINT)
								SWITCH(l, printThisLimb(1));
						}
					}
				} else {

					for (auto& l : SPECIALlimb) {
						if (PRIMEID(l) == primeid) {
							// STREAM(l).wait(s);
							// STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
							// STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));
							if (!used[primeid]) {
								SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
								used[primeid]++;
								if constexpr (PRINT)
									std::cout << "Init3 " << primeid; // << std::endl;
							} else {
								SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
								if constexpr (PRINT)
									std::cout << "Acc3 " << primeid; // << std::endl;
							}
							if constexpr (PRINT)
								SWITCH(l, printThisLimb(1));
						}
					}
				}
				if constexpr (PRINT)
					SWITCH(src.DIGITlimb.at(i).at(j), printThisLimb(1));
				if constexpr (PRINT)
					SWITCH(ksk.DIGITlimb.at(i).at(j), printThisLimb(1));
			}

			if constexpr (PRINT) {
				for (auto& i : limb) {
					if constexpr (PRINT)
						SWITCH(i, printThisLimb(1));
				}
				for (auto& i : SPECIALlimb) {
					if constexpr (PRINT)
						SWITCH(i, printThisLimb(1));
				}
			}
		}

		for (auto& l : limb)
			s.wait(STREAM(l));
		for (auto& l : SPECIALlimb)
			s.wait(STREAM(l));
	} else {

		int start	= 0;
		int special = SPECIALmeta.size();

		for (uint32_t i = 0; i < DECOMPmeta.size(); ++i) {
			int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
			if (size <= 0) {
				//  std::cout << "Out on " << i << std::endl;
				break;
			}
			Mult_<<<{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, s.ptr()>>>(
			  inplace ? auxptr.data + start : limbptr.data + start, ksk.DECOMPlimbptr[i].data, limbsrc ? limbsrc->limbptr.data + start : src.limbptr.data + start, start);
			start += DECOMPmeta[i].size();
		}

		start = 0;
		for (uint32_t i = 0; i < DIGITmeta.size(); ++i) {
			if (start >= limbsize) {
				// std::cout << "Out on " << i << std::endl;
				break;
			}
			if (start > 0) {
				int size = start;
				addMult_<<<{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, s.ptr()>>>(
				  inplace ? auxptr.data : limbptr.data, ksk.DIGITlimbptr[i].data + special, src.DIGITlimbptr[i].data + special, 0);
			}
			start += DECOMPmeta[i].size();
			if (start < limbsize) {
				int size = limbsize - start;
				addMult_<<<{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, s.ptr()>>>(inplace ? auxptr.data + start : limbptr.data + start,
				  ksk.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(),
				  src.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(),
				  start);
			}
		}

		start = 0;
		for (uint32_t i = 0; i < DIGITmeta.size(); ++i) {
			if (start >= limbsize)
				break;
			start += DECOMPmeta.at(i).size();
			if (i == 0) {
				Mult_<<<{ (uint32_t)cc.N / 128, (uint32_t)special }, 128, 0, s.ptr()>>>(
				  inplace ? SPECIALauxptr.data : SPECIALlimbptr.data, ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data, SPECIAL(id, 0));
			} else {
				addMult_<<<{ (uint32_t)cc.N / 128, (uint32_t)special }, 128, 0, s.ptr()>>>(
				  inplace ? SPECIALauxptr.data : SPECIALlimbptr.data, ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data, SPECIAL(id, 0));
			}
		}
	}

	src.getS().wait(s);
	ksk.getS().wait(s);

	if (inplace) {
		for (auto& l : limb) {
			if (l.index() == U32) {
				std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
			} else {
				std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
			}
		}
		std::swap(limbptr.data, auxptr.data);

		for (auto& l : SPECIALlimb) {
			if (l.index() == U32) {
				std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
			} else {
				std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
			}
		}
		std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
	}

	if constexpr (PRINT) {
		for (auto& i : limb) {
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
		for (auto& i : SPECIALlimb) {
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
}

void LimbPartition::multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0, const LimbPartition& c0tilde, const LimbPartition& ksk_a, const LimbPartition& ksk_b) {

	const int level_plus_1 = *level + 1;
	constexpr ALGO algo	   = ALGO_SHOUP;
	constexpr bool PRINT   = false;
	assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
	assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
	cudaSetDevice(device);

	// std::map<int, int> used;
	s.wait(c0.getS());
	s.wait(c1.getS());
	s.wait(c0tilde.getS());
	s.wait(c1tilde.getS());
	s.wait(ksk_a.getS());
	s.wait(ksk_b.getS());

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			if (d == 0) {
				std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
				SWITCH(limb[0], printThisLimb());
			}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			for (int i = 0; i < size; i += cc.batch) {
				STREAM(limb.at(start + i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				INTT_<false, algo, INTT_MULT_AND_SAVE>
				  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(getGlobals(),
					c1.limbptr.data + start + i,
					start + i,
					c1.auxptr.data + start + i,
					c1tilde.limbptr.data + start + i,
					c0.limbptr.data + start + i,
					c1.limbptr.data + start + i,
					ksk_a.DECOMPlimbptr[d].data + i,
					ksk_b.DECOMPlimbptr[d].data + i,
					c0.limbptr.data + start + i,
					c0tilde.limbptr.data + start + i);

				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
			}
			for (int32_t i = 0; i < size; i += cc.batch) {
				s.wait(STREAM(limb.at(start + i)));
			}
		}

		if constexpr (PRINT) {
			std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
			SWITCH(DECOMPlimb[d][0], printThisLimb());
		}

		{
			dim3 blockSize{ 64, 2 };
			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
			DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
		}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			int size = c0.SPECIALlimb.size();
			for (int i = 0; i < size; i += cc.batch) {
				STREAM(c0.SPECIALlimb.at(i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
				  getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

				if (d == 0) {
					NTT_<true, algo, NTT_KSK_DOT>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				} else {
					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				}
			}
		}
	}

	// cudaDeviceSynchronize();

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}

		if constexpr (1) // batched
		{
			int start = 0;
			for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
				if (j == d)
					continue;

				int Dstart = start + c0.SPECIALlimb.size();
				int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

				int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
				if (size <= 0)
					break;

				constexpr int M = 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				for (int i = 0; i < size; i += cc.batch) {
					STREAM(c0.limb.at(Lstart + i)).wait(s);
					uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

					NTT_<false, algo, NTT_NONE>
					  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
						getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr, nullptr);

					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(getGlobals(),
						c1.auxptr.data + Lstart + i,
						Lstart + i,
						c0.limbptr.data + Lstart + i,
						ksk_a.DIGITlimbptr[d].data + Dstart + i,
						0,
						c1.limbptr.data + Lstart + i,
						ksk_b.DIGITlimbptr[d].data + Dstart + i);
				}

				start += DECOMPlimb[j].size();
			}
		}

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
	}
	for (auto& l : c0.limb)
		s.wait(STREAM(l));
	for (auto& l : c0.SPECIALlimb)
		s.wait(STREAM(l));

	c0.getS().wait(s);
	c1.getS().wait(s);
	c0tilde.getS().wait(s);
	c1tilde.getS().wait(s);
	ksk_a.getS().wait(s);
	ksk_b.getS().wait(s);
}

size_t LimbPartition::getLimbSize(int level) {
	size_t size = 0;
	if (level == cc.L + 1) {
		level = level - 1 - (cc.rescaleTechnique == FIDESlib::CKKS::FLEXIBLEAUTOEXT);
	}
	while (size < meta.size() && meta[size].id <= level) {
		// assert(limb.size() > size);
		++size;
	}
	return size;
}

void LimbPartition::rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b) {

	const int level_plus_1 = *level + 1;
	constexpr ALGO algo	   = ALGO_SHOUP;
	constexpr bool PRINT   = false;
	assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
	assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
	cudaSetDevice(device);

	// std::map<int, int> used;
	s.wait(c0.getS());
	s.wait(c1.getS());
	s.wait(ksk_a.getS());
	s.wait(ksk_b.getS());

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			if (d == 0) {
				std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
				SWITCH(limb[0], printThisLimb());
			}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			for (int i = 0; i < size; i += cc.batch) {
				STREAM(limb.at(start + i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				INTT_<false, algo, INTT_ROTATE_AND_SAVE>
				  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(getGlobals(),
					c1.limbptr.data + start + i,
					start + i,
					c1.auxptr.data + start + i,
					nullptr,
					c0.limbptr.data + start + i,
					c1.limbptr.data + start + i,
					ksk_a.DECOMPlimbptr[d].data + i,
					ksk_b.DECOMPlimbptr[d].data + i,
					c0.limbptr.data + start + i,
					nullptr);

				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
			}
			for (int32_t i = 0; i < size; i += cc.batch) {
				s.wait(STREAM(limb.at(start + i)));
			}
		}

		if constexpr (PRINT) {
			std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
			SWITCH(DECOMPlimb[d][0], printThisLimb());
		}

		{
			dim3 blockSize{ 64, 2 };
			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
			DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
		}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			int size = c0.SPECIALlimb.size();
			for (int i = 0; i < size; i += cc.batch) {
				STREAM(c0.SPECIALlimb.at(i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
				  getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

				if (d == 0) {
					NTT_<true, algo, NTT_KSK_DOT>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				} else {
					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				}
			}
		}
	}

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}

		if constexpr (1) // batched
		{
			int start = 0;
			for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
				if (j == d)
					continue;

				int Dstart = start + c0.SPECIALlimb.size();
				int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

				int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
				if (size <= 0)
					break;

				constexpr int M = 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				for (int i = 0; i < size; i += cc.batch) {
					STREAM(c0.limb.at(Lstart + i)).wait(s);
					uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

					NTT_<false, algo, NTT_NONE>
					  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
						getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr, nullptr);

					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(getGlobals(),
						c1.auxptr.data + Lstart + i,
						Lstart + i,
						c0.limbptr.data + Lstart + i,
						ksk_a.DIGITlimbptr[d].data + Dstart + i,
						0,
						c1.limbptr.data + Lstart + i,
						ksk_b.DIGITlimbptr[d].data + Dstart + i);
				}

				start += DECOMPlimb[j].size();
			}
		}

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
	}

	for (auto& l : c0.limb)
		s.wait(STREAM(l));
	for (auto& l : c0.SPECIALlimb)
		s.wait(STREAM(l));

	c0.getS().wait(s);
	c1.getS().wait(s);
	ksk_a.getS().wait(s);
	ksk_b.getS().wait(s);
}

void LimbPartition::squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b) {

	const int level_plus_1 = *level + 1;
	constexpr ALGO algo	   = ALGO_SHOUP;
	constexpr bool PRINT   = false;
	assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
	assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
	cudaSetDevice(device);

	// std::map<int, int> used;
	s.wait(c0.getS());
	s.wait(c1.getS());
	s.wait(ksk_a.getS());
	s.wait(ksk_b.getS());

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			if (d == 0) {
				std::cout << cc.precom.constants[id].primes[PRIMEID(limb[0])] << ": ";
				SWITCH(limb[0], printThisLimb());
			}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			for (int i = 0; i < size; i += cc.batch) {
				STREAM(limb.at(start + i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				INTT_<false, algo, INTT_SQUARE_AND_SAVE>
				  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(getGlobals(),
					c1.limbptr.data + start + i,
					start + i,
					c1.auxptr.data + start + i,
					nullptr,
					c0.limbptr.data + start + i,
					c1.limbptr.data + start + i,
					ksk_a.DECOMPlimbptr[d].data + i,
					ksk_b.DECOMPlimbptr[d].data + i,
					c0.limbptr.data + start + i,
					nullptr);

				INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
				  getGlobals(), c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
			}
			for (int32_t i = 0; i < size; i += cc.batch) {
				s.wait(STREAM(limb.at(start + i)));
			}
		}

		if constexpr (PRINT) {
			std::cout << cc.precom.constants[id].primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
			SWITCH(DECOMPlimb[d][0], printThisLimb());
		}

		{
			dim3 blockSize{ 64, 2 };
			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
			DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(DECOMPlimbptr[d].data, level_plus_1, DIGITlimbptr[d].data, digitid[d], getGlobals());
		}

		if constexpr (1) { // Batched
			constexpr int M = 4;

			dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
			dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
			int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			int size = c0.SPECIALlimb.size();
			for (int i = 0; i < size; i += cc.batch) {
				STREAM(c0.SPECIALlimb.at(i)).wait(s);
				uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

				NTT_<false, algo, NTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(
				  getGlobals(), DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

				if (d == 0) {
					NTT_<true, algo, NTT_KSK_DOT>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				} else {
					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr()>>>(getGlobals(),
						c1.SPECIALauxptr.data + i,
						SPECIAL(id, i),
						c0.SPECIALlimbptr.data + i,
						ksk_a.DIGITlimbptr[d].data + i,
						0,
						c1.SPECIALlimbptr.data + i,
						ksk_b.DIGITlimbptr[d].data + i);
				}
			}
		}
	}

	for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPlimb[j].size();
		int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
		if (size <= 0)
			break;

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}

		if constexpr (1) // batched
		{
			int start = 0;
			for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
				if (j == d)
					continue;

				int Dstart = start + c0.SPECIALlimb.size();
				int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

				int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
				if (size <= 0)
					break;

				constexpr int M = 4;

				dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
				dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
				int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
				int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				for (int i = 0; i < size; i += cc.batch) {
					STREAM(c0.limb.at(Lstart + i)).wait(s);
					uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

					NTT_<false, algo, NTT_NONE>
					  <<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(
						getGlobals(), DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr, nullptr);

					NTT_<true, algo, NTT_KSK_DOT_ACC>
					  <<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(c0.limb.at(Lstart + i)).ptr()>>>(getGlobals(),
						c1.auxptr.data + Lstart + i,
						Lstart + i,
						c0.limbptr.data + Lstart + i,
						ksk_a.DIGITlimbptr[d].data + Dstart + i,
						0,
						c1.limbptr.data + Lstart + i,
						ksk_b.DIGITlimbptr[d].data + Dstart + i);
				}

				start += DECOMPlimb[j].size();
			}
		}

		if constexpr (PRINT)
			for (auto& i : DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
	}

	for (auto& l : c0.limb)
		s.wait(STREAM(l));
	for (auto& l : c0.SPECIALlimb)
		s.wait(STREAM(l));

	c0.getS().wait(s);
	c1.getS().wait(s);
	ksk_a.getS().wait(s);
	ksk_b.getS().wait(s);
}

template <ALGO algo> void LimbPartition::moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs) {
	assert(SPECIALlimb.size() == SPECIALmeta.size());
	const int limbsize = *level + 1;
	cudaSetDevice(device);
	constexpr bool PRINT = false;

	s.wait(auxLimbs.getS());
	{
		if constexpr (PRINT) {
			std::cout << "pre INTT Special GPU ";
			for (auto& i : SPECIALlimb) {
				SWITCH(i, printThisLimb(2));
			}
		}

		if (ntt) {
			for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
				STREAM(SPECIALlimb[i]).wait(s);
			}
			ApplyINTT<algo, INTT_NONE>(cc.batch, INTT_fusion_fields{}, SPECIALlimb, SPECIALlimbptr, SPECIALauxptr, cc, SPECIAL(id, 0), SPECIALlimb.size());
			for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
				s.wait(STREAM(SPECIALlimb[i]));
			}
		}

		if constexpr (PRINT) {
			std::cout << "post INTT Special GPU ";
			for (auto& i : SPECIALlimb) {
				SWITCH(i, printThisLimb(2));
			}
		}

		s.wait(auxLimbs.getS());

		{
			dim3 blockSize{ 64, 2 }; // blockSize.x * blockSize.y * blockSize.z <= 1024, blockSize.x a multiple of 32

			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;

			ModDown2<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(auxLimbs.limbptr.data, limbsize, SPECIALlimbptr.data, PARTITION(id, 0), getGlobals());
		}

		if constexpr (PRINT) {
			std::cout << "Output ModDown ";
			for (auto& i : auxLimbs.limb) {
				SWITCH(i, printThisLimb(2));
			}
		}

		for (int i = 0; i < limbsize; i += cc.batch) {
			STREAM(limb.at(i)).wait(s);
		}
		if (limbsize > 0)
			NTT<algo, NTT_MODDOWN>(cc.batch, false, NTT_fusion_fields{ .op2 = &auxLimbs });

		if constexpr (PRINT) {
			std::cout << "Output ModDown after sub mult.";
			for (auto& i : limb) {
				SWITCH(i, printThisLimb(2));
			}
		}

		for (int i = 0; i < limbsize; i += cc.batch) {
			s.wait(STREAM(limb.at(i)));
		}
	}
	auxLimbs.getS().wait(s);

	if (free_special_limbs) {
		freeSpecialLimbs();
	}
}

#define YY(algo) template void LimbPartition::moddown<algo>(LimbPartition & auxLimbs, bool ntt, bool free_special_limbs);
#include "ntt_types.inc"

#undef YY

void LimbPartition::automorph(const int index, const int br, LimbPartition* src, const bool ext) {
	cudaSetDevice(device);
	int limbsize = getLimbSize(*level);
	assert(src && "Don't use the inplace version of automorph");
	if (src)
		s.wait(src->getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		automorph_multi_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  src ? src->limbptr.data + i : limbptr.data + i, src ? limbptr.data + i : auxptr.data + i, index, br, PARTITION(id, i));
	}
	if (ext) {
		int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
		int num_limbs = cc.splitSpecialMeta.at(id).size();

		for (int32_t i = start; i < start + num_limbs; i += 1 /*cc.batch*/) {
			STREAM(SPECIALlimb[i - (SPECIALlimb.size() < cc.specialMeta.at(id).size()) * start]).wait(s);
			uint32_t size = std::min((int)start + num_limbs - (int)i, 1 /*cc.batch*/);
			automorph_multi_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i - (SPECIALlimb.size() < SPECIALmeta.size()) * start]).ptr()>>>(src ?
				src->SPECIALlimbptr.data + i - (src->SPECIALlimb.size() < SPECIALmeta.size()) * start :
				SPECIALlimbptr.data + i - (SPECIALlimb.size() < SPECIALmeta.size()) * start,
			  (src ? SPECIALlimbptr.data + i : SPECIALauxptr.data + i) - (SPECIALlimb.size() < SPECIALmeta.size()) * start,
			  index,
			  br,
			  SPECIAL(0, i));
		}
		for (int32_t i = start; i < start + num_limbs; i += 1 /*cc.batch*/) {
			s.wait(STREAM(SPECIALlimb[i - (SPECIALlimb.size() < SPECIALmeta.size()) * start]));
		}

		if (!src) {
			for (auto& i : SPECIALlimb) {
				if (i.index() == U32) {
					std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
				} else {
					std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
				}
			}
			std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
		}
	}
	if (!src) {
		for (auto& i : limb) {
			if (i.index() == U32) {
				std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
			} else {
				std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
			}
		}
		std::swap(limbptr.data, auxptr.data);
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	if (src)
		src->getS().wait(s);
}

void LimbPartition::modupInto(LimbPartition& partition, LimbPartition& aux_partition) {
	constexpr ALGO algo	 = ALGO_SHOUP;
	constexpr bool PRINT = false;
	// assert(SPECIALlimb.empty());
	cudaSetDevice(device);

	const int limbsize = *level + 1;

	s.wait(partition.getS());
	s.wait(aux_partition.getS());

	if constexpr (PRINT)
		for (auto& i : limb) {
			SWITCH(i, printThisLimb(2));
		}

	for (size_t d = 0; d < DECOMPmeta.size(); ++d) {

		int start = 0;
		for (uint32_t j = 0; j < d; ++j)
			start += DECOMPmeta[j].size();
		int size = std::min((int)DECOMPmeta[d].size(), limbsize - start);
		if (size <= 0)
			break;

		constexpr int M = 4;

		dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
		dim3 blockDimSecond = dim3{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		int bytesFirst		= 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
		int bytesSecond		= 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		for (int i = 0; i < size; i += cc.batch) {
			STREAM(limb.at(start + i)).wait(s);
			uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

			INTT_<false, algo, INTT_NONE><<<dim3{ cc.N / (blockDimFirst.x * M * 2), num_limbs }, blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr()>>>(
			  getGlobals(), limbptr.data + start + i, PARTITION(id, start + i), auxptr.data + start + i);

			INTT_<true, algo, INTT_NONE><<<dim3{ cc.N / (blockDimSecond.x * M * 2), num_limbs }, blockDimSecond, bytesSecond, STREAM(limb.at(start + i)).ptr()>>>(
			  getGlobals(), auxptr.data + start + i, PARTITION(id, start + i), partition.DECOMPlimbptr[d].data + i);
		}
		for (int32_t i = 0; i < size; i += cc.batch) {
			s.wait(STREAM(limb.at(start + i)));
		}
		if constexpr (PRINT)
			for (auto& i : partition.DECOMPlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}

		{
			dim3 blockSize{ 64, 2 };
			dim3 gridSize{ (uint32_t)cc.N / blockSize.x };
			int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
			DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr()>>>(
			  partition.DECOMPlimbptr[d].data, *level + 1, partition.DIGITlimbptr[d].data, digitid[d], getGlobals());
		}

		if constexpr (PRINT)
			for (auto& i : partition.DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}

		const int digitsize = cc.precom.constants[id].num_primeid_digit_to[digitid.at(d)][*level];

		for (int32_t i = 0; i < digitsize; i += cc.batch) {
			STREAM(partition.DIGITlimb.at(d).at(i)).wait(s);
		}

		ApplyNTT<algo, NTT_NONE>(
		  cc.batch, NTT_fusion_fields{}, partition.DIGITlimb.at(d), partition.DIGITlimbptr.at(d), aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

		for (int32_t i = 0; i < digitsize; i += cc.batch) {
			s.wait(STREAM(partition.DIGITlimb.at(d).at(i)));
		}

		if constexpr (PRINT)
			for (auto& i : partition.DIGITlimb[d]) {
				SWITCH(i, printThisLimb(2));
			}
	}

	aux_partition.getS().wait(s);
	partition.getS().wait(s);
}

void LimbPartition::multScalar(std::vector<uint64_t>& vector) {
	cudaSetDevice(device);
	/*
	cudaDeviceSynchronize();
	for (auto& l : limb) {
		if (l.index() == U64) {
			scalar_mult_<uint64_t, ALGO_BARRETT>
				<<<cc.N / 128, 128, 0, STREAM(l).ptr()>>>(std::get<U64>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
		} else {
			scalar_mult_<uint32_t, ALGO_BARRETT>
				<<<cc.N / 128, 128, 0, STREAM(l).ptr()>>>(std::get<U32>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
		}
	}
	cudaDeviceSynchronize();
	 */
	const int limbsize = getLimbSize(*level);

	uint64_t* elems = static_cast<uint64_t*>(FIDESlib::OpMallocAsync(vector.size() * sizeof(uint64_t), s.ptr()));
	// cudaMalloc(&elems, vector.size() * sizeof(uint64_t));
	UploadH2D(elems, vector.data(), vector.size() * sizeof(uint64_t), device, s.ptr());

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		Scalar_mult_<ALGO_BARRETT><<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, elems, PARTITION(id, i), nullptr);
	}
	if (*level == cc.L + 1 && SPECIALmeta.size() > 0 && SPECIALmeta.at(0).id == cc.L + 1) {
		Scalar_mult_<ALGO_BARRETT><<<dim3{ (uint32_t)cc.N / 128, 1 }, 128, 0, STREAM(SPECIALlimb[0]).ptr()>>>(SPECIALlimbptr.data, elems, SPECIAL(id, 0), nullptr);
		s.wait(STREAM(SPECIALlimb[0]));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	FIDESlib::OpFreeAsync(elems, s.ptr());
}

void LimbPartition::addScalar(std::vector<uint64_t>& vector) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	uint64_t* elems = static_cast<uint64_t*>(FIDESlib::OpMallocAsync(vector.size() * sizeof(uint64_t), s.ptr()));
	// cudaMalloc(&elems, vector.size() * sizeof(uint64_t));
	UploadH2D(elems, vector.data(), vector.size() * sizeof(uint64_t), device, s.ptr());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		int primeid_init   = PARTITION(id, i);
		scalar_add_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, elems, primeid_init);
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	FIDESlib::OpFreeAsync(elems, s.ptr());
}

void LimbPartition::subScalar(std::vector<uint64_t>& vector) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	uint64_t* elems = static_cast<uint64_t*>(FIDESlib::OpMallocAsync(vector.size() * sizeof(uint64_t), s.ptr()));
	// cudaMalloc(&elems, vector.size() * sizeof(uint64_t));
	UploadH2D(elems, vector.data(), vector.size() * sizeof(uint64_t), device, s.ptr());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		scalar_sub_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, elems, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	FIDESlib::OpFreeAsync(elems, s.ptr());
}

void LimbPartition::add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b) {
	cudaSetDevice(device);
	s.wait(a.getS());
	s.wait(b.getS());

	const int limbsize = getLimbSize(*level);

	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		if (!ext_a && ext_b) {
			addScaleB_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
			  limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
		} else if (!ext_b && ext_a) {
			addScaleB_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
			  limbptr.data + i, b.limbptr.data + i, a.limbptr.data + i, PARTITION(id, i));
		} else {
			add_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
		}
	}

	if (ext_a || ext_b) {
		int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
		int num_limbs = cc.splitSpecialMeta.at(id).size();
		for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
			STREAM(SPECIALlimb[i]).wait(s);
			uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

			if (!ext_a && ext_b) {
				copy_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(b.SPECIALlimbptr.data + i,
				  SPECIALlimbptr.data + i); // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
			} else if (!ext_b && ext_a) {
				copy_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(a.SPECIALlimbptr.data + i,
				  SPECIALlimbptr.data + i); // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
			} else {
				add_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(SPECIALlimbptr.data + i,
				  a.SPECIALlimbptr.data + i,
				  b.SPECIALlimbptr.data + i,
				  SPECIAL(id,
					i)); // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
			}
		}
		for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
			s.wait(STREAM(SPECIALlimb[i]));
		}
	}

	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	a.getS().wait(s);
	b.getS().wait(s);
}

void LimbPartition::squareElement(const LimbPartition& p) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	s.wait(p.getS());
	int size = std::min(limbsize, (int)p.limb.size());
	for (int i = 0; i < size; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)size - i, cc.batch);
		square_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < size; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	p.getS().wait(s);
}

void LimbPartition::binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0, const LimbPartition& c2_key_switched_1) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	s.wait(c0_res.getS());
	s.wait(c2_key_switched_0.getS());
	s.wait(c2_key_switched_1.getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		binomial_square_fold_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(
		  c0_res.limbptr.data + i, c2_key_switched_0.limbptr.data + i, limbptr.data + i, c2_key_switched_1.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	c0_res.getS().wait(s);
	c2_key_switched_0.getS().wait(s);
	c2_key_switched_1.getS().wait(s);
}

void LimbPartition::dropLimb() {
	cudaSetDevice(device);

	STREAM(limb.back()).wait(s);
	limb.pop_back();
}

void LimbPartition::freeNTTScratch() {
	cudaSetDevice(device);
	if (limb.empty())
		return;

	// Each limb reports what its scratch slot should now hold: nullptr if it was released, the
	// unchanged address if its aux is a view onto a shared buffer (which freeAux leaves alone).
	std::vector<void*> cpu_auxptr(limb.size(), nullptr);
	bool all_released = true;
	for (size_t i = 0; i < limb.size(); ++i) {
		SWITCH_RET(limb[i], freeAux(s), cpu_auxptr[i]);
		if (cpu_auxptr[i] != nullptr)
			all_released = false;
	}

	// Write the table the transform reads its scratch addresses from, so a later transform of this
	// polynomial faults on a null pointer instead of writing into freed memory.
	//
	// "Never transformed again" is not an assumption this path introduces — it is a load-bearing
	// invariant the library already runs on, and more strictly than here. A constant-grown limb has
	// no scratch at all (aux size 0), and generateLimbConstant passes generate() a null auxptrs, so
	// for a host-encoded plaintext this table is never written AT ALL: it holds whatever the
	// partition's pointer arena was allocated with. Transforming such a plaintext would already
	// scribble through an uninitialised address. Nulling the slots here is therefore strictly safer
	// than the state a loadConstant plaintext has always been in, not a new hazard.
	if (all_released) {
		// The common case, and the only one today's caller produces. A memset needs no host buffer,
		// so nothing here outlives the async copy.
		cudaMemsetAsync(auxptr.data, 0, limb.size() * sizeof(void*), s.ptr());
	} else {
		UploadH2D(auxptr.data, cpu_auxptr.data(), limb.size() * sizeof(void*), device, s.ptr());
		// cpu_auxptr is about to go out of scope and the copy is from pageable host memory, so the
		// upload has to have consumed it before we return. Only the mixed case pays this, and no
		// caller reaches it today.
		cudaStreamSynchronize(s.ptr());
	}
	CudaCheckErrorModNoSync;
}

void LimbPartition::addMult(const LimbPartition& a, const LimbPartition& b) {
	const int limbsize = getLimbSize(*level);
	assert(a.limb.size() >= limbsize);
	assert(b.limb.size() >= limbsize);
	cudaSetDevice(device);
	s.wait(a.getS());
	s.wait(b.getS());
	for (int i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
		addMult_<<<dim3{ (uint32_t)cc.N / 128, num_limbs }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
	}
	for (int i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	a.getS().wait(s);
	b.getS().wait(s);
}

void LimbPartition::broadcastLimb0() {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	assert(limbsize - 1 > 0);
	broadcastLimb0_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)limbsize - 1 }, 128, 0, s.ptr()>>>(limbptr.data);

	if (MODRAISE_WITH_P0) {
		// if (I HAVE P0)
		broadcastLimb0_mgpu_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)1 }, 128, 0, s.ptr()>>>(SPECIALlimbptr.data, SPECIAL(id, 0), limbptr.data);
	}
}

void LimbPartition::evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	for (uint32_t i = 0; i < n; ++i) {
		s.wait(ps[i]->getS());
	}

	// FIDESLIB_PERSIST_TABLES (on by default in the concurrent mode): these two per-call tables are the
	// reason this op is in the suspect class -- see CudaUtils.cuh PersistOpTables().
	const bool persist_tables = FIDESlib::PersistOpTables() || FIDESlib::PersistChurn();
	if (FIDESlib::WsumSrcSync())
		for (uint32_t i = 0; i < n; ++i)
			cudaStreamSynchronize(ps[i]->getS().ptr());   // DIAG: full source drain
	uint64_t* elems;
	if (persist_tables) {
		elems = static_cast<uint64_t*>(FIDESlib::OpTableBuffer(device, weights.size() * sizeof(uint64_t), 0, s.ptr()));
		if (FIDESlib::PersistChurn()) {   // DIAG: keep the allocator traffic, not the block
			void* dummy = FIDESlib::OpMallocAsync(weights.size() * sizeof(uint64_t), s.ptr());
			FIDESlib::OpFreeAsync(dummy, s.ptr());
		}
	} else {
		elems = static_cast<uint64_t*>(FIDESlib::OpMallocAsync(weights.size() * sizeof(uint64_t), s.ptr()));
		if (FIDESlib::TableTrace())
			FIDESlib::TableTraceAlloc(elems, "evalLinearWSum.weights");
	}
	UploadH2D(elems, weights.data(), weights.size() * sizeof(uint64_t), device, s.ptr());
	std::vector<void**> psptr(n, nullptr);
	for (uint32_t i = 0; i < n; ++i) {
		psptr[i] = ps[i]->limbptr.data;
		assert(ps[i]->limb.size() >= limbsize);
	}
	void*** d_psptr;
	if (persist_tables) {
		d_psptr = static_cast<void***>(FIDESlib::OpTableBuffer(device, psptr.size() * sizeof(void**), 1, s.ptr()));
		if (FIDESlib::PersistChurn()) {
			void* dummy = FIDESlib::OpMallocAsync(psptr.size() * sizeof(void**), s.ptr());
			FIDESlib::OpFreeAsync(dummy, s.ptr());
		}
	} else {
		d_psptr = static_cast<void***>(FIDESlib::OpMallocAsync(psptr.size() * sizeof(void**), s.ptr()));
		if (FIDESlib::TableTrace())
			FIDESlib::TableTraceAlloc(d_psptr, "evalLinearWSum.psptr");
	}
	UploadH2D(d_psptr, psptr.data(), psptr.size() * sizeof(void**), device, s.ptr());

	if (!limb.empty() && limbsize > 0)
		eval_linear_w_sum_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)limbsize }, 128, 0, s.ptr()>>>(n, limbptr.data, d_psptr, elems, PARTITION(id, 0));
	if (!persist_tables) {
		FIDESlib::OpFreeAsync(elems, s.ptr());
		FIDESlib::OpFreeAsync(d_psptr, s.ptr());
		if (FIDESlib::TableTrace()) {
			FIDESlib::TableTraceFree(elems, s.ptr());
			FIDESlib::TableTraceFree(d_psptr, s.ptr());
		}
	} else {
		FIDESlib::OpTableRelease(device, elems, 0, s.ptr());
		FIDESlib::OpTableRelease(device, d_psptr, 1, s.ptr());
	}
	for (uint32_t i = 0; i < n; ++i) {
		ps[i]->getS().wait(s);
	}
}

/**
  Only for MGPU key generation and extended limb partitions
 */
void LimbPartition::generatePartialSpecialLimb() {
	cudaSetDevice(device);
	if (SPECIALlimb.size() == 0 && cc.splitSpecialMeta.at(id).size() > 0 /*&& bufferSPECIAL == nullptr*/) {
		// if (bufferSPECIAL)
		//     GPUfree(bufferSPECIAL, id, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)),
		//             s.ptr());

		// bufferSPECIAL = (uint64_t*)GPUmalloc(
		//     id, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)), s.ptr());
		// cudaMallocAsync(&bufferSPECIAL, std::max(1ul, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t)),
		//                 s.ptr());
		// generate(cc.splitSpecialMeta.at(id), SPECIALlimb, SPECIALlimbptr, (int)cc.splitSpecialMeta.at(id).size() - 1,
		//          nullptr, bufferSPECIAL, 0);
		generate(cc.splitSpecialMeta.at(id), SPECIALlimb, SPECIALlimbptr, (int)cc.splitSpecialMeta.at(id).size() - 1, nullptr, nullptr, 0);
		// for (auto& l : SPECIALlimb)
		//     STREAM(l).wait(s);
	}
}

void LimbPartition::dotProductPt(LimbPartition& c1,
  const std::vector<const LimbPartition*>& c0s,
  const std::vector<const LimbPartition*>& c1s,
  const std::vector<const LimbPartition*>& pts,
  const bool ext) {

	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	uint32_t n = c0s.size();
	std::vector<void**> h_data(n * 3 * (1 + ext), nullptr);

	for (uint32_t i = 0; i < n; ++i) {
		assert(c0s[i]);
		assert(c1s[i]);
		assert(pts[i]);
		h_data[i]		  = c0s[i]->limbptr.data;
		h_data[i + n]	  = c1s[i]->limbptr.data;
		h_data[i + 2 * n] = pts[i]->limbptr.data;
		s.wait(c0s[i]->getS());
		s.wait(c1s[i]->getS());
		s.wait(pts[i]->getS());
		assert(c0s[i]->limb.size() >= limbsize);
		assert(c1s[i]->limb.size() >= limbsize);
		assert(pts[i]->limb.size() >= limbsize);
		if (ext) {
			int start	  = cc.splitSpecialMeta.at(id).at(0).id - cc.precom.constants[id].L;
			int num_limbs = cc.splitSpecialMeta.at(id).size();
			assert(c0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			assert(c1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			// assert(pts[i]->SPECIALlimb.size() >= 0);
			h_data[i + 3 * n] = c0s[i]->SPECIALlimbptr.data + start;
			h_data[i + 4 * n] = c1s[i]->SPECIALlimbptr.data + start;
			h_data[i + 5 * n] = pts[i]->SPECIALlimbptr.data;
		}
	}

	s.wait(c1.getS());

	VectorGPU<void**> data(s, n * 3 * (1 + ext), device, h_data.data());

	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		int size = std::min((int)limbsize - (int)i, cc.batch);
		dotProductPt_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);
	}

	if (ext) {
		uint32_t start	   = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
		uint32_t num_limbs = cc.splitSpecialMeta.at(id).size();
		for (uint32_t i = start; i < start + num_limbs; i += cc.batch) {
			STREAM(SPECIALlimb[i]).wait(s);
			uint32_t size = std::min(start + num_limbs - i, static_cast<uint32_t>(cc.batch));
			dotProductPt_<<<dim3{ (uint32_t)cc.N / 128, size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
			  SPECIALlimbptr.data + start, c1.SPECIALlimbptr.data + start, data.data + 3 * n, i - start, SPECIAL(id, i), n);
		}
		for (uint32_t i = start; i < start + num_limbs; i += cc.batch) {
			STREAM(SPECIALlimb[i]).wait(s);
		}
	}
	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}
	c1.getS().wait(s);
	for (size_t i = 0; i < c0s.size(); ++i) {
		c0s[i]->getS().wait(s);
		c1s[i]->getS().wait(s);
		pts[i]->getS().wait(s);
	}
	data.free(s);
}

void LimbPartition::binomialDotProduct(LimbPartition& c1,
  LimbPartition& c2,
  const std::vector<const LimbPartition*>& c0s,
  const std::vector<const LimbPartition*>& c1s,
  const std::vector<const LimbPartition*>& d0s,
  const std::vector<const LimbPartition*>& d1s,
  const bool ext) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);
	uint32_t n = c0s.size();
	std::vector<void**> h_data((n * 4 + 3) * (1 + ext), nullptr);

	for (uint32_t i = 0; i < n; ++i) {
		assert(c0s[i]);
		assert(c1s[i]);
		assert(d0s[i]);
		assert(d1s[i]);
		h_data[i]		  = c0s[i]->limbptr.data;
		h_data[i + n]	  = c1s[i]->limbptr.data;
		h_data[i + 2 * n] = d0s[i]->limbptr.data;
		h_data[i + 3 * n] = d1s[i]->limbptr.data;

		assert(c0s[i]->limb.size() >= limbsize);
		assert(c1s[i]->limb.size() >= limbsize);
		assert(d0s[i]->limb.size() >= limbsize);
		assert(d1s[i]->limb.size() >= limbsize);
		if (ext) {
			int start	  = cc.splitSpecialMeta.at(id).at(0).id - cc.precom.constants[id].L;
			int num_limbs = cc.splitSpecialMeta.at(id).size();
			assert(c0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			assert(c1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			assert(d0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			assert(d1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
			h_data[i + 4 * n] = c0s[i]->SPECIALlimbptr.data + start;
			h_data[i + 5 * n] = c1s[i]->SPECIALlimbptr.data + start;
			h_data[i + 6 * n] = d0s[i]->SPECIALlimbptr.data + start;
			h_data[i + 7 * n] = d1s[i]->SPECIALlimbptr.data + start;
		}
	}

	h_data[n * 4 * (1 + ext)]	  = limbptr.data;
	h_data[n * 4 * (1 + ext) + 1] = c1.limbptr.data;
	h_data[n * 4 * (1 + ext) + 2] = c2.limbptr.data;
	if (ext) {
		h_data[n * 4 * (1 + ext) + 3] = SPECIALlimbptr.data;
		h_data[n * 4 * (1 + ext) + 4] = c1.SPECIALlimbptr.data;
		h_data[n * 4 * (1 + ext) + 5] = c2.SPECIALlimbptr.data;
	}

	s.wait(c1.getS());
	s.wait(c2.getS());
	for (size_t i = 0; i < c0s.size(); ++i) {
		s.wait(c0s[i]->getS());
		s.wait(c1s[i]->getS());
		s.wait(d0s[i]->getS());
		s.wait(d1s[i]->getS());
	}

	VectorGPU<void**> data(s, (n * 4 + 3) * (1 + ext), device, h_data.data());

	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		int size = std::min((int)limbsize - (int)i, cc.batch);
		// dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
		//     limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);

		binomialDotProdBatched___<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(PARTITION(id, i),
		  data.data + 0,
		  data.data + n,
		  data.data + 2 * n,
		  data.data + 3 * n,
		  data.data + n * 4 * (1 + ext),
		  data.data + n * 4 * (1 + ext) + 1,
		  data.data + n * 4 * (1 + ext) + 2,
		  n,
		  1,
		  ext);
	}

	if (ext) {
		int start	  = cc.splitSpecialMeta.at(id).at(0).id - (cc.L + 1);
		int num_limbs = cc.splitSpecialMeta.at(id).size();
		for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
			STREAM(SPECIALlimb[i]).wait(s);
			int size = std::min((int)start + num_limbs - (int)i, cc.batch);
			// dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(
			//     SPECIALlimbptr.data + start, c1.SPECIALlimbptr.data + start, data.data + 3 * n, i - start,
			//    SPECIAL(id, i), n);

			binomialDotProdBatched___<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(SPECIALlimb[i]).ptr()>>>(SPECIAL(id, i),
			  data.data + 4 * n,
			  data.data + 5 * n,
			  data.data + 6 * n,
			  data.data + 7 * n,
			  data.data + n * 4 * (1 + ext) + 3,
			  data.data + n * 4 * (1 + ext) + 4,
			  data.data + n * 4 * (1 + ext) + 5,
			  n,
			  1,
			  false);
		}
		for (int32_t i = start; i < start + num_limbs; i += cc.batch) {
			STREAM(SPECIALlimb[i]).wait(s);
		}
	}
	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	c1.getS().wait(s);
	c2.getS().wait(s);
	for (size_t i = 0; i < c0s.size(); ++i) {
		c0s[i]->getS().wait(s);
		c1s[i]->getS().wait(s);
		d0s[i]->getS().wait(s);
		d1s[i]->getS().wait(s);
	}
	data.free(s);
}

void LimbPartition::binomialMult(LimbPartition& c1, LimbPartition& c2, const LimbPartition& d0, const LimbPartition& d1, bool extend_ins, bool square) {
	const int limbsize = getLimbSize(*level);
	cudaSetDevice(device);

	s.wait(c1.getS());
	s.wait(c2.getS());
	if (!square) {
		s.wait(d0.getS());
		s.wait(d1.getS());
	}

	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		STREAM(limb[i]).wait(s);
		int size = std::min((int)limbsize - (int)i, cc.batch);
		// dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr()>>>(
		//     limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);

		if (!square) {
			if (!extend_ins) {
				binomialMult_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(
				  PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i, d0.limbptr.data + i, d1.limbptr.data + i);
			} else {
				binomialMultExtend_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(
				  PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i, d0.limbptr.data + i, d1.limbptr.data + i);
			}
		} else {
			if (!extend_ins) {
				binomialSquare_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(
				  PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i);
			} else {
				binomialSquareExtend_<<<dim3{ (uint32_t)cc.N / 128, (uint32_t)size }, 128, 0, STREAM(limb[i]).ptr()>>>(
				  PARTITION(id, i), this->limbptr.data + i, c1.limbptr.data + i, c2.limbptr.data + i);
			}
		}
	}

	for (int32_t i = 0; i < limbsize; i += cc.batch) {
		s.wait(STREAM(limb[i]));
	}

	c1.getS().wait(s);
	c2.getS().wait(s);
	if (!square) {
		d0.getS().wait(s);
		d1.getS().wait(s);
	}
}

void LimbPartition::generateGatherLimb(bool iskey) {
	if (bufferGATHER == nullptr) {
		if (cc.GPUid.size() == 1 || iskey) {
			// bufferGATHER =
			//     (uint64_t*)GPUmalloc(device, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N), s.ptr());
			//  cudaMallocAsync(&bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N), s.ptr());

			if (iskey == false && DECOMPlimb.at(0).size() > 0) {
				std::vector<void*> h_gatherptr(GATHERptr.size, nullptr);

				int a = 0;

				for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
					for (size_t j = 0; j < DECOMPmeta[i].size(); ++j) {

						void* ptr = nullptr;
						SWITCH_RET(DECOMPlimb[i][j], v.data, ptr);
						h_gatherptr[a] = ptr;
						a++;
					}
				}

				if (GATHERptr.size * sizeof(void*) > 0) {
					UploadH2D(GATHERptr.data, h_gatherptr.data(), GATHERptr.size * sizeof(void*), device, s.ptr());
				}
			}
		} else {
#ifdef NCCL
			cudaStreamSynchronize(s.ptr());
			NCCLCHECK(ncclMemAlloc((void**)&bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N)));
			NCCLCHECK(ncclCommRegister(rank, bufferGATHER, std::max(1ul, GATHERmeta.size() * sizeof(uint64_t) * cc.N), &bufferGATHER_handle));
			if (bufferGATHER_handle == nullptr)
				bufferGATHER_handle = (void*)-1;
			cudaDeviceSynchronize();
#else
			assert(false);
#endif
			std::vector<void*> h_gatherptr(GATHERptr.size, nullptr);
			// std::cout << "BufferGATHER: " << bufferGATHER << std::endl;
			for (size_t i = 0; i < h_gatherptr.size(); ++i)
				h_gatherptr[i] = (void*)(bufferGATHER + cc.N * i);

			if (GATHERptr.size * sizeof(void*) > 0) {
				UploadH2D(GATHERptr.data, h_gatherptr.data(), GATHERptr.size * sizeof(void*), device, s.ptr());
			}
		}
	}
}

// namespace FIDESlib::CKKS

} // namespace FIDESlib::CKKS
