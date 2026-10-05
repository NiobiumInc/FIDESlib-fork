//
// Created by carlosad on 25/03/24.
//

#include "AddSub.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/Limb.cuh"
#include "CudaUtils.cuh" // PinnedStagingUpload
#include "ModMult.cuh"
#include "NTT.cuh"
#include "Rotation.cuh"

namespace FIDESlib::CKKS {
template <typename T>
Limb<T>::Limb(Limb<T>&& l) noexcept : cc(l.cc), primeid(l.primeid), stream(l.stream), v(std::move(l.v)), aux(std::move(l.aux)), id(l.id), raw(l.raw) {
}

template <typename T> Limb<T>::~Limb() noexcept {
	v.free(stream);
	// May already have been released by freeAux(); VectorGPU::free is idempotent.
	aux.free(stream);
}

template <typename T> void* Limb<T>::freeAux(Stream& s) {
	const bool owned = aux.owns();
	aux.free(s);
	// Released -> the slot must read nullptr (which is also what a never-allocated, constant-grown
	// limb's slot already holds). A view -> nothing was freed, so the address stays valid and stays.
	return owned ? nullptr : (void*)aux.data;
}

template <typename T>
Limb<T>::Limb(ContextData& context, const int id, Stream& stream, const int primeid, bool constant)
: cc(context), primeid(primeid), stream(stream /*StartStream(primeid, cc.L + cc.K + 1)*/), v(stream, context.N, cc.GPUid[id]),
  aux(stream, constant ? 0 : context.N, cc.GPUid[id]), id(id), raw(!cc.isValidPrimeId(primeid)) {
	// TODO: CryptoContext limb tracking.
	// id = cc.generateId(this);
	if (raw) {
		assert(primeid == -1);
	}
	// assert(stream.ptr() != nullptr);
	// assert(stream.ev != nullptr);
	assert(this->v.size == cc.N);
	if (DeviceChecks()) { // FIDESLIB_DEVICE_CHECKS=0 skips the driver query the assert consumes
		int dev = -1;
		cudaGetDevice(&dev);
		assert(dev == v.device);
		(void)dev;
	}
}

/**
 * Caution! Auxiliar vector is not managed and operations like INTT and NTT shouldn't be used on this.
 * TODO add assertions for correct use.
 */
template <typename T>
Limb<T>::Limb(ContextData& context, T* data, const int offset, const int id, Stream& stream, const int primeid, T* data_aux, const int offset_aux)
: cc(context), primeid(primeid), stream(stream /*StartStream(primeid, cc.L + cc.K + 1)*/), v(data, context.N, cc.GPUid[id], offset),
  aux(data_aux ? data_aux : data, context.N, cc.GPUid[id], data_aux ? offset_aux : offset), id(id), raw(!cc.isValidPrimeId(primeid)) {
	// TODO: CryptoContext limb tracking.
	// id = cc.generateId(this);
	if (raw) {
		assert(primeid == -1);
	}
	// assert(stream.ptr() != nullptr);
	// assert(stream.ev != nullptr);

	assert(this->v.size == cc.N);
	if (DeviceChecks()) { // FIDESLIB_DEVICE_CHECKS=0 skips the driver query the assert consumes
		int dev = -1;
		cudaGetDevice(&dev);
		assert(dev == v.device);
		(void)dev;
	}
}

template <typename T> Global::Globals* Limb<T>::getGlobals() {
	return cc.precom.globals->globals[id];
}

template <typename T> void Limb<T>::store(std::vector<T>& dat) const {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kStore);
	dat.resize(v.size);

	// cudaHostRegister((void*)dat.data(), dat.size() * sizeof(T), cudaHostRegisterDefault);
	// cudaMemcpyAsync((void *) dat.data(), v.data, dat.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.ptr());

	cudaMemcpyAsync((void*)dat.data(), v.data, dat.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.ptr());
	cudaStreamSynchronize(stream.ptr());
	// cudaDeviceSynchronize();
	// cudaHostUnregister((void*)dat.data());
}

/**
 * Lifetime contract for the H2D transfers below.
 *
 * Default (pageable) path: the source is ordinary pageable host memory, so cudaMemcpyAsync is not
 * truly asynchronous on the host side -- the driver stages the bytes through its own pinned buffer
 * and does not return until that staging copy has been made. The source is therefore free to die as
 * soon as the call returns, which is what lets the converting branch hand the kernel a
 * function-local vector.
 *
 * The same-width branch does not even need that guarantee: it transfers straight out of the
 * caller's vector, which by definition outlives the call, instead of copying it into a local
 * first. That is strictly safer than what it replaces, not less safe.
 *
 * Pinned path (FIDESLIB_PINNED_STAGING=1): PinnedStagingUpload copies the source into a slot of the
 * device's pinned staging ring before it issues the transfer, so the bytes leaving for the device
 * are the ring's, not the caller's. The source is therefore free to die as soon as the call
 * returns -- the same contract as the pageable path, established explicitly here instead of
 * inherited from driver behaviour, and it is the RING SLOT, not a stream synchronization, that
 * establishes it. (The earlier single-slot version did synchronize; it does not any more, and
 * nothing about this contract depended on that.) The upload itself stays asynchronous and ordered
 * within `st`, exactly as the pageable cudaMemcpyAsync below is. PinnedStagingUpload declines
 * (returns false) whenever it cannot do all that, and we fall back to the pageable copy.
 *
 * If the pageable path is ever moved onto pinned/registered memory WITHOUT going through
 * PinnedStagingUpload (i.e. transferring straight out of caller-owned pinned memory), the
 * converting branch must gain an owned staging buffer or a cudaStreamSynchronize before returning.
 *
 * THREADING: the ring is thread-safe per device, so nothing here forbids concurrent loads. In this
 * library they do not happen on one device anyway -- RNSPoly::load / loadConstant are serial loops
 * and RNSPoly::loadDecompDigit's omp loop is one thread PER GPU, so each thread reaches a different
 * device's ring. (Applications may encode plaintexts on worker threads, but the device load is a
 * separate step on the consumer thread.)
 */
template <typename T> template <typename Q> void Limb<T>::load(const std::vector<Q>& dat_) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kLoad);
	assert(dat_.size() <= v.size);
	cudaStream_t st = stream.ptr();

	if constexpr (std::is_same_v<T, Q>) {
		// No conversion needed: no intermediate at all.
		const size_t bytes = dat_.size() * sizeof(T);
		if (!PinnedStagingUpload(v.data, dat_.data(), bytes, v.device, st))
			cudaMemcpyAsync(v.data, dat_.data(), bytes, cudaMemcpyHostToDevice, st);
	} else {
		// Widths differ, so a converted staging vector is unavoidable. It is sized to the full limb
		// and zero-filled, so a short input leaves the tail zeroed; the previous version looped to
		// v.size while indexing dat_, reading past its end whenever dat_ was short.
		std::vector<T> dat(v.size, 0);
		for (size_t i = 0; i < dat_.size(); ++i) {
			dat[i] = static_cast<T>(dat_[i]);
		}
		const size_t bytes = dat.size() * sizeof(T);
		if (!PinnedStagingUpload(v.data, dat.data(), bytes, v.device, st))
			cudaMemcpyAsync(v.data, dat.data(), bytes, cudaMemcpyHostToDevice, st);
	}
}

template void Limb<uint32_t>::load<uint32_t>(const std::vector<uint32_t>& dat_);

template void Limb<uint32_t>::load<uint64_t>(const std::vector<uint64_t>& dat_);

template void Limb<uint64_t>::load<uint32_t>(const std::vector<uint32_t>& dat_);

template void Limb<uint64_t>::load<uint64_t>(const std::vector<uint64_t>& dat_);

template <typename T> void Limb<T>::load(const VectorGPU<T>& dat) {
	cudaMemcpyAsync(v.data, dat.data, v.size, cudaMemcpyDeviceToDevice, stream.ptr());
}

template <typename T> template <typename Q> void Limb<T>::load_convert(const std::vector<Q>& dat_raw) {
	assert(dat_raw.size() <= v.size);

	if constexpr (std::is_same_v<T, Q>) {
		// static_cast<T> is the identity here, so materialising a converted copy of the caller's
		// buffer only to memcpy it is pure overhead. Hand the caller's buffer straight to load(),
		// whose same-width branch transfers out of it directly.
		load(dat_raw);
	} else {
		std::vector<T> dat(dat_raw.size());
		for (size_t i = 0; i < dat.size(); ++i)
			dat[i] = static_cast<T>(dat_raw[i]);
		// `dat` outlives the transfer: see the lifetime contract on Limb<T>::load above.
		load(dat);
	}
}

template void Limb<uint32_t>::load_convert<uint64_t>(const std::vector<uint64_t>& dat_raw);

template void Limb<uint64_t>::load_convert<uint64_t>(const std::vector<uint64_t>& dat_raw);

template <typename T> template <typename Q> void Limb<T>::store_convert(std::vector<Q>& dat_raw) {
	dat_raw.resize(v.size);
	if constexpr (std::is_same_v<T, Q>) {
		store(dat_raw);
	} else {
		std::vector<T> dat(v.size);
		store(dat);
		for (size_t i = 0; i < dat.size(); ++i)
			dat_raw[i] = static_cast<Q>(dat[i]);
	}
}

template void Limb<uint32_t>::store_convert<uint64_t>(std::vector<uint64_t>& dat_raw);

template void Limb<uint64_t>::store_convert<uint64_t>(std::vector<uint64_t>& dat_raw);

/**
 * @brief Device encode, one limb: OpenFHE's CKKSPackedEncoding::FitToNativeVector, on the GPU.
 *
 * Reproduces, coefficient by coefficient, the host loop in OpenFHE's
 * src/pke/lib/encoding/ckkspackedencoding.cpp (FitToNativeVector), which is the ONLY step between
 * the encoder's `temp` array and the RNS towers:
 *
 *     NativeInteger n(vec[i]);
 *     if (n > bigValueHf) (*nativeVec)[gap * i] = n.ModSub(diff, modulus);
 *     else                (*nativeVec)[gap * i] = n.Mod(modulus);
 *
 * with `diff = bigBound - modulus` and `bigValueHf = bigBound >> 1`. `ModSub(b, q)` in OpenFHE
 * reduces BOTH operands first and then subtracts with a borrow-correcting add, which is what the
 * `(r >= d) ? r - d : r + q - d` below is; writing it as `(n - diff) % q` instead would be wrong,
 * because `n - diff` underflows for the small-`n` half of the negative branch.
 *
 * Deliberately NOT a `gap`-strided scatter: this entry point takes the caller's already-expanded
 * length-N coefficient vector (gap == 1 after expansion), so the zero-fill for sparse encodings is
 * the caller's job and is visible in the uploaded buffer. Keeping the stride out of the kernel is
 * what lets every limb share one buffer with a plain contiguous read.
 *
 * ALGO_BARRETT (not ALGO_BARRETT_FP64, which is an unimplemented device-side assert) reduces via
 * Neal_reduce_64, whose error bound holds for any input below prime^2 — biased coefficients are
 * bounded by 2^63 - 2^9 - 1 and the primes here are ~2^55 and up, so the input is far inside the
 * valid domain and the single conditional subtraction at the end of the reduce is sufficient.
 */
template <typename T>
__global__ void fit_to_native_(T* __restrict__ dst, const uint64_t* __restrict__ biased, const int primeid, const uint64_t bigValueHf, const uint64_t diff) {
	const int i		 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t n = biased[i];
	const uint64_t q = C_.primes[primeid];

	uint64_t r = modreduce<ALGO_BARRETT>((__uint128_t)n, primeid);
	if (n > bigValueHf) {
		const uint64_t d = modreduce<ALGO_BARRETT>((__uint128_t)diff, primeid);
		r				 = (r >= d) ? (r - d) : (r + q - d);
	}
	dst[i] = static_cast<T>(r);
}

template <> void Limb<uint64_t>::loadCoefficients(const uint64_t* biased, const uint64_t bigValueHf, const uint64_t diff) {
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	fit_to_native_<uint64_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, biased, primeid, bigValueHf, diff);
}

template <> void Limb<uint32_t>::loadCoefficients(const uint64_t* biased, const uint64_t bigValueHf, const uint64_t diff) {
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	// The residue is < prime < 2^32 for a 32-bit tower, so the narrowing store in the kernel is exact.
	fit_to_native_<uint32_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, biased, primeid, bigValueHf, diff);
}

template <typename T> void Limb<T>::add(const LimbImpl& l) {
	switch (l.index()) {
	case U32: add(std::get<U32>(l)); break;
	case U64: add(std::get<U64>(l)); break;
	}
}

template <> void Limb<uint64_t>::add(const Limb<uint64_t>& l) {
	// stream.wait(l.stream);
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	add_<uint64_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <> void Limb<uint64_t>::add(const Limb<uint32_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint32_t>::add(const Limb<uint64_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint32_t>::add(const Limb<uint32_t>& l) {
	// stream.wait(l.stream);
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	add_<uint32_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <typename T> void Limb<T>::sub(const LimbImpl& l) {
	switch (l.index()) {
	case U32: sub(std::get<U32>(l)); break;
	case U64: sub(std::get<U64>(l)); break;
	}
}

template <> void Limb<uint32_t>::sub(const Limb<uint32_t>& l) {
	stream.wait(l.stream);
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	sub_<uint32_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <> void Limb<uint32_t>::sub(const Limb<uint64_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint64_t>::sub(const Limb<uint32_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint64_t>::sub(const Limb<uint64_t>& l) {
	stream.wait(l.stream);
	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	sub_<uint64_t><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <typename T> void Limb<T>::mult(const LimbImpl& l) {
	switch (l.index()) {
	case U32: mult(std::get<U32>(l)); break;
	case U64: mult(std::get<U64>(l)); break;
	}
}

template <> void Limb<uint64_t>::mult(const Limb<uint32_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint32_t>::mult(const Limb<uint32_t>& l) {
	assert(v.size == l.v.size);
	mult_<uint32_t, ALGO_BARRETT><<<v.size / block, block, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <> void Limb<uint32_t>::mult(const Limb<uint64_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint64_t>::mult(const Limb<uint64_t>& l) {
	assert(v.size == l.v.size);
	mult_<uint64_t, ALGO_BARRETT><<<v.size / block, block, 0, stream.ptr()>>>(v.data, l.v.data, primeid);
}

template <> void Limb<uint64_t>::mult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace) {
	assert(_l1.index() == U64);
	assert(_l2.index() == U64);
	using T			  = uint64_t;
	const Limb<T>& l1 = std::get<U64>(_l1);
	const Limb<T>& l2 = std::get<U64>(_l2);

	mult_<T, ALGO_BARRETT><<<v.size / block, block, 0, stream.ptr()>>>(inplace ? aux.data : v.data, l1.v.data, l2.v.data, primeid);
}

template <> void Limb<uint32_t>::mult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace) {
	assert(_l1.index() == U32);
	assert(_l2.index() == U32);
	using T			  = uint32_t;
	const Limb<T>& l1 = std::get<U32>(_l1);
	const Limb<T>& l2 = std::get<U32>(_l2);

	mult_<T, ALGO_BARRETT><<<v.size / block, block, 0, stream.ptr()>>>(inplace ? aux.data : v.data, l1.v.data, l2.v.data, primeid);
}

template <typename T> template <ALGO algo> void Limb<T>::INTT() {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kNtt);
	assert(primeid >= 0);
	constexpr int M = sizeof(T) == 8 ? 4 : 8;

	if constexpr (1) {
		dim3 blockDim = 1 << ((cc.logN) / 2 - 1);
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, false, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), v.data, primeid, aux.data);

		blockDim = (1 << ((cc.logN + 1) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, true, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data);
	}
}

template <typename T> template <ALGO algo> void Limb<T>::NTT() {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kNtt);

	if constexpr (0) {
		assert(primeid >= 0);
		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		constexpr int K = 1;

		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)), K };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		// T *psi_arr = (T *) G_::psi[primeid];
		// T *psi_arr_middle_scale = (T *) G_::psi_middle_scale[primeid];

		int primeid_  = primeid;
		void* args[6] = { getGlobals(), &v.data, (void*)&primeid_, &aux.data, (void*)&primeid_, (void*)&primeid_ };
		cudaLaunchCooperativeKernel(get_NTT_reference(false) /*(void *) test_kernel*/ /*(void *) NTT_<T, false, algo>*/, gridDim, blockDim, args, bytes, stream.ptr());
		// CudaCheckErrorModNoSync;
	} else if constexpr (1) {
		// thread_local, not a bare function-local static (same reasoning as LimbPartition::multPt):
		// this is reached by every NTT, and `exec[primeid]` is a std::map insertion the first time a
		// prime is seen. With FIDESLIB_CONCURRENT_OPS several host threads reach it at once; the
		// default mode has one thread, one map, so its behaviour is unchanged.
		static thread_local std::map<int, cudaGraphExec_t> exec;

		run_in_graph<false>(exec[primeid], stream, [&]() {
			assert(primeid >= 0);
			constexpr int M = sizeof(T) == 8 ? 4 : 8;

			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1 + (cc.logN > 13 ? 0 : 0)) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, false, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), v.data, primeid, aux.data);

			{
				blockDim = dim3{ (uint32_t)(1 << ((cc.logN + (cc.logN > 13 ? 0 : 0)) / 2 - 1)) };
				gridDim	 = { v.size / blockDim.x / 2 / M };
				bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

				NTT_<T, true, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data);
			}
		});

		// aux.free(stream);
	}
}

template <> void Limb<uint64_t>::NTT_rescale_fused(const Limb<uint64_t>& l) {

	constexpr ALGO algo = ALGO_SHOUP;
	using T				= uint64_t;
	constexpr int M		= sizeof(T) == 8 ? 4 : 8;

	assert(primeid >= 0);
	assert(l.primeid >= 0);

	dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
	dim3 gridDim{ v.size / blockDim.x / 2 / M };
	int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

	NTT_<T, false, algo, NTT_RESCALE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, aux.data, nullptr, l.primeid);
	blockDim = (1 << ((cc.logN) / 2 - 1));
	gridDim	 = { v.size / blockDim.x / 2 / M };
	bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

	NTT_<T, true, algo, NTT_RESCALE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data, nullptr, l.primeid);
}

template <> void Limb<uint32_t>::NTT_rescale_fused(const Limb<uint32_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint64_t>::NTT_rescale_fused(const Limb<uint32_t>& l) {
	assert("Not implemented." == nullptr);
}

template <> void Limb<uint32_t>::NTT_rescale_fused(const Limb<uint64_t>& l) {
	assert("Not implemented." == nullptr);
}

template <typename T> void Limb<T>::NTT_rescale_fused(const LimbImpl& l) {
	switch (l.index()) {
	case U32: NTT_rescale_fused(std::get<U32>(l)); break;
	case U64: NTT_rescale_fused(std::get<U64>(l)); break;
	}
}

template <> void Limb<uint64_t>::NTT_moddown_fused(const LimbImpl& _l) {
	assert(_l.index() == U64);
	using T = uint64_t;
	{
		const Limb<uint64_t>& l = std::get<U64>(_l);

		constexpr ALGO algo = ALGO_SHOUP;
		using T				= uint64_t;
		constexpr int M		= sizeof(T) == 8 ? 4 : 8;

		assert(primeid >= 0);
		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, false, algo, NTT_MODDOWN><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, aux.data);
		blockDim = (1 << ((cc.logN) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, true, algo, NTT_MODDOWN><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data);
	}
}

template <> void Limb<uint32_t>::NTT_moddown_fused(const LimbImpl& _l) {
	assert(_l.index() == U32);
	using T = uint32_t;
	{
		const Limb<T>& l = std::get<U32>(_l);

		constexpr ALGO algo = ALGO_SHOUP;
		constexpr int M		= sizeof(T) == 8 ? 4 : 8;

		assert(primeid >= 0);
		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, false, algo, NTT_MODDOWN><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, aux.data);
		blockDim = (1 << ((cc.logN) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, true, algo, NTT_MODDOWN><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data);
	}
}

template <> void Limb<uint64_t>::NTT_multpt_fused(const LimbImpl& _l, const LimbImpl& _pt) {
	assert(_l.index() == U64);
	assert(_pt.index() == U64);

	// thread_local, not a bare function-local static (same reasoning as LimbPartition::multPt):
		// this is reached by every NTT, and `exec[primeid]` is a std::map insertion the first time a
		// prime is seen. With FIDESLIB_CONCURRENT_OPS several host threads reach it at once; the
		// default mode has one thread, one map, so its behaviour is unchanged.
		static thread_local std::map<int, cudaGraphExec_t> exec;

	// run_in_graph<false>(exec[primeid], stream, [&]()
	{
		const Limb<uint64_t>& l	 = std::get<U64>(_l);
		const Limb<uint64_t>& pt = std::get<U64>(_pt);

		constexpr ALGO algo = ALGO_SHOUP;
		using T				= uint64_t;
		constexpr int M		= sizeof(T) == 8 ? 4 : 8;

		assert(primeid >= 0);
		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, false, algo, NTT_MULTPT><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, aux.data, nullptr, l.primeid);
		blockDim = (1 << ((cc.logN) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		NTT_<T, true, algo, NTT_MULTPT><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data, pt.v.data, l.primeid);
	}
	//);
}

template <> void Limb<uint32_t>::NTT_multpt_fused(const LimbImpl& _l, const LimbImpl& _pt) {
	assert(_l.index() == U32);
	assert(_pt.index() == U32);
	const Limb<uint32_t>& l	 = std::get<U32>(_l);
	const Limb<uint32_t>& pt = std::get<U32>(_pt);

	constexpr ALGO algo = ALGO_SHOUP;
	using T				= uint32_t;
	constexpr int M		= sizeof(T) == 8 ? 4 : 8;

	assert(primeid >= 0);
	dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
	dim3 gridDim{ v.size / blockDim.x / 2 / M };
	int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

	NTT_<T, false, algo, NTT_MULTPT><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, aux.data, nullptr, l.primeid);
	blockDim = (1 << ((cc.logN) / 2 - 1));
	gridDim	 = { v.size / blockDim.x / 2 / M };
	bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

	NTT_<T, true, algo, NTT_MULTPT><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), aux.data, primeid, v.data, pt.v.data, l.primeid);
}

template <typename T> Limb<T> Limb<T>::clone() {
	Limb<T> res(cc, v.device, stream, primeid);

	stream.wait(res.stream);

	cudaMemcpyAsync(res.v.data, v.data, v.size * sizeof(T), cudaMemcpyDeviceToDevice, stream.ptr());

	res.stream.wait(stream);

	return res;
}

template <typename T> void Limb<T>::copyV(const LimbImpl& l) {
	switch (l.index()) {
	case U32: copyV(std::get<U32>(l)); break;
	case U64: copyV(std::get<U64>(l)); break;
	}
}

template <> void Limb<uint32_t>::copyV(const Limb<uint32_t>& l) {
	// stream.wait(l.stream);
	cudaMemcpyAsync(v.data, l.v.data, sizeof(uint32_t) * v.size, cudaMemcpyDefault, stream.ptr());
}

template <> void Limb<uint64_t>::copyV(const Limb<uint64_t>& l) {
	// stream.wait(l.stream);
	cudaMemcpyAsync(v.data, l.v.data, sizeof(uint64_t) * v.size, cudaMemcpyDefault, stream.ptr());
}

template <> void Limb<uint64_t>::copyV(const Limb<uint32_t>& l) {
	assert("Not implemented" == 0);
}

template <> void Limb<uint32_t>::copyV(const Limb<uint64_t>& l) {
	assert("Not implemented" == 0);
}

template <> void Limb<uint64_t>::INTT_from(LimbImpl& _l) {
	assert(primeid == PRIMEID(_l));
	assert(_l.index() == U64);
	constexpr ALGO algo = ALGO_SHOUP;
	using T				= uint64_t;
	{
		const Limb<uint64_t>& l = std::get<U64>(_l);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;

		assert(primeid >= 0);
		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, false, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, l.aux.data);
		blockDim = (1 << ((cc.logN) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, true, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.aux.data, primeid, v.data);
	}
}

template <> void Limb<uint32_t>::INTT_from(LimbImpl& _l) {
	assert(primeid == PRIMEID(_l));
	assert(_l.index() == U32);
	using T				= uint32_t;
	constexpr ALGO algo = ALGO_SHOUP;
	{
		const Limb<T>& l = std::get<U32>(_l);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
		dim3 gridDim{ v.size / blockDim.x / 2 / M };
		int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, false, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.v.data, primeid, l.aux.data);
		blockDim = (1 << ((cc.logN) / 2 - 1));
		gridDim	 = { v.size / blockDim.x / 2 / M };
		bytes	 = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

		INTT_<T, true, algo><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), l.aux.data, primeid, v.data);
	}
}

template <> void Limb<uint64_t>::addMult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace) {
	using T			  = uint64_t;
	const Limb<T>& l1 = std::get<U64>(_l1);
	const Limb<T>& l2 = std::get<U64>(_l2);

	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	addMult_<T><<<gridDim, blockDim, 0, stream.ptr()>>>(inplace ? aux.data : v.data, l1.v.data, l2.v.data, primeid);
}

template <> void Limb<uint32_t>::addMult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace) {
	using T			  = uint32_t;
	const Limb<T>& l1 = std::get<U32>(_l1);
	const Limb<T>& l2 = std::get<U32>(_l2);

	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };
	addMult_<T><<<gridDim, blockDim, 0, stream.ptr()>>>(inplace ? aux.data : v.data, l1.v.data, l2.v.data, primeid);
}

template <typename T> void Limb<T>::printThisLimb(int num) const {
	std::vector<T> cpu(v.size);
	store(cpu);
	std::cout << "(" << cc.precom.constants[id].primes[primeid] << ", ";
	for (int i = 0; i < num; ++i) {
		std::cout << cpu[i] << ((i == num - 1) ? ") " : " ");
	}
}

template <typename T>
void Limb<T>::INTT_from_mult(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& c1_, const LimbImpl& c1tilde_, const LimbImpl& c0_, const LimbImpl& c0tilde_, const LimbImpl& kska_, const LimbImpl& kskb_) {
	constexpr ALGO algo = ALGO_SHOUP;

	if constexpr (sizeof(T) == 8) {
		Limb<T>& res0		   = std::get<U64>(res0_);
		Limb<T>& res1		   = std::get<U64>(res1_);
		const Limb<T>& c1	   = std::get<U64>(c1_);
		const Limb<T>& c1tilde = std::get<U64>(c1tilde_);
		const Limb<T>& kska	   = std::get<U64>(kska_);
		const Limb<T>& kskb	   = std::get<U64>(kskb_);
		const Limb<T>& c0	   = std::get<U64>(c0_);
		const Limb<T>& c0tilde = std::get<U64>(c0tilde_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, false, algo, INTT_MODE::INTT_MULT_AND_SAVE><<<gridDim, blockDim, bytes, stream.ptr()>>>(
			  getGlobals(), c1.v.data, primeid, c1.aux.data, c1tilde.v.data, res0.v.data, res1.v.data, kska.v.data, kskb.v.data, c0.v.data, c0tilde.v.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, true, algo, INTT_MODE::INTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), c1.aux.data, primeid, v.data);
		}
	} else {
		Limb<T>& res0		   = std::get<U32>(res0_);
		Limb<T>& res1		   = std::get<U32>(res1_);
		const Limb<T>& c1	   = std::get<U32>(c1_);
		const Limb<T>& c1tilde = std::get<U32>(c1tilde_);
		const Limb<T>& kska	   = std::get<U32>(kska_);
		const Limb<T>& kskb	   = std::get<U32>(kskb_);
		const Limb<T>& c0	   = std::get<U32>(c0_);
		const Limb<T>& c0tilde = std::get<U32>(c0tilde_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, false, algo, INTT_MODE::INTT_MULT_AND_SAVE><<<gridDim, blockDim, bytes, stream.ptr()>>>(
			  getGlobals(), c1.v.data, primeid, c1.aux.data, c1tilde.v.data, res0.v.data, res1.v.data, kska.v.data, kskb.v.data, c0.v.data, c0tilde.v.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, true, algo, INTT_MODE::INTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), c1.aux.data, primeid, v.data);
		}
	}
}

template <typename T>
void Limb<T>::INTT_from_mult_acc(LimbImpl& res0_,
  LimbImpl& res1_,
  const LimbImpl& c1_,
  const LimbImpl& c1tilde_,
  const LimbImpl& c0_,
  const LimbImpl& c0tilde_,
  const LimbImpl& kska_,
  const LimbImpl& kskb_) {
	constexpr ALGO algo = ALGO_SHOUP;

	if constexpr (sizeof(T) == 8) {
		Limb<T>& res0		   = std::get<U64>(res0_);
		Limb<T>& res1		   = std::get<U64>(res1_);
		const Limb<T>& c1	   = std::get<U64>(c1_);
		const Limb<T>& c1tilde = std::get<U64>(c1tilde_);
		const Limb<T>& kska	   = std::get<U64>(kska_);
		const Limb<T>& kskb	   = std::get<U64>(kskb_);
		const Limb<T>& c0	   = std::get<U64>(c0_);
		const Limb<T>& c0tilde = std::get<U64>(c0tilde_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, false, algo, INTT_MODE::INTT_MULT_AND_ACC><<<gridDim, blockDim, bytes, stream.ptr()>>>(
			  getGlobals(), c1.v.data, primeid, c1.aux.data, nullptr, res0.v.data, res1.v.data, kska.v.data, kskb.v.data, c0.v.data, nullptr);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, true, algo, INTT_MODE::INTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), c1.aux.data, primeid, v.data);
		}
	} else {
		Limb<T>& res0		   = std::get<U32>(res0_);
		Limb<T>& res1		   = std::get<U32>(res1_);
		const Limb<T>& c1	   = std::get<U32>(c1_);
		const Limb<T>& c1tilde = std::get<U32>(c1tilde_);
		const Limb<T>& kska	   = std::get<U32>(kska_);
		const Limb<T>& kskb	   = std::get<U32>(kskb_);
		const Limb<T>& c0	   = std::get<U32>(c0_);
		const Limb<T>& c0tilde = std::get<U32>(c0tilde_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, false, algo, INTT_MODE::INTT_MULT_AND_ACC><<<gridDim, blockDim, bytes, stream.ptr()>>>(
			  getGlobals(), c1.v.data, primeid, c1.aux.data, c1tilde.v.data, res0.v.data, res1.v.data, kska.v.data, kskb.v.data, c0.v.data, c0tilde.v.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			INTT_<T, true, algo, INTT_MODE::INTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), c1.aux.data, primeid, v.data);
		}
	}
}

template <typename T> void Limb<T>::NTT_and_ksk_dot(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_) {
	constexpr ALGO algo = ALGO_SHOUP;

	if constexpr (sizeof(T) == 8) {
		Limb<T>& res0		= std::get<U64>(res0_);
		Limb<T>& res1		= std::get<U64>(res1_);
		const Limb<T>& kska = std::get<U64>(kska_);
		const Limb<T>& kskb = std::get<U64>(kskb_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, false, algo, NTT_MODE::NTT_NONE><<<gridDim, blockDim, bytes, res0.stream.ptr()>>>(getGlobals(), v.data, primeid, res0.aux.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, true, algo, NTT_MODE::NTT_KSK_DOT>
			  <<<gridDim, blockDim, bytes, res0.stream.ptr()>>>(getGlobals(), res0.aux.data, primeid, res0.v.data, kska.v.data, 0, res1.v.data, kskb.v.data);
		}
	} else {
		Limb<T>& res0		= std::get<U32>(res0_);
		Limb<T>& res1		= std::get<U32>(res1_);
		const Limb<T>& kska = std::get<U32>(kska_);
		const Limb<T>& kskb = std::get<U32>(kskb_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, false, algo, NTT_MODE::NTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), v.data, primeid, res0.aux.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, true, algo, NTT_MODE::NTT_KSK_DOT>
			  <<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), res0.aux.data, primeid, res0.v.data, kska.v.data, 0, res1.v.data, kskb.v.data);
		}
	}
}

template <typename T> void Limb<T>::NTT_and_ksk_dot_acc(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_) {
	constexpr ALGO algo = ALGO_SHOUP;

	if constexpr (sizeof(T) == 8) {
		Limb<T>& res0		= std::get<U64>(res0_);
		Limb<T>& res1		= std::get<U64>(res1_);
		const Limb<T>& kska = std::get<U64>(kska_);
		const Limb<T>& kskb = std::get<U64>(kskb_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, false, algo, NTT_MODE::NTT_NONE><<<gridDim, blockDim, bytes, res0.stream.ptr()>>>(getGlobals(), v.data, primeid, res0.aux.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, true, algo, NTT_MODE::NTT_KSK_DOT_ACC>
			  <<<gridDim, blockDim, bytes, res0.stream.ptr()>>>(getGlobals(), res0.aux.data, primeid, res0.v.data, kska.v.data, 0, res1.v.data, kskb.v.data);
		}
	} else {
		Limb<T>& res0		= std::get<U32>(res0_);
		Limb<T>& res1		= std::get<U32>(res1_);
		const Limb<T>& kska = std::get<U32>(kska_);
		const Limb<T>& kskb = std::get<U32>(kskb_);

		constexpr int M = sizeof(T) == 8 ? 4 : 8;
		assert(primeid >= 0);
		{
			dim3 blockDim{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
			dim3 gridDim{ v.size / blockDim.x / 2 / M };
			int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, false, algo, NTT_MODE::NTT_NONE><<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), v.data, primeid, res0.aux.data);
		}
		{
			dim3 blockDim = (1 << ((cc.logN) / 2 - 1));
			dim3 gridDim  = { v.size / blockDim.x / 2 / M };
			int bytes	  = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

			NTT_<T, true, algo, NTT_MODE::NTT_KSK_DOT_ACC>
			  <<<gridDim, blockDim, bytes, stream.ptr()>>>(getGlobals(), res0.aux.data, primeid, res0.v.data, kska.v.data, 0, res1.v.data, kskb.v.data);
		}
	}
}

template <typename T> void Limb<T>::automorph(const int index, const int br) {

	dim3 blockDim{ block };
	dim3 gridDim{ (uint32_t)(cc.N) / block };

	automorph_<T><<<gridDim, blockDim, 0, stream.ptr()>>>(v.data, aux.data, index, br);

	std::swap(v.data, aux.data);
}

#define Y(type, algo) template void Limb<type>::INTT<algo>();
#include "ntt_types.inc"
#undef Y

#define Y(type, algo) template void Limb<type>::NTT<algo>();
#include "ntt_types.inc"
#undef Y

#define X(type) template class Limb<type>;
#include "ntt_types.inc"

#undef X

} // namespace FIDESlib::CKKS