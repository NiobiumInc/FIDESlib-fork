//
// Batched device encode: build MANY plaintexts' RNS towers from MANY coefficient vectors in one
// pass of kernels, instead of one pass per plaintext.
//
// WHY THIS EXISTS. RNSPoly::loadCoefficients already moves the expensive half of a CKKS encode onto
// the device: the host keeps the inverse special FFT and the rounding (cheap, precision-critical),
// and the device does the L modular reductions and the L forward NTTs (expensive, exactly
// reproducible), with ONE N-word upload instead of L. What it does NOT change is how often the host
// talks to the driver, and that is what an application encoding hundreds of thousands of diagonals
// on many worker threads actually spends its time on. Per plaintext the single-call path issues:
//
//   * one coefficient upload out of pageable host memory -- which the driver stages through its own
//     internal pinned buffer and does NOT return from until that staging copy is done (the same
//     property Limb<T>::load documents at Limb.cu:89-95), so it BLOCKS the calling thread;
//   * one reduce launch per limb (Limb<T>::loadCoefficients, Limb.cu:233);
//   * one forward-NTT launch pair per partition (RNSPoly::NTT -> LimbPartition::ApplyNTT,
//     LimbPartition.cu:339-359);
//   * the stream fences around both.
//
// A BSGS giant step needs 32 diagonals at one level, i.e. 32 of those, back to back, on each of
// many threads. This path takes the whole group at once: ONE gathered upload for all 32 coefficient
// vectors, ONE reduce launch covering (batch x limbs x N), and ONE NTT launch pair covering the
// same, per GPU partition.
//
// WHAT IT DOES NOT CHANGE. The host half is untouched -- the caller still hands in exactly the
// coefficient vectors CryptoContextImpl::MakeCKKSPackedPlaintextDevice computes, one per plaintext,
// built by the same transcription of OpenFHE's CKKSPackedEncoding::Encode. The arithmetic is
// untouched: fit_to_native_batch_ below is fit_to_native_ (Limb.cu:220-243) with two extra grid
// dimensions, and NTTBatch_ (NTT.cu) calls the very same NTT__ device function the ordinary NTT
// kernel calls, with the same arguments. So every output plaintext is bit-identical to what the
// single-plaintext path would have produced from the same input -- that is the contract, and
// DeviceEncodeTests pins it.
//
// OWNERSHIP IS DELIBERATELY NOT BATCHED. Every plaintext still grows its own limbs through the
// ordinary pooled allocator, so each one is independently destructible and no plaintext's device
// memory is kept alive by a sibling's. Allocating the batch as one slab would cut the pool traffic
// as well, but it would tie the whole batch's lifetime together and make an individual release
// impossible without refcounting -- and the pool's contention is a separate problem with its own
// separate fix. What this file removes is upload, launch and fence count; what it leaves alone is
// the allocator.
//
// CUDA-ONLY BY CONSTRUCTION: this is a .cu file, and the root CMakeLists globs .cu sources only
// when FIDESLIB_ENABLE_CUDA is ON.
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <mutex>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/RNSPoly.cuh"
#include "CudaUtils.cuh" // PinnedStagingUploadGather
#include "LimbUtils.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "VectorGPU.cuh"

namespace FIDESlib::CKKS {

namespace {

/// Threads per block for the reduce kernel. Same 128 Limb<T>::loadCoefficients uses, so the launch
/// geometry per (plaintext, limb) is identical to the single-plaintext path.
constexpr uint32_t kReduceBlock = 128;

/// @brief Plaintexts per device pass (FIDESLIB_DEVICE_ENCODE_BATCH_MAX, default 32).
///
/// Two reasons there is a cap at all. (1) The chunk's shared staging buffer is `cap * N` words and
/// is allocated from the ordinary pool, so it must stay ONE size class -- hence a power of two, and
/// hence allocated at the full cap even for a short chunk, so a tail never opens a second class.
/// At logN = 16 the default is 32 * 65536 * 8 B = 16 MiB, which is one more pooled size class per
/// device (the pool refills a class 1 GiB at a time, GPUmalloc in CudaUtils.cu). (2) `int` is the
/// width of VectorGPU's size and of GPUmalloc's byte count, so the buffer has to stay well inside
/// it: 64 * 65536 * 8 B = 32 MiB is the ceiling here, far below INT_MAX.
///
/// Clamped to [1, 64] and rounded DOWN to a power of two.
size_t DeviceEncodeBatchMax() {
	static const size_t n = [] {
		constexpr size_t kDefault = 32;
		constexpr size_t kMax	  = 64;
		const char* env			  = std::getenv("FIDESLIB_DEVICE_ENCODE_BATCH_MAX");
		size_t want				  = kDefault;
		if (env != nullptr && env[0] != '\0') {
			char* end			 = nullptr;
			unsigned long long v = std::strtoull(env, &end, 10);
			if (end != env && *end == '\0' && v != 0)
				want = static_cast<size_t>(v < kMax ? v : kMax);
		}
		size_t pow2 = 1;
		while (pow2 * 2 <= want)
			pow2 *= 2;
		return pow2;
	}();
	return n;
}

void* LimbData(LimbImpl& l) {
	return l.index() == U32 ? static_cast<void*>(std::get<U32>(l).v.data) : static_cast<void*>(std::get<U64>(l).v.data);
}

void* LimbAux(LimbImpl& l) {
	return l.index() == U32 ? static_cast<void*>(std::get<U32>(l).aux.data) : static_cast<void*>(std::get<U64>(l).aux.data);
}

/**
 * @brief Device encode's CRT reduction for a whole batch: fit_to_native_ with two extra grid
 *        dimensions.
 *
 * Per thread this is line for line Limb<T>::loadCoefficients's kernel (Limb.cu:220-243) -- the same
 * ALGO_BARRETT reduce, the same `n > bigValueHf` negative branch, the same borrow-correcting
 * `(r >= d) ? r - d : r + q - d`. Read that kernel's comment for why each of those is what it is;
 * nothing about the arithmetic changes here, which is what makes the output bit-identical.
 *
 * The three grid dimensions are: x over the ring position, y over the LIMB, z over the PLAINTEXT.
 *
 *   * the prime comes from y alone (`primeid_flattened[primeid_init + blockIdx.y]`, the indexing
 *     every other batched kernel in this library uses), because limb j of every plaintext in the
 *     batch is the same tower -- that is precisely the precondition the caller enforces;
 *   * `diff` is derived on the device as `bigBound - q` instead of being passed in. The single-limb
 *     kernel takes the host's `bigBound - moduli[i]` and then uses `C_.primes[primeid]` for `q` in
 *     the same expression, so it already treats the two as the same number; the caller additionally
 *     rejects any modulus that is not this context's prime at that tower. Deriving it removes a
 *     per-limb argument table without changing a single computed value.
 *   * the source is the chunk's shared staging buffer, read at `blockIdx.z * N`, where
 *     `N == gridDim.x * blockDim.x`.
 *
 * Deliberately NOT a `gap`-strided scatter, for the same reason the single-limb kernel is not: the
 * caller uploads an already-expanded length-N vector, so the sparse-encoding zero fill is visible
 * in the uploaded bytes rather than hidden in the kernel.
 */
__global__ void fit_to_native_batch_(void** __restrict__ dst,
  const uint64_t* __restrict__ biased,
  const int __grid_constant__ primeid_init,
  const int __grid_constant__ limbs,
  const uint64_t bigValueHf,
  const uint64_t bigBound) {

	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const uint64_t q  = C_.primes[primeid];
	const uint64_t d0 = bigBound - q;

	const int i		 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t n = biased[(size_t)blockIdx.z * (size_t)(gridDim.x * blockDim.x) + (size_t)i];

	uint64_t r = modreduce<ALGO_BARRETT>((__uint128_t)n, primeid);
	if (n > bigValueHf) {
		const uint64_t d = modreduce<ALGO_BARRETT>((__uint128_t)d0, primeid);
		r				 = (r >= d) ? (r - d) : (r + q - d);
	}

	const int k = (int)blockIdx.z * limbs + (int)blockIdx.y;
	if (ISU64(primeid)) {
		((uint64_t*)dst[k])[i] = r;
	} else {
		// The residue is < prime < 2^32 for a 32-bit tower, so the narrowing store is exact --
		// same argument as the uint32_t specialization of Limb<T>::loadCoefficients.
		((uint32_t*)dst[k])[i] = (uint32_t)r;
	}
}

/// @brief One chunk of the batch: `n` polynomials, all already validated and all sharing `moduli`.
///
/// Takes ContextData explicitly so it needs no access to RNSPoly's private members; everything else
/// it touches (GPU, grow, freeNTTScratch) is public.
/// The encode NTT of a chunk: dst -> aux (stage one), aux -> dst (stage two), per limb, batched.
///
/// Per-limb mode (the default): `auxTab` holds each limb's own NTT scratch (Limb::aux), exactly
/// the two launches the shipped path has always made.
///
/// Scratch mode (FIDESLIB_ENCODE_SCRATCH=1): the limbs were grown constant and have no scratch;
/// one pooled buffer per chunk (kEncodeScratchBytes, a power of two so it is a pool class, not a
/// cudaMallocAsync on the legacy stream) serves as the scratch of `m` plaintexts at a time, and the
/// two launches run per sub-batch of `m` with the table pointers offset by `base * Lg` -- which is
/// how NTTBatch_ indexes its tables (k = blockIdx.z * limbs + blockIdx.y). The pointer table into
/// the scratch is the same for every sub-batch, so it is uploaded once. The kernels, their order
/// and their arguments per limb are the per-limb mode's; only where stage one writes differs.
constexpr size_t kEncodeScratchBytes = 64ull << 20;

void LaunchEncodeNtt(ContextData& cc, LimbPartition& lead, Stream& s, VectorGPU<void*>& dstTab, VectorGPU<void*>& auxTab, const size_t n, const int Lg, const bool scratch_mode) {
	constexpr int M = 4;
	const dim3 blockDimFirst{ (uint32_t)(1 << ((cc.logN + 1) / 2 - 1)) };
	const dim3 blockDimSecond{ (uint32_t)(1 << ((cc.logN) / 2 - 1)) };
	const int bytesFirst  = 8 * blockDimFirst.x * (2 * M + 1 + 1);
	const int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + 1);
	if (!scratch_mode) {
		NTTBatch_<false, ALGO_SHOUP><<<dim3{ cc.N / (blockDimFirst.x * M * 2), (uint32_t)Lg, (uint32_t)n }, blockDimFirst, bytesFirst, s.ptr()>>>(
		  lead.getGlobals(), dstTab.data, PARTITION(lead.id, 0), auxTab.data, Lg);
		NTTBatch_<true, ALGO_SHOUP><<<dim3{ cc.N / (blockDimSecond.x * M * 2), (uint32_t)Lg, (uint32_t)n }, blockDimSecond, bytesSecond, s.ptr()>>>(
		  lead.getGlobals(), auxTab.data, PARTITION(lead.id, 0), dstTab.data, Lg);
		return;
	}
	const size_t wordsPerPt	  = (size_t)Lg * (size_t)cc.N;
	const size_t scratchWords = kEncodeScratchBytes / sizeof(uint64_t);
	const size_t m			  = std::max<size_t>(1, std::min(n, scratchWords / wordsPerPt));
	const size_t allocWords	  = std::max(scratchWords, m * wordsPerPt); // m == 1 on a chain taller than the buffer: size to one plaintext
	VectorGPU<uint64_t> scratch(s, (int)allocWords, lead.device);
	std::vector<void*> h_sub(m * (size_t)Lg, nullptr);
	for (size_t b = 0; b < m; ++b) {
		for (int j = 0; j < Lg; ++j) {
			h_sub[b * (size_t)Lg + (size_t)j] = scratch.data + (b * (size_t)Lg + (size_t)j) * (size_t)cc.N;
		}
	}
	VectorGPU<void*> subTab(s, (int)(m * (size_t)Lg), lead.device, h_sub.data());
	for (size_t base = 0; base < n; base += m) {
		const uint32_t cur = (uint32_t)std::min(m, n - base);
		void** dst		   = dstTab.data + base * (size_t)Lg;
		NTTBatch_<false, ALGO_SHOUP><<<dim3{ cc.N / (blockDimFirst.x * M * 2), (uint32_t)Lg, cur }, blockDimFirst, bytesFirst, s.ptr()>>>(
		  lead.getGlobals(), dst, PARTITION(lead.id, 0), subTab.data, Lg);
		NTTBatch_<true, ALGO_SHOUP><<<dim3{ cc.N / (blockDimSecond.x * M * 2), (uint32_t)Lg, cur }, blockDimSecond, bytesSecond, s.ptr()>>>(
		  lead.getGlobals(), subTab.data, PARTITION(lead.id, 0), dst, Lg);
	}
	subTab.free(s);
	scratch.free(s);
}

void EncodeChunk(ContextData& cc,
  RNSPoly* const* polys,
  const std::vector<uint64_t>* const* biased,
  const size_t n,
  const size_t cap,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound) {

	const int limbsize		  = (int)moduli.size();
	const uint64_t bigValueHf = bigBound >> 1;
	const bool arena		  = PlaintextArena();			 // FIDESLIB_PT_ARENA: one pooled buffer per plaintext for its limbs
	const bool scratch_mode   = EncodeScratch() || arena; // FIDESLIB_ENCODE_SCRATCH: constant limbs, one NTT scratch per chunk

	// ---- per-polynomial allocation, UNCHANGED ----
	// grow(..., constant = false) because this path TRANSFORMS: ApplyNTT stages through Limb::aux,
	// and a constant-grown limb has no aux at all. Same call, same arguments, same pooled
	// allocations as the single-plaintext path -- see RNSPoly::loadCoefficients for the full
	// argument about why constant-grown is wrong here and why the scratch is released at the end.
	for (size_t b = 0; b < n; ++b) {
		if (arena)
			polys[b]->growArena(limbsize - 1);
		else
			polys[b]->grow(limbsize - 1, false, scratch_mode);
	}

	for (size_t g = 0; g < cc.GPUid.size(); ++g) {
		LimbPartition& lead = polys[0]->GPU[g];
		const int Lg		= (int)lead.getLimbSize(limbsize - 1);
		if (Lg <= 0)
			continue;

		cudaSetDevice(lead.device);
		// The whole chunk is driven on the FIRST polynomial's partition stream. Every other
		// polynomial's limbs were allocated on ITS OWN partition stream, so this stream must not run
		// ahead of those allocations; the matching fence in the other direction is at the bottom.
		Stream& s = lead.s;
		for (size_t b = 1; b < n; ++b) {
			s.wait(polys[b]->GPU[g].s);
		}

		// ---- the chunk's pointer tables, built on the host FIRST ----
		// Before any device allocation, deliberately: everything below that can throw throws here,
		// where there is no VectorGPU in scope whose destructor would fire without a free().
		std::vector<void*> h_dst((size_t)n * (size_t)Lg, nullptr);
		std::vector<void*> h_aux((size_t)n * (size_t)Lg, nullptr);
		// Limb streams that are NOT their partition's stream, and so need their own fence. Normally
		// empty: LimbPartition::generate binds every limb it creates to the partition stream
		// (USE_PARTITION_STREAM, LimbPartition.cu:187), which is what makes fencing the n partition
		// streams above sufficient. Checked rather than asserted because the cost of being wrong is
		// a wrong plaintext in a release build, and the cost of checking is a pointer compare.
		std::vector<Stream*> extra_fenced;
		for (size_t b = 0; b < n; ++b) {
			LimbPartition& part = polys[b]->GPU[g];
			assert((int)part.limb.size() == Lg);
			for (int j = 0; j < Lg; ++j) {
				Stream& ls = STREAM(part.limb[j]);
				if (&ls != &part.s) {
					s.wait(ls);
					extra_fenced.push_back(&ls);
				}
				h_dst[b * (size_t)Lg + (size_t)j] = LimbData(part.limb[j]);
				h_aux[b * (size_t)Lg + (size_t)j] = LimbAux(part.limb[j]);
				// A device encode transforms, so the NTT scratch has to be there. grow(constant =
				// false) allocates it; this catches a caller that handed us a constant-grown
				// polynomial before the NTT writes through a null pointer.
				if (!scratch_mode && h_aux[b * (size_t)Lg + (size_t)j] == nullptr) {
					throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: polynomial " + std::to_string(b) + " has no NTT scratch on limb " + std::to_string(j));
				}
			}
		}
		// ---- the chunk's shared device scratch: one staging buffer and the two tables ----
		// The staging buffer is always `cap` slots wide, never `n`: see DeviceEncodeBatchMax for
		// why the size class is pinned. Only the first n * N words are written and read.
		VectorGPU<uint64_t> staging(s, (int)(cap * (size_t)cc.N), lead.device);
		VectorGPU<void*> dstTab(s, (int)((size_t)n * (size_t)Lg), lead.device, h_dst.data());
		VectorGPU<void*> auxTab(s, (int)((size_t)n * (size_t)Lg), lead.device, h_aux.data());

		// ---- ONE host-to-device copy for the whole chunk's coefficients ----
		{
			const size_t bytes_each = (size_t)cc.N * sizeof(uint64_t);
			std::vector<const void*> srcs(n, nullptr);
			for (size_t b = 0; b < n; ++b) {
				srcs[b] = biased[b]->data();
			}
			if (!PinnedStagingUploadGather(staging.data, srcs.data(), n, bytes_each, lead.device, s.ptr())) {
				// The ring declined: build the contiguous image ourselves and take the pageable
				// copy. `flat` may die at the end of this block -- a pageable cudaMemcpyAsync does
				// not return until the driver has staged the bytes, which is the same lifetime
				// contract Limb<T>::load's converting branch relies on (Limb.cu:89-95).
				std::vector<uint64_t> flat(n * (size_t)cc.N);
				for (size_t b = 0; b < n; ++b) {
					std::memcpy(flat.data() + b * (size_t)cc.N, biased[b]->data(), bytes_each);
				}
				cudaMemcpyAsync(staging.data, flat.data(), n * bytes_each, cudaMemcpyHostToDevice, s.ptr());
			}
		}

		// ---- ONE CRT reduction for the whole chunk ----
		fit_to_native_batch_<<<dim3{ (uint32_t)cc.N / kReduceBlock, (uint32_t)Lg, (uint32_t)n }, dim3{ kReduceBlock }, 0, s.ptr()>>>(
		  dstTab.data, staging.data, PARTITION(lead.id, 0), Lg, bigValueHf, bigBound);

		// ---- ONE forward NTT for the whole chunk: COEFFICIENT -> EVALUATION ----
		// Geometry transcribed from LimbPartition::ApplyNTT (LimbPartition.cu:322-337) for
		// algo = ALGO_SHOUP, which is what RNSPoly::NTT's default instantiation uses; only the grid's
		// y/z extent differs (all Lg limbs of all n plaintexts at once instead of `cc.batch` limbs of
		// one). The +1 in the shared-memory term is ApplyNTT's `(algo == 2 || algo == ALGO_SHOUP ? 1
		// : 0)`, resolved for ALGO_SHOUP.
		LaunchEncodeNtt(cc, lead, s, dstTab, auxTab, n, Lg, scratch_mode);

		// ---- release the chunk's scratch ----
		// Safe here and only here, and for exactly the reason RNSPoly::loadCoefficients gives for
		// its own staging free: the pool's release is stream-ordered on `s`, and every kernel that
		// reads these three buffers was enqueued on `s` above, so the pool cannot hand them out
		// again ahead of that work.
		staging.free(s);
		dstTab.free(s);
		auxTab.free(s);

		// The other polynomials' partition streams must not run ahead of the transform that built
		// their limbs -- in particular not into freeNTTScratch below, which releases the very
		// scratch the second NTT stage just read.
		for (size_t b = 1; b < n; ++b) {
			polys[b]->GPU[g].s.wait(s);
		}
		for (Stream* ls : extra_fenced) {
			ls->wait(s);
		}
	}

	// ---- and the NTT scratch, exactly as the single-plaintext path does ----
	// Each release is stream-ordered on that polynomial's own partition streams, which the fence
	// above has just ordered behind the transform. See RNSPoly::loadCoefficients for why a plaintext
	// may give its scratch up: it is an operand, never a destination, and is never transformed again.
	if (!scratch_mode) {
		for (size_t b = 0; b < n; ++b) {
			polys[b]->freeNTTScratch();
		}
	}
}

// ---------------------------------------------------------------------------------------------
// THE TRANSFORM ON THE DEVICE (FIDESLIB_DEVICE_IFFT; see RNSPoly::loadSlotsBatch).
//
// DiscreteFourierTransform::FFTSpecialInv (openfhe src/core/lib/math/dftransform.cpp), stage for
// stage: a Gentleman-Sande ladder over len = size, size/2, ..., 2 whose twiddle for butterfly j of a
// block of length len is ksiPows[(lenq - rotGroup[j] % lenq) * (M / lenq)], lenq = 4 len, M = 2N;
// then a bit reversal and a division by size. CKKSPackedEncoding::Encode then multiplies by the
// scaling factor, llrounds the real and imaginary parts independently, biases a negative integer by
// Max64BitValue(), and scatters reals to positions gap*k and imaginaries to gap*(k + slots). The
// device does exactly these steps in this order. Its doubles are rounded-to-nearest at every op
// (the __d*_rn intrinsics, so nvcc contracts nothing into an FMA), which is what the host's
// std::complex arithmetic does when the compiler does not contract either; the two agree to the
// last bit except where a contraction on either side moves one, and the integers they round to are
// therefore numerically the same plaintext, not a guaranteed byte-identical one. An end-to-end GPU
// comparison with the host encode is the measurement of that.
// ---------------------------------------------------------------------------------------------
struct IfftTables {
	uint32_t* rotGroup = nullptr; // Nh entries: 5^i mod M
	double2* ksiPows   = nullptr; // M + 1 entries: (cos, sin)(2 pi j / M), [M] = [0]
	uint32_t M		   = 0;
	uint32_t Nh		   = 0;
};

/// One table set per (device, M), built once, exactly as PrecomputedValues builds the host's (same
/// expression, same libm), and kept for the process: 2 MB at N = 2^16.
const IfftTables& ifftTables(const int device, const uint32_t M, const uint32_t Nh) {
	static std::mutex mu;
	static std::map<std::pair<int, uint32_t>, IfftTables> cache;
	std::lock_guard<std::mutex> lk(mu);
	const auto key = std::make_pair(device, M);
	auto it		   = cache.find(key);
	if (it != cache.end()) {
		return it->second;
	}
	std::vector<uint32_t> rot(Nh);
	uint32_t fivePows = 1;
	for (size_t i = 0; i < Nh; ++i) {
		rot[i] = fivePows;
		fivePows *= 5;
		fivePows %= M;
	}
	std::vector<double2> ksi(M + 1);
	for (size_t j = 0; j < M; ++j) {
		double angle = 2.0 * M_PI * j / M;
		ksi[j]		 = make_double2(cos(angle), sin(angle));
	}
	ksi[M] = ksi[0];
	IfftTables t;
	t.M	 = M;
	t.Nh = Nh;
	cudaSetDevice(device);
	cudaMalloc(&t.rotGroup, Nh * sizeof(uint32_t));
	cudaMemcpy(t.rotGroup, rot.data(), Nh * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMalloc(&t.ksiPows, (M + 1) * sizeof(double2));
	cudaMemcpy(t.ksiPows, ksi.data(), (M + 1) * sizeof(double2), cudaMemcpyHostToDevice);
	CudaCheckErrorModNoSync;
	return cache.emplace(key, t).first->second;
}

constexpr uint32_t kIfftBlock = 256;

__device__ __forceinline__ double2 cmul_rn_(const double2 a, const double2 b) {
	// std::complex<double> operator*: (ac - bd, ad + bc), each product and sum rounded once.
	return make_double2(__dsub_rn(__dmul_rn(a.x, b.x), __dmul_rn(a.y, b.y)), __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x)));
}

/// Reals in, complex (re, 0) out; blockIdx.y is the plaintext.
__global__ void ifft_load_(double2* __restrict__ work, const double* __restrict__ vals, const uint32_t size) {
	const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= size)
		return;
	const size_t o = (size_t)blockIdx.y * size + k;
	work[o]		   = make_double2(vals[o], 0.0);
}

/// One Gentleman-Sande stage (one `len` of FFTSpecialInv's outer loop); one thread per butterfly.
__global__ void ifft_stage_(double2* __restrict__ work, const uint32_t* __restrict__ rotGroup, const double2* __restrict__ ksi, const uint32_t M, const uint32_t size, const uint32_t len) {
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= (size >> 1))
		return;
	double2* a			= work + (size_t)blockIdx.y * size;
	const uint32_t lenh = len >> 1;
	const uint32_t lenq = len << 2;
	const uint32_t gap	= M / lenq;
	const uint32_t blk	= t / lenh;
	const uint32_t j	= t - blk * lenh;
	const uint32_t i	= blk * len;
	const uint32_t idx	= (lenq - (rotGroup[j] % lenq)) * gap;
	const double2 x		= a[i + j];
	const double2 y		= a[i + j + lenh];
	a[i + j]			= make_double2(__dadd_rn(x.x, y.x), __dadd_rn(x.y, y.y));
	a[i + j + lenh]		= cmul_rn_(make_double2(__dsub_rn(x.x, y.x), __dsub_rn(x.y, y.y)), ksi[idx]);
}

/// The stages whose blocks fit in shared memory (len <= kIfftSmemElems), all in one launch: a
/// block owns `topLen` contiguous elements of one plaintext, runs len = topLen, topLen/2, ..., 2
/// on them in shared memory with a barrier between stages, and writes them back. The same
/// butterflies in the same order as ifft_stage_, so the result is bit-identical to the one-stage-
/// per-launch form; what changes is one read and one write of the array instead of two per stage.
constexpr uint32_t kIfftSmemElems = 2048; // double2 -> 32 KB of static shared memory

__global__ void ifft_stages_smem_(double2* __restrict__ work, const uint32_t* __restrict__ rotGroup, const double2* __restrict__ ksi, const uint32_t M, const uint32_t size, const uint32_t topLen) {
	__shared__ double2 sh[kIfftSmemElems];
	double2* a = work + (size_t)blockIdx.y * size + (size_t)blockIdx.x * topLen;
	for (uint32_t e = threadIdx.x; e < topLen; e += blockDim.x)
		sh[e] = a[e];
	__syncthreads();
	const uint32_t half = topLen >> 1;
	for (uint32_t len = topLen; len >= 2; len >>= 1) {
		const uint32_t lenh = len >> 1;
		const uint32_t lenq = len << 2;
		const uint32_t gap	= M / lenq;
		for (uint32_t t = threadIdx.x; t < half; t += blockDim.x) {
			const uint32_t blk = t / lenh;
			const uint32_t j   = t - blk * lenh;
			const uint32_t i   = blk * len;
			const uint32_t idx = (lenq - (rotGroup[j] % lenq)) * gap;
			const double2 x	   = sh[i + j];
			const double2 y	   = sh[i + j + lenh];
			sh[i + j]		   = make_double2(__dadd_rn(x.x, y.x), __dadd_rn(x.y, y.y));
			sh[i + j + lenh]   = cmul_rn_(make_double2(__dsub_rn(x.x, y.x), __dsub_rn(x.y, y.y)), ksi[idx]);
		}
		__syncthreads();
	}
	for (uint32_t e = threadIdx.x; e < topLen; e += blockDim.x)
		a[e] = sh[e];
}

/// Bit reversal, 1/size, the scaling factor, llround, the bias and the gap scatter into the biased
/// coefficient image fit_to_native_batch_ reads. `flag` (optional) is set when a coefficient is
/// larger than `magBound` (2^MAX_BITS_IN_WORD), where Encode would take its approxFactor path.
__global__ void ifft_finish_(uint64_t* __restrict__ staging,
  const double2* __restrict__ work,
  const uint32_t size,
  const uint32_t logSize,
  const uint32_t gap,
  const uint32_t N,
  const double scFact,
  const uint64_t bigBound,
  const double magBound,
  int* __restrict__ flag) {
	const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= size)
		return;
	const uint32_t r   = (logSize == 0) ? 0u : (__brev(k) >> (32 - logSize));
	const double2 x	   = work[(size_t)blockIdx.y * size + r];
	const double dsize = (double)size;
	const double re	   = __dmul_rn(__ddiv_rn(x.x, dsize), scFact);
	const double im	   = __dmul_rn(__ddiv_rn(x.y, dsize), scFact);
	if (flag != nullptr && (fabs(re) > magBound || fabs(im) > magBound))
		atomicOr(flag, 1);
	const long long ire = llround(re);
	const long long iim = llround(im);
	uint64_t* out		= staging + (size_t)blockIdx.y * N;
	out[(size_t)gap * k]		  = (ire < 0) ? (uint64_t)((long long)bigBound + ire) : (uint64_t)ire;
	out[(size_t)gap * (k + size)] = (iim < 0) ? (uint64_t)((long long)bigBound + iim) : (uint64_t)iim;
}

/// EncodeChunk with the transform on the device: the same grow / fence / table / reduce / NTT
/// structure (kept side by side rather than shared, so the shipped coefficient path is untouched),
/// with the coefficient upload replaced by a slot-value upload and the three kernels above.
void EncodeChunkSlots(ContextData& cc,
  RNSPoly* const* polys,
  const std::vector<double>* const* values,
  const size_t n,
  const size_t cap,
  const int slots,
  const double scFact,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound) {
	const int limbsize		  = (int)moduli.size();
	const uint64_t bigValueHf = bigBound >> 1;
	const bool arena		  = PlaintextArena();			 // FIDESLIB_PT_ARENA: one pooled buffer per plaintext for its limbs
	const bool scratch_mode   = EncodeScratch() || arena; // FIDESLIB_ENCODE_SCRATCH: constant limbs, one NTT scratch per chunk
	const bool check		  = DeviceIfftCheck();
	uint32_t logSize		  = 0;
	while ((1u << logSize) < (uint32_t)slots)
		++logSize;
	const uint32_t gap = (uint32_t)cc.N / (2u * (uint32_t)slots);
	for (size_t b = 0; b < n; ++b) {
		if (arena)
			polys[b]->growArena(limbsize - 1);
		else
			polys[b]->grow(limbsize - 1, false, scratch_mode);
	}
	for (size_t g = 0; g < cc.GPUid.size(); ++g) {
		LimbPartition& lead = polys[0]->GPU[g];
		const int Lg		= (int)lead.getLimbSize(limbsize - 1);
		if (Lg <= 0)
			continue;
		cudaSetDevice(lead.device);
		Stream& s = lead.s;
		for (size_t b = 1; b < n; ++b) {
			s.wait(polys[b]->GPU[g].s);
		}
		std::vector<void*> h_dst((size_t)n * (size_t)Lg, nullptr);
		std::vector<void*> h_aux((size_t)n * (size_t)Lg, nullptr);
		std::vector<Stream*> extra_fenced;
		for (size_t b = 0; b < n; ++b) {
			LimbPartition& part = polys[b]->GPU[g];
			assert((int)part.limb.size() == Lg);
			for (int j = 0; j < Lg; ++j) {
				Stream& ls = STREAM(part.limb[j]);
				if (&ls != &part.s) {
					s.wait(ls);
					extra_fenced.push_back(&ls);
				}
				h_dst[b * (size_t)Lg + (size_t)j] = LimbData(part.limb[j]);
				h_aux[b * (size_t)Lg + (size_t)j] = LimbAux(part.limb[j]);
				if (!scratch_mode && h_aux[b * (size_t)Lg + (size_t)j] == nullptr) {
					throw std::invalid_argument("RNSPoly::loadSlotsBatch: polynomial " + std::to_string(b) + " has no NTT scratch on limb " + std::to_string(j));
				}
			}
		}
		VectorGPU<uint64_t> staging(s, (int)(cap * (size_t)cc.N), lead.device);
		VectorGPU<uint64_t> dvals(s, (int)(n * (size_t)slots), lead.device);	   // the slot values, doubles bit-stored
		VectorGPU<uint64_t> work(s, (int)(n * (size_t)slots * 2u), lead.device); // double2 per slot
		VectorGPU<int> flag(s, 1, lead.device);
		VectorGPU<void*> dstTab(s, (int)((size_t)n * (size_t)Lg), lead.device, h_dst.data());
		VectorGPU<void*> auxTab(s, (int)((size_t)n * (size_t)Lg), lead.device, h_aux.data());
		{
			const size_t bytes_each = (size_t)slots * sizeof(double);
			std::vector<const void*> srcs(n, nullptr);
			for (size_t b = 0; b < n; ++b) {
				srcs[b] = values[b]->data();
			}
			if (!PinnedStagingUploadGather(dvals.data, srcs.data(), n, bytes_each, lead.device, s.ptr())) {
				std::vector<double> flat(n * (size_t)slots);
				for (size_t b = 0; b < n; ++b) {
					std::memcpy(flat.data() + b * (size_t)slots, values[b]->data(), bytes_each);
				}
				cudaMemcpyAsync(dvals.data, flat.data(), n * bytes_each, cudaMemcpyHostToDevice, s.ptr());
			}
		}
		if (check)
			cudaMemsetAsync(flag.data, 0, sizeof(int), s.ptr());
		if (gap > 1) // a sparse packing leaves the strided-out positions at zero, as NativeVector's zero-init does
			cudaMemsetAsync(staging.data, 0, n * (size_t)cc.N * sizeof(uint64_t), s.ptr());
		const IfftTables& tb   = ifftTables(lead.device, 2u * (uint32_t)cc.N, (uint32_t)cc.N / 2u);
		const uint32_t perSlot = ((uint32_t)slots + kIfftBlock - 1) / kIfftBlock;
		const uint32_t perBfly = (((uint32_t)slots >> 1) + kIfftBlock - 1) / kIfftBlock;
		ifft_load_<<<dim3{ perSlot, (uint32_t)n }, dim3{ kIfftBlock }, 0, s.ptr()>>>(reinterpret_cast<double2*>(work.data), reinterpret_cast<const double*>(dvals.data), (uint32_t)slots);
		// The stages whose blocks are wider than shared memory, one launch each; then every
		// remaining stage in one shared-memory launch per 2048-element block.
		for (uint32_t len = (uint32_t)slots; len > kIfftSmemElems; len >>= 1) {
			ifft_stage_<<<dim3{ perBfly, (uint32_t)n }, dim3{ kIfftBlock }, 0, s.ptr()>>>(reinterpret_cast<double2*>(work.data), tb.rotGroup, tb.ksiPows, tb.M, (uint32_t)slots, len);
		}
		if (slots >= 2) {
			const uint32_t topLen  = std::min<uint32_t>((uint32_t)slots, kIfftSmemElems);
			const uint32_t threads = std::min<uint32_t>(1024u, topLen >> 1);
			ifft_stages_smem_<<<dim3{ (uint32_t)slots / topLen, (uint32_t)n }, dim3{ threads }, 0, s.ptr()>>>(reinterpret_cast<double2*>(work.data), tb.rotGroup, tb.ksiPows, tb.M, (uint32_t)slots, topLen);
		}
		ifft_finish_<<<dim3{ perSlot, (uint32_t)n }, dim3{ kIfftBlock }, 0, s.ptr()>>>(staging.data,
		  reinterpret_cast<const double2*>(work.data),
		  (uint32_t)slots,
		  logSize,
		  gap,
		  (uint32_t)cc.N,
		  scFact,
		  bigBound,
		  std::ldexp(1.0, 61), // 2^MAX_BITS_IN_WORD (openfhe constants-defs.h)
		  check ? flag.data : nullptr);
		fit_to_native_batch_<<<dim3{ (uint32_t)cc.N / kReduceBlock, (uint32_t)Lg, (uint32_t)n }, dim3{ kReduceBlock }, 0, s.ptr()>>>(
		  dstTab.data, staging.data, PARTITION(lead.id, 0), Lg, bigValueHf, bigBound);
		LaunchEncodeNtt(cc, lead, s, dstTab, auxTab, n, Lg, scratch_mode);
		int overflow = 0;
		if (check) {
			cudaMemcpyAsync(&overflow, flag.data, sizeof(int), cudaMemcpyDeviceToHost, s.ptr());
			cudaStreamSynchronize(s.ptr());
			CudaCheckErrorModNoSync;
		}
		staging.free(s);
		dvals.free(s);
		work.free(s);
		flag.free(s);
		dstTab.free(s);
		auxTab.free(s);
		for (size_t b = 1; b < n; ++b) {
			polys[b]->GPU[g].s.wait(s);
		}
		for (Stream* ls : extra_fenced) {
			ls->wait(s);
		}
		if (overflow != 0) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: a coefficient exceeds 2^61 (OpenFHE's MAX_BITS_IN_WORD), where CKKSPackedEncoding::Encode takes its approxFactor path; "
										"use the host transform (FIDESLIB_DEVICE_IFFT=0) for this batch");
		}
	}
	if (!scratch_mode) {
		for (size_t b = 0; b < n; ++b) {
			polys[b]->freeNTTScratch();
		}
	}
}

} // namespace

void RNSPoly::loadCoefficientsBatch(const std::vector<RNSPoly*>& polys,
  const std::vector<const std::vector<uint64_t>*>& biased,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound) {

	if (polys.empty()) {
		return;
	}
	if (biased.size() != polys.size()) {
		throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: " + std::to_string(polys.size()) + " polynomials but " + std::to_string(biased.size()) +
		  " coefficient vectors");
	}
	if (polys[0] == nullptr) {
		throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: null polynomial at 0");
	}

	ContextData& cc	   = polys[0]->cc;
	const int limbsize = (int)moduli.size();

	// Preconditions are hard errors, never silent fallbacks -- the same set RNSPoly::loadCoefficients
	// enforces, for the same reason: this path produces limbs that nothing downstream re-checks, so
	// a wrong basis or a stale bias constant would surface only as a decryption that is quietly
	// garbage. Checked ONCE for the batch where the value is shared, and per polynomial otherwise.
	if (limbsize <= 0 || limbsize > cc.L + 1) {
		throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: expected 1.." + std::to_string(cc.L + 1) + " Q moduli, got " + std::to_string(limbsize));
	}
	if (bigBound == 0) {
		throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: bigBound must be the encoder's bias constant, not 0");
	}
	for (int i = 0; i < limbsize; ++i) {
		if (moduli[i] != cc.prime.at(i).p) {
			throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: modulus " + std::to_string(i) +
			  " is not this context's Q prime at that tower (device encode is Q-basis only)");
		}
	}
	for (size_t b = 0; b < polys.size(); ++b) {
		if (polys[b] == nullptr || biased[b] == nullptr) {
			throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: null entry at " + std::to_string(b));
		}
		// One context per batch: the shared staging buffer, the shared pointer tables and the shared
		// prime indexing all assume one ring dimension and one tower layout.
		if (&polys[b]->cc != &cc) {
			throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: polynomial " + std::to_string(b) + " belongs to a different context");
		}
		if (biased[b]->size() != (size_t)cc.N) {
			throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: polynomial " + std::to_string(b) + ": expected exactly N=" + std::to_string(cc.N) +
			  " coefficients, got " + std::to_string(biased[b]->size()) + " (the caller must expand a sparse encoding to full length before uploading)");
		}
		if (polys[b]->level != -1) {
			throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: polynomial " + std::to_string(b) + ": expected a freshly constructed polynomial (level -1), got level " +
			  std::to_string(polys[b]->level) + "; device encode allocates its own NTT scratch and cannot adopt a constant-grown polynomial");
		}
		// Aliasing would mean two entries of the pointer tables naming the same limbs, i.e. one
		// plaintext silently overwriting another's towers. Cheap to rule out at these batch sizes.
		for (size_t a = 0; a < b; ++a) {
			if (polys[a] == polys[b]) {
				throw std::invalid_argument("RNSPoly::loadCoefficientsBatch: polynomial " + std::to_string(b) + " appears twice in the batch");
			}
		}
	}

	const size_t cap = DeviceEncodeBatchMax();
	for (size_t base = 0; base < polys.size(); base += cap) {
		const size_t n = std::min(cap, polys.size() - base);
		EncodeChunk(cc, polys.data() + base, biased.data() + base, n, cap, moduli, bigBound);
	}
}

void Plaintext::loadCoefficientsBatch(const std::vector<Plaintext*>& pts,
  const std::vector<const std::vector<uint64_t>*>& biased,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound,
  const double noiseFactor,
  const int noiseLevel,
  const int slots_) {

	if (pts.empty()) {
		return;
	}
	CKKS::SetCurrentContext(pts[0]->cc_);

	std::vector<RNSPoly*> polys;
	polys.reserve(pts.size());
	for (size_t b = 0; b < pts.size(); ++b) {
		if (pts[b] == nullptr) {
			throw std::invalid_argument("Plaintext::loadCoefficientsBatch: null plaintext at " + std::to_string(b));
		}
		polys.push_back(&pts[b]->c0);
	}

	RNSPoly::loadCoefficientsBatch(polys, biased, moduli, bigBound);

	// Metadata last, and only once every polynomial is built: a throw above must leave no plaintext
	// claiming a scale it does not have.
	for (Plaintext* pt : pts) {
		pt->NoiseFactor = noiseFactor;
		pt->NoiseLevel	= noiseLevel;
		pt->slots		= slots_;
	}
}

void RNSPoly::loadSlotsBatch(const std::vector<RNSPoly*>& polys,
  const std::vector<const std::vector<double>*>& values,
  const int slots,
  const double scalingFactor,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound) {
	if (polys.empty()) {
		return;
	}
	if (values.size() != polys.size()) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: " + std::to_string(polys.size()) + " polynomials but " + std::to_string(values.size()) + " value vectors");
	}
	if (polys[0] == nullptr) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: null polynomial at 0");
	}
	ContextData& cc	   = polys[0]->cc;
	const int limbsize = (int)moduli.size();
	if (limbsize <= 0 || limbsize > cc.L + 1) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: expected 1.." + std::to_string(cc.L + 1) + " Q moduli, got " + std::to_string(limbsize));
	}
	if (bigBound == 0) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: bigBound must be the encoder's bias constant, not 0");
	}
	if (slots <= 0 || (slots & (slots - 1)) != 0 || slots > cc.N / 2) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: slots must be a power of two in 1..N/2, got " + std::to_string(slots));
	}
	if (!(scalingFactor > 0.0)) {
		throw std::invalid_argument("RNSPoly::loadSlotsBatch: the scaling factor must be positive");
	}
	for (int i = 0; i < limbsize; ++i) {
		if (moduli[i] != cc.prime.at(i).p) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: modulus " + std::to_string(i) + " is not this context's Q prime at that tower (device encode is Q-basis only)");
		}
	}
	for (size_t b = 0; b < polys.size(); ++b) {
		if (polys[b] == nullptr || values[b] == nullptr) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: null entry at " + std::to_string(b));
		}
		if (&polys[b]->cc != &cc) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: polynomial " + std::to_string(b) + " belongs to a different context");
		}
		if (values[b]->size() != (size_t)slots) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: polynomial " + std::to_string(b) + ": expected exactly " + std::to_string(slots) + " slot values, got " +
			  std::to_string(values[b]->size()) + " (the caller pads a sparse vector before uploading)");
		}
		if (polys[b]->level != -1) {
			throw std::invalid_argument("RNSPoly::loadSlotsBatch: polynomial " + std::to_string(b) + ": expected a freshly constructed polynomial (level -1), got level " +
			  std::to_string(polys[b]->level));
		}
		for (size_t a = 0; a < b; ++a) {
			if (polys[a] == polys[b]) {
				throw std::invalid_argument("RNSPoly::loadSlotsBatch: polynomial " + std::to_string(b) + " appears twice in the batch");
			}
		}
	}
	const size_t cap = DeviceEncodeBatchMax();
	for (size_t base = 0; base < polys.size(); base += cap) {
		const size_t n = std::min(cap, polys.size() - base);
		EncodeChunkSlots(cc, polys.data() + base, values.data() + base, n, cap, slots, scalingFactor, moduli, bigBound);
	}
}

void Plaintext::loadSlotsBatch(const std::vector<Plaintext*>& pts,
  const std::vector<const std::vector<double>*>& values,
  const int slots,
  const double scalingFactor,
  const std::vector<uint64_t>& moduli,
  const uint64_t bigBound,
  const double noiseFactor,
  const int noiseLevel) {
	if (pts.empty()) {
		return;
	}
	CKKS::SetCurrentContext(pts[0]->cc_);
	std::vector<RNSPoly*> polys;
	polys.reserve(pts.size());
	for (size_t b = 0; b < pts.size(); ++b) {
		if (pts[b] == nullptr) {
			throw std::invalid_argument("Plaintext::loadSlotsBatch: null plaintext at " + std::to_string(b));
		}
		polys.push_back(&pts[b]->c0);
	}
	RNSPoly::loadSlotsBatch(polys, values, slots, scalingFactor, moduli, bigBound);
	for (Plaintext* pt : pts) {
		pt->NoiseFactor = noiseFactor;
		pt->NoiseLevel	= noiseLevel;
		pt->slots		= slots;
	}
}

} // namespace FIDESlib::CKKS
