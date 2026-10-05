//
// A RESIDENT device pointer table for the batched linear-transform MAC.
//

#include "CKKS/PreparedLT.cuh"

#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "CudaUtils.cuh"

#include <cassert>
#include <stdexcept>
#include <string>

namespace FIDESlib::CKKS {

int PreparedLTTable::chunkGiantSteps(int c) const {
	const int g_in = c * kGiantChunk;
	const int rest = gStep_ - g_in;
	return rest < kGiantChunk ? rest : kGiantChunk;
}

int PreparedLTTable::ptEntries(int c) const {
	return num_LT_ * chunkGiantSteps(c) * bStep_;
}

int PreparedLTTable::chunkOffset(int chunk, bool special) const {
	int off = 0;
	for (int c = 0; c < chunk; ++c)
		off += ptEntries(c) * (ext_ ? 2 : 1);
	return special ? off + ptEntries(chunk) : off;
}

void*** PreparedLTTable::ptQ(int partition, int chunk) const {
	return buffer_[static_cast<size_t>(partition)] + chunkOffset(chunk, false);
}

void*** PreparedLTTable::ptP(int partition, int chunk) const {
	if (!ext_)
		return nullptr;
	return buffer_[static_cast<size_t>(partition)] + chunkOffset(chunk, true);
}

// The table is built and uploaded HERE, once, and never touched again. Everything in this
// constructor is deliberately off the hot path: a blocking cudaMalloc and a blocking cudaMemcpy
// per GPU partition. Both are the kind of call the steady-state path must never make (the
// blocking malloc would serialize the pool's streams, the blocking copy would stall the host) and
// both are exactly right once per group lifetime, which is what "prepared" means.
PreparedLTTable::PreparedLTTable(ContextData& cc, const std::vector<std::vector<Plaintext*>>& pts, int rowSize, int bStep, int stride, bool ext)
: num_LT_(static_cast<int>(pts.size())), rowSize_(rowSize), bStep_(bStep), stride_(stride), ext_(ext) {
	if (pts.empty())
		throw std::runtime_error("PreparedLTTable: needs at least one diagonal set");
	if (rowSize <= 0 || bStep <= 0)
		throw std::runtime_error("PreparedLTTable: rowSize and bStep must be positive");
	gStep_	= (rowSize + bStep - 1) / bStep;
	chunks_ = (gStep_ + kGiantChunk - 1) / kGiantChunk;

	entries_ = 0;
	for (int c = 0; c < chunks_; ++c)
		entries_ += ptEntries(c) * (ext_ ? 2 : 1);

	const size_t parts = pts[0].empty() ? 0 : pts[0][0]->c0.GPU.size();
	if (parts == 0)
		throw std::runtime_error("PreparedLTTable: the diagonals are not resident on any device");
	for (const auto& set : pts) {
		if (static_cast<int>(set.size()) < rowSize)
			throw std::runtime_error("PreparedLTTable: a diagonal set is shorter than rowSize");
		for (int k = 0; k < rowSize; ++k) {
			if (set[static_cast<size_t>(k)] == nullptr)
				throw std::runtime_error("PreparedLTTable: diagonal " + std::to_string(k) + " is null");
			if (set[static_cast<size_t>(k)]->c0.GPU.size() != parts)
				throw std::runtime_error("PreparedLTTable: the diagonals do not agree on a GPU partition count");
		}
	}

	buffer_.assign(parts, nullptr);
	device_.assign(parts, -1);
	host_.assign(parts, {});

	for (size_t p = 0; p < parts; ++p) {
		// Build the host image in EXACTLY the index order LimbPartition::LTdotProductPtBatch uses
		// for its `offset_pt` / `soffset_pt` sections, chunk by chunk. The index arithmetic below
		// is the same arithmetic, read against the GLOBAL diagonal list instead of the per-chunk
		// slice RNSPoly::LTdotProductPtBatch would have cut.
		std::vector<void**>& h = host_[p];
		h.assign(static_cast<size_t>(entries_), nullptr);
		for (int c = 0; c < chunks_; ++c) {
			const int g_in	= c * kGiantChunk;
			const int g_len = chunkGiantSteps(c);
			const int base	= chunkOffset(c, false);
			const int sbase = ext_ ? chunkOffset(c, true) : 0;
			for (int t = 0; t < num_LT_; ++t) {
				for (int j = 0; j < g_len; ++j) {
					for (int k = 0; k < bStep_; ++k) {
						const int local	 = t * g_len * bStep_ + j * bStep_ + k;
						const int global = (g_in + j) * bStep_ + k;
						// The tail of the last giant step is null-padded here, exactly as the
						// transform null-pads it: the kernel reads a null diagonal as "skip".
						Plaintext* pt = global < rowSize_ ? pts[static_cast<size_t>(t)][static_cast<size_t>(global)] : nullptr;
						h[static_cast<size_t>(base + local)] = pt ? pt->c0.GPU[p].limbptr.data : nullptr;
						if (ext_)
							h[static_cast<size_t>(sbase + local)] = pt ? pt->c0.GPU[p].SPECIALlimbptr.data : nullptr;
					}
				}
			}
		}

		const int dev = cc.GPUid[p];
		device_[p]	  = dev;
		cudaSetDevice(dev);
		void*** d = nullptr;
		// A plain cudaMalloc, deliberately NOT the pooled GPUmalloc: the pool orders every
		// allocation behind the most recent free on any stream through one shared event
		// (src/CudaUtils.cu:402,444-446), and this buffer outlives every stream that will read it,
		// so it has no business in the pool's lifetime bookkeeping.
		const cudaError_t e = cudaMalloc(&d, sizeof(void**) * static_cast<size_t>(entries_));
		if (e != cudaSuccess)
			throw std::runtime_error(std::string("PreparedLTTable: cudaMalloc failed: ") + cudaGetErrorString(e));
		const cudaError_t e2 = cudaMemcpy(d, h.data(), sizeof(void**) * static_cast<size_t>(entries_), cudaMemcpyHostToDevice);
		if (e2 != cudaSuccess) {
			cudaFree(d);
			throw std::runtime_error(std::string("PreparedLTTable: upload failed: ") + cudaGetErrorString(e2));
		}
		buffer_[p] = d;
	}
}

PreparedLTTable::~PreparedLTTable() {
	for (size_t p = 0; p < buffer_.size(); ++p) {
		if (buffer_[p] == nullptr)
			continue;
		cudaSetDevice(device_[p]);
		cudaFree(buffer_[p]);
		buffer_[p] = nullptr;
	}
}

} // namespace FIDESlib::CKKS
