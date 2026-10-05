//
// Created by carlosad on 1/10/25.
//
#include <algorithm>
#include <array>
#include <atomic>
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

namespace FIDESlib::CKKS {

void LimbPartition::addBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb, int stride, double usage, bool sub, bool exta, bool extb) {
	ContextData& cc = parta[0]->cc;
	int limbsize	= parta[0]->getLimbSize(*parta[0]->level);
	int slimbsize	= extb ? parta[1]->SPECIALlimb.size() : 0;
	int n			= parta.size();

	dim3 block	= { 128u, 1u, 1u };
	int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
	dim3 grid	= { cc.N / 128u, (uint32_t)limbsize, 1u };
	uint32_t num_parallel_parts =
	  std::max(1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))), (uint32_t)parta.size()));
	grid.z = num_parallel_parts;

	int its	  = stride;
	int split = 1;
	if (num_parallel_parts > partb.size()) {
		split = (num_parallel_parts + partb.size() - 1) / partb.size();

		its	   = stride / split;
		grid.z = split * partb.size();
	}

	int size	= its * split * partb.size() + split * partb.size() + extb * (its * split * partb.size() + split * partb.size());
	int offset1 = its * split * partb.size(), offset2 = its * split * partb.size() + split * partb.size(),
		offset3 = its * split * partb.size() + split * partb.size() + its * split * partb.size();
	std::vector<void**> data_ptrs(size, nullptr);
	for (uint32_t i = 0; i < partb.size(); ++i) {

		for (int j = 0; j < stride; ++j) {
			data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
		}
		for (int j = 0; j < split; ++j) {
			data_ptrs[offset1 + i * split + j] = partb[i]->limbptr.data;
		}

		if (extb) {
			for (int j = 0; j < stride; ++j) {
				data_ptrs[offset2 + its * split * i + j] = parta[stride * i + j]->SPECIALlimbptr.data;
			}
			for (int j = 0; j < split; ++j) {
				data_ptrs[offset3 + i * split + j] = partb[i]->SPECIALlimbptr.data;
			}
		}
	}

	dim3 sgrid = { grid.x, (uint32_t)slimbsize, grid.z };

	Stream& s = parta[0]->s;

	void*** data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, parta[0]->device, s.ptr());
	s.wait(partb[0]->s);
	if (!sub) {
		if (!exta && !extb) {
			if (limbsize > 0)
				add_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
		}
		if (exta && extb) {
			if (limbsize > 0)
				add_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
			if (slimbsize > 0)
				add_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3, SPECIAL(parta[0]->id, 0), n, its);
		}
		if (exta && !extb) {
			if (limbsize > 0)
				add_reuse_scale_p_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
		}
		if (!exta && extb) {
			if (limbsize > 0)
				add_scale_p_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
			if (slimbsize > 0)
				copy_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3, SPECIAL(parta[0]->id, 0), n, its);
		}

	} else {
		if (!exta && !extb) {
			if (limbsize > 0)
				sub_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
		}
		if (exta && extb) {
			if (limbsize > 0)
				sub_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
			if (slimbsize > 0)
				sub_reuse_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3, SPECIAL(parta[0]->id, 0), n, its);
		}
		if (exta && !extb) {
			if (limbsize > 0)
				sub_reuse_scale_p_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
		}
		if (!exta && extb) {
			if (limbsize > 0)
				sub_scale_p_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + offset1, PARTITION(parta[0]->id, 0), n, its);
			if (slimbsize > 0)
				copy_reuse_negative_b___<<<sgrid, block, 0, s.ptr()>>>(data_ptrs_d + offset2, data_ptrs_d + offset3, SPECIAL(parta[0]->id, 0), n, its);
		}
	}
	partb[0]->s.wait(s);
	FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

void LimbPartition::multPtBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb, int stride, double usage) {
	ContextData& cc = parta[0]->cc;
	int limbsize	= parta[0]->getLimbSize(*parta[0]->level);
	int n			= parta.size();

	dim3 block	= { 128u, 1u, 1u };
	int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
	dim3 grid	= { cc.N / 128u, (uint32_t)limbsize, 1u };
	uint32_t num_parallel_parts =
	  std::max(1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))), (uint32_t)parta.size()));
	grid.z = num_parallel_parts;

	int its	  = stride;
	int split = 1;
	if (num_parallel_parts > partb.size()) {
		split = (num_parallel_parts + partb.size() - 1) / partb.size();

		its	   = stride / split;
		grid.z = split * partb.size();
	}

	int size = its * split * partb.size() + split * partb.size();
	std::vector<void**> data_ptrs(size, nullptr);
	for (uint32_t i = 0; i < partb.size(); ++i) {

		for (int j = 0; j < stride; ++j) {
			data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
		}
		for (int j = 0; j < split; ++j) {
			data_ptrs[its * split * partb.size() + i * split + j] = partb[i]->limbptr.data;
		}
	}

	Stream& s = parta[0]->s;

	void*** data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, parta[0]->device, s.ptr());
	s.wait(partb[0]->s);
	if (limbsize > 0)
		mult_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + its * split * partb.size(), PARTITION(parta[0]->id, 0), n, its);
	partb[0]->s.wait(s);
	FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

void LimbPartition::addScalarBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage) {
	ContextData& cc = parta[0]->cc;
	int limbsize	= parta[0]->getLimbSize(*parta[0]->level);
	int n			= parta.size();

	dim3 block	= { 128u, 1u, 1u };
	int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
	dim3 grid	= { cc.N / 128u, (uint32_t)limbsize, 1u };
	uint32_t num_parallel_parts =
	  std::max(1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))), (uint32_t)parta.size()));
	grid.z = num_parallel_parts;

	int its	  = stride;
	int split = 1;
	if (num_parallel_parts > vector.size()) {
		split = (num_parallel_parts + vector.size() - 1) / vector.size();

		its	   = stride / split;
		grid.z = split * vector.size();
	}

	int size = its * split * vector.size() + split * vector.size() * MAXP;
	std::vector<void**> data_ptrs(size, nullptr);
	for (uint32_t i = 0; i < vector.size(); ++i) {

		for (int j = 0; j < stride; ++j) {
			data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
		}
		for (int j = 0; j < split; ++j) {
			std::memcpy(&data_ptrs[its * split * vector.size() + (i * split + j) * MAXP], vector[i].data(), vector[i].size() * sizeof(uint64_t));
			// data_ptrs[its * split * vector.size() + (i * split + j) * MAXP] = vector[i].data;
		}
	}

	Stream& s = parta[0]->s;

	void*** data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, parta[0]->device, s.ptr());

	if (limbsize > 0)
		add_scalar_reuse_b___<<<grid, block, 0, s.ptr()>>>(data_ptrs_d, data_ptrs_d + its * split * vector.size(), PARTITION(parta[0]->id, 0), n, its);
	FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

void LimbPartition::multScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
  const std::vector<std::vector<unsigned long int>>& vector,
  const std::vector<std::vector<unsigned long int>>& vector_shoup,
  int stride,
  double usage) {
	ContextData& cc = parta[0]->cc;
	int limbsize	= parta[0]->getLimbSize(*parta[0]->level);
	int n			= parta.size();

	dim3 block	= { 128u, 1u, 1u };
	int threads = GetTargetThreads(cc.GPUid[parta[0]->id]) * usage;
	dim3 grid	= { cc.N / 128u, (uint32_t)limbsize, 1u };
	uint32_t num_parallel_parts =
	  std::max(1u, std::min((uint32_t)((threads + block.x * grid.x * grid.y - 1) / (block.x * grid.x * std::max(1u, grid.y))), (uint32_t)parta.size()));
	grid.z = num_parallel_parts;

	int its	  = stride;
	int split = 1;
	if (num_parallel_parts > vector.size()) {
		split = (num_parallel_parts + vector.size() - 1) / vector.size();

		its	   = stride / split;
		grid.z = split * vector.size();
	}

	int size = its * split * vector.size() + split * vector.size() * MAXP * 2;
	std::vector<void**> data_ptrs(size, nullptr);
	for (uint32_t i = 0; i < vector.size(); ++i) {

		for (int j = 0; j < stride; ++j) {
			data_ptrs[its * split * i + j] = parta[stride * i + j]->limbptr.data;
		}
		for (int j = 0; j < split; ++j) {
			std::memcpy(&data_ptrs[its * split * vector.size() + (i * split + j) * MAXP], vector[i].data(), vector[i].size() * sizeof(uint64_t));
		}
		for (int j = 0; j < split; ++j) {
			std::memcpy(&data_ptrs[its * split * vector.size() + split * vector.size() * MAXP + (i * split + j) * MAXP],
			  vector_shoup[i].data(),
			  vector_shoup[i].size() * sizeof(uint64_t));
		}
	}

	Stream& s = parta[0]->s;

	void*** data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, parta[0]->device, s.ptr());

	if (limbsize > 0)
		mult_scalar_reuse_b___<<<grid, block, 0, s.ptr()>>>(
		  data_ptrs_d, data_ptrs_d + its * split * vector.size(), data_ptrs_d + its * split * vector.size() + split * vector.size() * MAXP, PARTITION(parta[0]->id, 0), n, its);
	FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

void LimbPartition::LTdotProductPtBatch(std::vector<LimbPartition*>& out,
  const std::vector<LimbPartition*>& in,
  const std::vector<LimbPartition*>& pt,
  int bStep,
  int gStep,
  int stride,
  double usage,
  bool ext,
  void*** prepared_pt_q,
  void*** prepared_pt_p) {

	constexpr bool VER2 = true;
	constexpr bool VER3 = false;

	cudaSetDevice(out[0]->device);
	ContextData& cc	  = out[0]->cc;
	int limbsize	  = out[0]->getLimbSize(*out[0]->level);
	int slimbsize	  = cc.splitSpecialMeta.at(out[0]->id).size();
	int special_start = 0;
	for (int i = 0; i < out[0]->id; ++i) {
		special_start += cc.splitSpecialMeta.at(i).size();
	}

	dim3 block;
	dim3 grid;
	if constexpr (!VER2) {
		int blockdimx = 16;
		while (bStep * 2 * blockdimx > 256 && blockdimx > 2) {
			blockdimx /= 2;
		}
		block = { (uint32_t)blockdimx, 2, (uint32_t)bStep };
		grid  = { (uint32_t)cc.N / blockdimx, (uint32_t)limbsize, 1u };
	} else {
		block = { (uint32_t)16, 2, 4 };
		grid  = { (uint32_t)cc.N / 64, (uint32_t)limbsize, 1u };
	}
	int threads = GetTargetThreads(cc.GPUid[out[0]->id]) * usage;

	uint32_t num_parallel_parts = std::max(1u,
	  std::min((uint32_t)((threads + block.x * block.y * block.z * grid.x * grid.y - 1) / (block.x * block.y * block.z * grid.x * std::max(1u, grid.y))),
		(uint32_t)out.size()));
	grid.z						= num_parallel_parts;

	dim3 sgrid = grid;
	sgrid.y	   = slimbsize;

	assert(out.size() % (2 * gStep * stride) == 0);
	assert(in.size() % (2 * bStep * stride) == 0);
	assert(pt.size() % (gStep * bStep) == 0);
	assert(out.size() / (2 * gStep * stride) == in.size() / (2 * bStep * stride));
	assert(out.size() / (2 * gStep * stride) == pt.size() / (gStep * bStep));
	uint32_t shmem_bytes;
	int num_LT;
	if constexpr (!VER2) {
		shmem_bytes = sizeof(uint64_t) * (stride * 2 + 4 + 1) * block.x * block.z;
		num_LT		= out.size() / (2 * stride * gStep);
	} else {
		shmem_bytes = sizeof(uint64_t) * (gStep * (VER3 ? 2 : 1)) * block.y * block.x * block.z;
		num_LT		= out.size() / (2 * gStep);
	}

	// `prepared` means the caller handed us a device diagonal table that was uploaded once
	// and outlives this call (CKKS/PreparedLT.cuh). The diagonal third of the pointer table is then
	// neither rebuilt on the host nor re-uploaded, so `size` covers the per-call thirds alone and
	// the kernels read their `pt` base straight out of the resident buffer. Everything else — the
	// index arithmetic, the launch shapes, the stream waits, the pointers the kernel ends up
	// reading — is unchanged, which is why a prepared call and an unprepared one are the same
	// computation and not two implementations of it.
	const bool prepared = prepared_pt_q != nullptr;
	assert(!prepared || !ext || prepared_pt_p != nullptr);

	int size		   = (out.size() + in.size() + (prepared ? 0u : pt.size())) * (1 + ext);
	int offset_out_c0  = 0;
	int offset_out_c1  = out.size() / 2;
	int offset_in_c0   = out.size();
	int offset_in_c1   = out.size() + in.size() / 2;
	int offset_pt	   = out.size() + in.size();
	int soffset_out_c0 = size / 2 + 0;
	int soffset_out_c1 = size / 2 + out.size() / 2;
	int soffset_in_c0  = size / 2 + out.size();
	int soffset_in_c1  = size / 2 + out.size() + in.size() / 2;
	int soffset_pt	   = size / 2 + out.size() + in.size();

	std::vector<void**> data_ptrs(size, nullptr);
	for (int i = 0; i < num_LT; ++i) {
		for (int j = 0; j < stride; ++j) {
			for (int k = 0; k < gStep; ++k) {
				data_ptrs[offset_out_c0 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k)]->limbptr.data;
			}
		}

		for (int j = 0; j < stride; ++j) {
			for (int k = 0; k < gStep; ++k) {
				data_ptrs[offset_out_c1 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k) + 1]->limbptr.data;
			}
		}

		for (int j = 0; j < stride; ++j) {
			for (int k = 0; k < bStep; ++k) {
				data_ptrs[offset_in_c0 + i * stride * bStep + j * bStep + k] = in[2 * (i * stride * bStep + j * bStep + k)]->limbptr.data;
			}
		}

		for (int j = 0; j < stride; ++j) {
			for (int k = 0; k < bStep; ++k) {
				data_ptrs[offset_in_c1 + i * stride * bStep + j * bStep + k] = in[2 * (i * stride * bStep + j * bStep + k) + 1]->limbptr.data;
			}
		}

		// Skipped when the caller handed us a resident diagonal table — these are exactly
		// the entries that table already holds, at exactly these indices.
		if (!prepared) {
			for (int j = 0; j < gStep; ++j) {
				for (int k = 0; k < bStep; ++k) {
					data_ptrs[offset_pt + i * gStep * bStep + j * bStep + k] =
					  pt[i * gStep * bStep + j * bStep + k] ? pt[i * gStep * bStep + j * bStep + k]->limbptr.data : nullptr;
				}
			}
		}

		if (ext) {
			for (int j = 0; j < stride; ++j) {
				for (int k = 0; k < gStep; ++k) {
					data_ptrs[soffset_out_c0 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k)]->SPECIALlimbptr.data + special_start;
				}
			}

			for (int j = 0; j < stride; ++j) {
				for (int k = 0; k < gStep; ++k) {
					data_ptrs[soffset_out_c1 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k) + 1]->SPECIALlimbptr.data + special_start;
				}
			}

			for (int j = 0; j < stride; ++j) {
				for (int k = 0; k < bStep; ++k) {
					data_ptrs[soffset_in_c0 + i * stride * bStep + j * bStep + k] = in[2 * (i * stride * bStep + j * bStep + k)]->SPECIALlimbptr.data + special_start;
				}
			}

			for (int j = 0; j < stride; ++j) {
				for (int k = 0; k < bStep; ++k) {
					data_ptrs[soffset_in_c1 + i * stride * bStep + j * bStep + k] = in[2 * (i * stride * bStep + j * bStep + k) + 1]->SPECIALlimbptr.data + special_start;
				}
			}

			if (!prepared) {
				for (int j = 0; j < gStep; ++j) {
					for (int k = 0; k < bStep; ++k) {
						data_ptrs[soffset_pt + i * gStep * bStep + j * bStep + k] =
						  pt[i * gStep * bStep + j * bStep + k] ? pt[i * gStep * bStep + j * bStep + k]->SPECIALlimbptr.data : nullptr;
					}
				}
			}
		}
	}

	Stream& s = in[0]->s;

	// FIDESLIB_PERSIST_TABLES (2026-10-03): the batched linear transform's pointer table
	// has the same output-pointer structure as fusedHoistRotate's digits and was never on the
	// persist ring either. Class 4. See LimbPartitionMGPU.cu fusedHoistRotate for the rationale.
	const bool persist_tables = FIDESlib::PersistOpTables() || FIDESlib::PersistChurn();
	void*** data_ptrs_d = nullptr;
	if (persist_tables) {
		data_ptrs_d = static_cast<void***>(FIDESlib::OpTableBuffer(in[0]->device, sizeof(void**) * size, 4));
		if (FIDESlib::PersistChurn()) {
			void* dummy = FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr());
			FIDESlib::OpFreeAsync(dummy, s.ptr());
		}
	} else {
		data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	}
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, in[0]->device, s.ptr());
	s.wait(out[0]->s);
	s.wait(pt[0]->s);

	// The diagonal bases the kernels read: the resident table when there is one, otherwise the
	// slice of the per-call table we just filled. Nothing downstream can tell the difference.
	void*** const pt_q_base = prepared ? prepared_pt_q : data_ptrs_d + offset_pt;
	void*** const pt_p_base = prepared ? prepared_pt_p : (ext ? data_ptrs_d + soffset_pt : nullptr);

	if constexpr (!VER2) {
		if (limbsize > 0) {
			dotProductLtBatchedPt___<<<grid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + offset_out_c0,
			  data_ptrs_d + offset_out_c1,
			  data_ptrs_d + offset_in_c0,
			  data_ptrs_d + offset_in_c1,
			  pt_q_base,
			  stride,
			  gStep,
			  PARTITION(out[0]->id, 0),
			  num_LT);
		}
		if (ext && slimbsize > 0) {
			dotProductLtBatchedPt___<<<sgrid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + soffset_out_c0,
			  data_ptrs_d + soffset_out_c1,
			  data_ptrs_d + soffset_in_c0,
			  data_ptrs_d + soffset_in_c1,
			  pt_p_base,
			  stride,
			  gStep,
			  SPECIAL(out[0]->id, special_start),
			  num_LT);
		}
	} else {
		if constexpr (!VER3) {
			if (limbsize > 0) {
				dotProductLtBatchedPt2___<<<grid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + offset_out_c0,
				  data_ptrs_d + offset_out_c1,
				  data_ptrs_d + offset_in_c0,
				  data_ptrs_d + offset_in_c1,
				  pt_q_base,
				  bStep,
				  gStep,
				  PARTITION(out[0]->id, 0),
				  num_LT);
			}
			if (ext && slimbsize > 0) {
				dotProductLtBatchedPt2___<<<sgrid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + soffset_out_c0,
				  data_ptrs_d + soffset_out_c1,
				  data_ptrs_d + soffset_in_c0,
				  data_ptrs_d + soffset_in_c1,
				  pt_p_base,
				  bStep,
				  gStep,
				  SPECIAL(out[0]->id, special_start),
				  num_LT);
			}
		} else {
			if (limbsize > 0) {
				dotProductLtBatchedPt3___<<<grid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + offset_out_c0,
				  data_ptrs_d + offset_out_c1,
				  data_ptrs_d + offset_in_c0,
				  data_ptrs_d + offset_in_c1,
				  pt_q_base,
				  bStep,
				  gStep,
				  PARTITION(out[0]->id, 0),
				  num_LT);
			}
			if (ext && slimbsize > 0) {
				dotProductLtBatchedPt3___<<<sgrid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + soffset_out_c0,
				  data_ptrs_d + soffset_out_c1,
				  data_ptrs_d + soffset_in_c0,
				  data_ptrs_d + soffset_in_c1,
				  pt_p_base,
				  bStep,
				  gStep,
				  SPECIAL(out[0]->id, special_start),
				  num_LT);
			}
		}
	}
	out[0]->s.wait(s);
	pt[0]->s.wait(s);
	if (!persist_tables)
		FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

void LimbPartition::fusedHoistedRotateBatch(std::vector<LimbPartition*>& out,
  const std::vector<LimbPartition*>& in,
  const std::vector<LimbPartition*>& ksk_a,
  const std::vector<LimbPartition*>& ksk_b,
  const std::vector<int>& indexes,
  int n,
  int stride,
  double usage,
  bool c0_modup) {
	ContextData& cc = out[0]->cc;
	int limbsize	= out[0]->getLimbSize(*out[0]->level);
	int slimbsize	= out[0]->SPECIALlimb.size();
	int blockdimx	= 64;
	while (stride * 2 * blockdimx > 256 && blockdimx > 2) {
		blockdimx /= 2;
	}

	dim3 block = { (uint32_t)blockdimx, 2, (uint32_t)stride };

	int threads = GetTargetThreads(cc.GPUid[out[0]->id]) * usage;
	dim3 grid	= { (uint32_t)cc.N / blockdimx, (uint32_t)slimbsize + limbsize, 1u };

	uint32_t num_parallel_parts = std::max(1u,
	  std::min((uint32_t)((threads + block.x * block.y * block.z * grid.x * grid.y - 1) / (block.x * block.y * block.z * grid.x * std::max(1u, grid.y))),
		(uint32_t)out.size()));
	// grid.z = num_parallel_parts;
	// ONE-SHOT: this is a tuning note, not an error. EvalRotateMany (the score-path ladder arm)
	// calls this thousands of times per pass, and a per-call line would bury the run's real output.
	if (num_parallel_parts > 1) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set())
			std::cerr << "fusedHoistRotateBatch detected underutilization, should optimize" << std::endl;
	}

	// dim3 sgrid = grid;
	//  sgrid.y = slimbsize;

	size_t num_d = 0;

	for (; num_d < in[1]->DIGITlimb.size(); ++num_d) {
		{
			int start = 0;
			for (uint32_t j = 0; j < num_d; ++j)
				start += in[1]->DECOMPmeta[j].size();
			int size = std::min((int)in[1]->DECOMPmeta[num_d].size(), (int)limbsize - start);
			if (size <= 0)
				break;
		}
	}

	uint32_t shmem_bytes = sizeof(uint64_t) * (num_d * block.x * block.z + 2 * num_d * block.x);

	assert(out.size() == in.size() * n);
	assert(ksk_a.size() * in.size() == out.size());
	assert(ksk_b.size() * in.size() == out.size());

	int size		   = 2 * out.size() + in.size() + (num_d + 1) * in.size() / 2 + (num_d + 1) * (ksk_a.size() + ksk_b.size()) + (indexes.size() + 1) / 2;
	int offset_out_c0  = 0;
	int offset_out_sc0 = out.size() / 2;
	int offset_out_c1  = out.size();
	int offset_out_sc1 = 3 * out.size() / 2;
	int offset_in_c0   = 2 * out.size();
	int offset_in_sc0  = 2 * out.size() + in.size() / 2;
	int offset_in_c1   = 2 * out.size() + 2 * in.size() / 2;
	int offset_in_dc1  = 2 * out.size() + 3 * in.size() / 2;
	int offset_ksk	   = 2 * out.size() + (3 + num_d) * in.size() / 2;
	int offset_index   = 2 * out.size() + (3 + num_d) * in.size() / 2 + (num_d + 1) * (ksk_a.size() + ksk_b.size());

	std::vector<void**> data_ptrs(size, nullptr);

	for (uint32_t j = 0; j < out.size() / 2; ++j) {
		data_ptrs[offset_out_c0 + j]  = out[2 * j]->limbptr.data;
		data_ptrs[offset_out_sc0 + j] = out[2 * j]->SPECIALlimbptr.data;
		data_ptrs[offset_out_c1 + j]  = out[2 * j + 1]->limbptr.data;
		data_ptrs[offset_out_sc1 + j] = out[2 * j + 1]->SPECIALlimbptr.data;
	}

	for (uint32_t j = 0; j < in.size() / 2; ++j) {
		data_ptrs[offset_in_c0 + j]	 = in[2 * j]->limbptr.data;
		data_ptrs[offset_in_sc0 + j] = in[2 * j]->SPECIALlimbptr.data;
		data_ptrs[offset_in_c1 + j]	 = in[2 * j + 1]->limbptr.data;
		for (uint32_t k = 0; k < num_d; ++k)
			data_ptrs[offset_in_dc1 + num_d * j + k] = in[2 * j + 1]->DIGITlimbptr[k].data;
	}

	for (uint32_t j = 0; j < ksk_a.size(); ++j) {
		for (uint32_t k = 0; k < num_d; ++k) {
			data_ptrs[offset_ksk + (num_d + 1) * 2 * j + k]		  = ksk_a[j] ? ksk_a[j]->DIGITlimbptr[k].data : nullptr;
			data_ptrs[offset_ksk + (num_d + 1) * (2 * j + 1) + k] = ksk_b[j] ? ksk_b[j]->DIGITlimbptr[k].data : nullptr;
		}

		data_ptrs[offset_ksk + (num_d + 1) * 2 * j + num_d]		  = ksk_a[j] ? ksk_a[j]->limbptr.data : nullptr;
		data_ptrs[offset_ksk + (num_d + 1) * (2 * j + 1) + num_d] = ksk_b[j] ? ksk_b[j]->limbptr.data : nullptr;
		((int*)&data_ptrs[offset_index])[j]						  = indexes[j];
	}

	Stream& s = in[0]->s;

	void*** data_ptrs_d = static_cast<void***>(FIDESlib::OpMallocAsync(sizeof(void**) * size, s.ptr()));
	// cudaMalloc(&data_ptrs_d, sizeof(void**) * size);
	UploadH2D(data_ptrs_d, data_ptrs.data(), sizeof(void**) * size, in[0]->device, s.ptr());
	// EVERY input and EVERY output, not just the first of each. One source per batch entry means
	// one stream per batch entry, and the kernel reads all of them: waiting only on in[0]/out[0]
	// was sufficient while the only conceivable caller was one-source/many-indexes (where every
	// other `in` is the same poly), and races as soon as a caller batches independent sources --
	// which is exactly what Ciphertext::rotate_many does. Mirrors the wait set of the single-source
	// LimbPartition::fusedHoistRotate (LimbPartitionMGPU.cu).
	for (auto* p : in)
		if (p)
			s.wait(p->s);
	for (auto* p : out)
		if (p)
			s.wait(p->s);
	s.wait(ksk_a[0] ? ksk_a[0]->s : ksk_a[1]->s);

	if (limbsize + slimbsize > 0)
		hoistedRotateDotKSKBatched___<<<grid, block, shmem_bytes, s.ptr()>>>(data_ptrs_d + offset_in_c1,
		  data_ptrs_d + offset_in_dc1,
		  data_ptrs_d + offset_in_c0,
		  data_ptrs_d + offset_in_sc0,
		  data_ptrs_d + offset_out_c1,
		  data_ptrs_d + offset_out_sc1,
		  data_ptrs_d + offset_out_c0,
		  data_ptrs_d + offset_out_sc0,
		  n,
		  (int*)(data_ptrs_d + offset_index),
		  data_ptrs_d + offset_ksk,
		  num_d,
		  out[0]->id,
		  slimbsize,
		  0,
		  c0_modup);

	for (auto* p : in)
		if (p)
			p->s.wait(s);
	for (auto* p : out)
		if (p)
			p->s.wait(s);
	(ksk_a[0] ? ksk_a[0]->s : ksk_a[1]->s).wait(s);
	FIDESlib::OpFreeAsync(data_ptrs_d, s.ptr());
}

} // namespace FIDESlib::CKKS