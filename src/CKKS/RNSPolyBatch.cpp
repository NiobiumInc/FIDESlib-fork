//
// Created by carlosad on 1/10/25.
//
#include <errno.h>

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/PreparedLT.cuh"
#include "CKKS/RNSPoly.cuh"

#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace FIDESlib::CKKS {

void RNSPoly::addBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage, bool sub, bool exta, bool extb) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		std::vector<LimbPartition*> partb;
		partb.reserve(polyb.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		for (auto j : polyb) {
			partb.push_back(&j->GPU[i]);
		}
		LimbPartition::addBatchManyToOne(parta, partb, stride, usage, sub, exta, extb);
	}
}

void RNSPoly::multPtBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		std::vector<LimbPartition*> partb;
		partb.reserve(polyb.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		for (auto j : polyb) {
			partb.push_back(&j->GPU[i]);
		}
		LimbPartition::multPtBatchManyToOne(parta, partb, stride, usage);
	}
}

void RNSPoly::addScalarBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		LimbPartition::addScalarBatchManyToOne(parta, vector, stride, usage);
	}
}

void RNSPoly::multScalarBatchManyToOne(std::vector<RNSPoly*>& polya,
  const std::vector<std::vector<unsigned long int>>& vector,
  const std::vector<std::vector<unsigned long int>>& vectors,
  int stride,
  double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		LimbPartition::multScalarBatchManyToOne(parta, vector, vectors, stride, usage);
	}
}

namespace {
// Verification hook for the resident diagonal table (FIDESLIB_R2_VERIFY_TABLE=1, default off).
//
// The resident diagonal table is built from the api layer's view of the diagonals and consumed deep
// inside the batch kernel dispatch, and the two index it by arithmetic written in two different
// files. There is no way to prove they agree by inspection, and a disagreement would be a WRONG
// ANSWER rather than a crash — so this rebuilds, on the host, exactly the entries
// LimbPartition::LTdotProductPtBatch would have built and compares them against what the table
// holds. Off by default and read once; on, it costs one host pass over the diagonal pointers per
// call and turns any drift into a throw naming the entry. Run a gate once with it set.
bool R2VerifyTable() {
	static const bool on = [] {
		const char* v = std::getenv("FIDESLIB_R2_VERIFY_TABLE");
		return v != nullptr && std::atoi(v) != 0;
	}();
	return on;
}

void R2CheckChunk(const PreparedLTTable& prepared, int partition, int chunk, const std::vector<LimbPartition*>& pts, int bStep, int g_len) {
	const auto& image = prepared.hostImage(partition);
	const int base	  = prepared.chunkOffset(chunk, false);
	const int sbase	  = prepared.ext() ? prepared.chunkOffset(chunk, true) : 0;
	const int num_LT  = prepared.numLT();
	for (int t = 0; t < num_LT; ++t) {
		for (int j = 0; j < g_len; ++j) {
			for (int k = 0; k < bStep; ++k) {
				const int local		   = t * g_len * bStep + j * bStep + k;
				const LimbPartition* p = pts[static_cast<size_t>(local)];
				void* const want	   = p ? static_cast<void*>(p->limbptr.data) : nullptr;
				if (static_cast<void*>(image[static_cast<size_t>(base + local)]) != want) {
					throw std::runtime_error("LTdotProductPtBatch: the prepared diagonal table disagrees with the call at partition " +
					  std::to_string(partition) + " chunk " + std::to_string(chunk) + " entry " + std::to_string(local) + " (Q basis)");
				}
				if (prepared.ext()) {
					void* const wantS = p ? static_cast<void*>(p->SPECIALlimbptr.data) : nullptr;
					if (static_cast<void*>(image[static_cast<size_t>(sbase + local)]) != wantS) {
						throw std::runtime_error("LTdotProductPtBatch: the prepared diagonal table disagrees with the call at partition " +
						  std::to_string(partition) + " chunk " + std::to_string(chunk) + " entry " + std::to_string(local) + " (P basis)");
					}
				}
			}
		}
	}
}
} // namespace

void RNSPoly::LTdotProductPtBatch(std::vector<RNSPoly*>& out,
  const std::vector<RNSPoly*>& in,
  const std::vector<RNSPoly*>& pt,
  int bStep,
  int gStep,
  int stride,
  double usage,
  bool ext,
  const PreparedLTTable* prepared) {
	ContextData& cc = out[0]->cc;
	// The resident diagonal table is indexed by (GPU partition, giant-step CHUNK), and the
	// chunking is the loop below — so this is the one place that knows which chunk a given
	// LimbPartition call is. Check the shape here, once, rather than letting a table built for
	// another transform be indexed as if it were this one: a wrong table is a wrong answer.
	if (prepared != nullptr) {
		const int num_LT = static_cast<int>(out.size()) / (2 * gStep);
		if (prepared->bStep() != bStep || prepared->gStep() != gStep || prepared->stride() != stride || prepared->ext() != ext || prepared->numLT() != num_LT) {
			throw std::runtime_error("LTdotProductPtBatch: the prepared diagonal table was built for a different transform shape");
		}
		if (prepared->partitions() != static_cast<int>(cc.GPUid.size()))
			throw std::runtime_error("LTdotProductPtBatch: the prepared diagonal table was built for a different GPU partition count");
	}
	// One chunk covers the whole transform here; the `else` below is the multi-chunk form.
	if (gStep <= PreparedLTTable::kGiantChunk) {
#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
		for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
			std::vector<LimbPartition*> outs;
			std::vector<LimbPartition*> ins;
			std::vector<LimbPartition*> pts;
			outs.reserve(out.size());
			for (auto j : out) {
				outs.push_back(&j->GPU[i]);
			}
			ins.reserve(in.size());
			for (auto j : in) {
				ins.push_back(&j->GPU[i]);
			}
			pts.reserve(pt.size());
			for (auto j : pt) {
				if (j)
					pts.push_back(&j->GPU[i]);
				else
					pts.push_back(nullptr);
			}

			if (prepared && R2VerifyTable())
				R2CheckChunk(*prepared, static_cast<int>(i), 0, pts, bStep, gStep);
			LimbPartition::LTdotProductPtBatch(outs, ins, pts, bStep, gStep, stride, usage, ext,
			  prepared ? prepared->ptQ(static_cast<int>(i), 0) : nullptr,
			  prepared ? prepared->ptP(static_cast<int>(i), 0) : nullptr);
		}

		for (auto i : out) {
			i->SetModUp(ext);
		}
	} else {
		if (1) {
			int num_LT = out.size() / (2 * gStep);
			for (int g_in = 0; g_in < gStep; g_in += PreparedLTTable::kGiantChunk) {
				int g_internal	 = std::min(PreparedLTTable::kGiantChunk, gStep - g_in);
				const int chunk_ = g_in / PreparedLTTable::kGiantChunk;
				// int num_out	   = num_LT * stride * g_in;
#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
				for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
					assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
					std::vector<LimbPartition*> outs;
					std::vector<LimbPartition*> ins;
					std::vector<LimbPartition*> pts;
					outs.reserve(out.size());

					for (int i_ = 0; i_ < num_LT; ++i_) {
						for (int j = 0; j < stride; ++j) {
							for (int k = g_in; k < g_in + g_internal; ++k) {
								outs.push_back(&out[2 * (i_ * stride * gStep + j * gStep + k)]->GPU[i]);
								outs.push_back(&out[2 * (i_ * stride * gStep + j * gStep + k) + 1]->GPU[i]);
								// data_ptrs[offset_out_c0 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k)]->limbptr.data;
							}
						}
					}

					ins.reserve(in.size());
					for (auto j : in) {
						ins.push_back(&j->GPU[i]);
					}
					pts.reserve(pt.size());

					for (int i_ = 0; i_ < num_LT; i_++) {
						for (int j = g_in; j < g_in + g_internal; ++j) {
							for (int k = 0; k < bStep; ++k) {
								pts.push_back(pt[i_ * gStep * bStep + j * bStep + k] ? &pt[i_ * gStep * bStep + j * bStep + k]->GPU[i] : nullptr);
							}
						}
					}

					cudaSetDevice(cc.GPUid[i]);

					outs[0]->s.wait(out[0]->GPU[i].s);
					pts[0]->s.wait(pt[0]->GPU[i].s);
					if (prepared && R2VerifyTable())
						R2CheckChunk(*prepared, static_cast<int>(i), chunk_, pts, bStep, g_internal);
					LimbPartition::LTdotProductPtBatch(outs, ins, pts, bStep, g_internal, stride, usage, ext,
					  prepared ? prepared->ptQ(static_cast<int>(i), chunk_) : nullptr,
					  prepared ? prepared->ptP(static_cast<int>(i), chunk_) : nullptr);
					out[0]->GPU[i].s.wait(outs[0]->s);
					pt[0]->GPU[i].s.wait(pts[0]->s);
				}
			}

			for (auto i : out) {
				i->SetModUp(ext);
			}

		} else { // TODO correct for stride != 1
			assert(stride == 1);
			// ---- Added safety checks ----
			// The fallback implementation assumes an even number of output polynomials.
			assert(out.size() % 2 == 0 && "LTdotProductPtBatch fallback requires an even number of output polynomials");

			// const std::size_t nBatches = out.size() / 2; // Number of output pairs processed
			//  Plain‑text vector must be large enough to provide `bStep` factors per batch.
			// const std::size_t minPtSize = static_cast<std::size_t>(bStep) * nBatches;
			// assert(pt.size() >= minPtSize && "Plain‑text vector `pt` is too small for the requested batch configuration");

			// Input vector must contain enough operands for all batches.
			// For each group of `gStep` batches we need `2 * bStep` inputs per batch.
			// const std::size_t groups	= (nBatches + static_cast<std::size_t>(gStep) - 1) / static_cast<std::size_t>(gStep);
			// const std::size_t minInSize = static_cast<std::size_t>(bStep) * 2 * groups;
			// assert(in.size() >= minInSize && "Input vector `in` is too small for the requested batch configuration");
			// ---- End of safety checks ----

			// #pragma omp parallel for
			for (uint32_t i = 0; i < out.size() / 2; ++i) {
				std::vector<const RNSPoly*> c0s;
				std::vector<const RNSPoly*> c1s;
				std::vector<const RNSPoly*> pts_;
				for (int j = 0; j < bStep; ++j) {
					if (!pt[bStep * i + j])
						break;
					c0s.emplace_back(in[bStep * 2 * (i / gStep) + 2 * j]);
					c1s.emplace_back(in[bStep * 2 * (i / gStep) + 2 * j + 1]);
					pts_.emplace_back(pt[bStep * i + j]);
				}

				out[2 * i]->dotProductPt(*out[2 * i + 1], c0s, c1s, pts_, ext);
			}
		}
	}
}

void RNSPoly::fusedHoistedRotateBatch(std::vector<RNSPoly*>& out,
  const std::vector<RNSPoly*>& in,
  const std::vector<RNSPoly*>& ksk_a,
  const std::vector<RNSPoly*>& ksk_b,
  const std::vector<int>& indexes,
  int stride,
  double usage,
  bool c0_modup) {

	uint32_t n = indexes.size();
	std::vector<int> index(n);
	for (uint32_t j = 0; j < n; ++j) {
		index[j] = in[0]->automorph_index_precomp(indexes[j]);
	}

#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
	for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
		std::vector<LimbPartition*> outs;
		std::vector<LimbPartition*> ins;
		std::vector<LimbPartition*> ksk_as;
		std::vector<LimbPartition*> ksk_bs;

		outs.reserve(out.size());
		for (auto j : out) {
			outs.push_back(&j->GPU[i]);
		}
		ins.reserve(in.size());
		for (auto j : in) {
			ins.push_back(&j->GPU[i]);
		}
		ksk_as.reserve(ksk_a.size());
		for (auto j : ksk_a) {
			ksk_as.push_back(j ? &j->GPU[i] : nullptr);
		}
		ksk_bs.reserve(ksk_b.size());
		for (auto j : ksk_b) {
			ksk_bs.push_back(j ? &j->GPU[i] : nullptr);
		}

		LimbPartition::fusedHoistedRotateBatch(outs, ins, ksk_as, ksk_bs, index, n, stride, usage, c0_modup);

		for (auto j : out) {
			j->SetModUp(true);
		}
	}
}

} // namespace FIDESlib::CKKS