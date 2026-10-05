//
// Created by carlosad on 25/04/24.
//
#include <errno.h>
#include "CudaUtils.cuh"

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "../parallel_for.hpp"

// #define OMP omp_disabled
#define OMP omp
/**
#define OMP_ASSERT(x) \
	do {              \
		x;            \
	} while (0)
*/
#define OMP_ASSERT(x) assert(x);

namespace FIDESlib::CKKS {
void RNSPoly::grow(int new_level, bool single_malloc, bool constant) {
	if (level >= new_level)
		return;
	level = new_level;

	// single_malloc = false;
	// if (level == -1) {
	// std::cout << "from 0" << std::endl;
	if (!constant && (!single_malloc || (GPU.at(0).limb.size() > 0)) && GPU.at(0).bufferLIMB == nullptr) {
		// TODO fix bug (check that limb size matches level)
		int init = 0;
		for (auto& g : GPU)
			init += g.limb.size();

		for (auto& g : GPU) {
			g.generateLimbToLevel(new_level);
		}
		// for (int i = init; i <= new_level; ++i) {
		//     GPU.at(cc.limbGPUid.at(i).x).generateLimb();
		// }
	} else {
		// #pragma omp parallel for num_threads(GPU.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			//          OMP_ASSERT(omp_get_num_threads() == (int)GPU.size());
			if (!constant) {
				GPU.at(i).generateLimbSingleMalloc();
			} else {
				GPU.at(i).generateLimbConstant();
			}
		}
	}
}

void RNSPoly::growArena(const int new_level) {
	if (level != -1)
		throw std::logic_error("FIDESlib: growArena needs a freshly constructed polynomial (level -1)");
	level = new_level;
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		const int n = GPU.at(i).getLimbSize(new_level);
		if (n > 0)
			GPU.at(i).generateLimbArena(n);
	}
}

RNSPoly::RNSPoly(ContextData& context, int level, bool single_malloc, bool def_stream)
	: uid(next_uid++), cc(context), level(-1) {

	// #pragma omp parallel for num_threads(context.GPUid.size())
	for (size_t i = 0u; i < context.GPUid.size(); ++i) {
		cudaSetDevice(context.GPUid.at(i));
		GPU.emplace_back(context, uid, &this->level, i, def_stream);
	}
	assert(level >= -1 && level <= cc.L);
	grow(level, single_malloc);
}

RNSPoly::RNSPoly(ContextData& context, const std::vector<std::vector<uint64_t>>& data)
	: RNSPoly(context, data.size() - 1) {

	assert(data.size() <= cc.prime.size());
	std::vector<uint64_t> moduli(data.size());
	for (size_t i = 0; i < data.size(); ++i)
		moduli[i] = cc.prime[i].p;
	load(data, moduli);
}

RNSPoly::RNSPoly(RNSPoly&& src) noexcept
	: uid(src.uid), cc(src.cc), level(src.level), modUp(src.modUp), GPU(std::move(src.GPU)), aux_slot(src.aux_slot) {
	for (auto& g : GPU) {
		g.level = &(this->level);
	}
	// src.setLevel(-2);
}

int32_t RNSPoly::getLevel() const {
	return level;
}

void RNSPoly::freeSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).freeSpecialLimbs();
	}
	this->SetModUp(false);
}

void RNSPoly::generateSpecialLimbs(const bool zero_out, const bool for_communication) {
	if (!GPU[0].SPECIALlimb.empty()) {
		if (zero_out) {
#pragma omp parallel for num_threads(cc.GPUid.size())
			for (size_t i = 0; i < cc.GPUid.size(); ++i) {
				assert(omp_get_num_threads() == (int)cc.GPUid.size());
				GPU[i].generateSpecialLimb(zero_out, for_communication);
			}
		}
		return;
	}

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generateSpecialLimb(zero_out, for_communication);
	}

	if (PEER_ACCESS && for_communication) {
		std::vector<void*> cpu_ptr(GPU[0].SPECIALlimb.size(), nullptr);
		for (uint32_t j = 0; j < cc.splitSpecialMeta.size(); ++j) {
			for (uint32_t g = 0; g < cc.splitSpecialMeta[j].size(); ++g) {
				for (uint32_t i = 0; i < cc.specialMeta[j].size(); ++i) {
					if (cc.specialMeta[0][i].id == cc.splitSpecialMeta[j][g].id) {
						cpu_ptr[i] = GPU[j].SPECIALlimb[i].index() == U32 ?
							(void*)std::get<U32>(GPU[j].SPECIALlimb[i]).v.data :
							(void*)std::get<U64>(GPU[j].SPECIALlimb[i]).v.data;
					}
				}
			}
		}

		if (GPU[0].SPECIALlimb.size() * sizeof(void*) > 0) {
			CudaCheckErrorMod;
			for (size_t g = 0; g < GPU.size(); ++g) {
				// std::cout << GPU[g].DECOMPlimbptr[i].data << " " << cpu_ptr.data() << " " << GPU[g].DECOMPmeta.at(i).size() * sizeof(void*) << " "
				//		  << cudaMemcpyHostToDevice << " " << GPU[g].s.ptr() << std::endl;
				cudaSetDevice(cc.GPUid[g]);
				UploadH2D(GPU[g].SPECIALlimbptr.data, cpu_ptr.data(), GPU[g].SPECIALmeta.size() * sizeof(void*), GPU[g].device, GPU[g].s.ptr());
				CudaCheckErrorMod;
			}
		}
	}
}

void RNSPoly::generateDecompAndDigit(bool iskey) {

	if (!GPU[0].DECOMPlimb[0].empty())
		return;

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generateAllDecompAndDigit(iskey);
	}

	/** To support peer access kernels we only need to change the pointers to the other libs*/
	if (PEER_ACCESS) {
		for (size_t i = 0; i < GPU[0].DECOMPmeta.size(); ++i) {
			std::vector<void*> cpu_ptr(GPU[0].DECOMPmeta.at(i).size(), nullptr);
			for (uint32_t j = 0; j < GPU[0].DECOMPlimb[i].size(); ++j) {
				for (size_t g = 0; g < cc.meta.size(); ++g) {
					for (auto& m : cc.meta[g]) {
						if (m.id == cc.decompMeta[0][i][j].id) {
							cpu_ptr[j] = GPU[g].DECOMPlimb[i][j].index() == U32 ?
								(void*)std::get<U32>(GPU[g].DECOMPlimb[i][j]).v.data :
								(void*)std::get<U64>(GPU[g].DECOMPlimb[i][j]).v.data;
						}
					}
				}
			}

			if (GPU[0].DECOMPmeta.at(i).size() * sizeof(void*) > 0) {
				CudaCheckErrorMod;
				for (size_t g = 0; g < GPU.size(); ++g) {
					// std::cout << GPU[g].DECOMPlimbptr[i].data << " " << cpu_ptr.data() << " " << GPU[g].DECOMPmeta.at(i).size() * sizeof(void*) << " "
					//		  << cudaMemcpyHostToDevice << " " << GPU[g].s.ptr() << std::endl;
					cudaSetDevice(cc.GPUid[g]);
					cudaMemcpyAsync(
						GPU[g].DECOMPlimbptr[i].data,
						cpu_ptr.data(),
						GPU[g].DECOMPmeta.at(i).size() * sizeof(void*),
						cudaMemcpyHostToDevice,
						GPU[g].s.ptr());
					CudaCheckErrorMod;
				}
			}
		}
	}
}

void RNSPoly::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).loadDecompDigit(data, moduli);
	}
}

void RNSPoly::store(std::vector<std::vector<uint64_t>>& data) {
	data.resize(level + 1);
	for (size_t i = 0; i < data.size(); ++i) {
		// auto& rec = cc.meta[cc.limbGPUid[i].x][cc.limbGPUid[i].y];
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], store_convert(data[i]));
	}
}

bool RNSPoly::isModUp() const {
	return modUp;
}

void RNSPoly::SetModUp(bool newValue) {
	modUp = newValue;
}

void RNSPoly::scaleByP() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).scaleByP();
	}
	this->SetModUp(true);
}

void RNSPoly::multNoModdownEnd(RNSPoly& c0, const RNSPoly& bc0, const RNSPoly& bc1, const RNSPoly& in, const RNSPoly& aux) {
	assert(in.isModUp());
	assert(aux.isModUp());
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multNoModdownEnd(c0.GPU.at(i), bc0.GPU.at(i), bc1.GPU.at(i), in.GPU.at(i), aux.GPU.at(i));
	}
	this->SetModUp(true);
	c0.SetModUp(true);
}

void RNSPoly::binomialMult(RNSPoly& c1, RNSPoly& in, const RNSPoly& d0, const RNSPoly& d1, bool moddown, bool square) {
	assert(!this->isModUp() && !c1.isModUp() && !d0.isModUp() && !d1.isModUp());

	if (!moddown) {
		this->generateSpecialLimbs(true, false);
		CudaCheckErrorMod;
		c1.generateSpecialLimbs(true, false);
	}

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).binomialMult(c1.GPU.at(i), in.GPU.at(i), d0.GPU.at(i), d1.GPU.at(i), !moddown, square);
	}
	this->SetModUp(!moddown);
	c1.SetModUp(!moddown);
	in.SetModUp(!moddown);
}

void RNSPoly::add(const RNSPoly& p) {

	if (p.isModUp() && !this->isModUp()) {
		// std::cout << "Adapt non modup destination add" << std::endl;
		generateSpecialLimbs(false, false);
		// scaleByP();
	}
	assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).add(p.GPU.at(i), this->isModUp(), p.isModUp());
	}
	this->SetModUp(this->isModUp() || p.isModUp());
}

void RNSPoly::sub(const RNSPoly& p) {
	assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).sub(p.GPU.at(i));
	}
}

void RNSPoly::modup() {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kModup);
	//  assert(GPU.size() == 1 || 0 == "ModUp Multi-GPU not implemented.");
	RNSPoly& aux = cc.getKeySwitchAux2();

	{
		ConcurrentOpsDiagScope diag_alloc(ConcurrentOpsDiag::kModupAlloc); // DIAG sub-kind: scratch allocation
		generateDecompAndDigit(false);
		aux.generateDecompAndDigit(false);
	}

	std::vector<std::atomic_uint64_t> thread_stop_buffer(cc.GPUid.size() * 8);
	std::vector<std::atomic_uint64_t*> thread_stop(cc.GPUid.size(), nullptr);
	for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
		thread_stop[k]            = &thread_stop_buffer[8 * k];
		thread_stop_buffer[8 * k] = 0;
	}

	std::vector<Stream*> external_s;
	for (uint32_t i = 0; i < GPU.size(); ++i) {
		external_s.push_back(&GPU[i].s);
	}

	std::vector<uint64_t*> buffergather;
	for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
		buffergather.push_back(GPU.at(i).bufferGATHER);
	}
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			GPU.at(i).modupMGPU(aux.GPU.at(i), buffergather, thread_stop, external_s);
		}
	} else {
#pragma omp parallel num_threads(GPU.size())
		{
			int i = omp_get_thread_num();

			if (omp_get_num_threads() != (int)GPU.size())
				throw std::invalid_argument("OMP didn't create enough threads");
			assert(omp_get_num_threads() == (int)GPU.size());
			assert(static_cast<size_t>(i) < GPU.size());
			GPU.at(i).modupMGPU(aux.GPU.at(i), buffergather, thread_stop, external_s);
		}
	}
}

void RNSPoly::sync() {
	for (auto& i : GPU) {
		for (auto& j : i.limb) {
			cudaStreamSynchronize(STREAM(j).ptr());
		}
		cudaStreamSynchronize(i.s.ptr());
	}
}

void RNSPoly::rescale() {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kRescale);
	//    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
	if (GPU.size() == 1) {
		for (auto& i : GPU) {
			i.rescale();
		}
		level -= 1;
	} else {
		int more_than_0 = 0;
		for (size_t i = 0; i < GPU.size(); ++i)
			if (GPU[i].getLimbSize(level) != 0)
				more_than_0++;

		if (more_than_0 == 1) {
			for (auto& i : GPU) {
				if (i.getLimbSize(level) > 0)
					i.rescale();
			}
		} else {
#pragma omp parallel num_threads(GPU.size())
			{
#pragma omp for
				for (size_t i = 0; i < GPU.size(); ++i) {
					if (omp_get_num_threads() != (int)GPU.size())
						throw std::invalid_argument("OMP didn't create enough threads");
					assert(omp_get_num_threads() == (int)GPU.size());
					GPU[i].rescaleMGPU();
				}
			}
		}
		level -= 1;
	}
}

void RNSPoly::rescaleDouble(RNSPoly& poly) {
	// DIAG (2026-10-03): the SAME scope RNSPoly::rescale() carries. Ciphertext::rescale routes to this
	// function (RESCALE_DOUBLE = true), so every "DIAG=rescale" sweep of the hunt serialized a path the
	// bootstrap never takes and left this one, the one it does take, unlocked. Measuring nothing.
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kRescale);
	//    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
	if (0 && GPU.size() == 1) {
		for (auto& i : GPU) {
			i.rescale();
		}
		level -= 1;
		for (auto& i : poly.GPU) {
			i.rescale();
		}
		poly.level -= 1;
	} else {

		int more_than_0 = 0;
		for (size_t i = 0; i < GPU.size(); ++i)
			if (GPU[i].getLimbSize(level) > 0)
				more_than_0++;

		if (0 && more_than_0 == 1) {
			for (size_t i = 0; i < GPU.size(); ++i) {
				if (GPU[i].getLimbSize(level) > 0) {
					GPU[i].rescale();
					poly.GPU[i].rescale();
				}
			}
		} else {

			if (0 && MEMCPY_PEER) {

				int id = cc.limbGPUid[level].x;
				GPU[id].doubleRescaleMGPU(poly.GPU[id]);
				for (size_t i = 0; i < GPU.size(); ++i) {

					// if (omp_get_num_threads() != (int)GPU.size())
					//     throw std::invalid_argument("OMP didn't create enough threads");
					// assert(omp_get_num_threads() == (int)GPU.size());
					// GPU[i].rescaleMGPU();
					if ((int32_t)i != id)
						GPU[i].doubleRescaleMGPU(poly.GPU[i]);
				}
			} else {
#pragma omp parallel num_threads(GPU.size())
				{
					int i = omp_get_thread_num();

					if (omp_get_num_threads() != (int)GPU.size())
						throw std::invalid_argument("OMP didn't create enough threads");
					assert(omp_get_num_threads() == (int)GPU.size());
					assert(static_cast<size_t>(i) < GPU.size());
					// GPU[i].rescaleMGPU();
					GPU[i].doubleRescaleMGPU(poly.GPU[i]);
				}
			}
			level -= 1 + (level == cc.L + 1 && cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT);
			poly.level -= 1 + (poly.level == cc.L + 1 && cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT);
		}
	}
}

void RNSPoly::multPt(const RNSPoly& p, bool rescale) {
	if (rescale) {
		if (GPU.size() == 1) {
			for (size_t i = 0; i < GPU.size(); ++i) {
				GPU.at(i).multPt(p.GPU.at(i));
			}
			--level;
		} else {
#pragma omp parallel for num_threads(GPU.size())
			for (size_t i = 0; i < GPU.size(); ++i) {
				assert(omp_get_num_threads() == (int)GPU.size());
				GPU.at(i).multElement(p.GPU.at(i));
				GPU.at(i).rescaleMGPU();
			}
			--level;
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).multElement(p.GPU.at(i));
		}
	}
}

template <ALGO algo> void RNSPoly::NTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).NTT<algo>(batch, sync);
	}
}

#define YY(algo) template void RNSPoly::NTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

template <ALGO algo> void RNSPoly::INTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).INTT<algo>(batch, sync);
	}
}

#define YY(algo) template void RNSPoly::INTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

/*
std::array<RNSPoly, 2> RNSPoly::dotKSK(const KeySwitchingKey& ksk) {
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	std::array<RNSPoly, 2> result{RNSPoly(cc, level, true), RNSPoly(cc, level, true)};
	result[0].generateSpecialLimbs(false);
	result[1].generateSpecialLimbs(false);

	if constexpr (PRINT)
		for (auto& i : ksk.b.GPU) {
			for (auto& j : i.DECOMPlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}

			for (auto& j : i.DIGITlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
		}
	for (size_t i = 0; i < GPU.size(); ++i) {
		dotKSKinto(result[0], ksk.b, level);
		dotKSKinto(result[1], ksk.a, level);
	}

	Out(KEYSWITCH, "dotKSK out");
	return result;
}
*/

void RNSPoly::multElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multElement(poly.GPU.at(i));
	}
}

void RNSPoly::multElement(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multElement(poly1.GPU.at(i), poly2.GPU.at(i));
	}
}

void RNSPoly::mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3, const RNSPoly& poly4) {

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).mult1AddMult23Add4(poly1.GPU.at(i), poly2.GPU.at(i), poly3.GPU.at(i), poly4.GPU.at(i));
	}
}

void RNSPoly::mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).mult1Add2(poly1.GPU.at(i), poly2.GPU.at(i));
	}
}

void RNSPoly::dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, const RNSPoly* limbsrc) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		acc.GPU.at(i).dotKSK(GPU.at(i), ksk.GPU.at(i), false, limbsrc ? &limbsrc->GPU.at(i) : nullptr);
	}
}

void RNSPoly::multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "multModupDotKSK Multi-GPU not implemented.");
	assert(c1.level <= c1tilde.level);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(level == c1.level);
		assert(level == c1tilde.level);
		assert(level == c0.level);
		assert(level == c0tilde.level);
	}
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
	for (size_t i = 0; i < GPU.size(); ++i) {
		GPU.at(i).multModupDotKSK(c1.GPU.at(i), c1tilde.GPU.at(i), c0.GPU.at(i), c0tilde.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c0.SetModUp(true);
	c1.SetModUp(true);
}

void RNSPoly::rotateModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "rotateModupDotKSK Multi-GPU not implemented.");
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
	for (size_t i = 0; i < GPU.size(); ++i) {
		GPU.at(i).rotateModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c1.SetModUp(true);
	c0.SetModUp(true);
}

template <ALGO algo> void RNSPoly::moddown(bool ntt, bool free, int aux_num) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kModdown);
	if (!this->isModUp()) {
		std::cout << "RNSPoly calling MOdDown on non-modup polynomial." << std::endl;
	}
	assert(this->isModUp());

	if (cc.GPUid.size() == 1) {
		for (int i = 0; i < (int)GPU.size(); ++i) {
			GPU.at(i).moddown<algo>(cc.getModdownAux(aux_num).GPU.at(i), ntt, free);
		}
	} else {
		RNSPoly& aux = cc.getModdownAux(aux_num);
		bool regular = true;
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			regular = regular && (cc.specialMeta.at(i).size() == cc.splitSpecialMeta.at(i).size());
		}

		std::vector<uint64_t*> bufferSpecial;
		for (uint32_t i = 0; i < GPU.size(); ++i) {
			bufferSpecial.push_back(aux.GPU[i].bufferSPECIAL);
		}

#pragma omp parallel num_threads(GPU.size())
		{
			int i = omp_get_thread_num();

			if (omp_get_num_threads() != (int)GPU.size())
				throw std::invalid_argument("OMP didn't create enough threads");
			assert(omp_get_num_threads() == (int)GPU.size());
			assert(static_cast<size_t>(i) < GPU.size());

			if (!regular) {
				GPU.at(i).moddownMGPU(aux.GPU.at(i), ntt, free, bufferSpecial);
			} else {
				GPU.at(i).moddown(aux.GPU.at(i), ntt, free);
			}
		}
		// assert(nullptr == "ModDown Multi-GPU not implemented.");
	}
	this->SetModUp(false);
}

#define YY(algo) template void RNSPoly::moddown<algo>(bool ntt, bool free, int aux_num);

#include "ntt_types.inc"

#undef YY

int RNSPoly::automorph_index_precomp(const int idx) const {
	return modpow(5, 2 * cc.N - idx, cc.N * 2);
}

void RNSPoly::automorph(const int idx, const int br, RNSPoly* src) {
	int k = automorph_index_precomp(idx);
	// int k2 = modpow(5, idx, cc.N * 2);

	// std::cout << k << " " << k2 << std::endl;
	if (src && src->isModUp() && !this->isModUp()) {
		generateSpecialLimbs(false, false);
	}
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).automorph(k, br, src ? &src->GPU.at(i) : nullptr, src ? src->isModUp() : this->isModUp());
	}
	if (src)
		this->SetModUp(src->isModUp());
}

RNSPoly& RNSPoly::dotKSKInPlace(const KeySwitchingKey& ksk, RNSPoly* limb_src) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKsk);
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	if (cc.GPUid.size() == 1) {
		if (limb_src) {
			std::cerr << "RNSPoly::dotKSKInPlace: limb_src: parameter ignored, fix this" << std::endl;
		}
		// RNSPoly result{RNSPoly(cc, level, true)};
		cc.getKeySwitchAux2().setLevel(level);
		cc.getKeySwitchAux2().generateSpecialLimbs(false, false);
		generateSpecialLimbs(false, false);
		if constexpr (PRINT)
			for (auto& i : ksk.b.GPU) {
				for (auto& j : i.DECOMPlimb) {
					for (auto& k : j) {
						SWITCH(k, printThisLimb(1));
					}
				}

				for (auto& j : i.DIGITlimb) {
					for (auto& k : j) {
						SWITCH(k, printThisLimb(1));
					}
				}
			}
		// dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level);
		// dotKSKInPlace(ksk.a, level);

		this->dotKSKfused(cc.getKeySwitchAux2(), *this, ksk.a, ksk.b, limb_src ? limb_src : this);
	} else {

		RNSPoly& aux = cc.getKeySwitchAux2();
		aux.setLevel(level);
		aux.generateSpecialLimbs(false, false);
		generateSpecialLimbs(false, false);
		this->dotKSKfused(aux, *this, ksk.a, ksk.b, limb_src ? limb_src : this);
	}

	this->SetModUp(true);
	cc.getKeySwitchAux2().SetModUp(true);
	Out(KEYSWITCH, "dotKSK out");
	return cc.getKeySwitchAux2();
}

/*
void RNSPoly::dotKSKInPlace(const RNSPoly& ksk_b, int level) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKsk);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).dotKSK(GPU.at(i), ksk_b.GPU.at(i), level, true);
	}
}
*/

void RNSPoly::setLevel(const int level) {
	assert(level >= -1 && (MODRAISE_WITH_P0 ? level <= cc.L + 1 : level <= cc.L));
	this->level = level;
}

void RNSPoly::modupInto(RNSPoly& poly) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kModup);
	assert(level == poly.level);
	auto& aux = cc.getKeySwitchAux2();
	aux.setLevel(level);

	if (GPU.size() > 1 || true) {
		{
			ConcurrentOpsDiagScope diag_copy(ConcurrentOpsDiag::kModupCopy); // DIAG sub-kind: the copy into the scratch
			poly.copy(*this);
		}
		poly.modup();
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).modupInto(poly.GPU.at(i), aux.GPU.at(i));
		}
	}
}

RNSPoly& RNSPoly::dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, const RNSPoly* limbsrc) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKsk);
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	assert(level == poly.level);
	cc.getKeySwitchAux2().setLevel(level);
	cc.getKeySwitchAux2().generateSpecialLimbs(false, false);
	generateSpecialLimbs(false, false);
	if constexpr (PRINT)
		for (auto& i : ksk.b.GPU) {
			for (auto& j : i.DECOMPlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
			for (auto& j : i.DIGITlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
		}
	// poly.dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level, limbsrc ? limbsrc : this);
	// poly.dotKSKinto(*this, ksk.a, level, limbsrc ? limbsrc : this);

	this->dotKSKfused(cc.getKeySwitchAux2(), poly, ksk.a, ksk.b, limbsrc ? limbsrc : this);

	cc.getKeySwitchAux2().SetModUp(true);
	this->SetModUp(true);
	Out(KEYSWITCH, "dotKSK out");
	return cc.getKeySwitchAux2();
}

void RNSPoly::multScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].multScalar(vector1);
	}
}

void RNSPoly::add(const RNSPoly& a, const RNSPoly& b) {
	assert(level <= a.level);
	assert(level <= b.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).add(a.GPU.at(i), b.GPU.at(i), a.isModUp(), b.isModUp());
	}

	this->SetModUp(a.isModUp() || b.isModUp());
}

void RNSPoly::squareElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).squareElement(poly.GPU.at(i));
	}
}

void RNSPoly::binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).binomialSquareFold(c0_res.GPU.at(i), c2_key_switched_0.GPU.at(i), c2_key_switched_1.GPU.at(i));
	}
}

void RNSPoly::addScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].addScalar(vector1);
	}
}

void RNSPoly::subScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].subScalar(vector1);
	}
}

void RNSPoly::copy(const RNSPoly& poly) {
	// std::cout << "Copy level: " << poly.level << std::endl;
	this->dropToLevel(poly.level);
	this->grow(poly.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).copyLimb(poly.GPU.at(i));
		if (poly.isModUp())
			GPU.at(i).copySpecialLimb(poly.GPU.at(i));
	}
	this->SetModUp(poly.isModUp());
}

void RNSPoly::shapeLike(const RNSPoly& poly) {
	// Same resize path as copy() above, minus the per-partition copyLimb/copySpecialLimb transfers.
	// Special (mod-up) limbs are deliberately not handled: no out-of-place caller needs them, and
	// silently skipping them would be worse than refusing.
	assert(!poly.isModUp());
	this->dropToLevel(poly.level);
	this->grow(poly.level);
	this->SetModUp(false);
}

/** Copy contents without extra checks or resizing */
void RNSPoly::copyShallow(const RNSPoly& poly) {
	this->level = poly.level;
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).copyLimb(poly.GPU.at(i));
	}
}

void RNSPoly::aliasPrefixOf(const RNSPoly& src, const int new_level) {
	if (level != -1)
		throw std::logic_error("FIDESlib: aliasPrefixOf needs a freshly constructed polynomial");
	if (new_level < 0 || new_level > src.level)
		throw std::logic_error("FIDESlib: aliasPrefixOf level " + std::to_string(new_level) + " is outside its source's 0.." + std::to_string(src.level));
	// The partitions size their view from *level, which points here.
	level = new_level;
	for (size_t i = 0; i < GPU.size(); ++i)
		GPU.at(i).aliasLimbsOf(src.GPU.at(i));
	SetModUp(src.isModUp());
}

void RNSPoly::dropToLevel(int level) {
	if (0 && GPU.at(0).bufferLIMB == nullptr) {
		for (auto& g : GPU) {
			cudaSetDevice(g.device);
			int limbSize = g.getLimbSize(level);
			while ((int)g.limb.size() > limbSize) {
				g.dropLimb();
			}
		}
	}
	if (this->level > level)
		this->level = level;
}

void RNSPoly::addMult(const RNSPoly& poly, const RNSPoly& poly1) {
	assert(level <= poly1.level && level <= poly.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).addMult(poly.GPU.at(i), poly1.GPU.at(i));
	}
}

void RNSPoly::load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
	int limbsize  = 0;
	int Slimbsize = 0;
	for (int i = 0; i < (int)data.size(); ++i) {
		if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
			limbsize++;
		} else {
			Slimbsize++;
		}
	}
	// std::cout << "Load " << limbsize << " limbs" << std::endl;

	assert(limbsize - 1 <= cc.L);
	if (level < limbsize - 1)
		grow(limbsize - 1, false);
	if (level > limbsize - 1)
		dropToLevel(limbsize - 1);
	assert(level == limbsize - 1);
	for (int i = 0; i < limbsize; ++i) {
		// std::cout << "Load limb" << i << " into gpu " << cc.limbGPUid[i].x << std::endl;
		assert(moduli[i] == cc.prime.at(i).p);
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
	}

	if ((int)data.size() > limbsize)
		generateSpecialLimbs(false, false);
	for (int i = limbsize; i < (int)data.size(); ++i) {
		for (auto& j : GPU) {
			cudaSetDevice(j.device);
			assert(moduli[i] == cc.specialPrime.at(i - limbsize).p);
			SWITCH(j.SPECIALlimb[i - limbsize], load_convert(data[i]));
		}
	}
	if (Slimbsize == 1)
		this->setLevel(level + 1);
}

void RNSPoly::loadConstant(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
	int limbsize  = 0;
	int Slimbsize = 0;
	for (int i = 0; i < (int)data.size(); ++i) {
		if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
			limbsize++;
		} else {
			Slimbsize++;
		}
	}

	assert(limbsize <= cc.L + 1);
	if (level < limbsize - 1) {
		grow(limbsize - 1, false, true);
	} else {
		dropToLevel(limbsize - 1);
	}
	assert(level == limbsize - 1);
	for (int i = 0; i < limbsize; ++i) {
		assert(moduli[i] == cc.prime.at(i).p);
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
	}

	if ((int)data.size() > limbsize) {
		generatePartialSpecialLimbs();
		this->SetModUp(true);
	}
	for (size_t i = limbsize; i < data.size(); ++i) {
		for (size_t j = 0; j < GPU.size(); ++j) {
			for (size_t k = 0; k < cc.splitSpecialMeta.at(j).size(); ++k) {
				if (cc.specialPrime.at(cc.splitSpecialMeta.at(j).at(k).id - cc.L - 1).p == moduli[i]) {
					cudaSetDevice(GPU[j].device);
					SWITCH(GPU[j].SPECIALlimb[k], load_convert(data[i]));
				}
			}
		}
	}
}

void RNSPoly::loadCoefficients(const std::vector<uint64_t>& biased, const std::vector<uint64_t>& moduli, const uint64_t bigBound) {
	const int limbsize = (int)moduli.size();

	// Preconditions are hard errors, never silent fallbacks: this path produces limbs that nothing
	// downstream re-checks, so a wrong basis or a stale bias constant would surface only as a
	// decryption that is quietly garbage.
	if (limbsize <= 0 || limbsize > cc.L + 1) {
		throw std::invalid_argument("RNSPoly::loadCoefficients: expected 1.." + std::to_string(cc.L + 1) + " Q moduli, got " + std::to_string(limbsize));
	}
	if (biased.size() != (size_t)cc.N) {
		throw std::invalid_argument("RNSPoly::loadCoefficients: expected exactly N=" + std::to_string(cc.N) + " coefficients, got " + std::to_string(biased.size()) +
		  " (the caller must expand a sparse encoding to full length before uploading)");
	}
	if (bigBound == 0) {
		throw std::invalid_argument("RNSPoly::loadCoefficients: bigBound must be the encoder's bias constant, not 0");
	}
	for (int i = 0; i < limbsize; ++i) {
		if (moduli[i] != cc.prime.at(i).p) {
			throw std::invalid_argument("RNSPoly::loadCoefficients: modulus " + std::to_string(i) + " is not this context's Q prime at that tower (device encode is Q-basis only; " +
			  "an extended Q||P plaintext must go through loadConstant)");
		}
	}

	// A FRESH polynomial only, and grown NON-constant. Both halves of that matter:
	//
	// loadConstant grows with constant = true, which routes through
	// LimbPartition::generateLimbConstant and deliberately passes no auxptr — a plaintext arrives
	// already in EVALUATION form, so nothing in this library has ever needed to transform one, and
	// the NTT scratch (Limb::aux, N words per limb) is simply not allocated. This path DOES
	// transform, so it must have that scratch: ApplyNTT writes stage one into auxptr and reads
	// stage two back out of it, and on a constant partition auxptr is empty.
	//
	// The scratch is TRANSIENT, not part of the finished plaintext: it is released at the end of
	// this function, once the NTT is fenced, so what the caller ends up holding is v alone —
	// byte for byte a loadConstant plaintext's footprint. (It used to be held for the plaintext's
	// whole lifetime, doubling the device memory of every device-encoded diagonal, because
	// VectorGPU::free was not idempotent: it left `managed` set, so an early release meant Limb's
	// destructor freed a second time — an assert in a debug build and a double free in a release
	// one. VectorGPU::free now returns early when the vector is already freed, which makes "release
	// early, let the destructor mop up" the intended pattern rather than a hazard.)
	//
	// Requiring level == -1 is what makes the scratch precondition structural rather than a
	// comment: it is the state a just-constructed Plaintext is in, and it forecloses being handed
	// a polynomial that some other path already grew constant.
	if (level != -1) {
		throw std::invalid_argument("RNSPoly::loadCoefficients: expected a freshly constructed polynomial (level -1), got level " + std::to_string(level) +
		  "; device encode allocates its own NTT scratch and cannot adopt a constant-grown polynomial");
	}
	grow(limbsize - 1, false, false);
	assert(level == limbsize - 1);

	const uint64_t bigValueHf = bigBound >> 1;

	// ONE upload per device: every limb on that device reduces the same N coefficients. This is the
	// entire point of the arm — loadConstant moves L*N words, this moves N.
	std::vector<std::unique_ptr<VectorGPU<uint64_t>>> staging(GPU.size());
	for (int i = 0; i < limbsize; ++i) {
		const int g = cc.limbGPUid[i].x;
		if (!staging[g]) {
			cudaSetDevice(GPU[g].device);
			staging[g] = std::make_unique<VectorGPU<uint64_t>>(GPU[g].s, cc.N, GPU[g].device, biased.data());
		}
	}

	for (int i = 0; i < limbsize; ++i) {
		const int g		 = cc.limbGPUid[i].x;
		auto& l			 = GPU[g].limb[cc.limbGPUid[i].y];
		cudaSetDevice(GPU[g].device);
		// The upload was enqueued on the partition stream; the limb's own stream must not run ahead of it.
		STREAM(l).wait(GPU[g].s);
		SWITCH(l, loadCoefficients(staging[g]->data, bigValueHf, bigBound - moduli[i]));
	}

	// NTT(sync=true) fences only STREAM(limb[i]) for i stepping by `batch`, and then launches the
	// whole batch on that one stream — so the per-limb kernels above must be visible to it through
	// the partition stream, not merely enqueued on their own limb streams.
	for (int i = 0; i < limbsize; ++i) {
		const int g = cc.limbGPUid[i].x;
		cudaSetDevice(GPU[g].device);
		GPU[g].s.wait(STREAM(GPU[g].limb[cc.limbGPUid[i].y]));
	}

	// COEFFICIENT -> EVALUATION, leaving exactly what OpenFHE's Encode (which ends in
	// SetFormat(Format::EVALUATION)) would have handed loadConstant.
	this->NTT<ALGO_SHOUP>(cc.batch, true);

	// Safe here and only here: NTT(sync=true) closed with s.wait(limb streams), so the pool's
	// stream-ordered free cannot be reordered ahead of the kernels that read the staging buffer.
	for (size_t g = 0; g < GPU.size(); ++g) {
		if (staging[g]) {
			cudaSetDevice(GPU[g].device);
			staging[g]->free(GPU[g].s);
		}
	}

	// And the NTT scratch, on exactly the same fence and for exactly the same reason: the transform
	// that needed it has completed, and this polynomial is a PLAINTEXT — an operand, never a
	// destination, and never transformed again by anything in this library. That is not an
	// assumption this path introduces: a loadConstant plaintext is grown with no scratch at all and
	// with its scratch pointer table left unwritten, so the invariant is already load-bearing, and
	// more strictly than here. Releasing makes the two encoders produce identically shaped
	// plaintexts, which is the point — device residency can then be sized against one figure.
	freeNTTScratch();
}

void RNSPoly::freeNTTScratch() {
	for (size_t g = 0; g < GPU.size(); ++g) {
		cudaSetDevice(GPU[g].device);
		GPU[g].freeNTTScratch();
	}
}

void RNSPoly::broadcastLimb0() {
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			GPU.at(i).broadcastLimb0();
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).broadcastLimb0_mgpu();
		}
	}
}

void RNSPoly::evalLinearWSum(uint32_t n, std::vector<const RNSPoly*>& vec, std::vector<uint64_t>& elem) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> ps(n);
		for (int j = 0; j < (int)n; ++j) {
			ps[j] = &vec[j]->GPU.at(i);
		}
		GPU.at(i).evalLinearWSum(n, ps, elem);
	}
}

void RNSPoly::squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "squareModupDotKSK Multi-GPU not implemented.");
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).squareModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c0.SetModUp(true);
	c1.SetModUp(true);
}

void RNSPoly::generatePartialSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generatePartialSpecialLimb();
	}
}

void RNSPoly::dotKSKfused(RNSPoly& out2, const RNSPoly& digitSrc, const RNSPoly& ksk_a, const RNSPoly& ksk_b, const RNSPoly* source) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKsk);
	RNSPoly& out1      = *this;
	const RNSPoly& src = source ? *source : *this;
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			out1.GPU[i].dotKSKfusedMGPU(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			out1.GPU[i].dotKSKfusedMGPU(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
		}
	}
}

void RNSPoly::dotProductPt(RNSPoly& c1_,
                           const std::vector<const RNSPoly*>& c0s_,
                           const std::vector<const RNSPoly*>& c1s_,
                           const std::vector<const RNSPoly*>& pts_,
                           const bool ext) {

	if (ext) {
		generateSpecialLimbs(false, false);
		c1_.generateSpecialLimbs(false, false);
	}
	int n = pts_.size();
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);
		for (int i = 0; i < n; ++i) {
			c0s[i] = &(c0s_[i]->GPU[j]);
			c1s[i] = &(c1s_[i]->GPU[j]);
			pts[i] = &(pts_[i]->GPU[j]);
		}
		GPU[j].dotProductPt(c1_.GPU[j], c0s, c1s, pts, ext);
	}
	c1_.SetModUp(ext);
	this->SetModUp(ext);
}

RNSPoly& RNSPoly::dotProduct(RNSPoly& c1,
                             const RNSPoly& kskb,
                             const RNSPoly& kska,
                             const std::vector<const RNSPoly*>& c0in,
                             const std::vector<const RNSPoly*>& c1in,
                             const std::vector<const RNSPoly*>& d0in,
                             const std::vector<const RNSPoly*>& d1in,
                             bool ext_in,
                             bool ext_out) {

	auto& c2 = cc.getKeySwitchAux();

	if (ext_in) {
		generateSpecialLimbs(false, false);
		c1.generateSpecialLimbs(false, false);
	}
	c2.setLevel(c1.getLevel());

	int n = c0in.size();
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), d0s(n, nullptr), d1s(n, nullptr);
		for (int i = 0; i < n; ++i) {
			c0s[i] = &(c0in[i]->GPU[j]);
			c1s[i] = &(c1in[i]->GPU[j]);
			d0s[i] = &(d0in[i]->GPU[j]);
			d1s[i] = &(d1in[i]->GPU[j]);
		}
		GPU[j].binomialDotProduct(c1.GPU[j], c2.GPU[j], c0s, c1s, d0s, d1s, ext_in);
	}

	SetModUp(ext_in);
	c1.SetModUp(ext_in);
	c2.SetModUp(ext_in);

	return c2;
}

void RNSPoly::hoistedRotationFused(std::vector<int> indexes,
                                   std::vector<RNSPoly*>& c0,
                                   std::vector<RNSPoly*>& c1,
                                   const std::vector<RNSPoly*>& ksk_a,
                                   const std::vector<RNSPoly*>& ksk_b,
                                   const RNSPoly& src_c0,
                                   const RNSPoly& src_c1) {
	uint32_t n = indexes.size();
	for (uint32_t j = 0; j < n; ++j) {
		c0[j]->generateSpecialLimbs(false, false);
		c1[j]->generateSpecialLimbs(false, false);
		indexes[j] = indexes[j] == 2 * cc.N - 1 ? 2 * cc.N - 1 : automorph_index_precomp(indexes[j]);
	}
	//    assert(src_c0.isModUp() == false);

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), ksk_as(n, nullptr), ksk_bs(n, nullptr);
		for (uint32_t i = 0; i < n; ++i) {
			c0s[i]    = &(c0[i]->GPU[j]);
			c1s[i]    = &(c1[i]->GPU[j]);
			ksk_as[i] = &(ksk_a[i]->GPU[j]);
			ksk_bs[i] = &(ksk_b[i]->GPU[j]);
		}
		GPU[j].fusedHoistRotate(n, indexes, c0s, c1s, ksk_as, ksk_bs, src_c0.GPU[j], src_c1.GPU[j], src_c0.isModUp());
	}

	for (uint32_t j = 0; j < n; ++j) {
		c0[j]->SetModUp(true);
		c1[j]->SetModUp(true);
	}
}

/*
void RNSPoly::generateGatherLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[j].generateGatherLimb(false);
	}
}
*/

RNSPoly& RNSPoly::modup_ksk_moddown_mgpu(const KeySwitchingKey& key, const bool moddown) {
	// DIAGNOSTIC SCOPE (FIDESLIB_CONCURRENT_OPS_DIAG=ksk): this fused key switch is what every
	// rotation, relinearisation and conjugation in a bootstrap actually runs, and until 2026-09-21 no
	// diag scope covered it -- the sub-step sweeps (lt, modraise, evalmod, keyswitch, ...) serialized
	// their callers but never this body, so "only the whole-boot lock closes it" could not tell a
	// defect in here from one between sub-steps. One cached-bool read when the diag is unset.
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kKsk);
	RNSPoly& aux = cc.getKeySwitchAux2();
	aux.setLevel(level);
	RNSPoly& aux_limbs1 = cc.getModdownAux(0);
	RNSPoly& aux_limbs2 = cc.getModdownAux(1);

	static std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>> signals;
	/*
		if (signals.size() < 2 * (cc.dnum + 2) || (signals.size() > 0 && signals[0].size() < cc.GPUid.size())) {
			signals.resize(2 * (cc.dnum + 2));
			for (int k = 0; k < cc.GPUid.size(); ++k) {
				cudaSetDevice(cc.GPUid[k]);
				cudaDeviceSynchronize();
			}
			for (auto& i : signals) {
				i.resize(std::max(i.size(), cc.GPUid.size()));
				parallel_for(0, cc.GPUid.size(), 1, [&](int j) {
					// for (int j = 0; j < cc.GPUid.size(); ++j) {
					i[j].resize(std::max(i.size(), cc.GPUid.size()));
					// cudaSetDevice(cc.GPUid[j]);
					// cudaDeviceSynchronize();
					for (int k = 0; k < cc.GPUid.size(); ++k) {
						if (i[j][k].second != nullptr) {

							cudaFreeHost(i[j][k].second);
							i[j][k].second = nullptr;
							CudaCheckErrorModNoSync;
							// cudaFree(i[j][k].second);
						}
						CudaCheckErrorModNoSync;
						cudaHostAlloc(&i[j][k].second, sizeof(TimelineSemaphore), cudaHostAllocPortable);
						CudaCheckErrorModNoSync;
						// cudaMalloc(&i[j][k].second, sizeof(TimelineSemaphore));
						i[j][k].second->value = 0;
						// cudaMemset(i[j][k].second, 0, 128);
						CudaCheckErrorModNoSync;
						i[j][k].first = 1;
					}
				});
			}
		}
	*/
	std::vector<std::atomic_uint64_t> thread_stop_buffer(cc.GPUid.size() * 8);
	std::vector<std::atomic_uint64_t*> thread_stop(cc.GPUid.size(), nullptr);
	for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
		thread_stop[k]            = &thread_stop_buffer[8 * k];
		thread_stop_buffer[8 * k] = 0;
	}

	if (1 || moddown) {

		std::vector<uint64_t*> bufferGather;
		std::vector<uint64_t*> bufferSpecial_c0;
		std::vector<uint64_t*> bufferSpecial_c1;
		std::vector<Stream*> external_s;
		std::vector<Stream*> external_s0;
		for (uint32_t i = 0; i < GPU.size(); ++i) {
			bufferGather.push_back(GPU[i].bufferGATHER);
			bufferSpecial_c0.push_back(aux_limbs2.GPU[i].bufferSPECIAL);
			bufferSpecial_c1.push_back(aux_limbs1.GPU[i].bufferSPECIAL);
			external_s.push_back(&GPU[i].s);
			external_s0.push_back(&aux.GPU[i].s);
		}

		if (!MEMCPY_PEER || !GRAPH_CAPTURE) {
#pragma omp parallel num_threads(GPU.size())
			{
				int j = omp_get_thread_num();

				if (omp_get_num_threads() != (int)GPU.size())
					throw std::invalid_argument("OMP didn't create enough threads");
				assert(omp_get_num_threads() == (int)GPU.size());
				assert(static_cast<size_t>(j) < GPU.size());
				GPU[j].modup_ksk_moddown_mgpu(
					aux.GPU[j],
					key.a.GPU[j],
					key.b.GPU[j],
					aux_limbs1.GPU[j],
					aux_limbs2.GPU[j],
					moddown,
					bufferGather,
					bufferSpecial_c0,
					bufferSpecial_c1,
					external_s,
					signals,
					thread_stop,
					external_s0);
			}
		} else {

			// #pragma omp parallel num_threads(GPU.size())
			//{
			parallel_for(0,
			             cc.GPUid.size(),
			             1,
			             [&](int j) {
				             // for (int j = 0; j < cc.GPUid.size(); ++j) {
				             // int j = omp_get_thread_num();
				             // if (omp_get_num_threads() != (int)GPU.size())
				             //     throw std::invalid_argument("OMP didn't create enough threads");
				             // assert(omp_get_num_threads() == (int)GPU.size());
				             // assert(j < GPU.size());
				             GPU[j].modup_ksk_moddown_mgpu(
					             aux.GPU[j],
					             key.a.GPU[j],
					             key.b.GPU[j],
					             aux_limbs1.GPU[j],
					             aux_limbs2.GPU[j],
					             moddown,
					             bufferGather,
					             bufferSpecial_c0,
					             bufferSpecial_c1,
					             external_s,
					             signals,
					             thread_stop,
					             external_s0);
			             });
		}

		if (MEMCPY_PEER) {
			for (auto& i : signals) {
				for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
					cudaSetDevice(cc.GPUid[j]);

					for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
						i[j][k].first = i[j][k].first + 4;
					}
				}
			}
		}
		this->SetModUp(!moddown);
		aux.SetModUp(!moddown);
		return aux;
	} else {
		this->modup();
		RNSPoly& result = this->dotKSKInPlace(key, nullptr);
		if (moddown) {
			result.moddown(true, false);
			this->moddown(true, false);
		}
		return result;
	}
}

} // namespace FIDESlib::CKKS