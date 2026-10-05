//
// Created by carlosad on 12/06/25.
//

#include "CKKS/AccumulateBroadcast.cuh"
#include "CudaUtils.cuh"

#include "CKKS/Context.cuh"

#include <bit>
#include <stdexcept>
#include <string>

namespace {
// Accumulate/Broadcast advance the rotation stride by `logbStep = bit_width(bStep) - 1` and
// generate `bStep - 1` hoisted rotations per round, so they are only correct when bStep is a
// power of two >= 2: bStep == 1 gives logbStep == 0 and the round loop never terminates, and a
// non-power-of-two bStep sums some shifts twice. Reject those instead of hanging or returning
// wrong results.
void checkBStep(const int bStep, const char* where) {
	if (bStep < 2 || (bStep & (bStep - 1)) != 0) {
		throw std::invalid_argument(std::string(where) + ": bStep must be a power of two >= 2, got " + std::to_string(bStep));
	}
}
} // namespace

/// THE DIAGNOSTIC LOCK MUST NOT SPAN `rotate_hoisted`. `FIDESLIB_CONCURRENT_OPS_DIAG=accum`
/// serializes Accumulate against the other lanes by holding the recursive diagnostic lock; held
/// across the whole body it also spans `Ciphertext::rotate_hoisted`, which reaches
/// `RNSPoly::modup_ksk_moddown_mgpu` and enters a `#pragma omp parallel` region unconditionally --
/// even at one GPU, where `num_threads(1)` still opens a team (RNSPoly.cpp, the omp region around
/// `GPU[j].modup_ksk_moddown_mgpu`). A std::recursive_mutex is owned by the thread that locked it,
/// and unlocking it from a thread of that team is undefined -- the `accum` sweep crashed there
/// instead of running.
///
/// So the lock is taken in blocks AROUND the hoisted rotation, never across it. What `accum` still
/// serializes: the auxiliary ciphertext construction, the index bookkeeping, the adds, the extend
/// and the mod-down. What it no longer serializes: the hoisted rotation itself -- `kKsk`, `kLt`
/// and `kBoot` are the scopes that cover that, and they are the ones a sweep should read for it.
/// Read an `accum`-clean run as "everything around the rotation", not as "Accumulate as a whole".

void AccumulateCascadeImpl(FIDESlib::CKKS::Ciphertext& ctxt, const int bStep, const int stride, const int size, const int startFactor) {
	checkBStep(bStep, "Accumulate");
	if (startFactor <= 0 || size <= 0) {
		return;
	}

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	std::vector<FIDESlib::CKKS::Ciphertext> aux;
	{
		FIDESlib::ConcurrentOpsDiagScope diag(FIDESlib::ConcurrentOpsDiag::kAccum);
		for (int i = 0; i < bStep - 1; ++i) {
			aux.emplace_back(cc_);
		}
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = startFactor; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<FIDESlib::CKKS::Ciphertext*> auxptr;
		for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
			indexes.push_back(idx);
			auxptr.emplace_back(&aux[idx / stride / s - 1]);
		}

		if (indexes.empty()) {
			continue;
		}

		// Outside the diagnostic lock on purpose -- see the note at the top of this file.
		ctxt.rotate_hoisted(indexes, auxptr, true);
		{
			FIDESlib::ConcurrentOpsDiagScope diag(FIDESlib::ConcurrentOpsDiag::kAccum);
			ctxt.extend();
			for (size_t i = 0; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}
			ctxt.modDown(false);
		}
	}

	if (size * stride == ctxt.slots)
		ctxt.slots = stride * startFactor;
} // namespace

std::vector<int> FIDESlib::CKKS::GetAccumulateRotationIndices(const int bStep, const int stride, const int size) {
	checkBStep(bStep, "GetAccumulateRotationIndices");
	std::vector<int> indices;
	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = stride; s < stride * size; s <<= logbStep) {
		for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
			indices.push_back(idx);
		}
	}
	return indices;
}

std::vector<int> FIDESlib::CKKS::GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize) {
	checkBStep(bStep, "GetbroadcastRotationIndices");
	const int size	 = outsize / initsize;
	const int stride = initsize;
	std::vector<int> indices;

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = stride; s < stride * size; s <<= logbStep) {
		if (stride * s * bStep >= stride * size) {
			for (int i = 1; i <= bStep; ++i) {
				int idx = -(i * s) + 1;
				if (-idx < outsize) {
					indices.push_back(idx);
				}
			}
		} else {
			for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
				indices.push_back(idx);
			}
		}
	}
	return indices;
}

void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size) {
	checkBStep(bStep, "Accumulate");
	Context& cc_ = ctxt.cc_;
	std::vector<Ciphertext> aux;

	{
		ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kAccum);
		for (int i = 0; i < bStep - 1; ++i) {
			aux.emplace_back(cc_);
		}
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = 1; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<Ciphertext*> auxptr;
		for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
			indexes.push_back(idx);
			auxptr.emplace_back(&aux[idx / stride / s - 1]);
		}
		// Outside the diagnostic lock on purpose -- see the note at the top of this file.
		ctxt.rotate_hoisted(indexes, auxptr, true);
		{
			ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kAccum);
			for (size_t i = 0; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}
			ctxt.c1.moddown();
		}
	}
	{
		ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kAccum);
		if (ctxt.c0.isModUp())
			ctxt.c0.moddown();
	}

	if (size * stride == ctxt.slots)
		ctxt.slots = stride;
}

void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size, const int startFactor) {
	// The kAccum scopes live INSIDE AccumulateCascadeImpl, around the hoisted rotation rather than
	// across it -- see the note at the top of this file.
	AccumulateCascadeImpl(ctxt, bStep, stride, size, startFactor);
}

void FIDESlib::CKKS::Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize) {
	checkBStep(bStep, "Broadcast");
	const int size	 = outsize / initsize;
	const int stride = initsize;
	Context& cc_	 = ctxt.cc_;
	// ContextData& cc	 = ctxt.cc;
	std::vector<Ciphertext> aux;

	for (int i = 0; i < bStep; ++i) {
		aux.emplace_back(cc_);
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = 1; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<Ciphertext*> auxptr;
		if (s * bStep >= size) {
			for (int i = 1; i <= bStep; ++i) {
				int idx = -(i * stride * s) + 1;
				if (-idx < outsize) {
					//  std::cout << idx << std::endl;
					indexes.push_back(idx);
					auxptr.emplace_back(&aux[i - 1]);
				}
			}
			ctxt.rotate_hoisted(indexes, auxptr, true);
			ctxt.add(*auxptr[0], *auxptr[1]);
			for (size_t i = 2; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}

		} else {
			for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
				//  std::cout << idx << std::endl;
				indexes.push_back(idx);
				auxptr.emplace_back(&aux[idx / stride / s - 1]);
			}
			ctxt.rotate_hoisted(indexes, auxptr, true);
			ctxt.extend();
			for (size_t i = 0; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}
		}
		ctxt.c1.moddown(false);
	}
	if (ctxt.c0.isModUp())
		ctxt.c0.moddown();
	
	if (outsize == ctxt.slots)
		ctxt.slots = initsize;
}
