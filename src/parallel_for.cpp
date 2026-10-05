//
// Created by carlosad on 27/03/25.
//
#include "parallel_for.hpp"
#include <cassert>
#include <omp.h>

void FIDESlib::parallel_for(int init, int end, int increment, const std::function<void(int)>& f) {
	const int n = increment > 0 ? (end - init + increment - 1) / increment : 0;
	if (n <= 0)
		return;
	// One index per thread when OpenMP grants the team asked for; a strided loop over the rest
	// when it grants fewer (OMP_THREAD_LIMIT, a nested region with nesting disabled, a runtime
	// that caps teams). The old body ran f() exactly once per thread and asserted the team size,
	// so under any of those it silently skipped every index past the team -- SetupConstants
	// (ConstantsGPU.cu) hands it one index per prime, and a one-thread team left every prime but
	// the first without its tables.
#pragma omp parallel num_threads(n)
	{
		const int nt = omp_get_num_threads();
		for (int k = omp_get_thread_num(); k < n; k += nt)
			f(init + increment * k);
	}
}

void FIDESlib::openmp_synchronize() {
#pragma omp barrier
}
