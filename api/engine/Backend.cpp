#include "engine/Backend.hpp"

#include "engine/Engine.hpp"
#include "engine/cpu/OpenFheEngine.hpp"
#ifdef FIDESLIB_ENABLE_CUDA
#include "engine/cuda/CudaEngine.hpp"
#endif
#ifdef FIDESLIB_ENABLE_HAZE
#include "engine/haze/HazeEngine.hpp"
#endif

#include <stdexcept>

namespace fideslib {

bool IsBackendAvailable(Backend backend) {
	switch (backend) {
	case Backend::CPU: return true;
	case Backend::CUDA:
#ifdef FIDESLIB_ENABLE_CUDA
		return true;
#else
		return false;
#endif
	case Backend::HAZE:
#ifdef FIDESLIB_ENABLE_HAZE
		return true;
#else
		return false;
#endif
	}
	return false;
}

std::unique_ptr<Engine> MakeEngine(Backend backend, bool reducedNoise) {
	switch (backend) {
	case Backend::CPU: return std::make_unique<OpenFheEngine>();
	case Backend::CUDA:
#ifdef FIDESLIB_ENABLE_CUDA
		return std::make_unique<CudaEngine>();
#else
		throw std::runtime_error("CUDA backend not compiled in; rebuild with FIDESLIB_ENABLE_CUDA=ON (i.e. with CUDA).");
#endif
	case Backend::HAZE:
#ifdef FIDESLIB_ENABLE_HAZE
		return std::make_unique<HazeEngine>(reducedNoise);
#else
		(void)reducedNoise;
		throw std::runtime_error("haze (FHETCH) backend not compiled in; rebuild with FIDESLIB_ENABLE_HAZE=ON.");
#endif
	}
	throw std::runtime_error("unknown backend");
}

} // namespace fideslib
