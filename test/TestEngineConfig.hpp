//
// Shared test-only helper for configuring a CCParams's FBC variant and backend: every oracle
// comparison must run the same variant as the linked OpenFHE, and GenCryptoContext refuses an
// unset reducedNoise.
//
#ifndef FIDESLIB_TEST_TESTENGINECONFIG_HPP
#define FIDESLIB_TEST_TESTENGINECONFIG_HPP

#include "CCParams.hpp"
#include "OpenFheVariant.hpp" // fideslib::LinkedOpenFheReducedNoise
#include "engine/Backend.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace FIDESlib::Testing {

// The api test fixtures run against any backend: set FIDESLIB_TEST_BACKEND=cuda (on a CUDA
// build) or =haze (on a haze build) to exercise that engine end-to-end. Unset means cpu, the
// documented default; any other value, or a backend this build lacks, fails hard rather than
// silently falling back to cpu.
enum class TestBackend { CPU, CUDA, HAZE };

inline TestBackend GetTestBackend() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	if (b == nullptr) {
		return TestBackend::CPU;
	}
	const std::string name(b);
	if (name == "cpu") {
		return TestBackend::CPU;
	}
	if (name == "cuda") {
		if (!fideslib::IsBackendAvailable(fideslib::Backend::CUDA)) {
			throw std::runtime_error("FIDESLIB_TEST_BACKEND=cuda: this build has no CUDA backend");
		}
		return TestBackend::CUDA;
	}
	if (name == "haze") {
		if (!fideslib::IsBackendAvailable(fideslib::Backend::HAZE)) {
			throw std::runtime_error("FIDESLIB_TEST_BACKEND=haze: this build has no haze backend");
		}
		return TestBackend::HAZE;
	}
	throw std::runtime_error("FIDESLIB_TEST_BACKEND: unknown backend '" + name + "' (expected cpu, cuda or haze; unset means cpu)");
}

// Sets the FBC variant to the one this build's OpenFHE was compiled with, then picks the backend
// GetTestBackend() resolves, so a call site is one line instead of a flag-plus-branch block.
inline void ConfigureTestEngine(fideslib::CCParams<fideslib::CryptoContextCKKSRNS>& params) {
	params.SetReducedNoise(fideslib::LinkedOpenFheReducedNoise());
	switch (GetTestBackend()) {
	case TestBackend::CUDA: params.SetBackend(fideslib::Backend::CUDA); break;
	case TestBackend::HAZE: params.SetBackend(fideslib::Backend::HAZE); break;
	case TestBackend::CPU: break;
	}
}

} // namespace FIDESlib::Testing

#endif // FIDESLIB_TEST_TESTENGINECONFIG_HPP
