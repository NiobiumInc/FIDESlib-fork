#ifndef API_BACKENDENV_HPP
#define API_BACKENDENV_HPP

#include "engine/Backend.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fideslib {

/// @brief Backend selection for the examples: FIDESLIB_BACKEND ∈ {cpu, cuda, haze} selects
/// the backend at run time; when unset, the default is CUDA when this build has it,
/// otherwise CPU. Failures are loud: an unrecognized name throws here, and a recognized
/// name that this build lacks is returned as-is so SetBackend rejects it with the
/// backend-not-available error (never a silent fallback).
inline Backend BackendFromEnv() {
	const char* env = std::getenv("FIDESLIB_BACKEND");
	if (env == nullptr || env[0] == '\0') {
		return IsBackendAvailable(Backend::CUDA) ? Backend::CUDA : Backend::CPU;
	}
	const std::string name(env);
	if (name == "cpu") {
		return Backend::CPU;
	}
	if (name == "cuda") {
		return Backend::CUDA;
	}
	if (name == "haze") {
		return Backend::HAZE;
	}
	throw std::runtime_error("FIDESLIB_BACKEND: unknown backend '" + name + "' (expected cpu, cuda, or haze)");
}

} // namespace fideslib

#endif // API_BACKENDENV_HPP
