#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeEngine.hpp"

// haze C API — the presence of this include (and the hazeGetDeviceCount call below)
// proves the haze link is exercised at compile/link time.
#include <haze/haze.h>

#include <stdexcept>
#include <vector>

namespace fideslib {

namespace {

/// @brief Probe hazeGetDeviceCount and return the count; used by HazeEngine::devices().
/// Isolated in an anonymous-namespace helper so the haze link is exercised even in the
/// Stub (no call to hazeGetDeviceCount in header-inlined code).
int probeHazeDeviceCount() {
	int count = 0;
	hazeError_t err = hazeGetDeviceCount(&count);
	if (err != HAZE_SUCCESS) {
		return 0;
	}
	return count;
}

} // namespace

std::vector<int> HazeEngine::devices() const {
	// haze exposes exactly one logical device (the FHETCH simulator/hardware).
	// Probe the count so the hazeGetDeviceCount symbol is linked; always return {0}.
	(void)probeHazeDeviceCount();
	return { 0 };
}

void HazeEngine::teardown() {
	// Stub: no haze state to tear down yet.
	// The full implementation will: clear outputs_/keys, call
	// hazeReplayBridgeReset(), hazeDeviceReset(), set loaded_ = false,
	// and decrement the liveEngines_ guard.
}

HazeEngine::~HazeEngine() {
	teardown();
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
