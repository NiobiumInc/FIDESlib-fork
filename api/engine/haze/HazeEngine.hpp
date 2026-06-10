#ifndef API_HAZEENGINE_HPP
#define API_HAZEENGINE_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/Engine.hpp"

#include <vector>

namespace fideslib {

/// @brief haze (FHETCH) record/replay backend targeting the Niobium accelerator.
/// Records CKKS operations as FHETCH IR via the haze C API; executes the recorded program
/// lazily on the first readback (hazeFlush). One program per context; explicit output
/// declaration via MarkOutput before the first Decrypt; no compute after readback.
///
/// Stub: only name(), backend(), teardown(), and devices() are implemented.
/// All compute methods inherit the throwing defaults from Engine.
class HazeEngine final : public Engine {
  public:
	const char* name() const override {
		return "haze (FHETCH)";
	}

	Backend backend() const override {
		return Backend::HAZE;
	}

	/// @brief Return the single logical haze device index {0}.
	std::vector<int> devices() const override;

	/// @brief Tear down haze state (no-op in the stub; full implementation clears
	/// outputs/keys, calls hazeReplayBridgeReset() + hazeDeviceReset(), and decrements
	/// the live-engine guard).
	void teardown() override;

	~HazeEngine() override;
};

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
#endif // API_HAZEENGINE_HPP
