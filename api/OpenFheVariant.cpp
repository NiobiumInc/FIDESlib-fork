#include "OpenFheVariant.hpp"

#include "config_core.h" // WITH_REDUCED_NOISE

namespace fideslib {

bool LinkedOpenFheReducedNoise() {
#ifdef WITH_REDUCED_NOISE
	return true;
#else
	return false;
#endif
}

} // namespace fideslib
