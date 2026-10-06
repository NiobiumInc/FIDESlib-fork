#ifndef API_OPENFHEVARIANT_HPP
#define API_OPENFHEVARIANT_HPP

namespace fideslib {

/// @brief True if this library's own linked OpenFHE was built with WITH_REDUCED_NOISE. A real
/// function, not a header constexpr, so every caller gets the library's one compiled-in value
/// instead of whatever config_core.h its own translation unit happens to see.
bool LinkedOpenFheReducedNoise();

} // namespace fideslib

#endif // API_OPENFHEVARIANT_HPP
