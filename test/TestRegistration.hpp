//
// Registers a test only where it applies. One test binary serves every backend and the backend is chosen when
// it runs (FIDESLIB_TEST_BACKEND), so a test written for one backend would otherwise be present on all of them
// and have to skip or be left out by a list. These macros register it only when the condition holds, so on any
// other backend it does not exist.
//
//   FIDES_TEST_IF(OnDevice(), Suite, Name) { ... }          like TEST(Suite, Name)
//   FIDES_TEST_F_IF(OnHaze(), Fixture, Name) { ... }        like TEST_F(Fixture, Name)
//   INSTANTIATE_TEST_SUITE_P(Prefix, Suite, OnlyIf<Suite::ParamType>(OnDevice(), generator), name)
//                                                           a TEST_P suite with no cases elsewhere
//
// The condition is evaluated while the binary starts, before main, from the environment alone (no library
// state), so it must be one of the functions below or a combination of them.
//
#ifndef FIDESLIB_TEST_TESTREGISTRATION_HPP
#define FIDESLIB_TEST_TESTREGISTRATION_HPP

#include "ConcurrentOps.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace FIDESlib::Testing {

namespace detail {
inline const char* BackendName() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	return b != nullptr ? b : "cpu";
}
inline bool Is(const char* name) {
	return std::strcmp(BackendName(), name) == 0;
}
// A name that is none of the three is an error that GetTestBackend() (TestEngineConfig.hpp) reports when a
// test builds a context. Every test registers meanwhile, so that the error shows and the run cannot pass
// with nothing in it.
inline bool Unknown() {
	return !Is("cpu") && !Is("cuda") && !Is("haze");
}
} // namespace detail

inline bool OnCpu() {
	return detail::Is("cpu") || detail::Unknown();
}
inline bool OnCuda() {
	return detail::Is("cuda") || detail::Unknown();
}
inline bool OnHaze() {
	return detail::Is("haze") || detail::Unknown();
}
/// A backend that computes on a device (CUDA, haze): what the comparison with OpenFHE is about.
inline bool OnDevice() {
	return OnCuda() || OnHaze();
}
/// A backend that computes on the host (CPU, haze's host-side paths): the one a device-encode fallback is about.
inline bool OnHostBackend() {
	return detail::Is("cpu") || detail::Is("haze") || detail::Unknown();
}
/// The test needs the concurrent-operations mode of the CUDA backend (FIDESLIB_CONCURRENT_OPS=1).
inline bool InConcurrentOpsMode() {
	return fideslib::ConcurrentOpsEnabled();
}

/// The generator of a TEST_P suite where `condition` holds, no cases where it does not.
template <typename T, typename Generator>
::testing::internal::ParamGenerator<T> OnlyIf(bool condition, Generator generator) {
	if (condition)
		return generator;
	return ::testing::ValuesIn(std::vector<T>{});
}

} // namespace FIDESlib::Testing

#define FIDES_TEST_REGISTERED_(CONDITION, Suite, Name, FixtureType)                                                    \
	class Suite##_##Name##_FidesTest : public FixtureType {                                                            \
	  public:                                                                                                          \
		void TestBody() override;                                                                                      \
	};                                                                                                                 \
	[[maybe_unused]] static const bool Suite##_##Name##_fides_registered = [] {                                        \
		if (!(CONDITION))                                                                                              \
			return false;                                                                                              \
		::testing::RegisterTest(#Suite, #Name, nullptr, nullptr, __FILE__, __LINE__,                                   \
		  []() -> FixtureType* { return new Suite##_##Name##_FidesTest; });                                            \
		return true;                                                                                                   \
	}();                                                                                                               \
	void Suite##_##Name##_FidesTest::TestBody()

#define FIDES_TEST_IF(CONDITION, Suite, Name) FIDES_TEST_REGISTERED_(CONDITION, Suite, Name, ::testing::Test)

#define FIDES_TEST_F_IF(CONDITION, Fixture, Name) FIDES_TEST_REGISTERED_(CONDITION, Fixture, Name, Fixture)

#endif
