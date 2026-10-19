// PROBE (not for merge): a per-thread upload delay that Plaintext::load and KeySwitchingKey::Initialize switch on
// for the duration of their own uploads, read by Limb::load.
#pragma once
#include <cstdlib>

namespace FIDESlib::ProbeDelay {

inline thread_local long long g_cycles = 0;

inline long long EnvCycles(const char* name) {
	const char* env = std::getenv(name);
	return (env != nullptr && env[0] != '\0') ? std::atoll(env) * 2000LL : 0LL; // about 2000 cycles per microsecond
}
inline long long PlaintextCycles() {
	static const long long c = EnvCycles("FIDESLIB_PROBE_PT_DELAY_US");
	return c;
}
inline long long KeyCycles() {
	static const long long c = EnvCycles("FIDESLIB_PROBE_KEY_DELAY_US");
	return c;
}

struct Scope {
	long long saved;
	explicit Scope(long long cycles) : saved(g_cycles) { g_cycles = cycles; }
	~Scope() { g_cycles = saved; }
};

} // namespace FIDESlib::ProbeDelay
