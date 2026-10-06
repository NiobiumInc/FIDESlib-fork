#ifndef FIDESLIB_CKKS_CHECKSUM_CUH
#define FIDESLIB_CKKS_CHECKSUM_CUH
#include <functional>
#include <string>

namespace FIDESlib::CKKS {
class Ciphertext;

/// DIAGNOSTIC (FIDESLIB_CKSUM=1, off by default; see Checksum.cu). Per-thread asynchronous
/// digests of a ciphertext at named points, printed by ChecksumFlush() as
/// `[cksum] <label> probe=<i> tag=<tag> c0=<hex> c1=<hex>`.
bool ChecksumEnabled();
/// Sets this thread's label (the test names stage / mode / rep / lane). No-op when disabled.
void ChecksumLabel(const std::string& label);
/// Enqueues the digest kernels on the ciphertext's partition stream. No-op when disabled.
void ChecksumProbe(const Ciphertext& ct, const char* tag);
inline void ChecksumProbe(const Ciphertext& ct, const std::string& tag) {
	ChecksumProbe(ct, tag.c_str());
}
/// Synchronises the device, prints every probe taken since the last flush, clears them.
void ChecksumFlush();

/// TEST-ONLY STAGE HOOK (null by default). When set, every ChecksumProbe() call -- the named
/// points of the bootstrap pipeline (in, post-modraise, pre-cts, post-cts, pre-evalmod, the em-*
/// EvalMod steps, post-evalmod, pre-stc, post-stc, out) -- hands the ciphertext to the hook
/// synchronously, independent of FIDESLIB_CKSUM. A test holding the secret key uses it to decrypt
/// each stage and attribute a precision loss to the stage that adds it. Unset, it costs one branch.
using StageHook = std::function<void(const char* tag, const Ciphertext& ct)>;
void SetStageHook(StageHook hook);
} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_CHECKSUM_CUH
