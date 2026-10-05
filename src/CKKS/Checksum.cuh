#ifndef FIDESLIB_CKKS_CHECKSUM_CUH
#define FIDESLIB_CKKS_CHECKSUM_CUH
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
} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_CHECKSUM_CUH
