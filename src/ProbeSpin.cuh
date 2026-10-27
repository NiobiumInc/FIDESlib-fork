// PROBE (not for merge): race amplifiers for the adapter-side stream experiment.
//
// FIDESLIB_PROBE_UPLOAD_DELAY_US=N: once a plaintext has been loaded, ProbeScramble enqueues on the
// buffer's own stream: save it, zero it, spin N microseconds, restore it. All device work, so the host
// is not held back. A reader that waits for the buffer's stream sees the restored data; one that does
// not reads zeros. This amplifies the "before" edge (read after write).
// FIDESLIB_PROBE_KEY_DELAY_US=N: the same for the key-switching keys, with its own (longer) delay,
// because the first key switch can come long after the upload.
// FIDESLIB_PROBE_KEY_B_DELAY_US=N: the delay for the b half of every key (default: the key delay).
// Longer than the a half, it opens a window that only a reader waiting on ksk.b's own stream closes.
// FIDESLIB_PROBE_WAR=1: amplifies the "after" edge (write after read): the reader is delayed by a spin
// on its own stream, and the plaintext is scrambled on its stream as soon as the call returns. Unless
// the plaintext's stream waits for the reader, the scramble lands under the read.
#pragma once
#include <chrono>
#include <cstddef>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

namespace FIDESlib {
long long ProbeUploadDelayCycles();
long long ProbeKeyDelayCycles();
long long ProbeKeyBDelayCycles();
bool ProbeWar();
void ProbeSpin(cudaStream_t stream, long long cycles);
/// The last scramble's spin: requested cycles, launch status, and events around it.
struct ProbeSpinStats {
	long long cycles = 0;
	int launch		 = 0;
	cudaEvent_t start = nullptr, end = nullptr;
};
ProbeSpinStats& ProbeLastSpin();
/// When the last ProbeScramble was enqueued (host clock).
std::chrono::steady_clock::time_point& ProbeLastScrambleTime();
void ProbeScramble(cudaStream_t stream, const std::vector<std::pair<void*, size_t>>& buffers, long long cycles);
} // namespace FIDESlib
