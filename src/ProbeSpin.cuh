// PROBE (not for merge): race amplifier for the adapter-side stream experiment.
// FIDESLIB_PROBE_UPLOAD_DELAY_US=N makes ProbeScramble, once a buffer has been loaded, enqueue on the
// buffer's own stream: save it, zero it, spin N microseconds, restore it. All four are device work, so
// the host is not held back (a pageable upload would block the host on an earlier spin instead of
// opening a window). A reader that waits for the buffer's stream sees the restored data; a reader that
// does not reads zeros for N microseconds.
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace FIDESlib {
long long ProbeUploadDelayCycles();
void ProbeScramble(cudaStream_t stream, void* data, size_t bytes);
} // namespace FIDESlib
