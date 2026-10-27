// PROBE (not for merge): upload-delay amplifier for the adapter-side stream experiment.
// FIDESLIB_PROBE_UPLOAD_DELAY_US=N makes ProbeUploadDelay put an N-microsecond spin kernel on a stream,
// so that whatever is enqueued on it next (a plaintext's or a key's upload) lands late, and any reader
// that does not wait for that stream reads stale data.
#pragma once
#include <cuda_runtime.h>

namespace FIDESlib {
long long ProbeUploadDelayCycles();
void ProbeUploadDelay(cudaStream_t stream);
} // namespace FIDESlib
