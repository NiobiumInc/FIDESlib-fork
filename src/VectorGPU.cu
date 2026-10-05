//
// Created by carlosad on 2/05/24.
//
#include "VectorGPU.cuh"
#include "CudaUtils.cuh"

namespace FIDESlib {
template <typename T>
VectorGPU<T>::VectorGPU(VectorGPU<T>&& v) noexcept : freeing(v.freeing), managed(v.managed), data(v.data), size(v.size), device(v.device) {
	v.freeing = true;
	v.managed = false;
}

template <typename T>
VectorGPU<T>::VectorGPU(T* data, const int size, const int device, const int offset)
: data(data + offset), size(size), device(device), managed(false), freeing(true) {
	assert(data != nullptr);
	if (DeviceChecks()) { // FIDESLIB_DEVICE_CHECKS=0 skips the driver query the asserts below consume
		cudaPointerAttributes att{};
		cudaPointerGetAttributes(&att, data);
		assert(att.type == cudaMemoryTypeManaged || att.type == cudaMemoryTypeDevice);
		assert(att.device == this->device);
		(void)att;
	}
	CudaCheckErrorModNoSync;
	assert(size > 0);

	Out(MEMORY, "Unmanaged vector construct OK");
}

template <typename T> VectorGPU<T>::~VectorGPU() {
	assert(freeing == true);
	Out(MEMORY, "Vector destruct OK");
}

/// IDEMPOTENT. Freeing an already-freed vector is a no-op, not a double free.
///
/// It used to be neither: the guard tested only `managed`, which free() never clears, so a second
/// call fell straight through to GPUfree — an assert in a debug build and a genuine double free in
/// a release one. That made "release a buffer early and let the destructor mop up" unexpressible,
/// which is why the NTT scratch of a device-encoded plaintext was held for the plaintext's whole
/// lifetime (see RNSPoly::loadCoefficients) even though it is dead the moment the transform ends.
///
/// The `assert(!freeing)` that used to stand here is deliberately gone rather than relaxed: with
/// this guard an early release followed by the owner's destructor is the INTENDED pattern, so the
/// second call is not a bug to catch. Callers that only ever free once are unaffected — the first
/// call still does exactly what it did.
///
/// Unmanaged vectors (a view onto someone else's buffer, and every zero-size vector) return here
/// as they always have, so an early release can never free memory this vector does not own.
template <typename T> void VectorGPU<T>::free(Stream& stream) {
	if (!managed || freeing) {
		return;
	}
	// cudaDeviceSynchronize();
	GPUfree(data, device, sizeof(T) * size, stream.ptr(), true);
	// cudaFreeAsync((void*)data, stream.ptr());
	freeing = true;
	Out(MEMORY, "Managed vector free OK");
}

template <typename T>
VectorGPU<T>::VectorGPU(Stream& stream, const int size, const int device, const T* src)
: data(nullptr), size(size), device(device), freeing(false), managed(true) {
	assert(device >= 0);
	if (DeviceChecks()) { // FIDESLIB_DEVICE_CHECKS=0 skips the two driver queries the asserts consume
		int device_count = -1;
		const cudaError_t rc_count = cudaGetDeviceCount(&device_count);
		assert(rc_count == cudaSuccess);
		assert(device < device_count);
		(void)device_count;
		(void)rc_count;
		int dev			   = -1;
		const cudaError_t rc_dev = cudaGetDevice(&dev);
		assert(rc_dev == cudaSuccess);
		assert(dev == device);
		(void)dev;
		(void)rc_dev;
	}
	int bytes = size * sizeof(T);

	if (size == 0) {
		managed = false;
		freeing = true;
	} else {
		// cudaDeviceSynchronize();
		data = (T*)GPUmalloc(device, bytes, stream.ptr(), true);
		// cudaDeviceSynchronize();
		// cudaMallocAsync(&data, bytes, stream.ptr());

		if (src != nullptr) {
			UploadH2D(data, src, bytes, device, stream.ptr());
		}
	}
	Out(MEMORY, "Managed vector construct OK");
}

template class VectorGPU<int>;
template class VectorGPU<void*>;
template class VectorGPU<void**>;
template class VectorGPU<uint32_t>;
template class VectorGPU<uint64_t>;
} // namespace FIDESlib