#ifndef API_HAZEPAYLOAD_HPP
#define API_HAZEPAYLOAD_HPP

#ifdef FIDESLIB_ENABLE_HAZE

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fideslib::hazebk {

/// @brief Residency of a haze-backed value in the one-program execution model.
/// - Uploaded: created by H2D (encrypted inputs, plaintexts, key limbs). A program input;
///   legal compute source; D2H is legal without a flush (the shadow already holds the bytes).
/// - Recorded: produced by recorded compute in the in-flight program. D2H is illegal until
///   the program executes (hazeFlush); reading one triggers the engine's single materialize().
/// - Flushed: a declared output materialized by the program's single flush; D2H reads the shadow.
enum class Residency { Uploaded, Recorded, Flushed };

/// @brief RAII move-only vector of haze polynomial allocations, one per RNS limb
/// (port of haze test/e2e/ops.cpp `Allocs`). Every allocation is exactly the configured
/// polynomial size (ring_dim * 8 bytes — haze's hazeMalloc contract). The destructor
/// swallows hazeFree errors: after a hazeDeviceReset (context teardown) outstanding
/// addresses are stale and freeing them reports INVALID_VALUE, which is expected.
class LimbChain {
  public:
	LimbChain() = default;
	/// @brief Allocate `count` fresh device polynomials of `polyBytes` each (hazeMalloc).
	LimbChain(std::size_t count, std::size_t polyBytes);
	/// @brief Allocate one device polynomial per row and H2D-upload the row into it.
	explicit LimbChain(const std::vector<std::vector<uint64_t>>& rows);

	~LimbChain();
	LimbChain(LimbChain&& other) noexcept;
	LimbChain& operator=(LimbChain&& other) noexcept;
	LimbChain(const LimbChain&)			   = delete;
	LimbChain& operator=(const LimbChain&) = delete;

	/// @brief Free the trailing limbs so only `newCount` remain (level drop; no IR is recorded —
	/// OpenFHE's DropLastElements does no arithmetic either).
	void truncate(std::size_t newCount);

	void* const* data() const noexcept { return ptrs_.data(); }
	void** data() noexcept { return ptrs_.data(); }
	std::size_t size() const noexcept { return ptrs_.size(); }
	bool empty() const noexcept { return ptrs_.empty(); }
	void* operator[](std::size_t i) const noexcept { return ptrs_[i]; }

	/// @brief Pointer view for the `const void* const*` source arguments of the haze MRP ops.
	std::vector<const void*> asConst() const;

  private:
	void freeAll() noexcept;
	std::vector<void*> ptrs_;
};

/// @brief Device payload of a haze-backed ciphertext: one LimbChain per component plus the
/// host-side CKKS metadata the engine maintains. Stored as a shared_ptr in the
/// value type's `device` slot.
struct HazePayload {
	LimbChain c0;
	LimbChain c1;
	/// @brief #RNS limbs; OpenFHE level = |Q| − towers.
	std::size_t towers = 0;
	/// @brief OpenFHE m_noiseScaleDeg.
	std::size_t noiseScaleDeg = 1;
	/// @brief OpenFHE m_scalingFactor (parity-critical: follows OpenFHE's exact op order).
	double scalingFactor = 0.0;
	/// @brief Batch size.
	std::size_t slots = 0;
	Residency state	  = Residency::Uploaded;
};

/// @brief Device payload of a haze-backed plaintext: a single uploaded chain + metadata.
/// Plaintexts are always program inputs (Uploaded); they are never computed on the device.
struct HazePtPayload {
	LimbChain chain;
	std::size_t towers		  = 0;
	std::size_t noiseScaleDeg = 1;
	double scalingFactor	  = 0.0;
	std::size_t slots		  = 0;
};

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
#endif // API_HAZEPAYLOAD_HPP
