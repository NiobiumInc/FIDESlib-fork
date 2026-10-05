#ifndef API_CIPHERTEXT_HPP
#define API_CIPHERTEXT_HPP

#include <any>
#include <memory>
#include <vector>

#include "constants-defs.h"
#include "lattice/hal/lat-backend.h"

#include "CryptoContext.hpp"
#include "Definitions.hpp"

namespace fideslib {

/// @brief Specialization of Ciphertext for the DCRTPoly representation.
template <> class CiphertextImpl<DCRTPoly> {
  public:
	CiphertextImpl()  = delete;
	~CiphertextImpl() = default; // device slot frees its backend payload via RAII

	CiphertextImpl(const CryptoContext<DCRTPoly>&& context);

	// ---- Copy ----

	CiphertextImpl(const CiphertextImpl<DCRTPoly>&);
	CiphertextImpl(const Ciphertext<DCRTPoly>&);
	CiphertextImpl& operator=(const CiphertextImpl<DCRTPoly>&) = delete;
	CiphertextImpl& operator=(const Ciphertext<DCRTPoly>&)	   = delete;

	// ---- Move ----

	CiphertextImpl(CiphertextImpl<DCRTPoly>&&)			  = delete;
	CiphertextImpl(Ciphertext<DCRTPoly>&&)				  = delete;
	CiphertextImpl& operator=(CiphertextImpl<DCRTPoly>&&) = delete;
	CiphertextImpl& operator=(Ciphertext<DCRTPoly>&&)	  = delete;

	// ---- Clone ----
	Ciphertext<DCRTPoly> Clone() const;

	// ---- Getters / setters (delegate to the backend) ----

	size_t GetLevel() const;
	size_t GetNoiseScaleDeg() const;
	double GetScalingFactor() const;
	size_t GetSlots() const;
	/// @brief OpenFHE NumberCiphertextElements: 2 normally, 3 after EvalMultNoRelin.
	size_t GetNumElements() const;
	/// @brief OpenFHE spelling of GetNumElements(), for drop-in compatibility.
	size_t NumberCiphertextElements() const {
		return GetNumElements();
	}
	lbcrypto::PlaintextEncodings GetEncodingType() const;
	/// @brief Current RNS elements. Refreshes the host shadow from the backend first, so the
	/// result reflects backend-resident work. On haze this requires the value to have been
	/// flushed already (MarkOutput + Decrypt); it never triggers a flush of its own.
	const std::vector<lbcrypto::DCRTPoly>& GetElements();
	void SetSlots(size_t slots);
	void SetLevel(size_t level);

	// ---- Host primitives (the host computation the engine backends reuse; no backend logic) ----

	size_t GetLevelHost() const;
	size_t GetNoiseScaleDegHost() const;
	double GetScalingFactorHost() const;
	size_t GetSlotsHost() const;
	size_t GetNumElementsHost() const;
	void SetSlotsHost(size_t slots);
	void SetLevelHost(size_t level);
	void EnsureLazyHostCopy();

	// ---- Internal State ----

	bool need_lazy_copy = false;
	/// @brief Canonical host value (lbcrypto::Ciphertext); used by host ops and as the CUDA readback shadow.
	std::any host;
	/// @brief Backend-resident payload (the CUDA backend stores a shared_ptr to its device ciphertext); empty == not resident.
	std::any device;
	/// @brief Parent context.
	CryptoContext<DCRTPoly> parent_context;
};

// ---- Override Operators ----
Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs);
bool operator==(const CiphertextImpl<DCRTPoly>& lhs, const CiphertextImpl<DCRTPoly>& rhs);
bool operator!=(const CiphertextImpl<DCRTPoly>& lhs, const CiphertextImpl<DCRTPoly>& rhs);

} // namespace fideslib

#endif // API_CIPHERTEXT_HPP
