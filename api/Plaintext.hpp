#ifndef API_PLAINTEXT_HPP
#define API_PLAINTEXT_HPP

#include <any>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl {
  public:
	PlaintextImpl()	 = default;
	~PlaintextImpl() = default; // device slot frees its backend payload via RAII

	PlaintextImpl(const CryptoContext<DCRTPoly>&& context);

	// ---- Copy ----

	PlaintextImpl(const PlaintextImpl&);
	PlaintextImpl(const Plaintext&);
	PlaintextImpl& operator=(const PlaintextImpl&) = delete;
	PlaintextImpl& operator=(const Plaintext&)	   = delete;

	// ---- Move ----

	PlaintextImpl(PlaintextImpl&&)			  = delete;
	PlaintextImpl(Plaintext&&)				  = delete;
	PlaintextImpl& operator=(PlaintextImpl&&) = delete;
	PlaintextImpl& operator=(Plaintext&&)	  = delete;

	// ---- Functions ----

	void SetLength(size_t length);
	void SetSlots(uint32_t slots);

	double GetLogPrecision() const;
	uint32_t GetLevel() const;
	uint32_t GetSlots() const;
	std::vector<std::complex<double>> GetCKKSPackedValue() const;
	std::vector<double> GetRealPackedValue() const;

	// ---- Friend Operators ----

	friend std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt);

	// ---- Internal State ----

	/// @brief Canonical host value (lbcrypto::Plaintext).
	std::any host;
	/// @brief Backend-resident payload (the CUDA backend stores a shared_ptr to its device plaintext); empty == not resident.
	std::any device;
	/// @brief The caller asked for this plaintext in the extended (mod-up, Q||P) basis, via
	/// CryptoContextImpl::MakeCKKSPackedPlaintextExtended.
	///
	/// This records INTENT and is set on every backend, including the ones that ignore the
	/// request. Whether `host` actually carries a Q||P encoding is a separate, backend-dependent
	/// fact: only backends whose device basis has the special primes (CUDA) encode differently;
	/// CPU and haze hand back the ordinary Q-basis encoding. LinearTransformInPlace's `ext`
	/// precondition is checked against this flag so it means the same thing on every backend.
	///
	/// An extended plaintext is ONLY valid as a LinearTransformInPlace diagonal. It is not a
	/// drop-in for EvalMult/EvalAdd against a ciphertext: on a device backend the extra P limbs
	/// change what those ops compute, because nothing in them performs the compensating ModDown.
	bool extended = false;
	/// @brief This plaintext exists ONLY on the backend: `host` is empty and will stay empty.
	///
	/// Set by CryptoContextImpl::MakeCKKSPackedPlaintextDevice on a backend that encodes on the
	/// device. Materialising the host value would mean performing exactly the encode that call
	/// skips, so it is not done lazily either — the flag exists to turn a host-side read into a
	/// legible error rather than a std::bad_any_cast from an empty `std::any`.
	///
	/// A device-only plaintext is valid wherever only the backend value is consumed, which is the
	/// case for LinearTransformInPlace diagonals — the one and only consumer this was built for.
	bool device_only = false;
	/// @brief Parent context.
	CryptoContext<DCRTPoly> parent_context;
};

// ---- Override Operators ----

std::ostream& operator<<(std::ostream& os, const Plaintext& pt);
bool operator==(const PlaintextImpl& lhs, const PlaintextImpl& rhs);
bool operator!=(const PlaintextImpl& lhs, const PlaintextImpl& rhs);
} // namespace fideslib

#endif // API_PLAINTEXT_HPP