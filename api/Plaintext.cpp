#include "Plaintext.hpp"
#include "CryptoContext.hpp"

#include <openfhe.h>

namespace fideslib {

PlaintextImpl::PlaintextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

namespace {

/// Every accessor below treats "no host value" as "not encoded yet" and answers with a neutral
/// default. For a device-only plaintext that default would be silently WRONG (level 0, slots 0, an
/// empty value vector) rather than merely uninformative, so device-only is separated out and made
/// loud before the has_value() test is reached.
void RejectDeviceOnly(bool device_only, const char* op) {
	if (device_only) {
		OPENFHE_THROW(std::string(op) +
		  ": this plaintext was encoded on the device (MakeCKKSPackedPlaintextDevice) and has no host value. "
		  "Device-only plaintexts are valid only where the backend value is consumed — e.g. as LinearTransformInPlace diagonals. "
		  "Use MakeCKKSPackedPlaintext if the host value is needed.");
	}
}

} // namespace

// ---- Functions ----

void PlaintextImpl::SetLength(size_t length) {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::SetLength");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->host);
		impl->SetLength(length);
	}
}

void PlaintextImpl::SetSlots(uint32_t slots) {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::SetSlots");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->host);
		impl->SetSlots(slots);
	}
}

double PlaintextImpl::GetLogPrecision() const {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::GetLogPrecision");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetLogPrecision();
	}
	return 0.0;
}

uint32_t PlaintextImpl::GetLevel() const {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::GetLevel");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetLevel();
	}
	return 0;
}

uint32_t PlaintextImpl::GetSlots() const {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::GetSlots");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetSlots();
	}
	return 0;
}

std::vector<std::complex<double>> PlaintextImpl::GetCKKSPackedValue() const {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::GetCKKSPackedValue");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetCKKSPackedValue();
	}
	return {};
}

std::vector<double> PlaintextImpl::GetRealPackedValue() const {
	RejectDeviceOnly(this->device_only, "PlaintextImpl::GetRealPackedValue");
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetRealPackedValue();
	}
	return {};
}

// ---- Friend Operators ----

std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt) {
	if (pt.host.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt.host);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, const Plaintext& pt) {
	if (pt && pt->host.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

bool operator==(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
	if (!lhs.host.has_value() || !rhs.host.has_value())
		return lhs.host.has_value() == rhs.host.has_value();
	const auto& l = std::any_cast<const lbcrypto::Plaintext&>(lhs.host);
	const auto& r = std::any_cast<const lbcrypto::Plaintext&>(rhs.host);
	if (!l || !r)
		return l == r;
	return *l == *r;
}

bool operator!=(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
	return !(lhs == rhs);
}

} // namespace fideslib
