#include "Ciphertext.hpp"
#include "ConcurrentOps.hpp"
#include "Definitions.hpp"

#include <iostream>
#include <mutex>
#include <openfhe.h>

namespace fideslib {

namespace {
/// THE HOST SHADOW'S COPY-ON-WRITE LOCK. Taken ONLY when several host threads may issue ops on one
/// CryptoContext (`FIDESLIB_CONCURRENT_OPS`); in the default mode every path below is exactly the
/// unlocked code it has always been.
///
/// WHAT IS SHARED. A CiphertextImpl copy SHARES the underlying `lbcrypto::Ciphertext` (a
/// shared_ptr) with its source and sets `need_lazy_copy`; the value is detached on first host
/// mutation. That is a plain copy-on-write, and its "am I the only owner?" test is
/// `ct_host.use_count() <= 1`.
///
/// WHY IT IS NOT THREAD SAFE AS WRITTEN. `use_count()` is a snapshot, not a claim: another thread
/// copying this ciphertext raises it the instant after it is read. Two windows follow, and both
/// end in one `lbcrypto::CiphertextImpl` being written while another thread reads or writes it:
///
///   * A reads `use_count() == 1`, concludes the value is private and mutates it in place; B's
///     copy constructor has meanwhile taken a second reference to that same value, so B now owns
///     a ciphertext that A is rewriting underneath it.
///   * A and B both detach from one shared value at the same time; both copy-construct from
///     `*ct_host` while the other may already be mutating it.
///
/// A backend that keeps its values on a device never reaches this -- its ops never touch the host
/// shadow. The CUDA backend's two-iteration bootstrap does: Meta-BTS reads the ciphertext back and
/// runs its glue arithmetic through lbcrypto (CudaEngine::evalBootstrap, numIterations == 2). So
/// an application that bootstraps from several threads hits it on every call, while one that only
/// multiplies never does.
///
/// ONE process-wide mutex, held across the test AND the copy, covers both windows: the reference
/// count cannot move between the test and the decision, and no two threads copy-construct from one
/// value at once. It is uncontended in the default mode and never taken there at all.
std::mutex& hostShadowLock() {
	static std::mutex m;
	return m;
}

/// Locks `hostShadowLock()` in the concurrent mode and nothing in the default mode.
struct MaybeHostShadowLock {
	std::unique_lock<std::mutex> lk;
	MaybeHostShadowLock() {
		if (ConcurrentOpsEnabled())
			lk = std::unique_lock<std::mutex>(hostShadowLock());
	}
};
} // namespace

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

// ---- Copy ----

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CiphertextImpl<DCRTPoly>& other) {
	// Share host ciphertext and detach only on first host mutation. Under hostShadowLock in the
	// concurrent mode so that the reference this takes cannot land between another thread's
	// use_count() test and its decision to mutate in place -- see hostShadowLock().
	{
		MaybeHostShadowLock guard;
		auto const& other_host = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(other.host);
		this->host			   = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(other_host);
		this->need_lazy_copy   = true;
	}

	// Copy parent context.
	this->parent_context = other.parent_context;

	// Cloning any backend-resident payload is the engine's decision, not ours: it returns an empty
	// slot when `other` is not resident (always so on the CPU backend).
	this->device = other.parent_context->CloneCiphertextBackend(other);
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const Ciphertext<DCRTPoly>& other) : CiphertextImpl<DCRTPoly>(static_cast<const CiphertextImpl<DCRTPoly>&>(other)) {
}

// ---- Clone ----

Ciphertext<DCRTPoly> CiphertextImpl<DCRTPoly>::Clone() const {
	Ciphertext<DCRTPoly> clone = std::make_shared<CiphertextImpl<DCRTPoly>>(*this);
	return clone;
}

// ---- Getters / setters ----
//
// These delegate unconditionally to the backend; the engine decides whether to read the host value or
// the device-resident copy. The host primitives below hold the host computation the engines reuse.

size_t CiphertextImpl<DCRTPoly>::GetLevel() const {
	return this->parent_context->CiphertextLevel(*this);
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDeg() const {
	return this->parent_context->CiphertextNoiseScaleDeg(*this);
}

double CiphertextImpl<DCRTPoly>::GetScalingFactor() const {
	return this->parent_context->CiphertextScalingFactor(*this);
}

size_t CiphertextImpl<DCRTPoly>::GetSlots() const {
	return this->parent_context->CiphertextSlots(*this);
}

size_t CiphertextImpl<DCRTPoly>::GetNumElements() const {
	return this->parent_context->CiphertextNumElements(*this);
}

lbcrypto::PlaintextEncodings CiphertextImpl<DCRTPoly>::GetEncodingType() const {
	// Fixed at creation and never changed by a backend, so the host shadow is always current:
	// no readback and no engine dispatch.
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetEncodingType();
}

const std::vector<lbcrypto::DCRTPoly>& CiphertextImpl<DCRTPoly>::GetElements() {
	this->parent_context->RefreshCiphertextHost(*this);
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetElements();
}

void CiphertextImpl<DCRTPoly>::SetSlots(size_t slots) {
	this->parent_context->SetCiphertextSlots(*this, slots);
}

void CiphertextImpl<DCRTPoly>::SetLevel(size_t level) {
	this->parent_context->SetCiphertextLevel(*this, level);
}

// ---- Host primitives (invoked by the engine backends; no backend logic of their own) ----

size_t CiphertextImpl<DCRTPoly>::GetLevelHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetLevel();
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDegHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetNoiseScaleDeg();
}

void CiphertextImpl<DCRTPoly>::SetSlotsHost(size_t slots) {
	this->EnsureLazyHostCopy();
	auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	ct->SetSlots(slots);
}

void CiphertextImpl<DCRTPoly>::SetLevelHost(size_t level) {
	this->EnsureLazyHostCopy();
	auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);

	size_t currentTowers = ct->GetElements()[0].GetNumOfElements();
	size_t currentLevel	 = ct->GetLevel();

	size_t totalPrimes	= currentTowers + currentLevel;
	size_t targetTowers = totalPrimes - level;

	if (currentTowers > targetTowers) {
		// Need to drop towers
		size_t towersToDrop = currentTowers - targetTowers;

		auto& elements = ct->GetElements();
		for (auto& elem : elements) {
			elem.DropLastElements(towersToDrop);
		}
	}

	ct->SetLevel(level);
}

void CiphertextImpl<DCRTPoly>::EnsureLazyHostCopy() {
	// The test and the copy are ONE critical section in the concurrent mode: the count this reads
	// must still be true when it decides to mutate in place, and no two threads may copy-construct
	// from one shared value at once. See hostShadowLock().
	MaybeHostShadowLock guard;
	auto const& ct_host = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	// Detach before in-place mutation if still lazily shared or the underlying
	// OpenFHE ciphertext has other owners (e.g. a Clone): copy-on-write.
	if (!this->need_lazy_copy && ct_host.use_count() <= 1) {
		return;
	}
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> host_copy = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_host);
	this->host										   = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(host_copy));
	this->need_lazy_copy							   = false;
}

double CiphertextImpl<DCRTPoly>::GetScalingFactorHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetScalingFactor();
}

size_t CiphertextImpl<DCRTPoly>::GetSlotsHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetSlots();
}

size_t CiphertextImpl<DCRTPoly>::GetNumElementsHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetElements().size();
}

// ---- Operators ----

Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs) {
	if (lhs->parent_context.get() != rhs->parent_context.get())
		OPENFHE_THROW("Cannot add ciphertexts from different contexts");
	return lhs->parent_context->EvalAdd(lhs, rhs);
}

bool operator==(const CiphertextImpl<DCRTPoly>& lhs, const CiphertextImpl<DCRTPoly>& rhs) {
	// GetElements() only refreshes the host shadow from the backend: the ciphertext value is
	// logically unchanged, so the const_cast is safe. Move towards mutable ctxt state
	// variables to remove need for const_cast.
	auto& l = const_cast<CiphertextImpl<DCRTPoly>&>(lhs);
	auto& r = const_cast<CiphertextImpl<DCRTPoly>&>(rhs);
	return lhs.GetNoiseScaleDeg() == rhs.GetNoiseScaleDeg() && lhs.GetScalingFactor() == rhs.GetScalingFactor() &&
	  lhs.GetEncodingType() == rhs.GetEncodingType() && l.GetElements() == r.GetElements();
}

bool operator!=(const CiphertextImpl<DCRTPoly>& lhs, const CiphertextImpl<DCRTPoly>& rhs) {
	return !(lhs == rhs);
}

} // namespace fideslib
