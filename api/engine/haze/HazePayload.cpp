#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazePayload.hpp"

#include <haze/haze.h>

#include <openfhe.h> // OPENFHE_THROW

#include <string>

namespace fideslib::hazebk {

namespace {
void hazeCheck(hazeError_t err, const char* what) {
	if (err != HAZE_SUCCESS) {
		OPENFHE_THROW(std::string("haze backend: ") + what + " failed: " + hazeGetErrorString(err));
	}
}
} // namespace

LimbChain::LimbChain(std::size_t count, std::size_t polyBytes) {
	ptrs_.assign(count, nullptr);
	// On a mid-loop hazeMalloc failure the destructor does not run (the object never finished
	// constructing), so free the earlier allocations before rethrowing.
	try {
		for (std::size_t i = 0; i < count; ++i) {
			hazeCheck(hazeMalloc(&ptrs_[i], polyBytes), "hazeMalloc");
		}
	} catch (...) {
		freeAll();
		throw;
	}
}

LimbChain::LimbChain(const std::vector<std::vector<uint64_t>>& rows) {
	ptrs_.assign(rows.size(), nullptr);
	try {
		for (std::size_t i = 0; i < rows.size(); ++i) {
			const std::size_t bytes = rows[i].size() * sizeof(uint64_t);
			hazeCheck(hazeMalloc(&ptrs_[i], bytes), "hazeMalloc");
			hazeCheck(hazeMemcpy(ptrs_[i], rows[i].data(), bytes, HAZE_MEMCPY_HOST_TO_DEVICE), "hazeMemcpy(H2D)");
		}
	} catch (...) {
		freeAll();
		throw;
	}
}

LimbChain::~LimbChain() {
	freeAll();
}

LimbChain::LimbChain(LimbChain&& other) noexcept : ptrs_(std::move(other.ptrs_)) {
	other.ptrs_.clear();
}

LimbChain& LimbChain::operator=(LimbChain&& other) noexcept {
	if (this != &other) {
		freeAll();
		ptrs_ = std::move(other.ptrs_);
		other.ptrs_.clear();
	}
	return *this;
}

void LimbChain::truncate(std::size_t newCount) {
	if (newCount >= ptrs_.size()) {
		return;
	}
	for (std::size_t i = newCount; i < ptrs_.size(); ++i) {
		if (ptrs_[i] != nullptr) {
			(void)hazeFree(ptrs_[i]);
		}
	}
	ptrs_.resize(newCount);
}

std::vector<const void*> LimbChain::asConst() const {
	return { ptrs_.begin(), ptrs_.end() };
}

void LimbChain::freeAll() noexcept {
	// Swallow INVALID_VALUE so a stale post-reset address doesn't trip the destructor;
	// hazeFree(nullptr) is already a no-op (ops.cpp free_all precedent).
	for (void* p : ptrs_) {
		if (p != nullptr) {
			(void)hazeFree(p);
		}
	}
	ptrs_.clear();
}

} // namespace fideslib::hazebk

#endif // FIDESLIB_ENABLE_HAZE
