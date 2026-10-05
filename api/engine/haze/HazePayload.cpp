#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazePayload.hpp"

#include <haze/haze.h>

#include <openfhe.h>

#include <stdexcept>
#include <string>

namespace fideslib::hazebk {

namespace {
void hazeCheck(hazeError_t err, const char* what) {
	if (err != HAZE_SUCCESS) {
		throw std::runtime_error(std::string("haze backend: ") + what + " failed: " + hazeGetErrorString(err));
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

LimbChain::LimbChain(const std::vector<std::vector<uint64_t>>& rows, const std::vector<uint64_t>& base) {
	if (base.size() != rows.size()) {
		throw std::runtime_error("haze backend: LimbChain upload base names " + std::to_string(base.size()) + " prime(s) for " + std::to_string(rows.size()) + " row(s)");
	}
	ptrs_.assign(rows.size(), nullptr);
	// Allocation is still one polynomial at a time, so keep the rollback: on a mid-loop failure
	// the destructor does not run (the object never finished constructing).
	try {
		std::size_t bytes = 0;
		std::vector<const void*> srcs;
		srcs.reserve(rows.size());
		for (std::size_t i = 0; i < rows.size(); ++i) {
			const std::size_t rowBytes = rows[i].size() * sizeof(uint64_t);
			if (i == 0) {
				bytes = rowBytes;
			} else if (rowBytes != bytes) {
				throw std::runtime_error("haze backend: LimbChain rows disagree on length");
			}
			hazeCheck(hazeMalloc(&ptrs_[i], rowBytes), "hazeMalloc");
			srcs.push_back(rows[i].data());
		}
		if (!rows.empty()) {
			// One MRP input entry for the whole chain, each residue under its declared prime.
			hazeCheck(hazeMemcpyMrp(ptrs_.data(), srcs.data(), bytes, HAZE_MEMCPY_HOST_TO_DEVICE, base.data(), base.size()), "hazeMemcpyMrp(H2D)");
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
