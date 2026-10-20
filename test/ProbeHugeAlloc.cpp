// PROBE (not for merge): while g_probe_huge_alloc is set, every allocation of 256 KiB or more comes from a
// 2 MiB-aligned block that is advised to use huge pages, so cudaHostRegister can pin it on huge pages.
#include <sys/mman.h>

#include <atomic>
#include <cstdlib>
#include <new>

std::atomic<bool> g_probe_huge_alloc{ false };

void* operator new(std::size_t size) {
	void* p = nullptr;
	if (g_probe_huge_alloc.load(std::memory_order_relaxed) && size >= (256u << 10)) {
		const std::size_t huge	  = std::size_t(2) << 20;
		const std::size_t rounded = ((size + huge - 1) / huge) * huge;
		p						  = std::aligned_alloc(huge, rounded);
		if (p != nullptr)
			madvise(p, rounded, MADV_HUGEPAGE);
	} else {
		p = std::malloc(size == 0 ? 1 : size);
	}
	if (p == nullptr)
		throw std::bad_alloc();
	return p;
}

void operator delete(void* p) noexcept {
	std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
	std::free(p);
}
