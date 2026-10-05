//
// Created by carlosad on 16/03/24.
//
#ifndef FIDESLIB_CONSTANTSGPU_CUH
#define FIDESLIB_CONSTANTSGPU_CUH

#include "LimbUtils.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib {
constexpr int MAXP = 64;
// MAXD is the maximum hybrid key-switching digit count (dnum): it bounds the
// first index of the [MAXD][MAXP] digit arrays in `Constants` below, which are
// written from digitGPUid values running 0..dnum-1. `Constants host_constants`
// is a STACK LOCAL in SetupConstants, so exceeding MAXD used to smash the
// stack rather than fail -- see the note on ContextData::validateDnum, which
// now rejects it.
//
// RAISED 8 -> 13 (2026-09-01). dnum is a SECURITY parameter as well as a
// performance one: |P| = ceil(|Q|/dnum), and hybrid key switching publishes
// evaluation keys mod Q*P, so growing dnum shrinks the modulus an attacker
// sees. At N=2^16, L=26 the useful values are dnum 9 (log2(QP) 1612) and
// dnum 13 (1552); both exceeded the old ceiling of 8.
//
// THE CEILING IS CONSTANT MEMORY, not taste. CUDA gives a module 64 KB of
// __constant__ space and `constants` is essentially all of it: at MAXD 8 the
// struct is already ~52 KB, and each +1 costs ~1.5 KB across six [MAXD][MAXP]
// int arrays plus the flattened union. MAXD 16 lands within a few hundred
// bytes of the limit and 32 blows straight past it. The static_assert after
// the struct makes that a BUILD failure instead of a load-time surprise.
constexpr int MAXD = 13;

enum version { BARRET, DHEM, NEIL };

constexpr version VERSION = NEIL;

namespace CKKS {
struct Scheme {
	struct Constants;
	struct Global;
};

struct Scheme::Constants {};

struct Scheme::Global {};
} // namespace CKKS

struct Constants {
	int N, L, K, logN;
	uint64_t type;
	uint64_t primes[MAXP];
	uint64_t prime_better_barret_mu[MAXP];
	uint32_t prime_bits[MAXP];
	uint8_t table[MAXP * MAXP * 8];

	uint64_t N_shoup[MAXP];
	uint64_t one_shoup[MAXP];
	uint64_t N_inv[MAXP];
	uint64_t N_inv_shoup[MAXP];
	uint64_t root[MAXP];
	uint64_t root_shoup[MAXP];
	uint64_t inv_root[MAXP];
	uint64_t inv_root_shoup[MAXP];

	int dnum;
	bool reduced_noise;
	uint64_t P[MAXP];
	uint64_t P_shoup[MAXP];
	uint64_t P_inv[MAXP];
	uint64_t P_inv_shoup[MAXP];

	int num_primeid_digit_from[MAXD][MAXP];
	int num_primeid_digit_to[MAXD][MAXP];
	int primeid_digit[MAXP];

	int pos_in_digit[MAXD][MAXP];

	union {
		struct {
			int primeid_partition[MAXD][MAXP];	 // Limb primeids
			int primeid_digit_from[MAXD][MAXP];	 // Decomp primeids
			int primeid_digit_to[MAXD][MAXP];	 // Digit primeids
			int primeid_special_partition[MAXP]; // Special limb primeids
		};

		int primeid_flattened[(MAXD * 3 + 1) * MAXP];
	};

	union {
		CKKS::Scheme::Constants ckks;
		void* none;
	};
};

// `constants` is declared __constant__ below, and CUDA gives each module 64 KB
// of constant memory total. Raising MAXD grows this struct; fail the build here
// rather than at module load, where the error is far less obvious.
static_assert(sizeof(Constants) <= 64 * 1024,
			  "Constants exceeds CUDA's 64 KB __constant__ budget -- lower MAXD "
			  "(or move a digit-indexed array out of constant memory).");

constexpr int PARTITION(int id, int j) {
	return (offsetof(Constants, primeid_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + id * MAXP + j;
}

constexpr int SPECIAL(int d, int j) {
	return (offsetof(Constants, primeid_special_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + j;
}

constexpr int DECOMP(int d, int j) {
	return (offsetof(Constants, primeid_digit_from) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP + j;
}

constexpr int DIGIT(int d, int j) {
	return (offsetof(Constants, primeid_digit_to) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP + j;
}

struct Global {
	// NTT
	uint64_t type;
	void* psi[MAXP];
	void* inv_psi[MAXP];
	void* psi_no[MAXP];
	void* inv_psi_no[MAXP];
	void* psi_middle_scale[MAXP];
	void* inv_psi_middle_scale[MAXP];
	void* psi_shoup[MAXP];
	void* inv_psi_shoup[MAXP];
	// GPU

	void* psi_ptr[MAXD][MAXP];
	void* inv_psi_ptr[MAXD][MAXP];
	void* psi_no_ptr[MAXD][MAXP];
	void* inv_psi_no_ptr[MAXD][MAXP];
	void* psi_middle_scale_ptr[MAXD][MAXP];
	void* inv_psi_middle_scale_ptr[MAXD][MAXP];
	void* psi_shoup_ptr[MAXD][MAXP];
	void* inv_psi_shoup_ptr[MAXD][MAXP];

	uint64_t root[MAXP];
	uint64_t inv_root[MAXP];

	// Conv
	// uint64_t q_[MAXP][MAXP];
	// uint64_t Q_[MAXP][MAXP];
	uint64_t q_inv[MAXP][MAXP];
	uint64_t QlQlInvModqlDivqlModq[MAXP][MAXP];
	uint64_t aux;

	uint64_t ModDown_pre_scale[MAXP];
	uint64_t ModDown_pre_scale_shoup[MAXP];
	uint64_t ModDown_matrix[MAXP][MAXP];
	uint64_t ModDown_matrix_shoup[MAXP][MAXP];

	uint64_t DecompAndModUp_pre_scale[MAXD][MAXP][MAXP];
	uint64_t DecompAndModUp_pre_scale_shoup[MAXD][MAXP][MAXP];
	uint64_t DecompAndModUp_matrix[MAXP][MAXP][MAXP];
	uint64_t DecompAndModUp_matrix_shoup[MAXP][MAXP][MAXP];
	// ReducedNoise FBC: -Q_{d,s} mod p_t, see SetupConstants's fill for what d/s/t index.
	uint64_t DecompAndModUp_negQ[MAXD][MAXP][MAXP];

	union {
		CKKS::Scheme::Global ckks;
		void* none;
	};

	struct Globals {
		void* psi[MAXP];
		void* inv_psi[MAXP];
		void* psi_middle_scale[MAXP];
		void* inv_psi_middle_scale[MAXP];
		void* psi_no[MAXP];
		void* inv_psi_no[MAXP];
		void* psi_shoup[MAXP];
		void* inv_psi_shoup[MAXP];

		//__device__ uint64_t q_[MAXP][MAXP];
		//__device__ uint64_t Q_[MAXP][MAXP];
		uint64_t q_inv[MAXP * MAXP];
		uint64_t QlQlInvModqlDivqlModq[MAXP * MAXP];

		uint64_t ModDown_pre_scale[MAXP];
		uint64_t ModDown_pre_scale_shoup[MAXP];
		uint64_t ModDown_matrix[MAXP * MAXP];
		uint64_t ModDown_matrix_shoup[MAXP * MAXP];

		uint64_t DecompAndModUp_pre_scale[MAXD * MAXP * MAXP];
		uint64_t DecompAndModUp_pre_scale_shoup[MAXD * MAXP * MAXP];
		uint64_t DecompAndModUp_matrix[MAXP * /*MAXD */ MAXP * MAXP];
		uint64_t DecompAndModUp_matrix_shoup[MAXP * /*MAXD */ MAXP * MAXP];
		uint64_t DecompAndModUp_negQ[MAXD * MAXP * MAXP];
	};

	Globals* globals[MAXD];

	~Global();
};

extern __constant__ Constants constants;
// extern Constants host_constants;
// extern Constants host_constants_per_gpu[8];
// extern Global host_global;

#define C_ (constants)
#define G_ Globals
#define TABLE32(i, j) (constants.table[MAXP * 8 * (i) + 4 * (j)])
#define TABLE64(i, j) (constants.table[MAXP * 8 * (i) + 8 * (j)])

#define MODDOWN_MATRIX(i, j) (MAXP * (i) + (j))
#define MODUPIDX_SCALE(i, j, k) (MAXP * MAXP * (i) + MAXP * (j) + (k))
// #define MODUPIDX_MATRIX(i, j, k, l) (MAXD * MAXP * MAXP * (i) + MAXP * MAXP * (j) + MAXP * (k) + (l))
#define MODUPIDX_MATRIX(i, j, k, l) (MAXP * MAXP * (i) + MAXP * (k) + (l))
#define hC_ (host_constants)
#define hG_ (host_global)

#define ISU64(x) (constants.type & (((uint64_t)1) << (x)))
#define HISU64(x) (host_constants.type & (((uint64_t)1) << (x)))

uint64_t shoup_precomp(uint64_t val, int primeid, Constants& host_constants_);

template <typename Scheme> __global__ void printConstants();

template <typename Scheme>
std::pair<std::vector<Constants>, std::unique_ptr<Global>> SetupConstants(const std::vector<PrimeRecord>& q,
  const std::vector<std::vector<LimbRecord>>& meta,
  const std::vector<PrimeRecord>& p,
  const std::vector<LimbRecord>& smeta,
  const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
  const std::vector<std::vector<std::vector<LimbRecord>>>& DIGITmeta,
  const std::vector<std::vector<int>>& digitGPUid,
  const std::vector<int>& GPUid,
  const int N,
  const Scheme& parameters);

} // namespace FIDESlib
#endif // FIDESLIB_CONSTANTSGPU_CUH
