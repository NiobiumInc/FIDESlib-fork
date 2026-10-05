//
// CONCURRENT OPS ON ONE CryptoContext.
//
// What this pins down: ops issued from SEVERAL host threads, on DISTINCT ciphertexts of ONE
// context, produce bit-identical ciphertexts to the same ops issued serially from one thread.
// CKKS here is exact integer arithmetic in the RNS domain, so "bit-identical" is the right bar --
// concurrency changes only which host thread enqueues a kernel and which stream it lands on, never
// what the kernel computes.
//
// What it exercises, per lane: a rotation (hybrid key switch -> key_switch_aux/aux2, the digit
// streams), a ct x pt multiply with a rescale (moddown_aux, the top-limb rescale scratch), a
// ct x ct multiply with relinearization and a rescale (key_switch_aux again, plus the auxiliary
// polynomial pool through the degree-2 component), a Chebyshev series (a long dependent chain of
// all of the above) and a bootstrap (multMonomial's monomial cache, the precomputed linear
// transforms, and every one of the above at once).
//
// Every one of those is ONE object per context in the default mode; FIDESLIB_CONCURRENT_OPS=1
// gives each issuing thread its own (see src/CudaUtils.cuh, ContextData::OpScratch). THIS TEST
// REQUIRES THE CONCURRENT MODE and skips without it: the default mode is documented as one issuing
// thread per context, so running these lanes concurrently there is out of contract, not a failure.
//

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Checksum.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ConcurrentOps.hpp"
#include "CudaUtils.cuh"
#include "ParametrizedTest.cuh"

using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

namespace {

/// Exact comparison of two stored ciphertexts, limb by limb and coefficient by coefficient.
void ExpectRawEqual(const FIDESlib::CKKS::RawCipherText& a, const FIDESlib::CKKS::RawCipherText& b, const std::string& what) {
	ASSERT_EQ(a.numRes, b.numRes) << what << ": residue count";
	ASSERT_EQ(a.N, b.N) << what << ": ring dimension";
	ASSERT_EQ(a.NoiseLevel, b.NoiseLevel) << what << ": noise level";
	ASSERT_EQ(a.sub_0.size(), b.sub_0.size()) << what << ": c0 limbs";
	ASSERT_EQ(a.sub_1.size(), b.sub_1.size()) << what << ": c1 limbs";
	// COUNT the differences in BOTH components before failing (2026-10-03). The previous form
	// ASSERT_EQ'd coefficient by coefficient and RETURNED at the first c0 mismatch, so c1 was never
	// compared and no other coefficient was: "c0 limb 0 coefficient 0 wrong, c1 intact" -- which the
	// hunt reasoned from for two weeks -- was the shape of this function, not a measurement.
	size_t bad0 = 0, bad1 = 0, first0_l = 0, first0_i = 0;
	bool have0 = false;
	for (size_t l = 0; l < a.sub_0.size(); ++l) {
		ASSERT_EQ(a.sub_0[l].size(), b.sub_0[l].size()) << what << ": c0 limb " << l << " size";
		for (size_t i = 0; i < a.sub_0[l].size(); ++i)
			if (a.sub_0[l][i] != b.sub_0[l][i]) {
				if (!have0) { first0_l = l; first0_i = i; have0 = true; }
				++bad0;
			}
	}
	for (size_t l = 0; l < a.sub_1.size(); ++l) {
		ASSERT_EQ(a.sub_1[l].size(), b.sub_1[l].size()) << what << ": c1 limb " << l << " size";
		for (size_t i = 0; i < a.sub_1[l].size(); ++i)
			if (a.sub_1[l][i] != b.sub_1[l][i])
				++bad1;
	}
	size_t total0 = 0, total1 = 0;
	for (auto& l : a.sub_0) total0 += l.size();
	for (auto& l : a.sub_1) total1 += l.size();
	if (bad0 != 0 || bad1 != 0)
		printf("[rawdiff] %s: c0 %zu/%zu coefficients differ (first limb %zu coeff %zu), c1 %zu/%zu differ\n",
		       what.c_str(), bad0, total0, first0_l, first0_i, bad1, total1);
	ASSERT_EQ(bad0 + bad1, size_t{ 0 }) << what << ": c0 " << bad0 << "/" << total0 << " and c1 " << bad1 << "/"
	                                   << total1 << " coefficients differ";
}

/// Bring a ciphertext to the level a bootstrap is meant to start from.
///
/// Bootstrap gains levels; handed a full-depth ciphertext it has nothing to give back, and
/// ModRaise's own precondition is that at least one usable tower remains after the noise level is
/// accounted for (Bootstrap.cu ModRaise: getLevel() - NoiseLevel + 1 >= 1). Level 1 with noise
/// level 1 is exactly that boundary and is the depletion the GPU bootstrap tests use
/// (OpenFheInterfaceTests.cu: `GPUct1.dropToLevel(1)` before Bootstrap).
void DepleteForBootstrap(FIDESlib::CKKS::Ciphertext& ct) {
	if (ct.NoiseLevel == 2)
		ct.rescale();
	if (ct.getLevel() > 1)
		ct.dropToLevel(1);
}

struct RawDiff {
	bool shape_ok	   = true;
	size_t c0_limbs_bad = 0, c1_limbs_bad = 0, coeffs_bad = 0;
	size_t first_bad_limb = static_cast<size_t>(-1);
	size_t c0_limbs = 0, c1_limbs = 0;
	bool ok() const { return shape_ok && c0_limbs_bad == 0 && c1_limbs_bad == 0; }
};

RawDiff DiffRaw(const FIDESlib::CKKS::RawCipherText& a, const FIDESlib::CKKS::RawCipherText& b) {
	RawDiff d;
	d.c0_limbs = a.sub_0.size();
	d.c1_limbs = a.sub_1.size();
	if (a.numRes != b.numRes || a.N != b.N || a.NoiseLevel != b.NoiseLevel || a.sub_0.size() != b.sub_0.size() ||
		a.sub_1.size() != b.sub_1.size()) {
		d.shape_ok = false;
		return d;
	}
	auto cmp = [&](const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y, size_t& limbs_bad) {
		for (size_t l = 0; l < x.size(); ++l) {
			if (x[l].size() != y[l].size()) {
				d.shape_ok = false;
				return;
			}
			bool bad = false;
			for (size_t i = 0; i < x[l].size(); ++i)
				if (x[l][i] != y[l][i]) {
					bad = true;
					++d.coeffs_bad;
				}
			if (bad) {
				++limbs_bad;
				if (d.first_bad_limb == static_cast<size_t>(-1))
					d.first_bad_limb = l;
			}
		}
	};
	cmp(a.sub_0, b.sub_0, d.c0_limbs_bad);
	cmp(a.sub_1, b.sub_1, d.c1_limbs_bad);
	return d;
}

/// Everything a provenance dump needs about one lane's run, captured while the ciphertexts are
/// still alive. The Ciphertext objects are destroyed at the end of `lane_work`, which hands their
/// blocks back to the pool -- but the tracer's rings are keyed by DEVICE POINTER and outlive the
/// objects, so the pointers are all a dump needs (and the FREE that returned each block is itself
/// one of the events the dump will print).
struct LaneTrace {
	std::vector<fideslib::LimbBlockRef> in;	 ///< The lane's INPUT ciphertext, at construction.
	std::vector<fideslib::LimbBlockRef> out; ///< The lane's OUTPUT ciphertext, just before store.
};

/// Print the provenance of every block behind a wrong output, and of that lane's input, followed
/// by the tail of the global fence ring. Silent unless the tracer is on.
///
/// WHAT TO READ IT FOR. Each block's ring is a hand-off history: CUT, then alternating TAKE and
/// FREE with a lane, a thread, a stream and the fence event on each. The question a dump answers
/// is whether the lane that WROTE the wrong polynomial (the OUTPUT mark) took its block over a
/// fence that actually covered the previous owner's last READ -- or over one that named a
/// different stream, or none at all (`owner_stream=0 ev=0`, a drained block), or one whose record
/// another thread had moved in between (visible in the fence ring as a second record of the same
/// event pointer between this block's FREE and its TAKE).
void DumpMismatchProvenance(const std::string& what, int lane, const LaneTrace& t) {
	if (!fideslib::PoolTraceEnabled())
		return;
	std::cerr << "[trace] ================ MISMATCH " << what << " lane " << lane << " ================" << std::endl;
	const auto dump_set = [](const char* title, const std::vector<fideslib::LimbBlockRef>& refs) {
		std::cerr << "[trace] " << title << ": " << refs.size() << " limb block(s)" << std::endl;
		for (const fideslib::LimbBlockRef& r : refs) {
			std::cerr << "[trace] " << title << " c" << r.poly << " partition " << r.partition << " limb " << r.limb << " prime "
					  << r.primeid << " ptr " << r.data << " limb_stream " << r.stream << std::endl;
			const std::string owners = fideslib::PoolTraceOwnersLine(r.data);
			if (!owners.empty())
				std::cerr << "[trace]   " << owners << std::endl;
			fideslib::PoolTraceDumpFor(r.data, std::cerr);
		}
	};
	dump_set("OUTPUT(wrong)", t.out);
	dump_set("INPUT", t.in);
	fideslib::PoolTraceDumpRecent(std::cerr, 2000);
	std::cerr << "[trace] ================ end MISMATCH " << what << " lane " << lane << " ================" << std::endl;
	std::cerr.flush();
}

} // namespace

class ConcurrentOpsTest : public GeneralParametrizedTest {
  protected:
	/// Everything both tests below need: a clean context registry, a GPU context, the eval and
	/// rotation keys and the bootstrap precomputation. Returns the GPU context.
	FIDESlib::CKKS::Context Prepare(const std::vector<int>& rot_index, int& numSlots) {
		CKKS::DeregisterAllContexts();
		ClearCachedContexts();

		cc->Enable(lbcrypto::PKE);
		cc->Enable(lbcrypto::KEYSWITCH);
		cc->Enable(lbcrypto::LEVELEDSHE);
		cc->Enable(lbcrypto::ADVANCEDSHE);
		cc->Enable(lbcrypto::FHE);

		FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
		FIDESlib::CKKS::Context GPUcc_		= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
		FIDESlib::CKKS::ContextData& GPUcc	= *GPUcc_;

		numSlots = cc->GetRingDimension() / 2;
		keys	 = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalRotateKeyGen(keys.secretKey, rot_index);

		cc->EvalBootstrapSetup({ 3, 3 }, { 16, 16 }, numSlots, 0, true, false);
		cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
		FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

		{
			FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc_);
			FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
			kskEval.Initialize(rawKskEval);
			GPUcc.AddEvalKey(std::move(kskEval));
		}
		for (int idx : rot_index) {
			FIDESlib::CKKS::KeySwitchingKey kskRot(GPUcc_);
			FIDESlib::CKKS::RawKeySwitchKey rawKskRot = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, idx, cc);
			kskRot.Initialize(rawKskRot);
			GPUcc.AddRotationKey(idx, std::move(kskRot));
		}
		return GPUcc_;
	}

	void ClearCachedContexts() {
		for (auto& i : cached_cc) {
			i.second.first->ClearEvalAutomorphismKeys();
			i.second.first->ClearEvalMultKeys();
			if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
				std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
		}
	}
};

/// N threads x M ops, each lane on its own ciphertexts, against the same lanes run serially.
TEST_P(ConcurrentOpsTest, LanesMatchSerial) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	constexpr int kLanes = 4; // N threads
	const std::vector<int> rot_index{ 1, 2, 3, 4 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	FIDESlib::CKKS::Context GPUcc_		= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc	= *GPUcc_;

	const int numSlots = cc->GetRingDimension() / 2;
	keys			   = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, rot_index);

	cc->EvalBootstrapSetup({ 3, 3 }, { 16, 16 }, numSlots, 0, true, false);
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	// Eval-mult (relinearization) key.
	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		GPUcc.AddEvalKey(std::move(kskEval));
	}
	// Rotation keys.
	for (int idx : rot_index) {
		FIDESlib::CKKS::KeySwitchingKey kskRot(GPUcc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskRot = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, idx, cc);
		kskRot.Initialize(rawKskRot);
		GPUcc.AddRotationKey(idx, std::move(kskRot));
	}

	// Per-lane inputs, encoded at FULL DEPTH (level 0).
	//
	// THE LEVEL THESE USED TO BE ENCODED AT WAS A BUG, and it made every case throw before any
	// concurrency was reached ("vector::_M_range_check: __n (which is
	// 18446744073709551615) >= this->size() (which is 25)" in all nine cases, with the flag on and
	// therefore never seen while they only ever skipped). `GPUcc.L - 1` is the right level for a
	// ciphertext you are about to hand STRAIGHT to Bootstrap -- it is the idiom the bootstrap tests
	// in OpenFheInterfaceTests.cu use, because bootstrap has to start depleted or it gains nothing.
	// It is the wrong level for a ciphertext you first rotate, multiply twice and run a Chebyshev
	// series over: that leaves one tower, the first rescale drops to zero, and the next op indexes
	// a per-level table at -1 (which is the 2^64-1 above, and the 2^32-1 in the shared-operand
	// case: the same mistake reaching a differently-typed .at()).
	//
	// So: encode at level 0 and deplete deliberately with dropToLevel just before the bootstrap
	// (below), which is what makes BOTH halves of the lane real -- a full-depth chain of ops AND a
	// bootstrap that actually has work to do.
	//
	// Distinct values per lane so a lane that picked up another lane's intermediate would
	// differ in every coefficient, not just in the last bits.
	std::vector<FIDESlib::CKKS::RawCipherText> raw_in;
	std::vector<FIDESlib::CKKS::RawPlainText> raw_pt;
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> host_in;
	for (int lane = 0; lane < kLanes; ++lane) {
		std::vector<double> x(8);
		for (int i = 0; i < 8; ++i)
			x[i] = 0.1 * (lane + 1) * (i + 1);
		lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
		auto c				  = cc->Encrypt(keys.publicKey, p);
		host_in.push_back(c);
		raw_in.push_back(FIDESlib::CKKS::GetRawCipherText(cc, c));

		std::vector<double> y(8);
		for (int i = 0; i < 8; ++i)
			y[i] = 0.05 * (lane + 2) * (i + 2);
		lbcrypto::Plaintext q = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
		raw_pt.push_back(FIDESlib::CKKS::GetRawPlainText(cc, q));
	}

	// A short Chebyshev series (the coefficients themselves do not matter; what matters is the
	// dependent chain of mults/rescales/relinearizations it builds).
	const std::vector<double> cheb_coeffs{ 0.5, 0.25, 0.125, 0.0625, 0.03125 };

	// M ops on one lane, all on the lane's OWN ciphertexts.
	auto lane_work = [&](int lane, bool do_boot, FIDESlib::CKKS::RawCipherText& out) {
		FIDESlib::CKKS::Ciphertext ct(GPUcc_, raw_in[lane]);
		FIDESlib::CKKS::Ciphertext other(GPUcc_, raw_in[(lane + 1) % kLanes]);
		FIDESlib::CKKS::Plaintext pt(GPUcc_, raw_pt[lane]);

		ct.rotate(rot_index[lane % rot_index.size()]);		 // key switch
		ct.multPt(pt, true);								 // ct x pt + rescale
		ct.mult(other, true);								 // ct x ct + relin + rescale
		// Its own copy: evalChebyshevSeries takes the coefficients by non-const reference.
		std::vector<double> coeffs = cheb_coeffs;
		evalChebyshevSeries(ct, coeffs, -1.0, 1.0);			 // a dependent chain of the above
		if (do_boot) {
			DepleteForBootstrap(ct);						 // a bootstrap that really has work
			Bootstrap(ct, numSlots, false);					 // the whole machine at once
		}
		ct.store(out);
	};

	// ---- serial reference ----
	std::vector<FIDESlib::CKKS::RawCipherText> serial(kLanes);
	for (int lane = 0; lane < kLanes; ++lane)
		lane_work(lane, /*do_boot=*/true, serial[lane]);
	cudaDeviceSynchronize();

	// ---- the same lanes, concurrently ----
	std::vector<FIDESlib::CKKS::RawCipherText> parallel(kLanes);
	std::atomic<int> failures{ 0 };
	{
		std::vector<std::thread> pool;
		pool.reserve(kLanes);
		for (int lane = 0; lane < kLanes; ++lane) {
			pool.emplace_back([&, lane] {
				try {
					// M repetitions per thread, so the threads interleave for longer than one op.
					for (int rep = 0; rep < 3; ++rep)
						lane_work(lane, /*do_boot=*/true, parallel[lane]);
				} catch (const std::exception& e) {
					++failures;
					std::cerr << "lane " << lane << " threw: " << e.what() << std::endl;
				}
			});
		}
		for (auto& th : pool)
			th.join();
	}
	cudaDeviceSynchronize();
	ASSERT_EQ(failures.load(), 0) << "a lane threw under concurrency";

	for (int lane = 0; lane < kLanes; ++lane)
		ExpectRawEqual(serial[lane], parallel[lane], "lane " + std::to_string(lane));

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}


// ---------------------------------------------------------------------------------------------
// THE SAME, BUT WITH SHARED OPERANDS.
//
// LanesMatchSerial above gives every lane its OWN Ciphertext and its OWN Plaintext, so it can
// only find state that is shared because it lives on the CONTEXT (the op scratch). It cannot see
// a race on a VALUE that two threads both read, and that is the shape a real concurrent circuit
// has:
//
//   * ONE SOURCE CIPHERTEXT, read by several branches. Three projections of one activation is the
//     canonical case; each branch copies the source and then goes its own way. A copy is not
//     read-only on the device: LimbPartition::copyLimb fences the destination on the source
//     (LimbPartition.cu:855) and then BACK-FENCES the source on the destination
//     (LimbPartition.cu:870), so every reader WRITES to the shared source's Stream objects --
//     `this` is the shared one in that second fence, which is the direction Stream::wait's lock
//     used to leave unguarded.
//
//   * ONE PLAINTEXT, multiplied by several branches. Same fence shape through
//     LimbPartition::multPt / the linear-transform MAC, which fences the operand stream
//     (LimbPartitionBatch.cu:414).
//
//   * THE BOOTSTRAP PRECOMPUTATION, read by several concurrent bootstraps: the CtS/StC/LT
//     plaintexts are one set per context, and every bootstrap multiplies by them.
//
// A dropped fence here does not throw and does not corrupt memory: it lets a kernel read a buffer
// its producer has not finished writing, which shows up as a ciphertext that decrypts to noise.
// So the bar is the same as above -- bit-identical to the serial run.
TEST_P(ConcurrentOpsTest, SharedOperandsMatchSerial) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}

	constexpr int kLanes = 4;
	const std::vector<int> rot_index{ 1, 2, 3, 4 };
	int numSlots = 0;

	FIDESlib::CKKS::Context GPUcc_	   = Prepare(rot_index, numSlots);
	FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;

	// ONE source ciphertext and ONE plaintext for every lane.
	std::vector<double> x(8);
	for (int i = 0; i < 8; ++i)
		x[i] = 0.1 * (i + 1);
	lbcrypto::Plaintext p_src				= cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
	FIDESlib::CKKS::RawCipherText raw_src	= FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, p_src));

	std::vector<double> y(8);
	for (int i = 0; i < 8; ++i)
		y[i] = 0.05 * (i + 2);
	lbcrypto::Plaintext q				 = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
	FIDESlib::CKKS::RawPlainText raw_shared_pt = FIDESlib::CKKS::GetRawPlainText(cc, q);

	// The shared device objects, built ONCE on this thread and then only read by the lanes.
	FIDESlib::CKKS::Ciphertext shared_src(GPUcc_, raw_src);
	FIDESlib::CKKS::Plaintext shared_pt(GPUcc_, raw_shared_pt);
	cudaDeviceSynchronize();
	// Baseline readback of the shared source, so the "nobody wrote to it" check below compares
	// store() against store() and not against the load format.
	FIDESlib::CKKS::RawCipherText src_before;
	shared_src.store(src_before);
	cudaDeviceSynchronize();

	// One lane: copy the shared source, multiply by the shared plaintext, take a lane-specific
	// branch, then bootstrap -- which reads the shared precomputation. Nothing here writes to
	// either shared object, so every lane must get exactly what it gets on its own.
	auto lane_work = [&](int lane, FIDESlib::CKKS::RawCipherText& out) {
		FIDESlib::CKKS::Ciphertext ct(GPUcc_);
		ct.copy(shared_src);					   // shared SOURCE: forward fence + back fence
		ct.multPt(shared_pt, true);				   // shared PLAINTEXT
		ct.rotate(rot_index[lane % rot_index.size()]);
		ct.multScalar(1.0 + 0.25 * lane, true);	   // distinct per lane, so a mix-up is visible
		DepleteForBootstrap(ct);
		Bootstrap(ct, numSlots, false);			   // shared bootstrap precomputation
		ct.store(out);
	};

	// ---- serial reference ----
	std::vector<FIDESlib::CKKS::RawCipherText> serial(kLanes);
	for (int lane = 0; lane < kLanes; ++lane)
		lane_work(lane, serial[lane]);
	cudaDeviceSynchronize();

	// ---- the same lanes, concurrently, against the SAME shared objects ----
	std::vector<FIDESlib::CKKS::RawCipherText> parallel(kLanes);
	std::atomic<int> failures{ 0 };
	{
		std::vector<std::thread> pool;
		pool.reserve(kLanes);
		for (int lane = 0; lane < kLanes; ++lane) {
			pool.emplace_back([&, lane] {
				try {
					for (int rep = 0; rep < 3; ++rep)
						lane_work(lane, parallel[lane]);
				} catch (const std::exception& e) {
					++failures;
					std::cerr << "lane " << lane << " threw: " << e.what() << std::endl;
				}
			});
		}
		for (auto& th : pool)
			th.join();
	}
	cudaDeviceSynchronize();
	ASSERT_EQ(failures.load(), 0) << "a lane threw under concurrency";

	for (int lane = 0; lane < kLanes; ++lane)
		ExpectRawEqual(serial[lane], parallel[lane], "shared-operand lane " + std::to_string(lane));

	// The shared source must be untouched: read it back and compare against what it was loaded
	// with. A back-fence that wrote through a dangling record would not change these bytes, but a
	// lane that mistook the shared source for its own destination would.
	{
		FIDESlib::CKKS::RawCipherText src_now;
		shared_src.store(src_now);
		ExpectRawEqual(src_before, src_now, "shared source ciphertext");
	}

	CKKS::DeregisterAllContexts();
	ClearCachedContexts();
}


// ---------------------------------------------------------------------------------------------
// SEVERAL THREADS, EACH BOOTSTRAPPING.
//
// WHY THIS CASE EXISTS SEPARATELY. A bisection of the application that motivated the concurrent
// mode split its parallel regions cleanly: every region that
// issues only MATMULS -- linear transforms, hoisted rotations, key switches, rescales, the
// auxiliary-polynomial pool, the device memory pool -- reproduced the serial result exactly, and
// the one region that issues BOOTSTRAPS concurrently corrupted its output from the first block.
// So the defect class lives behind Bootstrap and nothing else, and it deserves a case that is
// nothing but concurrent bootstraps on one context: the mod raise, the sparse encapsulation
// switch, CoeffsToSlots/SlotsToCoeffs over the ONE shared precomputation, the monomial cache, the
// Chebyshev series and the double-angle iterations, several at a time.
//
// WHAT IT CANNOT REACH, and where the defect actually was. This is the RAW layer, and
// FIDESlib::CKKS::Bootstrap is single-iteration by construction. Two-iteration bootstrapping
// (Meta-BTS) is composed one layer up, in CudaEngine::evalBootstrap, and it is the only op on this
// backend that leaves the device: it reads the ciphertext back and runs its glue arithmetic
// through lbcrypto. That readback is what made the application's bootstrap region differ from its
// matmul regions, and the fix is there (api/Ciphertext.cpp hostShadowLock,
// CudaEngine.cpp g_metaBtsHostLock). This case is the CONTROL for it: if the single-iteration
// device path is clean under concurrency, a remaining failure is in the host composition, not in
// the kernels.
TEST_P(ConcurrentOpsTest, BootstrapLanesMatchSerial) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}

	constexpr int kLanes = 4;
	const std::vector<int> rot_index{ 1, 2, 3, 4 };
	int numSlots = 0;

	FIDESlib::CKKS::Context GPUcc_	   = Prepare(rot_index, numSlots);
	FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;
	(void)GPUcc;

	std::vector<FIDESlib::CKKS::RawCipherText> raw_in;
	std::vector<FIDESlib::CKKS::RawPlainText> raw_pt;
	for (int lane = 0; lane < kLanes; ++lane) {
		std::vector<double> x(8);
		for (int i = 0; i < 8; ++i)
			x[i] = 0.1 * (lane + 1) * (i + 1);
		lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
		raw_in.push_back(FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, p)));

		std::vector<double> y(8);
		for (int i = 0; i < 8; ++i)
			y[i] = 0.05 * (lane + 2) * (i + 2);
		lbcrypto::Plaintext q = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
		raw_pt.push_back(FIDESlib::CKKS::GetRawPlainText(cc, q));
	}

	// The shape the application's failing region has: an independent chain per lane in which a
	// bootstrap sits between two pieces of ordinary work, TWICE, so the lanes are inside the
	// bootstrap machinery at overlapping times rather than merely near it.
	// Provenance capture, written only by this lane's own thread. `in` is taken at construction
	// (the blocks the lane's input arrived on) and `out` immediately before the store, which is
	// the last moment the wrong polynomial's blocks are still named by a live object.
	std::vector<LaneTrace> lane_trace(kLanes);

	auto lane_work = [&](int lane, FIDESlib::CKKS::RawCipherText& out) {
		FIDESlib::CKKS::Ciphertext ct(GPUcc_, raw_in[lane]);
		FIDESlib::CKKS::Plaintext pt(GPUcc_, raw_pt[lane]);
		if (fideslib::PoolTraceEnabled())
			lane_trace[lane].in = fideslib::CiphertextLimbPointers(ct);
		for (int round = 0; round < 2; ++round) {
			ct.multPt(pt, true);
			DepleteForBootstrap(ct);
			Bootstrap(ct, numSlots, false);
			ct.rotate(rot_index[(lane + round) % rot_index.size()]);
		}
		if (fideslib::PoolTraceEnabled())
			lane_trace[lane].out = fideslib::CiphertextLimbPointers(ct);
		ct.store(out);
	};

	// ---------------------------------------------------------------------------------------
	// SHARED-OBJECT DIGEST (2026-10-03). Reads only; prints only; the verdict below is untouched.
	//
	// WHAT IT ANSWERS. The serial_after datum says the corruption is PERSISTENT: it survives four
	// thread joins and a cudaDeviceSynchronize and is still there when the main thread re-runs the
	// reference. Only two kinds of object can carry it -- the four rotation keys (lane-selective:
	// lane L uses rot_index[L] and rot_index[(L+1)%4], so lane 0 touches keys 1 and 2 only) and the
	// bootstrap precomputation diagonals (shared identically by all four lanes). A digest of both,
	// taken at three points, says WHICH, and says it by equality rather than by inference.
	//
	// HOW EACH OBJECT IS READ. A KeySwitchingKey is a and b, two RNSPolys built with def_stream and
	// never grown on one GPU, so their `limb` vector is EMPTY and RNSPoly::store (which is driven by
	// `level`, still -1) would read nothing: the key material is in DECOMPlimb / DIGITlimb, written
	// once by KeySwitchingKey::Initialize. So the digest walks the partitions itself and brings
	// every limb of all three vectors to the host with the public Limb::store_convert, via the
	// existing SWITCH macro. The diagonals go through Plaintext::store into a RawPlainText, which is
	// how every other test reads a plaintext back. No new method, no new kernel, no allocation on
	// the device, and the keys are reached through ContextData::precom (public) with find(), never
	// operator[], so the lookup cannot insert.
	auto fold = [](uint64_t& h, const std::vector<uint64_t>& w) {
		for (uint64_t x : w)
			for (int byte = 0; byte < 8; ++byte) {   // FNV-1a 64, byte at a time, little end first
				h ^= (x >> (8 * byte)) & 0xffull;
				h *= 1099511628211ull;
			}
	};
	auto key_digest = [&](FIDESlib::CKKS::KeySwitchingKey& k) {
		uint64_t h = 1469598103934665603ull;
		std::vector<uint64_t> w;
		for (FIDESlib::CKKS::RNSPoly* P : { &k.a, &k.b })
			for (auto& g : P->GPU) {
				for (auto& l : g.limb) {
					SWITCH(l, store_convert(w));
					fold(h, w);
				}
				for (auto& d : g.DECOMPlimb)
					for (auto& l : d) {
						SWITCH(l, store_convert(w));
						fold(h, w);
					}
				for (auto& d : g.DIGITlimb)
					for (auto& l : d) {
						SWITCH(l, store_convert(w));
						fold(h, w);
					}
			}
		return h;
	};
	auto diag_digest = [&]() {
		uint64_t h								   = 1469598103934665603ull;
		FIDESlib::CKKS::BootstrapPrecomputation& B = GPUcc.GetBootPrecomputation(numSlots, 0);
		FIDESlib::CKKS::RawPlainText r;
		auto fold_pt = [&](FIDESlib::CKKS::Plaintext& pt) {
			pt.store(r);
			for (auto& limb : r.sub_0)
				fold(h, limb);
		};
		for (auto* S : { &B.CtS, &B.StC })
			for (auto& st : *S)
				for (auto& pt : st.A)
					fold_pt(pt);
		for (auto* V : { &B.LT.A, &B.LT.invA })
			for (auto& pt : *V)
				fold_pt(pt);
		return h;
	};
	// The whole thing is inside a catch-all for one reason: it is a probe, and a probe must not be
	// able to decide a test. GetBootPrecomputation throws on a missing (slots, levelsToDrop) key,
	// and an escaping exception here would fail the case for a reason that has nothing to do with
	// what it measures.
	auto digest = [&](const char* phase) {
		try {
			cudaDeviceSynchronize();
			printf("[digest] %s", phase);
			for (size_t n = 0; n < rot_index.size(); ++n) {
				bool found = false;
				uint64_t h = 0;
				for (auto& kv : GPUcc.precom.keys) {
					const auto it = kv.second.rot_keys.find(rot_index[n]);
					if (it != kv.second.rot_keys.end()) {
						h	  = key_digest(it->second);
						found = true;
						break;
					}
				}
				if (found)
					printf(" key%zu=%016llx", n + 1, (unsigned long long)h);
				else
					printf(" key%zu=ABSENT", n + 1);
			}
			printf(" diagonals=%016llx\n", (unsigned long long)diag_digest());
			fflush(stdout);
		} catch (const std::exception& e) {
			printf(" DIGEST FAILED: %s\n", e.what());
			fflush(stdout);
		}
	};

	std::vector<FIDESlib::CKKS::RawCipherText> serial(kLanes);
	for (int lane = 0; lane < kLanes; ++lane)
		lane_work(lane, serial[lane]);
	cudaDeviceSynchronize();
	digest("serial");

	std::vector<FIDESlib::CKKS::RawCipherText> parallel(kLanes);
	// IDENTIFICATION (2026-10-03). A run measured that a failing lane differs in EVERY
	// coefficient of c0 AND c1: a different polynomial, not a corrupted one. Keep each rep's
	// output separately so that, on a mismatch, the wrong result can be matched against the
	// other lanes' serial outputs (a lane swap), the lane's own input (a no-op) and the lane's
	// other rep (stale), and the class is named by equality rather than argued.
	std::vector<std::vector<FIDESlib::CKKS::RawCipherText>> parallel_rep(kLanes, std::vector<FIDESlib::CKKS::RawCipherText>(2));
	std::atomic<int> failures{ 0 };
	{
		std::vector<std::thread> pool;
		pool.reserve(kLanes);
		for (int lane = 0; lane < kLanes; ++lane) {
			pool.emplace_back([&, lane] {
				try {
					for (int rep = 0; rep < 2; ++rep)
						lane_work(lane, parallel_rep[lane][rep]);
					parallel[lane] = parallel_rep[lane][1];   // the verdict below is unchanged: the last rep
				} catch (const std::exception& e) {
					++failures;
					std::cerr << "lane " << lane << " threw: " << e.what() << std::endl;
				}
			});
		}
		for (auto& th : pool)
			th.join();
	}
	cudaDeviceSynchronize();
	digest("concurrent");   // before the ASSERT: it returns on a throw, and the digest is the datum
	ASSERT_EQ(failures.load(), 0) << "a lane threw under concurrency";

	// DID THE SHARED STATE CHANGE (2026-10-03)? A wrong lane is wrong identically on both
	// reps, and the staged one_thread mode (a worker slot, no overlap) matches slot 0 -- so overlap
	// corrupts something PERSISTENT once and both reps read it. Re-run the serial reference on the
	// MAIN thread now, after the concurrent phase: if it reproduces the "wrong" answer, the object
	// whose state changed is shared with slot 0 (the precomputation plaintexts, a key, a stream);
	// if it still matches the first reference, the corruption is lane-private. Reads only.
	{
		std::vector<FIDESlib::CKKS::RawCipherText> serial_after(kLanes);
		for (int lane = 0; lane < kLanes; ++lane)
			lane_work(lane, serial_after[lane]);
		cudaDeviceSynchronize();
		digest("serial_after");
		// WHICH OF THE FOUR IS ACTUALLY RIGHT (2026-10-03)? Everything above is a COMPARISON. The
		// test has never decrypted: it takes the first serial pass for the truth and calls every
		// disagreement a corruption. In the latest run lane 2's two parallel reps AND serial_after
		// agree with each other and disagree with that first pass -- which is exactly the shape in
		// which the majority is right and the reference is the wrong answer. So decrypt all four
		// outputs of every lane and score each one against the cleartext the chain is supposed to
		// compute. Reads only, prints only, and wrapped in a catch-all: a probe must not be able to
		// turn a measured run into a crash, nor to decide the verdict below.
		try {
			// THE CONVERSION is the one every other test uses: a stored RawCipherText is poured
			// onto a FRESH, full-tower, level-0 OpenFHE ciphertext with
			// FIDESlib::CKKS::GetOpenFHECipherText (declared
			// src/CKKS/openfhe-interface/RawCiphertext.cuh:104, defined RawCiphertext.cu:96); the
			// holder supplies the DCRTPoly containers and the helper overwrites its elements,
			// level, scaling factor, noise degree, key tag and slot count from the raw. Same
			// pattern as OpenFheInterfaceTests.cu's bootstrap case (`auto cResGPU(c2);
			// GetOpenFHECipherText(cResGPU, raw_res1); cc->Decrypt(keys.secretKey, cResGPU, ...)`),
			// with a fresh holder per output so no two scores share an object. Default REV on both
			// sides is 1, so neither direction bit-reverses, which is how every call site runs.
			std::vector<double> zero(8, 0.0);
			lbcrypto::Plaintext tmpl_pt = cc->MakeCKKSPackedPlaintext(zero, 1, 0, nullptr, numSlots);

			// THE EXPECTATION, in doubles on the host. Same chain as lane_work, with the bootstrap
			// as the identity: slotwise multiply by y, rotate, twice. ROTATION CONVENTION: the GPU
			// Ciphertext::rotate(idx) is OpenFHE's EvalRotate(idx) -- OpenFheInterfaceTests.cu's
			// Rotate cases decrypt GPUct.rotate(k) against cc->EvalRotate(c1, k) for the SAME k,
			// k = 1..4 -- and EvalRotate's sign is the CKKS LEFT rotation, stated in
			// api/CryptoContext.hpp: "Rot(x, r) is the CKKS left rotation (slot i of the result is
			// slot i+r of the input, cyclic over the packed slots), i.e. EvalRotate's sign". So
			// new[j] = old[(j + idx) mod numSlots].
			auto expectation = [&](int lane) {
				std::vector<double> v(numSlots, 0.0), y(numSlots, 0.0), t(numSlots, 0.0);
				for (int i = 0; i < 8; ++i) {
					v[i] = 0.1 * (lane + 1) * (i + 1);
					y[i] = 0.05 * (lane + 2) * (i + 2);
				}
				for (int round = 0; round < 2; ++round) {
					for (int j = 0; j < numSlots; ++j)
						v[j] *= y[j];
					const int idx = rot_index[(lane + round) % rot_index.size()];
					for (int j = 0; j < numSlots; ++j)
						t[j] = v[(j + idx) % numSlots];
					v.swap(t);
				}
				return v;
			};

			// Max absolute error over the first 8 slots. -1 marks an output that was never written
			// (the same emptiness test the [rawwho] loop below uses); -2 marks one whose decrypt
			// THREW, which is the case this block exists for: in the three FIXEDMANUAL parameter
			// sets of an earlier run a single try/catch around the whole loop swallowed the first
			// Decode() failure, so not one per-lane line was printed and no other output of those
			// sets was ever read. One try/catch PER OUTPUT, so an undecodable output costs its own
			// field and nothing else.
			auto score = [&](const FIDESlib::CKKS::RawCipherText& raw, const std::vector<double>& want, std::string& err) {
				if (raw.sub_0.empty())
					return -1.0;
				try {
					lbcrypto::Ciphertext<lbcrypto::DCRTPoly> holder = cc->Encrypt(keys.publicKey, tmpl_pt);
					FIDESlib::CKKS::GetOpenFHECipherText(holder, raw);
					lbcrypto::Plaintext got;
					cc->Decrypt(keys.secretKey, holder, &got);
					const std::vector<double> v = got->GetRealPackedValue();
					double e					= 0.0;
					for (int i = 0; i < 8; ++i) {
						const double d	 = (i < (int)v.size() ? v[i] : 0.0) - want[i];
						const double mag = d < 0.0 ? -d : d;
						if (mag > e)
							e = mag;
					}
					return e;
				} catch (const std::exception& ex) {
					err = ex.what();
				} catch (...) {
					err = "unknown exception";
				}
				if (err.size() > 120)
					err.resize(120);
				return -2.0;
			};
			auto fmt = [](char* buf, double e) {
				if (e <= -2.0)
					snprintf(buf, 16, "FAIL");
				else if (e < 0.0)
					snprintf(buf, 16, "n/a");
				else
					snprintf(buf, 16, "%.3e", e);
				return buf;
			};
			// REPRESENTATION WITHOUT DECODING. The three RawCipherText fields GetOpenFHECipherText
			// reads to re-stamp the holder (RawCiphertext.cuh:29, 32, 33): `numRes` drives
			// SetLevel(holder_level + holder_towers - numRes), `Noise` is SetScalingFactor and
			// `NoiseLevel` is SetNoiseScaleDeg (RawCiphertext.cu:126, 170, 171). A Decode() that
			// throws is usually a disagreement about exactly these, so print them for every output
			// whether or not it decoded -- no decryption is involved. `level` is what the
			// conversion stamps on a FRESH level-0 holder, i.e. holder_towers - numRes.
			const int holder_towers = (int)cc->Encrypt(keys.publicKey, tmpl_pt)->GetElements().at(0).GetNumOfElements();
			auto meta				= [&](char* buf, const FIDESlib::CKKS::RawCipherText& raw) {
				   if (raw.sub_0.empty())
					   snprintf(buf, 96, "numRes=n/a level=n/a deg=n/a scale=n/a");
				   else
					   snprintf(buf, 96, "numRes=%d level=%d deg=%d scale=%.6g", raw.numRes, holder_towers - raw.numRes, raw.NoiseLevel, raw.Noise);
				   return buf;
			};
			static const char* const kPhase[4] = { "serial", "par0", "par1", "serial_after" };
			for (int lane = 0; lane < kLanes; ++lane) {
				const std::vector<double> want = expectation(lane);
				std::string why[4];
				const double e0 = score(serial[lane], want, why[0]);
				const double e1 = score(parallel_rep[lane][0], want, why[1]);
				const double e2 = score(parallel_rep[lane][1], want, why[2]);
				const double e3 = score(serial_after[lane], want, why[3]);
				char b0[16], b1[16], b2[16], b3[16];
				printf("[decrypt] BootstrapLanesMatchSerial lane %d: serial %s par0 %s par1 %s serial_after %s\n", lane,
				       fmt(b0, e0), fmt(b1, e1), fmt(b2, e2), fmt(b3, e3));
				const double scored[4] = { e0, e1, e2, e3 };
				for (int k = 0; k < 4; ++k)
					if (scored[k] <= -2.0)
						printf("[decrypt] lane %d phase %s: FAILED %s\n", lane, kPhase[k], why[k].c_str());
				char m0[96], m1[96], m2[96], m3[96];
				printf("[decrypt-meta] lane %d: serial (%s) par0 (%s) par1 (%s) serial_after (%s)\n", lane, meta(m0, serial[lane]),
				       meta(m1, parallel_rep[lane][0]), meta(m2, parallel_rep[lane][1]), meta(m3, serial_after[lane]));
			}
			fflush(stdout);
		} catch (const std::exception& e) {
			printf("[decrypt] FAILED: %s\n", e.what());
			fflush(stdout);
		} catch (...) {
			printf("[decrypt] FAILED: unknown exception\n");
			fflush(stdout);
		}
		for (int lane = 0; lane < kLanes; ++lane) {
			const bool par_ok	 = DiffRaw(serial[lane], parallel[lane]).ok();
			const bool after_ok	 = DiffRaw(serial[lane], serial_after[lane]).ok();
			const bool after_par = DiffRaw(serial_after[lane], parallel[lane]).ok();
			printf("[serialafter] BootstrapLanesMatchSerial lane %d: parallel==serial %d, serial_after==serial %d, serial_after==parallel %d\n",
			       lane, par_ok ? 1 : 0, after_ok ? 1 : 0, after_par ? 1 : 0);
		}
	}

	// PROVENANCE BEFORE THE ASSERT. ExpectRawEqual's ASSERT_EQ returns from this function on the
	// first wrong coefficient, so every dump has to be issued first -- and a dump per WRONG lane,
	// not one for the first, since a repetition can lose more than one. The pass/fail verdict
	// below is exactly the one this test always had.
	for (int lane = 0; lane < kLanes; ++lane) {
		const RawDiff d = DiffRaw(serial[lane], parallel[lane]);
		if (!d.ok())
			DumpMismatchProvenance("BootstrapLanesMatchSerial", lane, lane_trace[lane]);
	}
	// WHOSE POLYNOMIAL IS IT. For every rep of every lane that disagrees with its own serial
	// result, test exact equality against every candidate and print the verdict. Reads only.
	for (int lane = 0; lane < kLanes; ++lane) {
		for (int rep = 0; rep < 2; ++rep) {
			const auto& got = parallel_rep[lane][rep];
			if (got.sub_0.empty() || DiffRaw(serial[lane], got).ok())
				continue;
			std::string who;
			for (int o = 0; o < kLanes; ++o)
				if (o != lane && DiffRaw(serial[o], got).ok())
					who += " =serial[lane " + std::to_string(o) + "]";
			if (DiffRaw(raw_in[lane], got).ok())
				who += " =own-input";
			for (int o = 0; o < kLanes; ++o)
				if (o != lane && DiffRaw(raw_in[o], got).ok())
					who += " =input[lane " + std::to_string(o) + "]";
			const int other = 1 - rep;
			if (!parallel_rep[lane][other].sub_0.empty() && DiffRaw(parallel_rep[lane][other], got).ok())
				who += " =own-other-rep";
			for (int o = 0; o < kLanes; ++o)
				for (int r = 0; r < 2; ++r)
					if (o != lane && !parallel_rep[o][r].sub_0.empty() && DiffRaw(parallel_rep[o][r], got).ok())
						who += " =parallel[lane " + std::to_string(o) + " rep " + std::to_string(r) + "]";
			printf("[rawwho] BootstrapLanesMatchSerial lane %d rep %d: wrong;%s\n", lane, rep,
			       who.empty() ? " matches NOTHING (garbage)" : who.c_str());
		}
	}

	for (int lane = 0; lane < kLanes; ++lane)
		ExpectRawEqual(serial[lane], parallel[lane], "bootstrap lane " + std::to_string(lane));

	CKKS::DeregisterAllContexts();
	ClearCachedContexts();
}


// ---------------------------------------------------------------------------------------------
// STAGED: WHICH PRIMITIVE DIVERGES FIRST, AND IS IT OVERLAP OR PER-THREAD STATE?
//
// LanesMatchSerial reports one verdict per lane for the whole chain. When it fails, that says
// nothing about WHICH primitive produced the first wrong coefficient, nor whether the wrong value
// needs two threads to be live at once (an overlap defect: a missing stream dependency, a shared
// buffer) or merely a thread other than the one that built the context (a per-thread-state
// defect: scratch-slot plumbing, thread_local caches, the pool lane). This test answers both with
// one run, printing one `[staged]` line per (stage, mode, lane):
//
//   stage 1 rotate | 2 + multPt | 3 + mult | 4 + Chebyshev | 5 + bootstrap   (cumulative)
//   mode one_thread   -- the four lanes run back to back on ONE thread that is not the main one
//   mode serialized   -- four threads, but a mutex lets only one lane_work run at a time
//   mode concurrent   -- four threads, free-running (what LanesMatchSerial does)
//
// Reading: a mismatch already in one_thread/serialized is per-thread state; a mismatch only in
// concurrent is overlap. The first stage that mismatches names the primitive.
TEST_P(ConcurrentOpsTest, StagedLanesMatchSerial) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}
	constexpr int kLanes = 4;
	const std::vector<int> rot_index{ 1, 2, 3, 4 };
	int numSlots = 0;
	FIDESlib::CKKS::Context GPUcc_ = Prepare(rot_index, numSlots);

	std::vector<FIDESlib::CKKS::RawCipherText> raw_in;
	std::vector<FIDESlib::CKKS::RawPlainText> raw_pt;
	for (int lane = 0; lane < kLanes; ++lane) {
		std::vector<double> x(8);
		for (int i = 0; i < 8; ++i)
			x[i] = 0.1 * (lane + 1) * (i + 1);
		lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
		raw_in.push_back(FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, p)));
		std::vector<double> y(8);
		for (int i = 0; i < 8; ++i)
			y[i] = 0.05 * (lane + 2) * (i + 2);
		lbcrypto::Plaintext q = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
		raw_pt.push_back(FIDESlib::CKKS::GetRawPlainText(cc, q));
	}
	const std::vector<double> cheb_coeffs{ 0.5, 0.25, 0.125, 0.0625, 0.03125 };
	static const char* const kStageName[] = { "", "rotate", "multPt", "mult", "cheb", "boot" };
	constexpr int kStages				  = 5;

	// The first `upto` primitives of LanesMatchSerial's chain, on the lane's own ciphertexts.
	// `per_op` non-null: every primitive runs under that lock, so lanes interleave OP BY OP but no
	// two threads are ever inside a primitive at once (the composite ones -- the series and the
	// bootstrap -- count as one primitive here).
	// Provenance capture, one slot per lane, written only by that lane's own thread and
	// overwritten by each repetition -- so what a dump prints is the repetition that just failed.
	std::vector<LaneTrace> lane_trace(kLanes);

	auto lane_work = [&](int lane, int upto, FIDESlib::CKKS::RawCipherText& out, std::mutex* per_op) {
		auto guard = [&]() { return per_op ? std::unique_lock<std::mutex>(*per_op) : std::unique_lock<std::mutex>(); };
		FIDESlib::CKKS::Ciphertext ct(GPUcc_, raw_in[lane]);
		FIDESlib::CKKS::Ciphertext other(GPUcc_, raw_in[(lane + 1) % kLanes]);
		FIDESlib::CKKS::Plaintext pt(GPUcc_, raw_pt[lane]);
		if (fideslib::PoolTraceEnabled())
			lane_trace[lane].in = fideslib::CiphertextLimbPointers(ct);
		if (upto >= 1) {
			auto g = guard();
			ct.rotate(rot_index[lane % rot_index.size()]);
		}
		if (upto >= 2) {
			auto g = guard();
			ct.multPt(pt, true);
		}
		if (upto >= 3) {
			auto g = guard();
			ct.mult(other, true);
		}
		if (upto >= 4) {
			auto g					   = guard();
			std::vector<double> coeffs = cheb_coeffs;
			evalChebyshevSeries(ct, coeffs, -1.0, 1.0);
		}
		if (upto >= 5) {
			auto g = guard();
			DepleteForBootstrap(ct);
			Bootstrap(ct, numSlots, false);
			FIDESlib::CKKS::ChecksumProbe(ct, "post-boot"); // DIAG (FIDESLIB_CKSUM=1)
		}
		if (fideslib::PoolTraceEnabled())
			lane_trace[lane].out = fideslib::CiphertextLimbPointers(ct);
		ct.store(out);
		FIDESlib::CKKS::ChecksumProbe(ct, "post-store"); // DIAG (FIDESLIB_CKSUM=1): a digest that differs from post-boot means a write landed AFTER this lane's last kernel
		FIDESlib::CKKS::ChecksumFlush(); // DIAG (FIDESLIB_CKSUM=1): print this lane's per-sub-step digests
	};

	// one_thread / serialized: one repetition (they are controls). op_serialized / concurrent:
	// kReps repetitions, because the concurrent mismatch is intermittent for the shorter chains
	// (Stage 3 mismatched in one lane of one param and was clean on the next).
	enum Mode { kOneThread = 0, kSerialized = 1, kOpSerialized = 2, kConcurrent = 3 };
	static const char* const kModeName[] = { "one_thread", "serialized", "op_serialized", "concurrent" };
	constexpr int kReps					  = 8;

	bool all_ok = true;
	for (int stage = 1; stage <= kStages; ++stage) {
		std::vector<FIDESlib::CKKS::RawCipherText> serial(kLanes);
		for (int lane = 0; lane < kLanes; ++lane) {
			FIDESlib::CKKS::ChecksumLabel("stage=" + std::to_string(stage) + " mode=serial rep=0 lane=" + std::to_string(lane)); // DIAG (FIDESLIB_CKSUM=1)
			lane_work(lane, stage, serial[lane], nullptr);
		}
		cudaDeviceSynchronize();

		for (int mode = kOneThread; mode <= kConcurrent; ++mode) {
			const int reps = (mode == kOpSerialized || mode == kConcurrent) ? kReps : 1;
			std::vector<int> bad_reps(kLanes, 0);
			RawDiff worst[kLanes];
			int threw = 0;
			for (int rep = 0; rep < reps; ++rep) {
				std::vector<FIDESlib::CKKS::RawCipherText> parallel(kLanes);
				std::atomic<int> failures{ 0 };
				std::mutex gate;
				auto run_lane = [&](int lane) {
					FIDESlib::CKKS::ChecksumLabel("stage=" + std::to_string(stage) + " mode=" + kModeName[mode] + " rep=" + std::to_string(rep) + " lane=" + std::to_string(lane)); // DIAG (FIDESLIB_CKSUM=1)
					try {
						if (mode == kSerialized) {
							std::lock_guard<std::mutex> lk(gate);
							lane_work(lane, stage, parallel[lane], nullptr);
						} else if (mode == kOpSerialized) {
							lane_work(lane, stage, parallel[lane], &gate);
						} else {
							lane_work(lane, stage, parallel[lane], nullptr);
						}
					} catch (const std::exception& e) {
						++failures;
						std::cerr << "[staged] stage=" << kStageName[stage] << " mode=" << kModeName[mode] << " lane " << lane
								  << " threw: " << e.what() << std::endl;
					}
				};
				if (mode == kOneThread) {
					std::thread t([&] {
						for (int lane = 0; lane < kLanes; ++lane)
							run_lane(lane);
					});
					t.join();
				} else {
					std::vector<std::thread> pool;
					pool.reserve(kLanes);
					for (int lane = 0; lane < kLanes; ++lane)
						pool.emplace_back([&, lane] { run_lane(lane); });
					for (auto& th : pool)
						th.join();
				}
				cudaDeviceSynchronize();
				threw += failures.load();
				for (int lane = 0; lane < kLanes; ++lane) {
					const RawDiff d = DiffRaw(serial[lane], parallel[lane]);
					if (!d.ok()) {
						++bad_reps[lane];
						worst[lane] = d;
						// HERE, not after the loop over repetitions: `lane_trace` holds the
						// repetition that just ran, and the device has been synchronized and the
						// lanes joined, so the rings are quiescent and complete for THIS failure.
						DumpMismatchProvenance(std::string("StagedLanesMatchSerial stage=") + kStageName[stage] + " mode=" +
												 kModeName[mode] + " rep=" + std::to_string(rep),
											   lane, lane_trace[lane]);
					}
				}
			}
			for (int lane = 0; lane < kLanes; ++lane) {
				if (bad_reps[lane] == 0) {
					printf("[staged] stage=%d:%s mode=%s lane=%d OK reps=%d\n", stage, kStageName[stage], kModeName[mode], lane, reps);
				} else {
					all_ok			= false;
					const RawDiff& d = worst[lane];
					printf("[staged] stage=%d:%s mode=%s lane=%d MISMATCH bad_reps=%d/%d shape_ok=%d c0 %zu/%zu limbs c1 %zu/%zu limbs coeffs %zu first_bad_limb %zd\n",
						   stage, kStageName[stage], kModeName[mode], lane, bad_reps[lane], reps, d.shape_ok ? 1 : 0, d.c0_limbs_bad,
						   d.c0_limbs, d.c1_limbs_bad, d.c1_limbs, d.coeffs_bad,
						   d.first_bad_limb == static_cast<size_t>(-1) ? -1 : (ssize_t)d.first_bad_limb);
				}
			}
			if (threw != 0) {
				all_ok = false;
				printf("[staged] stage=%d:%s mode=%s: %d lane run(s) threw\n", stage, kStageName[stage], kModeName[mode], threw);
			}
			fflush(stdout);
		}
	}
	EXPECT_TRUE(all_ok) << "one or more [staged] lines report MISMATCH; the first mismatching stage/mode names the defect class";

	CKKS::DeregisterAllContexts();
	ClearCachedContexts();
}

// =================================================================================================
// PRODUCER / CONSUMER DEVICE-MEMORY POOL (the mechanism of an out-of-memory GPU run).
//
// In the concurrent mode a freed block goes onto the FREEING lane's list and only that lane takes
// it back. The application's shape is asymmetric: ~24 prefetch workers device-encode plaintexts on
// their lanes and ONE consumer thread frees them on its, so blocks flow producer-lane ->
// consumer-lane and, before the drain, never came back. The producers' fresh tier emptied forever,
// GPUmalloc cut another chunk every time it did, and the pool grew until the card was full --
// ~113k limb blocks (~58 GiB) per transformer block, two blocks into a 96 GB card.
//
// So this case reproduces that shape exactly and asserts the thing the drain owes: the CHUNK COUNT
// IS BOUNDED. It is a footprint assertion, not a timing one, and its control is arithmetic rather
// than a second run -- 32,768 allocations of a class that fits 1024 blocks to a chunk cut 32 chunks
// when nothing is ever reclaimed, and the bound below is 2 (derived in the case itself: the live
// set can never exceed 265 blocks, so a second chunk's worth of 1024 can never all be live at the
// instant a cut is decided). It also closes with an accounting identity, so a run that merely
// stopped CUTTING while LEAKING blocks onto the lanes does not pass.
//
// It also keeps the correctness shape of PoolStressTests (fill / spin / verify on the producer's
// own stream), because the drain's whole cost is a cudaDeviceSynchronize that makes reclaimed
// blocks quiescent: if that fence were dropped, the blocks the drain hands to a producer would
// still be live on the consumer's stream and the tag check is what would say so.
//
// CUDA-ONLY, and NOT RUNNABLE ON A LAPTOP. Run it on the runner as part of fideslib-test with
// FIDESLIB_CONCURRENT_OPS=1 (it skips otherwise -- the lane pool does not exist in the default
// mode) and FIDESLIB_SCRATCH_SLOTS at or above the thread count, so every producer gets its own
// lane instead of falling back to the shared lane 0, which would hide the asymmetry.
// =================================================================================================

namespace {

constexpr int kPoolDevice = 0;

/// GPUmalloc force-caches every request under 64 KB and cuts a chunk of (bytes/1024) MB for it,
/// which is 1024 blocks per chunk at ANY size in that range. 8 KB therefore keeps the whole case
/// inside 256 MB of device memory even in the failing case where nothing is ever reclaimed.
constexpr int kPoolBlockBytes	 = 8192;
constexpr size_t kPoolBlockWords = kPoolBlockBytes / sizeof(uint64_t);
constexpr size_t kBlocksPerChunk = 1024;

__global__ void drain_fill_(uint64_t* __restrict__ p, const size_t n, const uint64_t tag) {
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		p[i] = tag ^ (uint64_t)i;
}

__global__ void drain_check_(const uint64_t* __restrict__ p, const size_t n, const uint64_t tag, int* __restrict__ errors) {
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && p[i] != (tag ^ (uint64_t)i))
		atomicAdd(errors, 1);
}

/// Device-side delay, so a racing stream would land inside our window. A host sleep would not: the
/// hazard is device overlap.
__global__ void drain_spin_(const uint64_t iters, uint64_t* __restrict__ sink) {
	uint64_t x = 1;
	for (uint64_t i = 0; i < iters; ++i)
		x = x * 6364136223846793005ull + 1442695040888963407ull;
	if (threadIdx.x == 0 && blockIdx.x == 0 && x == 0)
		*sink = x; // never taken; keeps the loop alive
}

/// One size class out of a snapshot, zeroed if the class has not been touched yet.
FIDESlib::MemPoolSizeClassStats PoolClassRow(const FIDESlib::MemPoolStats& st, const int bytes) {
	for (const FIDESlib::MemPoolSizeClassStats& c : st.classes)
		if (c.bytes == bytes)
			return c;
	FIDESlib::MemPoolSizeClassStats zero;
	zero.bytes = bytes;
	return zero;
}

/// The hand-off between the producers and the one consumer. BOUNDED on purpose: the capacity is
/// what makes the number of blocks that can be live at once a known quantity, which is what turns
/// the chunk count into an assertion instead of an observation.
struct BlockQueue {
	std::mutex mu;
	std::condition_variable not_full, not_empty;
	std::deque<uint64_t*> q;
	size_t cap	 = 0;
	bool closed = false;

	/// False if the queue is closed and the block was NOT taken -- the consumer is gone and will
	/// never free it, so the caller must. `closed` is part of the wait predicate on purpose: a
	/// producer blocked on a full queue whose consumer has died would otherwise wait forever, and
	/// a test that hangs reports nothing at all.
	bool push(uint64_t* p) {
		std::unique_lock<std::mutex> lk(mu);
		not_full.wait(lk, [&] { return q.size() < cap || closed; });
		if (closed)
			return false;
		q.push_back(p);
		not_empty.notify_one();
		return true;
	}
	/// Returns nullptr once the producers are done and the queue has drained.
	uint64_t* pop() {
		std::unique_lock<std::mutex> lk(mu);
		not_empty.wait(lk, [&] { return !q.empty() || closed; });
		if (q.empty())
			return nullptr;
		uint64_t* p = q.front();
		q.pop_front();
		not_full.notify_one();
		return p;
	}
	/// Wakes BOTH waits. not_empty releases the consumer at the end of a normal run; not_full
	/// releases producers that a failed consumer would otherwise have stranded.
	void close() {
		std::lock_guard<std::mutex> lk(mu);
		closed = true;
		not_empty.notify_all();
		not_full.notify_all();
	}
};

} // namespace

TEST(ConcurrentOpsPoolDrainTest, ProducerConsumerBoundsChunkCount) {
	if (!FIDESlib::ConcurrentOps())
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1: the lane pool this case is about does not exist in the default mode";
	int devices = 0;
	if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
		GTEST_SKIP() << "no CUDA device";
	// The drain's device guard declines to run when more than one GPU is visible, because a peer
	// kernel could be reading a block through P2P and cudaDeviceSynchronize does not cover another
	// card's streams (see drainFreedLanes). On such a host the allocator is deliberately back to
	// its pre-drain behaviour, so the bound below does not apply and this case has nothing to say.
	// It is a skip and not a relaxed bound: the point of the case is the drain, and a run in which
	// the drain never fires would assert nothing while looking like a pass.
	if (devices > 1)
		GTEST_SKIP() << "more than one GPU visible: the drain's device guard disables it (P2P readers are outside "
						"cudaDeviceSynchronize), so the chunk bound this case asserts does not hold";
	cudaSetDevice(kPoolDevice);
	FIDESlib::initGPUprop();

	constexpr int kProducers   = 8;
	constexpr int kPerProducer = 4096; ///< 32,768 blocks = 32 chunks if a block is never reclaimed.
	constexpr size_t kQueueCap = 256;

	// ---- WHY THE BOUND IS 2 ----------------------------------------------------------------
	//
	// LIVE CEILING. A block of this class is "live" between its GPUmalloc and its GPUfree. At any
	// instant that is: at most kQueueCap in the queue, at most one per producer (allocated, not
	// yet pushed), and at most one in the consumer's hand (popped, not yet freed). Nothing else
	// holds one -- producers never free, and the consumer frees immediately.
	constexpr size_t kLiveCeiling = kQueueCap + kProducers + 1; // 256 + 8 + 1 = 265
	static_assert(kLiveCeiling < kBlocksPerChunk, "one chunk has to be able to cover the live set");
	//
	// CUT CONDITION. GPUmalloc cuts a chunk only when, under mempool_lock: the taker's own freed
	// list is empty, the fresh tier is empty, AND the drain left the fresh tier empty. The drain
	// moves EVERY lane's freed blocks of the class, so "still empty afterwards" means every lane's
	// freed list was empty too. Those four lists are the only places a non-live block of the class
	// can be. So at the instant of a cut, every block the pool has ever created for this class is
	// LIVE.
	//
	// THEREFORE. After the first chunk there are kBlocksPerChunk = 1024 blocks. A second cut would
	// require all 1024 to be live at once, and the ceiling is 265. The derivation gives 1.
	//
	// The assertion is 2, i.e. one chunk of slack, because the derivation assumes this case is the
	// only user of the 8 KB class in the process; another case that had left a block of it live
	// would shift the count by at most the chunk that block sits in. Anything above 2 is the
	// growth this branch exists to stop (the unfixed allocator cuts 32 here).
	constexpr unsigned long long kChunkBound = 2;

	int* d_errors	= nullptr;
	uint64_t* d_sink = nullptr;
	ASSERT_EQ(cudaMalloc(&d_errors, sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaMemset(d_errors, 0, sizeof(int)), cudaSuccess);
	ASSERT_EQ(cudaMalloc(&d_sink, sizeof(uint64_t)), cudaSuccess);
	ASSERT_EQ(cudaMemset(d_sink, 0, sizeof(uint64_t)), cudaSuccess);

	const FIDESlib::MemPoolSizeClassStats before =
		PoolClassRow(FIDESlib::MemPoolStatsSnapshot(kPoolDevice), kPoolBlockBytes);

	BlockQueue queue;
	queue.cap = kQueueCap;
	std::atomic<int> nulls{ 0 };
	std::atomic<int> threw{ 0 };

	const dim3 block{ 256 };
	const dim3 grid{ (uint32_t)((kPoolBlockWords + 255) / 256) };

	// --- producers: each on its OWN stream and therefore its own pool lane ---
	std::vector<std::thread> producers;
	producers.reserve(kProducers);
	for (int t = 0; t < kProducers; ++t) {
		producers.emplace_back([&, t] {
			cudaSetDevice(kPoolDevice);
			cudaStream_t stream = nullptr;
			if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
				threw.fetch_add(1);
				return;
			}
			for (int i = 0; i < kPerProducer; ++i) {
				auto* p = (uint64_t*)FIDESlib::GPUmalloc(kPoolDevice, kPoolBlockBytes, stream, true);
				if (p == nullptr) {
					nulls.fetch_add(1);
					continue;
				}
				const uint64_t tag = ((uint64_t)t << 40) ^ ((uint64_t)i << 8) ^ 0x5bd1e995ull;
				drain_fill_<<<grid, block, 0, stream>>>(p, kPoolBlockWords, tag);
				drain_spin_<<<1, 32, 0, stream>>>(1000, d_sink);
				// Verified on OUR stream, while the block is still ours: a nonzero count means the
				// pool handed this block to somebody else before our work on it finished.
				drain_check_<<<grid, block, 0, stream>>>(p, kPoolBlockWords, tag, d_errors);
				if (!queue.push(p)) {
					// The consumer is gone (its stream failed); the test has already failed on
					// `threw`. Free the block ourselves so the pool's accounting stays whole and
					// the identity assertion below still means something, then stop.
					FIDESlib::GPUfree(p, kPoolDevice, kPoolBlockBytes, stream, true);
					break;
				}
			}
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
		});
	}

	// --- consumer: ONE thread, ONE lane, frees everything the producers allocated ---
	std::thread consumer([&] {
		cudaSetDevice(kPoolDevice);
		cudaStream_t stream = nullptr;
		if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
			// CLOSE BEFORE RETURNING. Without this the producers fill the queue, block in push()
			// waiting for a consumer that no longer exists, and the test hangs instead of failing
			// -- a hang in CI is a timeout with no diagnosis, whereas closing here lets every
			// producer finish and the EXPECT on `threw` below say exactly what went wrong.
			threw.fetch_add(1);
			queue.close();
			return;
		}
		for (;;) {
			uint64_t* p = queue.pop();
			if (p == nullptr)
				break;
			FIDESlib::GPUfree(p, kPoolDevice, kPoolBlockBytes, stream, true);
		}
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	});

	for (auto& th : producers)
		th.join();
	queue.close();
	consumer.join();
	cudaDeviceSynchronize();

	const FIDESlib::MemPoolStats after_all		= FIDESlib::MemPoolStatsSnapshot(kPoolDevice);
	const FIDESlib::MemPoolSizeClassStats after = PoolClassRow(after_all, kPoolBlockBytes);
	const unsigned long long chunks				= after.chunks_cut - before.chunks_cut;
	const unsigned long long drains				= after.drains - before.drains;
	const unsigned long long reclaimed			= after.blocks_reclaimed - before.blocks_reclaimed;
	const unsigned long long skips				= after.drain_skips - before.drain_skips;

	int h_errors = -1;
	ASSERT_EQ(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

	printf("[pool-drain] class=%d B allocations=%d chunks=%llu drains=%llu reclaimed=%llu skips=%llu errors=%d\n",
		   kPoolBlockBytes, kProducers * kPerProducer, chunks, drains, reclaimed, skips, h_errors);
	printf("[pool-drain] held: fresh=%zu (no-fence %zu) lane-freed=%zu\n", after.fresh_blocks,
		   after.fresh_no_fence_blocks, after.lane_freed_blocks);
	printf("[pool-drain] %s\n", FIDESlib::MemPoolStatsLine(kPoolDevice).c_str());
	fflush(stdout);

	EXPECT_EQ(threw.load(), 0) << "a worker could not create its stream";
	EXPECT_EQ(nulls.load(), 0) << "the pool failed to serve an allocation";
	EXPECT_EQ(h_errors, 0) << "a pooled block was handed to a new owner while its previous owner was still using it";
	EXPECT_GT(drains, 0ull) << "the fresh tier never emptied, so this run did not exercise the drain at all";
	EXPECT_GT(reclaimed, 0ull) << "the drain fired but reclaimed nothing: the consumer's lane list was not the one drained";
	// Single device, and every thread here calls cudaSetDevice(kPoolDevice) first, so the guard
	// has no reason to fire. If it did, the chunk bound below is measuring the pre-drain allocator
	// and this is the line that explains why.
	EXPECT_EQ(skips, 0ull) << "the drain's device guard declined: a worker was not on device " << kPoolDevice;
	EXPECT_LE(chunks, kChunkBound) << "the pool kept cutting chunks instead of reclaiming the consumer's lane -- "
									  "this is the chunk growth seen before the drain existed";

	// ---- CLOSING IDENTITY: every block ever cut is still accounted for ----------------------
	//
	// Both threads have joined and the consumer freed everything it popped, so no block of this
	// class is live. A block of the class is then in exactly one of two places: the shared fresh
	// tier, or some lane's freed list. Nothing needs adding for the drain's sentinel -- a
	// reclaimed block sits IN the fresh tier and is counted in `fresh_blocks`; the sentinel only
	// changes its `lane` tag, and `lane_freed_blocks` is summed over the freed lists, which
	// GPUfree always tags with a real lane. So:
	//
	//     fresh_blocks + lane_freed_blocks == chunks_cut * kBlocksPerChunk
	//
	// A shortfall is a leak (a block handed out and never returned to the pool); a surplus is
	// double-counting (one block on two lists). These are ABSOLUTE counts, not deltas, so the
	// identity covers the whole process. The precondition is that the same held before this case
	// ran: if an earlier case in this binary leaked a block of the 8 KB class, THAT expectation is
	// the one that fires and the attribution stays honest.
	EXPECT_EQ(before.fresh_blocks + before.lane_freed_blocks, (size_t)before.chunks_cut * kBlocksPerChunk)
		<< "the 8 KB class was already unbalanced before this case ran; the identity below is not this case's to fail";
	EXPECT_EQ(after.fresh_blocks + after.lane_freed_blocks, (size_t)after.chunks_cut * kBlocksPerChunk)
		<< "blocks of the 8 KB class went missing: cut " << after.chunks_cut * kBlocksPerChunk << ", holding "
		<< after.fresh_blocks << " fresh + " << after.lane_freed_blocks << " on lanes";
	EXPECT_LE(after.fresh_no_fence_blocks, after.fresh_blocks)
		<< "the no-fence count is a subset of the fresh tier, not a separate population";

	cudaFree(d_errors);
	cudaFree(d_sink);
}

// =================================================================================================
// THE PASS-BOUNDARY RELEASE (the mechanism of an out-of-memory GPU run).
//
// The case above is about the pool's own drift: blocks freed on one lane that only that lane can
// take back. This one is about the drift ABOVE the pool, which no drain can touch. A Ciphertext
// draws two auxiliary polynomials in its constructor and returns them in its destructor, and the
// list is the CALLING THREAD'S slot's. The application's shape is a parallel region: sub-tasks are
// issued on pool WORKER threads and their results are merged on the PASS thread, so intermediates
// are born on a worker and die on the pass thread. The pass thread's list grows every block, the
// workers' lists stay empty and keep CONSTRUCTING, and every parked polynomial is LIVE pool memory
// -- on no freed list, so no drain can reclaim it. At application scale that ran a 96 GB card out of
// memory in the third generation pass.
//
// So this case reproduces THAT shape -- workers construct, the pass thread destroys -- and asserts
// what ContextData::releaseParkedScratch owes at a pass boundary:
//   1. the parked polynomials are gone (the count drops to zero, not merely down);
//   2. their blocks reached the SHARED FRESH TIER rather than one lane's private freed list, and
//      the lanes are empty afterwards;
//   3. an allocation burst of the same shape then cuts NO NEW CHUNK -- i.e. the released memory is
//      genuinely reusable and not just re-labelled.
// (3) is the assertion that matters: (1) and (2) could both pass on a release that handed back
// memory the allocator cannot draw from.
//
// CUDA-ONLY. Run it with FIDESLIB_CONCURRENT_OPS=1 (it skips otherwise -- in the default mode
// there is one list, one thread and nothing to drift) and FIDESLIB_SCRATCH_SLOTS above the worker
// count, so every worker gets its own slot rather than sharing slot 0, which is what makes the
// drift visible at all.
// =================================================================================================

namespace {

/// The parked-polynomial totals a boundary assertion needs, from ONE pool snapshot.
struct PoolHeld {
	size_t fresh	 = 0;
	size_t no_fence	 = 0;
	size_t lane_held = 0;
};

PoolHeld PoolHeldFrom(const FIDESlib::MemPoolStats& st) {
	PoolHeld h;
	for (const FIDESlib::MemPoolSizeClassStats& c : st.classes) {
		h.fresh += c.fresh_blocks;
		h.no_fence += c.fresh_no_fence_blocks;
	}
	for (const size_t n : st.lane_freed_blocks)
		h.lane_held += n;
	return h;
}

} // namespace

TEST_P(ConcurrentOpsTest, PassBoundaryReleaseReturnsParkedScratch) {
	if (!FIDESlib::ConcurrentOps())
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1: there is no per-slot list to drift in the default mode";
	int visible = 0;
	if (cudaGetDeviceCount(&visible) != cudaSuccess || visible == 0)
		GTEST_SKIP() << "no CUDA device";
	// Same guard as the drain's, and the same reasoning: cudaDeviceSynchronize does not cover a
	// peer card's streams, so MemPoolReleaseAllLanes declines on a multi-GPU host and counts a
	// boundary skip. The polynomial half would still work, but the block assertions below would be
	// asserting the pre-release allocator. A skip says that; a relaxed bound would not.
	if (visible > 1)
		GTEST_SKIP() << "more than one GPU visible: the boundary release's device guard disables it";

	int numSlots					   = 0;
	FIDESlib::CKKS::Context GPUcc_	   = Prepare({ 1 }, numSlots);
	FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;
	ASSERT_EQ(GPUcc.GPUid.size(), 1u) << "this case is single-device by construction";
	const int dev = GPUcc.GPUid[0];
	cudaSetDevice(dev);

	// Prepare() issues ops on THIS thread (the key-switching keys, the bootstrap precomputation),
	// so slot 0 may already be holding polynomials that have nothing to do with the drift below.
	// Every count that follows is a DELTA against this, which is also why the release is expected
	// to hand back more than the drift created: it hands back everything parked, not only ours.
	const size_t parked_baseline = GPUcc.auxPolyParkedCount();

	constexpr int kWorkers	  = 3;
	constexpr int kPerWorker  = 8;
	constexpr size_t kPolysPerCt = 2; ///< c0 and c1; no op here raises a ciphertext to degree 2.

	// The hand-off. Bounded like the drain case's queue, for the same reason: what is IN FLIGHT has
	// to be a known quantity for the counts below to be exact rather than approximate.
	std::mutex qm;
	std::condition_variable q_not_empty, q_not_full;
	std::deque<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> q;
	constexpr size_t kQueueCap = 4;
	int producers_live		   = kWorkers;

	std::atomic<int> threw{ 0 };

	// THE GATE. Slots are recycled LIFO at thread exit (CudaUtils.cu: SlotHolder::~SlotHolder
	// pushes the slot onto `freed`, ScratchSlot pops the back of it), and a slot carries its
	// parked polynomials with it. So a worker that exits before another worker has first asked for
	// scratch hands that worker its slot WITH two polynomials still on the list, and the new
	// owner's next Ciphertext constructor draws them straight back out. That is the counting the
	// /4 and /5 shapes measured: 52 parked instead of 54, 3 hoarding slots instead of 4 -- not
	// drift the release failed to reach, but two polynomials that were not parked at the moment of
	// measuring, on a slot that had been handed on. The expectations below are the right ones; the
	// race was the test's.
	//
	// So no worker exits until the pass thread has measured AND released: each parks its
	// polynomials, says so, and blocks here holding its own slot.
	std::mutex gate_m;
	std::condition_variable gate_cv, all_parked;
	bool gate_open	   = false;
	int parked_workers = 0;

	std::vector<std::thread> workers;
	workers.reserve(kWorkers);
	for (int t = 0; t < kWorkers; ++t) {
		workers.emplace_back([&] {
			cudaSetDevice(dev);
			try {
				for (int i = 0; i < kPerWorker; ++i) {
					// Constructed HERE, on this worker's slot: its list is empty, so both
					// polynomials are CONSTRUCTED rather than reused, which is the starvation half
					// of the drift.
					auto ct = std::make_unique<FIDESlib::CKKS::Ciphertext>(GPUcc_);
					std::unique_lock<std::mutex> lk(qm);
					q_not_full.wait(lk, [&] { return q.size() < kQueueCap; });
					q.push_back(std::move(ct));
					q_not_empty.notify_one();
				}
				// ...and ONE that this worker destroys itself, so its own slot list is not empty
				// either. Without it every parked polynomial would sit on slot 0 and the case
				// would not show that the release walks the OTHER slots -- which is the whole
				// reason trimAuxilarPoly (the calling thread's list only) was not the fix.
				{ FIDESlib::CKKS::Ciphertext local(GPUcc_); }
			} catch (...) {
				threw.fetch_add(1);
			}
			{
				std::lock_guard<std::mutex> lk(qm);
				--producers_live;
			}
			q_not_empty.notify_all();
			// Parked. Hold this slot -- and everything on its list -- until the gate opens.
			{
				std::unique_lock<std::mutex> gl(gate_m);
				++parked_workers;
				all_parked.notify_all();
				gate_cv.wait(gl, [&] { return gate_open; });
			}
		});
	}

	// Any early return from here on (a failed ASSERT) must still open the gate and join, or the
	// worker threads would be destroyed joinable and the run would abort instead of reporting.
	struct GateGuard {
		std::mutex& m;
		std::condition_variable& cv;
		bool& open;
		std::vector<std::thread>& ts;
		~GateGuard() {
			{
				std::lock_guard<std::mutex> lk(m);
				open = true;
			}
			cv.notify_all();
			for (auto& t : ts)
				if (t.joinable())
					t.join();
		}
	} gate_guard{ gate_m, gate_cv, gate_open, workers };

	// The PASS THREAD: destroys everything the workers made. Every destructor parks two
	// polynomials on slot 0's list, which is the hoarding half of the drift.
	int destroyed = 0;
	for (;;) {
		std::unique_ptr<FIDESlib::CKKS::Ciphertext> ct;
		{
			std::unique_lock<std::mutex> lk(qm);
			q_not_empty.wait(lk, [&] { return !q.empty() || producers_live == 0; });
			if (q.empty())
				break;
			ct = std::move(q.front());
			q.pop_front();
			q_not_full.notify_one();
		}
		ct.reset();
		++destroyed;
	}
	// Every worker has parked its two polynomials and is now waiting at the gate, still holding its
	// slot: 1 + kWorkers hoarding slots, none of them recycled.
	{
		std::unique_lock<std::mutex> gl(gate_m);
		all_parked.wait(gl, [&] { return parked_workers == kWorkers; });
	}
	ASSERT_EQ(destroyed, kWorkers * kPerWorker) << "the pass thread did not receive every ciphertext";
	cudaDeviceSynchronize();

	// ---- the drift, measured -----------------------------------------------------------------
	// kWorkers * kPerWorker ciphertexts died on the pass thread and one more died on each worker,
	// two polynomials each, and nothing has drawn one back out. So the parked total is exact.
	const size_t expected_parked = kPolysPerCt * (size_t)(kWorkers * kPerWorker + kWorkers);
	const size_t parked_before	 = GPUcc.auxPolyParkedCount();
	ASSERT_GE(parked_before, parked_baseline) << "the parked total fell while nothing was drawing polynomials out";
	EXPECT_EQ(parked_before - parked_baseline, expected_parked)
		<< "the drift did not form: every ciphertext's polynomials should be parked on the list of the thread that "
		   "DESTROYED it";
	// THE ASYMMETRY ITSELF, which is what the measured run shows and what a total cannot: the
	// thread that FREES holds the mass. Slot 0 IS `precom.auxPoly` (ContextData::auxPolyList), and
	// this thread is the one that destroyed every ciphertext the workers made, so its list carries
	// two polynomials per ciphertext while each worker's carries two in total. A GPU run
	// measured the same shape at application scale: slot lists 0:1664, 1:44, 2:50, 3:50, 4:50 -- ~25
	// GiB parked on the pass thread against ~50 polynomials on each worker.
	const size_t slot0_before = GPUcc.precom.auxPoly.size();
	EXPECT_GE(slot0_before, kPolysPerCt * (size_t)(kWorkers * kPerWorker))
		<< "the pass thread's own list is not where the polynomials it destroyed went";
	EXPECT_GT(slot0_before, parked_before - slot0_before)
		<< "the pass thread should be hoarding MORE than every worker slot put together";

	const FIDESlib::MemPoolStats pool_before = FIDESlib::MemPoolStatsSnapshot(dev);
	const PoolHeld held_before				 = PoolHeldFrom(pool_before);

	// ---- THE PASS BOUNDARY --------------------------------------------------------------------
	const FIDESlib::CKKS::ContextData::ScratchRelease rel = GPUcc.releaseParkedScratch();

	const FIDESlib::MemPoolStats pool_after = FIDESlib::MemPoolStatsSnapshot(dev);
	const PoolHeld held_after				= PoolHeldFrom(pool_after);

	// Measured and released, so the workers may exit and give their slots back. Nothing they do on
	// the way out touches the pool -- their lists are empty and a slot release only returns an
	// index -- so the snapshots above still describe the state every expectation below reads.
	{
		std::lock_guard<std::mutex> gl(gate_m);
		gate_open = true;
	}
	gate_cv.notify_all();
	for (auto& th : workers)
		th.join();
	ASSERT_EQ(threw.load(), 0) << "a worker threw while building ciphertexts";

	printf("[pass-boundary] parked %zu -> %zu, released %zu polys from %zu slots, moved %zu blocks\n", parked_before,
		   GPUcc.auxPolyParkedCount(), rel.polys_released, rel.slots_drained, rel.blocks_moved);
	printf("[pass-boundary] held: fresh %zu -> %zu (no-fence %zu -> %zu), lane-held %zu -> %zu\n", held_before.fresh,
		   held_after.fresh, held_before.no_fence, held_after.no_fence, held_before.lane_held, held_after.lane_held);
	printf("[pass-boundary] %s\n", FIDESlib::MemPoolStatsLine(dev).c_str());
	fflush(stdout);

	// 1. the polynomials are gone, and the release says so in the same numbers.
	EXPECT_TRUE(rel.ran) << "the release declined in a run that has the concurrent mode on";
	EXPECT_EQ(rel.polys_released, parked_before);
	EXPECT_EQ(GPUcc.auxPolyParkedCount(), 0u) << "a slot list still holds parked polynomials after the boundary";
	EXPECT_TRUE(GPUcc.precom.auxPoly.empty())
		<< "the PASS THREAD'S slot -- the one holding the mass -- still has " << GPUcc.precom.auxPoly.size()
		<< " polynomials parked";
	// Slot 0 plus every worker's slot: the drift is ACROSS slots, and a release that emptied only
	// the caller's list would pass every other assertion here.
	EXPECT_EQ(rel.slots_drained, (size_t)(1 + kWorkers))
		<< "the release did not reach every hoarding slot (slot 0 plus one per worker)";

	// 2. their blocks reached the SHARED fresh tier, and no lane is still holding any.
	EXPECT_GT(rel.blocks_moved, 0u) << "no pooled block was moved: the polynomials' limbs did not reach the pool";
	EXPECT_EQ(pool_after.boundary_releases, pool_before.boundary_releases + 1);
	EXPECT_EQ(pool_after.boundary_skips, pool_before.boundary_skips) << "the release's device guard declined";
	EXPECT_EQ(pool_after.boundary_blocks_moved - pool_before.boundary_blocks_moved, (unsigned long long)rel.blocks_moved)
		<< "the context's count of moved blocks and the pool's own disagree";
	EXPECT_EQ(held_after.lane_held, 0u) << "a lane is still holding freed blocks after a release that claims to move every one";
	// The polynomials were destroyed BEFORE the lanes were swept (that ordering is the point), so
	// their blocks are inside `blocks_moved` and the fresh tier grew by exactly that many.
	EXPECT_EQ(held_after.fresh, held_before.fresh + rel.blocks_moved)
		<< "the fresh tier did not gain the blocks the lanes gave up";
	EXPECT_GE(held_after.no_fence, rel.blocks_moved)
		<< "a moved block was not tagged kNoFenceLane, so its next owner will fence on a lane that did not free it";
	EXPECT_LE(held_after.no_fence, held_after.fresh) << "the no-fence count is a subset of the fresh tier";

	// 3. and the memory is REUSABLE: the same burst again, on this thread, cuts no new chunk.
	//    The pass thread's own freed list was swept too, so every block it takes here comes from
	//    the shared fresh tier -- which is exactly what the boundary release was for.
	const unsigned long long chunks_before = pool_after.chunks_cut;
	{
		std::vector<std::unique_ptr<FIDESlib::CKKS::Ciphertext>> burst;
		burst.reserve(kWorkers * kPerWorker);
		for (int i = 0; i < kWorkers * kPerWorker; ++i)
			burst.emplace_back(std::make_unique<FIDESlib::CKKS::Ciphertext>(GPUcc_));
	}
	cudaDeviceSynchronize();
	const FIDESlib::MemPoolStats pool_burst = FIDESlib::MemPoolStatsSnapshot(dev);
	printf("[pass-boundary] after the burst: %s\n", FIDESlib::MemPoolStatsLine(dev).c_str());
	fflush(stdout);
	EXPECT_EQ(pool_burst.chunks_cut, chunks_before)
		<< "the pool cut a new chunk for a burst the boundary release had just handed it the blocks for";
}

// ---- NOTHING REACHES THE DEVICE POOL ON THE NULL STREAM --------------------------------------
//
// THE DEFECT THIS PINS. `Plaintext` (Plaintext.cu:53, :59) and `KeySwitchingKey`
// (KeySwitchingKey.cu:60) build their polynomials with RNSPoly's `def_stream` flag, which used to
// mean `Stream::initDefault()` -- `ptr_ = 0` -- in every mode. That null is not private to the
// partition: `LimbPartition::generate` binds every limb it creates to the partition's stream
// (USE_PARTITION_STREAM) and `Limb` holds it as a `Stream&` (Limb.cuh:24), so every limb of a
// plaintext or of a key LAUNCHED on the legacy default stream and reached the device pool with
// `GPUmalloc(..., 0)` / `GPUfree(..., 0)`. In the lane pool that free takes
// `Stream::wait(cudaStream_t)`'s `s == 0` early out and issues NO fence at all: the block returns
// to the free list owing nothing, the next lane's `record_and_wait` fences against a handshake
// stream that was never recorded on, and it overwrites the block while the previous owner's
// kernels are still reading it -- one lane's whole output wrong with its shape intact, about one
// repetition in eight (StagedLanesMatchSerial; a GPU run traced 96 of 96 pool FREE
// events carrying stream = 0).
//
// TWO CHECKS, because there are exactly two ways a `Stream::ptr_` becomes null:
//   1. BORN NULL -- `Stream::initDefault()`, the case above. Asserted directly on the partitions
//      and limbs of a plaintext, a ciphertext and the evaluation key.
//   2. NULLED BY A MOVE -- `Stream::Stream(Stream&&)` sets the SOURCE's `ptr_` to null, so a
//      `Limb` still holding a `Stream&` to a moved-from partition would behave identically. The
//      objects are therefore moved -- by hand and by a `std::vector` reallocation, which is the
//      only way an application makes the library move a polynomial at all -- and then USED and
//      DESTROYED, so that every allocation and every free of the moved objects goes through the
//      pool afterwards.
//
// THE ACCEPTANCE CRITERION IS THE POOL'S COUNTER, not the stream values on their own: it is the
// hand-off, not the object, that a null stream actually breaks, and the counter sees every path
// into the pool including the ones this test does not name. `StagedLanesMatchSerial` remains the
// end-to-end acceptance test; this one says WHICH invariant broke when that one fails again.
TEST_P(ConcurrentOpsTest, NoNullStreamAtThePool) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}
	const std::vector<int> rot_index{ 1, 2 };
	int numSlots				   = 0;
	FIDESlib::CKKS::Context GPUcc_ = Prepare(rot_index, numSlots);
	FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;

	std::vector<double> x(8), y(8);
	for (int i = 0; i < 8; ++i) {
		x[i] = 0.1 * (i + 1);
		y[i] = 0.05 * (i + 2);
	}
	lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
	lbcrypto::Plaintext q = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
	const FIDESlib::CKKS::RawCipherText raw_in = FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, p));
	const FIDESlib::CKKS::RawPlainText raw_pt  = FIDESlib::CKKS::GetRawPlainText(cc, q);

	/// Every partition of `poly`, and every Q limb bound to it, must name a real stream.
	const auto expect_real_streams = [](const FIDESlib::CKKS::RNSPoly& poly, const std::string& what) {
		for (size_t g = 0; g < poly.GPU.size(); ++g) {
			const FIDESlib::CKKS::LimbPartition& part = poly.GPU[g];
			EXPECT_NE(part.getS().raw(), nullptr) << what << ": partition " << g << " launches on the null stream";
			for (size_t l = 0; l < part.limb.size(); ++l) {
				EXPECT_NE(STREAM(part.limb[l]).raw(), nullptr)
					<< what << ": partition " << g << " limb " << l << " is bound to a null stream";
			}
		}
	};

	// The baseline is taken AFTER Prepare: the context build is its own (single-threaded) story,
	// and what this test is about is every allocation and free the work below performs.
	cudaDeviceSynchronize();
	const unsigned long long null_before = FIDESlib::PoolNullStreamEvents();

	// (1) BORN NULL. The three owners that used to be built with `def_stream`, plus one op each so
	// that their limbs are actually launched on and their scratch actually allocated and freed.
	{
		FIDESlib::CKKS::Plaintext pt(GPUcc_, raw_pt);
		FIDESlib::CKKS::Ciphertext ct(GPUcc_, raw_in);
		expect_real_streams(pt.c0, "plaintext c0");
		expect_real_streams(ct.c0, "ciphertext c0");
		expect_real_streams(ct.c1, "ciphertext c1");
		FIDESlib::CKKS::KeySwitchingKey& evk = GPUcc.GetEvalKey(ct.keyID);
		expect_real_streams(evk.a, "evaluation key a");
		expect_real_streams(evk.b, "evaluation key b");
		ct.multPt(pt, true);
		ct.rotate(rot_index[0]);
	}

	// (2) NULLED BY A MOVE. No `reserve`, so the vectors reallocate and move-construct everything
	// they already hold; then an explicit move of a ciphertext out of one of them. Every object is
	// USED after its move and DESTROYED at the end of the scope, which is where its limb blocks go
	// back to the pool -- the free that the null stream used to leave unfenced.
	{
		std::vector<FIDESlib::CKKS::Ciphertext> cts;
		std::vector<FIDESlib::CKKS::Plaintext> pts;
		for (int i = 0; i < 4; ++i) {
			cts.emplace_back(GPUcc_, raw_in);
			pts.emplace_back(GPUcc_, raw_pt);
		}
		for (size_t i = 0; i < cts.size(); ++i) {
			expect_real_streams(cts[i].c0, "moved ciphertext c0 [" + std::to_string(i) + "]");
			expect_real_streams(cts[i].c1, "moved ciphertext c1 [" + std::to_string(i) + "]");
			expect_real_streams(pts[i].c0, "moved plaintext c0 [" + std::to_string(i) + "]");
		}
		for (size_t i = 0; i < cts.size(); ++i)
			cts[i].multPt(pts[i], true);

		FIDESlib::CKKS::Ciphertext moved(std::move(cts.back()));
		cts.pop_back(); // destroys the moved-FROM ciphertext, whose polynomials are now empty
		expect_real_streams(moved.c0, "explicitly moved ciphertext c0");
		expect_real_streams(moved.c1, "explicitly moved ciphertext c1");
		moved.rotate(rot_index[1]);
	}
	cudaDeviceSynchronize();

	const unsigned long long null_after = FIDESlib::PoolNullStreamEvents();
	printf("[null-stream] pool null-stream events: %llu before, %llu after\n", null_before, null_after);
	fflush(stdout);
	EXPECT_EQ(null_after, null_before)
		<< "a NULL stream reached GPUmalloc/GPUfree in the concurrent mode: that hand-off is unfenced "
		   "(Stream::wait(cudaStream_t) early-outs on s == 0), which is the concurrent-bootstrap race. Its "
		   "owner has a Stream whose ptr_ is null -- a partition built with def_stream, or a moved-from "
		   "Stream still referenced by a Limb";

	CKKS::DeregisterAllContexts();
	ClearCachedContexts();
}

// ---- The cross-thread fence handshake, on its own -------------------------------------------
//
// StagedLanesMatchSerial reaches the fence primitives through a bootstrap, so a dropped
// dependency shows up as a wrong ciphertext some repetitions later and says nothing about which
// fence dropped it. This test drives `Stream::wait` directly, in the two shapes the concurrent
// mode adds and the default mode never produces:
//
//   1. ALIASING. `Stream::init` hands streams out of a 37-entry per-device pool, so two Stream
//      objects owned by two threads routinely carry the same `cudaStream_t`. The consumers below
//      are chosen so that at least one of them aliases the producer, which is exactly the case
//      `Stream::wait`'s `ptr_ == s.ptr_` early-out used to skip.
//   2. A SHARED PRODUCER, FENCED IN BOTH DIRECTIONS. Every thread fences on ONE producer Stream
//      and then back-fences it -- `x.wait(P)` then `P.wait(x)` -- which is the shape the
//      bootstrap's precomputation plaintexts and key-switching keys are fenced with
//      (LimbPartitionBatch.cu:433/:521, Bootstrap.cu:235/:239). With one `cudaEvent_t` per
//      Stream, all four threads record and wait on the same event; with the per-lane events they
//      each record their own.
//
// The producer's last write is the round's byte value, behind a long chain of writes of the
// PREVIOUS value, so a consumer that is not ordered behind it copies the previous round's bytes
// and the check fails with a value that names the round it came from.
TEST(ConcurrentOpsFence, SharedProducerFenceAcrossLanes) {
	if (!FIDESlib::ConcurrentOps()) {
		GTEST_SKIP() << "set FIDESLIB_CONCURRENT_OPS=1 to run the concurrent-op suite";
	}
	int devices = 0;
	ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
	if (devices < 1)
		GTEST_SKIP() << "no CUDA device";
	ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

	constexpr int kThreads	  = 4;
	constexpr int kRounds	  = 32;
	constexpr int kChainDepth = 96; ///< writes of the PREVIOUS value queued ahead of the real one
	constexpr size_t kBytes	  = 1u << 18;

	// More Streams than the pool has (37), so the producer is guaranteed to share its
	// `cudaStream_t` with at least one of them.
	constexpr int kCandidates = 96;
	FIDESlib::Stream producer;
	producer.init();
	const cudaStream_t producer_raw = producer.ptr();
	ASSERT_NE(producer_raw, nullptr);

	std::vector<std::unique_ptr<FIDESlib::Stream>> candidates;
	candidates.reserve(kCandidates);
	for (int i = 0; i < kCandidates; ++i) {
		auto s = std::make_unique<FIDESlib::Stream>();
		s->init();
		candidates.emplace_back(std::move(s));
	}

	// Consumer 0 aliases the producer if any candidate does. The others are distinct objects,
	// aliased or not -- the pool decides, which is the point.
	std::vector<FIDESlib::Stream*> consumers;
	for (auto& c : candidates) {
		if (c->ptr() == producer_raw) {
			consumers.push_back(c.get());
			break;
		}
	}
	EXPECT_FALSE(consumers.empty()) << "expected the 37-entry stream pool to alias at least one of " << kCandidates
									<< " streams onto the producer; the aliasing half of this test did not run";
	for (auto& c : candidates) {
		if (static_cast<int>(consumers.size()) >= kThreads)
			break;
		if (!consumers.empty() && c.get() == consumers.front())
			continue;
		consumers.push_back(c.get());
	}
	ASSERT_EQ(static_cast<int>(consumers.size()), kThreads);

	uint8_t* src = nullptr;
	ASSERT_EQ(cudaMalloc(&src, kBytes), cudaSuccess);
	std::vector<uint8_t*> dst(kThreads, nullptr);
	for (int t = 0; t < kThreads; ++t)
		ASSERT_EQ(cudaMalloc(&dst[t], kBytes), cudaSuccess);

	std::vector<std::vector<uint8_t>> host(kThreads, std::vector<uint8_t>(kBytes, 0));

	// Round 0 leaves the producer holding a known value so round 1's chain has something to be
	// mistaken for.
	ASSERT_EQ(cudaMemsetAsync(src, 0, kBytes, producer.ptr()), cudaSuccess);
	ASSERT_EQ(cudaStreamSynchronize(producer.ptr()), cudaSuccess);

	int bad = 0;
	for (int round = 1; round <= kRounds; ++round) {
		const uint8_t want = static_cast<uint8_t>(round);
		const uint8_t prev = static_cast<uint8_t>(round - 1);
		for (int i = 0; i < kChainDepth; ++i)
			ASSERT_EQ(cudaMemsetAsync(src, prev, kBytes, producer.ptr()), cudaSuccess);
		ASSERT_EQ(cudaMemsetAsync(src, want, kBytes, producer.ptr()), cudaSuccess);

		std::vector<std::thread> pool;
		pool.reserve(kThreads);
		for (int t = 0; t < kThreads; ++t) {
			pool.emplace_back([&, t] {
				cudaSetDevice(0);
				FIDESlib::Stream& mine = *consumers[t];
				// The fence under test: order this thread's copy behind the shared producer.
				mine.wait(producer);
				cudaMemcpyAsync(dst[t], src, kBytes, cudaMemcpyDeviceToDevice, mine.ptr());
				// ...and the back-fence, so the producer's next write cannot run underneath the
				// copy. Four threads do this to ONE producer at once.
				producer.wait(mine);
			});
		}
		for (auto& th : pool)
			th.join();
		ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

		for (int t = 0; t < kThreads; ++t) {
			ASSERT_EQ(cudaMemcpy(host[t].data(), dst[t], kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
			size_t first_bad = kBytes;
			for (size_t i = 0; i < kBytes; ++i) {
				if (host[t][i] != want) {
					first_bad = i;
					break;
				}
			}
			if (first_bad != kBytes) {
				++bad;
				if (bad <= 4) {
					printf("[fence] round %d thread %d: byte %zu is %u, expected %u (aliases producer: %s)\n",
						   round,
						   t,
						   first_bad,
						   static_cast<unsigned>(host[t][first_bad]),
						   static_cast<unsigned>(want),
						   consumers[t]->ptr() == producer_raw ? "yes" : "no");
					fflush(stdout);
				}
			}
		}
	}

	for (int t = 0; t < kThreads; ++t)
		cudaFree(dst[t]);
	cudaFree(src);

	EXPECT_EQ(bad, 0) << "a consumer ran ahead of the shared producer: the cross-thread fence was dropped";
}

/// WHICH STEP OF THE CHAIN LOSES DECODABILITY, AND IN WHICH MODE (2026-10-03).
///
/// An earlier run's [decrypt] probe measured something the concurrency verdict cannot explain: in
/// parameter sets 3, 5 and 7 -- the FIXEDMANUAL ones at dnum >= 2 -- EVERY output of EVERY lane,
/// the single-threaded serial reference included, threw OpenFHE's "approximation error is too
/// high" out of Decode (ckkspackedencoding.cpp:453). So in those sets the chain is already
/// undecodable before any second thread exists, and BootstrapLanesMatchSerial's raw comparisons
/// there are comparing garbage with garbage. Two things that run did not say: WHICH op in the
/// chain loses it, and whether the default (non-concurrent) mode loses it too -- every test in
/// this file GTEST_SKIPs without FIDESLIB_CONCURRENT_OPS, so this chain has never been seen there.
///
/// This probe answers both. It runs lane 0's chain ONCE on the calling thread -- same Prepare,
/// same values, same ops in the same order -- and after EVERY step stores the ciphertext and
/// decrypts it the way the [decrypt] block does (fresh level-0 holder, GetOpenFHECipherText,
/// Decrypt), against a cleartext expectation carried along step by step. It does NOT skip without
/// the concurrent mode, so the `_0` and `_1` invocations of the binary each print their own nine
/// lines per parameter set and `mode=` says which is which.
///
/// It asserts NOTHING. Every line must print even when every decrypt fails, so each step owns its
/// try/catch and the whole body sits under a catch-all; a probe must not be able to turn a
/// measured run into a failure or a crash. The what() is tailed, not headed: OpenFHE's message
/// opens with the file path and the first 120 characters of it -- which is what 51b0f77 printed --
/// are the path, never the reason.
TEST_P(ConcurrentOpsTest, ChainDecodesPerStep) {
	// mode is read per invocation of the binary, not per parameter set: this is the one test here
	// that runs in both, and the line has to say which one produced it.
	const char* const mode = FIDESlib::ConcurrentOps() ? "concurrent" : "default";

	// THE PARAMETER. The fixture is TestWithParam<tuple<tuple<GeneralTestParams, Parameters>,
	// ScalingTechnique>> (ParametrizedTest.cuh): the technique is the OUTER tuple's second
	// element, and dnum is a field of GeneralTestParams, the INNER tuple's first. GeneralParamet-
	// rizedTest::SetUp already unpacked the inner one into generalTestParams, so dnum is read from
	// there; the technique it only forwards to OpenFHE, so it is taken from GetParam() directly.
	const lbcrypto::ScalingTechnique tech = std::get<1>(GetParam());
	const unsigned long long dnum		  = static_cast<unsigned long long>(generalTestParams.dnum);
	std::string tech_name;
	try {
		std::ostringstream os; // lbcrypto::operator<<(ostream&, ScalingTechnique), pke/constants.h
		os << tech;
		tech_name = os.str();
	} catch (...) {
		tech_name.clear();
	}
	if (tech_name.empty())
		tech_name = std::to_string(static_cast<int>(tech));

	try {
		constexpr int kLane = 0; // lane 0 of BootstrapLanesMatchSerial, values and rotations alike
		const std::vector<int> rot_index{ 1, 2, 3, 4 };
		int numSlots = 0;

		FIDESlib::CKKS::Context GPUcc_	   = Prepare(rot_index, numSlots);
		FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;
		(void)GPUcc;

		std::vector<double> x(8), y(8);
		for (int i = 0; i < 8; ++i) {
			x[i] = 0.1 * (kLane + 1) * (i + 1);
			y[i] = 0.05 * (kLane + 2) * (i + 2);
		}
		lbcrypto::Plaintext p				 = cc->MakeCKKSPackedPlaintext(x, 1, 0, nullptr, numSlots);
		FIDESlib::CKKS::RawCipherText raw_in = FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, p));
		lbcrypto::Plaintext q				 = cc->MakeCKKSPackedPlaintext(y, 1, 0, nullptr, numSlots);
		FIDESlib::CKKS::RawPlainText raw_pt	 = FIDESlib::CKKS::GetRawPlainText(cc, q);

		// The holder the conversion pours onto: a FRESH full-tower level-0 ciphertext per step, so
		// no two decrypts share an object.
		std::vector<double> zero(8, 0.0);
		lbcrypto::Plaintext tmpl_pt = cc->MakeCKKSPackedPlaintext(zero, 1, 0, nullptr, numSlots);

		// THE EXPECTATION, carried step by step rather than computed at the end: multPt is the
		// slotwise product with y, Deplete and Bootstrap are the identity, and rotate is the CKKS
		// LEFT rotation new[j] = old[(j + idx) mod numSlots] -- the convention 51b0f77 verified
		// against OpenFheInterfaceTests.cu's Rotate cases.
		std::vector<double> want(numSlots, 0.0), yfull(numSlots, 0.0), tmp(numSlots, 0.0);
		for (int i = 0; i < 8; ++i) {
			want[i]	 = x[i];
			yfull[i] = y[i];
		}

		FIDESlib::CKKS::Ciphertext ct(GPUcc_, raw_in);
		FIDESlib::CKKS::Plaintext pt(GPUcc_, raw_pt);

		auto probe = [&](const char* step) {
			FIDESlib::CKKS::RawCipherText raw;
			std::string err;
			double e	  = 0.0;
			double scale  = 0.0;
			int numRes	  = -1;
			int deg		  = -1;
			try {
				ct.store(raw);
				numRes = raw.numRes;
				deg	   = raw.NoiseLevel;
				scale  = raw.Noise;
				lbcrypto::Ciphertext<lbcrypto::DCRTPoly> holder = cc->Encrypt(keys.publicKey, tmpl_pt);
				FIDESlib::CKKS::GetOpenFHECipherText(holder, raw);
				lbcrypto::Plaintext got;
				cc->Decrypt(keys.secretKey, holder, &got);
				const std::vector<double> v = got->GetRealPackedValue();
				for (int i = 0; i < 8; ++i) {
					const double d	 = (i < (int)v.size() ? v[i] : 0.0) - want[i];
					const double mag = d < 0.0 ? -d : d;
					if (mag > e)
						e = mag;
				}
			} catch (const std::exception& ex) {
				err = ex.what();
			} catch (...) {
				err = "unknown exception";
			}
			// THE TAIL, not the head: OpenFHE prefixes what() with the source path, so the first
			// 120 characters 51b0f77 kept were the path and the reason was the part it cut.
			if (err.size() > 100)
				err = err.substr(err.size() - 100);
			char ebuf[160];
			if (err.empty())
				snprintf(ebuf, sizeof(ebuf), "%.3e", e);
			else
				snprintf(ebuf, sizeof(ebuf), "FAIL: %s", err.c_str());
			printf("[chain] mode=%s tech=%s dnum=%llu step=%s numRes=%d deg=%d scale=%.6g err=%s\n", mode,
			       tech_name.c_str(), dnum, step, numRes, deg, scale, ebuf);
			fflush(stdout);
		};

		probe("input");
		static const char* const kMult[2]  = { "r0.multPt", "r1.multPt" };
		static const char* const kDepl[2]  = { "r0.deplete", "r1.deplete" };
		static const char* const kBoot[2]  = { "r0.bootstrap", "r1.bootstrap" };
		static const char* const kRot[2]   = { "r0.rotate", "r1.rotate" };
		for (int round = 0; round < 2; ++round) {
			ct.multPt(pt, true);
			for (int j = 0; j < numSlots; ++j)
				want[j] *= yfull[j];
			probe(kMult[round]);

			DepleteForBootstrap(ct); // identity on the cleartext
			probe(kDepl[round]);

			Bootstrap(ct, numSlots, false); // identity on the cleartext
			probe(kBoot[round]);

			const int idx = rot_index[(kLane + round) % rot_index.size()];
			ct.rotate(idx);
			for (int j = 0; j < numSlots; ++j)
				tmp[j] = want[(j + idx) % numSlots];
			want.swap(tmp);
			probe(kRot[round]);
		}
	} catch (const std::exception& e) {
		std::string w = e.what();
		if (w.size() > 100)
			w = w.substr(w.size() - 100);
		printf("[chain] FAILED: %s\n", w.c_str());
		fflush(stdout);
	} catch (...) {
		printf("[chain] FAILED: unknown exception\n");
		fflush(stdout);
	}
	// Same teardown as every other case here; the GPU objects died with the try block above.
	CKKS::DeregisterAllContexts();
	ClearCachedContexts();
}

INSTANTIATE_TEST_SUITE_P(ConcurrencyTests, ConcurrentOpsTest, testing::Values(TTALL64BOOT));

} // namespace FIDESlib::Testing
