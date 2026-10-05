//
// Created by carlosad on 4/12/24.
//

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Checksum.cuh"
#include "CudaUtils.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh" // copy_ (and, transitively, AddSub.cuh: sub_)
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Rescale.cuh" // SwitchModulus<T> global (device modulus switch + sign recentering)
#include "Math.cuh"
#include <cstdlib>
#include <stdexcept>
#include <string>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

// ---------------------------------------------------------------------------
// In-context SPARSE_ENCAPSULATED instrumentation + device primitive.
// ---------------------------------------------------------------------------

// Env-gated fingerprint (OFF by default). Set FIDESLIB_ENCAPS_FINGERPRINT to any
// value to print a stable digest of the ciphertext at the three ENCAPS dance
// points, so they can be diffed against the CPU prints in stock
// ckksrns-fhe.cpp (post-sparse-switch -> line 830; post-raise -> lines 833-840;
// post-switch-back -> line 843). Prints the first 8 coefficients of tower 0 of
// c0 and c1, in whatever NTT domain the ciphertext currently holds (EVALUATION at
// all three points, matching the corresponding CPU element state).
namespace {
void EncapsFingerprint(FIDESlib::CKKS::Ciphertext& ctxt, const char* label) {
	if (std::getenv("FIDESLIB_ENCAPS_FINGERPRINT") == nullptr)
		return;
	cudaDeviceSynchronize();
	std::cout << "[ENCAPS_FP] " << label << " c0[t0]:";
	for (auto& j : ctxt.c0.GPU) {
		cudaSetDevice(j.device);
		if (!j.limb.empty()) {
			SWITCH(j.limb[0], printThisLimb(8));
		}
		break;
	}
	std::cout << " c1[t0]:";
	for (auto& j : ctxt.c1.GPU) {
		cudaSetDevice(j.device);
		if (!j.limb.empty()) {
			SWITCH(j.limb[0], printThisLimb(8));
		}
		break;
	}
	std::cout << std::endl;
	cudaDeviceSynchronize();
}
} // namespace

// --- Two single-limb coefficient-domain rebase kernels the sparse switch needs ---
//
// Both helpers are single-tower coefficient-domain modulus switches with the sign
// recentering stock does via NativeVector::SwitchModulus. FIDESlib provides that exact
// recentering as the device `SwitchModulus(src, om_pid, res, nm_pid)` global (Rescale.cuh /
// Rescale.cu), keyed on RAW primeids into the global C_.primes table; a single-limb NTT/INTT
// is the LimbPartition::ApplyNTT/ApplyINTT pair (the same call the mod-down path uses for its
// lone special limb, LimbPartition.cu:384). Everything below composes those primitives — no
// new raw CUDA.
//
// REPRESENTATION. The (q0,p) polynomial is a level-0 poly: one Q limb (tower 0, q0, raw
// primeid 0) plus one SPECIAL limb (p). generateSpecialLimbs() allocates the p slot (data +
// its own NTT aux, SPECIALauxptr); SetModUp(true) marks the special limb live so the caller's
// copy()/add() carry it. MODRAISE_WITH_P0 == false, so broadcastLimb0() would NOT fill the
// special slot — these helpers fill/consume it explicitly instead.
//
// WORKSPACE. To avoid a temporary RNSPoly (and the cross-object stream/lifetime coordination
// that its async free would require), both helpers use the SPECIAL limb's own N-word buffer as
// the scratch tower. A single-limb NTT/INTT is table-agnostic w.r.t. where the data lives:
// ApplyNTT/ApplyINTT (mode *_NONE) take the prime purely from the primeid_init argument, never
// from the limb's stored primeid, so the p buffer can be transformed under q0's tables when it
// transiently holds q0 residues (verified: NTT_NONE/INTT_NONE read PRIMEID(limb) only in the
// RESCALE/MULTPT fusions, LimbPartition.cu ApplyNTT). That is the one non-obvious step and is
// called out at each use below.
//
// FLAG: this device code was written without a CUDA build; its first run is on a GPU.
namespace {

// Small accessor: the u64 device data pointer of a limb. q0 and p are ~60-bit CKKS primes,
// hence U64; a U32 limb here would be a parameter error.
inline uint64_t* u64data(FIDESlib::CKKS::LimbImpl& l) {
	assert(l.index() == FIDESlib::U64);
	return std::get<FIDESlib::U64>(l).v.data;
}

// Stock 3821-3827: extend a level-0 poly (one Q limb, EVALUATION) to (q0,p):
//   poly' = COPY(tower0); INTT_q0(poly'); SwitchModulus(q0->p, poly'); NTT_p(poly') -> special.
// Stock operates on a COPY, leaving tower 0 (element 0) untouched in EVALUATION — which is
// load-bearing, because the caller immediately multElement()s tower 0 against the key in
// EVALUATION. We mirror that by doing all the work in the p special buffer and never writing
// tower 0.
void sparseExtendTower0ToSpecial(FIDESlib::CKKS::RNSPoly& poly) {
	using namespace FIDESlib;
	using namespace FIDESlib::CKKS;

	// Allocate the p special limb (data + NTT aux). Not zeroed, not for communication.
	poly.generateSpecialLimbs(/*zero_out=*/false, /*for_communication=*/false);

	ContextData& cc = poly.GPU[0].cc;

	for (size_t i = 0; i < poly.GPU.size(); ++i) {
		LimbPartition& P = poly.GPU[i];
		// Single-GPU: tower 0 and the lone p limb are co-located. (Multi-GPU placement of the
		// special limb on a non-tower-0 device is NOT handled here — see report.)
		if (P.SPECIALlimb.empty() || P.limb.empty())
			continue;
		cudaSetDevice(P.device);

		uint64_t* sp		= u64data(P.SPECIALlimb[0]); // p special buffer = our scratch tower
		const int q0_primeid = PRIMEID(P.limb[0]);		 // raw q0 primeid (0)
		const int p_primeid = PRIMEID(P.SPECIALlimb[0]); // raw p primeid

		Stream& ss = STREAM(P.SPECIALlimb[0]);
		ss.wait(P.s);				  // after any prior partition work
		ss.wait(STREAM(P.limb[0]));	  // and after tower 0's own stream (we read it once, below)

		// COPY tower0 (EVALUATION, q0 residues) into the p buffer. All subsequent steps touch
		// only the p buffer; tower 0 is never modified.
		copy_<<<dim3{ (uint32_t)cc.N / 128, 1 }, 128, 0, ss.ptr()>>>(P.limbptr.data, P.SPECIALlimbptr.data);

		// INTT under q0 (cross-table: p buffer transiently holds q0 residues) -> tower0 coeffs.
		P.ApplyINTT<ALGO_SHOUP, INTT_NONE>(
		  cc.batch, LimbPartition::INTT_fusion_fields{}, P.SPECIALlimb, P.SPECIALlimbptr, P.SPECIALauxptr, cc, PARTITION(P.id, 0), 1);

		// Stock 3824: SwitchModulus(q0 -> p) on the coefficients, in place in the p buffer.
		SwitchModulus<uint64_t><<<dim3{ (uint32_t)cc.N / 128 }, 128, 0, ss.ptr()>>>(sp, q0_primeid, sp, p_primeid);

		// Stock 3825: NTT under p -> the p special limb in EVALUATION.
		P.ApplyNTT<ALGO_SHOUP, NTT_NONE>(
		  cc.batch, LimbPartition::NTT_fusion_fields{}, P.SPECIALlimb, P.SPECIALlimbptr, P.SPECIALauxptr, cc, SPECIAL(P.id, 0), 1);

		P.s.wait(ss); // fold the special-limb work back onto the partition stream
	}

	// Mark the special limb live so the caller's cvRes.copy(c1) carries it. tower 0 unchanged.
	poly.SetModUp(true);
}

// Stock 3837-3847: exact RNS mod-switch (q0,p) -> q0 for one poly (eprint 2021/204 App B.2.2,
// t=1; NO floating rescale). Per stock:
//   polyP = NTT_q0( SwitchModulus(p->q0, INTT_p(special)) );   // reconstruct the p part in q0
//   DropLastElement;                                           // drop the p limb
//   tower0 = tower0 - polyP;  tower0 = tower0 * pModInvq        // subtract THEN scale (order!)
// with pModInvq = p^-1 mod q0. The subtract and scale are done in EVALUATION, matching stock.
void sparseFoldSpecialIntoTower0(FIDESlib::CKKS::RNSPoly& poly, uint64_t pModInvq) {
	using namespace FIDESlib;
	using namespace FIDESlib::CKKS;

	ContextData& cc = poly.GPU[0].cc;

	for (size_t i = 0; i < poly.GPU.size(); ++i) {
		LimbPartition& P = poly.GPU[i];
		if (P.SPECIALlimb.empty() || P.limb.empty())
			continue;
		cudaSetDevice(P.device);

		uint64_t* sp		= u64data(P.SPECIALlimb[0]); // p buffer; becomes the q0 "tmp" (polyP)
		const int q0_primeid = PRIMEID(P.limb[0]);		 // raw q0 primeid (0)
		const int p_primeid = PRIMEID(P.SPECIALlimb[0]); // raw p primeid

		Stream& ss = STREAM(P.SPECIALlimb[0]);
		ss.wait(P.s);
		ss.wait(STREAM(P.limb[0]));

		// INTT under p -> p coefficients (same call the mod-down path makes for its special limb).
		P.ApplyINTT<ALGO_SHOUP, INTT_NONE>(
		  cc.batch, LimbPartition::INTT_fusion_fields{}, P.SPECIALlimb, P.SPECIALlimbptr, P.SPECIALauxptr, cc, SPECIAL(P.id, 0), 1);

		// Stock 3840: SwitchModulus(p -> q0) in place; the p buffer now holds polyP as q0 coeffs.
		SwitchModulus<uint64_t><<<dim3{ (uint32_t)cc.N / 128 }, 128, 0, ss.ptr()>>>(sp, p_primeid, sp, q0_primeid);

		// Stock 3841: NTT under q0 (cross-table: q0 residues in the p buffer) -> polyP EVALUATION.
		P.ApplyNTT<ALGO_SHOUP, NTT_NONE>(
		  cc.batch, LimbPartition::NTT_fusion_fields{}, P.SPECIALlimb, P.SPECIALlimbptr, P.SPECIALauxptr, cc, PARTITION(P.id, 0), 1);

		// Stock 3843-3844 (subtract half): tower0 = tower0 - polyP (both EVALUATION, mod q0).
		FIDESlib::sub_<<<dim3{ (uint32_t)cc.N / 128, 1 }, 128, 0, ss.ptr()>>>(P.limbptr.data, P.SPECIALlimbptr.data, PARTITION(P.id, 0));

		P.s.wait(ss);
		STREAM(P.limb[0]).wait(ss); // tower 0 was written on ss; publish it to tower 0's stream
	}

	// Stock 3845 (scale half): tower0 = tower0 * pModInvq (mod q0), AFTER the subtract. multScalar
	// scales only the level-0 Q limb (tower 0); the still-present p limb is untouched.
	std::vector<uint64_t> pInv(cc.prime.size(), 0);
	pInv[0] = pModInvq;
	poly.multScalar(pInv);

	// Stock 3842 DropLastElement: drop the p limb (also clears the mod-up flag).
	poly.freeSpecialLimbs();
}

// The p-tower half of stock 3830 (cvRes = c1Ext * ek). RNSPoly::multElement ->
// LimbPartition::multElement (LimbPartition.cu:536/554) multiplies only the Q limbs
// (getLimbSize(*level) of them); it never touches the SPECIAL(p) limb. For a full
// (q0,p) DCRTPoly multiply the p component must be multiplied too, or the exact
// mod-switch below folds an unmultiplied p-tower back in and silently corrupts the
// result. This mirrors the Q-limb `Mult_(l, l, p, PARTITION(id,i))` in-place multiply,
// one special limb per partition under SPECIAL(id,0). Localized here (rather than in
// multElement) so no other multElement caller changes behaviour — the hybrid key-switch
// and plaintext-mult callers handle P/special limbs through their own dedicated paths
// and rely on multElement leaving specials alone.
void sparseMultSpecialLimb(FIDESlib::CKKS::RNSPoly& dst, const FIDESlib::CKKS::RNSPoly& key) {
	using namespace FIDESlib;
	using namespace FIDESlib::CKKS;

	for (size_t i = 0; i < dst.GPU.size(); ++i) {
		LimbPartition& D	   = dst.GPU[i];
		const LimbPartition& K = key.GPU[i];
		if (D.SPECIALlimb.empty() || K.SPECIALlimb.empty())
			continue;
		cudaSetDevice(D.device);
		ContextData& cc = D.cc;

		Stream& ds = STREAM(D.SPECIALlimb[0]);
		ds.wait(D.s);
		ds.wait(K.getS());
		// dst.p = dst.p * key.p  (EVALUATION, elementwise, mod p)
		Mult_<<<dim3{ (uint32_t)cc.N / 128, 1 }, 128, 0, ds.ptr()>>>(D.SPECIALlimbptr.data, D.SPECIALlimbptr.data, K.SPECIALlimbptr.data, SPECIAL(D.id, 0));
		D.s.wait(ds);
		K.getS().wait(ds);
	}
}

} // namespace

// keySwitchSparse — GPU transcription of FHECKKSRNS::KeySwitchSparse
// (ckksrns-fhe.cpp:3812). Operates on tower 0 only, over the two-prime basis
// (q0, p) carried by the 2N-4 GHS single-digit key. NOT a hybrid key switch.
//
// The ciphertext enters at level 0 (a single Q limb, EVALUATION), the state stock
// reaches after ModReduce + AdjustCiphertext, BEFORE the modulus raise. `ek` is the
// 2N-4 key loaded by KeySwitchingKey::InitializeSparse: ek.a and ek.b are each a
// (q0, p) polynomial = one Q limb (q0, primeid 0) + one special limb (p), EVALUATION.
//
// In fideslib the special prime p lives as a SPECIAL limb, so the (q0,p) polynomial
// is the level-0 + one-special-limb ("mod-up") representation. See RNSPoly::load.
void FIDESlib::CKKS::keySwitchSparse(FIDESlib::CKKS::Ciphertext& ctxt, const FIDESlib::CKKS::KeySwitchingKey& ek) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	ContextData& cc = ctxt.cc;

	// Stock 3819-3827: extend cv[1] (=c1) from q0 to (q0,p).
	RNSPoly& c1 = ctxt.c1;
	sparseExtendTower0ToSpecial(c1);

	// Stock 3830: cvRes = { c1Ext * B, c1Ext * A } pointwise over both towers.
	// RNSPoly::multElement multiplies the Q limb only; sparseMultSpecialLimb supplies
	// the p-tower multiply it skips (see that helper), completing the (q0,p) multiply.
	RNSPoly cvRes0(cc), cvRes1(cc);
	cvRes0.copy(c1);                 // (q0,p) shape (copy carries the p limb: c1 is mod-up)
	cvRes0.multElement(ek.b);        // c1Ext * B  (q0 tower)
	sparseMultSpecialLimb(cvRes0, ek.b); // c1Ext * B  (p tower)
	cvRes1.copy(c1);
	cvRes1.multElement(ek.a);        // c1Ext * A  (q0 tower)
	sparseMultSpecialLimb(cvRes1, ek.a); // c1Ext * A  (p tower)

	// Stock 3832-3847: exact RNS mod-switch (q0*p) -> q0 per component.
	// pModInvq = p^-1 mod q0. Stock: modulusp.ModInverse(modulusq).
	uint64_t q0       = cc.prime[0].p;
	uint64_t p        = cc.specialPrime[0].p;
	uint64_t pModInvq = FIDESlib::modinv(p % q0, q0);
	sparseFoldSpecialIntoTower0(cvRes0, pModInvq);
	sparseFoldSpecialIntoTower0(cvRes1, pModInvq);

	// Stock 3850: cvRes[0] += cv[0] (fold original c0 back in).
	cvRes0.add(ctxt.c0);

	// Stock 3852-3853: result elements = { cvRes0, cvRes1 }.
	ctxt.c0.copy(cvRes0);
	ctxt.c1.copy(cvRes1);
}

void FIDESlib::CKKS::BootstrapCPUraise(Ciphertext& ctxt,
  const int slots,
  std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
  const bool prescaled) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;
	Ciphertext aux(cc_);
	bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

	/////////////////////////////////////////////////////////////////////
	// NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
	uint64_t q	   = cc.prime[0].p;
	double qDouble = (double)q; // q.ConvertToDouble();

	if constexpr (PRINT) {
		std::cout << "q: " << q << " ";
		std::cout << qDouble << std::endl;
	}
	const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
	double powP	 = pow(2, p);

	if constexpr (PRINT) {
		std::cout << "p: " << p << std::endl;
	}
	int32_t deg = std::round(std::log2(qDouble / powP));
	/*
#if NATIVEINT != 128
	if (deg > static_cast<int32_t>(m_correctionFactor)) {
		OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
					  std::to_string(m_correctionFactor) + "].");
	}
#endif
	*/
	uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;
	if constexpr (PRINT)
		std::cout << cc.GetBootPrecomputation(slots).correctionFactor << " " << deg << std::endl;
	double post = std::pow(2, static_cast<double>(deg));

	double pre		= 1. / post;
	uint64_t scalar = std::llround(post);

	//////////////////////////////////////////////////////////////////////

	{

		ModRaise(ctxt, slots, correction, prescaled);

		//------------------------------------------------------------------------------
		// SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
		//------------------------------------------------------------------------------

		// Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
		double k = cc.GetBootK();

		double constantEvalMult = pre * (1.0 / (k * cc.N));

		if constexpr (PRINT)
			std::cout << "mult: " << constantEvalMult << std::endl;
		ctxt.multScalar(constantEvalMult, false);

		if constexpr (PRINT) {
			std::cout << "Raise scaled ";
			for (auto& j : ctxt.c0.GPU) {
				cudaSetDevice(j.device);
				for (auto& i : j.limb) {
					SWITCH(i, printThisLimb(1));
				}
			}
			std::cout << std::endl;
		}

		////////////////////////////////////////////////////////////////

		Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
	}

	if (ctxt.NoiseLevel == 2) {
		ctxt.rescale();
	}

	//   std::cout << "LT" << std::endl;

	if (isLT) {
		EvalLinearTransform(ctxt, slots, false);
	} else {
		EvalCoeffsToSlots(ctxt, slots, false);
	}
	//  std::cout << "ModRed" << std::endl;

	if (cc.N / 2 == slots) {
		aux.conjugate(ctxt);
		Ciphertext ctxtEncI(cc_);
		ctxtEncI.sub(ctxt, aux);
		ctxt.add(aux);
		ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxt.rescale();
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxtEncI.rescale();

		approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
	} else {
		aux.conjugate(ctxt);
		ctxt.add(aux);
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxt.rescale();
		approxModReductionSparse(ctxt, scalar);
	}

	if (ctxt.NoiseLevel == 2) {
		ctxt.rescale();
	}

	//  std::cout << "LT" << std::endl;

	if (isLT) {
		EvalLinearTransform(ctxt, slots, true);
	} else {
		EvalCoeffsToSlots(ctxt, slots, true);
	}

	if (cc.N / 2 != slots) {
		aux.rotate(ctxt, slots);
		ctxt.add(aux);
	}

	uint64_t corFactor = (uint64_t)1 << std::llround(correction);
	multIntScalar(ctxt, corFactor);
	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "End bootstrap ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(2));
			}
		}
		std::cout << std::endl;
		cudaDeviceSynchronize();
	}
}

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled, const int levelsToDrop) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kBoot);
	CudaNvtxRange r(std::string{ sc::current().function_name() });

	assert(slots >= ctxt.slots);
	int old_slots = ctxt.slots;

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;
	Ciphertext aux(cc_);

	// VARIABLE-HEIGHT RAISE (opt-in; levelsToDrop == 0 is the behaviour this function has
	// always had). Stopping the mod-raise `levelsToDrop` towers below the chain top runs
	// CoeffsToSlots, EvalMod and SlotsToCoeffs on that many fewer limbs and returns the
	// ciphertext that many levels deeper -- for a caller that does not need the levels a
	// full-height raise would hand back. The transform diagonals must sit at the
	// ciphertext's own level (LinearTransform asserts it), so a shortened raise reads its
	// own precomputation; GetBootPrecomputation throws by name if it was never installed.
	if (levelsToDrop < 0)
		throw std::invalid_argument("FIDESlib: Bootstrap levelsToDrop must be non-negative");
	BootstrapPrecomputation& precom = cc.GetBootPrecomputation(slots, levelsToDrop);
	ChecksumProbe(ctxt, "in");
	bool isLT						= precom.LT.slots == slots;

	/////////////////////////////////////////////////////////////////////
	// NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
	uint64_t q	   = cc.prime[0].p;
	double qDouble = (double)q; // q.ConvertToDouble();

	if constexpr (PRINT) {
		std::cout << "q: " << q << " ";
		std::cout << qDouble << std::endl;
	}
	const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
	double powP	 = pow(2, p);

	if constexpr (PRINT) {
		std::cout << "p: " << p << std::endl;
	}
	int32_t deg = std::round(std::log2(qDouble / powP));
	/*
#if NATIVEINT != 128
	if (deg > static_cast<int32_t>(m_correctionFactor)) {
		OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
					  std::to_string(m_correctionFactor) + "].");
	}
#endif
	*/
	uint32_t correction = precom.correctionFactor - deg;
	if constexpr (PRINT)
		std::cout << precom.correctionFactor << " " << deg << std::endl;
	double post = std::pow(2, static_cast<double>(deg));

	double pre		= 1. / post;
	uint64_t scalar = std::llround(post);

	//////////////////////////////////////////////////////////////////////
	bool sparse_encaps = precom.sparse_encaps;

	{
		// In-context ENCAPS op order (stock ckksrns-fhe.cpp ModRaise):
		//   (1) keySwitchSparse(2N-4) at the INPUT level, before the raise  [inside ModRaise]
		//   (2) tower-0 reinterpret raise (same as non-ENCAPS)              [inside ModRaise]
		//   (3) keySwitch(2N-2) at the RAISED level                         [here, below]
		//   (4) multScalar(constantEvalMult)                                [here, below]
		// ROLE SWAP vs. the old dual-context path: 2N-4 (sparse) is now FIRST, 2N-2 SECOND.
		ModRaise(ctxt, slots, correction, prescaled, sparse_encaps, levelsToDrop);

		if (sparse_encaps) {
			// Stock ckksrns-fhe.cpp:843 — algo->KeySwitchInPlace(raised, evalKeyMap.at(2*N-2)).
			// Ordinary hybrid key switch at the raised level: sparse mod-raise secret -> dense
			// main secret. Loaded as a rotation key at index 2N-2 (see AddBootstrapKeys).
			ctxt.keySwitch(cc.GetRotationKey(2 * cc.N - 2, ctxt.keyID));
			EncapsFingerprint(ctxt, "post-switch-back");
		}
		ChecksumProbe(ctxt, "post-modraise");

		//------------------------------------------------------------------------------
		// SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
		//------------------------------------------------------------------------------

		// Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x).
		// For SPARSE_ENCAPSULATED, k = 1.0 (RawParams bootK, in-context rework step 5): the 1/K is baked
		// into the CtS precomputation, so constantEvalMult = pre*(1/(k*N)) matches stock's
		// EvalMultInPlace(raised, pre*(1.0/(k*N))) at ckksrns-fhe.cpp:892 with k=1.0.
		double k = cc.GetBootK();

		double constantEvalMult = pre * (1.0 / (k * cc.N));
		if (precom.cts_fold != 1.0)
			constantEvalMult *= precom.cts_fold;

		if constexpr (PRINT)
			std::cout << "mult: " << constantEvalMult << std::endl;
		ctxt.multScalar(constantEvalMult, false);
		if (MODRAISE_WITH_P0) {
			ctxt.rescale();
		}

		if constexpr (PRINT) {
			std::cout << "Raise scaled ";
			for (auto& j : ctxt.c0.GPU) {
				cudaSetDevice(j.device);
				for (auto& i : j.limb) {
					SWITCH(i, printThisLimb(1));
				}
			}
			std::cout << std::endl;
		}

		////////////////////////////////////////////////////////////////

		Accumulate(ctxt, precom.accumulate_bStep, slots, cc.N / 2 / slots);
	}

	ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

	if (ctxt.NoiseLevel == 2) {
		ctxt.rescale();
	}

	//   std::cout << "LT" << std::endl;

	ChecksumProbe(ctxt, "pre-cts");
	if (isLT) {
		EvalLinearTransform(ctxt, slots, false, levelsToDrop);
	} else {
		EvalCoeffsToSlots(ctxt, slots, false, levelsToDrop);
	}
	ChecksumProbe(ctxt, "post-cts");

	//  std::cout << "ModRed" << std::endl;

	if (cc.N / 2 == slots) {
		aux.conjugate(ctxt);
		Ciphertext ctxtEncI(cc_);
		ctxtEncI.sub(ctxt, aux);
		ctxt.add(aux);
		ChecksumProbe(ctxt, "post-conj");
		ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
		ChecksumProbe(ctxt, "post-monomial");
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxt.rescale();
		else if (ctxt.NoiseLevel == 2)
			ctxt.rescale();  // FIDESlib bit-compat: mirror OpenFHE's ModReduceInternal before the Chebyshev series
		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxtEncI.rescale();
		else if (ctxtEncI.NoiseLevel == 2)
			ctxtEncI.rescale();  // FIDESlib bit-compat: same for the imaginary component
		ChecksumProbe(ctxt, "pre-evalmod");
		approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar, precom.stc_fold);
		ChecksumProbe(ctxt, "post-evalmod");
	} else {
		aux.conjugate(ctxt);
		ctxt.add(aux);
		ChecksumProbe(ctxt, "post-conj");

		if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
			ctxt.rescale();
		else if (ctxt.NoiseLevel == 2)
			ctxt.rescale();  // FIDESlib bit-compat: mirror OpenFHE's ModReduceInternal before the Chebyshev series
		ChecksumProbe(ctxt, "pre-evalmod");
		approxModReductionSparse(ctxt, scalar, precom.stc_fold);
		ChecksumProbe(ctxt, "post-evalmod");
	}


	if (ctxt.NoiseLevel == 2) {
		ctxt.rescale();
	}

	//  std::cout << "LT" << std::endl;

	ChecksumProbe(ctxt, "pre-stc");
	if (isLT) {
		EvalLinearTransform(ctxt, slots, true, levelsToDrop);
	} else {
		EvalCoeffsToSlots(ctxt, slots, true, levelsToDrop);
	}
	ChecksumProbe(ctxt, "post-stc");

	if (cc.N / 2 != slots) {
		aux.rotate(ctxt, slots);
		ctxt.add(aux);
		ChecksumProbe(ctxt, "post-rot");
	}


	uint64_t corFactor = (uint64_t)1 << std::llround(correction);
	multIntScalar(ctxt, corFactor);
	ChecksumProbe(ctxt, "out");
	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "End bootstrap ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(2));
			}
		}
		std::cout << std::endl;
		cudaDeviceSynchronize();
	}

	// PROVENANCE. `multIntScalar` above is this bootstrap's LAST write to `ctxt`, so this is the
	// point at which the blocks backing `ctxt`'s limbs hold this lane's answer. Marking them here
	// is what lets a dump of a wrong output say which lane wrote the block and on which stream,
	// next to which lane freed it and which fence the taker inherited. One cached-bool read unless
	// FIDESLIB_POOL_TRACE=1 (and the concurrent mode) -- see PoolTraceMarkCiphertextOutput.
	PoolTraceMarkCiphertextOutput(ctxt);

	ctxt.slots = old_slots;
}

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
	ContextData& cc = *cc_;
	SetCurrentContext(cc_);
	/////////////////////////////////////////////////////////////////////
	// NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
	uint64_t q	   = cc.prime[0].p;
	double qDouble = (double)q; // q.ConvertToDouble();

	if constexpr (PRINT) {
		std::cout << "q: " << q << " ";
		std::cout << qDouble << std::endl;
	}
	const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
	double powP	 = pow(2, p);

	if constexpr (PRINT) {
		std::cout << "p: " << p << std::endl;
	}
	int32_t deg = std::round(std::log2(qDouble / powP));
	/*
	#if NATIVEINT != 128
		if (deg > static_cast<int32_t>(m_correctionFactor)) {
			OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
						  std::to_string(m_correctionFactor) + "].");
		}
	#endif
		*/
	uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;

	double res = 0.0;
	if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
		uint32_t lvl	   = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
		double targetSF	   = cc.param.ScalingFactorReal[cc.L - lvl];
		double sourceSF	   = cc.param.ScalingFactorReal[1]; // ciphertext->GetScalingFactor();
		uint32_t numTowers = 2;								// ciphertext->GetElements()[0].GetNumOfElements();
		double modToDrop   = static_cast<double>(cc.prime.at(numTowers - 1).p);
		// cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
		//  in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
		//  a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
		//  So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
		//  for NATIVEINT = 128.
		//  Scaling down the message by a correction factor to emulate using a larger q0.
		//  This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
		double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
		double pow				= std::pow((double)2.0, (double)-1.0 * (double)correction);
		adjustmentFactor *= pow;
		if constexpr (PRINT)
			std::cout << adjustmentFactor << std::endl;
		res = adjustmentFactor;
	} else { // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
			 // Scaling down the message by a correction factor to emulate using a larger q0.
			 // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
		res = std::pow((double)2.0, (double)-1.0 * (double)correction);
	}

	return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled, const bool sparse_encaps, const int levelsToDrop) {
	ConcurrentOpsDiagScope diag(ConcurrentOpsDiag::kModRaise);
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	ContextData& cc = ctxt.cc;
	// The top of the chain this raise reinterprets tower 0 onto. `levelsToDrop` towers below
	// cc.L the raise is just as exact -- broadcastLimb0 replicates the same residue into
	// however many towers were grown -- and everything downstream runs on fewer limbs.
	const int raiseTop = cc.L - levelsToDrop;
	if (levelsToDrop < 0 || raiseTop < 1)
		throw std::invalid_argument("FIDESlib: ModRaise levelsToDrop leaves no modulus chain to raise onto");
	//------------------------------------------------------------------------------
	// RAISING THE MODULUS
	//------------------------------------------------------------------------------

	// Input precondition. This used to be an assert(), i.e. absent from Release builds: the modulus raise below first
	// consumes the pending rescale of a NoiseLevel-2 input and then one more limb (multScalar + rescale before
	// dropToLevel(0)), so the input needs level >= NoiseLevel (NoiseLevel 1: level >= 1, NoiseLevel 2: level >= 2).
	// Otherwise rescale() runs on a single-limb polynomial and the result is silently garbage/NaN (or a crash).
	if (!prescaled) {
		if (ctxt.getLevel() < ctxt.NoiseLevel) {
			throw std::invalid_argument("FIDESlib::CKKS::Bootstrap: ciphertext at level " + std::to_string(ctxt.getLevel()) + " with NoiseLevel " +
			  std::to_string(ctxt.NoiseLevel) + " cannot be bootstrapped: the modulus raise requires level >= NoiseLevel, i.e. level >= " +
			  std::to_string(ctxt.NoiseLevel) + " for this input (rescale() a NoiseLevel-2 ciphertext or keep one more limb).");
		}
	} else {
		if (ctxt.getLevel() - ctxt.NoiseLevel + 1 != 0) {
			throw std::invalid_argument("FIDESlib::CKKS::Bootstrap(prescaled = true): expected level == NoiseLevel - 1, got level " +
			  std::to_string(ctxt.getLevel()) + " and NoiseLevel " + std::to_string(ctxt.NoiseLevel) + ".");
		}
	}
	// In FLEXIBLEAUTO, raising the ciphertext to a larger number
	// of towers is a bit more complex, because we need to adjust
	// it's scaling factor to the one that corresponds to the level
	// it's being raised to.
	// Increasing the modulus
	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "Initial ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb)
				SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
		cudaDeviceSynchronize();
		CudaCheckErrorMod;
	}
	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();
	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "Initial 2 ";
		CudaCheckErrorMod;
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb)
				SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
		std::cout << correction << std::endl;
		std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
		cudaDeviceSynchronize();
		CudaCheckErrorMod;
	}

	double targetSF = ctxt.NoiseFactor;
	if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
		uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
		if (MODRAISE_WITH_P0) {
			targetSF = cc.param.ScalingFactorReal[raiseTop - lvl]; // Will multiply with scalar scaled by P_0 and rescaled down by P_0.
		} else {
			targetSF = cc.param.ScalingFactorReal[raiseTop - lvl];
		}
		double sourceSF	   = ctxt.NoiseFactor;	  // ciphertext->GetScalingFactor();
		uint32_t numTowers = ctxt.getLevel() + 1; // ciphertext->GetElements()[0].GetNumOfElements();
		double modToDrop   = static_cast<double>(cc.prime.at(numTowers - 1).p);
		// cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

		// in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
		// a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
		// So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
		// for NATIVEINT = 128.
		// Scaling down the message by a correction factor to emulate using a larger q0.
		// This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
		double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
		double pow				= std::pow((double)2.0, (double)-1.0 * (double)correction);
		adjustmentFactor *= pow;
		if constexpr (PRINT)
			std::cout << adjustmentFactor << std::endl;

		if (!prescaled) {
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			ctxt.multScalar(adjustmentFactor);
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			// cc->EvalMultInPlace(ciphertext, adjustmentFactor);
			ctxt.rescale();
			ctxt.dropToLevel(0, true);
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
		} else {
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Prescale path ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			if (ctxt.NoiseLevel == 2) {
				ctxt.dropToLevel(1, true);
				ctxt.rescale();
			} else {
				ctxt.dropToLevel(0, true);
			}
		}
		ctxt.NoiseFactor = targetSF;
	} else { // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
			 // Scaling down the message by a correction factor to emulate using a larger q0.
			 // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
		if (!prescaled) {
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			ctxt.rescale();
			ctxt.dropToLevel(0);
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Initial ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
		} else {
			if constexpr (PRINT) {
				cudaDeviceSynchronize();
				std::cout << "Prescale path ";
				for (auto& j : ctxt.c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& i : j.limb)
						SWITCH(i, printThisLimb(1));
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
				CudaCheckErrorMod;
			}
			if (ctxt.NoiseLevel == 2) {
				ctxt.dropToLevel(1);
				ctxt.rescale();
			} else {
				ctxt.dropToLevel(0);
			}
		}
	}

	if (sparse_encaps) {
		// Stock ckksrns-fhe.cpp:830 — raised = KeySwitchSparse(raised, evalKeyMap.at(2*N-4)).
		// The 2N-4 GHS single-digit (q0,p) sparse switch, at the INPUT level, BEFORE the raise:
		// transforms from the dense main secret to the sparse mod-raise secret. In-context — no
		// second context, no reinterpretContext, no NoiseFactor override (the old dual-context
		// path lost the scale metadata across contexts and re-set NoiseFactor = targetSF here;
		// the in-context keySwitch preserves the scale set by the adjustment above, so the reset
		// is gone). See in-context rework step 4.
		keySwitchSparse(ctxt, cc.GetRotationKey(2 * cc.N - 4, ctxt.keyID));
		EncapsFingerprint(ctxt, "post-sparse-switch");
	}

	//   std::cout << "Boot start " << std::endl;
	// auto ctxtDCRT = raised->GetElements();
	if constexpr (PRINT) {
		std::cout << "Adjustment 1: ";
		CudaCheckErrorMod;
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}

	ctxt.c0.INTT(cc.batch, true);

	if constexpr (PRINT) {
		CudaCheckErrorMod;
		std::cout << "Adjustment ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	//   std::cout << "Grow" << std::endl;
	ctxt.c0.grow(raiseTop - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
	if (MODRAISE_WITH_P0) {
		ctxt.c0.generateSpecialLimbs(false, true);
		ctxt.c0.setLevel(raiseTop + 1);
	}
	//   std::cout << "Broadcast" << std::endl;
	if constexpr (PRINT) {
		CudaCheckErrorMod;
		std::cout << "Adjustment ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	ctxt.c0.broadcastLimb0();
	if constexpr (PRINT) {
		CudaCheckErrorMod;
		std::cout << "Adjustment ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	ctxt.c0.NTT(cc.batch, true);
	// std::cout << cc.batch << std::endl;
	if constexpr (PRINT) {
		std::cout << "ModRaise ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	ctxt.c1.INTT(cc.batch, true);
	if constexpr (PRINT) {
		std::cout << "Adjustment c1 ";
		for (auto& j : ctxt.c1.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	//  std::cout << "Grow" << std::endl;
	ctxt.c1.grow(raiseTop - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
	if (MODRAISE_WITH_P0) {
		ctxt.c1.generateSpecialLimbs(false, true);
		ctxt.c1.setLevel(raiseTop + 1);
	}
	//  std::cout << "Broadcast" << std::endl;
	if constexpr (PRINT) {
		std::cout << "Adjustment c1  ";
		for (auto& j : ctxt.c1.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	ctxt.c1.broadcastLimb0();
	if constexpr (PRINT) {
		std::cout << "Adjustment c1";
		for (auto& j : ctxt.c1.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}
	ctxt.c1.NTT(cc.batch, true);
	if constexpr (PRINT) {
		std::cout << "Adjustment c1";
		for (auto& j : ctxt.c1.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
	}

	// Fingerprint point 2: post-raise (tower-0 reinterpret done, before the 2N-2
	// switch-back). Diff vs. stock CPU prints at ckksrns-fhe.cpp:833-840.
	if (sparse_encaps)
		EncapsFingerprint(ctxt, "post-raise");

	ctxt.slots = cc.N / 2;
}
