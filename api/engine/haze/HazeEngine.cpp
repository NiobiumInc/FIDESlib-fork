#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeEngine.hpp"

#include "engine/haze/HazeScalarEncode.hpp"

#include "CryptoContext.hpp"
#include "engine/EngineCommon.hpp"

#include <functional>

#include <haze/haze.h>
#include <haze/replay_bridge.h>

#include <openfhe.h>

#include <any>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fideslib {

namespace {

using hazebk::HazePayload;
using hazebk::HazePtPayload;
using hazebk::LimbChain;
using hazebk::Residency;

void hazeCheck(hazeError_t err, const char* what) {
	if (err != HAZE_SUCCESS) {
		throw std::runtime_error(std::string("haze backend: ") + what + " failed: " + hazeGetErrorString(err));
	}
}

// Unwrap the haze payload from the value types' `device` std::any (parity with engine/cuda's
// deviceCt/devicePt). An empty slot means "not resident"; the engine ensures residency via
// loadCiphertext/loadPlaintext before every use.
std::shared_ptr<HazePayload> devicePayload(const CiphertextImpl<DCRTPoly>& ct) {
	return std::any_cast<std::shared_ptr<HazePayload>>(ct.device);
}

std::shared_ptr<HazePayload> devicePayload(const Ciphertext<DCRTPoly>& ct) {
	return devicePayload(*ct);
}

std::shared_ptr<HazePtPayload> devicePtPayload(const Plaintext& pt) {
	return std::any_cast<std::shared_ptr<HazePtPayload>>(pt->device);
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& hostContext(CryptoContextImpl<DCRTPoly>& ctx) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& hostCt(const CiphertextImpl<DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(const_cast<CiphertextImpl<DCRTPoly>&>(ct).host);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& hostCt(const Ciphertext<DCRTPoly>& ct) {
	return hostCt(*ct);
}

// Per-tower uint64 rows of one DCRTPoly element, forced to EVALUATION form on a copy
// (fresh OpenFHE ciphertexts/plaintexts are EVAL already; the copy keeps the host value
// untouched if it was COEFFICIENT).
std::vector<std::vector<uint64_t>> extractRows(const lbcrypto::DCRTPoly& src, uint64_t ringDim) {
	lbcrypto::DCRTPoly elem = src;
	elem.SetFormat(Format::EVALUATION);
	const std::size_t towers = elem.GetNumOfElements();
	std::vector<std::vector<uint64_t>> rows(towers);
	for (std::size_t t = 0; t < towers; ++t) {
		const auto& np	 = elem.GetElementAtIndex(static_cast<uint32_t>(t));
		const auto& vals = np.GetValues();
		if (vals.GetLength() != ringDim) {
			throw std::runtime_error("haze backend: polynomial length " + std::to_string(vals.GetLength()) + " does not match ring dimension " + std::to_string(ringDim));
		}
		rows[t].resize(ringDim);
		for (std::size_t i = 0; i < ringDim; ++i) {
			rows[t][i] = vals[i].template ConvertToInt<uint64_t>();
		}
	}
	return rows;
}

// Wrap a device-computed result in a new value-type Ciphertext parented to `ctx`. The host
// slot shares `proto`'s lbcrypto ciphertext as a lazily-detached readback shell (it is stale
// until recoverHostCiphertext overwrites it); the device slot holds the new payload.
Ciphertext<DCRTPoly> wrapDeviceResult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& proto, std::shared_ptr<HazePayload> payload) {
	Ciphertext<DCRTPoly> res = std::make_shared<CiphertextImpl<DCRTPoly>>(ctx.self_reference.lock());
	res->host				 = proto->host;
	res->need_lazy_copy		 = true;
	res->device				 = std::make_any<std::shared_ptr<HazePayload>>(std::move(payload));
	return res;
}

} // namespace

// ---- Lifecycle ----

void HazeEngine::loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) {
	if (loaded_) {
		return;
	}
	// haze configuration is process-global (it locks at hazeConfigureDevice), so two
	// concurrently-loaded haze contexts cannot work. Sequential contexts are fine: the
	// previous context's teardown() resets haze and decrements the guard. (The check and
	// the later increment are not atomic together — concurrent loadContext from two threads
	// is not supported, matching haze's own single-threaded configuration model.)
	if (liveEngines_.load() != 0) {
		throw std::runtime_error("haze backend: another haze context is already loaded in this process; "
					  "haze configuration is process-global, so destroy the previous context first");
	}

	auto& context = hostContext(ctx);
	ringDim_	  = context->GetRingDimension();
	polyBytes_	  = static_cast<size_t>(ringDim_) * sizeof(uint64_t);

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	if (!cryptoParams) {
		throw std::runtime_error("haze backend: context does not carry CKKS-RNS crypto parameters");
	}
	scalingTech_ = static_cast<int>(cryptoParams->GetScalingTechnique());

	qBase_.clear();
	const auto& qParams = cryptoParams->GetElementParams()->GetParams();
	qBase_.reserve(qParams.size());
	for (const auto& p : qParams) {
		qBase_.push_back(p->GetModulus().template ConvertToInt<uint64_t>());
	}
	if (qBase_.empty()) {
		throw std::runtime_error("haze backend: context has an empty modulus chain");
	}
	pBase_.clear();
	if (const auto paramsP = cryptoParams->GetParamsP()) {
		const auto& pParams = paramsP->GetParams();
		pBase_.reserve(pParams.size());
		for (const auto& p : pParams) {
			pBase_.push_back(p->GetModulus().template ConvertToInt<uint64_t>());
		}
	}
	// haze's ciphertext-modulus table holds 64 entries (kMaxCiphertextModuli), but the last slot
	// is reserved for the aux prime the device generates at hazeConfigureDevice, so at most 63
	// may be supplied.
	if (qBase_.size() + pBase_.size() > 63) {
		throw std::runtime_error("haze backend: |Q|+|P| = " + std::to_string(qBase_.size()) + "+" + std::to_string(pBase_.size()) + " exceeds haze's 63 supplied moduli (64-entry table, last slot reserved for the generated aux prime)");
	}

	// Per-level scale-factor caches, OpenFHE level orientation.
	const size_t sizeQ = qBase_.size();
	sfReal_.resize(sizeQ);
	for (size_t lvl = 0; lvl < sizeQ; ++lvl) {
		sfReal_[lvl] = cryptoParams->GetScalingFactorReal(static_cast<uint32_t>(lvl));
	}
	sfRealBig_.resize(sizeQ > 0 ? sizeQ - 1 : 0); // OpenFHE sizes m_scalingFactorsRealBig at sizeQ-1
	for (size_t lvl = 0; lvl + 1 < sizeQ; ++lvl) {
		sfRealBig_[lvl] = cryptoParams->GetScalingFactorRealBig(static_cast<uint32_t>(lvl));
	}
	modReduceFactor_.resize(sizeQ); // tower-indexed GetModReduceFactor(i): approxSF for FIXED modes, double(q_i) for FLEXIBLE
	for (size_t i = 0; i < sizeQ; ++i) {
		modReduceFactor_[i] = cryptoParams->GetModReduceFactor(static_cast<uint32_t>(i));
	}

	// Host key extraction (HazeKeyExtract; host-only, before any haze state is touched):
	// relin key if EvalMultKeyGen ran, one rotation key per registered slot step (keyed by
	// step; FindAutomorphismIndex handles negative steps). Limbs are H2D-uploaded lazily at
	// the first keyswitch use and then reused as program inputs every epoch.
	const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);
	keyTag_			   = pkImpl->GetKeyTag();
	plaintextModulus_  = cryptoParams->GetPlaintextModulus();
	numPartQ_		   = cryptoParams->GetNumPartQ(); // hybrid digit count (#19 hoisting prefix)
	const auto& keyMap = context->GetAllEvalMultKeys();
	if (keyMap.find(keyTag_) != keyMap.end()) {
		relinKey_	   = KsKey{};
		relinKey_.host = hazebk::extractEvalMultKeyLimbs(context, keyTag_);
		haveRelinKey_  = true;
	}
	rotKeys_.clear();
	autoKeys_.clear();
	sparseKey_.reset();
	monomials_.clear();
	for (const int32_t step : ctx.rotation_indexes) {
		if (rotKeys_.count(step) != 0) {
			continue;
		}
		RotKey rk;
		rk.autoIndex = context->FindAutomorphismIndex(step);
		rk.key.host	 = hazebk::extractAutomorphismKeyLimbs(context, keyTag_, rk.autoIndex);
		rotKeys_.emplace(step, std::move(rk));
	}
	// Index the registered keys by CUDA's normalized rotation index ((step mod N/2), positive),
	// so rotateCore can replicate GetRotationKey's slots-aware actual_index selection (#7).
	rotKeyIndex_.clear();
	if (ringDim_ >= 2) {
		const int32_t half = static_cast<int32_t>(ringDim_ / 2);
		for (const auto& [step, rk] : rotKeys_) {
			const int32_t idx = ((step % half) + half) % half;
			rotKeyIndex_.emplace(idx, step); // first registered step wins for a given normalized index
		}
	}
	// Bootstrap precomputation per registered slot count (host-only). The rotation
	// and conjugation keys it needs resolve lazily through autoKeyFor at first use.
	boot_.clear();
	for (const uint32_t slots : ctx.slots_bootstrap) {
		extractBootPrecom(ctx, slots);
	}

	// haze bring-up, exact order (program info/directory must precede the first
	// H2D or compute call — the first of either brings up the compiler backend). On any
	// failure, reset haze so a partially-configured process-global state cannot survive
	// (hazeConfigureDevice cannot be re-attempted without a preceding hazeDeviceReset).
	try {
		// One-shot immutable device configuration (haze one-shot config contract): all bring-up
		// values are gathered into two caller-owned structs and handed to hazeConfigureDevice in
		// a single call. haze copies what it needs and retains no pointer, but both structs (and
		// the storage they point at — the modulus vector and programDir_) must stay alive across
		// the call. hazeConfigureDevice cannot be re-attempted without a preceding hazeDeviceReset.
		hazeReplayConfig replay{};
		replay.program_name		   = "fideslib";
		replay.program_version	   = "1.0";
		replay.program_description = "FIDESlib HazeEngine";

		// FBC mode (constructor-selected), latched at bring-up before the first compute:
		// reduced-noise selects the centered FBC matching a WITH_REDUCED_NOISE OpenFHE reference
		// (bit-exact parity). It is the only mode there is to select — an algorithmic choice, not
		// a data format. haze records ordinary-form traces for every target; the hardware data
		// format (Montgomery residues, bit-reversed order) is applied by the replay driver, which
		// decides from the target's device spec (montgomery_enabled) and decodes results back
		// before they are read.
		replay.reduced_noise = reducedNoise() ? 1 : 0;
		// Replay target (default: in-process local simulator, i.e. NULL). This is also the sole
		// selector of the hardware datapath: point it at a Montgomery device (func_sim_hw, any
		// fpga*) to get one. Configure it before the bridge brings up the compiler so replay
		// dispatch routes there.
		const char* tgt = std::getenv("FIDESLIB_HAZE_TARGET");
		if (tgt != nullptr && tgt[0] != '\0')
			replay.target = tgt;

		// Program directory: <FIDESLIB_HAZE_RUNS_DIR | /tmp/fideslib-haze-runs>/<key>/ctx<N>,
		// where <key> is the pid unless FIDESLIB_HAZE_PROGRAM_KEY overrides it.
		// Each flush writes a multi-hundred-MB project dir, so (a) keep it off slow shared
		// filesystems and (b) clean it up at teardown unless FIDESLIB_HAZE_KEEP_RUNS is set.
		//
		// The pid keeps concurrent processes, and successive runs of *different* programs, from
		// colliding. A caller-supplied key deliberately gives that up so that successive runs of
		// the SAME program reuse one directory: the lowered artifacts persist there and the
		// replay skips lowering the second time (measured 14.4 s of a ~19 s re-replay, against
		// ~85 s when the project is fresh). The caller owns the key's correctness — reusing a
		// directory across two genuinely different programs would replay the wrong one.
		// Implies KEEP_RUNS; see teardown().
		static std::atomic<int> ctxCounter{ 0 };
		const char* runsRoot   = std::getenv("FIDESLIB_HAZE_RUNS_DIR");
		const char* progKey	   = std::getenv("FIDESLIB_HAZE_PROGRAM_KEY");
		const std::string root = (runsRoot != nullptr && runsRoot[0] != '\0') ? runsRoot : "/tmp/fideslib-haze-runs";
		const std::string key =
			(progKey != nullptr && progKey[0] != '\0') ? std::string(progKey) : std::to_string(getpid());
		programDir_			   = root + "/" + key + "/ctx" + std::to_string(ctxCounter.fetch_add(1));
		std::filesystem::create_directories(programDir_);
		replay.program_directory = programDir_.c_str();

		// FHE parameters. The full Q∥P chain is conveyed as one contiguous moduli array; leave
		// the twiddle generators unset — haze's e2e suite never sets them and is bit-exact vs
		// OpenFHE (FHETCH derives per-modulus roots internally). Documented fallback if a context
		// ever disagrees: per-limb GetRootOfUnity().
		std::vector<uint64_t> moduli;
		moduli.reserve(qBase_.size() + pBase_.size());
		moduli.insert(moduli.end(), qBase_.begin(), qBase_.end());
		moduli.insert(moduli.end(), pBase_.begin(), pBase_.end());

		hazeFheParams fhe{};
		fhe.ring_dim		   = ringDim_;
		fhe.moduli			   = moduli.data();
		fhe.moduli_count	   = moduli.size();
		fhe.twiddle_generators = nullptr;
		fhe.twiddle_count	   = 0;

		// Configure the device before the bridge: the one-shot config contract requires
		// hazeConfigureDevice to run first, then hazeReplayBridgeInitCryptoContext.
		hazeCheck(hazeConfigureDevice(&fhe, &replay), "hazeConfigureDevice");

		// Pure-C bridge: haze rebuilds a matching CryptoContext from (ring_dim, first Q prime).
		// Must be re-called after every hazeDeviceReset (replay_bridge.h contract), i.e. on every
		// loadContext. `picked` is now advisory only — the authoritative primes are the ones
		// passed above via fhe.moduli — but a zero return still signals a failed bring-up.
		uint64_t picked = 0;
		hazeCheck(hazeReplayBridgeInitCryptoContext(ringDim_, qBase_.front(), &picked), "hazeReplayBridgeInitCryptoContext");
		if (picked == 0) {
			throw std::runtime_error("haze backend: replay bridge returned a zero modulus");
		}
	} catch (...) {
		hazeReplayBridgeReset();
		(void)hazeDeviceReset();
		programDir_.clear();
		throw;
	}

	liveEngines_.fetch_add(1);
	loaded_	  = true;
	executed_ = false;
}

void HazeEngine::teardown() {
	if (!loaded_) {
		return;
	}
	outputs_.clear();
	// Drop uploaded key chains before the device reset frees their meaning; the host limbs
	// are re-extracted by the next loadContext.
	relinKey_	  = KsKey{};
	haveRelinKey_ = false;
	rotKeys_.clear();
	autoKeys_.clear();
	sparseKey_.reset();
	monomials_.clear();
	boot_.clear();
	numPartQ_ = 0;
	// Reset order per replay_bridge.h: drop the bridge's cached CryptoContexts so they don't
	// outlive the device reset, then clear all process-global haze state. Outstanding
	// LimbChain destructors of still-live ciphertexts will see stale addresses afterwards;
	// they swallow the resulting hazeFree errors (ops.cpp free_all precedent).
	hazeReplayBridgeReset();
	(void)hazeDeviceReset();
	loaded_	  = false;
	executed_ = false;
	liveEngines_.fetch_sub(1);

	// Each context's program dir holds a full FHETCH project (hundreds of MB per flush);
	// gtest runs one context per test, so keeping them would fill /tmp across a suite run.
	// A caller-supplied program key means the directory is meant to outlive this process:
	// deleting it would destroy the lowered artifacts it exists to preserve.
	const bool keepRuns = std::getenv("FIDESLIB_HAZE_KEEP_RUNS") != nullptr ||
						  std::getenv("FIDESLIB_HAZE_PROGRAM_KEY") != nullptr;
	if (!programDir_.empty() && !keepRuns) {
		std::error_code ec;
		std::filesystem::remove_all(programDir_, ec); // best-effort; ignore errors
	}
	programDir_.clear();
}

HazeEngine::~HazeEngine() {
	teardown();
}

bool HazeEngine::isContextLoaded() const {
	return loaded_;
}

void HazeEngine::synchronize() const {
	// No-op: haze records synchronously; nothing runs in the background and the
	// program executes only at the first readback's flush.
}

void HazeEngine::setDevices(const std::vector<int>& /*devices*/) {
	// haze exposes exactly one logical device.
}

std::vector<int> HazeEngine::devices() const {
	return { 0 };
}

// ---- Residency ----

void HazeEngine::loadCiphertext(CryptoContextImpl<DCRTPoly>& /*ctx*/, Ciphertext<DCRTPoly>& ct) {
	if (ct->device.has_value()) {
		return;
	}
	if (!loaded_) {
		throw std::runtime_error("CryptoContext not loaded to any device");
	}
	auto& host			 = hostCt(ct);
	const auto& elements = host->GetElements();
	// Degree 1 (2 components) or degree 2 (3 components, an un-relinearized EvalMultNoRelin
	// result). Higher degrees have no keyswitch path here — only one relin key exists.
	if (elements.size() != 2 && elements.size() != 3) {
		throw std::runtime_error("haze backend: only degree-1 (2 components) and degree-2 (3 components) ciphertexts are supported; got " + std::to_string(elements.size()));
	}

	auto payload = std::make_shared<HazePayload>();
	payload->components.reserve(elements.size());
	for (const auto& element : elements) {
		auto rows = extractRows(element, ringDim_);
		if (payload->components.empty()) {
			payload->towers = rows.size();
		} else if (rows.size() != payload->towers) {
			throw std::runtime_error("haze backend: ciphertext components disagree on tower count");
		}
		payload->components.push_back(LimbChain(rows, qPrefix(rows.size())));
	}
	payload->noiseScaleDeg = host->GetNoiseScaleDeg();
	payload->scalingFactor = host->GetScalingFactor();
	payload->slots		   = host->GetSlots();
	payload->state		   = Residency::Uploaded;

	ct->device = std::make_any<std::shared_ptr<HazePayload>>(std::move(payload));

	// Opt-in: harnesses that never read an uploaded input's host residues can drop them here
	// instead of retaining one host copy per uploaded input for the context's lifetime (default off).
	if (dropInputHostAfterUpload_) {
		host->SetElements(std::vector<lbcrypto::DCRTPoly>{});
	}
}

void HazeEngine::loadPlaintext(CryptoContextImpl<DCRTPoly>& /*ctx*/, Plaintext& pt) {
	if (pt->device.has_value()) {
		return;
	}
	if (!loaded_) {
		throw std::runtime_error("CryptoContext not loaded to any device");
	}
	const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->host);

	auto rows = extractRows(ptImpl->GetElement<lbcrypto::DCRTPoly>(), ringDim_);

	// A plaintext is encoded either over a prefix of Q or over the extended basis Q_l∥P. The
	// bootstrap linear-transform precomputations are the extended case — OpenFHE builds them in
	// P*Q so the BSGS stages can multiply without a mod-down per baby step
	// (EvalCoeffsToSlotsPrecompute, EvalLinearTransformPrecompute) — and extBsgsStage reads them
	// back as `towers` Q rows plus |P| tail rows.
	//
	// Which shape it is must be read off the plaintext's own moduli, never inferred from the row
	// count: an extended encoding at a low residual level (the SlotsToCoeffs direction runs at
	// ~11 Q towers, after EvalMod has consumed most of the chain) has FEWER rows than |Q|, so a
	// count test silently mislabels it a Q prefix. The primes named here are what the replay
	// driver converts each row to Montgomery form under, so mislabelling them corrupts the value
	// on every montgomery_enabled target while leaving the plain simulator correct.
	const auto& element = ptImpl->GetElement<lbcrypto::DCRTPoly>();
	const bool extended = [&] {
		if (pBase_.empty() || rows.size() <= pBase_.size()) {
			return false;
		}
		const size_t tail = rows.size() - pBase_.size();
		for (size_t i = 0; i < pBase_.size(); ++i) {
			if (element.GetElementAtIndex(static_cast<uint32_t>(tail + i)).GetModulus().template ConvertToInt<uint64_t>() != pBase_[i]) {
				return false;
			}
		}
		return true;
	}();
	std::vector<uint64_t> base;
	if (extended) {
		if ((rows.size() - pBase_.size()) > qBase_.size()) {
			throw std::runtime_error("haze backend: plaintext has " + std::to_string(rows.size()) + " rows ending in the P basis, but its Q part exceeds |Q| = " + std::to_string(qBase_.size()));
		}
		base = qPrefix(rows.size() - pBase_.size());
		base.insert(base.end(), pBase_.begin(), pBase_.end());
	} else {
		if (rows.size() > qBase_.size()) {
			throw std::runtime_error("haze backend: plaintext has " + std::to_string(rows.size()) + " rows, which is neither a Q prefix (|Q| = " + std::to_string(qBase_.size())
									 + ") nor an extended Q_l∥P encoding (|P| = " + std::to_string(pBase_.size()) + ")");
		}
		base = qPrefix(rows.size());
	}

	auto payload			  = std::make_shared<HazePtPayload>();
	payload->towers			  = rows.size();
	payload->chain			  = LimbChain(rows, base);
	payload->noiseScaleDeg = ptImpl->GetNoiseScaleDeg();
	payload->scalingFactor = ptImpl->GetScalingFactor();
	payload->slots		   = ptImpl->GetSlots();

	pt->device = std::make_any<std::shared_ptr<HazePtPayload>>(std::move(payload));
}

std::shared_ptr<HazePayload> HazeEngine::ensureCt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	return devicePayload(ct);
}

// ---- One-program execution ----

void HazeEngine::markOutput(CryptoContextImpl<DCRTPoly>& /*ctx*/, Ciphertext<DCRTPoly>& ct) {
	if (!ct->device.has_value()) {
		return; // host-only value: a Decrypt reads ct->host directly, nothing to declare
	}
	auto payload = devicePayload(ct);
	if (payload->state != Residency::Recorded) {
		return; // Uploaded/Flushed values are D2H-readable without a tag
	}
	if (executed_) {
		throw std::runtime_error("haze backend: value was a program-local; declare it with MarkOutput before the first Decrypt");
	}
	for (const auto& existing : outputs_) {
		if (existing.lock() == payload) {
			return;
		}
	}
	outputs_.emplace_back(payload);
}

void HazeEngine::materialize() {
	// Tag every residue chain of every declared output (tagging one residue of an MRP group
	// tags the group, but tag each chain to be explicit — ops.cpp tag_ct), then run the
	// program's single flush. A tag is readback DMA on real hardware: only declared outputs,
	// never intermediates.
	for (const auto& weak : outputs_) {
		auto payload = weak.lock();
		if (!payload || payload->state != Residency::Recorded) {
			continue;
		}
		for (const auto& comp : payload->components) {
			for (std::size_t t = 0; t < payload->towers; ++t) {
				hazeCheck(hazeTagOutput(comp[t]), "hazeTagOutput");
			}
		}
	}
	hazeCheck(hazeFlush(), "hazeFlush");
	for (const auto& weak : outputs_) {
		if (auto payload = weak.lock(); payload && payload->state == Residency::Recorded) {
			payload->state = Residency::Flushed;
		}
	}
	outputs_.clear();
	executed_ = true;
}

void HazeEngine::requireComputable(HazePayload& p, const char* op) {
	(void)p; // operand residency no longer matters: any compute after the single flush is illegal
	if (executed_) {
		throw std::runtime_error(std::string("haze backend: ") + op + ": compute after readback is a new epoch; restructure the program so all compute precedes the first Decrypt");
	}
}

// ---- Readback ----

void HazeEngine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	if (!ct->device.has_value()) {
		return; // host-resident; nothing to read back
	}
	auto payload = devicePayload(ct);
	if (payload->state == Residency::Recorded) {
		if (executed_) {
			throw std::runtime_error("haze backend: value was a program-local; declare it with MarkOutput before the first Decrypt");
		}
		markOutput(ctx, ct); // implicit declaration of the value being read (first-readback case)
		materialize();		 // tags ALL declared outputs, ONE hazeFlush, sets executed_
	}

	refreshHostShadow(ctx, *ct);
}

// Pure shadow refresh: no output declaration, no flush. Splitting this out of recoverHostCiphertext
// lets GetElements() sync the host value without the MarkOutput/materialize step, which needs the
// owning shared_ptr and would silently execute the program.
void HazeEngine::refreshHostShadow(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return; // host-resident; nothing to read back
	}
	auto payload = devicePayload(ct);
	if (payload->state == Residency::Recorded) {
		// Still a program-local: its residues do not exist yet. Forcing a flush here would turn a
		// plain getter into program execution, so this is a hard error instead.
		throw std::runtime_error("haze backend: ciphertext has not been flushed; declare it with MarkOutput and Decrypt before reading its elements");
	}

	ct.EnsureLazyHostCopy();
	auto& host = hostCt(ct);

	// Make the host shell carry exactly payload->components.size() elements of payload->towers
	// limbs each. The element count follows the degree (EvalMultNoRelin grows the shell 2 -> 3, an
	// in-place Relinearize shrinks it back), so a mismatch is reshaped here; the limb count follows
	// the level — grow via a zero ciphertext at the full level when the device result has MORE limbs
	// (post-ModRaise — CudaEngine:536-545), then trim down with DropLastElements (no arithmetic).
	// Check the element count BEFORE indexing GetElements()[0]: an upload-dropped or
	// never-populated host shell has zero elements, and indexing it first is out of bounds.
	const size_t comps = payload->components.size();
	bool needsFresh	   = host->GetElements().size() != comps;
	size_t hostTowers  = 0;
	if (!needsFresh) {
		hostTowers = host->GetElements()[0].GetNumOfElements();
		needsFresh = hostTowers < payload->towers;
	}
	if (needsFresh) {
		auto& context		  = hostContext(ctx);
		const auto elemParams = context->GetCryptoParameters()->GetElementParams();
		lbcrypto::DCRTPoly zero(elemParams, Format::EVALUATION, true);
		auto fresh = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*host);
		fresh->SetElements(std::vector<lbcrypto::DCRTPoly>(comps, zero));
		host	   = fresh;
		hostTowers = host->GetElements()[0].GetNumOfElements();
	}
	if (hostTowers > payload->towers) {
		for (auto& elem : host->GetElements()) {
			elem.DropLastElements(hostTowers - payload->towers);
		}
	}

	// Pure shadow D2H per limb (Uploaded: the upload bytes; Flushed: the program's result).
	std::vector<uint64_t> buf(ringDim_);
	for (size_t e = 0; e < comps; ++e) {
		const LimbChain& chain = payload->components[e];
		auto& towersVec		   = host->GetElements()[e].GetAllElements();
		for (std::size_t t = 0; t < payload->towers; ++t) {
			hazeCheck(hazeMemcpy(buf.data(), chain[t], polyBytes_, HAZE_MEMCPY_DEVICE_TO_HOST), "hazeMemcpy(D2H)");
			auto& np = towersVec[t];
			lbcrypto::NativeVector nv(static_cast<uint32_t>(ringDim_), np.GetModulus());
			for (std::size_t i = 0; i < ringDim_; ++i) {
				nv[i] = lbcrypto::NativeInteger(buf[i]);
			}
			np.SetValues(std::move(nv), np.GetFormat());
		}
	}

	host->SetLevel(qBase_.size() - payload->towers);
	host->SetNoiseScaleDeg(payload->noiseScaleDeg);
	host->SetScalingFactor(payload->scalingFactor); // parity-critical
	host->SetSlots(payload->slots);
}

// ---- Ciphertext backend hooks ----

std::any HazeEngine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& src) {
	if (!src.device.has_value()) {
		return std::any{}; // not resident; the value type copies the host value itself
	}
	auto srcPayload		 = devicePayload(src);
	auto clone			 = std::make_shared<HazePayload>();
	clone->towers		 = srcPayload->towers;
	clone->noiseScaleDeg = srcPayload->noiseScaleDeg;
	clone->scalingFactor = srcPayload->scalingFactor;
	clone->slots		 = srcPayload->slots;
	clone->components.reserve(srcPayload->components.size());

	if (srcPayload->state == Residency::Recorded) {
		// In-flight value: record per-residue pass-through D2D copies (legal mid-epoch;
		// epoch.cpp copy_device_to_device). The clone is itself Recorded.
		const auto base = qPrefix(srcPayload->towers);
		for (const auto& srcComp : srcPayload->components) {
			LimbChain comp(srcPayload->towers, polyBytes_);
			hazeCheck(hazeMemcpyMrp(comp.data(), srcComp.asConst().data(), polyBytes_, HAZE_MEMCPY_DEVICE_TO_DEVICE, base.data(), base.size()), "hazeMemcpyMrp(D2D)");
			clone->components.push_back(std::move(comp));
		}
		clone->state = Residency::Recorded;
	} else {
		// Uploaded/Flushed: shadows hold the bytes — D2H to scratch + H2D into fresh buffers
		// (no epoch interaction). The clone is a fresh program input.
		auto pull = [&](const LimbChain& chain) {
			std::vector<std::vector<uint64_t>> rows(srcPayload->towers, std::vector<uint64_t>(ringDim_));
			for (std::size_t t = 0; t < srcPayload->towers; ++t) {
				hazeCheck(hazeMemcpy(rows[t].data(), chain[t], polyBytes_, HAZE_MEMCPY_DEVICE_TO_HOST), "hazeMemcpy(D2H)");
			}
			return rows;
		};
		for (const auto& srcComp : srcPayload->components) {
			clone->components.push_back(LimbChain(pull(srcComp), qPrefix(srcPayload->towers)));
		}
		clone->state = Residency::Uploaded;
	}
	return std::make_any<std::shared_ptr<HazePayload>>(std::move(clone));
}

size_t HazeEngine::ciphertextLevel(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetLevelHost();
	}
	// OpenFHE level = |Q| − towers, directly (no FIDESlib depth inversion).
	return qBase_.size() - devicePayload(ct)->towers;
}

size_t HazeEngine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetNoiseScaleDegHost();
	}
	return devicePayload(ct)->noiseScaleDeg;
}

double HazeEngine::ciphertextScalingFactor(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetScalingFactorHost();
	}
	return devicePayload(ct)->scalingFactor;
}

size_t HazeEngine::ciphertextSlots(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetSlotsHost();
	}
	return devicePayload(ct)->slots;
}

size_t HazeEngine::ciphertextNumElements(CryptoContextImpl<DCRTPoly>& /*ctx*/, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetNumElementsHost();
	}
	// The host shadow of a device-resident value is a stale shell (it still has 2 elements after an
	// EvalMultNoRelin), so the degree comes from the payload. No flush and no refresh: a Recorded
	// payload's component count is host-side bookkeeping, and the facade's degree-1 guard calls
	// this on every guarded op.
	return devicePayload(ct)->components.size();
}

void HazeEngine::requireRelinearized(const HazePayload& p, const char* op) const {
	if (p.components.size() != 2) {
		throw std::runtime_error(std::string("haze backend: ") + op + ": ciphertext should be relinearized before (has " + std::to_string(p.components.size()) + " components; call Relinearize)");
	}
}

void HazeEngine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>& /*ctx*/, CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	if (!ct.device.has_value()) {
		ct.SetSlotsHost(slots);
		return;
	}
	devicePayload(ct)->slots = slots;
}

void HazeEngine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>& /*ctx*/, CiphertextImpl<DCRTPoly>& ct, size_t level) {
	if (!ct.device.has_value()) {
		ct.SetLevelHost(level);
		return;
	}
	auto payload = devicePayload(ct);
	if (level > qBase_.size()) {
		throw std::runtime_error("haze backend: SetLevel(" + std::to_string(level) + ") exceeds the modulus chain length");
	}
	const size_t targetTowers = qBase_.size() - level;
	if (targetTowers > payload->towers) {
		throw std::runtime_error("haze backend: SetLevel can only drop levels (have " + std::to_string(qBase_.size() - payload->towers) + ", requested " + std::to_string(level) + ")");
	}
	// Level drop = truncation: hazeFree the tail limbs, no IR, no flush (OpenFHE's
	// DropLastElements does no arithmetic either). Freeing recorded intermediates mid-epoch
	// is safe — the recorded IR holds its own references (ops.cpp precedent).
	for (auto& comp : payload->components) {
		comp.truncate(targetTowers);
	}
	payload->towers = targetTowers;
}

// ---- FIXEDAUTO adjust + scalar-op cores (OpenFHE ckksrns-leveledshe is the oracle) ----

HazeEngine::Operand HazeEngine::asOperand(const std::shared_ptr<HazePayload>& p) const {
	return Operand{ p, p->towers, p->noiseScaleDeg, p->scalingFactor, p->slots };
}

hazebk::ScalarEncodeParams HazeEngine::scalarParams() const {
	return hazebk::ScalarEncodeParams{ qBase_, sfReal_, sfRealBig_, modReduceFactor_, static_cast<lbcrypto::ScalingTechnique>(scalingTech_) };
}

LimbChain HazeEngine::evalModDown(const std::vector<const void*>& survivorsEval,
								  const LimbChain& droppedCoeff,
								  const std::vector<uint64_t>& rescaleBase,
								  const std::vector<uint64_t>& targetBase,
								  ModDownLift lift) {
	// OpenFHE ApproxModDown / the CUDA NTT_RESCALE+NTT_MODDOWN fused path, expressed in
	// haze's pointwise primitives. The caller has INTT'd ONLY the dropped `rescaleBase`
	// limbs (coeff, in droppedCoeff); the `targetBase` survivors stay in eval. Steps:
	//   1. lift  = basis-convert(droppedCoeff -> targetBase)         (coeff)
	//   2. liftN = NTT(lift)                                         (eval)
	//   3. z[q]  = (survivorsEval[q] - liftN[q]) * (prod rescaleBase)^{-1} mod q   (eval)
	// The lift is centered for a rescale and follows the configured FBC variant for a keyswitch
	// mod-down; the subtract + scalar-multiply commute with the NTT, so only the dropped limbs are INTT'd.
	const size_t target = targetBase.size();

	const hazeBasisConvertParams bcParams = {
		/*.src_base =*/rescaleBase.data(),
		/*.src_base_len =*/rescaleBase.size(),
		/*.dst_base =*/targetBase.data(),
		/*.dst_base_len =*/target,
	};
	LimbChain liftCoeff(target, polyBytes_);
	if (lift == ModDownLift::Centered) {
		hazeCheck(hazeBasisConvertCentered(liftCoeff.data(), droppedCoeff.asConst().data(), &bcParams, nullptr), "hazeBasisConvertCentered");
	} else {
		hazeCheck(hazeBasisConvert(liftCoeff.data(), droppedCoeff.asConst().data(), &bcParams, nullptr), "hazeBasisConvert");
	}

	LimbChain liftEval(target, polyBytes_);
	hazeCheck(hazeNTTMrp(liftEval.data(), liftCoeff.asConst().data(), targetBase.data(), target, nullptr), "hazeNTTMrp");

	LimbChain diff(target, polyBytes_);
	hazeCheck(hazeSubMrp(diff.data(), survivorsEval.data(), liftEval.asConst().data(), targetBase.data(), target, nullptr), "hazeSubMrp");

	// P^{-1} mod q_i, P = prod(rescaleBase). The inverse is unique, so OpenFHE's
	// NativeInteger arithmetic yields the same factor libnbfhetch's rescale_fbc applies.
	std::vector<uint64_t> pInv(target);
	for (size_t i = 0; i < target; ++i) {
		const lbcrypto::NativeInteger q(targetBase[i]);
		lbcrypto::NativeInteger prod(1);
		for (const uint64_t p : rescaleBase) {
			prod = prod.ModMul(lbcrypto::NativeInteger(p), q);
		}
		pInv[i] = prod.ModInverse(q).ConvertToInt();
	}
	LimbChain out(target, polyBytes_);
	hazeCheck(hazeMulScalarMrp(out.data(), diff.asConst().data(), pInv.data(), targetBase.data(), target, nullptr), "hazeMulScalarMrp");
	return out;
}

LimbChain HazeEngine::rescaleChainOneTower(const LimbChain& src, size_t srcTowers) {
	// OpenFHE ModReduce / the CUDA NTT_RESCALE path: INTT ONLY the dropped trailing Q prime;
	// the surviving limbs stay in eval. evalModDown lifts the dropped residue, NTTs it into
	// each survivor's eval domain, and does the centered subtract + q_l^{-1} scale there.
	// Bit-identical to the old INTT-all -> hazeModDown -> NTT-all chain, with L-1 fewer INTTs.
	if (srcTowers < 2) {
		throw std::runtime_error("haze backend: rescale needs at least 2 towers");
	}
	const size_t dstTowers = srcTowers - 1;
	const auto srcBase	   = qPrefix(srcTowers);
	const auto dstBase	   = qPrefix(dstTowers);
	const std::vector<uint64_t> rescaleBase = { srcBase.back() };

	// INTT only the dropped (trailing) limb; src is eval and src[dstTowers] == src[srcTowers-1].
	LimbChain droppedCoeff(1, polyBytes_);
	const std::vector<const void*> droppedSrc = { src[dstTowers] };
	hazeCheck(hazeINTTMrp(droppedCoeff.data(), droppedSrc.data(), rescaleBase.data(), rescaleBase.size(), nullptr), "hazeINTTMrp");

	// Surviving limbs (eval), src[0 .. dstTowers-1] in dstBase order.
	std::vector<const void*> survivorsEval(dstTowers);
	for (size_t i = 0; i < dstTowers; ++i) {
		survivorsEval[i] = src[i];
	}

	return evalModDown(survivorsEval, droppedCoeff, rescaleBase, dstBase, ModDownLift::Centered);
}

HazeEngine::Operand HazeEngine::multScalarCore(const Operand& x, double operand) {
	// OpenFHE EvalMultCoreInPlace: per-limb CRT factors at the current level, NSD+1,
	// sf ×= ScalingFactorReal[level]. No pre-rescale (callers add the precheck).
	// Every component is scaled by the same per-limb factors (OpenFHE loops over all elements), so
	// this covers 1-component morphed plaintexts, ordinary ciphertexts and degree-2 results alike.
	const auto factors = hazebk::elemForEvalMult(scalarParams(), x.towers, operand);
	const auto base	   = qPrefix(x.towers);
	std::vector<LimbChain> outs;
	outs.reserve(x.p->components.size());
	for (const auto& comp : x.p->components) {
		LimbChain out(x.towers, polyBytes_);
		hazeCheck(hazeMulScalarMrp(out.data(), comp.asConst().data(), factors.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
		outs.push_back(std::move(out));
	}

	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + 1;
	res.scalingFactor = x.scalingFactor * sfReal_[levelOf(x)];
	res.p			  = finishPayload(std::move(outs), res);
	return res;
}

HazeEngine::Operand HazeEngine::rescaleCore(const Operand& x) {
	// OpenFHE ModReduceInternalInPlace (one level): towers−1, NSD−1,
	// sf ÷= ModReduceFactor[oldTowers−1].
	if (x.noiseScaleDeg < 1) {
		throw std::runtime_error("haze backend: rescale on noiseScaleDeg 0 would underflow");
	}
	// Mirrors CUDA's guard in Ciphertext::rescale() (src/CKKS/Ciphertext.cpp) for cross-backend
	// api-contract parity (BITCOMPAT.md O13): a degree-2 (3-component) ciphertext at the
	// FLEXIBLEAUTOEXT extra level (towers == Q, i.e. levelOf == 0 -- reachable only under
	// FLEXIBLEAUTOEXT, whose context provisions one more modulus than FIXEDAUTO/FLEXIBLEAUTO do
	// for the same depth; levelOf(x) == 0 alone is just "freshly encrypted" under every other
	// technique) is refused rather than rescaled. CUDA's version of this guard exists because its
	// fused two-poly rescaleDouble (c0, c1) drops the extra limb differently than the
	// single-polynomial path available for c2 -- a mechanism this per-component loop does not
	// share -- but the state is kept untested/unsupported on both backends rather than only on CUDA.
	if (static_cast<lbcrypto::ScalingTechnique>(scalingTech_) == lbcrypto::FLEXIBLEAUTOEXT && x.p->components.size() == 3 && levelOf(x) == 0) {
		throw std::runtime_error("haze backend: rescale of a degree-2 ciphertext at the FLEXIBLEAUTOEXT extra level is not supported; relinearize first");
	}
	// OpenFHE ModReduceInternalInPlace rescales EVERY element, so this loops over all components
	// (1-component morphed plaintexts, ciphertexts, degree-2 results).
	std::vector<LimbChain> outs;
	outs.reserve(x.p->components.size());
	for (const auto& comp : x.p->components) {
		outs.push_back(rescaleChainOneTower(comp, x.towers));
	}

	Operand res		  = x;
	res.towers		  = x.towers - 1;
	res.noiseScaleDeg = x.noiseScaleDeg - 1;
	res.scalingFactor = x.scalingFactor / modReduceFactor_[x.towers - 1];
	res.p			  = finishPayload(std::move(outs), res);
	return res;
}

HazeEngine::Operand HazeEngine::negateCore(const Operand& x) {
	// Degree-preserving EvalNegate: per-limb scalar q_i − 1 (≡ −1 mod q_i); metadata unchanged.
	// CUDA's evalNegate is multScalar(-1.0) instead (CudaEngine.cpp:104-110): residues become the
	// ElemForEvalMult(-1.0) scaled const, bumping NSD 1→2 and sf *= ScalingFactorReal[level] with a
	// depth-2 precheck (known GPU fidelity gap, ApiParityTest.cpp:221) — NOT OpenFHE's q_i−1. We keep
	// the q_i−1 multiply: component count is unchanged (per Ryan's rule, only a degree/component-count
	// change forces literal CUDA reproduction), and it is not obvious CUDA's NSD 1→2 bump is safe for
	// how FIDESlib consumes noiseScaleDeg downstream, so we deliberately do not replicate it.
	// OpenFHE EvalNegate flips the sign of EVERY element, so this loops over all components.
	std::vector<LimbChain> outs;
	outs.reserve(x.p->components.size());
	for (const auto& comp : x.p->components) {
		outs.push_back(negateChain(comp, x.towers));
	}

	Operand res = x;
	res.p		= finishPayload(std::move(outs), res);
	return res;
}

LimbChain HazeEngine::negateChain(const LimbChain& src, size_t towers) {
	// Per-limb scalar multiply by q_i − 1 (≡ −1 mod q_i) — the exact OpenFHE sign flip, shared by
	// negateCore and the EvalSub tail rule (the tail components of the subtrahend are negated).
	std::vector<uint64_t> scalars(towers);
	for (size_t i = 0; i < towers; ++i) {
		scalars[i] = qBase_[i] - 1;
	}
	const auto base = qPrefix(towers);
	LimbChain out(towers, polyBytes_);
	hazeCheck(hazeMulScalarMrp(out.data(), src.asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	return out;
}

void HazeEngine::adjustForAddOrSub(Operand& a, Operand& b) {
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st == lbcrypto::FIXEDMANUAL) {
		// OpenFHE AdjustLevelsInPlace: level-align by dropping towers of the fresher operand
		// (pure truncation — a view change here, no IR). FIXEDMANUAL never auto-rescales, so the
		// depth-fix below is not applied (CUDA does not call adjustForAddOrSub for FIXEDMANUAL).
		const size_t towers = std::min(a.towers, b.towers);
		a.towers			= towers;
		b.towers			= towers;
		return;
	}
	// FIXEDAUTO / FLEXIBLEAUTO / FLEXIBLEAUTOEXT: OpenFHE AdjustLevelsAndDepthInPlace (CUDA
	// adjustScaleAndLevel, which follows it since the FIXEDAUTO alignment fix) via the generic
	// ciphertext helper (towers-space; OpenFHE level = |Q| − towers). The "fresher" operand (more
	// towers / lower level) is adjusted toward the other; equal level only needs a depth bump.
	if (levelOf(a) < levelOf(b)) {
		adjustOperandToward(a, b);
	} else if (levelOf(a) > levelOf(b)) {
		adjustOperandToward(b, a);
	} else {
		if (a.noiseScaleDeg < b.noiseScaleDeg) {
			a = multScalarCore(a, 1.0);
		} else if (b.noiseScaleDeg < a.noiseScaleDeg) {
			b = multScalarCore(b, 1.0);
		}
	}
}

void HazeEngine::adjustOperandToward(Operand& x, const Operand& tgt) {
	// OpenFHE AdjustLevelsAndDepthInPlace, the c1lvl < c2lvl branch, operation for operation: the same
	// floating-point evaluation order for the adjustment factor, the mod-reduce factor of the operand's own
	// top tower, and the rescale before the level reduction (a pure truncation here).
	const size_t c1lvl	 = levelOf(x);
	const size_t c2lvl	 = levelOf(tgt);
	const size_t c1depth = x.noiseScaleDeg;
	const size_t c2depth = tgt.noiseScaleDeg;
	// The scaling factor of a depth-2 operand at a level. OpenFHE's table holds the factor itself under FIXEDAUTO
	// (every level has the same one), which makes the factors below come out as 1/factor and the adjusted operand
	// as its value divided by the scaling factor; the square is what the formulas mean (CUDA does the same).
	auto bigFactor = [&](size_t i) {
		return static_cast<lbcrypto::ScalingTechnique>(scalingTech_) == lbcrypto::FIXEDAUTO ? sfReal_[i] * sfReal_[i] : sfRealBig_[i];
	};
	if (c1depth == 2) {
		if (c2depth == 2) {
			const double scf1 = x.scalingFactor;
			const double scf2 = tgt.scalingFactor;
			const double scf  = sfReal_[c1lvl];
			const double q1	  = modReduceFactor_[x.towers - 1];
			x				  = multScalarCore(x, scf2 / scf1 * q1 / scf);
			x				  = rescaleCore(x);
			if (c1lvl + 1 < c2lvl) {
				x.towers -= c2lvl - c1lvl - 1;
			}
			x.scalingFactor = tgt.scalingFactor;
		} else {
			if (c1lvl + 1 == c2lvl) {
				x = rescaleCore(x);
			} else {
				const double scf1 = x.scalingFactor;
				const double scf2 = bigFactor(c2lvl - 1);
				const double scf  = sfReal_[c1lvl];
				const double q1	  = modReduceFactor_[x.towers - 1];
				x				  = multScalarCore(x, scf2 / scf1 * q1 / scf);
				x				  = rescaleCore(x);
				if (c1lvl + 2 < c2lvl) {
					x.towers -= c2lvl - c1lvl - 2;
				}
				x				= rescaleCore(x);
				x.scalingFactor = tgt.scalingFactor;
			}
		}
	} else {
		if (c2depth == 2) {
			const double scf1 = x.scalingFactor;
			const double scf2 = tgt.scalingFactor;
			const double scf  = sfReal_[c1lvl];
			x				  = multScalarCore(x, scf2 / scf1 / scf);
			x.towers -= c2lvl - c1lvl;
			x.scalingFactor = scf2;
		} else {
			const double scf1 = x.scalingFactor;
			const double scf2 = bigFactor(c2lvl - 1);
			const double scf  = sfReal_[c1lvl];
			x				  = multScalarCore(x, scf2 / scf1 / scf);
			if (c1lvl + 1 < c2lvl) {
				x.towers -= c2lvl - c1lvl - 1;
			}
			x				= rescaleCore(x);
			x.scalingFactor = tgt.scalingFactor;
		}
	}
}

void HazeEngine::adjustPtToward(Operand& ptOp, const Operand& ct) {
	// GPU Plaintext::adjustPlaintextToCiphertext port (src/CKKS/Plaintext.cu:195-225) in
	// towers-space (FIDESlib level = towers − 1). The pt is a 1-component operand; the
	// cores skip its absent c1.
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st == lbcrypto::FIXEDAUTO) {
		if (ptOp.noiseScaleDeg < ct.noiseScaleDeg && ptOp.towers >= ct.towers) {
			// Depth-1 pt meeting a depth-2 ct: the PLAINTEXT is scaled up, the ciphertext is left
			// alone. OpenFHE AdjustLevelsAndDepthInPlace (ckksrns-leveledshe.cpp:726-732 for equal
			// levels, :705-713 when the pt sits at the shallower level) raises the lower-depth side
			// with EvalMultCore by a constant — NSD 1 → 2, sf ×= ScalingFactorReal — and then
			// level-reduces it onto the ciphertext. Under FIXEDAUTO the scaling factor is
			// level-constant, so :710's scf1/scf2/scf is exactly 1.0 and both branches collapse to
			// multScalar(1.0) + tower truncation. Multiplying before the truncation is what OpenFHE
			// does and is bit-identical to truncating first (the limbs are independent).
			// The GPU port below cannot express this: its leading level-difference test swallows the
			// case and drops the depth mismatch on the floor, which the applyPt guard then rejects.
			ptOp		= multScalarCore(ptOp, 1.0);
			ptOp.towers = ct.towers;
			return;
		}
		if (ptOp.towers - ptOp.noiseScaleDeg > ct.towers - ct.noiseScaleDeg) {
			if (ct.noiseScaleDeg == 1 && ptOp.noiseScaleDeg == 2) {
				ptOp.towers = ct.towers + 1; // dropToLevel(c.level + 1): view truncation
				ptOp		= rescaleCore(ptOp);
			} else {
				ptOp.towers = ct.towers; // dropToLevel(c.level)
			}
			return;
		}
		if (ct.noiseScaleDeg == 1 && ptOp.noiseScaleDeg == 2) {
			ptOp = rescaleCore(ptOp);
			return;
		}
		if (ptOp.noiseScaleDeg == 1 && ct.noiseScaleDeg == 2) {
			throw std::runtime_error("haze backend: plaintext adjust failed — rescale the ciphertext first (depth-2 ct vs depth-1 pt)");
		}
		return;
	}
	if (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) {
		// Plaintext::adjustScaleAndLevel(c.NSD, c.level, c.sf): same generic adjust as
		// ciphertext operands, including the equal-level depth bump.
		if (ptOp.towers < ct.towers) {
			throw std::runtime_error("haze backend: plaintext is encoded deeper than the ciphertext (cannot raise a plaintext)");
		}
		if (levelOf(ptOp) < levelOf(ct)) {
			adjustOperandToward(ptOp, ct);
		} else if (ptOp.noiseScaleDeg < ct.noiseScaleDeg) {
			ptOp = multScalarCore(ptOp, 1.0);
		} else if (ptOp.noiseScaleDeg > ct.noiseScaleDeg) {
			throw std::runtime_error("haze backend: plaintext adjust failed — pt depth exceeds ct depth at equal level");
		}
		return;
	}
	throw std::runtime_error("haze backend: adjustPtToward is meaningful only for the AUTO scaling modes");
}

HazeEngine::Operand HazeEngine::multScalarWithPrecheck(const Operand& x, double scalar) {
	// FIXEDAUTO multScalar precheck (FIDESlib Ciphertext.cpp:761-775): rescale a depth-2
	// operand before the scalar multiply so the result re-enters depth 2 cleanly.
	Operand cur	  = x;
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && cur.noiseScaleDeg == 2) {
		cur = rescaleCore(cur);
	}
	return multScalarCore(cur, scalar);
}

LimbChain HazeEngine::passThroughChain(const LimbChain& src, size_t towers) {
	// Recorded per-residue pass-through copy (epoch.cpp copy_device_to_device): the result
	// component is unchanged by the op but must live in its own fresh chains.
	const auto base = qPrefix(towers);
	LimbChain out(towers, polyBytes_);
	hazeCheck(hazeMemcpyMrp(out.data(), src.asConst().data(), polyBytes_, HAZE_MEMCPY_DEVICE_TO_DEVICE, base.data(), base.size()), "hazeMemcpyMrp(D2D)");
	return out;
}

HazeEngine::Operand HazeEngine::addScalarCore(const Operand& x, double scalar) {
	// OpenFHE EvalAdd(ct, double): encode |scalar| at the ct's NSD (ElemForEvalAddOrSub),
	// flip per-limb for negative scalars, add onto c0 only. Metadata unchanged.
	LimbChain out0;
	if (scalar != 0.0) {
		auto factors = hazebk::elemForEvalAddOrSub(scalarParams(), x.towers, std::fabs(scalar), x.noiseScaleDeg);
		if (scalar < 0.0) {
			for (size_t i = 0; i < factors.size(); ++i) {
				// CUDA flips unconditionally: elem[i] = prime.p − elem[i] (Ciphertext.cpp:786), so a
				// 0 residue records the literal q_i, not 0. OpenFHE/old Haze guarded 0→0; we match CUDA.
				factors[i] = qBase_[i] - factors[i];
			}
		}
		const auto base = qPrefix(x.towers);
		out0			= LimbChain(x.towers, polyBytes_);
		hazeCheck(hazeAddScalarMrp(out0.data(), x.p->c0().asConst().data(), factors.data(), base.data(), base.size(), nullptr), "hazeAddScalarMrp");
	} else {
		out0 = passThroughChain(x.p->c0(), x.towers);
	}
	Operand res = x;
	res.p		= finishPayload(passThroughTail(std::move(out0), x), res);
	return res;
}

std::vector<LimbChain> HazeEngine::passThroughTail(LimbChain c0, const Operand& x) {
	// Result component vector of an op that touches c0 only (plaintext / scalar addition): the
	// computed c0 followed by a recorded pass-through of every remaining component of the view.
	std::vector<LimbChain> outs;
	outs.reserve(x.p->components.size());
	outs.push_back(std::move(c0));
	for (size_t e = 1; e < x.p->components.size(); ++e) {
		outs.push_back(passThroughChain(x.p->components[e], x.towers));
	}
	return outs;
}

HazeEngine::Operand HazeEngine::applyPt(const Operand& ct, const hazebk::HazePtPayload& pt, bool subtract) {
	Operand x	  = ct;
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool autoMode = (st != lbcrypto::FIXEDMANUAL);
	if (autoMode) {
		// Rescale-if rule (GPU Ciphertext.cpp:250-252, gated for all AUTO modes there): a
		// depth-2 ct meeting a depth-1 pt encoded one level deeper rescales first, which
		// aligns both level and depth.
		if (pt.noiseScaleDeg == 1 && x.noiseScaleDeg == 2 && pt.towers == x.towers - 1) {
			x = rescaleCore(x);
		}
	}

	const bool fixedTrimOk = (st == lbcrypto::FIXEDAUTO || st == lbcrypto::FIXEDMANUAL) && pt.noiseScaleDeg == x.noiseScaleDeg && pt.towers >= x.towers;
	const bool flexMatchOk = (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) && pt.noiseScaleDeg == x.noiseScaleDeg && pt.towers == x.towers;
	if (fixedTrimOk || flexMatchOk) {
		// The pt chain is trimmed to the ct's towers by passing only the leading pointers
		// (exact for FIXED modes: level-constant scaling factor).
		const auto base = qPrefix(x.towers);
		LimbChain out0(x.towers, polyBytes_);
		if (subtract) {
			hazeCheck(hazeSubMrp(out0.data(), x.p->c0().asConst().data(), pt.chain.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
		} else {
			hazeCheck(hazeAddMrp(out0.data(), x.p->c0().asConst().data(), pt.chain.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
		}

		// slots is ciphertext-only in OpenFHE (EvalAddCore clones ciphertext, never reads the
		// plaintext's GetSlots()); res already carries x.slots via the copy, so leave it alone.
		Operand res = x;
		res.p		= finishPayload(passThroughTail(std::move(out0), x), res);
		return res;
	}
	if (!autoMode) {
		notImplemented("EvalAdd/EvalSub(ct, pt) with mismatched levels under FIXEDMANUAL");
	}

	// Morph + adjust (GPU addPt:254-262 -> adjustPlaintextToCiphertext), then add on c0.
	Operand ptOp;
	ptOp.towers		   = pt.towers;
	ptOp.noiseScaleDeg = pt.noiseScaleDeg;
	ptOp.scalingFactor = pt.scalingFactor;
	ptOp.slots		   = pt.slots;
	ptOp.p = morphPlaintext(pt);
	adjustPtToward(ptOp, x);
	if (ptOp.towers < x.towers || ptOp.noiseScaleDeg != x.noiseScaleDeg) {
		throw std::runtime_error("haze backend: internal error — pt/ct disagree after adjustPlaintextToCiphertext");
	}

	const auto base = qPrefix(x.towers);
	LimbChain out0(x.towers, polyBytes_);
	if (subtract) {
		hazeCheck(hazeSubMrp(out0.data(), x.p->c0().asConst().data(), ptOp.p->c0().asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
	} else {
		hazeCheck(hazeAddMrp(out0.data(), x.p->c0().asConst().data(), ptOp.p->c0().asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	}

	// See the fast path above: slots is ciphertext-only, already x.slots via the copy below.
	Operand res = x;
	res.p		= finishPayload(passThroughTail(std::move(out0), x), res);
	return res;
}

std::shared_ptr<HazePayload> HazeEngine::morphPlaintext(const hazebk::HazePtPayload& pt) {
	// A plaintext operand entering the adjust paths becomes a SINGLE-component payload (a recorded
	// pass-through copy of its chain). The component count is the degree, so one component here
	// means "no c1 to transform" — every core loops over components and therefore skips it.
	auto morph		  = std::make_shared<HazePayload>();
	morph->components = hazebk::makeComponents(passThroughChain(pt.chain, pt.towers));
	morph->towers	  = pt.towers;
	morph->state	  = Residency::Recorded;
	return morph;
}

std::shared_ptr<HazePayload> HazeEngine::finishPayload(std::vector<LimbChain> comps, const Operand& meta) const {
	auto payload		   = std::make_shared<HazePayload>();
	payload->components	   = std::move(comps);
	payload->towers		   = meta.towers;
	payload->noiseScaleDeg = meta.noiseScaleDeg;
	payload->scalingFactor = meta.scalingFactor;
	payload->slots		   = meta.slots;
	payload->state		   = Residency::Recorded;
	return payload;
}

void HazeEngine::rebindPayload(HazePayload& dst, std::vector<LimbChain> comps, const Operand& meta) const {
	dst.components	  = std::move(comps); // frees the old chains (the allocator recycles addrs)
	dst.towers		  = meta.towers;
	dst.noiseScaleDeg = meta.noiseScaleDeg;
	dst.scalingFactor = meta.scalingFactor;
	dst.slots		  = meta.slots;
	dst.state		  = Residency::Recorded;
}

std::shared_ptr<HazePtPayload> HazeEngine::ensurePt(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) {
	ctx.LoadPlaintext(pt);
	auto payload = devicePtPayload(pt);
	return payload;
}

// ---- Linear operations ----
// In-place ops compute into fresh chains and rebind them onto the target's existing payload
// object, so facade-level identity (aliases, declared outputs) is preserved.

HazeEngine::Operand HazeEngine::addSubCore(Operand a, Operand b, bool subtract) {
	adjustForAddOrSub(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}
	// OpenFHE EvalAddCoreInPlace / EvalSubCoreInPlace: operate elementwise over the components both
	// operands have, then take the tail of whichever has more — copied for add, and negated for sub
	// when the tail belongs to the subtrahend (cv1[i] = cv2[i].Negate()). Both operands are degree 1
	// in the common case, so this reduces to the previous two elementwise ops.
	const auto base		= qPrefix(a.towers);
	const size_t sizeA	= a.p->components.size();
	const size_t sizeB	= b.p->components.size();
	const size_t common = std::min(sizeA, sizeB);
	std::vector<LimbChain> outs;
	outs.reserve(std::max(sizeA, sizeB));
	for (size_t e = 0; e < common; ++e) {
		LimbChain out(a.towers, polyBytes_);
		// The view's towers, not the payload's: an adjusted operand may be a truncated view.
		if (subtract) {
			hazeCheck(hazeSubMrp(out.data(), a.p->components[e].asConst().data(), b.p->components[e].asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
		} else {
			hazeCheck(hazeAddMrp(out.data(), a.p->components[e].asConst().data(), b.p->components[e].asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
		}
		outs.push_back(std::move(out));
	}
	for (size_t e = common; e < sizeA; ++e) {
		outs.push_back(passThroughChain(a.p->components[e], a.towers)); // minuend tail: unchanged
	}
	for (size_t e = common; e < sizeB; ++e) {
		outs.push_back(subtract ? negateChain(b.p->components[e], a.towers) : passThroughChain(b.p->components[e], a.towers));
	}

	Operand meta	   = a;
	meta.noiseScaleDeg = std::max(a.noiseScaleDeg, b.noiseScaleDeg); // equal after FIXEDAUTO adjust
	// slots stays a.slots (meta's copy above): OpenFHE clones ciphertext1 alone for EvalAdd/EvalSub
	// and never reads ciphertext2's GetSlots().
	meta.p			   = finishPayload(std::move(outs), meta);
	return meta;
}

Ciphertext<DCRTPoly> HazeEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAdd");
	requireComputable(*p2, "EvalAdd");
	Operand res = addSubCore(asOperand(p1), asOperand(p2), /*subtract=*/false);
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

void HazeEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAddInPlace");
	requireComputable(*p2, "EvalAddInPlace");
	Operand res = addSubCore(asOperand(p1), asOperand(p2), /*subtract=*/false);
	rebindPayload(*p1, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalSub");
	requireComputable(*p2, "EvalSub");
	Operand res = addSubCore(asOperand(p1), asOperand(p2), /*subtract=*/true);
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

void HazeEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalSubInPlace");
	requireComputable(*p2, "EvalSubInPlace");
	Operand res = addSubCore(asOperand(p1), asOperand(p2), /*subtract=*/true);
	rebindPayload(*p1, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalNegate");
	Operand res = negateCore(asOperand(p));
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalNegateInPlace");
	Operand res = negateCore(asOperand(p));
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalAdd(scalar)");
	Operand res = addScalarCore(asOperand(p), scalar);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto p = ensureCt(ctx, ct1);
	requireComputable(*p, "EvalAddInPlace(scalar)");
	Operand res = addScalarCore(asOperand(p), scalar);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSub(scalar)");
	Operand res = addScalarCore(asOperand(p), -scalar);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto p = ensureCt(ctx, ct1);
	requireComputable(*p, "EvalSubInPlace(scalar)");
	Operand res = addScalarCore(asOperand(p), -scalar);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) {
	// scalar − ct = (−ct) + scalar via the exact negate, honoring the oracle's s − ct contract
	// (OpenFheEngine.cpp:174). CUDA's evalSub(scalar, ct) instead does multScalar(-1);addScalar(s);
	// multScalar(-1) (CudaEngine.cpp:252-260), which actually computes ct − s (double-negate sign bug)
	// plus an NSD bump and a dropped tower from the multScalar prechecks. The result stays 2-component,
	// so per Ryan's rule we keep this correct s − ct and deliberately do NOT replicate CUDA's sign/level
	// bug. (The negate primitive itself still differs per #1.)
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSub(scalar, ct)");
	Operand res = addScalarCore(negateCore(asOperand(p)), scalar);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) {
	auto p = ensureCt(ctx, ct1);
	requireComputable(*p, "EvalSubInPlace(scalar, ct)");
	Operand res = addScalarCore(negateCore(asOperand(p)), scalar);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	auto p	  = ensureCt(ctx, ct);
	auto ptp  = ensurePt(ctx, pt);
	requireComputable(*p, "EvalAdd(pt)");
	Operand res = applyPt(asOperand(p), *ptp, /*subtract=*/false);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto p	  = ensureCt(ctx, ct1);
	auto ptp  = ensurePt(ctx, pt);
	requireComputable(*p, "EvalAddInPlace(pt)");
	Operand res = applyPt(asOperand(p), *ptp, /*subtract=*/false);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	auto p	  = ensureCt(ctx, ct);
	auto ptp  = ensurePt(ctx, pt);
	requireComputable(*p, "EvalSub(pt)");
	Operand res = applyPt(asOperand(p), *ptp, /*subtract=*/true);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	// pt − ct = (−ct) + pt via the exact negate.
	auto p	  = ensureCt(ctx, ct);
	auto ptp  = ensurePt(ctx, pt);
	requireComputable(*p, "EvalSub(pt, ct)");
	Operand res = applyPt(negateCore(asOperand(p)), *ptp, /*subtract=*/false);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

Ciphertext<DCRTPoly> HazeEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto p = ensureCt(ctx, ct1);
	requireComputable(*p, "EvalMult(scalar)");
	Operand res = multScalarWithPrecheck(asOperand(p), scalar);
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

void HazeEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	auto p = ensureCt(ctx, ct1);
	requireComputable(*p, "EvalMultInPlace(scalar)");
	Operand res = multScalarWithPrecheck(asOperand(p), scalar);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	// Facade-level left fold, structurally from CudaEngine minus its up-front LoadCiphertext
	// loop (each EvalAdd loads its operands lazily via ensureCt). The association must match
	// OpenFHE's EvalAddMany exactly — it is bit-significant.
	if (ciphertexts.empty()) {
		throw std::runtime_error("EvalAddMany: input ciphertext vector is empty");
	}

	size_t first = 0;
	while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
		++first;
	}
	if (first == ciphertexts.size()) {
		throw std::runtime_error("EvalAddMany: input ciphertext vector has no non-null entries");
	}
	size_t second = first + 1;
	while (second < ciphertexts.size() && ciphertexts[second] == nullptr) {
		++second;
	}
	if (second == ciphertexts.size()) {
		// Single non-null entry: independent copy, like the CPU implementation's clone.
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertexts[first]);
	}

	Ciphertext<DCRTPoly> result = ctx.EvalAdd(ciphertexts[first], ciphertexts[second]);
	for (size_t i = second + 1; i < ciphertexts.size(); ++i) {
		if (ciphertexts[i] != nullptr) {
			ctx.EvalAddInPlace(result, ciphertexts[i]);
		}
	}

	return result;
}

void HazeEngine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	// Facade-level serial fold into slot 0, structurally from CudaEngine minus its up-front
	// LoadCiphertext loop (each EvalAddInPlace loads its operands lazily via ensureCt).
	if (ciphertexts.empty()) {
		throw std::runtime_error("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	size_t first = 0;
	while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
		++first;
	}
	if (first == ciphertexts.size()) {
		throw std::runtime_error("EvalAddManyInPlace: input ciphertext vector has no non-null entries");
	}
	for (size_t i = first + 1; i < ciphertexts.size(); ++i) {
		if (ciphertexts[i] != nullptr) {
			ctx.EvalAddInPlace(ciphertexts[first], ciphertexts[i]);
		}
	}
	if (first != 0) {
		ciphertexts[0] = ciphertexts[first];
	}
}

// ---- Hybrid keyswitch + multiplication family ----

namespace {
LimbChain mulChain(size_t polyBytes, const std::vector<uint64_t>& base, const void* const* x, const void* const* y) {
	LimbChain out(base.size(), polyBytes);
	hazeCheck(hazeMulMrp(out.data(), x, y, base.data(), base.size(), nullptr), "hazeMulMrp");
	return out;
}

LimbChain addChain(size_t polyBytes, const std::vector<uint64_t>& base, const void* const* x, const void* const* y) {
	LimbChain out(base.size(), polyBytes);
	hazeCheck(hazeAddMrp(out.data(), x, y, base.data(), base.size(), nullptr), "hazeAddMrp");
	return out;
}
} // namespace

void HazeEngine::ensureKeyUploaded(KsKey& key) {
	if (key.uploaded) {
		return; // uploaded once; reused within the single program (see the one-program contract)
	}
	const size_t digits = key.host.a_limbs.size();
	key.aDigits.clear();
	key.bDigits.clear();
	key.aDigits.reserve(digits);
	key.bDigits.reserve(digits);
	// Key rows are the FULL Q∥P basis, not the per-op Q-prefix∥P hoistedKeyswitchPrefix builds:
	// naming the shorter one here would trip base validation or mislabel key residues.
	std::vector<uint64_t> qpBaseFull = qBase_;
	qpBaseFull.insert(qpBaseFull.end(), pBase_.begin(), pBase_.end());
	for (size_t d = 0; d < digits; ++d) {
		key.aDigits.emplace_back(key.host.a_limbs[d], qpBaseFull); // H2D: full Q∥P rows
		key.bDigits.emplace_back(key.host.b_limbs[d], qpBaseFull);
	}
	key.uploaded = true;
	// hybridKeyswitch only needs numPartQ_ after this point; drop the per-key host limbs, which
	// would otherwise be retained after upload for no further use.
	key.host = hazebk::HybridKeyswitchLimbs{};
}

HazeEngine::HoistedDigits HazeEngine::hoistedKeyswitchPrefix(const LimbChain& src, size_t towers, size_t numPartQ) {
	// #19 step 1-2: the key-INDEPENDENT prefix of hybridKeyswitch (CUDA c1.modupInto,
	// Ciphertext.cpp:965). INTT(src→coeff) → ModUp(per-digit decompose to Q∥P) → per-digit
	// NTT(→EVAL). The digit partition (alpha, digit_base_lens) is fixed by numPartQ alone, so
	// this depends only on (src, towers) and is shared across every rotation key downstream.
	if (numPartQ == 0) {
		throw std::runtime_error("haze backend: keyswitch key has no digits");
	}
	const size_t sizeQ = qBase_.size();
	const size_t alpha = (sizeQ + numPartQ - 1) / numPartQ;

	const auto qSub				 = qPrefix(towers);
	std::vector<uint64_t> qpBase = qSub;
	qpBase.insert(qpBase.end(), pBase_.begin(), pBase_.end());
	const size_t qpTowers = qpBase.size();

	std::vector<uint64_t> digitBasesFlat;
	std::vector<std::size_t> digitBaseLens;
	digitBaseLens.reserve(numPartQ);
	for (size_t part = 0; part < numPartQ; ++part) {
		const size_t start = part * alpha;
		if (start >= towers) {
			break;
		}
		const size_t end = std::min(start + alpha, towers);
		digitBaseLens.push_back(end - start);
		for (size_t i = start; i < end; ++i) {
			digitBasesFlat.push_back(qBase_[i]);
		}
	}
	const size_t numDigits = digitBaseLens.size();
	if (numDigits == 0) {
		throw std::runtime_error("haze backend: keyswitch digit decomposition is empty");
	}

	const hazeModUpParams modupParams = {
		/*.src_base =*/qSub.data(),
		/*.src_base_len =*/qSub.size(),
		/*.digit_bases =*/digitBasesFlat.data(),
		/*.digit_bases_total_len =*/digitBasesFlat.size(),
		/*.digit_base_lens =*/digitBaseLens.data(),
		/*.digit_count =*/numDigits,
		/*.p_base =*/pBase_.data(),
		/*.p_base_len =*/pBase_.size(),
	};

	LimbChain srcCoeff(towers, polyBytes_);
	hazeCheck(hazeINTTMrp(srcCoeff.data(), src.asConst().data(), qSub.data(), qSub.size(), nullptr), "hazeINTTMrp");

	LimbChain digitsFlat(numDigits * qpTowers, polyBytes_);
	hazeCheck(hazeModUp(digitsFlat.data(), srcCoeff.asConst().data(), &modupParams, nullptr), "hazeModUp");

	// Per-digit NTT back to EVAL form over Q∥P.
	LimbChain digitsEval(numDigits * qpTowers, polyBytes_);
	for (size_t d = 0; d < numDigits; ++d) {
		std::vector<const void*> in(qpTowers);
		std::vector<void*> out(qpTowers);
		for (size_t t = 0; t < qpTowers; ++t) {
			in[t]  = digitsFlat[(d * qpTowers) + t];
			out[t] = digitsEval[(d * qpTowers) + t];
		}
		hazeCheck(hazeNTTMrp(out.data(), in.data(), qpBase.data(), qpBase.size(), nullptr), "hazeNTTMrp");
	}

	HoistedDigits h;
	h.digitsEval = std::move(digitsEval);
	h.towers	 = towers;
	h.numDigits	 = numDigits;
	h.qpTowers	 = qpTowers;
	h.qpBase	 = std::move(qpBase);
	return h;
}

HazeEngine::KsContribution HazeEngine::hybridKeyswitchFromDigits(const HoistedDigits& digits, KsKey& key, bool ext) {
	// #19 step 3: the per-KEY work — the part that varies across rotation indexes. Reuses the
	// shared digitsEval handle (CUDA dotKSKInPlaceFrom, Ciphertext.cpp:991) instead of redoing
	// the ModUp/NTT. ext=false → ModDown(drop P)+NTT to base-Q (the plain-rotate contribution,
	// byte-identical to the old monolithic hybridKeyswitch). ext=true (#10) → KEEP the extended
	// Q∥P basis and SKIP the ModDown+NTT (CUDA rotate_hoisted ext=true skips moddown,
	// Ciphertext.cpp:994-998); the caller folds addFirst and does the moddown later.
	ensureKeyUploaded(key);
	const size_t sizeQ	  = qBase_.size();
	const size_t towers	  = digits.towers;
	const size_t qpTowers = digits.qpTowers;
	const size_t numDigits = digits.numDigits;
	const auto& qpBase	  = digits.qpBase;

	// Key rows for this towers count: indices [0, towers) ∪ [|Q|, |Q|+|P|) of the full rows.
	auto keyRows = [&](const LimbChain& full) {
		std::vector<const void*> rows;
		rows.reserve(qpTowers);
		for (size_t t = 0; t < towers; ++t) {
			rows.push_back(full[t]);
		}
		for (size_t t = 0; t < pBase_.size(); ++t) {
			rows.push_back(full[sizeQ + t]);
		}
		return rows;
	};

	// Inner products against the key digits; same-shape in-place accumulation is the one
	// sanctioned reuse.
	LimbChain accumA(qpTowers, polyBytes_);
	LimbChain accumB(qpTowers, polyBytes_);
	for (size_t d = 0; d < numDigits; ++d) {
		std::vector<const void*> dig(qpTowers);
		for (size_t t = 0; t < qpTowers; ++t) {
			dig[t] = digits.digitsEval[(d * qpTowers) + t];
		}
		const auto aRows = keyRows(key.aDigits[d]);
		const auto bRows = keyRows(key.bDigits[d]);
		if (d == 0) {
			hazeCheck(hazeMulMrp(accumB.data(), dig.data(), bRows.data(), qpBase.data(), qpBase.size(), nullptr), "hazeMulMrp");
			hazeCheck(hazeMulMrp(accumA.data(), dig.data(), aRows.data(), qpBase.data(), qpBase.size(), nullptr), "hazeMulMrp");
		} else {
			LimbChain prodB(qpTowers, polyBytes_);
			LimbChain prodA(qpTowers, polyBytes_);
			hazeCheck(hazeMulMrp(prodB.data(), dig.data(), bRows.data(), qpBase.data(), qpBase.size(), nullptr), "hazeMulMrp");
			hazeCheck(hazeMulMrp(prodA.data(), dig.data(), aRows.data(), qpBase.data(), qpBase.size(), nullptr), "hazeMulMrp");
			hazeCheck(hazeAddMrp(accumB.data(), accumB.asConst().data(), prodB.asConst().data(), qpBase.data(), qpBase.size(), nullptr), "hazeAddMrp");
			hazeCheck(hazeAddMrp(accumA.data(), accumA.asConst().data(), prodA.asConst().data(), qpBase.data(), qpBase.size(), nullptr), "hazeAddMrp");
		}
	}

	if (ext) {
		// #10 extended-basis return: the dot product is already EVAL form over Q∥P — hand it back
		// without the ModDown (CUDA keeps Q∥P; the final moddown is deferred to the giant-step
		// boundary). NOTE: this is the contribution before the c0+ks.b fold and automorph.
		return KsContribution{ std::move(accumB), std::move(accumA) };
	}

	// Final mod-down: drop the P special primes (CUDA NTT_MODDOWN path). INTT ONLY the |P|
	// special limbs (the trailing limbs of qpBase = qSub ++ pBase_); the `towers` Q limbs
	// stay in eval. evalModDown lifts P -> Q, NTTs the lift, and combines in eval.
	// Bit-identical to the old INTT-all(Q|P) -> hazeModDown -> NTT-all(Q) chain.
	const size_t numP = pBase_.size();
	const auto qSub	  = qPrefix(towers);
	LimbChain pCoeffA(numP, polyBytes_);
	LimbChain pCoeffB(numP, polyBytes_);
	std::vector<const void*> pSrcA(numP);
	std::vector<const void*> pSrcB(numP);
	for (size_t j = 0; j < numP; ++j) {
		pSrcA[j] = accumA[towers + j];
		pSrcB[j] = accumB[towers + j];
	}
	hazeCheck(hazeINTTMrp(pCoeffA.data(), pSrcA.data(), pBase_.data(), numP, nullptr), "hazeINTTMrp");
	hazeCheck(hazeINTTMrp(pCoeffB.data(), pSrcB.data(), pBase_.data(), numP, nullptr), "hazeINTTMrp");

	// Surviving Q limbs (eval): accum[0 .. towers-1] in qSub order.
	std::vector<const void*> survA(towers);
	std::vector<const void*> survB(towers);
	for (size_t i = 0; i < towers; ++i) {
		survA[i] = accumA[i];
		survB[i] = accumB[i];
	}
	LimbChain outA = evalModDown(survA, pCoeffA, pBase_, qSub, ModDownLift::Configured);
	LimbChain outB = evalModDown(survB, pCoeffB, pBase_, qSub, ModDownLift::Configured);

	return KsContribution{ std::move(outB), std::move(outA) };
}

HazeEngine::KsContribution HazeEngine::hybridKeyswitch(const LimbChain& src, size_t towers, KsKey& key) {
	// Port of ops.cpp hybrid_keyswitch, now a thin composition of the shared input-only prefix
	// (#19 step 1-2) + the per-key step (step 3). Relin (ct×ct) and non-hoisted rotation callers
	// route through here unchanged, so their results stay byte-identical to the old monolithic
	// implementation. The full-Q∥P key is uploaded once; trimming to (first `towers` Q rows, all
	// P rows) is pointer selection, not re-upload. numPartQ_ is the same digit count that sized
	// key.host.a_limbs, read here instead so this call works after ensureKeyUploaded drops it.
	HoistedDigits digits = hoistedKeyswitchPrefix(src, towers, numPartQ_);
	return hybridKeyswitchFromDigits(digits, key, /*ext=*/false);
}

void HazeEngine::adjustForMult(Operand& a, Operand& b) {
	// OpenFHE AdjustLevelsAndDepthToOneInPlace: equalize, then bring both to depth 1.
	const bool aliased = (a.p == b.p && a.towers == b.towers && a.noiseScaleDeg == b.noiseScaleDeg);
	adjustForAddOrSub(a, b);
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && a.noiseScaleDeg == 2) {
		a = rescaleCore(a);
		b = aliased ? a : rescaleCore(b);
	}
}

HazeEngine::Operand HazeEngine::multPtCore(const Operand& ct, const hazebk::HazePtPayload& pt) {
	// multPt adjust rules (GPU Ciphertext.cpp:356-421): rescale a depth-2 ct first; a pt at
	// a mismatched level/depth is adjusted toward the ct (adjustPlaintextToCiphertext) as a
	// morphed 1-component operand. For FIXED modes the scaling factor is level-constant, so
	// trimming a same-depth higher-level pt chain to the ct's towers is exact (fast path,
	// no morph). FLEXIBLE modes need exact level/scale agreement, hence the adjust.
	Operand x	  = ct;
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool autoMode = (st != lbcrypto::FIXEDMANUAL);
	if (autoMode && x.noiseScaleDeg == 2) {
		x = rescaleCore(x);
	}

	const bool fixedTrimOk = (st == lbcrypto::FIXEDAUTO || st == lbcrypto::FIXEDMANUAL) && pt.noiseScaleDeg == 1 && pt.towers >= x.towers;
	const bool flexMatchOk = (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) && pt.noiseScaleDeg == 1 && x.noiseScaleDeg == 1 && pt.towers == x.towers;
	if (fixedTrimOk || flexMatchOk) {
		// Full polynomial multiply of EVERY component against the pt chain ( never the
		// slot-constant mult_scalar shortcut; OpenFHE's pt multiply loops over all elements).
		const auto base = qPrefix(x.towers);
		std::vector<LimbChain> outs;
		outs.reserve(x.p->components.size());
		for (const auto& comp : x.p->components) {
			outs.push_back(mulChain(polyBytes_, base, comp.asConst().data(), pt.chain.asConst().data()));
		}

		Operand res		  = x;
		res.noiseScaleDeg = x.noiseScaleDeg + pt.noiseScaleDeg;
		res.scalingFactor = x.scalingFactor * pt.scalingFactor;
		// slots stays x.slots (already carried by the copy above): OpenFHE clones the ciphertext
		// for EvalMult(ct, pt) and never reads the plaintext's own GetSlots().
		res.p			  = finishPayload(std::move(outs), res);
		return res;
	}
	if (!autoMode) {
		notImplemented("EvalMult(ct, pt) with mismatched levels under FIXEDMANUAL");
	}

	// Morph the pt into a 1-component operand (a recorded pass-through copy of its chain)
	// and adjust it toward the ct, mirroring GPU multPt's adjustPlaintextToCiphertext path.
	Operand ptOp;
	ptOp.towers		   = pt.towers;
	ptOp.noiseScaleDeg = pt.noiseScaleDeg;
	ptOp.scalingFactor = pt.scalingFactor;
	ptOp.slots		   = pt.slots;
	ptOp.p			   = morphPlaintext(pt);
	adjustPtToward(ptOp, x);
	if (x.noiseScaleDeg == 2) { // GPU multPt:396-402 post-adjust rescues
		x = rescaleCore(x);
	}
	if (ptOp.noiseScaleDeg == 2) {
		ptOp = rescaleCore(ptOp);
	}
	if (ptOp.towers < x.towers || ptOp.noiseScaleDeg != x.noiseScaleDeg) {
		throw std::runtime_error("haze backend: internal error — pt/ct disagree after adjustPlaintextToCiphertext");
	}

	const auto base = qPrefix(x.towers);
	std::vector<LimbChain> outs;
	outs.reserve(x.p->components.size());
	for (const auto& comp : x.p->components) {
		outs.push_back(mulChain(polyBytes_, base, comp.asConst().data(), ptOp.p->c0().asConst().data()));
	}

	// See the fast path above: slots stays x.slots, already carried by the copy below.
	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + ptOp.noiseScaleDeg;
	res.scalingFactor = x.scalingFactor * ptOp.scalingFactor;
	res.p			  = finishPayload(std::move(outs), res);
	return res;
}

HazeEngine::Operand HazeEngine::multNoRelinCore(Operand a, Operand b, const char* op) {
	// OpenFHE EvalMultNoRelin == EvalMult minus the keyswitch+fold tail: the keyed EvalMult
	// (base-leveledshe.cpp:202-215) is literally this tensor product followed by
	// KeySwitchCore(cv[2]) + fold + resize(2), which is exactly relinearizeCore. NO relin key is
	// required here — Relinearize owns that requirement.
	requireRelinearized(*a.p, op);
	requireRelinearized(*b.p, op);
	adjustForMult(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}

	// Tensor product (ops.cpp order): d0 = a0·b0, d1 = a0·b1 + a1·b0, d2 = a1·b1.
	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0().asConst().data(), b.p->c0().asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0().asConst().data(), b.p->c1().asConst().data());
	LimbChain u		= mulChain(polyBytes_, base, a.p->c1().asConst().data(), b.p->c0().asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), u.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1().asConst().data(), b.p->c1().asConst().data());

	Operand res		  = a;
	res.noiseScaleDeg = a.noiseScaleDeg + b.noiseScaleDeg;
	res.scalingFactor = a.scalingFactor * b.scalingFactor; // exact double op order
	// slots stays a.slots (already carried by the copy above): OpenFHE clones ctxt1 alone
	// (CloneEmpty) for EvalMult(ct1, ct2) and never reads ctxt2->GetSlots().
	res.p			  = finishPayload(hazebk::makeComponents(std::move(d0), std::move(d1), std::move(d2)), res);
	return res;
}

HazeEngine::Operand HazeEngine::relinearizeCore(const Operand& x) {
	// OpenFHE's Relinearize/RelinearizeInPlace call GetEvalMultKeyVector(tag) FIRST, before any
	// degree check (cryptocontext.h), and that accessor throws unconditionally when no key is
	// registered. So the key check must precede the degree branch here too, or a degree-1
	// relinearize without EvalMultKeyGen would wrongly succeed as a no-op.
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: Relinearize requires EvalMultKeyGen before LoadContext");
	}
	const size_t comps = x.p->components.size();
	if (comps == 2) {
		return x; // OpenFHE RelinearizeInPlace on a degree-1 ciphertext is a no-op (cv.resize(2))
	}
	if (comps != 3) {
		throw std::runtime_error("haze backend: Relinearize supports degree-2 ciphertexts (3 components) only; got " + std::to_string(comps));
	}
	// KeySwitchCore(cv[2]) + fold, exactly OpenFHE's RelinearizeInPlace loop body. All metadata
	// (level, NSD, scaling factor, slots) is untouched.
	const auto base	  = qPrefix(x.towers);
	KsContribution ks = hybridKeyswitch(x.p->components[2], x.towers, relinKey_);
	LimbChain out0	  = addChain(polyBytes_, base, x.p->components[0].asConst().data(), ks.b.asConst().data());
	LimbChain out1	  = addChain(polyBytes_, base, x.p->components[1].asConst().data(), ks.a.asConst().data());

	Operand res = x;
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
	return res;
}

Ciphertext<DCRTPoly> HazeEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalMult");
	requireComputable(*p2, "EvalMult");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalMult requires EvalMultKeyGen before LoadContext");
	}
	// Same recorded IR as the pre-split monolithic implementation: tensor product, then keyswitch
	// of d2 and the two folds.
	Operand res = relinearizeCore(multNoRelinCore(asOperand(p1), asOperand(p2), "EvalMult"));
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

void HazeEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalMultInPlace");
	requireComputable(*p2, "EvalMultInPlace");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalMult requires EvalMultKeyGen before LoadContext");
	}
	Operand res = relinearizeCore(multNoRelinCore(asOperand(p1), asOperand(p2), "EvalMultInPlace"));
	rebindPayload(*p1, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalMultNoRelin(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalMultNoRelin");
	requireComputable(*p2, "EvalMultNoRelin");
	// Deliberately no relin-key guard: OpenFHE's EvalMultNoRelin is a pure tensor product.
	Operand res = multNoRelinCore(asOperand(p1), asOperand(p2), "EvalMultNoRelin");
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

Ciphertext<DCRTPoly> HazeEngine::relinearize(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "Relinearize");
	Operand res = relinearizeCore(asOperand(p));
	if (res.p == p) {
		// Degree-1 input: OpenFHE's Relinearize is Clone() + a no-op RelinearizeInPlace, so it still
		// hands back an INDEPENDENT ciphertext. Record pass-through copies rather than aliasing the
		// input's payload, which a later in-place op would corrupt.
		std::vector<LimbChain> outs;
		outs.reserve(p->components.size());
		for (const auto& comp : p->components) {
			outs.push_back(passThroughChain(comp, res.towers));
		}
		res.p = finishPayload(std::move(outs), res);
	}
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::relinearizeInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "RelinearizeInPlace");
	Operand res = relinearizeCore(asOperand(p));
	if (res.p == p) {
		return; // degree-1: OpenFHE's no-op, and rebinding a payload onto itself would free it
	}
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSquare");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalSquare requires EvalMultKeyGen before LoadContext");
	}

	requireRelinearized(*p, "EvalSquare");

	// OpenFHE EvalSquare: AUTO modes mod-reduce a depth-2 input first, then square.
	Operand a	  = asOperand(p);
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && a.noiseScaleDeg == 2) {
		a = rescaleCore(a);
	}

	// Squaring keeps its own tensor product (a·a shares the cross term), but the keyswitch+fold
	// tail is relinearizeCore — the fold is never duplicated.
	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0().asConst().data(), a.p->c0().asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0().asConst().data(), a.p->c1().asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), t.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1().asConst().data(), a.p->c1().asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = 2 * a.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * a.scalingFactor;
	meta.p			   = finishPayload(hazebk::makeComponents(std::move(d0), std::move(d1), std::move(d2)), meta);
	Operand res		   = relinearizeCore(meta);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSquareInPlace");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalSquare requires EvalMultKeyGen before LoadContext");
	}
	requireRelinearized(*p, "EvalSquareInPlace");

	Operand a	  = asOperand(p);
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && a.noiseScaleDeg == 2) {
		a = rescaleCore(a);
	}

	// Squaring keeps its own tensor product (a·a shares the cross term), but the keyswitch+fold
	// tail is relinearizeCore — the fold is never duplicated.
	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0().asConst().data(), a.p->c0().asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0().asConst().data(), a.p->c1().asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), t.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1().asConst().data(), a.p->c1().asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = 2 * a.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * a.scalingFactor;
	meta.p			   = finishPayload(hazebk::makeComponents(std::move(d0), std::move(d1), std::move(d2)), meta);
	Operand res		   = relinearizeCore(meta);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto p	 = ensureCt(ctx, ct1);
	auto ptp = ensurePt(ctx, pt);
	requireComputable(*p, "EvalMult(pt)");
	Operand res = multPtCore(asOperand(p), *ptp);
	return wrapDeviceResult(ctx, ct1, std::move(res.p));
}

void HazeEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	auto p	 = ensureCt(ctx, ct1);
	auto ptp = ensurePt(ctx, pt);
	requireComputable(*p, "EvalMultInPlace(pt)");
	Operand res = multPtCore(asOperand(p), *ptp);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::multByMonomial(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t power) {
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "MultByMonomial");
	Operand res = multByMonomialCore(ctx, asOperand(p), power);
	return wrapDeviceResult(ctx, ciphertext, std::move(res.p));
}

void HazeEngine::multByMonomialInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t power) {
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "MultByMonomialInPlace");
	Operand res = multByMonomialCore(ctx, asOperand(p), power);
	rebindPayload(*p, std::move(res.p->components), res);
}

Ciphertext<DCRTPoly> HazeEngine::rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	// CUDA rescales eagerly under ALL scaling techniques (drop a limb, NoiseFactor /= ModReduceFactor,
	// NoiseLevel −= 1; Ciphertext.cpp:428-442 via CudaEngine.cpp:459,466), unlike OpenFHE's public
	// ModReduce which mod-reduces only under FIXEDMANUAL and clones under the AUTO modes. We match CUDA
	// and always mod-reduce. (Task 02 #16's convolution special-exit rescale depends on this being
	// unconditional.)
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "Rescale");
	Operand res = rescaleCore(asOperand(p));
	return wrapDeviceResult(ctx, ciphertext, std::move(res.p));
}

void HazeEngine::rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) {
	// CUDA mod-reduces eagerly under every technique (see rescale() above), not OpenFHE's
	// FIXEDMANUAL-only ModReduceInPlace.
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "RescaleInPlace");
	Operand res = rescaleCore(asOperand(p));
	rebindPayload(*p, std::move(res.p->components), res);
}

// ---- Rotation + accumulate ----

HazeEngine::KsKey& HazeEngine::autoKeyFor(CryptoContextImpl<DCRTPoly>& ctx, uint32_t autoIndex) {
	auto it = autoKeys_.find(autoIndex);
	if (it == autoKeys_.end()) {
		auto& context = hostContext(ctx);
		KsKey key;
		key.host = hazebk::extractAutomorphismKeyLimbs(context, keyTag_, autoIndex);
		it		 = autoKeys_.emplace(autoIndex, std::move(key)).first;
	}
	return it->second;
}

HazeEngine::SparseSwitchKey& HazeEngine::sparseKeyFor(CryptoContextImpl<DCRTPoly>& ctx) {
	if (!sparseKey_) {
		auto& context = hostContext(ctx);
		SparseSwitchKey key;
		key.host = hazebk::extractSparseSwitchKeyLimbs(context, keyTag_, 2 * static_cast<uint32_t>(ringDim_) - 4);
		if (key.host.q0 != qBase_.front() || pBase_.empty() || key.host.p0 != pBase_.front()) {
			throw std::runtime_error("haze backend: the sparse switching key is not over the first primes of Q and P");
		}
		const std::vector<uint64_t> base = { key.host.q0, key.host.p0 };
		key.a							 = LimbChain(key.host.a_limbs, base);
		key.b							 = LimbChain(key.host.b_limbs, base);
		key.uploaded					 = true;
		key.host.a_limbs.clear();
		key.host.b_limbs.clear();
		sparseKey_ = std::move(key);
	}
	return *sparseKey_;
}

HazeEngine::Operand HazeEngine::keySwitchSparseCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x) {
	requireRelinearized(*x.p, "EvalBootstrap");
	SparseSwitchKey& key = sparseKeyFor(ctx);
	const uint64_t q0	 = key.host.q0;
	const uint64_t p0	 = key.host.p0;
	const std::vector<uint64_t> baseQ	= { q0 };
	const std::vector<uint64_t> baseP	= { p0 };
	const std::vector<uint64_t> baseQP = { q0, p0 };

	// c1 mod q0, extended to q0*p0: the p0 row is the centered lift of the coefficients.
	const void* const c1Row0[1] = { x.p->c1()[0] };
	LimbChain c1Coeff(1, polyBytes_);
	hazeCheck(hazeINTTMrp(c1Coeff.data(), c1Row0, baseQ.data(), 1, nullptr), "hazeINTTMrp(sparse)");
	const hazeBasisConvertParams toP = {
		/*.src_base =*/baseQ.data(),
		/*.src_base_len =*/1,
		/*.dst_base =*/baseP.data(),
		/*.dst_base_len =*/1,
	};
	LimbChain c1P(1, polyBytes_);
	hazeCheck(hazeBasisConvertCentered(c1P.data(), c1Coeff.asConst().data(), &toP, nullptr), "hazeBasisConvertCentered(sparse)");
	LimbChain c1PEval(1, polyBytes_);
	hazeCheck(hazeNTTMrp(c1PEval.data(), c1P.asConst().data(), baseP.data(), 1, nullptr), "hazeNTTMrp(sparse)");
	const void* const c1Ext[2] = { x.p->c1()[0], c1PEval[0] };

	// Multiply by the key, then divide by p0 back to q0 (ApproxModDown with a centered lift).
	auto switchOne = [&](const LimbChain& keyPoly) {
		LimbChain prod(2, polyBytes_);
		hazeCheck(hazeMulMrp(prod.data(), c1Ext, keyPoly.asConst().data(), baseQP.data(), 2, nullptr), "hazeMulMrp(sparse)");
		const void* const pRow[1] = { prod[1] };
		LimbChain pCoeff(1, polyBytes_);
		hazeCheck(hazeINTTMrp(pCoeff.data(), pRow, baseP.data(), 1, nullptr), "hazeINTTMrp(sparse)");
		const std::vector<const void*> surv = { prod[0] };
		return evalModDown(surv, pCoeff, baseP, baseQ, ModDownLift::Centered);
	};
	LimbChain ksB = switchOne(key.b);
	LimbChain ksA = switchOne(key.a);

	// c0' = c0 + ks.b (first prime only), c1' = ks.a.
	const void* const c0Row0[1] = { x.p->c0()[0] };
	LimbChain out0				= addChain(polyBytes_, baseQ, c0Row0, ksB.asConst().data());

	Operand res = x; // metadata unchanged
	res.towers	= 1;
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(ksA)), res);
	return res;
}

HazeEngine::Operand HazeEngine::keySwitchByAutoIndex(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, uint32_t autoIndex) {
	requireRelinearized(*x.p, "EvalBootstrap");
	KsKey& key		  = autoKeyFor(ctx, autoIndex);
	KsContribution ks = hybridKeyswitch(x.p->c1(), x.towers, key);
	const auto base	  = qPrefix(x.towers);
	LimbChain out0	  = addChain(polyBytes_, base, x.p->c0().asConst().data(), ks.b.asConst().data());

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(ks.a)), res);
	return res;
}

HazeEngine::Operand HazeEngine::rotateByAutoIndex(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, uint32_t autoIndex) {
	// Keyswitching an automorphism only has a c1 to switch: a degree-2 operand has no valid path
	// here (bootstrap reaches this core without passing the facade's guard).
	requireRelinearized(*x.p, "EvalRotate");
	KsKey& key = autoKeyFor(ctx, autoIndex);

	// Keyswitch first, automorphism last (OpenFHE EvalAtIndex order; ops.cpp rotate).
	KsContribution ks = hybridKeyswitch(x.p->c1(), x.towers, key);
	const auto base	  = qPrefix(x.towers);
	LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0().asConst().data(), ks.b.asConst().data());

	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
	hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
	return res;
}

HazeEngine::RotKeyResolved HazeEngine::resolveRotationKey(int32_t step, size_t slotsArg) {
	// Slots-aware rotation-key selection (CUDA Ciphertext::rotate, Ciphertext.cpp:822-887):
	// normalyzeIndex maps the logical step to a full-ring rotation given the ct's slot count, then
	// GetRotationKey selects the actual_index — the registered key whose normalized index is
	// slot-compatible (the alternate-key search adds multiples of `slots`, which are no-ops on
	// slots-periodic sparse data). OpenFHE's EvalAtIndex (the old path) used 5^step with slots
	// ignored, so steps > slots/2 picked the wrong key. Extracted so rotateCore (plain) and
	// fastRotateFromDigits (hoisted) pick the IDENTICAL key+autoIndex (#19 byte-identity).
	const int32_t half	= (ringDim_ >= 2) ? static_cast<int32_t>(ringDim_ / 2) : 1;
	const int32_t slots = (slotsArg > 0) ? static_cast<int32_t>(slotsArg) : half;
	// normalyzeIndex(step, slots, N) — Context.cu:1088-1095.
	int32_t norm = step % slots;
	if (norm < 0)
		norm += slots;
	if (norm > slots / 2)
		norm += half - slots; // N/2 − slots
	// GetRotationKey actual_index (Context.cu:562-589): reduce mod N/2, then the slot-compatible search.
	const int32_t idx		 = ((norm % half) + half) % half;
	const int32_t* foundStep = nullptr;
	if (auto hit = rotKeyIndex_.find(idx); hit != rotKeyIndex_.end()) {
		foundStep = &hit->second;
	} else if (slots != half) {
		for (int32_t i = 1; i < half / slots; ++i) {
			const int32_t idx_ = (idx + i * slots) % half;
			if (auto alt = rotKeyIndex_.find(idx_); alt != rotKeyIndex_.end()) {
				foundStep = &alt->second;
				break;
			}
		}
	}
	RotKeyResolved r;
	if (foundStep != nullptr) {
		// Pre-extracted EvalRotateKeyGen key; rk.autoIndex is the OpenFHE automorphism for the
		// found step, which is the key matching CUDA's 5^(2N − actual_index) for that index.
		RotKey& rk	 = rotKeys_.at(*foundStep);
		r.key		 = &rk.key;
		r.autoIndex	 = rk.autoIndex;
	}
	return r;
}

HazeEngine::Operand HazeEngine::rotateCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int32_t step) {
	requireRelinearized(*x.p, "EvalRotate");
	const RotKeyResolved rk = resolveRotationKey(step, x.slots);
	if (rk.key != nullptr) {
		KsContribution ks = hybridKeyswitch(x.p->c1(), x.towers, *rk.key);
		const auto base	  = qPrefix(x.towers);
		LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0().asConst().data(), ks.b.asConst().data());

		LimbChain out0(x.towers, polyBytes_);
		LimbChain out1(x.towers, polyBytes_);
		hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
		hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

		Operand res = x; // metadata unchanged
		res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
		return res;
	}
	// Fallback (bootstrap rotations and any step without a registered key): resolve the automorphism
	// index on the host and extract the key lazily. This path is not slots-aware (kept for full-slot
	// bootstrap); sparse-bootstrap rotation correctness is Task 05's scope.
	auto& context			 = hostContext(ctx);
	const uint32_t autoIndex = context->FindAutomorphismIndex(static_cast<uint32_t>(step));
	return rotateByAutoIndex(ctx, x, autoIndex);
}

HazeEngine::Operand HazeEngine::fastRotateFromDigits(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int32_t step, const HoistedDigits& digits) {
	// #19 hoisted single rotation: identical assembly to rotateCore's pre-extracted-key branch,
	// but the keyswitch reuses the shared digit decomposition (hybridKeyswitchFromDigits) instead
	// of recomputing INTT+ModUp+NTT. Same key (resolveRotationKey), same c0+ks.b fold, same
	// AutomorphMrp on BOTH components ⇒ byte-identical to evalRotate(x, step). No host fallback:
	// fast rotation only services pre-extracted keys (the bootstrap full-slot fallback uses plain
	// EvalRotate, not the hoisted path).
	requireRelinearized(*x.p, "EvalFastRotation");
	const RotKeyResolved rk = resolveRotationKey(step, x.slots);
	if (rk.key == nullptr) {
		// No pre-extracted key (e.g. a bootstrap full-slot rotation that resolves lazily via
		// FindAutomorphismIndex). The hoisted prefix only covers pre-extracted rotation keys, so
		// fall back to the plain rotate (which recomputes the prefix) — byte-identical, just not
		// hoisted for this index.
		return rotateCore(ctx, x, step);
	}
	KsContribution ks = hybridKeyswitchFromDigits(digits, *rk.key, /*ext=*/false);
	const auto base	  = qPrefix(x.towers);
	LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0().asConst().data(), ks.b.asConst().data());

	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
	hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(out1)), res);
	return res;
}

Ciphertext<DCRTPoly> HazeEngine::evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "EvalRotate");
	Operand res = rotateCore(ctx, asOperand(p), index);
	return wrapDeviceResult(ctx, ciphertext, std::move(res.p));
}

void HazeEngine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	auto p = ensureCt(ctx, ciphertext);
	requireComputable(*p, "EvalRotateInPlace");
	Operand res = rotateCore(ctx, asOperand(p), index);
	rebindPayload(*p, std::move(res.p->components), res);
}

std::shared_ptr<void> HazeEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	// #19: record the shared, key-independent ModUp/per-digit-NTT decomposition of c1 ONCE
	// (CUDA c1.modupInto across the baby steps, Ciphertext.cpp:965). The returned handle is reused
	// by every evalFastRotation index, so the expensive INTT+ModUp+NTT runs a single time instead
	// of once per rotation. numPartQ_ is the context's hybrid digit count (all keys share it).
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotationPrecompute");
	requireRelinearized(*p, "EvalFastRotationPrecompute");
	if (numPartQ_ == 0) {
		throw std::runtime_error("haze backend: EvalFastRotationPrecompute requires a loaded context with keyswitch params");
	}
	return std::make_shared<HoistedDigits>(hoistedKeyswitchPrefix(p->c1(), p->towers, numPartQ_));
}

Ciphertext<DCRTPoly>
HazeEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t /*m*/, const std::shared_ptr<void>& precomp) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotation");
	// A null/foreign precompute (or a stale handle at a different level) ⇒ fall back to the plain
	// rotate, which recomputes the prefix. Byte-identical either way (#19 is perf-only).
	auto digits = std::static_pointer_cast<HoistedDigits>(precomp);
	if (!digits || digits->towers != p->towers) {
		return evalRotate(ctx, ct, index);
	}
	Operand res = fastRotateFromDigits(ctx, asOperand(p), index, *digits);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

Ciphertext<DCRTPoly>
HazeEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	// #10: CUDA's evalFastRotationExt keeps the result in the EXTENDED (Q∥P) basis, SKIPS the
	// final ModDown, and folds addFirst*PModq into c0 (CudaEngine.cpp:375-387 → rotate(.,false);
	// Ciphertext.cpp:994-998,1030-1036). In the haze backend, however, NOTHING downstream consumes
	// an extended-basis ciphertext: the bootstrap linearTransform/CtS-StC drivers use plain
	// base-Q EvalRotate (not Ext), and the only Ext caller is the dispatch-only api test
	// (ApiTests.cpp:1225, EXPECT_NO_THROW, no decrypt). There is no extended-basis HazePayload
	// representation (towers + |P| limbs with no ModDown) that any later op could read, and the
	// fused giant-step ModDown that CUDA pairs with it (LinearTransform.cu:174-194,304) is not
	// wired here. Wiring a true extended return end-to-end would therefore have no correct
	// consumer and would risk a decrypt that the api test does not guard. Per the task's explicit
	// fallback guidance, we keep Ext CORRECT-but-base-Q: route through the hoisting primitive WITH
	// the ModDown (the same byte-identical result as evalRotate) when a handle is present, else
	// plain rotate. addFirst is therefore not folded (it only matters for the skipped-ModDown
	// extended form). See the report for why the extended return was not safely wireable.
	(void)addFirst;
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotationExt");
	auto h = std::static_pointer_cast<HoistedDigits>(digits);
	if (!h || h->towers != p->towers) {
		return evalRotate(ctx, ct, index);
	}
	Operand res = fastRotateFromDigits(ctx, asOperand(p), index, *h);
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

std::vector<Ciphertext<DCRTPoly>> HazeEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const uint32_t /*m*/,
  const std::shared_ptr<void>& precomp) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotation");
	auto digits = std::static_pointer_cast<HoistedDigits>(precomp);
	const bool hoisted = digits && digits->towers == p->towers;
	std::vector<Ciphertext<DCRTPoly>> results;
	results.reserve(indices.size());
	for (const int32_t index : indices) {
		if (index == 0) {
			results.push_back(std::make_shared<CiphertextImpl<DCRTPoly>>(*ct));
		} else if (hoisted) {
			Operand res = fastRotateFromDigits(ctx, asOperand(p), index, *digits);
			results.push_back(wrapDeviceResult(ctx, ct, std::move(res.p)));
		} else {
			results.push_back(evalRotate(ctx, ct, index));
		}
	}
	return results;
}

std::vector<Ciphertext<DCRTPoly>> HazeEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const std::shared_ptr<void>& digits,
  bool addFirst) {
	// See the single-index Ext above: kept correct-but-base-Q (no consumer for an extended return).
	(void)addFirst;
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotationExt");
	auto h = std::static_pointer_cast<HoistedDigits>(digits);
	const bool hoisted = h && h->towers == p->towers;
	std::vector<Ciphertext<DCRTPoly>> results;
	results.reserve(indices.size());
	for (const int32_t index : indices) {
		if (index == 0) {
			results.push_back(std::make_shared<CiphertextImpl<DCRTPoly>>(*ct));
		} else if (hoisted) {
			Operand res = fastRotateFromDigits(ctx, asOperand(p), index, *h);
			results.push_back(wrapDeviceResult(ctx, ct, std::move(res.p)));
		} else {
			results.push_back(evalRotate(ctx, ct, index));
		}
	}
	return results;
}

HazeEngine::Operand HazeEngine::partialSumCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int stride, int size, int startFactor) {
	requireRelinearized(*x.p, "AccumulateSum");
	constexpr int radix = ACCUMULATE_SUM_RADIX;
	if (startFactor <= 0 || size <= 0 || startFactor >= size) {
		return x;
	}
	const size_t towers			 = x.towers;
	const auto qSub				 = qPrefix(towers);
	std::vector<uint64_t> qpBase = qSub;
	qpBase.insert(qpBase.end(), pBase_.begin(), pBase_.end());
	auto& context = hostContext(ctx);

	auto addInPlace = [&](LimbChain& dst, const LimbChain& src) {
		hazeCheck(hazeAddMrp(dst.data(), dst.asConst().data(), src.asConst().data(), qpBase.data(), qpBase.size(), nullptr), "hazeAddMrp(partialSum)");
	};
	// In-place add of a base-Q chain into the leading Q rows of an extended chain (the P rows of the term are zero).
	auto addQIntoExt = [&](LimbChain& ext, const LimbChain& srcQ) {
		std::vector<void*> dst(towers);
		std::vector<const void*> s1(towers);
		for (size_t t = 0; t < towers; ++t) {
			dst[t] = ext[t];
			s1[t]  = ext[t];
		}
		hazeCheck(hazeAddMrp(dst.data(), s1.data(), srcQ.asConst().data(), qSub.data(), towers, nullptr), "hazeAddMrp(partialSum Q)");
	};
	auto automorphChain = [&](const LimbChain& src, uint32_t autoIndex) {
		LimbChain out(qpBase.size(), polyBytes_);
		hazeCheck(hazeAutomorphMrp(out.data(), src.asConst().data(), autoIndex, qpBase.data(), qpBase.size(), nullptr), "hazeAutomorphMrp(partialSum)");
		return out;
	};

	// acc starts as c0 · (P mod q_i) on the Q rows (the P rows are zero until the first level adds into them).
	const auto pmodq = pModqPrefix(towers);
	LimbChain raised0(towers, polyBytes_);
	hazeCheck(hazeMulScalarMrp(raised0.data(), x.p->c0().asConst().data(), pmodq.data(), qSub.data(), towers, nullptr), "hazeMulScalarMrp(partialSum raise)");
	std::optional<LimbChain> acc;

	// The current c1: the input's until the first level has settled its update.
	const LimbChain* c1 = &x.p->c1();
	LimbChain c1Owned;
	for (int s = startFactor; s < size; s *= radix) {
		const HoistedDigits digits = hoistedKeyswitchPrefix(*c1, towers, numPartQ_);
		std::optional<LimbChain> lvl0, lvl1;
		for (int idx = stride * s; idx < stride * size && idx < radix * stride * s; idx += stride * s) {
			const uint32_t autoIndex = context->FindAutomorphismIndex(static_cast<uint32_t>(idx));
			KsKey& key				 = autoKeyFor(ctx, autoIndex);
			KsContribution ks		 = hybridKeyswitchFromDigits(digits, key, /*ext=*/true);
			// ba[0] += acc, then the automorphism of both extended components.
			if (acc) {
				addInPlace(ks.b, *acc);
			} else {
				addQIntoExt(ks.b, raised0);
			}
			LimbChain ba0 = automorphChain(ks.b, autoIndex);
			LimbChain ba1 = automorphChain(ks.a, autoIndex);
			if (!lvl0) {
				lvl0 = std::move(ba0);
				lvl1 = std::move(ba1);
			} else {
				addInPlace(*lvl0, ba0);
				addInPlace(*lvl1, ba1);
			}
		}
		if (!lvl0) {
			continue; // no rotation at this level
		}
		// acc += lvl0 (the first level also carries c0·P on the Q rows).
		if (!acc) {
			acc = std::move(*lvl0);
			addQIntoExt(*acc, raised0);
		} else {
			addInPlace(*acc, *lvl0);
		}
		// c1 += ModDown(lvl1): settled once per level, the next level's digits need it.
		LimbChain down	= extModDownChain(*lvl1, towers);
		LimbChain next	= addChain(polyBytes_, qSub, c1->asConst().data(), down.asConst().data());
		c1Owned			= std::move(next);
		c1				= &c1Owned;
	}
	if (!acc) {
		return x;
	}
	LimbChain out0 = extModDownChain(*acc, towers);

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(hazebk::makeComponents(std::move(out0), std::move(c1Owned)), res);
	return res;
}

Ciphertext<DCRTPoly> HazeEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	auto p		= ensureCt(ctx, ct);
	Operand res = partialSumCore(ctx, asOperand(p), stride, slots, /*startFactor=*/1);
	if (res.p.get() == p.get()) { // nothing to fold: a copy of the input
		return std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	}
	// No slots-shrink: CudaEngine::accumulateSum now saves and restores the input slots around
	// Accumulate, so the CUDA broadcast-shrink this used to compensate for no longer happens, and the
	// CPU path never shrank. Keeping it here would diverge from both.
	return wrapDeviceResult(ctx, ct, std::move(res.p));
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	auto p		= ensureCt(ctx, ct);
	Operand res = partialSumCore(ctx, asOperand(p), stride, slots, /*startFactor=*/1);
	if (res.p.get() != p.get()) {
		rebindPayload(*p, std::move(res.p->components), res);
	}
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	auto p				 = ensureCt(ctx, ct);
	const size_t ctSlots = p->slots;
	Operand res			 = partialSumCore(ctx, asOperand(p), stride, slots, /*startFactor=*/start);
	if (res.p.get() != p.get()) {
		rebindPayload(*p, std::move(res.p->components), res);
	}
	// Cascade slots-shrink (AccumulateBroadcast.cu:41-42): result is a broadcast over stride*startFactor.
	if (static_cast<size_t>(slots) * static_cast<size_t>(stride) == ctSlots)
		ct->SetSlots(static_cast<size_t>(stride) * static_cast<size_t>(start));
}

// ---- Bootstrap setup hooks (staged compute lands in a later change) ----

BootstrapSetupPolicy HazeEngine::bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const {
	// CUDA parity (#9): mirror CudaEngine::bootstrapSetupPolicy (CudaEngine.cpp:489-493) —
	// forward {precompute, btsfirstboot, modEvalLevels} verbatim, instead of the CPU oracle's
	// hard-coded {true, modall!=0, -1} (OpenFheEngine.cpp:400-408). The Engine-dispatched
	// CUDA bootstrap (FIDESlib::CKKS::Bootstrap, Bootstrap.cu:169) is ALWAYS ModRaise-first
	// regardless of this flag; BTSlotsEncoding here only steers OpenFHE's host setup (correction
	// factor + StC encode level lDec, ckksrns-fhe.cpp:108-216), and for REAL data the precom
	// LAYOUT is identical either way (the `REAL || !BTSlotsEncoding` guards at :234,257 fire on
	// REAL). With btsfirstboot=false (the EvalBootstrap default), the OpenFHE host oracle the
	// parity test compares against also dispatches the normal ModRaise-first EvalBootstrap, so
	// haze's ModRaise-first port and the oracle stay in lockstep.
	return BootstrapSetupPolicy{ precompute, btsfirstboot, modEvalLevels };
}

void HazeEngine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	if (isContextLoaded()) {
		throw std::runtime_error("Context is already loaded");
	}
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);
	auto& context = hostContext(ctx);
	context->EvalBootstrapKeyGen(skImpl, slots);
	ctx.slots_bootstrap.push_back(slots);
}

std::vector<uint64_t> HazeEngine::qPrefix(size_t towers) const {
	return { qBase_.begin(), qBase_.begin() + static_cast<std::ptrdiff_t>(towers) };
}

// ---- Chebyshev series + convolution ----

namespace {

// ---------------------------------------------------------------------------
// Chebyshev Paterson-Stockmeyer is implemented as HazeEngine member functions
// (see below, after the convolution helpers) so the base-case weighted sums can
// call the Operand-level evalLinearWSumMutableCore. The recursion and affine map
// are ported from the CUDA spec src/CKKS/ApproxModEval.cu (NOT OpenFHE).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Convolution transform — ported from cpuConvolutionTransform
// (api/engine/cpu/OpenFheEngine.cpp:443-538) over the facade.
// ---------------------------------------------------------------------------

// kCpuInternalGStep must match INTERNAL_GSTEP in FIDESlib::CKKS::ConvolutionTransform
// and the GetConvolutionTransformRotationIndices logic so key coverage aligns.
constexpr uint32_t kHazeInternalGStep = 8;

// Tree-reduce v[0..count) into v[0]: pairwise when even, linear when odd —
// exact copy of cpuTreeAccumulate (OpenFheEngine.cpp:447-456) over the facade.
static void hazeTreeAccumulate(CryptoContextImpl<DCRTPoly>& ctx,
  std::vector<Ciphertext<DCRTPoly>>& v, uint32_t count) {
	if (count % 2 == 0) {
		for (uint32_t active = count; active > 1; active /= 2)
			for (uint32_t j = 0; j < active / 2; ++j)
				ctx.EvalAddInPlace(v[j], v[j + active / 2]);
	} else {
		for (uint32_t j = 1; j < count; ++j)
			ctx.EvalAddInPlace(v[0], v[j]);
	}
}

// Core convolution transform: hoisted baby steps, block giant steps,
// tree-reduce per block, inter-block rotation, rescale. Ported from
// cpuConvolutionTransform (OpenFheEngine.cpp:462-538). Plaintext pointers
// stay as fideslib Plaintext values; ctx.EvalMult(ct, pt) handles loading.
static Ciphertext<DCRTPoly> hazeConvolutionTransform(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly> ct,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int bStep,
  int gStep,
  int stride,
  int rowSize,
  Plaintext* mask,
  int maskRotationStride) {
	if (rowSize == 0)
		rowSize = bStep * gStep;

	// Rescale a freshly-multiplied (NSD==2) input, matching the GPU path.
	if (ct->GetNoiseScaleDeg() == 2)
		ctx.RescaleInPlace(ct);

	// Phase 1: hoisted baby-step rotations.
	auto precomp			= ctx.EvalFastRotationPrecompute(ct);
	uint32_t m				= ctx.GetCyclotomicOrder();
	std::vector<Ciphertext<DCRTPoly>> fastRotation(bStep);
	for (int i = 0; i < bStep; ++i)
		fastRotation[i] = (indexes[i] == 0) ? std::make_shared<CiphertextImpl<DCRTPoly>>(*ct) : ctx.EvalFastRotation(ct, indexes[i], m, precomp);

	// Phase 2: process blocks of kHazeInternalGStep giant steps.
	uint32_t blockCount = (static_cast<uint32_t>(gStep) + kHazeInternalGStep - 1) / kHazeInternalGStep;
	std::vector<Ciphertext<DCRTPoly>> blockResults;
	blockResults.reserve(blockCount);

	for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
		uint32_t blockStart		   = blockIdx * kHazeInternalGStep;
		uint32_t blockEnd		   = std::min(blockStart + kHazeInternalGStep, static_cast<uint32_t>(gStep));
		uint32_t currentBlockGStep = blockEnd - blockStart;

		std::vector<Ciphertext<DCRTPoly>> results(currentBlockGStep);
		for (uint32_t j = 0; j < currentBlockGStep; ++j) {
			uint32_t globalJ = blockStart + j;
			// Dot product: results[j] = sum_i fastRotation[i] * pts[bStep*globalJ + i].
			Plaintext pt0 = pts[static_cast<size_t>(bStep) * globalJ];
			results[j]	  = ctx.EvalMult(fastRotation[0], pt0);
			for (int i = 1; i < bStep; ++i) {
				int ptIdx = bStep * static_cast<int>(globalJ) + i;
				if (ptIdx < rowSize) {
					Plaintext pti = pts[static_cast<size_t>(ptIdx)];
					ctx.EvalAddInPlace(results[j], ctx.EvalMult(fastRotation[i], pti));
				}
			}

			if (mask != nullptr) {
				// temp = result + rot(result, s) + rot(result, 2s); result = temp * mask.
				auto temp = std::make_shared<CiphertextImpl<DCRTPoly>>(*results[j]);
				ctx.EvalAddInPlace(temp, ctx.EvalRotate(results[j], maskRotationStride));
				ctx.EvalAddInPlace(temp, ctx.EvalRotate(results[j], 2 * maskRotationStride));
				results[j] = ctx.EvalMult(temp, *mask);
			}

			// Intra-block rotation by stride * (currentBlockGStep - j).
			int rotation = stride * static_cast<int>(currentBlockGStep - j);
			if (rotation != 0)
				results[j] = ctx.EvalRotate(results[j], rotation);
		}

		hazeTreeAccumulate(ctx, results, currentBlockGStep);
		blockResults.push_back(results[0]);
	}

	// Phase 3: inter-block rotation and accumulation.
	if (blockCount > 1) {
		int baseRotation = static_cast<int>(kHazeInternalGStep) * stride;
		for (uint32_t blockIdx = 0; blockIdx < blockCount - 1; ++blockIdx) {
			int rotation = static_cast<int>(blockCount - 1 - blockIdx) * baseRotation;
			if (rotation != 0)
				blockResults[blockIdx] = ctx.EvalRotate(blockResults[blockIdx], rotation);
		}
		hazeTreeAccumulate(ctx, blockResults, blockCount);
	}

	// Exit rescale: only the SPECIAL (mask) path eager-rescales (NSD 2→1) at its end, matching CUDA
	// SpecialConvolutionTransform's trailing ctxt.rescale() (LinearTransform.cu:648); the regular
	// ConvolutionTransform has NO exit rescale (LinearTransform.cu:436 just copies). Now that Rescale is
	// unconditionally eager (parity #3), gating on the mask reproduces both paths' level/NSD exactly —
	// an unconditional Rescale here would over-drop the regular path a level.
	if (mask != nullptr)
		return ctx.Rescale(blockResults[0]);
	return blockResults[0];
}

} // namespace

// ---- Chebyshev series ----

size_t HazeEngine::towersOf(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	return ensureCt(ctx, ct)->towers;
}

void HazeEngine::dropCiphertextToTowers(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, size_t targetTowers) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "dropCiphertextToTowers");
	if (p->towers <= targetTowers)
		return;
	// LimbChain::truncate frees the trailing limbs in place (no IR, no arithmetic — the CUDA
	// dropToLevel/OpenFHE DropLastElements analog); towers is metadata alongside it.
	for (auto& comp : p->components)
		comp.truncate(targetTowers);
	p->towers = targetTowers;
}

Ciphertext<DCRTPoly> HazeEngine::openFheChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& x,
  const std::vector<double>& coefficients,
  double a, double b) {
	using Ct			  = Ciphertext<DCRTPoly>;
	const auto st		  = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	const bool manual	  = (st == lbcrypto::FIXEDMANUAL);
	const size_t sizeQ	  = qBase_.size();
	auto clone			  = [](const Ct& c) { return std::make_shared<CiphertextImpl<DCRTPoly>>(*c); };
	auto level			  = [&](const Ct& c) { return sizeQ - towersOf(ctx, c); }; // OpenFHE GetLevel
	// OpenFHE's public ModReduceInPlace and LevelReduceInPlace act only under FIXEDMANUAL (the automatic
	// techniques rescale lazily inside the operations).
	auto modReduce		  = [&](Ct& c) {
		if (manual)
			ctx.RescaleInPlace(c);
	};
	auto levelReduce = [&](Ct& c, size_t levels) {
		if (manual && levels > 0)
			dropCiphertextToTowers(ctx, c, towersOf(ctx, c) - levels);
	};
	// AdjustLevelsAndDepthInPlace(c1, c2) on two live ciphertexts (the automatic techniques).
	auto adjustPair = [&](Ct& c1, Ct& c2) {
		auto p1 = ensureCt(ctx, c1);
		auto p2 = ensureCt(ctx, c2);
		Operand o1 = asOperand(p1);
		Operand o2 = asOperand(p2);
		adjustForAddOrSub(o1, o2);
		auto commit = [&](Ct& c, const std::shared_ptr<hazebk::HazePayload>& orig, Operand& o) {
			if (o.p.get() != orig.get()) {
				c = wrapDeviceResult(ctx, c, std::move(o.p));
			}
			if (o.towers < towersOf(ctx, c)) {
				dropCiphertextToTowers(ctx, c, o.towers);
			}
			// The adjustment sets the scaling factor of the operand view, not of the payload.
			auto pc			  = ensureCt(ctx, c);
			pc->scalingFactor = o.scalingFactor;
			pc->noiseScaleDeg = o.noiseScaleDeg;
		};
		commit(c1, p1, o1);
		commit(c2, p2, o2);
	};

	// OpenFHE EvalPartialLinearWSum: the first `limit` ciphertexts weighted by constants[1..limit].
	auto partialWSum = [&](const std::vector<Ct>& cts, const std::vector<double>& constants, uint32_t limit) -> Ct {
		if (limit == 0)
			limit = static_cast<uint32_t>(cts.size());
		std::vector<Ct> w(limit);
		for (uint32_t i = 0; i < limit; ++i)
			w[i] = clone(cts[i]);
		if (!manual) {
			// Bring every term to the highest level among them (a depth-2 one at equal level wins).
			size_t maxLevel = level(w[0]);
			uint32_t maxIdx = 0;
			for (uint32_t i = 1; i < limit; ++i) {
				const size_t li = level(w[i]);
				if (li > maxLevel || (li == maxLevel && w[i]->GetNoiseScaleDeg() == 2)) {
					maxLevel = li;
					maxIdx	 = i;
				}
			}
			for (uint32_t i = 0; i < maxIdx; ++i)
				adjustPair(w[i], w[maxIdx]);
			for (uint32_t i = maxIdx + 1; i < limit; ++i)
				adjustPair(w[i], w[maxIdx]);
			if (w[maxIdx]->GetNoiseScaleDeg() == 2) {
				for (uint32_t i = 0; i < limit; ++i)
					ctx.RescaleInPlace(w[i]); // ModReduceInternalInPlace
			}
		}
		ctx.EvalMultInPlace(w[0], constants[1]);
		for (uint32_t i = 1; i < limit; ++i) {
			ctx.EvalMultInPlace(w[i], constants[i + 1]);
			ctx.EvalAddInPlace(w[0], w[i]);
		}
		modReduce(w[0]);
		return w[0];
	};

	// OpenFHE InnerEvalChebyshevPS.
	std::function<Ct(const std::vector<double>&, uint32_t, uint32_t, const std::vector<Ct>&, const std::vector<Ct>&)> inner;
	inner = [&](const std::vector<double>& coeffs, uint32_t k, uint32_t m, const std::vector<Ct>& T, const std::vector<Ct>& T2) -> Ct {
		const uint32_t k2m2k = k * (1u << (m - 1)) - k;
		std::vector<double> Tkm(k2m2k + k + 1);
		Tkm.back()	= 1;
		auto divqr	= lbcrypto::LongDivisionChebyshev(coeffs, Tkm);
		auto& r2	= divqr->r;
		if (const uint32_t n = lbcrypto::Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
			r2.resize(n + 1);
			r2[k2m2k] -= 1;
		} else {
			r2.resize(k2m2k + 1);
			r2.back() = -1;
		}
		auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

		Ct cu, qu, su;
		{
			if (lbcrypto::Degree(divqr->q) > k) {
				qu = inner(divqr->q, k, m - 1, T, T2);
			} else {
				qu					 = clone(T[k - 1]);
				const uint32_t limit = static_cast<uint32_t>(std::log2(divqr->q.back()));
				for (uint32_t i = 0; i < limit; ++i)
					ctx.EvalAddInPlace(qu, qu);
				ctx.EvalAddInPlace(qu, divqr->q.front() / 2.0);
				divqr->q.resize(k);
				if (const uint32_t n = lbcrypto::Degree(divqr->q); n > 0)
					ctx.EvalAddInPlace(qu, partialWSum(T, divqr->q, n));
			}
		}
		{
			auto& s2 = divcs->r;
			s2.resize(k2m2k + 1);
			s2.back() = 1;
			if (lbcrypto::Degree(s2) > k) {
				su = inner(s2, k, m - 1, T, T2);
			} else {
				su = clone(T[k - 1]);
				s2.resize(k);
				if (const uint32_t n = lbcrypto::Degree(s2); n > 0)
					ctx.EvalAddInPlace(su, partialWSum(T, s2, n));
				ctx.EvalAddInPlace(su, s2.front() / 2.0);
				levelReduce(su, 1);
			}
		}
		if (const uint32_t n = lbcrypto::Degree(divcs->q); n >= 1) {
			if (n == 1) {
				if (lbcrypto::IsNotEqualOne(divcs->q[1])) {
					cu = ctx.EvalMult(T.front(), divcs->q[1]);
					modReduce(cu);
				} else {
					cu = clone(T.front());
				}
			} else {
				cu = partialWSum(T, divcs->q, n);
			}
			ctx.EvalAddInPlace(cu, divcs->q.front() / 2.0);
			levelReduce(cu, level(T2[m - 1]) - level(cu));
		}
		cu = cu ? ctx.EvalAdd(T2[m - 1], cu) : ctx.EvalAdd(T2[m - 1], divcs->q.front() / 2.0);
		Ct result = ctx.EvalMult(cu, qu);
		modReduce(result);
		ctx.EvalAddInPlace(result, su);
		return result;
	};

	const uint32_t d = lbcrypto::Degree(coefficients);
	std::vector<Ct> T;
	// T_1(y) = y after the affine map to [-1, 1]; one level when the interval is not already that.
	auto firstPower = [&]() -> Ct {
		if (!lbcrypto::IsNotEqualNegOne(a) && !lbcrypto::IsNotEqualOne(b)) {
			return clone(x);
		}
		const double alpha = 2 / (b - a);
		const double beta  = a * alpha;
		Ct t			   = ctx.EvalMult(x, alpha);
		modReduce(t);
		ctx.EvalAddInPlace(t, -1.0 - beta);
		return t;
	};
	// T_i for i = 2..k by binary-tree multiplication.
	auto buildPowers = [&](std::vector<Ct>& P, uint32_t k) {
		for (uint32_t i = 2; i <= k; ++i) {
			if (i & 0x1) {
				P[i - 1] = ctx.EvalMult(P[i / 2 - 1], P[i / 2]);
				ctx.EvalAddInPlace(P[i - 1], P[i - 1]);
				modReduce(P[i - 1]);
				ctx.EvalSubInPlace(P[i - 1], P[0]);
			} else {
				P[i - 1] = ctx.EvalSquare(P[i / 2 - 1]);
				ctx.EvalAddInPlace(P[i - 1], P[i - 1]);
				modReduce(P[i - 1]);
				ctx.EvalAddInPlace(P[i - 1], -1.0);
			}
		}
	};

	if (d < 5) {
		// internalEvalChebyPolysLinear + internalEvalChebyshevSeriesLinearWithPrecomp.
		const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 1;
		T.resize(k);
		T[0] = firstPower();
		buildPowers(T, k);
		for (uint32_t i = 1; i < k; ++i)
			levelReduce(T[i - 1], level(T[k - 1]) - level(T[i - 1]));

		const uint32_t kk = static_cast<uint32_t>(coefficients.size()) - 2;
		Ct result		  = ctx.EvalMult(T[kk], coefficients[kk + 1]);
		for (uint32_t i = 0; i < kk; ++i) {
			if (lbcrypto::IsNotEqualZero(coefficients[i + 1])) {
				ctx.EvalMultInPlace(T[i], coefficients[i + 1]);
				ctx.EvalAddInPlace(result, T[i]);
			}
		}
		modReduce(result);
		ctx.EvalAddInPlace(result, coefficients[0] / 2.0);
		return result;
	}

	// internalEvalChebyPolysPS.
	const auto degs	   = lbcrypto::ComputeDegreesPS(d);
	const uint32_t k   = degs[0];
	const uint32_t m   = degs[1];
	T.resize(k);
	T[0] = firstPower();
	buildPowers(T, k);
	if (manual) {
		for (uint32_t i = 1; i < k; ++i)
			levelReduce(T[i - 1], level(T[k - 1]) - level(T[i - 1]));
	} else {
		for (uint32_t i = 1; i < k; ++i)
			adjustPair(T[i - 1], T[k - 1]);
	}
	std::vector<Ct> T2(m);
	T2[0]	  = T.back();
	Ct T2km1  = T.back();
	for (uint32_t i = 1; i < m; ++i) {
		T2[i] = ctx.EvalSquare(T2[i - 1]);
		ctx.EvalAddInPlace(T2[i], T2[i]);
		modReduce(T2[i]);
		ctx.EvalAddInPlace(T2[i], -1.0);

		T2km1 = ctx.EvalMult(T2km1, T2[i]);
		ctx.EvalAddInPlace(T2km1, T2km1);
		modReduce(T2km1);
		ctx.EvalSubInPlace(T2km1, T2[0]);
	}

	// internalEvalChebyshevSeriesPSWithPrecomp.
	const uint32_t k2m2k = k * (1u << (m - 1)) - k;
	auto f2				 = coefficients;
	f2.resize(lbcrypto::Degree(f2) + 1);
	f2.resize(2 * k2m2k + k + 1);
	f2.back() = 1;
	return ctx.EvalSub(inner(f2, k, m, T, T2), T2km1);
}

Ciphertext<DCRTPoly> HazeEngine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coeffs,
  double a, double b) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalChebyshevSeries");
	requireRelinearized(*p, "EvalChebyshevSeries");
	return openFheChebyshevSeries(ctx, ct, coeffs, a, b);
}

void HazeEngine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coeffs,
  double a, double b) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalChebyshevSeriesInPlace");
	requireRelinearized(*p, "EvalChebyshevSeriesInPlace");
	ct = openFheChebyshevSeries(ctx, ct, coeffs, a, b);
}

// ---- ConvolutionTransform ----

void HazeEngine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "ConvolutionTransformInPlace");
	ct = hazeConvolutionTransform(ctx, ct, pts, indexes, bStep, gStep, stride, rowSize, nullptr, 0);
}

void HazeEngine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "SpecialConvolutionTransformInPlace");
	ct = hazeConvolutionTransform(ctx, ct, pts, indexes, bStep, gStep, stride, rowSize, &mask, maskRotationStride);
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
