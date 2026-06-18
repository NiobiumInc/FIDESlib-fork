#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeEngine.hpp"

#include "engine/haze/HazeScalarEncode.hpp"

#include "CryptoContext.hpp"

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

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& hostCt(const Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
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
		const auto& np	 = elem.GetElementAtIndex(static_cast<usint>(t));
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
	// haze's ciphertext-modulus table caps at 64 entries (kMaxCiphertextModuli).
	if (qBase_.size() + pBase_.size() > 64) {
		throw std::runtime_error("haze backend: |Q|+|P| = " + std::to_string(qBase_.size()) + "+" + std::to_string(pBase_.size()) + " exceeds haze's 64-moduli table");
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
		hazeCheck(hazeSetProgramInfo("fideslib", "1.0", "FIDESlib HazeEngine"), "hazeSetProgramInfo");

		// FBC mode (constructor-selected), latched at bring-up before the first compute:
		// reduced-noise selects the centered FBC matching a WITH_REDUCED_NOISE OpenFHE reference
		// (bit-exact parity); Montgomery selects the 4-op SwitchModulus center shape. Montgomery
		// is rejected by the local simulator at first compute, so it is only valid against a
		// hardware/transport target.
		hazeCheck(hazeSetReducedNoise(reducedNoise_ ? 1 : 0), "hazeSetReducedNoise");
		hazeCheck(hazeSetMontgomery(montgomery_ ? 1 : 0), "hazeSetMontgomery");
		// The full hardware data format pairs Montgomery with bit-reversal (the
		// compiler's --niobium_hw), so couple them: a Montgomery run targets the
		// hardware datapath, an ordinary run leaves both off.
		hazeCheck(hazeSetBitReversal(montgomery_ ? 1 : 0), "hazeSetBitReversal");
		// Replay target (default: in-process local simulator). Montgomery is only
		// valid against a transport target such as FUNC_SIM / FUNC_SIM_HW (the
		// Niobium hardware simulator); set it before the bridge brings up the
		// compiler so replay dispatch routes there.
		if (const char* tgt = std::getenv("FIDESLIB_HAZE_TARGET"); tgt != nullptr && tgt[0] != '\0')
			hazeCheck(hazeSetTarget(tgt), "hazeSetTarget");

		// Program directory: <FIDESLIB_HAZE_RUNS_DIR | /tmp/fideslib-haze-runs>/<pid>/ctx<N>.
		// Each flush writes a multi-hundred-MB project dir, so (a) keep it off slow shared
		// filesystems and (b) clean it up at teardown unless FIDESLIB_HAZE_KEEP_RUNS is set.
		static std::atomic<int> ctxCounter{ 0 };
		const char* runsRoot   = std::getenv("FIDESLIB_HAZE_RUNS_DIR");
		const std::string root = (runsRoot != nullptr && runsRoot[0] != '\0') ? runsRoot : "/tmp/fideslib-haze-runs";
		programDir_			   = root + "/" + std::to_string(getpid()) + "/ctx" + std::to_string(ctxCounter.fetch_add(1));
		std::filesystem::create_directories(programDir_);
		hazeCheck(hazeSetProgramDirectory(programDir_.c_str()), "hazeSetProgramDirectory");

		hazeCheck(hazeSetRingDimension(ringDim_), "hazeSetRingDimension");

		// Pure-C bridge: haze rebuilds a matching CryptoContext from (ring_dim, first Q prime);
		// the full Q∥P chain is conveyed via hazeSetCiphertextModulus below. Must be re-called
		// after every hazeDeviceReset (replay_bridge.h contract), i.e. on every loadContext.
		uint64_t picked = 0;
		hazeCheck(hazeReplayBridgeInitCryptoContext(ringDim_, qBase_.front(), &picked), "hazeReplayBridgeInitCryptoContext");
		if (picked == 0) {
			throw std::runtime_error("haze backend: replay bridge returned a zero modulus");
		}

		int modIdx = 0;
		for (uint64_t q : qBase_) {
			hazeCheck(hazeSetCiphertextModulus(modIdx++, q), "hazeSetCiphertextModulus(Q)");
		}
		for (uint64_t p : pBase_) {
			hazeCheck(hazeSetCiphertextModulus(modIdx++, p), "hazeSetCiphertextModulus(P)");
		}
		// Do NOT call hazeSetTwiddleFactors: haze's e2e suite never sets them and is bit-exact
		// vs OpenFHE (FHETCH derives per-modulus roots internally). Documented fallback if a
		// context ever disagrees: per-limb GetRootOfUnity().
		hazeCheck(hazeConfigureDevice(), "hazeConfigureDevice");
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
	if (!programDir_.empty() && std::getenv("FIDESLIB_HAZE_KEEP_RUNS") == nullptr) {
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
	if (elements.size() != 2) {
		throw std::runtime_error("haze backend: only degree-1 ciphertexts (2 components) are supported; got " + std::to_string(elements.size()));
	}

	auto rows0 = extractRows(elements[0], ringDim_);
	auto rows1 = extractRows(elements[1], ringDim_);
	if (rows0.size() != rows1.size()) {
		throw std::runtime_error("haze backend: ciphertext components disagree on tower count");
	}

	auto payload		   = std::make_shared<HazePayload>();
	payload->towers		   = rows0.size();
	payload->c0			   = LimbChain(rows0);
	payload->c1			   = LimbChain(rows1);
	payload->noiseScaleDeg = host->GetNoiseScaleDeg();
	payload->scalingFactor = host->GetScalingFactor();
	payload->slots		   = host->GetSlots();
	payload->state		   = Residency::Uploaded;

	ct->device = std::make_any<std::shared_ptr<HazePayload>>(std::move(payload));
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

	auto payload			  = std::make_shared<HazePtPayload>();
	payload->towers			  = rows.size();
	payload->chain			  = LimbChain(rows);
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
		for (std::size_t t = 0; t < payload->towers; ++t) {
			hazeCheck(hazeTagOutput(payload->c0[t]), "hazeTagOutput");
			hazeCheck(hazeTagOutput(payload->c1[t]), "hazeTagOutput");
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

	ct->EnsureLazyHostCopy();
	auto& host = hostCt(ct);

	// Make the host shell carry exactly payload->towers limbs: grow it via a zero ciphertext
	// at the full level when the device result has MORE limbs (post-ModRaise — CudaEngine
	// :536-545), then trim down with DropLastElements (no arithmetic) when it has fewer.
	size_t hostTowers = host->GetElements()[0].GetNumOfElements();
	if (hostTowers < payload->towers) {
		auto& context		  = hostContext(ctx);
		const auto elemParams = context->GetCryptoParameters()->GetElementParams();
		lbcrypto::DCRTPoly zero(elemParams, Format::EVALUATION, true);
		auto fresh = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*host);
		fresh->SetElements({ zero, zero });
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
	const LimbChain* chains[2] = { &payload->c0, &payload->c1 };
	for (size_t e = 0; e < 2; ++e) {
		auto& towersVec = host->GetElements()[e].GetAllElements();
		for (std::size_t t = 0; t < payload->towers; ++t) {
			hazeCheck(hazeMemcpy(buf.data(), (*chains[e])[t], polyBytes_, HAZE_MEMCPY_DEVICE_TO_HOST), "hazeMemcpy(D2H)");
			auto& np = towersVec[t];
			lbcrypto::NativeVector nv(static_cast<usint>(ringDim_), np.GetModulus());
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

	if (srcPayload->state == Residency::Recorded) {
		// In-flight value: record per-residue pass-through D2D copies (legal mid-epoch;
		// epoch.cpp copy_device_to_device). The clone is itself Recorded.
		const auto base = qPrefix(srcPayload->towers);
		LimbChain c0(srcPayload->towers, polyBytes_);
		LimbChain c1(srcPayload->towers, polyBytes_);
		hazeCheck(hazeMemcpyMrp(c0.data(), srcPayload->c0.asConst().data(), polyBytes_, HAZE_MEMCPY_DEVICE_TO_DEVICE, base.data(), base.size()), "hazeMemcpyMrp(D2D)");
		hazeCheck(hazeMemcpyMrp(c1.data(), srcPayload->c1.asConst().data(), polyBytes_, HAZE_MEMCPY_DEVICE_TO_DEVICE, base.data(), base.size()), "hazeMemcpyMrp(D2D)");
		clone->c0	 = std::move(c0);
		clone->c1	 = std::move(c1);
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
		clone->c0			  = LimbChain(pull(srcPayload->c0));
		clone->c1			  = LimbChain(pull(srcPayload->c1));
		clone->state		  = Residency::Uploaded;
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
	payload->c0.truncate(targetTowers);
	payload->c1.truncate(targetTowers);
	payload->towers = targetTowers;
}

// ---- FIXEDAUTO adjust + scalar-op cores (OpenFHE ckksrns-leveledshe is the oracle) ----

HazeEngine::Operand HazeEngine::asOperand(const std::shared_ptr<HazePayload>& p) const {
	return Operand{ p, p->towers, p->noiseScaleDeg, p->scalingFactor, p->slots };
}

hazebk::ScalarEncodeParams HazeEngine::scalarParams() const {
	return hazebk::ScalarEncodeParams{ qBase_, sfReal_, sfRealBig_, modReduceFactor_, static_cast<lbcrypto::ScalingTechnique>(scalingTech_) };
}

LimbChain HazeEngine::rescaleChainOneTower(const LimbChain& src, size_t srcTowers) {
	// Port of ops.cpp rescale_chain_one_tower: INTT → ModDown dropping the trailing Q prime
	// (centered lift + q_l^{-1}, matching OpenFHE ModReduce exactly) → NTT.
	if (srcTowers < 2) {
		throw std::runtime_error("haze backend: rescale needs at least 2 towers");
	}
	const size_t dstTowers = srcTowers - 1;
	const auto srcBase	   = qPrefix(srcTowers);
	const auto dstBase	   = qPrefix(dstTowers);
	const std::vector<uint64_t> rescaleBase = { srcBase.back() };

	const hazeModDownParams mdParams = {
		/*.src_base =*/srcBase.data(),
		/*.src_base_len =*/srcBase.size(),
		/*.rescale_base =*/rescaleBase.data(),
		/*.rescale_base_len =*/rescaleBase.size(),
	};

	LimbChain intt(srcTowers, polyBytes_);
	LimbChain md(dstTowers, polyBytes_);
	LimbChain ntt(dstTowers, polyBytes_);
	hazeCheck(hazeINTTMrp(intt.data(), src.asConst().data(), srcBase.data(), srcBase.size(), nullptr), "hazeINTTMrp");
	hazeCheck(hazeModDown(md.data(), intt.asConst().data(), &mdParams, nullptr), "hazeModDown");
	hazeCheck(hazeNTTMrp(ntt.data(), md.asConst().data(), dstBase.data(), dstBase.size(), nullptr), "hazeNTTMrp");
	return ntt;
}

HazeEngine::Operand HazeEngine::multScalarCore(const Operand& x, double operand) {
	// OpenFHE EvalMultCoreInPlace: per-limb CRT factors at the current level, NSD+1,
	// sf ×= ScalingFactorReal[level]. No pre-rescale (callers add the precheck).
	const auto factors = hazebk::elemForEvalMult(scalarParams(), x.towers, operand);
	const auto base	   = qPrefix(x.towers);
	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1;
	hazeCheck(hazeMulScalarMrp(out0.data(), x.p->c0.asConst().data(), factors.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	if (!x.p->c1.empty()) { // 1-component operands (morphed plaintexts) have no c1
		out1 = LimbChain(x.towers, polyBytes_);
		hazeCheck(hazeMulScalarMrp(out1.data(), x.p->c1.asConst().data(), factors.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	}

	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + 1;
	res.scalingFactor = x.scalingFactor * sfReal_[levelOf(x)];
	res.p			  = finishPayload(std::move(out0), std::move(out1), res);
	return res;
}

HazeEngine::Operand HazeEngine::rescaleCore(const Operand& x) {
	// OpenFHE ModReduceInternalInPlace (one level): towers−1, NSD−1,
	// sf ÷= ModReduceFactor[oldTowers−1].
	if (x.noiseScaleDeg < 1) {
		throw std::runtime_error("haze backend: rescale on noiseScaleDeg 0 would underflow");
	}
	LimbChain out0 = rescaleChainOneTower(x.p->c0, x.towers);
	LimbChain out1;
	if (!x.p->c1.empty()) { // 1-component operands (morphed plaintexts) have no c1
		out1 = rescaleChainOneTower(x.p->c1, x.towers);
	}

	Operand res		  = x;
	res.towers		  = x.towers - 1;
	res.noiseScaleDeg = x.noiseScaleDeg - 1;
	res.scalingFactor = x.scalingFactor / modReduceFactor_[x.towers - 1];
	res.p			  = finishPayload(std::move(out0), std::move(out1), res);
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
	std::vector<uint64_t> scalars(x.towers);
	for (size_t i = 0; i < x.towers; ++i) {
		scalars[i] = qBase_[i] - 1;
	}
	const auto base = qPrefix(x.towers);
	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeMulScalarMrp(out0.data(), x.p->c0.asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	hazeCheck(hazeMulScalarMrp(out1.data(), x.p->c1.asConst().data(), scalars.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");

	Operand res = x;
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
	return res;
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
	if (st == lbcrypto::FIXEDAUTO) {
		// CUDA Ciphertext::adjustForAddOrSub FIXEDAUTO branch (Ciphertext.cpp:1473-1488): a SIMPLE
		// depth-fix only — rescale a depth-2 operand or multScalar(1.0)-bump a depth-1 operand so both
		// reach a common depth — and the level alignment is PURE TRUNCATION in the add/sub tail
		// (:187-195, dropToLevel@1140 drops the fresher/more-towers operand). NOT OpenFHE's generic
		// AdjustLevelsAndDepth (the FIXEDAUTO scale tables degenerate to a constant, so its
		// multScalar(~1/S)+rescale is wasted work that perturbs NSD); that generic path is reserved for
		// the FLEXIBLE modes below. The depth-adjusted-level comparison mirrors CUDA's
		// getLevel()-NoiseLevel (getLevel = towers−1, so the −1 cancels): towers − noiseScaleDeg.
		auto depthFix = [&](Operand& x, const Operand& y) -> bool {
			const long dlx = static_cast<long>(x.towers) - static_cast<long>(x.noiseScaleDeg);
			const long dly = static_cast<long>(y.towers) - static_cast<long>(y.noiseScaleDeg);
			if (dlx > dly) {
				if (y.noiseScaleDeg == 1 && x.noiseScaleDeg == 2) {
					x = rescaleCore(x);
				} else if (y.noiseScaleDeg == 2 && x.noiseScaleDeg == 1) {
					x = multScalarCore(x, 1.0);
				}
				return true;
			} else if (y.noiseScaleDeg == 1 && x.noiseScaleDeg == 2) {
				x = rescaleCore(x);
				return true;
			} else if (x.noiseScaleDeg == 1 && y.noiseScaleDeg == 2) {
				return false; // ask the caller to adjust the other operand instead (CUDA's swap+retry)
			}
			return true;
		};
		if (!depthFix(a, b)) {
			if (!depthFix(b, a)) {
				throw std::runtime_error("haze backend: FIXEDAUTO add/sub adjust failed to reconcile depth");
			}
		}
		// Tail: pure truncation of the fresher (more-towers) operand to the other's tower count.
		const size_t towers = std::min(a.towers, b.towers);
		a.towers			= towers;
		b.towers			= towers;
		return;
	}
	// FLEXIBLEAUTO / FLEXIBLEAUTOEXT: OpenFHE/CUDA AdjustScaleAndLevel via the generic ciphertext
	// helper (towers-space; OpenFHE level = |Q| − towers). The "fresher" operand (more towers / lower
	// level) is adjusted toward the other; equal level only needs a depth bump.
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

void HazeEngine::adjustOperandToward(Operand& x, const Operand& tgt, bool ctRule) {
	{
		const size_t c1lvl	 = levelOf(x);
		const size_t c2lvl	 = levelOf(tgt);
		const size_t c1depth = x.noiseScaleDeg;
		const size_t c2depth = tgt.noiseScaleDeg;
		if (c1depth == 2) {
			if (c2depth == 2) {
				const double scf1 = x.scalingFactor;
				const double scf2 = tgt.scalingFactor;
				const double scf  = sfReal_[c1lvl];
				if (ctRule) {
					// Ciphertext rule (CUDA Ciphertext::adjustScaleAndLevel, Ciphertext.cpp:1388-1400):
					// q1 = ModReduceFactor[c2lvl+1] = modReduceFactor_[tgt.towers] (tower-indexed, matches
					// CUDA param.ModReduceFactor), and TRIM the extra top towers BEFORE rescale (dropToLevel
					// then rescale) so the correct top tower is folded when operands are >1 level apart.
					const double q1 = modReduceFactor_[tgt.towers];
					x				= multScalarCore(x, scf2 * q1 / scf1 / scf);
					if (c1lvl + 1 < c2lvl) {
						x.towers -= c2lvl - c1lvl - 1;
					}
					x = rescaleCore(x);
				} else {
					// Plaintext rule (CUDA Plaintext::adjustScaleAndLevel, Plaintext.cu:119-134): q1 =
					// ModReduceFactor[c1lvl] = modReduceFactor_[x.towers−1], and rescale BEFORE trim. Kept
					// only for adjustPtToward (the pt operand), per parity #5.
					const double q1 = modReduceFactor_[x.towers - 1];
					x				= multScalarCore(x, scf2 / scf1 * q1 / scf);
					x				= rescaleCore(x);
					if (c1lvl + 1 < c2lvl) {
						x.towers -= c2lvl - c1lvl - 1;
					}
				}
				x.scalingFactor = tgt.scalingFactor;
			} else {
				if (c1lvl + 1 == c2lvl) {
					x = rescaleCore(x);
				} else {
					const double scf1 = x.scalingFactor;
					const double scf2 = sfRealBig_[c2lvl - 1];
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
				const double scf2 = sfRealBig_[c2lvl - 1];
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
}

void HazeEngine::adjustPtToward(Operand& ptOp, const Operand& ct) {
	// GPU Plaintext::adjustPlaintextToCiphertext port (src/CKKS/Plaintext.cu:195-225) in
	// towers-space (FIDESlib level = towers − 1). The pt is a 1-component operand; the
	// cores skip its absent c1.
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st == lbcrypto::FIXEDAUTO) {
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
		// ciphertext operands, including the equal-level depth bump. ctRule=false keeps the
		// Plaintext-rule depth2/depth2 branch1 (parity #5).
		if (ptOp.towers < ct.towers) {
			throw std::runtime_error("haze backend: plaintext is encoded deeper than the ciphertext (cannot raise a plaintext)");
		}
		if (levelOf(ptOp) < levelOf(ct)) {
			adjustOperandToward(ptOp, ct, /*ctRule=*/false);
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
		hazeCheck(hazeAddScalarMrp(out0.data(), x.p->c0.asConst().data(), factors.data(), base.data(), base.size(), nullptr), "hazeAddScalarMrp");
	} else {
		out0 = passThroughChain(x.p->c0, x.towers);
	}
	LimbChain out1 = passThroughChain(x.p->c1, x.towers);

	Operand res = x;
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
	return res;
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
			hazeCheck(hazeSubMrp(out0.data(), x.p->c0.asConst().data(), pt.chain.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
		} else {
			hazeCheck(hazeAddMrp(out0.data(), x.p->c0.asConst().data(), pt.chain.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
		}
		LimbChain out1 = passThroughChain(x.p->c1, x.towers);

		Operand res = x;
		res.slots	= std::max(x.slots, pt.slots);
		res.p		= finishPayload(std::move(out0), std::move(out1), res);
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
	{
		auto morph	  = std::make_shared<HazePayload>();
		morph->c0	  = passThroughChain(pt.chain, pt.towers);
		morph->towers = pt.towers;
		morph->state  = Residency::Recorded;
		ptOp.p		  = std::move(morph);
	}
	adjustPtToward(ptOp, x);
	if (ptOp.towers < x.towers || ptOp.noiseScaleDeg != x.noiseScaleDeg) {
		throw std::runtime_error("haze backend: internal error — pt/ct disagree after adjustPlaintextToCiphertext");
	}

	const auto base = qPrefix(x.towers);
	LimbChain out0(x.towers, polyBytes_);
	if (subtract) {
		hazeCheck(hazeSubMrp(out0.data(), x.p->c0.asConst().data(), ptOp.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
	} else {
		hazeCheck(hazeAddMrp(out0.data(), x.p->c0.asConst().data(), ptOp.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	}
	LimbChain out1 = passThroughChain(x.p->c1, x.towers);

	Operand res = x;
	res.slots	= std::max(x.slots, ptOp.slots);
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
	return res;
}

std::shared_ptr<HazePayload> HazeEngine::finishPayload(LimbChain c0, LimbChain c1, const Operand& meta) const {
	auto payload		   = std::make_shared<HazePayload>();
	payload->c0			   = std::move(c0);
	payload->c1			   = std::move(c1);
	payload->towers		   = meta.towers;
	payload->noiseScaleDeg = meta.noiseScaleDeg;
	payload->scalingFactor = meta.scalingFactor;
	payload->slots		   = meta.slots;
	payload->state		   = Residency::Recorded;
	return payload;
}

void HazeEngine::rebindPayload(HazePayload& dst, LimbChain c0, LimbChain c1, const Operand& meta) const {
	dst.c0			  = std::move(c0); // frees the old chains (the allocator recycles addrs)
	dst.c1			  = std::move(c1);
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

Ciphertext<DCRTPoly> HazeEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAdd");
	requireComputable(*p2, "EvalAdd");

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForAddOrSub(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}
	const auto base = qPrefix(a.towers);
	LimbChain out0(a.towers, polyBytes_);
	LimbChain out1(a.towers, polyBytes_);
	hazeCheck(hazeAddMrp(out0.data(), a.p->c0.asConst().data(), b.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	hazeCheck(hazeAddMrp(out1.data(), a.p->c1.asConst().data(), b.p->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");

	Operand meta	   = a;
	meta.noiseScaleDeg = std::max(a.noiseScaleDeg, b.noiseScaleDeg); // equal after FIXEDAUTO adjust
	meta.slots		   = std::max(a.slots, b.slots);
	return wrapDeviceResult(ctx, ct1, finishPayload(std::move(out0), std::move(out1), meta));
}

void HazeEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAddInPlace");
	requireComputable(*p2, "EvalAddInPlace");

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForAddOrSub(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}
	const auto base = qPrefix(a.towers);
	LimbChain out0(a.towers, polyBytes_);
	LimbChain out1(a.towers, polyBytes_);
	hazeCheck(hazeAddMrp(out0.data(), a.p->c0.asConst().data(), b.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	hazeCheck(hazeAddMrp(out1.data(), a.p->c1.asConst().data(), b.p->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");

	Operand meta	   = a;
	meta.noiseScaleDeg = std::max(a.noiseScaleDeg, b.noiseScaleDeg);
	meta.slots		   = std::max(a.slots, b.slots);
	rebindPayload(*p1, std::move(out0), std::move(out1), meta);
}

Ciphertext<DCRTPoly> HazeEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalSub");
	requireComputable(*p2, "EvalSub");

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForAddOrSub(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}
	const auto base = qPrefix(a.towers);
	LimbChain out0(a.towers, polyBytes_);
	LimbChain out1(a.towers, polyBytes_);
	hazeCheck(hazeSubMrp(out0.data(), a.p->c0.asConst().data(), b.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
	hazeCheck(hazeSubMrp(out1.data(), a.p->c1.asConst().data(), b.p->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");

	Operand meta	   = a;
	meta.noiseScaleDeg = std::max(a.noiseScaleDeg, b.noiseScaleDeg);
	meta.slots		   = std::max(a.slots, b.slots);
	return wrapDeviceResult(ctx, ct1, finishPayload(std::move(out0), std::move(out1), meta));
}

void HazeEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalSubInPlace");
	requireComputable(*p2, "EvalSubInPlace");

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForAddOrSub(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}
	const auto base = qPrefix(a.towers);
	LimbChain out0(a.towers, polyBytes_);
	LimbChain out1(a.towers, polyBytes_);
	hazeCheck(hazeSubMrp(out0.data(), a.p->c0.asConst().data(), b.p->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");
	hazeCheck(hazeSubMrp(out1.data(), a.p->c1.asConst().data(), b.p->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeSubMrp");

	Operand meta	   = a;
	meta.noiseScaleDeg = std::max(a.noiseScaleDeg, b.noiseScaleDeg);
	meta.slots		   = std::max(a.slots, b.slots);
	rebindPayload(*p1, std::move(out0), std::move(out1), meta);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
}

Ciphertext<DCRTPoly> HazeEngine::evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	// Facade-level add tree, structurally from CudaEngine minus its up-front LoadCiphertext
	// loop (each EvalAdd loads its operands lazily via ensureCt).
	if (ciphertexts.empty()) {
		throw std::runtime_error("EvalAddMany: input ciphertext vector is empty");
	}
	if (ciphertexts.size() == 1) {
		return ciphertexts[0]; // the CudaEngine tree would read past an empty sum vector
	}

	const size_t inSize = ciphertexts.size();
	const size_t lim	= inSize * 2 - 2;
	std::vector<Ciphertext<DCRTPoly>> ciphertextSumVec;
	ciphertextSumVec.resize(inSize - 1);
	size_t ctrIndex = 0;

	for (size_t i = 0; i < lim; i = i + 2) {
		ciphertextSumVec[ctrIndex++] =
		  ctx.EvalAdd(i < inSize ? ciphertexts[i] : ciphertextSumVec[i - inSize], i + 1 < inSize ? ciphertexts[i + 1] : ciphertextSumVec[i + 1 - inSize]);
	}

	return ciphertextSumVec.back();
}

void HazeEngine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	// Facade-level pairwise reduction, structurally from CudaEngine minus its up-front
	// LoadCiphertext loop (each EvalAddInPlace loads its operands lazily via ensureCt).
	if (ciphertexts.empty()) {
		throw std::runtime_error("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	for (size_t j = 1; j < ciphertexts.size(); j = j * 2) {
		for (size_t i = 0; i < ciphertexts.size(); i = i + 2 * j) {
			if ((i + j) < ciphertexts.size()) {
				if (ciphertexts[i] != nullptr && ciphertexts[i + j] != nullptr) {
					ctx.EvalAddInPlace(ciphertexts[i], ciphertexts[i + j]);
				} else if (ciphertexts[i] == nullptr && ciphertexts[i + j] != nullptr) {
					ciphertexts[i] = ciphertexts[i + j];
				}
			}
		}
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
	for (size_t d = 0; d < digits; ++d) {
		key.aDigits.emplace_back(key.host.a_limbs[d]); // H2D: full Q∥P rows
		key.bDigits.emplace_back(key.host.b_limbs[d]);
	}
	key.uploaded = true;
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

	LimbChain accumACoeff(qpTowers, polyBytes_);
	LimbChain accumBCoeff(qpTowers, polyBytes_);
	hazeCheck(hazeINTTMrp(accumACoeff.data(), accumA.asConst().data(), qpBase.data(), qpBase.size(), nullptr), "hazeINTTMrp");
	hazeCheck(hazeINTTMrp(accumBCoeff.data(), accumB.asConst().data(), qpBase.data(), qpBase.size(), nullptr), "hazeINTTMrp");

	const hazeModDownParams ksMdParams = {
		/*.src_base =*/qpBase.data(),
		/*.src_base_len =*/qpBase.size(),
		/*.rescale_base =*/pBase_.data(),
		/*.rescale_base_len =*/pBase_.size(),
	};
	const auto qSub = qPrefix(towers);
	LimbChain mdA(towers, polyBytes_);
	LimbChain mdB(towers, polyBytes_);
	hazeCheck(hazeModDown(mdA.data(), accumACoeff.asConst().data(), &ksMdParams, nullptr), "hazeModDown");
	hazeCheck(hazeModDown(mdB.data(), accumBCoeff.asConst().data(), &ksMdParams, nullptr), "hazeModDown");

	LimbChain outA(towers, polyBytes_);
	LimbChain outB(towers, polyBytes_);
	hazeCheck(hazeNTTMrp(outA.data(), mdA.asConst().data(), qSub.data(), qSub.size(), nullptr), "hazeNTTMrp");
	hazeCheck(hazeNTTMrp(outB.data(), mdB.asConst().data(), qSub.data(), qSub.size(), nullptr), "hazeNTTMrp");

	return KsContribution{ std::move(outB), std::move(outA) };
}

HazeEngine::KsContribution HazeEngine::hybridKeyswitch(const LimbChain& src, size_t towers, KsKey& key) {
	// Port of ops.cpp hybrid_keyswitch, now a thin composition of the shared input-only prefix
	// (#19 step 1-2) + the per-key step (step 3). Relin (ct×ct) and non-hoisted rotation callers
	// route through here unchanged, so their results stay byte-identical to the old monolithic
	// implementation. The full-Q∥P key is uploaded once; trimming to (first `towers` Q rows, all
	// P rows) is pointer selection, not re-upload.
	HoistedDigits digits = hoistedKeyswitchPrefix(src, towers, key.host.a_limbs.size());
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
		// Full polynomial multiply of BOTH components against the pt chain (
		// never the slot-constant mult_scalar shortcut).
		const auto base = qPrefix(x.towers);
		LimbChain out0	= mulChain(polyBytes_, base, x.p->c0.asConst().data(), pt.chain.asConst().data());
		LimbChain out1	= mulChain(polyBytes_, base, x.p->c1.asConst().data(), pt.chain.asConst().data());

		Operand res		  = x;
		res.noiseScaleDeg = x.noiseScaleDeg + pt.noiseScaleDeg;
		res.scalingFactor = x.scalingFactor * pt.scalingFactor;
		res.slots		  = std::max(x.slots, pt.slots);
		res.p			  = finishPayload(std::move(out0), std::move(out1), res);
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
	{
		auto morph	  = std::make_shared<HazePayload>();
		morph->c0	  = passThroughChain(pt.chain, pt.towers);
		morph->towers = pt.towers;
		morph->state  = Residency::Recorded;
		ptOp.p		  = std::move(morph);
	}
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
	LimbChain out0	= mulChain(polyBytes_, base, x.p->c0.asConst().data(), ptOp.p->c0.asConst().data());
	LimbChain out1	= mulChain(polyBytes_, base, x.p->c1.asConst().data(), ptOp.p->c0.asConst().data());

	Operand res		  = x;
	res.noiseScaleDeg = x.noiseScaleDeg + ptOp.noiseScaleDeg;
	res.scalingFactor = x.scalingFactor * ptOp.scalingFactor;
	res.slots		  = std::max(x.slots, ptOp.slots);
	res.p			  = finishPayload(std::move(out0), std::move(out1), res);
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

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForMult(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}

	// Tensor product (ops.cpp order): d0 = a0·b0, d1 = a0·b1 + a1·b0, d2 = a1·b1.
	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0.asConst().data(), b.p->c0.asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0.asConst().data(), b.p->c1.asConst().data());
	LimbChain u		= mulChain(polyBytes_, base, a.p->c1.asConst().data(), b.p->c0.asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), u.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1.asConst().data(), b.p->c1.asConst().data());

	KsContribution ks = hybridKeyswitch(d2, a.towers, relinKey_);
	LimbChain out0	  = addChain(polyBytes_, base, d0.asConst().data(), ks.b.asConst().data());
	LimbChain out1	  = addChain(polyBytes_, base, d1.asConst().data(), ks.a.asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = a.noiseScaleDeg + b.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * b.scalingFactor;
	meta.slots		   = std::max(a.slots, b.slots);
	return wrapDeviceResult(ctx, ct1, finishPayload(std::move(out0), std::move(out1), meta));
}

void HazeEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalMultInPlace");
	requireComputable(*p2, "EvalMultInPlace");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalMult requires EvalMultKeyGen before LoadContext");
	}

	Operand a = asOperand(p1);
	Operand b = asOperand(p2);
	adjustForMult(a, b);
	if (a.towers != b.towers) {
		throw std::runtime_error("haze backend: internal error — operands disagree on towers after adjust");
	}

	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0.asConst().data(), b.p->c0.asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0.asConst().data(), b.p->c1.asConst().data());
	LimbChain u		= mulChain(polyBytes_, base, a.p->c1.asConst().data(), b.p->c0.asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), u.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1.asConst().data(), b.p->c1.asConst().data());

	KsContribution ks = hybridKeyswitch(d2, a.towers, relinKey_);
	LimbChain out0	  = addChain(polyBytes_, base, d0.asConst().data(), ks.b.asConst().data());
	LimbChain out1	  = addChain(polyBytes_, base, d1.asConst().data(), ks.a.asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = a.noiseScaleDeg + b.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * b.scalingFactor;
	meta.slots		   = std::max(a.slots, b.slots);
	rebindPayload(*p1, std::move(out0), std::move(out1), meta);
}

Ciphertext<DCRTPoly> HazeEngine::evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSquare");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalSquare requires EvalMultKeyGen before LoadContext");
	}

	// OpenFHE EvalSquare: AUTO modes mod-reduce a depth-2 input first, then square.
	Operand a	  = asOperand(p);
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && a.noiseScaleDeg == 2) {
		a = rescaleCore(a);
	}

	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0.asConst().data(), a.p->c0.asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0.asConst().data(), a.p->c1.asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), t.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1.asConst().data(), a.p->c1.asConst().data());

	KsContribution ks = hybridKeyswitch(d2, a.towers, relinKey_);
	LimbChain out0	  = addChain(polyBytes_, base, d0.asConst().data(), ks.b.asConst().data());
	LimbChain out1	  = addChain(polyBytes_, base, d1.asConst().data(), ks.a.asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = 2 * a.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * a.scalingFactor;
	return wrapDeviceResult(ctx, ct, finishPayload(std::move(out0), std::move(out1), meta));
}

void HazeEngine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalSquareInPlace");
	if (!haveRelinKey_) {
		throw std::runtime_error("haze backend: EvalSquare requires EvalMultKeyGen before LoadContext");
	}

	Operand a	  = asOperand(p);
	const auto st = static_cast<lbcrypto::ScalingTechnique>(scalingTech_);
	if (st != lbcrypto::FIXEDMANUAL && a.noiseScaleDeg == 2) {
		a = rescaleCore(a);
	}

	const auto base = qPrefix(a.towers);
	LimbChain d0	= mulChain(polyBytes_, base, a.p->c0.asConst().data(), a.p->c0.asConst().data());
	LimbChain t		= mulChain(polyBytes_, base, a.p->c0.asConst().data(), a.p->c1.asConst().data());
	LimbChain d1	= addChain(polyBytes_, base, t.asConst().data(), t.asConst().data());
	LimbChain d2	= mulChain(polyBytes_, base, a.p->c1.asConst().data(), a.p->c1.asConst().data());

	KsContribution ks = hybridKeyswitch(d2, a.towers, relinKey_);
	LimbChain out0	  = addChain(polyBytes_, base, d0.asConst().data(), ks.b.asConst().data());
	LimbChain out1	  = addChain(polyBytes_, base, d1.asConst().data(), ks.a.asConst().data());

	Operand meta	   = a;
	meta.noiseScaleDeg = 2 * a.noiseScaleDeg;
	meta.scalingFactor = a.scalingFactor * a.scalingFactor;
	rebindPayload(*p, std::move(out0), std::move(out1), meta);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
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

HazeEngine::Operand HazeEngine::rotateByAutoIndex(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, uint32_t autoIndex) {
	KsKey& key = autoKeyFor(ctx, autoIndex);

	// Keyswitch first, automorphism last (OpenFHE EvalAtIndex order; ops.cpp rotate).
	KsContribution ks = hybridKeyswitch(x.p->c1, x.towers, key);
	const auto base	  = qPrefix(x.towers);
	LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0.asConst().data(), ks.b.asConst().data());

	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
	hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
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
	const RotKeyResolved rk = resolveRotationKey(step, x.slots);
	if (rk.key != nullptr) {
		KsContribution ks = hybridKeyswitch(x.p->c1, x.towers, *rk.key);
		const auto base	  = qPrefix(x.towers);
		LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0.asConst().data(), ks.b.asConst().data());

		LimbChain out0(x.towers, polyBytes_);
		LimbChain out1(x.towers, polyBytes_);
		hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
		hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

		Operand res = x; // metadata unchanged
		res.p		= finishPayload(std::move(out0), std::move(out1), res);
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
	LimbChain ksC0	  = addChain(polyBytes_, base, x.p->c0.asConst().data(), ks.b.asConst().data());

	LimbChain out0(x.towers, polyBytes_);
	LimbChain out1(x.towers, polyBytes_);
	hazeCheck(hazeAutomorphMrp(out0.data(), ksC0.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");
	hazeCheck(hazeAutomorphMrp(out1.data(), ks.a.asConst().data(), rk.autoIndex, base.data(), base.size(), nullptr), "hazeAutomorphMrp");

	Operand res = x; // metadata unchanged
	res.p		= finishPayload(std::move(out0), std::move(out1), res);
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
	rebindPayload(*p, std::move(res.p->c0), std::move(res.p->c1), res);
}

std::shared_ptr<void> HazeEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	// #19: record the shared, key-independent ModUp/per-digit-NTT decomposition of c1 ONCE
	// (CUDA c1.modupInto across the baby steps, Ciphertext.cpp:965). The returned handle is reused
	// by every evalFastRotation index, so the expensive INTT+ModUp+NTT runs a single time instead
	// of once per rotation. numPartQ_ is the context's hybrid digit count (all keys share it).
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalFastRotationPrecompute");
	if (numPartQ_ == 0) {
		throw std::runtime_error("haze backend: EvalFastRotationPrecompute requires a loaded context with keyswitch params");
	}
	return std::make_shared<HoistedDigits>(hoistedKeyswitchPrefix(p->c1, p->towers, numPartQ_));
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

namespace {
// Radix-4 reduction cascade — CUDA Accumulate / AccumulateCascadeImpl (AccumulateBroadcast.cu:9-113),
// expressed over facade rotate+add (the CUDA rotate_hoisted/extend/modDown are the hoisting form of the
// same rotate-then-add; hoisting itself is Task 06 #19). bStep=4 ⇒ logbStep=2: outer s <<= 2 from
// startFactor, inner adds rot(snapshot, stride*s*k) for k in {1,2,3} while stride*s*k < stride*size.
// Each step rotates the PRE-STEP snapshot (CUDA rotates ctxt for all indexes before any of the step's
// adds), unlike the old radix-2 byte-copy of the OpenFHE doubling loop which rotated the running sum.
void hazeAccumulateCascade(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int size, int stride, int startFactor) {
	if (startFactor <= 0 || size <= 0)
		return;
	for (int s = startFactor; s < size; s <<= 2) {
		Ciphertext<DCRTPoly> snapshot = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
		for (int idx = stride * s; idx < stride * size && idx < 4 * stride * s; idx += stride * s) {
			ctx.EvalAddInPlace(ct, ctx.EvalRotate(snapshot, idx));
		}
	}
}
} // namespace

Ciphertext<DCRTPoly> HazeEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	const size_t ctSlots		= ensureCt(ctx, ct)->slots;
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	hazeAccumulateCascade(ctx, result, slots, stride, /*startFactor=*/1);
	// Slots-shrink (CUDA Accumulate AccumulateBroadcast.cu:107-108): when the accumulate spans the
	// whole packing, the result is a broadcast over `stride` slots. OpenFHE's loop leaves slots intact.
	if (static_cast<size_t>(slots) * static_cast<size_t>(stride) == ctSlots)
		result->SetSlots(static_cast<size_t>(stride));
	return result;
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	const size_t ctSlots = ensureCt(ctx, ct)->slots;
	hazeAccumulateCascade(ctx, ct, slots, stride, /*startFactor=*/1);
	if (static_cast<size_t>(slots) * static_cast<size_t>(stride) == ctSlots)
		ct->SetSlots(static_cast<size_t>(stride));
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	const size_t ctSlots = ensureCt(ctx, ct)->slots;
	hazeAccumulateCascade(ctx, ct, slots, stride, /*startFactor=*/start);
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

// ---------------------------------------------------------------------------
// Chebyshev Paterson-Stockmeyer — port of the CUDA spec src/CKKS/ApproxModEval.cu
// (evalChebyshevSeries@370, innerEvalChebyshevPS@138). FIDESlib/CUDA is the spec;
// OpenFHE is reference-only. Divergences from OpenFHE are commented inline (#12-#15).
// ---------------------------------------------------------------------------

HazeEngine::Operand HazeEngine::evalLinearWSumMutableCore(size_t targetTowers,
  const std::vector<Operand>& ops,
  const std::vector<double>& weights) {
	// Port of CUDA Ciphertext::evalLinearWSumMutable (Ciphertext.cpp:1168): out = sum_i
	// weights[i]*ops[i] produced directly at `targetTowers` (NSD=2). CUDA grows a fresh ct to
	// the target level then fills it via per-limb ElemForEvalMult + evalLinearWSum; haze cannot
	// grow (setCiphertextLevel only drops), so we build the result at targetTowers here.
	// Each ops[i] is at NoiseLevel 1 and ops[i].towers >= targetTowers; reading the leading
	// targetTowers limbs is a valid Q-prefix view (CUDA ElemForEvalMult uses level_in=towersIn).
	const size_t n	= ops.size();
	const auto base = qPrefix(targetTowers);
	if (n == 0) {
		throw std::runtime_error("haze backend: evalLinearWSumMutableCore needs at least one operand");
	}

	// CUDA encodes each weight with level_in = ops[i].towers (the bootstrap-prescale fast path:
	// scFactorOut * ModReduceFactor[targetTowers-1] / scFactorIn), exact for FIXED modes.
	for (size_t i = 0; i < n; ++i) {
		// CUDA asserts target level <= ctxs[i] level, i.e. each op must have >= targetTowers limbs
		// (reading the leading targetTowers limbs as a Q-prefix view).
		if (ops[i].towers < targetTowers) {
			throw std::runtime_error("haze backend: evalLinearWSumMutableCore operand has fewer towers than target");
		}
	}
	auto encode = [&](size_t i) {
		return hazebk::elemForEvalMult(scalarParams(), targetTowers, weights[i], ops[i].towers);
	};

	LimbChain out0(targetTowers, polyBytes_);
	{
		const auto f0 = encode(0);
		hazeCheck(hazeMulScalarMrp(out0.data(), ops[0].p->c0.asConst().data(), f0.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	}
	LimbChain out1;
	const bool haveC1 = !ops[0].p->c1.empty();
	if (haveC1) {
		out1		  = LimbChain(targetTowers, polyBytes_);
		const auto f0 = encode(0);
		hazeCheck(hazeMulScalarMrp(out1.data(), ops[0].p->c1.asConst().data(), f0.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
	}
	for (size_t i = 1; i < n; ++i) {
		const auto fi = encode(i);
		LimbChain term0(targetTowers, polyBytes_);
		hazeCheck(hazeMulScalarMrp(term0.data(), ops[i].p->c0.asConst().data(), fi.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
		LimbChain acc0(targetTowers, polyBytes_);
		hazeCheck(hazeAddMrp(acc0.data(), out0.asConst().data(), term0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
		out0 = std::move(acc0);
		if (haveC1) {
			LimbChain term1(targetTowers, polyBytes_);
			hazeCheck(hazeMulScalarMrp(term1.data(), ops[i].p->c1.asConst().data(), fi.data(), base.data(), base.size(), nullptr), "hazeMulScalarMrp");
			LimbChain acc1(targetTowers, polyBytes_);
			hazeCheck(hazeAddMrp(acc1.data(), out1.asConst().data(), term1.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
			out1 = std::move(acc1);
		}
	}

	Operand res;
	res.towers		  = targetTowers;
	res.noiseScaleDeg = 2; // CUDA NoiseLevel=2 after the weighted sum
	const size_t level = qBase_.size() - targetTowers;
	res.scalingFactor = sfReal_[level] * sfReal_[level]; // CUDA NoiseFactor = ScalingFactorReal[level]^2
	res.slots		  = ops[0].slots;
	for (size_t i = 1; i < n; ++i)
		res.slots = std::max(res.slots, ops[i].slots);
	res.p = finishPayload(std::move(out0), std::move(out1), res);
	return res;
}

Ciphertext<DCRTPoly> HazeEngine::evalLinearWSumMutableFacade(CryptoContextImpl<DCRTPoly>& ctx,
  size_t targetTowers,
  const std::vector<Ciphertext<DCRTPoly>>& ctxs,
  const std::vector<double>& weights) {
	std::vector<Operand> ops;
	ops.reserve(ctxs.size());
	for (const auto& c : ctxs) {
		auto pc = ensureCt(ctx, c);
		ops.push_back(asOperand(pc));
	}
	Operand res = evalLinearWSumMutableCore(targetTowers, ops, weights);
	return wrapDeviceResult(ctx, ctxs[0], std::move(res.p));
}

Ciphertext<DCRTPoly> HazeEngine::hazeInnerEvalChebyshevPS(CryptoContextImpl<DCRTPoly>& ctx,
  const std::vector<double>& coefficients,
  uint32_t k, uint32_t m,
  const std::vector<Ciphertext<DCRTPoly>>& T,
  const std::vector<Ciphertext<DCRTPoly>>& T2,
  int level_offset, int max_m) {
	// CUDA getLevel of a facade ct = towers-1 = (|Q| - GetLevel()) - 1 (inverse of OpenFHE GetLevel).
	auto cudaLevel = [&](const Ciphertext<DCRTPoly>& c) -> int {
		return static_cast<int>(qBase_.size()) - static_cast<int>(c->GetLevel()) - 1;
	};
	auto noiseLevel = [&](const Ciphertext<DCRTPoly>& c) -> size_t { return c->GetNoiseScaleDeg(); };

	// --- Coefficient long division (identical to OpenFHE / CUDA, left as-is) ---
	uint32_t k2m2k = k * (1u << (m - 1)) - k;
	auto f2		   = coefficients;
	f2.resize(2 * k2m2k + k + 1, 0.0);
	if (f2.size() > coefficients.size())
		f2.back() = 1.0;
	std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
	Tkm.back()	= 1.0;
	auto divqr	= lbcrypto::LongDivisionChebyshev(f2, Tkm);
	std::vector<double> r2 = divqr->r;
	if (int32_t(k2m2k - lbcrypto::Degree(divqr->r)) <= 0) {
		r2[int32_t(k2m2k)] -= 1.0;
		r2.resize(lbcrypto::Degree(r2) + 1);
	} else {
		r2.resize(int32_t(k2m2k + 1), 0.0);
		r2.back() = -1.0;
	}
	auto divcs			   = lbcrypto::LongDivisionChebyshev(r2, divqr->q);
	std::vector<double> s2 = divcs->r;
	s2.resize(int32_t(k2m2k + 1), 0.0);
	s2.back() = 1.0;

	// --- Evaluate c at u (cu) ---
	Ciphertext<DCRTPoly> cu;
	uint32_t dc	  = lbcrypto::Degree(divcs->q);
	bool flag_c	  = false;
	if (dc >= 1) {
		if (dc == 1) {
			if (divcs->q[1] != 1) {
				// CUDA cu.multScalar(*T[0], divcs->q[1], true): under FIXEDAUTO the rescale arg is
				// inert (multScalar passes rescale&&FIXEDMANUAL), so this is just multScalar -> NSD2.
				cu = ctx.EvalMult(T[0], divcs->q[1]);
			} else {
				cu = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[0]);
			}
		} else {
			std::vector<double> weights(dc);
			for (uint32_t i = 0; i < dc; ++i)
				weights[i] = divcs->q[i + 1];
			// CUDA: cu.dropToLevel(target); cu.growToLevel(target); cu.evalLinearWSumMutable(dc, T, weights).
			int target = cudaLevel(T2[m - 1]) + (noiseLevel(T2[m - 1]) == 1 ? 1 : 0) - level_offset;
			std::vector<Ciphertext<DCRTPoly>> ctxs(T.begin(), T.begin() + dc);
			cu = evalLinearWSumMutableFacade(ctx, static_cast<size_t>(target) + 1, ctxs, weights);
		}
		ctx.EvalAddInPlace(cu, divcs->q.front() / 2.0);
		// FIXEDMANUAL rescale here is skipped on the FIXEDAUTO spec path.
		flag_c = true;
	}

	// --- Evaluate q at u (qu) ---
	Ciphertext<DCRTPoly> qu;
	if (lbcrypto::Degree(divqr->q) > k) {
		assert(m > 2);
		// q recurses at level_offset (CUDA ApproxModEval.cu:230).
		qu = hazeInnerEvalChebyshevPS(ctx, divqr->q, k, m - 1, T, T2, level_offset, max_m);
		if (noiseLevel(qu) == 2)
			ctx.RescaleInPlace(qu);
	} else {
		auto qcopy = divqr->q;
		qcopy.resize(k);
		if (lbcrypto::Degree(qcopy) > 0) {
			std::vector<double> weights;
			std::vector<Ciphertext<DCRTPoly>> ctxs;
			for (uint32_t i = 0; i < divqr->q.size() - 1; ++i) {
				if (divqr->q[i + 1] != 0) {
					weights.push_back(divqr->q[i + 1]);
					ctxs.push_back(T[i]);
				}
			}
			int target = cudaLevel(T2[m - 1]) + (noiseLevel(T2[m - 1]) == 1 ? 1 : 0) - level_offset;
			qu		   = evalLinearWSumMutableFacade(ctx, static_cast<size_t>(target) + 1, ctxs, weights);
			ctx.EvalAddInPlace(qu, divqr->q.front() / 2.0);
			// CUDA: if (T[k-1]->NoiseLevel==1) qu.rescale(); ... if (T[k-1]->NoiseLevel==2) qu.rescale();
			// Either way qu is rescaled exactly once (NSD2 -> NSD1) since T[k-1] NL is 1 or 2.
			ctx.RescaleInPlace(qu);
		} else {
			qu = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[k - 1]);
			if (divqr->q.back() > 0 && divqr->q.back() - std::round(divqr->q.back()) == 0.0) {
				// CUDA multIntScalar (integer scalar mult); for the degree path this is the leading
				// power-of-2 coefficient. EvalMult by a double is metadata-equivalent here.
				ctx.EvalMultInPlace(qu, divqr->q.back());
			} else {
				__builtin_unreachable();
			}
			ctx.EvalAddInPlace(qu, divqr->q.front() / 2.0);
			if (noiseLevel(qu) == 2)
				ctx.RescaleInPlace(qu);
		}
	}

	// --- Evaluate s2 at u (su) ---
	Ciphertext<DCRTPoly> su;
	if (lbcrypto::Degree(s2) > k) {
		assert(m > 2);
		// s recurses at level_offset + 1 (CUDA ApproxModEval.cu:308).
		su = hazeInnerEvalChebyshevPS(ctx, s2, k, m - 1, T, T2, level_offset + 1, max_m);
	} else {
		auto scopy = s2;
		scopy.resize(k);
		if (lbcrypto::Degree(scopy) > 0) {
			std::vector<Ciphertext<DCRTPoly>> ctxs;
			std::vector<double> weights;
			for (uint32_t i = 0; i < s2.size() - 1; ++i) {
				if (s2[i + 1] != 0) {
					ctxs.emplace_back(T[i]);
					weights.push_back(s2[i + 1]);
				}
			}
			int target = cudaLevel(T2[m - 1]) + (noiseLevel(T2[m - 1]) == 1 ? 1 : 0) - 1 - level_offset;
			// CUDA quirk: evalLinearWSumMutable(ctxs.size(), T, weights) passes the FULL T vector with
			// n=ctxs.size() (ApproxModEval.cu:328), not the filtered ctxs. Match CUDA verbatim.
			std::vector<Ciphertext<DCRTPoly>> ctxsFull(T.begin(), T.begin() + ctxs.size());
			su = evalLinearWSumMutableFacade(ctx, static_cast<size_t>(target) + 1, ctxsFull, weights);
			ctx.EvalAddInPlace(su, s2.front() / 2.0);
			// CUDA leaves su un-rescaled here (the rescale lines are commented out); s2 is monic.
			assert(s2.back() == 1.0);
		} else {
			su = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[k - 1]);
			ctx.EvalAddInPlace(su, s2.front() / 2.0);
		}
	}

	// --- Combine: cu = (T2[m-1] + cu) * qu + su ---
	if (flag_c) {
		// CUDA: for m>3 the required cu levels are not strictly decreasing, so cache-align cu to
		// T2[m-1] only when max_m - m <= 1 (ApproxModEval.cu:351). adjustForAddOrSub equalizes
		// level+depth in place; in the facade EvalAdd does the adjust, so we let EvalAdd handle it
		// when caching applies and otherwise still EvalAdd (the metadata outcome is the same).
		ctx.EvalAddInPlace(cu, T2[m - 1]);
	} else {
		cu = ctx.EvalAdd(T2[m - 1], divcs->q.front() / 2.0);
	}
	// FIXEDMANUAL out NSD==2 rescale skipped on the FIXEDAUTO spec path.
	cu = ctx.EvalMult(cu, qu); // CUDA cu.mult(qu, false): relin, no rescale
	ctx.EvalAddInPlace(cu, su);
	return cu;
}

Ciphertext<DCRTPoly> HazeEngine::hazeEvalChebyshevSeriesImpl(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coefficients,
  double a, double b) {
	double lower_bound = a;
	double upper_bound = b;

	// Working copy of the input; the affine map + T[0] build mutate it (CUDA mutates ctxt in place).
	Ciphertext<DCRTPoly> ctxt = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

	// #13 affine map to [-1,1] — CUDA pre-T centering (ApproxModEval.cu:377-387), NOT OpenFHE's
	// single y=-1+2(x-a)/(b-a) (which always costs a multScalar+ModReduce). CUDA has a span==2
	// no-multiply fast path and otherwise centers-then-scales.
	if (std::abs(lower_bound + 1.0) > 1e-9 || std::abs(upper_bound - 1.0) > 1e-9) {
		if (std::abs(upper_bound - lower_bound - 2.0) < 1e-8) {
			// span == 2 fast path: shift only, no multiply (OpenFHE would still multiply by 1).
			ctx.EvalAddInPlace(ctxt, -lower_bound + 1.0);
		} else {
			if (std::abs(lower_bound + upper_bound) > 1e-8)
				// Center on 0: subtract the interval midpoint (a+b)/2 == a + (b-a)/2.
				// CUDA writes addScalar(-a + (b-a)/2) (ApproxModEval.cu:382), distributing the minus
				// onto only `a` — a sign BUG that gives shift (b-3a)/2 instead of -(a+b)/2. CUDA's own
				// Chebyshev tests only use [-1,1] (the span==2 fast path above), so the bug is never
				// exercised there; the haze [0,1] test does hit it, so we use the OpenFHE-correct
				// shift -(a + (b-a)/2) here (the one deliberate deviation from a verbatim CUDA port).
				ctx.EvalAddInPlace(ctxt, -(lower_bound + (upper_bound - lower_bound) / 2.0));
			// FIXEDMANUAL pre-rescale (ApproxModEval.cu:383-384) is skipped on the FIXEDAUTO spec path.
			ctx.EvalMultInPlace(ctxt, 2.0 / (upper_bound - lower_bound));
		}
	}

	// #12 always Paterson-Stockmeyer — CUDA has NO degree<5 linear dispatch (ApproxModEval.cu:370-404).
	uint32_t n	= lbcrypto::Degree(coefficients);
	auto f2coef = coefficients;
	f2coef.resize(n + 1);

	auto degs  = lbcrypto::ComputeDegreesPS(n);
	uint32_t k = degs[0];
	uint32_t m = degs[1];

	std::vector<Ciphertext<DCRTPoly>> T(k);
	// CUDA copies the (already-mapped) ctxt into T[0] (ApproxModEval.cu:455), then rescales if NSD2.
	T[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*ctxt);
	if (T[0]->GetNoiseScaleDeg() == 2)
		ctx.RescaleInPlace(T[0]);

	for (uint32_t i = 2; i <= k; ++i) {
		if (i % 2 == 1) {
			// T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
			T[i - 1] = ctx.EvalMult(T[i / 2], T[i / 2 - 1]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]);
			// CUDA: ctxt.adjustForAddOrSub(*T[i-1]); if(ctxt.NL==1) T[i-1].rescale(); T[i-1].sub(ctxt).
			// The facade EvalSub performs the adjust+depth-fix internally (Task 02 parity).
			ctx.EvalSubInPlace(T[i - 1], ctxt);
		} else {
			// T_{2i}(y) = 2*T_i(y)^2 - 1
			T[i - 1] = ctx.EvalSquare(T[i / 2 - 1]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]);
			ctx.EvalAddInPlace(T[i - 1], -1.0);
		}
	}

	for (size_t i = 1; i <= k; ++i) {
		if (T[i - 1]->GetNoiseScaleDeg() == 2)
			ctx.RescaleInPlace(T[i - 1]);
	}
	// FIXEDMANUAL drops all T[i] to T[k-1]'s level here; under FIXEDAUTO the GPU spec leaves the
	// T-equalization to the lazy adjust inside the later EvalMult/EvalAdd (ApproxModEval.cu:563-583).

	// Build T2[0..m-1]: T2[i] = T_{k*2^i}(y); T2[0] is a placeholder = T[k-1].
	std::vector<Ciphertext<DCRTPoly>> T2(m);
	T2[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*T.back());
	for (uint32_t i = 1; i < m; ++i) {
		// FIXEDMANUAL pre/post rescales (ApproxModEval.cu:605-613) skipped on the FIXEDAUTO path.
		T2[i] = ctx.EvalSquare(T2[i - 1]);
		ctx.EvalAddInPlace(T2[i], T2[i]);
		ctx.EvalAddInPlace(T2[i], -1.0);
	}

	// T2km1 = T_{k(2m-1)}(y).
	Ciphertext<DCRTPoly> T2km1 = std::make_shared<CiphertextImpl<DCRTPoly>>(*T2[0]);
	// FIXEDMANUAL drops T2km1 to T2[1]'s level (ApproxModEval.cu:646-648); skipped on FIXEDAUTO.
	for (uint32_t i = 1; i < m; ++i) {
		// T_{k(2m-1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
		T2km1 = ctx.EvalMult(T2km1, T2[i]);
		ctx.EvalAddInPlace(T2km1, T2km1);
		// CUDA: T2[0].adjustForAddOrSub(T2km1); if(T2[0].NL==1) T2km1.rescale(); T2km1.sub(*T2[0]);
		// if(T2[0].NL==2 && i<m-1) T2km1.rescale(). The facade EvalSub does the adjust; the trailing
		// rescale matters only for m>3 (deep path), where we additionally rescale T2km1 below.
		ctx.EvalSubInPlace(T2km1, T2[0]);
		if (T2[0]->GetNoiseScaleDeg() == 2 && i < m - 1)
			ctx.RescaleInPlace(T2km1);
	}

	// Paterson-Stockmeyer recursion; top-level call uses level_offset=0, max_m=m.
	Ciphertext<DCRTPoly> result = hazeInnerEvalChebyshevPS(ctx, f2coef, k, m, T, T2, 0, static_cast<int>(m));

	// result - T2km1 (ApproxModEval.cu:700).
	return ctx.EvalSub(result, T2km1);
}

// ---- EvalChebyshevSeries ----

Ciphertext<DCRTPoly> HazeEngine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coeffs,
  double a, double b) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalChebyshevSeries");
	return hazeEvalChebyshevSeriesImpl(ctx, ct, coeffs, a, b);
}

void HazeEngine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coeffs,
  double a, double b) {
	auto p = ensureCt(ctx, ct);
	requireComputable(*p, "EvalChebyshevSeriesInPlace");
	ct = hazeEvalChebyshevSeriesImpl(ctx, ct, coeffs, a, b);
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
