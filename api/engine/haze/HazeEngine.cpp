#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeEngine.hpp"

#include "engine/haze/HazeScalarEncode.hpp"

#include "CryptoContext.hpp"

#include <haze/haze.h>
#include <haze/replay_bridge.h>

#include <openfhe.h>

#include <any>
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
		// (pure truncation — a view change here, no IR).
		const size_t towers = std::min(a.towers, b.towers);
		a.towers			= towers;
		b.towers			= towers;
		return;
	}
	// Port of OpenFHE LeveledSHECKKSRNS::AdjustLevelsAndDepthInPlace (compositeDegree == 1;
	// one shared implementation for FIXEDAUTO and the FLEXIBLE modes), in towers-space:
	// OpenFHE level = |Q| − towers; LevelReduceInternal is a view truncation. The "fresher"
	// operand (more towers / lower level) is adjusted toward the other.
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
				const double q1	  = modReduceFactor_[x.towers - 1];
				x				  = multScalarCore(x, scf2 / scf1 * q1 / scf);
				// CUDA does multScalar; dropToLevel(c2lvl+1); THEN rescale (Ciphertext.cpp:1393-1398):
				// trim the extra top towers FIRST, then rescale the now-correct top tower. OpenFHE / the
				// old Haze order rescaled first and folded the wrong top tower when operands were >1 level
				// apart. (q1 still uses the Plaintext-rule modReduceFactor_[x.towers−1]; the Ciphertext-
				// rule index is #5, a Stage 2 change.)
				if (c1lvl + 1 < c2lvl) {
					x.towers -= c2lvl - c1lvl - 1;
				}
				x				  = rescaleCore(x);
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

HazeEngine::KsContribution HazeEngine::hybridKeyswitch(const LimbChain& src, size_t towers, KsKey& key) {
	// Port of ops.cpp hybrid_keyswitch. The full-Q∥P key is uploaded once; trimming to
	// (first `towers` Q rows, all P rows) is pointer selection, not re-upload.
	ensureKeyUploaded(key);
	const size_t numPartQ = key.host.a_limbs.size();
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
			dig[t] = digitsEval[(d * qpTowers) + t];
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

HazeEngine::Operand HazeEngine::rotateCore(CryptoContextImpl<DCRTPoly>& ctx, const Operand& x, int32_t step) {
	const auto it = rotKeys_.find(step);
	if (it != rotKeys_.end()) {
		// Pre-extracted EvalRotateKeyGen key (keyed by slot step).
		RotKey& rk		  = it->second;
		KsContribution ks = hybridKeyswitch(x.p->c1, x.towers, rk.key);
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
	// Fallback (bootstrap rotations and any other keyed step): resolve the automorphism
	// index on the host and extract the key lazily from the tag's automorphism key map.
	auto& context			 = hostContext(ctx);
	const uint32_t autoIndex = context->FindAutomorphismIndex(static_cast<uint32_t>(step));
	return rotateByAutoIndex(ctx, x, autoIndex);
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

std::shared_ptr<void> HazeEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	// No hoisting in v1: fast rotations are plain rotations, so there is no
	// precompute handle. Correctness-equivalent; hoisting is a tracked future optimization.
	return nullptr;
}

Ciphertext<DCRTPoly>
HazeEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t /*m*/, const std::shared_ptr<void>& /*precomp*/) {
	return evalRotate(ctx, ct, index);
}

Ciphertext<DCRTPoly>
HazeEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& /*digits*/, bool /*addFirst*/) {
	// Plain rotation stand-in: the Ext variant's extended-basis intermediate is a hoisting
	// detail; the api test only exercises the dispatch.
	return evalRotate(ctx, ct, index);
}

std::vector<Ciphertext<DCRTPoly>> HazeEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const uint32_t /*m*/,
  const std::shared_ptr<void>& /*precomp*/) {
	std::vector<Ciphertext<DCRTPoly>> results;
	results.reserve(indices.size());
	for (const int32_t index : indices) {
		if (index == 0) {
			results.push_back(std::make_shared<CiphertextImpl<DCRTPoly>>(*ct));
		} else {
			results.push_back(evalRotate(ctx, ct, index));
		}
	}
	return results;
}

std::vector<Ciphertext<DCRTPoly>> HazeEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const std::shared_ptr<void>& /*digits*/,
  bool /*addFirst*/) {
	std::vector<Ciphertext<DCRTPoly>> results;
	results.reserve(indices.size());
	for (const int32_t index : indices) {
		if (index == 0) {
			results.push_back(std::make_shared<CiphertextImpl<DCRTPoly>>(*ct));
		} else {
			results.push_back(evalRotate(ctx, ct, index));
		}
	}
	return results;
}

Ciphertext<DCRTPoly> HazeEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	// Port of the CPU rotate+add doubling loop (OpenFheEngine.cpp:357-370) over engine
	// primitives, via the facade like evalAddMany.
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	for (int i = 0; i < std::log2(slots); i++) {
		const int rotIdx = stride * (1 << i);
		auto tmp		 = ctx.EvalRotate(result, rotIdx);
		ctx.EvalAddInPlace(result, tmp);
	}
	return result;
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	for (int i = 0; i < std::log2(slots); i++) {
		const int rotIdx = stride * (1 << i);
		auto tmp		 = ctx.EvalRotate(ct, rotIdx);
		ctx.EvalAddInPlace(ct, tmp);
	}
}

void HazeEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	for (int s = start; s < slots; s <<= 1) {
		const int rotIdx = stride * s;
		auto tmp		 = ctx.EvalRotate(ct, rotIdx);
		ctx.EvalAddInPlace(ct, tmp);
	}
}

// ---- Bootstrap setup hooks (staged compute lands in a later change) ----

BootstrapSetupPolicy HazeEngine::bootstrapSetupPolicy(bool /*precompute*/, bool /*btsfirstboot*/, int32_t modEvalLevels) const {
	// Verbatim CPU policy (OpenFheEngine.cpp:400-408): haze's host setup must match the CPU
	// oracle, including the pre-existing modall-lands-in-BTSlotsEncoding quirk flagged there.
	return BootstrapSetupPolicy{ /*precompute=*/true, /*btSlotsEncoding=*/modEvalLevels != 0, /*modEvalLevels=*/-1 };
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
// Chebyshev helpers — all ported from
//   deps/openfhe-src/src/pke/lib/scheme/ckksrns/ckksrns-advancedshe.cpp
// and
//   deps/openfhe-src/src/pke/include/scheme/ckksrns/ckksrns-utils.h
// expressed over the fideslib facade (CryptoContextImpl<DCRTPoly>& ctx).
// ---------------------------------------------------------------------------

// Under FIXEDAUTO (FIXEDMANUAL + AUTO modes):
//   ctx.Rescale = OpenFHE ModReduceInPlace = no-op (clone) for AUTO.
//   LevelReduce by n levels = SetLevel(ct, ct->GetLevel() + n): truncation only.
//
// compositeDegree = 1 (the default for FIXEDAUTO with standard params; see
// deps/openfhe-src/src/pke/include/scheme/gen-cryptocontext-params-defaults.h:84).

// Rescale one level (OpenFHE ModReduceInPlace). Under FIXEDAUTO this is a
// no-op/clone in the facade — any depth-equalization needed is handled inside
// EvalMult/EvalAdd adjust. We call it at exactly the same places OpenFHE does.
static void hazeModReduceInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.RescaleInPlace(ct);
}

// Drop `n` levels (OpenFHE LevelReduceInPlace with diff/compositeDegree = n).
// On the haze engine this calls setCiphertextLevel which truncates the limb chains
// (no IR, no flush) — exactly as OpenFHE's DropLastElements.
static void hazeLevelReduceInPlace(Ciphertext<DCRTPoly>& ct, size_t n) {
	if (n == 0)
		return;
	ct->SetLevel(ct->GetLevel() + n);
}

// EvalPartialLinearWSum(T, coefficients, limit): sum_{i=1}^{limit} T[i-1] * coefficients[i],
// rescaled by one level. Ported from ckksrns-advancedshe.cpp:142-193. Uses T[0..limit-1]
// with coefficient indices [1..limit] (i.e. index 0 is the constant term, handled separately).
static Ciphertext<DCRTPoly> hazeEvalPartialLinearWSum(CryptoContextImpl<DCRTPoly>& ctx,
  const std::vector<Ciphertext<DCRTPoly>>& T,
  const std::vector<double>& coefficients,
  uint32_t limit) {
	if (limit == 0)
		limit = static_cast<uint32_t>(T.size());

	// Under FIXEDAUTO, align all Ts to the same level (the max level + NSD).
	// We do a simple clone-and-level-reduce approach matching what OpenFHE does:
	// find the deepest (highest-level) T[i] up to limit, then drop others to match.
	// For our use case (degree 5, all Ts were equalized to T[k-1] before calling),
	// they are already equal, so this is mostly a no-op clone step.
	std::vector<Ciphertext<DCRTPoly>> cts(limit);
	size_t maxLevel = 0;
	uint32_t maxIdx = 0;
	for (uint32_t i = 0; i < limit; ++i) {
		cts[i] = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[i]);
		size_t lvl = cts[i]->GetLevel();
		if (lvl > maxLevel || (lvl == maxLevel && cts[i]->GetNoiseScaleDeg() == 2)) {
			maxLevel = lvl;
			maxIdx	 = i;
		}
	}
	// Drop all to the same level as the deepest.
	for (uint32_t i = 0; i < limit; ++i) {
		if (i != maxIdx) {
			size_t diff = cts[maxIdx]->GetLevel() - cts[i]->GetLevel();
			if (diff > 0)
				hazeLevelReduceInPlace(cts[i], diff);
		}
	}
	// Rescale depth-2 entries if the max is also depth 2.
	if (cts[maxIdx]->GetNoiseScaleDeg() == 2) {
		for (uint32_t i = 0; i < limit; ++i)
			hazeModReduceInPlace(ctx, cts[i]);
	}

	// EvalMult by coefficients[i+1] (index i+1 corresponds to T[i]).
	ctx.EvalMultInPlace(cts[0], coefficients[1]);
	for (uint32_t i = 1; i < limit; ++i) {
		ctx.EvalMultInPlace(cts[i], coefficients[i + 1]);
		ctx.EvalAddInPlace(cts[0], cts[i]);
	}
	ctx.RescaleInPlace(cts[0]);
	return cts[0];
}

// Forward declaration.
static Ciphertext<DCRTPoly> hazeInnerEvalChebyshevPS(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& x,
  const std::vector<double>& coefficients,
  uint32_t k, uint32_t m,
  const std::vector<Ciphertext<DCRTPoly>>& T,
  const std::vector<Ciphertext<DCRTPoly>>& T2);

// Port of InnerEvalChebyshevPS from ckksrns-advancedshe.cpp:650-760.
// Recurses on the Paterson-Stockmeyer tree.
static Ciphertext<DCRTPoly> hazeInnerEvalChebyshevPS(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& x,
  const std::vector<double>& coefficients,
  uint32_t k, uint32_t m,
  const std::vector<Ciphertext<DCRTPoly>>& T,
  const std::vector<Ciphertext<DCRTPoly>>& T2) {
	// k2m2k = k*2^{m-1} - k
	uint32_t k2m2k = k * (1u << (m - 1)) - k;

	// Divide coefficients by T^{k*2^{m-1}}
	std::vector<double> Tkm(k2m2k + k + 1, 0.0);
	Tkm.back() = 1.0;
	auto divqr = lbcrypto::LongDivisionChebyshev(coefficients, Tkm);

	// Subtract T^{k(2^{m-1} - 1)} from r
	auto& r2 = divqr->r;
	uint32_t n = lbcrypto::Degree(r2);
	if (static_cast<int32_t>(k2m2k - n) <= 0) {
		r2.resize(n + 1);
		r2[k2m2k] -= 1.0;
	} else {
		r2.resize(k2m2k + 1);
		r2.back() = -1.0;
	}

	auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

	Ciphertext<DCRTPoly> cu, qu, su;

	{
		// Evaluate q at u.
		if (lbcrypto::Degree(divqr->q) > k) {
			qu = hazeInnerEvalChebyshevPS(ctx, x, divqr->q, k, m - 1, T, T2);
		} else {
			// dq = k from construction; highest order is a power of 2 up to 2^{m-1}.
			qu = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[k - 1]);
			uint32_t limit = static_cast<uint32_t>(std::log2(lbcrypto::ToReal(divqr->q.back())));
			for (uint32_t i = 0; i < limit; ++i)
				ctx.EvalAddInPlace(qu, qu);
			// adds the free term (constant) /2
			ctx.EvalAddInPlace(qu, divqr->q.front() / 2.0);
			// Add partial linear sum for remaining coefficients
			divqr->q.resize(k);
			if (uint32_t nn = lbcrypto::Degree(divqr->q); nn > 0)
				ctx.EvalAddInPlace(qu, hazeEvalPartialLinearWSum(ctx, T, divqr->q, nn));
		}
	}

	{
		// Evaluate s2 at u: first add T^{k(2^{m-1}-1)} back to s.
		auto& s2 = divcs->r;
		s2.resize(k2m2k + 1);
		s2.back() = 1.0;

		if (lbcrypto::Degree(s2) > k) {
			su = hazeInnerEvalChebyshevPS(ctx, x, s2, k, m - 1, T, T2);
		} else {
			// Highest order is 1 (s2 is monic).
			su = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[k - 1]);
			s2.resize(k);
			if (uint32_t nn = lbcrypto::Degree(s2); nn > 0)
				ctx.EvalAddInPlace(su, hazeEvalPartialLinearWSum(ctx, T, s2, nn));
			// adds the free term /2
			ctx.EvalAddInPlace(su, s2.front() / 2.0);
			// reduce su to match T2[m-1] + 1 levels
			hazeLevelReduceInPlace(su, (su->GetLevel() > T2[m - 1]->GetLevel()) ? su->GetLevel() - T2[m - 1]->GetLevel() : 0);
		}
	}

	// Evaluate c (the quotient polynomial from divcs).
	{
		uint32_t n_c = lbcrypto::Degree(divcs->q);
		if (n_c >= 1) {
			if (n_c == 1) {
				if (lbcrypto::IsNotEqualOne(divcs->q[1])) {
					cu = ctx.EvalMult(T[0], divcs->q[1]);
					hazeModReduceInPlace(ctx, cu);
				} else {
					cu = std::make_shared<CiphertextImpl<DCRTPoly>>(*T[0]);
				}
			} else {
				cu = hazeEvalPartialLinearWSum(ctx, T, divcs->q, n_c);
			}
			// adds the free term /2
			ctx.EvalAddInPlace(cu, divcs->q.front() / 2.0);
			// level-reduce cu to T2[m-1]
			size_t cuLvl	 = cu->GetLevel();
			size_t t2mLvl	 = T2[m - 1]->GetLevel();
			if (cuLvl < t2mLvl)
				hazeLevelReduceInPlace(cu, t2mLvl - cuLvl);
		}
	}

	if (cu) {
		cu = ctx.EvalAdd(T2[m - 1], cu);
	} else {
		cu = ctx.EvalAdd(T2[m - 1], divcs->q.front() / 2.0);
	}

	auto result = ctx.EvalMult(cu, qu);
	hazeModReduceInPlace(ctx, result);
	ctx.EvalAddInPlace(result, su);
	return result;
}

// Port of internalEvalChebyPolysLinear (ckksrns-advancedshe.cpp:572-621):
// build T[0..k-1] (Chebyshev powers of the linearly-transformed argument) and
// level-equalize them to T[k-1].
static std::vector<Ciphertext<DCRTPoly>> hazeEvalChebyPolysLinear(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& x,
  const std::vector<double>& coefficients,
  double a, double b) {
	const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 1;
	std::vector<Ciphertext<DCRTPoly>> T(k);

	// Linear transformation: y = -1 + 2*(x-a)/(b-a), consuming one level when [a,b] != [-1,1].
	if (!lbcrypto::IsNotEqualNegOne(a) && !lbcrypto::IsNotEqualOne(b)) {
		T[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*x);
	} else {
		double alpha = 2.0 / (b - a);
		double beta	 = a * alpha;
		T[0]		 = ctx.EvalMult(x, alpha);
		hazeModReduceInPlace(ctx, T[0]);
		ctx.EvalAddInPlace(T[0], -1.0 - beta);
	}

	// Build Chebyshev powers up to degree k: T_{2i}(y) = 2*T_i(y)^2 - 1; T_{2i+1} = 2*T_i*T_{i+1} - y.
	for (uint32_t i = 2; i <= k; ++i) {
		if (i & 0x1u) {
			// T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
			T[i - 1] = ctx.EvalMult(T[i / 2 - 1], T[i / 2]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]); // ×2 (no rescale; NSD already 2)
			hazeModReduceInPlace(ctx, T[i - 1]);
			ctx.EvalSubInPlace(T[i - 1], T[0]);
		} else {
			// T_{2i}(y) = 2*T_i(y)^2 - 1
			T[i - 1] = ctx.EvalSquare(T[i / 2 - 1]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]); // ×2
			hazeModReduceInPlace(ctx, T[i - 1]);
			ctx.EvalAddInPlace(T[i - 1], -1.0);
		}
	}

	// Level-equalize: bring all T[0..k-2] to the level of T[k-1].
	for (uint32_t i = 1; i < k; ++i) {
		size_t diff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
		hazeLevelReduceInPlace(T[i - 1], diff);
	}

	return T;
}

// Port of internalEvalChebyshevSeriesLinearWithPrecomp (ckksrns-advancedshe.cpp:624-647):
// compute the linear weighted sum of the T powers, adding the free term at the end.
static Ciphertext<DCRTPoly> hazeEvalChebyshevSeriesLinearWithPrecomp(CryptoContextImpl<DCRTPoly>& ctx,
  std::vector<Ciphertext<DCRTPoly>>& T,
  const std::vector<double>& coefficients) {
	const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 2;

	// Scalar multiply for highest-order term.
	auto result = ctx.EvalMult(T[k], coefficients[k + 1]);

	// Scalar multiply for all other terms and accumulate.
	for (uint32_t i = 0; i < k; ++i) {
		if (lbcrypto::IsNotEqualZero(coefficients[i + 1])) {
			ctx.EvalMultInPlace(T[i], coefficients[i + 1]);
			ctx.EvalAddInPlace(result, T[i]);
		}
	}

	// Rescale after the scalar multiplications.
	hazeModReduceInPlace(ctx, result);

	// Add the free term (constant) / 2.
	ctx.EvalAddInPlace(result, coefficients[0] / 2.0);

	return result;
}

// Port of internalEvalChebyPolysPS (ckksrns-advancedshe.cpp:763-841):
// build T[0..k-1] (the base Chebyshev powers), T2[0..m-1] (the extended chain),
// and T2km1 = T_{k(2^m - 1)}.
struct HazeChebyPS {
	std::vector<Ciphertext<DCRTPoly>> T;	 // T[0] = T_1(y), ... T[k-1] = T_k(y)
	std::vector<Ciphertext<DCRTPoly>> T2; // T2[i] = T_{k*2^i}(y), T2[0] placeholder = T[k-1]
	Ciphertext<DCRTPoly> T2km1;			 // T_{k(2^m-1)}(y)
	uint32_t k, m;
};

static HazeChebyPS hazeEvalChebyPolysPS(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& x,
  uint32_t degree, double a, double b) {
	auto degs  = lbcrypto::ComputeDegreesPS(degree);
	uint32_t k = degs[0];
	uint32_t m = degs[1];

	std::vector<Ciphertext<DCRTPoly>> T(k);
	// Linear transformation (same as linear path).
	if (!lbcrypto::IsNotEqualNegOne(a) && !lbcrypto::IsNotEqualOne(b)) {
		T[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(*x);
	} else {
		double alpha = 2.0 / (b - a);
		double beta	 = a * alpha;
		T[0]		 = ctx.EvalMult(x, alpha);
		hazeModReduceInPlace(ctx, T[0]);
		ctx.EvalAddInPlace(T[0], -1.0 - beta);
	}

	// Build Chebyshev powers up to degree k.
	for (uint32_t i = 2; i <= k; ++i) {
		if (i & 0x1u) {
			T[i - 1] = ctx.EvalMult(T[i / 2 - 1], T[i / 2]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]); // ×2
			hazeModReduceInPlace(ctx, T[i - 1]);
			ctx.EvalSubInPlace(T[i - 1], T[0]);
		} else {
			T[i - 1] = ctx.EvalSquare(T[i / 2 - 1]);
			ctx.EvalAddInPlace(T[i - 1], T[i - 1]); // ×2
			hazeModReduceInPlace(ctx, T[i - 1]);
			ctx.EvalAddInPlace(T[i - 1], -1.0);
		}
	}

	// Under FIXEDAUTO, equalize T[0..k-2] to T[k-1] via AdjustLevelsAndDepthInPlace.
	// For FIXEDAUTO this means adjustForAddOrSub: the engine handles this inside EvalAdd,
	// but we must pre-equalize for the PS algorithm which accesses T directly.
	// Use the same approach as the PS path: AdjustLevelsAndDepthInPlace = SetLevel to max.
	for (uint32_t i = 1; i < k; ++i) {
		size_t diff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
		if (diff > 0)
			hazeLevelReduceInPlace(T[i - 1], diff);
	}

	// Build T2[0..m-1] and T2km1.
	std::vector<Ciphertext<DCRTPoly>> T2(m);
	T2[0] = T.back(); // placeholder

	auto T2km1 = T.back();

	for (uint32_t i = 1; i < m; ++i) {
		// T2[i] = 2*T2[i-1]^2 - 1 = T_{k*2^i}(y)
		T2[i] = ctx.EvalSquare(T2[i - 1]);
		ctx.EvalAddInPlace(T2[i], T2[i]); // ×2
		hazeModReduceInPlace(ctx, T2[i]);
		ctx.EvalAddInPlace(T2[i], -1.0);

		// T2km1 = 2*T2km1 * T2[i] - T2[0] = T_{k(2*m-1)}(y)
		T2km1 = ctx.EvalMult(T2km1, T2[i]);
		ctx.EvalAddInPlace(T2km1, T2km1); // ×2
		hazeModReduceInPlace(ctx, T2km1);
		ctx.EvalSubInPlace(T2km1, T2[0]);
	}

	return HazeChebyPS{ std::move(T), std::move(T2), std::move(T2km1), k, m };
}

// Port of internalEvalChebyshevSeriesPSWithPrecomp (ckksrns-advancedshe.cpp:844-861).
static Ciphertext<DCRTPoly> hazeEvalChebyshevSeriesPSWithPrecomp(CryptoContextImpl<DCRTPoly>& ctx,
  HazeChebyPS& poly,
  const std::vector<double>& coefficients) {
	auto& T	   = poly.T;
	auto& T2   = poly.T2;
	auto& T2km1 = poly.T2km1;
	uint32_t k = poly.k;
	uint32_t m = poly.m;

	uint32_t k2m2k = k * (1u << (m - 1)) - k;

	// f2 = coefficients extended + 1 at position 2*k2m2k+k (monic degree).
	auto f2 = coefficients;
	f2.resize(lbcrypto::Degree(f2) + 1);
	f2.resize(2 * k2m2k + k + 1);
	f2.back() = 1.0;

	return ctx.EvalSub(hazeInnerEvalChebyshevPS(ctx, T[0], f2, k, m, T, T2), T2km1);
}

// Top-level port of EvalChebyshevSeries: dispatch on degree < 5 (linear) vs >= 5 (PS).
// Port of AdvancedSHECKKSRNS::EvalChebyshevSeries (ckksrns-advancedshe.cpp:882-896).
static Ciphertext<DCRTPoly> hazeEvalChebyshevSeriesImpl(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  std::vector<double>& coeffs,
  double a, double b) {
	uint32_t degree = lbcrypto::Degree(coeffs);
	if (degree < 5) {
		// Linear path.
		auto T = hazeEvalChebyPolysLinear(ctx, ct, coeffs, a, b);
		return hazeEvalChebyshevSeriesLinearWithPrecomp(ctx, T, coeffs);
	} else {
		// Paterson-Stockmeyer path.
		auto poly = hazeEvalChebyPolysPS(ctx, ct, degree, a, b);
		return hazeEvalChebyshevSeriesPSWithPrecomp(ctx, poly, coeffs);
	}
}

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

	return ctx.Rescale(blockResults[0]);
}

} // namespace

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
