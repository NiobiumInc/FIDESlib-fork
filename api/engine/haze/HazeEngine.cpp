#ifdef FIDESLIB_ENABLE_HAZE

#include "engine/haze/HazeEngine.hpp"

#include "CryptoContext.hpp"

#include <haze/haze.h>
#include <haze/replay_bridge.h>

#include <openfhe.h>

#include <any>
#include <cstdlib>
#include <filesystem>
#include <memory>
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
		OPENFHE_THROW(std::string("haze backend: ") + what + " failed: " + hazeGetErrorString(err));
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
			OPENFHE_THROW("haze backend: polynomial length " + std::to_string(vals.GetLength()) + " does not match ring dimension " + std::to_string(ringDim));
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

void HazeEngine::loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& /*publicKey*/) {
	if (loaded_) {
		return;
	}
	// haze configuration is process-global (it locks at hazeConfigureDevice), so two
	// concurrently-loaded haze contexts cannot work. Sequential contexts are fine: the
	// previous context's teardown() resets haze and decrements the guard. (The check and
	// the later increment are not atomic together — concurrent loadContext from two threads
	// is not supported, matching haze's own single-threaded configuration model.)
	if (liveEngines_.load() != 0) {
		OPENFHE_THROW("haze backend: another haze context is already loaded in this process; "
					  "haze configuration is process-global, so destroy the previous context first");
	}

	auto& context = hostContext(ctx);
	ringDim_	  = context->GetRingDimension();
	polyBytes_	  = static_cast<size_t>(ringDim_) * sizeof(uint64_t);

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	if (!cryptoParams) {
		OPENFHE_THROW("haze backend: context does not carry CKKS-RNS crypto parameters");
	}
	scalingTech_ = static_cast<int>(cryptoParams->GetScalingTechnique());

	qBase_.clear();
	const auto& qParams = cryptoParams->GetElementParams()->GetParams();
	qBase_.reserve(qParams.size());
	for (const auto& p : qParams) {
		qBase_.push_back(p->GetModulus().template ConvertToInt<uint64_t>());
	}
	if (qBase_.empty()) {
		OPENFHE_THROW("haze backend: context has an empty modulus chain");
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
		OPENFHE_THROW("haze backend: |Q|+|P| = " + std::to_string(qBase_.size()) + "+" + std::to_string(pBase_.size()) + " exceeds haze's 64-moduli table");
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
			OPENFHE_THROW("haze backend: replay bridge returned a zero modulus");
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

	// Host key extraction (relin / rotation / bootstrap keys) lands with the keyswitch
	// phases; keys are extracted from the host context there and uploaded lazily.

	liveEngines_.fetch_add(1);
	loaded_	  = true;
	executed_ = false;
}

void HazeEngine::teardown() {
	if (!loaded_) {
		return;
	}
	outputs_.clear();
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
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	auto& host			 = hostCt(ct);
	const auto& elements = host->GetElements();
	if (elements.size() != 2) {
		OPENFHE_THROW("haze backend: only degree-1 ciphertexts (2 components) are supported; got " + std::to_string(elements.size()));
	}

	auto rows0 = extractRows(elements[0], ringDim_);
	auto rows1 = extractRows(elements[1], ringDim_);
	if (rows0.size() != rows1.size()) {
		OPENFHE_THROW("haze backend: ciphertext components disagree on tower count");
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
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->host);

	auto rows = extractRows(ptImpl->GetElement<lbcrypto::DCRTPoly>(), ringDim_);

	auto payload		   = std::make_shared<HazePtPayload>();
	payload->towers		   = rows.size();
	payload->chain		   = LimbChain(rows);
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
		OPENFHE_THROW("haze backend: value was a program-local; declare it with MarkOutput before the first Decrypt");
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

void HazeEngine::requireComputable(const HazePayload& p, const char* op) const {
	if (executed_ && p.state != Residency::Uploaded) {
		OPENFHE_THROW(std::string("haze backend: ") + op + ": compute after readback is a new epoch; restructure the program so all compute precedes the first Decrypt");
	}
}

void HazeEngine::beginNewProgramIfExecuted() {
	// Reached only after every device operand of the op passed requireComputable, i.e. all
	// are freshly Uploaded: host-in-the-loop (decrypt → host step → re-encrypt) legitimately
	// starts a new program with fresh inputs only.
	if (executed_) {
		executed_ = false;
		outputs_.clear();
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
			OPENFHE_THROW("haze backend: value was a program-local; declare it with MarkOutput before the first Decrypt");
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
		clone->c0	 = LimbChain(pull(srcPayload->c0));
		clone->c1	 = LimbChain(pull(srcPayload->c1));
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
		OPENFHE_THROW("haze backend: SetLevel(" + std::to_string(level) + ") exceeds the modulus chain length");
	}
	const size_t targetTowers = qBase_.size() - level;
	if (targetTowers > payload->towers) {
		OPENFHE_THROW("haze backend: SetLevel can only drop levels (have " + std::to_string(qBase_.size() - payload->towers) + ", requested " + std::to_string(level) + ")");
	}
	// Level drop = truncation: hazeFree the tail limbs, no IR, no flush (OpenFHE's
	// DropLastElements does no arithmetic either). Freeing recorded intermediates mid-epoch
	// is safe — the recorded IR holds its own references (ops.cpp precedent).
	payload->c0.truncate(targetTowers);
	payload->c1.truncate(targetTowers);
	payload->towers = targetTowers;
}

// ---- Operations (aligned ct+ct addition; the FIXEDAUTO adjust family and
// the remaining op surface land in later changes) ----

Ciphertext<DCRTPoly> HazeEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAdd");
	requireComputable(*p2, "EvalAdd");
	beginNewProgramIfExecuted();
	if (p1->towers != p2->towers || p1->noiseScaleDeg != p2->noiseScaleDeg) {
		notImplemented("EvalAdd with mismatched operand levels/depths (FIXEDAUTO adjust)");
	}

	const auto base = qPrefix(p1->towers);
	LimbChain out0(p1->towers, polyBytes_);
	LimbChain out1(p1->towers, polyBytes_);
	hazeCheck(hazeAddMrp(out0.data(), p1->c0.asConst().data(), p2->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	hazeCheck(hazeAddMrp(out1.data(), p1->c1.asConst().data(), p2->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");

	auto res		   = std::make_shared<HazePayload>();
	res->c0			   = std::move(out0);
	res->c1			   = std::move(out1);
	res->towers		   = p1->towers;
	res->noiseScaleDeg = p1->noiseScaleDeg; // add: unchanged (operands equal after adjust)
	res->scalingFactor = p1->scalingFactor;
	res->slots		   = std::max(p1->slots, p2->slots);
	res->state		   = Residency::Recorded;
	return wrapDeviceResult(ctx, ct1, std::move(res));
}

void HazeEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	auto p1 = ensureCt(ctx, ct1);
	auto p2 = ensureCt(ctx, ct2);
	requireComputable(*p1, "EvalAddInPlace");
	requireComputable(*p2, "EvalAddInPlace");
	beginNewProgramIfExecuted();
	if (p1->towers != p2->towers || p1->noiseScaleDeg != p2->noiseScaleDeg) {
		notImplemented("EvalAddInPlace with mismatched operand levels/depths (FIXEDAUTO adjust)");
	}

	// Fresh chains per op result (engine policy); "in place" = rebind the payload
	// contents and free the old chains via move-assignment.
	const auto base = qPrefix(p1->towers);
	LimbChain out0(p1->towers, polyBytes_);
	LimbChain out1(p1->towers, polyBytes_);
	hazeCheck(hazeAddMrp(out0.data(), p1->c0.asConst().data(), p2->c0.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	hazeCheck(hazeAddMrp(out1.data(), p1->c1.asConst().data(), p2->c1.asConst().data(), base.data(), base.size(), nullptr), "hazeAddMrp");
	p1->c0	  = std::move(out0);
	p1->c1	  = std::move(out1);
	p1->slots = std::max(p1->slots, p2->slots);
	p1->state = Residency::Recorded;
}

std::vector<uint64_t> HazeEngine::qPrefix(size_t towers) const {
	return { qBase_.begin(), qBase_.begin() + static_cast<std::ptrdiff_t>(towers) };
}

} // namespace fideslib

#endif // FIDESLIB_ENABLE_HAZE
