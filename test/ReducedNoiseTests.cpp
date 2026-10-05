// Tests for the reducedNoise (FBC variant) plumbing: MakeEngine's cpu refusal, GenCryptoContext's
// unset-value refusal, and the serialized .dev field's round trip. Deliberately separate from
// ApiTests.cpp so this file stays the one place that exercises engine construction and
// serialization directly against the flag, rather than through a fixture's already-built context.
// Run:    ./build/fideslib-test --gtest_filter='ReducedNoise.*'          (CUDA-enabled configure)
// Run:    ./build-cpu/fideslib-cpu-test --gtest_filter='ReducedNoise.*'  (CPU-only configure)

#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h> // getpid
#include <vector>

#include "Serialize.hpp"
#include "TestEngineConfig.hpp" // FIDESlib::Testing::ConfigureTestEngine
#include "engine/Engine.hpp"	// Engine::reducedNoise(), for cc->engine_->reducedNoise()
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;

namespace {

// A unique directory under the system temp root, recursively removed on destruction -- including
// when a test body exits early via an ASSERT_* failure, which unwinds the stack through this
// guard same as any other exception-driven unwind.
class ScopedTempDir {
  public:
	ScopedTempDir() {
		static std::atomic<int> counter{ 0 };
		path_ = std::filesystem::temp_directory_path() / ("fideslib_reduced_noise_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
		std::filesystem::create_directories(path_);
	}

	~ScopedTempDir() {
		std::error_code ec;
		std::filesystem::remove_all(path_, ec);
	}

	ScopedTempDir(const ScopedTempDir&)			   = delete;
	ScopedTempDir& operator=(const ScopedTempDir&) = delete;

	std::filesystem::path file(const std::string& name) const {
		return path_ / name;
	}

  private:
	std::filesystem::path path_;
};

// A minimal, cheap-to-build CCParams: these tests exercise engine construction and
// serialization, not FHE correctness, so depth/ring/slots are the smallest workable values.
// CCParams has no move/copy constructor, so this configures in place rather than returning one.
void ConfigureMinimalParams(CCParams<CryptoContextCKKSRNS>& params) {
	params.SetMultiplicativeDepth(1);
	params.SetScalingModSize(50);
	params.SetBatchSize(8);
	params.SetRingDim(1u << 12);
	params.SetSecurityLevel(HEStd_NotSet); // small ring on purpose; not meant to be secure
}
} // namespace

// MakeEngine(CPU, ...) has only one variant to run -- the one the linked OpenFHE was compiled
// with -- so the oracle's own value is always accepted and its opposite always refused.
TEST(ReducedNoise, CpuAcceptsOracleVariant) {
	EXPECT_NO_THROW(MakeEngine(Backend::CPU, LinkedOpenFheReducedNoise()));
}

TEST(ReducedNoise, CpuRefusesMismatch) {
	const bool requested = !LinkedOpenFheReducedNoise();
	try {
		MakeEngine(Backend::CPU, requested);
		FAIL() << "expected MakeEngine(CPU, " << requested << ") to throw";
	} catch (const std::runtime_error& e) {
		const std::string what = e.what();
		EXPECT_NE(what.find(requested ? "reducedNoise=true" : "reducedNoise=false"), std::string::npos) << what;
		EXPECT_NE(what.find(LinkedOpenFheReducedNoise() ? "WITH_REDUCED_NOISE=ON" : "WITH_REDUCED_NOISE=OFF"), std::string::npos) << what;
	}
}

// CCParams::reducedNoise has no default; GenCryptoContext must refuse to build from an unset one
// rather than silently picking a variant.
TEST(ReducedNoise, GenCryptoContextRefusesUnsetReducedNoise) {
	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	// Deliberately no SetReducedNoise call.
	EXPECT_THROW(GenCryptoContext(params), std::runtime_error);
}

// A .dev file predating the ReducedNoise field cannot say which FBC variant the engine was built
// for, so deserializing one must fail rather than silently default.
TEST(ReducedNoise, DeserializeRefusesMissingReducedNoise) {
	ScopedTempDir dir;
	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	ConfigureTestEngine(params);
	auto cc = GenCryptoContext(params);
	cc->Enable(PKE);

	const std::string path = dir.file("ctx_missing_reduced_noise").string();
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	{
		std::ifstream in(path + ".dev");
		std::vector<std::string> lines;
		std::string line;
		while (std::getline(in, line))
			lines.push_back(line);
		in.close();
		ASSERT_FALSE(lines.empty());
		ASSERT_EQ(lines.back().rfind(Serial::kReducedNoiseLabel, 0), 0u) << "last .dev line is not " << Serial::kReducedNoiseLabel;
		lines.pop_back();

		std::ofstream out(path + ".dev", std::ios::trunc);
		for (const auto& l : lines)
			out << l << "\n";
	}

	CryptoContext<DCRTPoly> cc2;
	EXPECT_FALSE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
}

// Round-trips a cuda context built with the OPPOSITE of the oracle's variant (cuda has no
// refusal gate, so this is a legal, distinct configuration), and confirms the deserialized
// engine reports the same reducedNoise() the original was built with -- i.e. that the .dev
// field, not some default, is what determines it on read-back.
TEST(ReducedNoise, SerializeDeserializeCudaReducedNoiseRoundTrip) {
	if (!IsBackendAvailable(Backend::CUDA))
		GTEST_SKIP() << "requires a CUDA build";

	ScopedTempDir dir;
	const bool kOppositeOfOracle = !LinkedOpenFheReducedNoise();

	CCParams<CryptoContextCKKSRNS> params;
	ConfigureMinimalParams(params);
	params.SetBackend(Backend::CUDA);
	params.SetReducedNoise(kOppositeOfOracle);
	auto cc = GenCryptoContext(params);
	ASSERT_EQ(cc->engine_->reducedNoise(), kOppositeOfOracle);

	const std::string path = dir.file("cuda_reduced_noise_roundtrip").string();
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	CryptoContext<DCRTPoly> cc2;
	ASSERT_TRUE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
	ASSERT_NE(cc2.get(), nullptr);
	EXPECT_EQ(cc2->engine_->reducedNoise(), kOppositeOfOracle);
}
