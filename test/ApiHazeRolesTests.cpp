//
// The haze engine's recording modes: trace-only recording (FIDESLIB_HAZE_TRACE_ONLY=1) and the
// roles file it writes beside every program (fideslib.input_roles.json), with
// FIDESLIB_HAZE_OMIT_INPUTS. Each test sets its own environment, records one small program in a
// fresh runs directory and reads what the engine left there. haze only: on any other backend
// nothing is recorded.
//
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "TestEngineConfig.hpp" // FIDESlib::Testing::ConfigureTestEngine
#include "TestRegistration.hpp" // FIDES_TEST_IF, OnHaze
#include "fideslib.hpp"

using namespace fideslib;
using FIDESlib::Testing::ConfigureTestEngine;
using FIDESlib::Testing::OnHaze;

namespace {

namespace fs = std::filesystem;

// Sets environment variables for one scope and restores what was there before.
class ScopedEnv {
  public:
	void set(const char* name, const std::string& value) {
		const char* old = std::getenv(name);
		saved_.emplace_back(name, old ? std::optional<std::string>(old) : std::nullopt);
		setenv(name, value.c_str(), 1);
	}

	void unset(const char* name) {
		const char* old = std::getenv(name);
		saved_.emplace_back(name, old ? std::optional<std::string>(old) : std::nullopt);
		unsetenv(name);
	}

	~ScopedEnv() {
		for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
			if (it->second)
				setenv(it->first.c_str(), it->second->c_str(), 1);
			else
				unsetenv(it->first.c_str());
		}
	}

  private:
	std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

// A fresh runs directory under the one the suite was given (or the temp dir), removed at scope exit.
class RunsDir {
  public:
	explicit RunsDir(const std::string& name) {
		const char* base = std::getenv("FIDESLIB_HAZE_RUNS_DIR");
		root_			 = fs::path(base && base[0] ? base : fs::temp_directory_path().string()) / ("roles-" + name + "-" + std::to_string(getpid()));
		fs::remove_all(root_);
		fs::create_directories(root_);
	}

	~RunsDir() {
		std::error_code ec;
		fs::remove_all(root_, ec);
	}

	const fs::path& root() const {
		return root_;
	}

	// The one program directory the engine made: <root>/<key>/ctx<N>.
	fs::path programDir() const {
		std::vector<fs::path> dirs;
		// Only the ctx<N> directories: the engine keeps other entries beside them.
		for (const auto& key : fs::directory_iterator(root_))
			if (key.is_directory())
				for (const auto& ctx : fs::directory_iterator(key.path()))
					if (ctx.is_directory() && ctx.path().filename().string().rfind("ctx", 0) == 0)
						dirs.push_back(ctx.path());
		EXPECT_EQ(dirs.size(), 1u) << "program directories under " << root_;
		return dirs.empty() ? fs::path() : dirs.front();
	}

  private:
	fs::path root_;
};

std::string Slurp(const fs::path& p) {
	std::ifstream in(p);
	return { std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
}

// One entry of the roles file's "inputs" array: the writer puts each on its own line.
struct RoleLine {
	std::string name, role, tag;
	bool omitted = false;
};

std::string Field(const std::string& line, const std::string& key) {
	const std::string open = "\"" + key + "\": \"";
	const size_t at		   = line.find(open);
	if (at == std::string::npos)
		return {};
	const size_t from = at + open.size();
	return line.substr(from, line.find('"', from) - from);
}

std::vector<RoleLine> InputRoles(const fs::path& rolesFile) {
	std::vector<RoleLine> roles;
	std::ifstream in(rolesFile);
	std::string line;
	bool inInputs = false;
	while (std::getline(in, line)) {
		if (line.find("\"inputs\": [") != std::string::npos) {
			inInputs = true;
			continue;
		}
		if (inInputs && line.find(']') == line.find_first_not_of(' ') && line.find("{\"name\"") == std::string::npos)
			break; // end of the inputs array
		if (inInputs && line.find("{\"name\"") != std::string::npos)
			roles.push_back({ Field(line, "name"), Field(line, "role"), Field(line, "tag"), line.find("\"omitted\": true") != std::string::npos });
	}
	return roles;
}

// The input names the program directory has a word file for: fideslib.input_<name>.bin.
std::vector<std::string> InputFiles(const fs::path& dir) {
	std::vector<std::string> names;
	const std::string prefix = "fideslib.input_";
	for (const auto& e : fs::directory_iterator(dir)) {
		const std::string f = e.path().filename().string();
		if (f.rfind(prefix, 0) == 0 && e.path().extension() == ".bin")
			names.push_back(f.substr(prefix.size(), f.size() - prefix.size() - 4));
	}
	return names;
}

struct Context {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
};

Context MakeContext() {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(2);
	params.SetScalingModSize(50);
	params.SetBatchSize(8);
	params.SetRingDim(1u << 13);
	params.SetSecurityLevel(HEStd_NotSet);
	params.SetScalingTechnique(FIXEDAUTO);
	ConfigureTestEngine(params);
	Context c;
	c.cc = GenCryptoContext(params);
	c.cc->Enable(PKE);
	c.cc->Enable(KEYSWITCH);
	c.cc->Enable(LEVELEDSHE);
	c.keys = c.cc->KeyGen();
	c.cc->EvalMultKeyGen(c.keys.secretKey);
	c.cc->LoadContext(c.keys.publicKey);
	return c;
}

std::vector<double> DecryptValues(Context& c, Ciphertext<DCRTPoly>& ct, size_t n) {
	Plaintext pt;
	c.cc->Decrypt(ct, c.keys.secretKey, &pt);
	pt->SetLength(n);
	return pt->GetRealPackedValue();
}

const std::vector<double> kValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

} // namespace

// Trace-only: reading an input back gives the caller's own value (its words were never uploaded,
// and the host value is the value), before and after the program is written; and the program
// directory, all a trace-only recording produces, outlives the context without
// FIDESLIB_HAZE_KEEP_RUNS.
FIDES_TEST_IF(OnHaze(), HazeRoles, TraceOnlyKeepsInputValuesAndProgram) {
	RunsDir runs("trace-only");
	ScopedEnv env;
	env.set("FIDESLIB_HAZE_TRACE_ONLY", "1");
	env.set("FIDESLIB_HAZE_RUNS_DIR", runs.root().string());
	env.unset("FIDESLIB_HAZE_KEEP_RUNS");
	env.unset("FIDESLIB_HAZE_PROGRAM_KEY");
	env.unset("FIDESLIB_HAZE_OMIT_INPUTS");
	fs::path program;
	{
		Context c  = MakeContext();
		auto input = c.cc->MakeCKKSPackedPlaintext(kValues);
		auto ct	   = c.cc->Encrypt(input, c.keys.publicKey);
		auto sum   = c.cc->EvalAdd(ct, ct);
		c.cc->MarkOutput(sum);

		auto before = DecryptValues(c, ct, kValues.size());
		for (size_t i = 0; i < kValues.size(); ++i)
			EXPECT_NEAR(before[i], kValues[i], 1e-6) << "input slot " << i << " before the program is written";

		(void)DecryptValues(c, sum, kValues.size()); // writes the program (a result reads back as zeros)
		program = runs.programDir();
		EXPECT_TRUE(fs::exists(program / "fideslib.input_roles.json"));

		auto after = DecryptValues(c, ct, kValues.size());
		for (size_t i = 0; i < kValues.size(); ++i)
			EXPECT_NEAR(after[i], kValues[i], 1e-6) << "input slot " << i << " after the program is written";
	}
	ASSERT_FALSE(program.empty());
	EXPECT_TRUE(fs::exists(program / "fideslib.input_roles.json")) << program << " was removed with the context";
}

// The roles file names every input the program has a word file for (the monomial MultByMonomial
// uploads included), and FIDESLIB_HAZE_OMIT_INPUTS=plaintext empties a tagged plaintext's words
// but keeps an untagged one's, a constant of the program no runtime can supply.
FIDES_TEST_IF(OnHaze(), HazeRoles, RolesNameEveryInputAndOmitKeepsConstants) {
	RunsDir runs("omit");
	ScopedEnv env;
	env.set("FIDESLIB_HAZE_RUNS_DIR", runs.root().string());
	env.set("FIDESLIB_HAZE_KEEP_RUNS", "1");
	env.set("FIDESLIB_HAZE_OMIT_INPUTS", "plaintext");
	env.unset("FIDESLIB_HAZE_TRACE_ONLY");
	env.unset("FIDESLIB_HAZE_PROGRAM_KEY");
	{
		Context c		= MakeContext();
		auto input		= c.cc->MakeCKKSPackedPlaintext(kValues);
		auto ct			= c.cc->Encrypt(input, c.keys.publicKey);
		auto constant	= c.cc->MakeCKKSPackedPlaintext(std::vector<double>(kValues.size(), 0.5));
		auto weights	= c.cc->MakeCKKSPackedPlaintext(std::vector<double>(kValues.size(), 2.0));
		weights->tag	= "weights";
		auto byConstant = c.cc->EvalMult(ct, constant);
		auto byWeights	= c.cc->EvalMult(ct, weights);
		auto shifted	= c.cc->MultByMonomial(ct, 1);
		c.cc->MarkOutput(byConstant);
		c.cc->MarkOutput(byWeights);
		c.cc->MarkOutput(shifted);
		auto got = DecryptValues(c, byWeights, kValues.size());
		for (size_t i = 0; i < kValues.size(); ++i)
			EXPECT_NEAR(got[i], 2.0 * kValues[i], 1e-3) << "slot " << i;

		const fs::path program = runs.programDir();
		ASSERT_FALSE(program.empty());
		const fs::path rolesFile = program / "fideslib.input_roles.json";
		ASSERT_TRUE(fs::exists(rolesFile));
		const auto roles = InputRoles(rolesFile);
		SCOPED_TRACE(Slurp(rolesFile));

		for (const std::string& name : InputFiles(program)) {
			bool listed = false;
			for (const auto& r : roles)
				listed = listed || r.name == name;
			EXPECT_TRUE(listed) << "input " << name << " has a word file but no role";
		}

		size_t constants = 0, tagged = 0;
		for (const auto& r : roles) {
			if (r.role != "plaintext")
				continue;
			const auto bin = program / ("fideslib.input_" + r.name + ".bin");
			if (r.tag.empty()) {
				++constants;
				EXPECT_FALSE(r.omitted) << "constant " << r.name << " marked omitted";
				EXPECT_GT(fs::file_size(bin), 0u) << "constant " << r.name << " lost its words";
			} else {
				++tagged;
				EXPECT_EQ(r.tag, "weights");
				EXPECT_TRUE(r.omitted) << "tagged plaintext " << r.name << " not omitted";
				EXPECT_EQ(fs::file_size(bin), 0u) << "tagged plaintext " << r.name << " kept its words";
			}
		}
		EXPECT_GE(constants, 2u) << "the untagged plaintext and the monomial";
		EXPECT_EQ(tagged, 1u);
	}
}
