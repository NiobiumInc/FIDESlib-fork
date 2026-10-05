// Copyright (C) 2026, All rights reserved by Niobium Microsystems.
//
// K copies of one atomic CKKS operation per process, written once against the
// FIDESlib facade and run on whichever backend FIDESLIB_BACKEND selects: `cuda`
// (RTX 5090) or `haze` (record -> compiler -> accelerator). Same source, same op
// sequence, same parameters — the strongest available "same program" claim for
// the GPU-vs-FPGA comparison.
//
// Sweeping K and regressing separates the two numbers a single measurement
// cannot: the slope is the op's marginal device cost, the intercept is the fixed
// per-program overhead. That overhead is not negligible — a mixed-op fit over
// earlier single-op runs put it at a significant share of a cheap op's total.
//
// The K operand pairs carry DISTINCT values and each op reads its own pair:
// identical values would let the compiler fold the repeats into one, and reusing
// a destination would serialise them into a dependency chain, which measures
// latency rather than throughput.
//
// One op *kind* per process because HazeEngine records ONE program per context
// and executes it at the first readback; that also makes the replay's own
// device time cover exactly the K ops plus one program's overhead.
//
// Ring dimension is pinned to 65536.
//
//   usage: atomic_ops_engine <add|sub|scalar-mult|pt-mult|mult|rotate|rescale> [repeat]

#include "BackendEnv.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fideslib.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fideslib;

namespace {

constexpr uint32_t kRing = 65536;
constexpr uint32_t kMultDepth = 2;
constexpr uint32_t kScaleBits = 50;
constexpr uint32_t kBatch = 8;

const std::vector<double> kA = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
const std::vector<double> kB = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
constexpr double kScalar = 4.0;

// Distinct per-repeat operands, kept the same order of magnitude so every repeat
// is equally hard and the regression is not skewed by operand size.
std::vector<double> operand(const std::vector<double> &base, size_t k, double step) {
    std::vector<double> v(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        v[i] = base[i] + step * static_cast<double>(k + 1);
    }
    return v;
}

double us_since(const std::chrono::steady_clock::time_point &t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

const char *backend_name(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::CUDA: return "cuda";
        case Backend::HAZE: return "haze";
    }
    return "?";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0]
                  << " <add|sub|scalar-mult|pt-mult|mult|rotate|rescale> [repeat]\n";
        return 2;
    }
    const std::string op = argv[1];
    const long repeat_arg = (argc == 3) ? std::atol(argv[2]) : 1;
    if (repeat_arg < 1) {
        std::cerr << "repeat must be >= 1\n";
        return 2;
    }
    const size_t repeat = static_cast<size_t>(repeat_arg);

    const Backend backend = BackendFromEnv();

    // Rescale is only meaningful under FIXEDMANUAL — the auto techniques rescale
    // inside the op that needs it. pt-mult shares the mode so the two are
    // directly differenceable.
    const bool manual = (op == "rescale" || op == "pt-mult");

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(kMultDepth);
    parameters.SetScalingModSize(kScaleBits);
    parameters.SetBatchSize(kBatch);
    parameters.SetRingDim(kRing);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingTechnique(manual ? FIXEDMANUAL : FLEXIBLEAUTO);
    parameters.SetBackend(backend);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise()); // matches the oracle's variant

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    if (cc->GetRingDimension() != kRing) {
        std::cerr << "ring dimension is " << cc->GetRingDimension() << ", expected " << kRing << "\n";
        return 1;
    }

    auto keys = cc->KeyGen();
    if (op == "mult") {
        cc->EvalMultKeyGen(keys.secretKey);
    }
    if (op == "rotate") {
        cc->EvalRotateKeyGen(keys.secretKey, {1});
    }
    cc->LoadContext(keys.publicKey);

    // One independent operand pair per repeat.
    std::vector<std::vector<double>> a_vals(repeat), b_vals(repeat);
    std::vector<Plaintext> pt_a(repeat), pt_b(repeat);
    std::vector<Ciphertext<DCRTPoly>> ct_a(repeat), ct_b(repeat);
    for (size_t k = 0; k < repeat; ++k) {
        a_vals[k] = operand(kA, k, 0.01);
        b_vals[k] = operand(kB, k, 0.02);
        pt_a[k] = cc->MakeCKKSPackedPlaintext(a_vals[k]);
        pt_b[k] = cc->MakeCKKSPackedPlaintext(b_vals[k]);
        ct_a[k] = cc->Encrypt(keys.publicKey, pt_a[k]);
        ct_b[k] = cc->Encrypt(keys.publicKey, pt_b[k]);
    }

    // Scale-doubling setup for the rescale case, deliberately outside the timer:
    // a bare rescale leaves nothing checkable, so the plaintext multiply gives it
    // something to undo.
    std::vector<Ciphertext<DCRTPoly>> ct_scaled(repeat);
    if (op == "rescale") {
        for (size_t k = 0; k < repeat; ++k) {
            ct_scaled[k] = cc->EvalMult(ct_a[k], pt_b[k]);
        }
    }

    std::vector<Ciphertext<DCRTPoly>> res(repeat);

    // Issue: on CUDA this is the work; on haze it only appends to the recording.
    const auto t_issue = std::chrono::steady_clock::now();
    for (size_t k = 0; k < repeat; ++k) {
        if (op == "add") {
            res[k] = cc->EvalAdd(ct_a[k], ct_b[k]);
        } else if (op == "sub") {
            res[k] = cc->EvalSub(ct_a[k], ct_b[k]);
        } else if (op == "scalar-mult") {
            res[k] = cc->EvalMult(ct_a[k], kScalar);
        } else if (op == "pt-mult") {
            res[k] = cc->EvalMult(ct_a[k], pt_b[k]);
        } else if (op == "mult") {
            res[k] = cc->EvalMult(ct_a[k], ct_b[k]);
        } else if (op == "rotate") {
            res[k] = cc->EvalRotate(ct_a[k], 1);
        } else if (op == "rescale") {
            res[k] = cc->Rescale(ct_scaled[k]);
        } else {
            std::cerr << "unknown op: " << op << "\n";
            return 2;
        }
    }
    const double issue_us = us_since(t_issue);

    std::vector<std::vector<double>> expected(repeat, std::vector<double>(kA.size()));
    for (size_t k = 0; k < repeat; ++k) {
        for (size_t i = 0; i < kA.size(); ++i) {
            if (op == "add") {
                expected[k][i] = a_vals[k][i] + b_vals[k][i];
            } else if (op == "sub") {
                expected[k][i] = a_vals[k][i] - b_vals[k][i];
            } else if (op == "scalar-mult") {
                expected[k][i] = a_vals[k][i] * kScalar;
            } else if (op == "rotate") {
                expected[k][i] = a_vals[k][(i + 1) % kA.size()];
            } else { // pt-mult, mult, rescale
                expected[k][i] = a_vals[k][i] * b_vals[k][i];
            }
        }
    }

    // Declare every output before any readback: haze executes the recorded
    // program at the first Decrypt, and only declared outputs are tagged.
    for (size_t k = 0; k < repeat; ++k) {
        cc->MarkOutput(res[k]);
    }

    // Readback: on haze this is where record -> lower -> DMA -> execute -> read
    // happens. On CUDA it is a device-to-host copy of already-computed values.
    const auto t_read = std::chrono::steady_clock::now();
    double max_err = 0.0;
    for (size_t k = 0; k < repeat; ++k) {
        Plaintext out;
        cc->Decrypt(keys.secretKey, res[k], &out);
        out->SetLength(kA.size());
        const auto slots = out->GetRealPackedValue();
        for (size_t i = 0; i < kA.size(); ++i) {
            max_err = std::max(max_err, std::fabs(slots[i] - expected[k][i]));
        }
    }
    const double readback_us = us_since(t_read);

    std::cout << std::fixed << std::setprecision(1)
              << "[ENGINE_TIMING] backend=" << backend_name(backend) << " op=" << op
              << " repeat=" << repeat << " ring=" << cc->GetRingDimension()
              << " mode=" << (manual ? "FIXEDMANUAL" : "FLEXIBLEAUTO")
              << " issue_us=" << issue_us << " readback_us=" << readback_us
              << " total_us=" << (issue_us + readback_us)
              << " issue_us_per_op=" << (issue_us / static_cast<double>(repeat)) << std::endl;
    std::cout << std::scientific << std::setprecision(3)
              << "[ENGINE_CHECK] op=" << op << " repeat=" << repeat
              << " max_abs_err=" << max_err << std::endl;

    // A timing number from an unverified computation is worse than no number.
    constexpr double kTol = 1e-4;
    if (!(max_err < kTol)) {
        std::cerr << "[ENGINE_CHECK] FAILED: max_abs_err " << max_err << " >= " << kTol << "\n";
        return 1;
    }
    return 0;
}
