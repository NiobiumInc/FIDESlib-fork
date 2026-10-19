// PROBE (not for merge): does LimbPartition::multPt read a plaintext before the plaintext's own
// stream has finished writing it?
//
// The plaintext is overwritten on its own stream by a device-to-device copy that starts only after a
// long spin kernel. A multiply issued right after must wait for that copy. With the plaintext on the
// legacy default stream the wait comes for free from the legacy stream's implicit ordering; on a
// pooled stream it has to be an explicit fence.
#include <openfhe.h>

#include <gtest/gtest.h>

#include <utility>
#include <variant>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

namespace FIDESlib::Testing {

namespace {

__global__ void multpt_probe_spin(const long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

std::pair<void*, size_t> ProbeLimbBytes(const FIDESlib::CKKS::LimbImpl& l) {
    return std::visit([](const auto& L) { return std::pair<void*, size_t>{ (void*)L.v.data, (size_t)L.v.size * sizeof(*L.v.data) }; }, l);
}

size_t CountDifferences(const FIDESlib::CKKS::RawCipherText& a, const FIDESlib::CKKS::RawCipherText& b) {
    size_t bad = 0;
    auto cmp = [&](const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
        if (x.size() != y.size()) {
            bad += 1u << 30;
            return;
        }
        for (size_t l = 0; l < x.size(); ++l) {
            if (x[l].size() != y[l].size()) {
                bad += 1u << 30;
                continue;
            }
            for (size_t i = 0; i < x[l].size(); ++i)
                if (x[l][i] != y[l][i])
                    ++bad;
        }
    };
    cmp(a.sub_0, b.sub_0);
    cmp(a.sub_1, b.sub_1);
    return bad;
}

} // namespace

class MultPtFenceProbe : public GeneralParametrizedTest {};

TEST_P(MultPtFenceProbe, MultPtWaitsForThePlaintext) {
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context& cc_ = GPUcc;
    cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);

    std::vector<double> x(8), y(8), z(8);
    for (int i = 0; i < 8; ++i) {
        x[i] = 0.1 * (i + 1);
        y[i] = 0.05 * (i + 2);
        z[i] = -0.3 * (i + 3); // what the plaintext holds until the copy lands
    }
    const FIDESlib::CKKS::RawCipherText raw_ct = FIDESlib::CKKS::GetRawCipherText(cc, cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(x)));
    const FIDESlib::CKKS::RawPlainText raw_y = FIDESlib::CKKS::GetRawPlainText(cc, cc->MakeCKKSPackedPlaintext(y));
    const FIDESlib::CKKS::RawPlainText raw_z = FIDESlib::CKKS::GetRawPlainText(cc, cc->MakeCKKSPackedPlaintext(z));

    // The reference: the plaintext loaded and the device idle before the multiply.
    FIDESlib::CKKS::RawCipherText want;
    {
        FIDESlib::CKKS::Ciphertext ct(cc_, raw_ct);
        FIDESlib::CKKS::Plaintext pt(cc_, raw_y);
        cudaDeviceSynchronize();
        ct.multPt(pt, true);
        ct.store(want);
    }
    cudaDeviceSynchronize();

    constexpr long long kSpinCycles = 200'000'000; // about 100 ms
    FIDESlib::CKKS::RawCipherText got;
    {
        FIDESlib::CKKS::Ciphertext ct(cc_, raw_ct);
        FIDESlib::CKKS::Plaintext pt(cc_, raw_z);
        FIDESlib::CKKS::Plaintext src(cc_, raw_y);
        cudaDeviceSynchronize();
        ASSERT_EQ(pt.c0.GPU.size(), src.c0.GPU.size());
        for (size_t g = 0; g < pt.c0.GPU.size(); ++g) {
            FIDESlib::CKKS::LimbPartition& to = pt.c0.GPU[g];
            const FIDESlib::CKKS::LimbPartition& from = src.c0.GPU[g];
            ASSERT_EQ(to.limb.size(), from.limb.size());
            cudaSetDevice(to.device);
            const cudaStream_t s = to.getS().ptr();
            multpt_probe_spin<<<1, 1, 0, s>>>(kSpinCycles);
            for (size_t i = 0; i < to.limb.size(); ++i) {
                const auto [dst, n] = ProbeLimbBytes(to.limb[i]);
                const auto [from_ptr, m] = ProbeLimbBytes(from.limb[i]);
                ASSERT_EQ(n, m);
                cudaMemcpyAsync(dst, from_ptr, n, cudaMemcpyDeviceToDevice, s);
            }
        }
        ct.multPt(pt, true);
        ct.store(got);
        cudaDeviceSynchronize();
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const size_t bad = CountDifferences(want, got);
    std::cout << "[probe] differing coefficients: " << bad << std::endl;
    EXPECT_EQ(bad, 0u) << "a multiply issued while its plaintext was still being written on the plaintext's own stream";
}

INSTANTIATE_TEST_SUITE_P(MultPtFenceProbes, MultPtFenceProbe, testing::Values(tparams64_13_fix));

} // namespace FIDESlib::Testing
