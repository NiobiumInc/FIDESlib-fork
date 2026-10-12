#include "ParametrizedTest.cuh"

#include <algorithm>
#include <any>
#include <fideslib.hpp>
#include <map>
#include <set>
#include <vector>

using namespace fideslib;

namespace FIDESlib::Testing {

TEST(OpenFHECompatTests, EvalBootstrap) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Deepest level the host EvalBootstrap accepts: L0 - (levelBudget_decode + 2 + (st ==
    // FLEXIBLEAUTOEXT)), from the guard at ckksrns-fhe.cpp:1252-1256. These tests take the CPU
    // reference through that host call, so multDepth - 1 throws "Not enough levels" before
    // reaching the comparison. The FLEXIBLEAUTOEXT cases subtract one more for the extra tower.
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    // EXPECT_EQ(cBoot1->GetElements(), cBoot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


TEST(OpenFHECompatTests, EvalBootstrapDense) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    // EXPECT_EQ(cBoot1->GetElements(), cBoot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}


// level budget {1,1} takes the isLT/EvalLinearTransform branch
// of CoeffsToSlots/SlotsToCoeffs instead of the FFT-decomposed Horner path.
TEST(OpenFHECompatTests, EvalBootstrapLT) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// a different sparse slot count changes the PartialSum depth
// and the CtS/StC FFT split parameters.
TEST(OpenFHECompatTests, EvalBootstrapSlots64) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 64;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// FIXEDMANUAL bootstrap exercises the manual-rescale branches
// of the raise/Chebyshev/StC transcriptions that the FLEXIBLE tests never reach.
TEST(OpenFHECompatTests, EvalBootstrapFixedManual) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// Fully-packed bootstrap under FIXEDMANUAL: the dense branch splits into
// real/imaginary Chebyshev evaluations and recombines, a pipeline the sparse
// FIXEDMANUAL test never enters.
TEST(OpenFHECompatTests, EvalBootstrapDenseFixedManual) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    // The slot-dependent default correction factor lands at 9 for fully-packed
    // slots at this toy ring size, below deg = log2(2^60/2^50) = 10, which
    // EvalBootstrap rejects. Production-scale parameters don't trip this;
    // pass an explicit correction factor to get the same moduli as the
    // FLEXIBLEAUTO dense test.
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 10);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}


// SPARSE_ENCAPSULATED secret-key distribution takes its own
// Chebyshev coefficient set (g_coefficientsSparseEncapsulated) and keygen path.
TEST(OpenFHECompatTests, DISABLED_EvalBootstrapSparseEncaps) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// FLEXIBLEAUTOEXT (OpenFHE's default) adds an extra level at
// encryption and changes the ModRaise handling.
TEST(OpenFHECompatTests, EvalBootstrapFlexExt) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    // The ModRaise-first refresh lands at level 20 at these parameters, and an input no deeper
    // than that comes back unchanged. multDepth - 2 leaves it levels to gain.
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 2, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// Combination sweep: fully packed + FLEXIBLEAUTOEXT — dense real/imaginary
// split downstream of the extra-level ModRaise.
TEST(OpenFHECompatTests, EvalBootstrapDenseFlexExt) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    // The ModRaise-first refresh lands at level 20 at these parameters, and an input no deeper
    // than that comes back unchanged. multDepth - 2 leaves it levels to gain.
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 2, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}


// Combination sweep: level budget {1,1} (EvalLinearTransform branch) + FIXEDMANUAL —
// the LT path's rescale gates were only ever exercised under FLEXIBLEAUTO.
TEST(OpenFHECompatTests, EvalBootstrapLTFixedManual) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// Combination sweep: SPARSE_TERNARY secret keys — selects the FIDESlib::SPARSE
// boot config (g_coefficientsSparse Chebyshev set, bootK=1.0, no encapsulation),
// a coefficient/keygen path no other test touches.
TEST(OpenFHECompatTests, EvalBootstrapSparseSecret) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}


// Combination sweep: FIXEDAUTO bootstrap — the fourth scaling technique,
// previously untested at any level. DISABLED: known divergence (O6e) — the GPU
// add/sub operand adjustment only runs under FLEXIBLEAUTO/FLEXIBLEAUTOEXT,
// while stock AdjustForAddOrSub adjusts for every technique except FIXEDMANUAL,
// FIXEDAUTO included. Acceptance test for the O6e fix.
TEST(OpenFHECompatTests, DISABLED_EvalBootstrapFixedAuto) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    parameters.SetReducedNoise(LinkedOpenFheReducedNoise());

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - (levelBudget[1] + 2), nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);
    ASSERT_LT(cBoot1->GetLevel(), ctxt->GetLevel()) << "bootstrap gained no levels; the comparison is vacuous";

    //====================================================================

    cc->SetCudaDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

} // namespace FIDESlib::Testing
