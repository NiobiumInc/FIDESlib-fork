// Device-encode correctness anchors.
//
// MakeCKKSPackedPlaintextDevice claims to produce the SAME plaintext as MakeCKKSPackedPlaintext —
// bit for bit, not approximately — while moving one coefficient vector to the device instead of L
// encoded towers. That claim rests on two things this suite pins down, because neither is provable
// by inspection and both fail silently if wrong (a plaintext that decrypts to noise, not a crash):
//
//   1. THE FORMAT CONVENTION. The host path uploads towers OpenFHE left in EVALUATION format and
//      FIDESlib stores them verbatim (RawCiphertext.cuh's REVERSE is false: "OpenFHE should be
//      bit_reversed"). The device path instead uploads COEFFICIENT-order data and runs FIDESlib's
//      own forward NTT. Those agree only if FIDESlib's NTT maps natural-order coefficients onto
//      exactly the layout OpenFHE's evaluation form already has. DeviceIntt_MatchesOpenFheCoeffs
//      tests that directly, by round-tripping a host-encoded plaintext back through the device INTT
//      and comparing against OpenFHE's own COEFFICIENT-format limbs.
//
//   2. THE ARITHMETIC. Everything from FFTSpecialInv through the biasing is transcribed from
//      CKKSPackedEncoding::Encode, and the per-tower reduction is transcribed from
//      FitToNativeVector. DeviceEncode_MatchesHostEncode_AtEveryLevel encodes the same values both
//      ways and compares every limb of every tower, at every level, so any drift in the rounding
//      rule, the bias constant, the scaling-factor selection or the modular reduction shows up as
//      an exact mismatch rather than as a precision loss someone could argue about.
//
// CUDA-ONLY BY CONSTRUCTION: this is a .cu file, and the root CMakeLists globs .cu sources only
// into fideslib-test, which is built only when FIDESLIB_ENABLE_CUDA is ON. The CPU-only target
// (fideslib-cpu-test) lists its three .cpp suites explicitly and can never pick this up. The
// fixture additionally skips at runtime unless the CUDA backend is the active one, so running the
// binary on a machine without a device reports a skip rather than a failure.

#include <gtest/gtest.h>
#include <openfhe.h>
#undef duration

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "fideslib.hpp"

namespace FIDESlib::Testing {

namespace {

constexpr uint32_t kRingDim = 1 << 13;
constexpr uint32_t kSlots	= kRingDim / 2; // fully packed: the app's layout, and gap == 1
constexpr uint32_t kDepth	= 5;

/// The device payload behind a facade plaintext. Deliberately not a helper on the api: reaching
/// into the backend slot is exactly what a test may do and production code may not.
std::shared_ptr<FIDESlib::CKKS::Plaintext> DevicePlaintext(const fideslib::Plaintext& pt) {
	return std::any_cast<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(pt->device);
}

/// Every RNS tower of a device plaintext, read back through the ordinary store path.
std::vector<std::vector<uint64_t>> ReadBackLimbs(const fideslib::Plaintext& pt) {
	FIDESlib::CKKS::RawPlainText raw;
	DevicePlaintext(pt)->store(raw);
	return raw.sub_0;
}

/// A slot vector with mixed signs and magnitudes. Sign coverage is the point: the biasing
/// (`c < 0 ? bigBound + c : c`) and the ModSub branch that undoes it are the two places where an
/// all-positive test vector would pass while the encoding was wrong for half the real inputs.
std::vector<double> MixedSignValues(uint32_t n) {
	std::vector<double> v(n);
	for (uint32_t i = 0; i < n; ++i) {
		const double mag = 0.001 + 0.9 * static_cast<double>(i % 97) / 97.0;
		v[i]			 = ((i % 3) == 0) ? -mag : mag;
	}
	// Pin the edge cases rather than hoping the pattern above happens to hit them.
	v[0] = 0.0;
	if (n > 1)
		v[1] = -0.0;
	if (n > 2)
		v[2] = 1.0;
	if (n > 3)
		v[3] = -1.0;
	return v;
}

} // namespace

class DeviceEncodeTest : public testing::Test {
  protected:
	fideslib::CryptoContext<fideslib::DCRTPoly> cc;
	fideslib::KeyPair<fideslib::DCRTPoly> keys;

	void SetUp() override {
		if (!fideslib::IsBackendAvailable(fideslib::Backend::CUDA)) {
			GTEST_SKIP() << "device encode is a CUDA-backend arm; this build has no CUDA backend.";
		}

		fideslib::CCParams<fideslib::CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
		params.SetBackend(fideslib::Backend::CUDA);

		cc = fideslib::GenCryptoContext(params);
		cc->Enable(fideslib::PKE);
		cc->Enable(fideslib::KEYSWITCH);
		cc->Enable(fideslib::LEVELEDSHE);
		keys = cc->KeyGen();
		cc->LoadContext(keys.publicKey);
	}
};

// ---------------------------------------------------------------------------------------------
// Anchor 1: the format convention.
//
// Load a plaintext OpenFHE encoded (EVALUATION form, uploaded verbatim), run the device INTT, and
// require the result to equal OpenFHE's own COEFFICIENT-format towers. If FIDESlib's transform
// disagreed with OpenFHE's about which coefficient order corresponds to which evaluation layout,
// this is where it shows — and it is the same convention, used in the other direction, that
// RNSPoly::loadCoefficients depends on when it NTTs uploaded coefficients into place.
// ---------------------------------------------------------------------------------------------
TEST_F(DeviceEncodeTest, DeviceIntt_MatchesOpenFheCoeffs) {
	ASSERT_TRUE(cc->SupportsDeviceEncode());

	const std::vector<double> v = MixedSignValues(kSlots);

	fideslib::Plaintext pt = cc->MakeCKKSPackedPlaintext(v, 1, 0);
	cc->LoadPlaintext(pt);

	// Device: EVALUATION -> COEFFICIENT.
	auto gpu = DevicePlaintext(pt);
	gpu->c0.INTT<FIDESlib::ALGO_SHOUP>(gpu->cc.batch, true);
	FIDESlib::CKKS::RawPlainText raw;
	gpu->store(raw);

	// Host oracle: the same plaintext's element, put into COEFFICIENT format by OpenFHE itself.
	const auto& hostPt		 = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
	lbcrypto::DCRTPoly coeff = hostPt->GetElement<lbcrypto::DCRTPoly>();
	coeff.SetFormat(Format::COEFFICIENT);

	ASSERT_EQ(raw.sub_0.size(), coeff.GetAllElements().size());
	for (size_t t = 0; t < raw.sub_0.size(); ++t) {
		const auto& expect = coeff.GetElementAtIndex(t).GetValues();
		ASSERT_EQ(raw.sub_0[t].size(), expect.GetLength()) << "tower " << t;
		for (size_t i = 0; i < raw.sub_0[t].size(); ++i) {
			ASSERT_EQ(raw.sub_0[t][i], expect[i].ConvertToInt()) << "tower " << t << ", coefficient " << i;
		}
	}
}

// ---------------------------------------------------------------------------------------------
// Anchor 2: the arithmetic, at every level.
//
// The level sweep is not padding. The scaling factor under FLEXIBLEAUTO is read from a per-level
// table of RNS moduli products, not a power of two, so a level-independent test would pass against
// a device encode that used the wrong entry — and the number of towers, which is what the level
// actually controls, changes which primes the reduction runs against.
// ---------------------------------------------------------------------------------------------
TEST_F(DeviceEncodeTest, DeviceEncode_MatchesHostEncode_AtEveryLevel) {
	ASSERT_TRUE(cc->SupportsDeviceEncode());

	const std::vector<double> v = MixedSignValues(kSlots);

	for (uint32_t level = 0; level <= kDepth; ++level) {
		fideslib::Plaintext host = cc->MakeCKKSPackedPlaintext(v, 1, level);
		cc->LoadPlaintext(host);

		fideslib::Plaintext dev = cc->MakeCKKSPackedPlaintextDevice(v, 1, level);

		const std::vector<std::vector<uint64_t>> a = ReadBackLimbs(host);
		const std::vector<std::vector<uint64_t>> b = ReadBackLimbs(dev);

		ASSERT_EQ(a.size(), b.size()) << "tower count differs at level " << level;
		for (size_t t = 0; t < a.size(); ++t) {
			ASSERT_EQ(a[t].size(), b[t].size()) << "level " << level << ", tower " << t;
			for (size_t i = 0; i < a[t].size(); ++i) {
				ASSERT_EQ(a[t][i], b[t][i]) << "level " << level << ", tower " << t << ", coefficient " << i;
			}
		}

		// Metadata has to travel too: the scale is what a later rescale/adjust reads.
		ASSERT_EQ(DevicePlaintext(host)->NoiseFactor, DevicePlaintext(dev)->NoiseFactor) << "level " << level;
		ASSERT_EQ(DevicePlaintext(host)->NoiseLevel, DevicePlaintext(dev)->NoiseLevel) << "level " << level;
		ASSERT_EQ(DevicePlaintext(host)->slots, DevicePlaintext(dev)->slots) << "level " << level;
	}
}

// A device-encoded plaintext is usable where only the backend value is read — the case it exists
// for — and says so clearly where the host value is needed, instead of answering 0.
TEST_F(DeviceEncodeTest, DeviceOnlyPlaintext_RejectsHostReads) {
	ASSERT_TRUE(cc->SupportsDeviceEncode());

	fideslib::Plaintext dev = cc->MakeCKKSPackedPlaintextDevice(MixedSignValues(kSlots), 1, 0);

	EXPECT_TRUE(dev->device.has_value());
	EXPECT_FALSE(dev->host.has_value());
	EXPECT_THROW((void)dev->GetLevel(), lbcrypto::OpenFHEException);
	EXPECT_THROW((void)dev->GetRealPackedValue(), lbcrypto::OpenFHEException);
}

// The refusals are part of the contract: each one marks a branch of OpenFHE's Encode that this path
// does not reproduce, and each has to fail loudly rather than quietly encode something else.
TEST_F(DeviceEncodeTest, RefusesUnsupportedEncodeRequests) {
	ASSERT_TRUE(cc->SupportsDeviceEncode());

	const std::vector<double> v = MixedSignValues(kSlots);

	// noiseScaleDeg > 1 would need the CRT power-up by llround(scalingFactor)^(deg-1).
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDevice(v, 2, 0), lbcrypto::OpenFHEException);
	// A level past the tower chain.
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDevice(v, 1, kDepth + 8), lbcrypto::OpenFHEException);
	// An empty value vector.
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDevice(std::vector<double>{}, 1, 0), lbcrypto::OpenFHEException);
	// A slot count that is not a power of two.
	EXPECT_THROW((void)cc->MakeCKKSPackedPlaintextDevice(v, 1, 0, 3), lbcrypto::OpenFHEException);
}


// ---------------------------------------------------------------------------------------------
// Anchor 3: the BATCH is the singles.
//
// MakeCKKSPackedPlaintextDeviceMany shares one host-to-device copy, one CRT-reduce launch and one
// NTT launch pair across a whole group. None of that is allowed to change a single limb, and none
// of it would fail visibly if it did: a mis-indexed pointer table would hand plaintext i the towers
// of plaintext j, which decrypts to garbage rather than crashing. So compare every limb of every
// tower, element by element, against the single call on the same values.
//
// The batch is deliberately LARGER than the default per-pass cap (FIDESLIB_DEVICE_ENCODE_BATCH_MAX,
// 32) so the chunking loop runs and the short tail chunk is exercised, and it deliberately mixes
// value vectors of different lengths, since the batch resolves ONE `slots` for all of them.
// ---------------------------------------------------------------------------------------------
TEST_F(DeviceEncodeTest, DeviceEncodeMany_MatchesSingleDeviceEncode) {
	ASSERT_TRUE(cc->SupportsDeviceEncode());
	ASSERT_TRUE(cc->SupportsDeviceEncodeMany());

	constexpr size_t kBatch = 37; // > the default cap of 32: one full chunk plus a tail of 5

	std::vector<std::vector<double>> values;
	values.reserve(kBatch);
	for (size_t b = 0; b < kBatch; ++b) {
		// A different vector per element, so a batch that quietly broadcast one member to all of
		// them (or reversed the order) fails rather than passing.
		std::vector<double> v = MixedSignValues(kSlots);
		for (size_t i = 0; i < v.size(); ++i)
			v[i] = v[i] * (1.0 + 0.01 * static_cast<double>(b)) + 0.001 * static_cast<double>(b);
		if (b % 5 == 0)
			v.resize(kSlots / 2); // shorter than `slots`: the zero-fill has to be per element
		values.push_back(std::move(v));
	}

	for (uint32_t level = 0; level <= kDepth; ++level) {
		std::vector<fideslib::Plaintext> many = cc->MakeCKKSPackedPlaintextDeviceMany(values, 1, level);
		ASSERT_EQ(many.size(), values.size()) << "level " << level;

		for (size_t b = 0; b < values.size(); ++b) {
			fideslib::Plaintext one = cc->MakeCKKSPackedPlaintextDevice(values[b], 1, level);

			const std::vector<std::vector<uint64_t>> a = ReadBackLimbs(one);
			const std::vector<std::vector<uint64_t>> c = ReadBackLimbs(many[b]);

			ASSERT_EQ(a.size(), c.size()) << "level " << level << ", element " << b;
			for (size_t t = 0; t < a.size(); ++t) {
				ASSERT_EQ(a[t].size(), c[t].size()) << "level " << level << ", element " << b << ", tower " << t;
				for (size_t i = 0; i < a[t].size(); ++i) {
					ASSERT_EQ(a[t][i], c[t][i]) << "level " << level << ", element " << b << ", tower " << t << ", coefficient " << i;
				}
			}

			ASSERT_EQ(DevicePlaintext(one)->NoiseFactor, DevicePlaintext(many[b])->NoiseFactor) << "level " << level << ", element " << b;
			ASSERT_EQ(DevicePlaintext(one)->NoiseLevel, DevicePlaintext(many[b])->NoiseLevel) << "level " << level << ", element " << b;
			ASSERT_EQ(DevicePlaintext(one)->slots, DevicePlaintext(many[b])->slots) << "level " << level << ", element " << b;
		}
	}
}

// A batched plaintext is an ORDINARY device-encoded plaintext in every other respect, and in
// particular each one owns its own device storage: releasing one must not disturb the rest. If the
// batch were a shared slab this is the case that would fail (or corrupt), which is exactly why it
// is not one.
TEST_F(DeviceEncodeTest, DeviceEncodeMany_ElementsAreIndependentlyReleasable) {
	ASSERT_TRUE(cc->SupportsDeviceEncodeMany());

	std::vector<std::vector<double>> values;
	for (size_t b = 0; b < 8; ++b) {
		std::vector<double> v = MixedSignValues(kSlots);
		v[0]				  = static_cast<double>(b);
		values.push_back(std::move(v));
	}

	std::vector<fideslib::Plaintext> many = cc->MakeCKKSPackedPlaintextDeviceMany(values, 1, 0);
	ASSERT_EQ(many.size(), values.size());

	// Keep a copy of what the survivors must still read back as, then drop half the batch.
	std::vector<std::vector<std::vector<uint64_t>>> expect;
	for (size_t b = 1; b < many.size(); b += 2)
		expect.push_back(ReadBackLimbs(many[b]));

	for (size_t b = 0; b < many.size(); b += 2)
		many[b].reset();

	size_t k = 0;
	for (size_t b = 1; b < many.size(); b += 2, ++k) {
		ASSERT_TRUE(many[b]) << "element " << b << " was released with its neighbours";
		const std::vector<std::vector<uint64_t>> got = ReadBackLimbs(many[b]);
		ASSERT_EQ(got.size(), expect[k].size()) << "element " << b;
		for (size_t t = 0; t < got.size(); ++t) {
			ASSERT_EQ(got[t], expect[k][t]) << "element " << b << ", tower " << t;
		}
	}

	// And the device-only contract is unchanged.
	EXPECT_TRUE(many[1]->device.has_value());
	EXPECT_FALSE(many[1]->host.has_value());
}

} // namespace FIDESlib::Testing
