//
// Created by carlos on 6/03/24.
//
#ifndef FIDESLIB_CKKS_LIMB_CUH
#define FIDESLIB_CKKS_LIMB_CUH

// #include <concepts>
#include "ConstantsGPU.cuh"
#include "VectorGPU.cuh"
#include "forwardDefs.cuh"
#include <iostream>
#include <variant>

namespace FIDESlib::CKKS {

template <typename T> class Limb {
	using LimbImpl = std::variant<Limb<uint32_t>, Limb<uint64_t>>;
	static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

	ContextData& cc;

  public:
	int primeid;
	Stream& stream;
	VectorGPU<T> v;
	VectorGPU<T> aux;

  private:
	const int id;
	const bool raw;
	static constexpr int block = 128;

  public:
	~Limb() noexcept;

	Limb(Limb<T>&& l) noexcept;

	Limb(ContextData& context, const int id, Stream& stream, const int primeid = -1, bool constant = false);

	Limb<T> clone();

	//    Limb(Context &context, const int device, const int primeid = -1);
	Limb(ContextData& context, T* data, const int offset, const int id, Stream& stream, const int primeid = -1, T* data_aux = nullptr, const int offset_aux = 0);
	// void free();

	Global::Globals* getGlobals();

	void add(const LimbImpl& l);

	void add(const Limb<uint64_t>& l);

	void add(const Limb<uint32_t>& l);

	void sub(const LimbImpl& l);

	void sub(const Limb<uint64_t>& l);

	void sub(const Limb<uint32_t>& l);

	void mult(const LimbImpl& l);

	void mult(const Limb<uint64_t>& l);

	void mult(const Limb<uint32_t>& l);

	void mult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace = false);

	template <typename Q> void load(const std::vector<Q>& dat_);

	void load(const VectorGPU<T>& dat);

	void store(std::vector<T>& dat) const;

	/// @brief Device encode, one limb: reduce a shared device-resident coefficient vector mod this
	/// limb's prime.
	///
	/// @param biased      Device pointer to cc.N biased coefficients, in OpenFHE's `temp` encoding
	///                    (a signed integer c is carried as `c < 0 ? bigBound + c : c`). The buffer
	///                    is SHARED by every limb of the polynomial — that sharing is the whole
	///                    point of the arm, since it replaces one H2D per limb with one H2D total.
	/// @param bigValueHf  `bigBound >> 1`; a biased value strictly above it denotes a negative
	///                    coefficient.
	/// @param diff        `bigBound - prime`, unreduced. Subtracted mod prime on the negative branch.
	///
	/// This is OpenFHE's CKKSPackedEncoding::FitToNativeVector for a single tower, so the limb is
	/// left in COEFFICIENT order — the caller still owes the forward NTT (RNSPoly::loadCoefficients
	/// does it). See RNSPoly::loadCoefficients for the full contract and the parity argument.
	void loadCoefficients(const uint64_t* biased, const uint64_t bigValueHf, const uint64_t diff);

	/// @brief Release this limb's NTT scratch (`aux`), keeping the limb and its data (`v`) alive.
	///
	/// For a limb that will never be transformed again. `aux` is scratch for the two-stage NTT/INTT
	/// and for the in-place destination of mult/addMult; a limb holding a PLAINTEXT is read only
	/// through `v` (it is an operand, never a destination, and this library never transforms a
	/// plaintext), which is why a constant-grown limb is allocated with no `aux` at all and every
	/// loadConstant plaintext in the library already runs without one.
	///
	/// STREAM ORDERING IS THE CALLER'S: the release is stream-ordered on `s`, so `s` must already be
	/// fenced against whatever last wrote the scratch. Idempotent, and a no-op on a limb whose `aux`
	/// is a view onto a shared buffer or was never allocated — see VectorGPU::free.
	///
	/// @return the address the partition's device scratch table should now hold for this limb:
	///   nullptr when the scratch was actually released (and when there never was any), or the
	///   unchanged address when `aux` is a view and nothing was freed. Returning it rather than
	///   assuming nullptr is what keeps a caller from nulling a shared buffer's slot.
	void* freeAux(Stream& s);

	template <typename Q> void load_convert(const std::vector<Q>& dat_raw);

	template <typename Q> void store_convert(std::vector<Q>& dat_raw);

	template <ALGO algo = ALGO_SHOUP> void INTT();

	template <ALGO algo = ALGO_SHOUP> void NTT();

	void NTT_rescale_fused(const LimbImpl& l);

	void NTT_rescale_fused(const Limb<uint32_t>& l);

	void NTT_rescale_fused(const Limb<uint64_t>& l);

	void NTT_moddown_fused(const LimbImpl& l);

	void NTT_multpt_fused(const LimbImpl& _l, const LimbImpl& _pt);

	void copyV(const LimbImpl& l);

	void copyV(const Limb<uint32_t>& l);

	void copyV(const Limb<uint64_t>& l);

	void INTT_from(LimbImpl& l);

	void addMult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace = false);

	void printThisLimb(int num = 32) const;

	void INTT_from_mult(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& c1_, const LimbImpl& c1tilde_, const LimbImpl& c0_, const LimbImpl& c0tilde_, const LimbImpl& kska_, const LimbImpl& kskb_);

	void INTT_from_mult_acc(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& c1_, const LimbImpl& c1tilde_, const LimbImpl& c0_, const LimbImpl& c0tilde_, const LimbImpl& kska_, const LimbImpl& kskb_);

	void NTT_and_ksk_dot(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_);

	void NTT_and_ksk_dot_acc(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_);

	void automorph(const int index, const int br);
};

using LimbImpl = std::variant<Limb<uint32_t>, Limb<uint64_t>>;
} // namespace FIDESlib::CKKS

#endif // FIDESLIB_LIMB_CUH