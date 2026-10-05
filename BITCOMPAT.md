# CPU/GPU Bit-Compatibility: Issues and Plan

> **Note:** this document tracks ongoing CPU/GPU bit-compatibility work and may change.

**Goal:** every FIDESlib GPU operation produces ciphertexts that are bit-identical to the
patched-OpenFHE CPU equivalent, verified by exact ciphertext comparison (`ASSERT_EQ_CIPHERTEXT`),
not decrypt-level tolerance. Exact plaintext-level comparison is impossible by design: OpenFHE
adds fresh Gaussian noise to every CKKS decryption (`CKKSPackedEncoding::Decode` noise-flooding
countermeasure), so decrypted values are compared within precision (`ASSERT_ERROR_OK`) only.

**Landing zones.** Changes live in one of three places, in order of preference:

1. **OpenFHE dev branch** — algorithmic changes with standalone value (formulation unification,
   optimizations that also help the CPU).
2. **`deps/fideslib-ref-1.5.1.6.patch`** — kept minimal; visibility/build-config shims only
   (applies on tag `fideslib-ref-v1.5.1.6`). Temporary home for changes queued for upstream.
3. **FIDESlib** — GPU-side fixes and anything that is a FIDESlib bug.

## Current test status (`test/OpenFheCompatTests.cu`)

> **Provenance warning (2026-08-26): the statuses below are NOT currently reproducible in this
> tree.** `test/OpenFheCompatTests.cu` was written on a line that predates the Engine abstraction
> (`e71e08e`, later merged into this tree), where `cc->SetDevices({0})` set `CryptoContextImpl::devices`
> and every call site dispatched on `devices.empty()`. In this tree the engine is fixed at
> `GenCryptoContext` from `CCParams::backend`, which defaults to `Backend::CPU`, and no test in this
> file calls `SetBackend`. `SetCudaDevices` and `LoadContext` are therefore both no-ops
> (`OpenFheEngine.cpp:593-595, 659-660`), so every test computes both sides on the CPU:
> `ASSERT_EQ_CIPHERTEXT` compares OpenFHE against OpenFHE and `ASSERT_ERROR_OK` compares a value
> against itself. The suite runs green in 15.7 s with 12 bootstraps at N=4096, which is itself the
> tell. The results below are real upstream GPU measurements; re-measuring them here requires
> converting the suite to the `ApiParityTest` oracle pattern (`LbCc`/`LbCt`/`HostCt`).

| Test | Status | Notes |
|---|---|---|
| `EvalFastRotation` | PASS | single-index hoisted rotation. **CUDA, 2026-09-18:** the GPU engine used to return `nullptr` from `EvalFastRotationPrecompute` and ignore the handle, so the single-index call fell through to `Ciphertext::rotate` and redid the ModUp per index — hoisting was a no-op on this backend while the CPU path really hoisted (values always agreed; only cost differed). `CudaEngine` now owns a real handle (the ModUp of the source `c1`, an `RNSPoly` of its own, not the context scratch) and `Ciphertext::rotate_precomputed` applies it per index — one index of `rotate_hoisted`. A handle used after its source descends a level THROWS (`std::invalid_argument`) rather than decompose a stale chain; the haze engine falls back to a plain rotate in the same situation. The vector overloads are unchanged. |
| `EvalRotate` | PASS | keyswitch-first path unified onto the hoisted core upstream (O1, `fideslib-ref-v1.5.1.2`) |
| `EvalBootstrap` | PASS | sparse, N=4096, slots=8, FLEXIBLEAUTO, budget {3,3} |
| `EvalBootstrapDense` | PASS | fully packed, slots=N/2 |
| `EvalArithmetic` | PASS | add/sub/negate/mult/square, ct∘ct and ct∘scalar |
| `EvalArithmeticPt` | PASS | ct∘plaintext variants |
| `EvalAdjust` | PASS | 9 mixed noise-degree / level-gap adjustment cases |
| `EvalChebyshev` | PASS | standalone degree-12 Paterson–Stockmeyer |
| `EvalAddMany` | PASS | |
| `AccumulateSum` | PASS | api rotation fold: CPU fallback ≡ GPU `Accumulate` at `PARTIAL_SUM_RADIX` (R11, O8) |
| `EvalFastRotationHoisted` | PASS | multi-index hoisted rotations; re-enabled after O2 turned out to be a test-macro bug |
| `EvalBootstrapLT` | PASS | Tier-2: level budget {1,1}, `isLT`/`EvalLinearTransform` branch (O6) |
| `EvalBootstrapSlots64` | PASS | Tier-2: slots=64 sparse bootstrap (O6) |
| `EvalBootstrapFlexExt` | PASS | Tier-2: FLEXIBLEAUTOEXT bootstrap (O6) |
| `EvalArithmeticFlexExt` | PASS | Tier-2: FLEXIBLEAUTOEXT arithmetic (O6) |
| `EvalArithmeticFixedManual` | PASS | Tier-2: FIXEDMANUAL arithmetic incl. explicit `Rescale` (O6) |
| `EvalChebyshevFixedManual` | PASS | Tier-2: re-enabled after O6a fix (T2km1 level alignment) |
| `EvalBootstrapFixedManual` | PASS | Tier-2: re-enabled after O6b (double-angle rescale placement) + O6a fixes |
| `EvalBootstrapDenseFixedManual` | PASS | fully packed + FIXEDMANUAL combination, gap found after the sweep (O6d) |
| `EvalBootstrapDenseFlexExt` | PASS | combination sweep: fully packed × FLEXIBLEAUTOEXT |
| `EvalBootstrapLTFixedManual` | PASS | combination sweep: level budget {1,1} (LT branch) × FIXEDMANUAL |
| `EvalBootstrapSparseSecret` | PASS (upstream) | combination sweep: SPARSE_TERNARY keys (`g_coefficientsSparse`, bootK=1.0 path). Re-established independently on this tree by `SparseSecretModEval.*` and `ApiParitySparseSecretBootstrapTest` - see R13 |
| `EvalArithmeticFixedAuto` | PASS | combination sweep: FIXEDAUTO arithmetic (same-level ops) |
| `DISABLED_EvalBootstrapFixedAuto` | **disabled** | combination sweep finding O6e: FIXEDAUTO add/sub operand adjustment missing on GPU |
| `DISABLED_EvalBootstrapSparseEncaps` | **disabled** | Tier-2 finding O6c: encapsulation designs differ structurally (dual-context GPU vs in-context stock) |

## Resolved issues

### R1. `EvalRotate`/`EvalAtIndex` vs `EvalFastRotation` are different formulations
The hoisted formulation (`EvalFastRotation` HYBRID branch: extended key-switch → fold `c0·P` →
automorph → `ApproxModDown`) and the KeySwitch-first formulation (`EvalRotate`: key-switch,
mod-down, add to `c0`, automorph last) produce different-but-equally-valid ciphertexts — the
key-switching noise enters automorphed in one and un-automorphed in the other, so they never
match bitwise. FIDESlib implements only the hoisted flavor (verified: GPU `rotate` == CPU
`EvalFastRotation` exactly, incl. at 26 towers).

**Interim fix (was in the patch):** the bootstrap's 4 `EvalRotate`/`EvalAtIndex` sites
(PartialSum loop, final sparse doubling, CtS/StC correction rotations) switched to
`EvalFastRotation`; `FHECKKSRNS::Conjugate` rewritten in the hoisted form with
`autoIndex = 2N−1`.

**Resolution (O1, `fideslib-ref-v1.5.1.2`):** `EvalRotate`/`EvalAtIndex`/`Conjugate` are routed
through the hoisted core upstream, so the original bootstrap call sites are bit-compatible as
written — the five interim hunks were reverted to stock upstream code and dropped from the
patch (they were never upstreamed as call-site changes).

### R2. `Accumulate` lazy extended-basis accumulation *(deliberate GPU optimization — perf recovered via P2b)*
The GPU PartialSum kept `c0` in the Q·P basis across all doubling levels with one deferred
mod-down. `ApproxModDown` rounds, so `moddown(Σx) ≠ Σ moddown(x)`; OpenFHE's per-rotation
mod-down can never match.

**Interim fix (FIDESlib `AccumulateBroadcast.cu`):** sequential full rotations, per-rotation
mod-down. Perf cost ≈ 7 extra `c0` mod-downs per bootstrap at the raised level (the hoisting
itself was moot: bootstrap uses `accumulate_bStep = 2`, one rotation/level).

**Resolution (P2b, `fideslib-ref-v1.5.1.3`):** the CPU PartialSum is lazy too
(`FHECKKSRNS::EvalPartialSumInPlace`, see P2b), and the lazy `Accumulate` was restored in
FIDESlib.

### R3. `LinearTransform` `ONLY_C1` lazy mod-down *(deliberate GPU optimization — perf recovered via P2c)*
Horner giant-step and correction rotations mod-downed only `c1`, keeping `c0` extended; CPU
`EvalHornerGiantRotate` does a full `KeySwitchDown` (with exact `·PModq` re-lift) per giant step.

**Interim fix (FIDESlib `LinearTransform.cu`):** `ONLY_C1 = false`. Perf cost ≈ 1 extra `c0`
mod-down per giant step per CtS/StC level.

**Resolution (P2c, `fideslib-ref-v1.5.1.3`):** `EvalHornerGiantRotate` settles only `c1` and
folds the extended `c0` directly into the key-switch product; the CtS/StC slot-ordering
corrections fold into the last level's extended output; `ONLY_C1` is back to `true` (see P2c).
The level-end `KeySwitchDown` is unchanged and settles `c0` once per level. FIDESlib's extended
representation is ·P-scaled exactly like the key-switch product, so the direct fold is
bit-compatible by construction (validated by the suite, no bisect cycle needed).

### R4. GPU Chebyshev was a stale port (values wrong, not just slower)
`evalChebyshevSeries`/inner PS ported an older OpenFHE: mutated shared powers `T[i]` in place
(sticky adjustments), skipped the `AdjustLevelsAndDepthInPlace` power alignment, and the inner
recursion didn't match the current qu/su/cu construction.

**Resolution (FIDESlib `ApproxModEval.cu`):** line-for-line transcription of current OpenFHE
(clone-based operand adjustment, power alignment, transcribed inner PS +
`EvalPartialLinearWSum`). Locked in by `EvalChebyshev` + both bootstrap tests.

### R5. `adjustScaleAndLevel` deg2→deg2 branch mismatched `AdjustLevelsAndDepthInPlace`
Three separate deltas in `Ciphertext.cpp`: floating-point evaluation order of the adjustment
factor (`scf2*q1/scf1/scf` vs OpenFHE's `scf2/scf1*q1/scf` — differs in ULPs, changes the encoded
scalar), the mod-reduce factor index (target+1 instead of the current top level), and
drop-before-rescale instead of rescale-before-drop. Locked in by `EvalAdjust`.

### R6. `LevelReduceInPlace` is a no-op outside FIXEDMANUAL
OpenFHE only drops levels under FIXEDMANUAL; the GPU inner PS transcription initially dropped a
real level for FLEXIBLE modes. Fixed with a FIXEDMANUAL gate (`su`/`cu` handling in inner PS).

### R7. Double-angle scalar floating-point form
Must be `-pow(2π, -2^i)`; `-1.0/pow(2π, 2^i)` is not FP-identical. Fixed in
`applyDoubleAngleIterations`.

### R8. Missing/misplaced deg-2 rescale before Chebyshev
CPU `EvalBootstrap` runs `ModReduceInternalInPlace` (deg2→deg1) on the ciphertext(s) before
`EvalChebyshevSeries`. Mirrored in `Bootstrap.cu` for the sparse branch and the dense branch
(both `ctxtEnc` and `ctxtEncI`).

### R9. api `EvalNegate` implemented as `multScalar(-1.0)`
Bumped noise degree to 2 and consumed scale — never equivalent to OpenFHE's exact negation, and
silently cost callers a level.

**Resolution (`api/CryptoContext.cpp`):** exact per-limb multiply by `q_i − 1`; no metadata
change. Applies to `EvalNegate` and `EvalNegateInPlace`.

### R10. api `bad any_cast` in ct×pt CPU fallbacks
`EvalMult(ct, pt)` / `EvalMultInPlace(ct, pt)` cast `pt->cpu` to `ConstPlaintext` where it holds
`Plaintext` — an `std::any` cast must name the stored type exactly, so any CPU-path ct×pt
multiply threw.

**Resolution (`api/CryptoContext.cpp`, both call sites):** cast to the stored type:

```diff
-		auto& ptImpl  = std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
+		auto& ptImpl  = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
```

This failure mode is inherent to the `std::any` design — see O7.

### R11. api `AccumulateSum` CPU fallback mismatched the GPU accumulation
The GPU path runs `FIDESlib::CKKS::Accumulate(ct, bStep=4, stride, slots)` — radix-4 levels,
each sharing one digit decomposition across up to 3 rotations, with the lazy extended-basis
`c0` accumulation — while the CPU fallback was an eager sequential doubling loop
(`EvalRotate` + `EvalAddInPlace`): a structurally different accumulation order, so the two
paths never matched bitwise. Additionally, the GPU path narrows the FIDESlib `slots` field to
`stride` after a full fold (its broadcast convention), which leaked through api `GetSlots` and
made the GPU result decode at a different length than the CPU result — a pre-existing
api-visible inconsistency.

**Resolution (`fideslib-ref-v1.5.1.4`):** `FHECKKSRNS::EvalPartialSumInPlace` provides a static
`(ct, stride, size)` radix-2 helper (doubling; one rotation per level, folded directly into the
extended accumulator) plus a general `(ct, stride, size, radix)` form that delegates to the
radix-2 helper when `radix == 2`; both mirror FIDESlib's `Accumulate` level/index structure
bit-for-bit. A single api constant `ACCUMULATE_SUM_RADIX` (since bound to OpenFHE's
compile-time `PARTIAL_SUM_RADIX` — see O8) drives both sides: the CPU
fallbacks call the general form with it, and the GPU calls `Accumulate(*, ACCUMULATE_SUM_RADIX,
…)` (FIDESlib's existing `bStep` parameter — no new GPU code). The api GPU paths also restore
the OpenFHE-visible slot count after the fold (whether that restore belongs in `Accumulate`
itself is an open decision — see O10). Radix-2 keeps the rotation-key footprint at the
power-of-two set (see O8, radix/key-size tradeoff — key minimization drove this choice).
Acceptance test: `OpenFHECompatTests.AccumulateSum`. The `start`-offset variant
(`AccumulateCascadeImpl`, which settles both elements per level) still uses `bStep=4` on the GPU
and keeps its eager doubling fallback — deferred (see O8).

### R12. OpenFHE's keyed EvalMult/EvalSquare silently drops components past cv[2]
`LeveledSHEBase::EvalMult`/`EvalSquare` overloads that take an `evalKey`
(`deps/openfhe-src/src/pke/lib/schemebase/base-leveledshe.cpp:202-215`) key-switch only `cv[2]`
and resize the element vector back to 2 — so a ciphertext that already carries 3+ elements
(e.g. a degree-2 result fed into another `EvalMult`) loses everything past `cv[2]` with no
error, just an undecryptable output.

**Resolution (`api/CryptoContext.cpp`):** the facade's `RequireDegree1` guard throws
`"<op>: ciphertext should be relinearized before (has N elements; call Relinearize)"` on any
non-degree-1 operand, on every backend including CPU, so callers never reach OpenFHE's silent
drop.

### R13. CUDA rejected SPARSE_TERNARY, and the GPU context cache aliased boot configurations
Two coupled defects, both fork-only (introduced by `ba11c3e`, the Engine-abstraction refactor):

1. `api/GenCryptoContext.cpp` threw for `Backend::CUDA` + `SPARSE_TERNARY` ("CUDA needs a uniform
   main key"). The premise was wrong. The GPU never sees a secret key: sparse-awareness reaches it
   only as the Chebyshev table + K + R selected in `GetRawParams`
   (`RawCiphertext.cu:604-612`) and as OpenFHE's precomputed CtS/StC plaintexts, both consumed as
   data. `SPARSE_TERNARY` also sets `sparse_encaps = false`, so it needs no secondary context and no
   a->b/b->a keys - strictly simpler than `SPARSE_ENCAPSULATED`, which CUDA already accepted. The
   guard also broke every `examples/resnet` preset (all six use `SPARSE_TERNARY`), plus
   `examples/bootstrap` and `examples/hpca/solutions/04`.

2. `Parameters::operator<` (`src/CKKS/Parameters.cuh`) ordered on ring dimension, level count, dnum,
   scaling technique and moduli, then compared only `raw.has_value()`. Contexts differing *only* in
   boot configuration therefore compared equal, and `GenCryptoContextGPU` returned the cached one.
   Demonstrated, not inferred: with two contexts alive, the sparse one reported
   `GetCoeffsChebyshev().size() == 89` and `GetDoubleAngleIts() == 6` - it had been handed the
   uniform context - while `GetRawParams` returned the correct 45/3 in the same run. This is exactly
   the "silent downgrade" the guard's comment feared, and `~CudaEngine`'s deregistration hides it
   whenever contexts are strictly sequential.

**Resolution.** The comparator now also orders on `coefficientsCheby.size()`, `bootK`,
`doubleAngleIts` and `sparse_encaps` - a tuple that discriminates all five `BOOT_CONFIG` values -
and the guard is deleted.

**Evidence the GPU genuinely runs the shorter approximation.** A sparse ternary key has low Hamming
weight, so the mod-reduction argument range is smaller and the Chebyshev approximation shrinks:
degree 88 -> 44 and R_UNIFORM=6 -> R_SPARSE=3, i.e. `GetDepthByDegree` 8+6=14 mod-eval levels
against 7+3=10. Measured level of the refreshed ciphertext (level budget {3,3}, depth 25, N=4096):

| backend | uniform | sparse | delta |
|---|--:|--:|--:|
| CUDA | 19 | 15 | 4 |
| haze (local target) | 19 | 15 | 4 |
| CPU | 16 | 12 | 4 |

The delta matches the table difference on all three backends, and haze reproduces CUDA's absolute
levels exactly. The absolute offsets differ because
`OpenFheEngine::bootstrapSetupPolicy` routes `modEvalLevels` into OpenFHE's `BTSlotsEncoding` slot
and passes `-1` for `modevallevels`, so `ckksrns-fhe.cpp:261-269` derives `lDec` from the budget
alone and ignores the approximation depth - a pre-existing divergence, flagged in place, that shifts
both arms equally.

**haze counts the saving directly.** Because HazeEngine records FHETCH IR rather than computing, the
trace length is the work. Same test run, same parameters, only the key distribution differing:

| secret key | FHETCH instructions |
|---|--:|
| `UNIFORM_TERNARY` | 199,588 |
| `SPARSE_TERNARY` | 181,883 |
| difference | **-17,705 (-8.9%)** |

Attribution is checked rather than assumed: the sparse-only test emits exactly 181,883, matching the
second trace of the pair. `HazeBootstrap.cpp:132-136` is the selection site (`g_coefficientsSparse`,
`k = 1.0` pre-divided, `numIter = R_SPARSE`). Sparse parity against the OpenFHE sparse oracle also
passes on haze at N=2^16 (264,637 instructions, 0 errors).

Full-suite state after the change, all 145 api-level tests:

| backend | passed | skipped | failed |
|---|--:|--:|--:|
| haze (local) | 142 | 3 | 0 |
| CUDA | 133 | 10 | 2 |
| CPU | 93 | 37 | 0 |

The 2 CUDA failures are `ApiParityTest.EvalMult`/`EvalSquare`, the relinearization key-switch
rounding baseline at `test/ApiParityTest.cpp:212-239`. haze's 3 skips are the FLEXIBLEAUTO and
FLEXIBLEAUTOEXT fully-packed parity cases the in-process replay bridge declines; CUDA skips all 5
fully-packed cases outright, so haze covers two configurations the GPU backend does not.

Note when selecting these tests: `ApiParityBootstrapTest` and `ApiParityFullPackBootstrapTest` are
parametrized, so gtest names them `LevelBudgets/...`. A `--gtest_filter='Api*'` silently misses all 8
of them plus `KeyAccessTest` and `NoEvalMultKeyTest` - 15 tests of 145.

Tests: `ContextCacheBootConfig.SparseAndUniformDoNotAlias` (`test/ContextCacheTests.cu`),
`SparseSecretModEval.*` (`test/ApiTests.cpp`), `ApiParitySparseSecretBootstrapTest`
(`test/ApiParityTest.cpp`, bit-level agreement with a sparse OpenFHE oracle at N=2^16).

### O1. `EvalRotate` unification *(resolved — landed as `fideslib-ref-v1.5.1.2`)*
The hoisted HYBRID formulation (already used by `EvalFastRotation`, added upstream for the GPU
backend) is extended to `EvalRotate`/`EvalAtIndex`/`EvalAutomorphism`/`Conjugate`, making
`EvalRotate(ct, i)` ≡ `EvalFastRotation(ct, i, m, EvalFastRotationPrecompute(ct))` bit-for-bit:
`EvalAutomorphism` and `EvalFastRotation` both delegate to a new `EvalAutomorphismCore`
(`base-leveledshe.cpp`) holding the former `EvalFastRotation` body, and `FHECKKSRNS::Conjugate`
delegates to `EvalAutomorphism(ct, 2N−1)`. Side effect: upstream `EvalRotate` output bit-patterns
change (same decrypted values and noise magnitude). Landed as commit `eb3e76cf`,
tagged `fideslib-ref-v1.5.1.2`; the patch rebased onto it (R1 hunks dropped).
`OpenFHECompatTests.EvalRotate` was the acceptance test and is green, and the api
`AccumulateSum` CPU fallback (then an `EvalRotate`+add doubling loop, since replaced by the
shared radix-2 helper — see R11) and `EvalRotateInPlace` became bit-compatible with it.

### O2. "Heap corruption" in multi-index `EvalFastRotation` *(resolved — test-harness bug, no memory bug exists)*
The `DISABLED_EvalFastRotationHoisted` reproducer crashed with what looked like a clobbered
`CiphertextImpl` (garbage jump through the `std::any` manager pointer), suspected to be a
pre-existing FIDESlib heap corruption requiring `compute-sanitizer` on native Linux.

**Resolution:** a host ASAN build (per P3) reported a heap-buffer-overflow *read* — not a
write — and the culprit was the test harness itself: `ASSERT_EQ_CIPHERTEXT`'s inner tower loop
was named `i`, shadowing the caller's loop variable inside the macro's textual argument
re-evaluations, so `ASSERT_EQ_CIPHERTEXT(cRots[i], gRots[i])` re-indexed the 2-element vectors
with the tower index (0..2) and read one `shared_ptr` slot past the buffer — a garbage object
pointer, hence the "clobbered" any manager, the layout dependence, and the silence of every
hardware watchpoint placed on real objects. No FIDESlib or api memory bug exists; rotation
values were bit-exact all along. Fixed by evaluating the macro arguments exactly once into
locals and renaming the loop variables (`test/ParametrizedTest.cuh`); the test is re-enabled
and green. Every other test was unaffected because this was the macro's only call site passing
indexed expressions as arguments.

## Open issues

### O3. api `Rescale` semantic divergence under AUTO scaling techniques — RESOLVED (device semantics)
Was: CPU fallback called `context->Rescale` (a no-op for FLEXIBLE*/FIXEDAUTO); the GPU path
performs a real rescale. Resolved by aligning the CPU engine on the device semantics: api
`Rescale` mod-reduces under EVERY scaling technique (towers−1, NSD−1 guard ≥1,
sf ÷= ModReduceFactor), routed through OpenFHE's `ModReduceInternalInPlace` for the AUTO
techniques (`OpenFheEngine.cpp modReduceOneLevelInPlace`). Rationale: 2 of 3 engines already
implemented the eager semantics; a caller hoisting one real rescale out of a ct×pt hot loop
(replacing OpenFHE's per-multiply hidden operand adjust, a significant cost in circuits
dominated by ct×pt products) needs it to be expressible on CPU; and `SetLevel`-style test
code already assumed Rescale consumes a level. O12 resolves with it (same mechanism).

### O4. Chebyshev range-transform prelude not aligned
For bounds other than (−1, 1), the GPU applies add-then-mult on the input; CPU builds
`T[0] = EvalMult(x, α); ModReduce; EvalAddInPlace(−1−β)`. Different rounding order. All current
tests use (−1, 1). Align the GPU prelude to the CPU order when needed.

### O5. Chebyshev degree < 5 takes `EvalChebyshevSeriesLinear` on CPU
The GPU has no mirrored linear path. Only matters if low-degree series are used through the api.

### O6. Tier-2 configurations *(swept; O6a/O6b/O6d fixed, O6c/O6e open)*
Each configuration now has a dedicated test in `OpenFheCompatTests.cu`. Of the five findings,
O6a, O6b, and O6d are fixed and their tests in place; O6c and O6e remain as `DISABLED_`
reproducers (the acceptance tests for their fixes). O6d was found after the sweep — the sweep varied one configuration axis at a time and missed the dense×FIXEDMANUAL
combination. That prompted a systematic **combination sweep** (packing × technique × budget ×
key distribution): dense×FLEXIBLEAUTOEXT, LT{1,1}×FIXEDMANUAL, SPARSE_TERNARY keys, and
FIXEDAUTO arithmetic all came back bit-exact with no changes; FIXEDAUTO bootstrap is O6e.
Known remaining coverage gaps: dense×LT{1,1} needs ≈2·slots raised-level LT plaintexts resident
(several GB at N=4096 fully packed) and could not run on a 4 GB GPU. **Stale as of
2026-08-26:** it is testable on a GPU with 32 GB; iterated bootstrap (`numIterations > 1`) and StC-first setups are api dispatch
gaps, not testable combinations (see O11).

**Bit-compatible with no changes needed (green):**
- Bootstrap level budget {1,1} → `isLT`/`EvalLinearTransform` branch (`EvalBootstrapLT`) —
  also exercises P2c's lazy giant steps in the LT path.
- slots=64 sparse bootstrap (`EvalBootstrapSlots64`) — different PartialSum depth + CtS/StC splits.
- FLEXIBLEAUTOEXT arithmetic and full bootstrap (`EvalArithmeticFlexExt`, `EvalBootstrapFlexExt`)
  — the extra-level ModRaise handling matches.
- FIXEDMANUAL arithmetic including an explicit `Rescale` (`EvalArithmeticFixedManual`) — the
  raw ModReduce path is bit-exact.

**Findings:**
- **O6a — FIXEDMANUAL Chebyshev formulation divergence** *(fixed)*
  (`EvalChebyshevFixedManual`, re-enabled): values agreed to ~1.7e-14 on both sides but the
  raw outputs differed bitwise. A checkpoint trace (coefficient fingerprints at every aligned
  stage: T powers, alignment, T2 powers, T2km1, per-depth qu/su/res/fin) showed **every stage
  bit-identical except `T2km1`** — including a one-tower level mismatch. Root cause: in the
  `T2km1` update loop the operands have mismatched levels; stock's `EvalMult` aligns levels
  internally (exact tower drop) before multiplying, while FIDESlib's `mult` does not under
  FIXEDMANUAL, so the relinearization key-switch ran one tower higher and rounded differently
  in every tower. Stock does this explicitly and by contract: `LeveledSHERNS::EvalMult` calls
  `AdjustForMultInPlace` (→ `AdjustLevelsInPlace` under FIXEDMANUAL — a pure tower drop, no
  rescale) whenever operand levels or tower counts differ, and `EvalMultCore` opens with
  `VerifyNumOfTowers`, rejecting misaligned operands outright. **Fixed in
  `Ciphertext::mult` itself** (FIXEDMANUAL-gated `dropToLevel(b.getLevel())` when `this` is
  deeper), mirroring the stock wrapper and matching the convention FIDESlib's `add`/`sub`
  already follow — so every FIXEDMANUAL mult call site is covered, not just T2km1. This also
  greens the FIXEDMANUAL bootstrap (its only remaining divergence was T2km1 propagating).
- **O6b — FIXEDMANUAL bootstrap livelock** *(fixed)*
  (`EvalBootstrapFixedManual`, re-enabled): the GPU phase effectively hung. Live stack:
  `Bootstrap → approxModReductionSparse → applyDoubleAngleIterations → addScalar →
  ElemForEvalAddOrSub → CRTMult`. Root cause: the GPU double-angle loop rescaled at the
  *start* of each iteration where stock's `ApplyDoubleAngleIterations` ends each iteration
  with `ModReduceInPlace` (a real rescale only under FIXEDMANUAL). On the deg-1 series output
  the start-rescale underflowed the noise degree to 0, blowing up the host-side big-integer
  scalar encoding. **Fixed** by moving the FIXEDMANUAL rescale to the iteration end in
  `applyDoubleAngleIterations` (`ApproxModEval.cu`) — FLEXIBLE paths untouched (the gate never
  fired there). This turned the livelock into a fast exact-compare failure whose cause was
  O6a upstream; with both fixes in, the test passes.
- **O6c — SPARSE_ENCAPSULATED structural pipeline divergence** *(investigation in progress)*
  (`DISABLED_EvalBootstrapSparseEncaps`): initially observed as a 9th-digit scaling-factor
  mismatch, but that is a symptom. Probing data and metadata separately shows the outputs are
  at **different levels entirely** (CPU 7 towers vs GPU 11 — the GPU pipeline consumes ~4
  fewer levels) with genuinely different data. Ruled out: the Chebyshev coefficients (the GPU's
  hardcoded degree-32 list is identical to `g_coefficientsSparseEncapsulated`) and the
  double-angle count (R_SPARSE both). Root difference found: the two encapsulated-bootstrap
  implementations are **different designs**. Stock does the dance in-context
  (`KeySwitchSparse` via the `2N−4` key, raise, switch back via `2N−2`); FIDESlib builds a
  **second "switchable" GPU context** (`createSwitchableContextBasedOnContext`, stored as
  `sparse_context` in the bootstrap precomputation), moves the ciphertext across contexts with
  `reinterpretContext`, key-switches there, and hand-sets `NoiseFactor = targetSF` afterward —
  same key material, different parameter regime and level trajectory. `TODO`s in the code
  ("1/32 will be pre-applied with OpenFHE v1.4, remove the flag") indicate the GPU path targets
  a planned upstream convention rather than the current fideslib-ref one. Worse: at this test's
  configuration the GPU output **decodes to ≈ zero** (all slots ~1e-14 where {0.25…5.0} were
  expected) — the GPU ENCAPS path is functionally broken here, not merely bit-divergent
  (whether it works at the parameters it was designed for was untestable on a 4 GB GPU; a
  32 GB GPU can test it).
  The divergence originates in host-side control flow (dual-context routing, level bookkeeping,
  hand-set `NoiseFactor`), not in device kernels.

  **Fix deferred.** Stock's in-context design is the reference — its output is correct, and
  the dual-context shape does not fit OpenFHE's single-context model.
  `DISABLED_EvalBootstrapSparseEncaps` stays as the acceptance test. What the fix entails, in
  dependency order:

  1. **A GPU `keySwitchSparse` primitive.** Stock's `FHECKKSRNS::KeySwitchSparse` is *not* the
     standard hybrid key switch: it operates on tower 0 only, over a two-prime basis (q₀, p)
     taken from the eval key's params. Steps: extend `c1` from q₀ to (q₀, p) by a
     coefficient-domain `SwitchModulus` round-trip; multiply the extended `c1` by the key's
     B/A vectors; exact mod-switch back to q₀ as `(x_q − convert(x_p)) · p⁻¹ mod q₀` per
     component; fold component 0 into `c0`. All single-limb — a small kernel sequence over
     existing `Limb` ops (NTT/INTT, `SwitchModulus`-style rebase, pointwise mult, the p⁻¹
     fold). Bit-compat requires matching stock's rounding exactly, so transcribe rather than
     re-derive.
  2. **Key loading into the main context.** `AddBootstrapPrecomputation`'s ENCAPS block
     currently creates the switchable context and loads the `2N−2` key into it (`ksk_atob`)
     and `2N−4` into the main one (`ksk_btoa`). Replace with: load `2N−4` as the sparse-switch
     key — note its shape is the two-prime (q₀, p) key, so `KeySwitchingKey::Initialize` needs
     a variant for that layout, it is not a standard dnum-digit hybrid key — and `2N−2` as an
     ordinary hybrid key at raised params, both in the main context. Delete
     `createSwitchableContextBasedOnContext`, the `sparse_context` member in
     `BootstrapPrecomputation`, `Add/GetSecretSwitchingKey`, and the `reinterpretContext`
     round-trips (grep for other users first).
  3. **Rewrite the ENCAPS branches in `Bootstrap.cu` to stock's op order.** Stock (ckksrns-fhe
     ModRaise): (i) `KeySwitchSparse(raised, key@2N−4)` at the *input* level, before the raise;
     (ii) the raise itself (tower-0 reinterpret at raised params — same as the non-ENCAPS
     path); (iii) plain `KeySwitchInPlace(raised, key@2N−2)` at the *raised* level (the
     existing generic GPU `keySwitch` works here). The GPU currently also places the
     switch-back *after* `multScalar(constantEvalMult)`; stock switches back before any
     post-raise scaling — reorder to match. Remove the `ctxt.NoiseFactor = targetSF` hand-set:
     with the in-context dance the scale factor evolves exactly as stock's and needs no
     override.
  4. **Constant conventions.** `RawParams` ENCAPS: `bootK` 16.0 → 1.0 (re-enable the
     commented-out line; the CPU uses k = 1.0 because the 1/K division is baked into the CtS
     precomputation, and the GPU consumes those same CPU-precomputed matrices via
     `AddBootstrapPlaintexts`). Delete the commented `/32` variant of `constantEvalMult` and
     its "OpenFHE v1.4" TODO — under the current fideslib-ref convention it is not needed.
     Reconcile or remove the unused `ENCAPS_2` config the same way. `doubleAngleIts` stays
     `R_SPARSE`.
  5. **Level/metadata bookkeeping.** The observed 4-level trajectory gap and the scf mismatch
     should disappear once the dance is in-context; verify api `GetLevel`/`original_level`
     handling needs no ENCAPS special-casing afterward.
  6. **Validation.** Re-enable the reproducer; if bits still differ, fingerprint-checkpoint the
     three dance steps (post-sparse-switch, post-raise, post-switch-back) against CPU prints at
     the corresponding `ckksrns-fhe.cpp` lines — the ModRaise stage is now the only place left
     to diverge.

  Open questions for the FIDESlib maintainers before starting: whether the dual-context design
  exists for performance (the sparse-phase key switch at reduced parameters) or as groundwork
  for a planned upstream convention (the "v1.4" TODOs), and whether anything else depends on
  the switchable-context machinery. Independent of bit-compat, the ≈zero output at small
  parameters is a functional bug worth reporting to them as-is.

  **2026-09-02 — a constraint the fix plan above has to respect, and one blocker it does NOT
  cover.**

  *The constraint.* Item 2's premise is that the switching keys come from stock's
  `EvalBootstrapKeyGen`, which only emits them when the HOST `CCParams` carry
  `SPARSE_ENCAPSULATED`. Flipping that mapping (`api/CCParams.cpp`) is not free: stock's
  `KeyGenInternal` (`src/pke/lib/schemebase/base-pke.cpp`) puts `SPARSE_TERNARY` and
  `SPARSE_ENCAPSULATED` in the **same case** and draws the MAIN secret at Hamming weight 192
  for both — OpenFHE's own README says so ("Hamming weight of 32 for the key used in
  bootstrapping and 192 for other operations"). FIDESlib maps `SPARSE_ENCAPSULATED` to
  `UNIFORM_TERNARY` on the host params precisely to avoid that: it wants a **uniform** main
  key with the sparse h=32 key confined to the mod-raise, which is a strictly stronger lattice
  instance than stock's h=192 main key and is the whole reason a deployment adopts
  encapsulation. So the in-context rewire must ALSO keep the main key uniform — e.g. by
  generating the key pair with the dist temporarily set to `UNIFORM_TERNARY`, or by patching
  `KeyGenInternal`. Landing item 2 without that is a silent security regression, not a
  refactor. Item 4 (`bootK` 16 → 1.0) is likewise conditional on the same flip: it is wrong
  while the host params say `UNIFORM_TERNARY`, because then the CPU-side CtS precomputation
  does not bake the 1/K in and the GPU has to apply it.

  *The blocker O6c does not cover.* Encapsulation was unusable on the split client/server
  topology for a reason unrelated to the dual-context design: the switching keys were
  generated ONLY inside `GenBootstrapKeys`'s `SSE` block, in `src/` — code that is compiled
  only when `FIDESLIB_ENABLE_CUDA` is ON. A client that key-gens on the CPU backend (the
  normal deployment: the server holds no secret key) therefore could not produce them at all,
  and the server could not either. Fixed 2026-09-02 by moving that pure-host OpenFHE code to
  `api/engine/SparseEncapsulation.{h,cpp}` (always compiled) and calling it from
  `OpenFheEngine::evalBootstrapKeyGen` as well as from `GenBootstrapKeys`. Same code, same
  call order, so the key material is unchanged; `ParameterSwitch.cuh` is now a shim over the
  new header. The keys land at 2N-2 / 2N-4 in the process-global automorphism map, which is
  what `SerializeEvalAutomorphismKey` writes, so they reach a CUDA server through the ordinary
  eval-key stream with no new format. This is orthogonal to the fix plan above: it makes the
  EXISTING dual-context GPU path reachable from a CPU client, and stays correct if the plan is
  later carried out.
- **O6d — dense + FIXEDMANUAL leftover tail rescale** *(fixed)*
  (`EvalBootstrapDenseFixedManual`, new test; found after the sweep): the fully-packed bootstrap
  under FIXEDMANUAL produced bit-different outputs with **identical metadata** (same towers,
  noise degree, scaling factor — only the data differed), and decrypt-level results were wrong.
  Root cause: the dense tail of `approxModReduction` (`ApproxModEval.cu`) carried a
  FIXEDMANUAL-gated `rescale()` after `multIntScalar(post)` — original FIDESlib code that
  compensated for the pre-O6b double-angle placement (which exited at noise degree 2). After
  O6b moved the double-angle rescale to iteration end (output degree 1), this became a second
  rescale on a degree-1 ciphertext, underflowing it to degree 0. Stock has no counterpart (its
  pre-StC `ModReduceInternalInPlace` is gated on `st != FIXEDMANUAL`), and the sparse tail
  (`approxModReductionSparse`) never had the rescale — which is why the sparse FIXEDMANUAL
  test stayed green after O6b. The metadata matched because the GPU CtS/StC rescales are
  degree-gated (`if (NoiseLevel == 2) rescale()`): the first StC level's mult only reached
  degree 1, its rescale didn't fire, and the skipped rescale exactly cancelled the extra one —
  same final level, wrong rounding trajectory. **Fixed** by deleting the tail rescale; the
  dense FIXEDMANUAL tail now matches the sparse tail and stock. All FIXEDMANUAL tests plus
  dense FLEXIBLEAUTO revalidated green. Test note: at this toy ring size the slot-dependent
  default correction factor lands at 9, below `deg = log2(2^60/2^50) = 10`, which
  `EvalBootstrap` rejects outright — the test passes an explicit correction factor of 10
  (production-scale parameters don't trip this check).

- **O6e — FIXEDAUTO add/sub operand adjustment missing** *(open; combination sweep)*
  (`DISABLED_EvalBootstrapFixedAuto`, the acceptance test): FIXEDAUTO bootstrap produces
  bit-different outputs with identical metadata (the O6d signature — data-only divergence),
  while FIXEDAUTO *arithmetic* (`EvalArithmeticFixedAuto`, same-level operands) is bit-exact.
  Root-cause hypothesis, from the gate audit: stock `LeveledSHERNS::AdjustForAddOrSub` runs
  `AdjustLevelsAndDepthInPlace` for **every technique except FIXEDMANUAL/NORESCALE — FIXEDAUTO
  included** — but the GPU add/sub adjustment gates in `Ciphertext.cpp` fire only for
  `FLEXIBLEAUTO || FLEXIBLEAUTOEXT` (the result-metadata blocks and the `skip_adjust` dispatch;
  by contrast `mult`'s gates correctly include FIXEDAUTO). So FIXEDAUTO adds/subs of
  mixed-level/degree operands — which the bootstrap does constantly (Chebyshev su/cu folds,
  Horner accumulation, conjugate combine) but the arithmetic test never does — skip the operand
  adjustment stock performs. Fix direction: widen those gates to match stock's contract
  (everything except FIXEDMANUAL), then check the adjustment arithmetic degenerates correctly
  for FIXEDAUTO's fixed scaling-factor table. Touches shared add/sub core paths — full-suite
  revalidation required.
### O7. api type-erasure design (`std::any`) — decide a direction
The api layer stores its OpenFHE/FIDESlib objects type-erased (`std::any cpu/gpu/pimpl`,
`shared_ptr<void>` GPU registry) so the public headers carry no OpenFHE or CUDA includes.
The types are fully known at compile time inside the `.cpp` files — the erasure exists only for
the header boundary, because OpenFHE's public types are backend-configured alias templates that
are unsafe to forward-declare. Costs: `any_cast` turns type errors into runtime throws (R10 is
the canonical failure mode), a per-access type check, and an extra heap allocation per `any`
holding a `shared_ptr` (exceeds libstdc++'s small-object buffer). The boundary is also already
punctured: `GetElements()`/`GetEncodingType()` return `lbcrypto` types, so `Ciphertext.hpp` now
includes two OpenFHE headers. Pick one direction:
- **A — restore header independence:** remove the OpenFHE includes from api headers again. Requires
  reworking `GetElements()` (opt-in `openfhe_interop.hpp` header, or an opaque/`std::any` return)
  and mirroring `PlaintextEncodings` in `Definitions.hpp` like the other mirrored enums. Keep
  `std::any` internally but centralize all casts in one internal helper header (typed accessors,
  each stored type spelled exactly once) so R10-class bugs cannot recur.
- **B — remove `std::any`:** store the concrete types (or a typed pimpl struct) — compile-time
  safety, no per-access checks, no extra allocation. api headers then include OpenFHE outright;
  consumers need OpenFHE headers on their include path (they already link against it).
Deciding factor: whether external consumers must compile against `fideslib.hpp` without an
OpenFHE installation. Either way, the centralized-cast hardening is worth doing immediately.

### O8. Rotation-fold radix — performance vs key-size scaling knob *(knob landed: `PARTIAL_SUM_RADIX`, default 4)*
Every rotation-accumulation fold (`sum_j Rotate(ct, j·stride)` over `size` summands) can be
evaluated at any power-of-two **radix** — the accumulation branching factor, exposed as the
`radix` parameter of `FHECKKSRNS::EvalPartialSumInPlace(ct, stride, size, radix)` (OpenFHE) and
the `bStep` parameter of FIDESlib's `Accumulate(ct, bStep, stride, size)` (GPU; `bStep` is a
BSGS-carryover name — for a pure rotation fold it is the radix, not a baby-step count). The
radix trades digit decompositions against rotation keys:

- **Digit decompositions (mod-ups, the dominant cost):** `log_radix(size)` levels, one
  decomposition each → `log2(size) / log2(radix)` mod-ups. Radix 4 does half as many as
  doubling; radix 8, a third.
- **Rotation keys:** `(radix − 1)` distinct rotations per level →
  `(radix − 1)·log_radix(size) = (radix − 1)/log2(radix) · log2(size)` keys. Doubling = `log2(size)`;
  radix-4 = `1.5·log2(size)`; radix-8 ≈ `2.33·log2(size)`.

The catch that makes doubling especially cheap on keys: **radix-2's index set is exactly the
power-of-two rotations `{stride·2^i}`** — the universal set essentially every CKKS program
already generates, so its *incremental* key cost is ≈ 0. Higher radices add *dedicated*
non-power-of-two indexes unlikely to be shared: radix-4 adds `{3·stride·4^j}`
(`{3, 12, 48, 192, …}` for stride=1), i.e. `floor(log4(size))` extra dedicated keys —
+1 at 8 slots, +5 at 1024, +7 at fully-packed 32768. At bootstrapping parameters a single
rotation key is tens of MB, so those extra keys are real storage.

**Current choice — a compile-time knob, `PARTIAL_SUM_RADIX` (default 4, by decision):**
the radix is now an OpenFHE CMake cache variable (`-DPARTIAL_SUM_RADIX=<power of two>`),
emitted into `config_core.h` and consumed as the single source of truth by every layer:

- **OpenFHE**: the bootstrap `EvalPartialSumInPlace(raised, slots)` wrapper folds at the
  configured radix, and keygen generates exactly the fold's index set —
  `{i·stride·radix^level, i ∈ [1, radix)}`, computed inline (no new public API) in place of the
  power-of-two loops at the three bootstrap `Find*RotationIndices` helpers and the two
  scheme-switching keygen sites. For radix > 2 the set is a superset of the power-of-two set,
  so nothing that relied on those keys breaks.
- **FIDESlib**: `accumulate_bStep` (bootstrap PartialSum) and the api `ACCUMULATE_SUM_RADIX`
  (AccumulateSum, both fallback and GPU) are set from the same macro.

Validated: the full compat suite is green at the default radix 4 — the first bit-exact
validation of the radix-4 fold (CPU 4-arg helper ≡ GPU `Accumulate(bStep=4)`), keygen included.
The key cost of the default is as derived above (radix-4 adds the dedicated `{3·stride·4^j}`
indices — `⌊log4(size)⌋` extra keys per fold); **key-minimizing deployments build with
`-DPARTIAL_SUM_RADIX=2`**, which reproduces the previous behavior exactly.

Audit of inline occurrences of the fold pattern (places not calling the subroutine): the
FHEW→CKKS sparse re-encode in `EvalFHEWtoCKKS` (`ckksrns-schemeswitching.cpp`) was an eager
`EvalAtIndex`+add doubling loop — converted to call `EvalPartialSumInPlace` at the configured
radix, with its two keygen sites generating the radix's index set (CPU-only path,
compile-validated; no GPU counterpart). The wide-matrix log-fold in
`EvalLTRectWithPrecomputeSwitch` is the same pattern but operates lazily on the extended
accumulator mid-transform, where the settled-ciphertext helper does not fit — left as is.

**EvalSum extension (knob renamed `CKKS_PARTIAL_SUM_RADIX` → `PARTIAL_SUM_RADIX`):** the
`EvalSum`/`EvalSumRows`/`EvalSumCols` family (`base-advancedshe`, scheme-generic) now folds at
the same radix. The doubling there lives in the automorphism-index group (generator g ∈
{5, 5^rowSize, 5⁻¹}, squared per level); the radix-r generalization uses g, g², …, g^(r−1)
within a level and advances g ← g^r. All four folds route through one shared
`EvalSumRadixFold`, which under HYBRID key switching accumulates **lazily in the extended
basis exactly like `EvalPartialSumInPlace`**: element 0 carries one deferred `ApproxModDown`
for the whole fold, element 1 settles once per level, and each level shares one digit
decomposition. (An eager hoisted fallback via a new `SchemeBase::EvalAutomorphismCore` wrapper
covers non-HYBRID key switching — EvalSum keys live in their own key map, so
`EvalFastRotation`'s key lookup does not apply.) The fully-packed BFV/BGV case
(`2·batchSize == m`) keeps the legacy doubling, since its tail is the conjugation-like
automorphism `m−1` with no radix generalization. The lazy accumulation changes EvalSum's
output bits at every radix relative to the previous eager doubling — the same class of change
P2b made upstream for PartialSum — and is value-validated by the ad-hoc test.

The laziness is what makes the radix pay: an earlier eager version of the fold (every rotation
settling through its own `ApproxModDown`) *lost* to dev at radix 4 and lost badly at radix 8 —
higher radix does 1.5–2.33× more rotations, and per-rotation mod-downs outweigh the saved
decompositions. Benchmarked single-threaded on a 2-socket Ice Lake Xeon (5 runs × 3 reps,
NUMA-pinned; EvalSum/Rows/Cols at ring 2^14/2^16, depths 5–25): the lazy fold beats dev by
~1.15× at radix 2 and ~1.5–1.65× at radix 4, with radix 8 consistently *behind* radix 4 — the
extra rotations outgrow the savings. The same run measured `EvalPartialSumInPlace` directly:
radix 4 ≈ 1.3× over radix 2 — the first CPU confirmation of the radix-4 default; radix 8 again
slightly behind radix 4. Radix 4 is the empirical optimum for both fold families on CPU.

Index generation is shared instead of duplicated: the `GenerateIndices*` helpers became static
and radix-aware (one `GenerateEvalSumIndices(g0, size, m)` core), and
`MultipartyBase::MultiEvalSumKeyGen` — which had its **own inline copy** of the doubling index
loop (a silent-drift risk under any radix change, and BFV-form-only) — now calls
`AdvancedSHEBase::GenerateIndexListForEvalSum` via a **friend declaration** (keeps the helpers
non-public; chosen over promoting them to the installed API). A multiparty audit found one more
duplication of the same shape: `MultiEvalAtIndexKeyGen` re-implemented the rotation→automorphism
index mapping with a hardcoded `isCKKS ? 2nComplex : 2n` dispatch instead of the scheme's
virtual `FindAutomorphismIndex` that single-party `EvalAtIndexKeyGen` uses — replaced with the
virtual (validated by a 2-party joint `EvalAtIndex(+1, −2)` value test). The rest of
`MultipartyBase` is genuinely multiparty math with nothing derivable to share; noted gap: no
`MultiEvalSumRows/ColsKeyGen` exists for threshold users. Key impact: EvalSum keys are
stored in their own map and were never shared with rotation keys even at radix 2, so radix 4
costs +50% of an already-dedicated log-scale set (batch 1024: 10 → 15 keys) and does not
interact with the bootstrap key budget. Validated by an ad-hoc value test at radix 4: CKKS
EvalSum/EvalSumRows/EvalSumCols, BFV EvalSum (radix path and fully-packed legacy path), and a
2-party `MultiEvalSumKeyGen` + joint EvalSum threshold flow — all pass.

Open sub-item: the `start`-offset `AccumulateSum` variant (`AccumulateCascadeImpl`) still runs
a literal `bStep=4` on the GPU with an unvalidated eager-doubling CPU fallback; binding it to
the knob (and putting it under the compat test) would close the last AccumulateSum gap.

### O9. `MODES(name)` macro ignores its parameter
`test/ParametrizedTest.cuh`'s `MODES(name)` declares identifiers literally named `name_fix`,
`name_fixauto`, `name_flex`, `name_flexext` — it never token-pastes with `##`, so the `name`
argument is ignored and every invocation declares the same four externs. Harmless as long as
nothing depends on per-name declarations (repeated identical externs are legal), but the macro
doesn't do what its signature advertises. Either fix it to `name##_fix` (and update/verify any
code relying on the literal names) or delete it if dead. Surfaced during the O2 macro audit;
not a correctness hazard for the assert paths, so left untouched.

### O10. `Accumulate` slots-narrowing — decide the fix layer
`FIDESlib::CKKS::Accumulate` rewrites the ciphertext's `slots` field after a full fold
(`AccumulateBroadcast.cu`: `if (size * stride == ctxt.slots) ctxt.slots = stride;` — same
pattern in `AccumulateCascadeImpl` and `Broadcast`). The guard is a caller-intent heuristic —
"a fold covering the whole slot range must mean the caller wants the result treated as a sparse
`stride`-slot ciphertext" — and it misfires: right for the bootstrap PartialSum, wrong for api
`AccumulateSum`, whose OpenFHE semantics keep the slot count unchanged (the CPU mirror
`EvalPartialSumInPlace` never touches slot metadata). The R11 fix compensates at the api
boundary (save/restore of `slots` around the `Accumulate` call in the two radix-2 GPU paths),
which works but must be remembered at every future call site.

In-tree evidence that the narrowing sits at the wrong layer: `Accumulate` has exactly two
internal callers, both in `Bootstrap.cu` — one (`Bootstrap.cu:264`) immediately **overwrites**
`ctxt.slots` on the next line, making the internal narrowing dead code there; only the other
(`Bootstrap.cu:100`) actually relies on it (downstream CtS rotations normalize indices through
`ctxt.slots`).

**Proposed FIDESlib-side fix (decision pending):**
1. Delete the `slots` writes from `Accumulate` and `AccumulateCascadeImpl` — the fold becomes
   metadata-neutral, matching its CPU mirror's contract.
2. Add an explicit `ctxt.slots = slots;` after the `Bootstrap.cu:100` call site (mirroring what
   the other call site already does), making the bootstrap's sparse re-interpretation visible
   where it is decided.
3. Drop the `inputSlots` save/restore from the api paths.

Bonus: the `start`-offset api variant calls `AccumulateCascadeImpl` (no internal callers) and
did **not** get the R11 save/restore — its slots leak (the O8 sub-item) is fixed for free.
Caveat: this changes the observable behavior of a public FIDESlib function; any out-of-tree
code relying on the narrowing convention would break — worth confirming with the FIDESlib
maintainers. `Broadcast` has the same trailing `slots` rewrite with no in-tree consumers;
leave it alone for this fix but it is the same pattern.

### O11. api `EvalBootstrap` GPU dispatch drops parameters
Two silent divergences between the api's CPU fallback and GPU path (`api/CryptoContext.cpp`,
`EvalBootstrap`/`EvalBootstrapInPlace`), found while auditing the dispatch during the
combination sweep:

- **`numIterations`/`precision` ignored on GPU.** The CPU fallback forwards both to
  `context->EvalBootstrap(ct, numIterations, precision)` (META-BTS iterated bootstrapping);
  the GPU path unconditionally runs a single `FIDESlib::CKKS::Bootstrap`. A caller requesting
  two iterations gets one, silently, with correspondingly lower precision.
- **`btsfirstboot` (StC-first) misroutes on GPU.** api `EvalBootstrapSetup` exposes
  `btsfirstboot` and forwards it into the CPU precomputation (`BTSlotsEncoding`), but FIDESlib
  has no StC-first pipeline (`Bootstrap.cuh` exposes only the CtS-first `Bootstrap` and
  `BootstrapCPUraise`); the GPU path runs CtS-first against StC-first precomputations.

Both are O3-class semantic gaps (api contract, not bit-compat): decide whether to fall back to
CPU when the request can't be honored on GPU (`numIterations > 1`, StC-first setups), throw, or
implement the missing GPU paths. Until then they fail silently rather than loudly.

### O12. Degree-2 Rescale on device backends inherits the O3 divergence
The api's degree-2 `Rescale` runs unconditionally on device backends (CUDA, haze); OpenFHE's
public API only rescales under FIXEDMANUAL. Same gap as O3 — degree-2 ciphertexts are not
exempt from it. See O3 rather than duplicating the analysis here.

### O13. Degree-2 Rescale at the FLEXIBLEAUTOEXT extra level throws on CUDA and haze (untested)
`Ciphertext::rescale()` throws `"relinearize first"` for a degree-2 ciphertext at the
FLEXIBLEAUTOEXT extra level (L+1): the fused two-poly `rescaleDouble` knows how to peel the
special limb off both of `c0`/`c1`, but the single-polynomial `rescale()` used for the third
component cannot follow it there, so the guard refuses rather than desynchronize the three
components. OpenFHE's public API succeeds in the equivalent case.

haze's compute is not blocked on FLEXIBLE scaling techniques in general — `scalingTech_` gates
real arithmetic throughout `api/engine/haze/HazeEngine.cpp` (e.g. `adjustForAddOrSub`,
`HazeScalarEncode.cpp`'s `towers == Q` handling), so the state is reachable there. Its
`rescaleCore` (`api/engine/haze/HazeEngine.cpp:765`) carries no fused two-poly path in the first
place — every component (2 or 3) goes through the identical single-tower
`rescaleChainOneTower`, so the specific desynchronization CUDA guards against cannot occur
mechanically. Even so, `rescaleCore` now mirrors CUDA's throw for a degree-2 operand at the
extra level, guarded on `scalingTech_ == FLEXIBLEAUTOEXT && levelOf(x) == 0` (the level check
alone is just "freshly encrypted" under every other technique, since haze's `qBase_` always
holds exactly the moduli provisioned for the active context; the FLEXIBLEAUTOEXT gate is what
makes it the one-modulus-larger extra state CUDA's `cc.L + 1` refers to), so the state stays
untested/unsupported on both backends instead of only on CUDA. FLEXIBLEAUTOEXT sits outside the
parity fixtures' scope, which target FIXEDAUTO.

## Plan forward

**P1 — Land the current state.** *(ongoing)* The `OpenFHECompatTests` branch carries the work
in per-milestone commits (FIDESlib fixes, `deps/fideslib-ref-1.5.1.6.patch` + `build.sh` bumps,
`test/OpenFheCompatTests.cu`, and this document); the upstream side is tagged
`fideslib-ref-v1.5.1.2`–`v1.5.1.6`. The green suite is the regression wall for everything below.

**P2 — Upstream OpenFHE PRs** (shrinks the patch back to visibility shims; each step re-validated
with the stage harness):
- **a.** *(done)* `EvalRotate`/`EvalAtIndex`/`EvalAutomorphism` + `Conjugate` unification onto
  the hoisted core (O1) — landed as `fideslib-ref-v1.5.1.2`; the R1 interim hunks are dropped
  from the patch and `OpenFHECompatTests.EvalRotate` is green. Remaining: propose the same
  change to the upstream OpenFHE dev branch proper.
- **b.** *(done)* Lazy extended-basis PartialSum, implemented as
  `FHECKKSRNS::EvalPartialSumInPlace` and used at all three PartialSum sites (`EvalBootstrap`,
  `EvalBootstrapStCFirst`, `EvalHomDecoding`): element 0 accumulates in the extended (QlP)
  basis starting from `P·cv[0]` with a single deferred `ApproxModDown`; element 1 settles per
  level (its digit decomposition feeds the next rotation). Also a CPU win: `log2(N/(2·slots))−1`
  fewer mod-downs per PartialSum (7 for slots=8, N=4096). The initial-lift convention matched
  FIDESlib directly — FIDESlib's mixed standard/extended adds P-scale the standard operand
  (`add_scale_p_*` kernels), identical to OpenFHE's `·PModq` lift, and
  `ApproxModDown(P·x + ks) = x + ApproxModDown(ks)` holds exactly, so no bisect cycle was
  needed. The lazy `Accumulate` is restored in FIDESlib (R2 perf recovered); full suite green.
  Landed with P2c as `fideslib-ref-v1.5.1.3` (commit `d31322ac`); remaining: propose upstream.
- **c.** *(done)* Lazy Horner giant-step: `EvalHornerGiantRotate` settles only
  `c1` (its digit decomposition feeds the key switch) and folds the extended `c0` into the
  key-switch product directly — no per-giant-step `ApproxModDown` + `·PModq` re-lift round-trip.
  Additionally, the final CtS/StC slot-ordering corrections (`EvalAtIndex(result, Delta)`) are
  folded into the last level's Horner output while it is still extended, matching the GPU's
  offset rotation (`LinearTransform`'s `offset` consumes the extended `c0` under `ONLY_C1`).
  `EvalLinearTransform` inherits the lazy giant steps automatically. FIDESlib `ONLY_C1` is back
  to `true` (R3 perf recovered); full suite green, dense bootstrap ~11% faster. The same lazy
  folds were applied to the scheme-switching linear transforms (`EvalLTWithPrecomputeSwitch`
  gains an extended-output flag; the sparse SlotsToCoeffs doubling and the wide-matrix log-fold
  in `EvalLTRectWithPrecomputeSwitch` accumulate extended and settle once) — CPU-only paths, no
  GPU counterpart, validated with OpenFHE's pke unit tests. Landed with P2b as
  `fideslib-ref-v1.5.1.3` (commit `d31322ac`); remaining: propose upstream.
- **d.** *(done)* Radix-configurable PartialSum: the generalized
  `EvalPartialSumInPlace(ct, stride, size, radix)` (R11) is now driven everywhere by the
  compile-time `PARTIAL_SUM_RADIX` CMake variable, **default 4 by decision** — halving the
  digit decompositions at the raised level (8 → 4 for slots=8, N=4096), the widest-tower, most
  expensive mod-ups of the bootstrap, at the cost of the dedicated `{3·stride·4^i}` rotation
  keys (~1.5× the PartialSum key count vs doubling). Keygen generates the radix's exact index
  set inline at each site, FIDESlib and the api read the same macro, and the full
  suite is green at radix 4. Key-minimizing deployments build OpenFHE with
  `-DPARTIAL_SUM_RADIX=2`, restoring the previous behavior exactly (see O8). Landed as
  `fideslib-ref-v1.5.1.5` (commit `fa5b49e3`); remaining: propose upstream with the rest of P2.
- **e.** *(in the post-v1.5.1.5 delta)* advancedshe/multiparty hygiene from the fold audit:
  `EvalAddMany` → left fold (one clone instead of n−1 allocating `EvalAdd`s);
  `EvalAddManyInPlace` → actually-in-place serial fold (zero allocations, no trailing clone,
  result in slot 0 per contract; null-leading vectors work for the first time — the old code
  dereferenced slot 0 before its null handling could run). Measured ~1.5–2× (`EvalAddMany`) and
  ~1.8–2.3× (`EvalAddManyInPlace`) vs dev on Ice Lake, growing with tower count. Both functions
  are now null-tolerant (entries skipped), and a size-1 **aliasing bug** is fixed: the
  `cryptocontext.h` wrappers for `EvalAddMany`/`EvalMultMany` returned the caller's input
  `shared_ptr` for single-element vectors, so mutating the "result" silently corrupted the
  input — `EvalAddMany`'s early return is dropped (the scheme clone covers all sizes) and
  `EvalMultMany`'s becomes a `Clone()` (it cannot be dropped outright: the wrapper fetches
  relin keys before dispatch, and the scheme-level tree had `.back()`-on-empty UB for size 1,
  now also guarded). The result of both APIs never aliases an input. The FIDESlib api paths
  mirror all of it (null-safe fallbacks, GPU null-skip fold, independent copy for
  single-operand cases — also fixing the GPU `EvalAddMany` size-1 UB); CPU≡GPU bitwise held by
  the compat suite. `EvalMultMany` keeps its balanced tree (depth bounds noise growth and level
  consumption — do not serialize like the adds) but frees each consumed partial immediately —
  live intermediates drop from n−1 to ~n/2. **Measurement pitfall found on the way: CKKS
  overrode it** (`AdvancedSHECKKSRNS::EvalMultMany`, which existed only for composite-scaling's
  `levelsToDrop = GetCompositeDegree()`) — early benchmarks of the base change alone ran
  unchanged CKKS code. Resolution: **the implementations are unified** — the base picks
  `levelsToDrop` from `GetCompositeDegree()` when the params are RNS (1 for everything except
  composite-scaling CKKS, ≡ `BASE_NUM_LEVELS_TO_DROP`) and the CKKS override is deleted, so the
  divergence class is gone. The unified fold keeps the override's three-phase structure
  (input×input, odd-size mixed node, partial×partial). Isolated dev+change-only build at n=32
  (ring 2^16, depth 25): **peak MultMany scratch 709 → 381 MB (−46%)** at 1.00× wall time; at
  n=8 the peak is transient-dominated (first round has nothing to free) and RSS is unchanged.
  Pairing order untouched, so bit-identical. Also from the multiparty audit:
  `MultiEvalAtIndexKeyGen` now uses the scheme's virtual `FindAutomorphismIndex` instead of a
  hardcoded `2n/2nComplex` dispatch (see O8), and the EvalSum radix work (lazy
  `EvalSumRadixFold`, friend-shared index generation, `PARTIAL_SUM_RADIX` rename) rides in the
  same delta. The delta surfaced one regression via OpenFHE's own unit tests: the fully-packed
  BFV/BGV EvalSum key set (legacy doubling + `m−1`) no longer covered smaller-batch folds —
  radix-2 sets were nested across batch sizes for free, and e.g. `EvalInnerProduct` with
  default keygen requested key `5³`. Fixed by radix-folding the fully-packed first phase
  (covering `batchSize/2`) with only the `m−1` conjugation tail kept special, in both keygen
  and the fold — the fully-packed set is a superset of every smaller batch's set again, at any
  radix (the sets are threshold-nested in the exponent). Full pke unit suite green at radix 2,
  4, and 8 (1892 tests each; radix 2 provably reproduces the legacy key set). Landed as
  `fideslib-ref-v1.5.1.6` (commits `b211e037`, `691908c9`).

**P3 — Fix the `rotate_hoisted` memory bug (O2).** *(done)* Resolved exactly as prescribed —
the host ASAN build identified it as a test-macro argument-re-evaluation bug, not a memory bug
(see O2). `EvalFastRotationHoisted` is re-enabled and green; nothing to hand off to the
FIDESlib maintainers.

**P4 — Tier-2 coverage (O6)** one configuration at a time — each red result is a mini-investigation
with the harness. *(swept)* Five configurations green with tests landed; five findings
characterized (O6a–O6e). O6a, O6b, and O6d are fixed (T2km1 level alignment; double-angle
FIXEDMANUAL rescale moved to iteration end; dense-tail leftover rescale deleted) with tests in
place — the full FIXEDMANUAL configuration, sparse and fully packed, is now bit-compatible.
The follow-up combination sweep added four more green configurations (see O6) and surfaced
O6e. Remaining: **O6c** — the SPARSE_ENCAPSULATED
implementations are structurally different designs (dual-context GPU vs in-context stock) and
the GPU output is value-wrong at the test configuration; fix deferred, plan documented (see
O6c) — and **O6e** — FIXEDAUTO add/sub operand adjustment (fix direction documented). Then
close out the smaller semantic gaps (O3–O5, O11) and the api type-erasure decision (O7).

## Validation methodology (reusable)

Env-gated bisection: add a `BOOT_STAGE_LIMIT` helper to both pipelines
(`ckksrns-fhe.cpp` / `Bootstrap.cu` + `ApproxModEval.cu`), truncating after aligned stages —
1 ModRaise, 2 PartialSum, 3 CtS, 4 conj+add, 5 approxMod (51x = Chebyshev sub-stages: baby powers,
giant powers, T2km1, inner PS; 52 = double-angle), 6 StC(+doubling); dense variants 71–82 —
then compare with the exact-equality test. One build+run per probe localizes any divergence to a
single stage. A lighter-weight variant (used for O6a): env-gated checkpoint prints of
coefficient fingerprints — `c0`/`c1` first coefficient of tower 0 (index 0 is invariant under
the GPU's bit-reversed NTT order, so it compares directly) plus deg/level — at aligned stages
on both sides; diffing the two traces names the first diverging op in one run. Gotchas learned:
`g_used_rot_indices`-style key instrumentation misses `GetRotationKey`'s alternative-key
early-return (the "unused key" that started all this was an artifact); `normalyzeIndex` maps
rotation indices through `ctxt.slots`; encryption is randomized, so CPU/GPU test phases must
share a single `Encrypt` call; test macros that re-evaluate their arguments under shadowed loop
variables can fake memory corruption (O2); a "clean" `Decode` of values near zero still means a
wrong result — check magnitudes, not just exceptions (O6c); sweeping configuration axes one at
a time misses combinations — dense×FIXEDMANUAL diverged where each axis alone was green (O6d);
matching output metadata does not imply a matching pipeline — degree-gated rescales can absorb
an extra upstream rescale and land at the right level via the wrong trajectory (O6d).

## Performance accounting (recovered via P2)

| Change | Cost of current (eager) form |
|---|---|
| R2 `Accumulate` | *recovered (P2b)* — lazy accumulation restored on both sides |
| R3 `ONLY_C1=false` | *recovered (P2c)* — lazy Horner restored on both sides |
| R4 Chebyshev transcription | temp-ciphertext clones + k−1 alignment ops (CPU does these too); optimize buffer reuse only if profiling warrants |

Numbers are op-count based (slots=8, budget {3,3}); profile before prioritizing.

### O14. `OpenFheInterfaceTests` red baseline is inherited from upstream, not introduced here
`test/OpenFheInterfaceTests.cu` and `test/OpenFheBootstrapTests` carry a large failing baseline that
no document in this repo recorded an expected state for. Measured 2026-08-27 on an RTX 5090,
`FIDESLIB_TEST_BACKEND=cuda`, `--gtest_filter='OpenFHEBootstrapTests/*:OpenFHEInterfaceTests/*'`
(~15 min per run):

| tree | OpenFHE | tests | stable failures | flaky |
|---|---|--:|--:|--:|
| this tree | fideslib-ref-v1.5.1.6 | 264 | **82** | 7 |
| upstream `786c760`, as it pins itself | v1.5.1 | 256 | 88 | 4 |
| upstream `786c760`, rebuilt against v1.5.1.6 | fideslib-ref-v1.5.1.6 | 256 | 87 | 2 |

`786c760` is this branch's exact merge-base with `upstream/main`. Test inventories are identical
apart from the 8 `RawRoundTrip3Poly` cases added here (none of which fail). "Stable" means failing
in every run of that tree; three runs for this tree, two per upstream configuration.

**Set comparison against the controlled upstream build** (only FIDESlib differs):

| | count |
|---|--:|
| fail in both - inherited | **82** |
| fail only here - divergence introduced by this branch | **0** |
| fail only upstream - fixed here | 5 |

All 82 also fail in *both* runs of *both* upstream configurations, so the inherited-ness is firm.
Rebuilding upstream against the newer OpenFHE moved only 3 tests, all inside the flaky families,
so the patch level is not a meaningful variable here.

**Failure kinds.** Roughly half are not bit-compat at all:

| kind | count | families |
|---|--:|---|
| `ASSERT_EQ_CIPHERTEXT` - bit-divergence, may still decrypt correctly | 42 | `Mult`, `MultAllLevels`, `MultRescale`, `Square`, `SquareAllLevels` |
| `ASSERT_ERROR_OK` - **decrypted values outside tolerance** | 40 | `OpenFHEBootstrap`, `OpenFHEBootstrapLT`, `OpenFHEBootstrapDense`, `LinearTransform`, `CoeffsToSlots` |

The 42 bit-divergence cases are the same relinearization key-switch rounding recorded for
`ApiParityTest.EvalMult`/`EvalSquare` (`test/ApiParityTest.cpp:212-239`), in a different suite. The
40 precision failures are genuine numerical breakage in the GPU bootstrap path and are the more
serious half.

**The 5 fixed-here cases corroborate O6 independently.** Per-parametrization scaling techniques:

- `ApproxModEvalSparse`: upstream fails /1 (FIXEDAUTO) and /3, /5, /7 (all FIXEDMANUAL); this tree
  fails only /1. So O6a/O6b (`88fa0f6`, T2km1 level alignment + double-angle rescale placement)
  fixed every FIXEDMANUAL case in this suite too, and O6e (FIXEDAUTO, still open) is exactly why /1
  still fails.
- `OpenFHEBootstrapDense`: upstream fails /0, /2, /4, /6 (FLEXIBLEAUTOEXT) and /3, /5, /7
  (FIXEDMANUAL); this tree fixed /3 but still fails /5 and /7, so the FIXEDMANUAL fix is only
  partial here. The four FLEXIBLEAUTOEXT dense cases fail in both trees - an inherited gap nothing
  has addressed.
- `AccumAllLevels/4` is in this tree's flaky set; treat it as noise, not a fix.

**Methodology warning: a single run of this suite cannot distinguish a regression from noise.**
7 tests here and 2-4 upstream flip with no source change, confined to
`OpenFHEInterfaceTest.{AccumAllLevels, Conjugate, HoistedRotateAllLevels, RotateAllLevels}` in both
trees. OpenFHE seeds its PRNG from `std::random_device` (`blake2engine.cpp:118`) and no test fixes a
seed, so each run encrypts different values and any divergence near a rounding threshold lands on
either side. Compare failing *families* across repeated runs, never exact counts or ids. This is how
R13's comparator change was cleared: its before/after symmetric difference (5 ids) sat inside the
same-binary noise floor (3 ids), in the same families.

Reproduce by building upstream FIDESlib at `786c760` twice: once against its own pinned OpenFHE
(v1.5.1) and once against `fideslib-ref-v1.5.1.6`.
