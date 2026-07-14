# Implementation audit

Audit of this repository against Boneh, Laufer, Tas,
*"Threshold Batch Identity-based Encryption without Epochs"* (the paper).
Section numbers refer to the paper unless noted.

Executive summary: the implementation covers every construction in the paper
(Sections 4-9, Appendices A.4, B, C).  All algebraic invariants are enforced
by asserts and tests.  Known deviations are enumerated in section 3 below;
none affect correctness of the schemes; three (S3.1, S3.2, S3.3) are relevant
to the fidelity of the *security argument* and are called out prominently.
This code is a research prototype: see section 5 for the security review and
the explicit non-goals.

---

## 1. Conformance map (paper -> code)

| Paper object | Code | Notes |
|---|---|---|
| Gadget matrix G_n, m~ (S3.4.1) | `gadget.gadget_matrix`, `gadget_dims` | g = (1, 2, ..., 2^{floor(log q)}), m~ = n(floor(log q)+1) exactly as defined |
| G^{-1} bit decomposition | `gadget.g_inverse` | binary, G G^{-1}(X) = X verified |
| Basis S of Lambda^perp(G) (Lemma 21) | `gadget.g_perp_basis` | MP12 basis; per-block det = +-q verified in tests |
| TrapGen (Lemma 1/23) | `trapdoor.trapgen` | A = [Abar \| G - Abar R], T = [R; I], A T = G, \|\|T\|\|_inf = 1; statistical closeness via LHL needs m >= m~ + (n+1) log q (warned if violated) |
| SamplePre (Lemma 1/23) | `trapdoor.PreimageSampler.sample_pre` | Klein/GPV sampler over an Ajtai basis derived from the gadget trapdooor; see S3.1 |
| SampleLeft (S A.2.3, [ABB10]) | `trapdoor.PreimageSampler.sample_left` | via DetExtBasis below |
| Explainable SampleLeft (Def. 12, Thm 1, App. C) | `sample_left(...; rb)` + `explain_left` | deterministic in the tape rb; ExplainSL recovers Klein coefficients exactly and re-samples tape intervals (Lu-Waters / CHW approach); round-trip tested on both basis paths |
| DetExtBasis (App. C.1) | `trapdoor.det_ext_basis` | T'_F = [[T'_A, X], [0, I]]; X solved deterministically (fixed-pivot Gaussian elimination), GS([.]) = [[GS(T'_A), 0], [0, I]] exploited exactly |
| Lemma 31 Ajtai trapdoor | `trapdoor.ajtai_basis` | literal [I - T G^{-1}(A) \| T S] leftmost-independent selection; **see S3.1** |
| WWW24 Gen/Expand/ExpandLocal (App. B.3) | `www24.gen/expand/expand_local` | crs = [A \| B]; A_u = [A \| B - u^T x G]; D_ell and T assembled per eq. (3); D T = I_ell x G and \|\|T\|\|_inf <= 1 asserted |
| EvalF / EvalFX for indicators (Thm 10, Lemma 29) | `www24.eval_f/eval_fx` | recursion exactly as App. B.4; identity (B - x x G) H = B_u - delta_u(x) G tested for x = u and x != u |
| StructTrapGen (Lemma 30) | `www24.struct_trapgen` | block layout [-H_{B,u_j,u_i}]_{ij}, bottom [G^{-1}(B_{u_j})]_j |
| SampleMultPre (Def. 24) | `www24.sample_mult_pre` | outputs (pi_i, c = -G c-hat); A_i pi_i = t_i + c tested |
| GenProg (somewhere programmability) | `www24.gen_prog` | perfect: Expand on programmed crs reproduces A* exactly (tested); crs distribution preserved (bijection, bin-tested) |
| H (Def. 13) | `hashes.hash_index` | injective for integer identities < 2^d; SHAKE-truncation otherwise (S3.6) |
| Hro (Def. 14) | `hashes.hash_ro` | SHAKE-256 XOF to Z_q^{n x m}, 64 surplus bits/element |
| Hsp (Def. 15) / Hsp^th (Def. 17) | `hashes.hash_sp/hash_sp_th` | binds (A, T_A, (r_i), u, sigma1) (+ sidsp); output = SampleLeft tape |
| Htr (Def. 16) | `hashes.hash_tr` | lambda_tr = 256 bits |
| BIBE.Setup/Enc/PreDec/Dec (S5) | `bibe.setup/encrypt/pre_dec/decrypt` | ct = (r, ct1, ct2, ct3) with eps3 = (eps3pre, R~^T eps3pre); PreDec aborts on index collisions; Dec re-derives v via the Hsp tape and rounds per the paper's round() |
| Batches of size < ell (Remark 1) | supported | tested |
| Multi-bit variant (S5.3) | `params.msg_bits`, u_1..u_N | one SampleLeft target/preimage per bit; see S3.7 |
| RO-compressed pk (Remark 2) | not implemented | noted; crs/u generation from a seed would be mechanical |
| TGS.Setup/RoundOne/RoundTwo/Comb (S6.1) | `tgs.*` | Shamir sharing of T_C entry-wise over Z_q; pairwise PRF masks; y' = G^{-1}(c') binary; combiner checks C sbk = c |
| TBIBE.Setup/PredecOne/PredecTwo/Comb/Dec (S6.2) | `tbibe.*` | sidsp = Htr(sid, act, (e_i)) domain-separates Hsp^th; combiner checks sidsp and c consistency; Dec re-checks C sbk = c |
| Threshold correctness bound B_th (Thm 4/6) | `params.threshold_bounds` | \|\|sbk\|\| <= B_th asserted in tests |
| Threshold GPV (S7) | `gpv.*` | c = HGPV(m, sidsp); verify checks the mod-q equation and the B_th norm bound |
| KP-ABE -> BIBE (S8) | `abe.*` | generic wrapper verbatim; KP-ABE instantiated BGG+14-style for F_{k,ell} via indicator sums (S3.8) |
| Trilinear BIBE (S9) + threshold (S9.4) | `trilinear.*` | generic-group simulation (no real trilinear map exists); partial-fraction threshold pre-decryption; see S3.9 |
| Lattice BEAT-MEV (App. A.4) | `beatmev.*` | BLMR PRF with binary chain matrices, GGM puncturing with p'-rounded sibling nodes, chunked-Regev key transport; see S3.10 |
| Theorem 2/3 parameters | `params.spec_params`, `check_theorem2` | full constraint system checked programmatically at several (lambda, ell) |

## 2. What is verified, and how

- **Algebraic identities**: every trapdoor relation (A T = G, D T = I x G,
  A T' = 0, F v = u, C sbk = G c-hat, Lemma 29's evaluation identity) is
  asserted at run time and covered by tests, including exact big-int
  cross-checks where int64 overflow was a risk.
- **Determinism**: SampleLeft is a pure function of (inputs, tape); PreDec and
  Dec derive bit-identical v (tested by clearing the cache and re-deriving).
- **Explainability (Theorem 1)**: ExplainSL round-trips: for sampled v,
  SampleLeft(inp; ExplainSL(inp, v)) = v exactly, on both basis paths, and at
  the 1-D level the tape interval invariant F(z-1) <= x < F(z) is tested per
  regime.
- **Distributions**: 1-D sampler moments and a chi-square test against the
  exact pmf (deterministic streams - cannot flake); preimage marginal
  moments; TrapGen/GenProg output uniformity bin tests.
- **End-to-end**: BIBE/TBIBE/GPV/ABE/trilinear/BEAT-MEV round-trips, negative
  tests (wrong keys, tampered shares, mixed transcripts, out-of-batch
  ciphertexts, norm-bound and sidsp-binding rejections), threshold runs over
  multiple active sets, and multi-batch reliability runs
  (`scripts/verify_deep.py` -> VERIFICATION.md).

## 3. Deviations from the paper (all documented, none hidden)

### S3.1 Lemma 31 selection can span a proper sublattice (potential erratum)

Lemma 31 (from [CHW25, Lemma 7.4]) forms W = [I_m - T_A G^{-1}(A) | T_A S] and
takes "the first m linearly-independent (in R) columns" as an "Ajtai trapdoor"
T'_A.  As stated, W's columns *generate* Lambda^perp(A) as a lattice (we
verified the generation argument), but an m-column full-rank *subset* need not
be a *basis*: on a small TrapGen instance (n = 2, q = 257) the literal
selection - which is exactly the block I - T G^{-1}(A) whenever it is
nonsingular - spans a sublattice of index ~3.9 x 10^8, and
`scripts/verify_deep.py` reports the same phenomenon at our toy sizes.
Klein sampling over a sublattice still yields *correct, short, deterministic,
explainable* preimages (everything the honest protocol needs, and everything
we test), but the preimage distribution is supported on a coset of the
sublattice rather than the full coset D_{Lambda^u, sigma} that the security
proof's hybrids (Lemma 18, Hybrid 5/6) simulate.  A security-faithful
implementation needs T'_A to be a genuine basis.

What we do about it:

- For TrapGen-structured matrices (the C side: pre-decryption keys, GPV
  signatures) we construct a **provably exact basis**
  `trapdoor.trapgen_exact_basis`:
  B = [[I - R G^{-1}(Abar), R S], [-G^{-1}(Abar), S]], whose determinant is
  +-det(S) = +-q^n by a unimodular-transformation argument
  (`test_exact_basis_is_genuine`).  This path is the default for C.
- For the WWW24 matrix D_ell (trapdoor from StructTrapGen, no TrapGen
  structure) we keep the paper-literal Lemma 31 selection and flag the
  exactness status per instance (`check_exactness`).  Constructing a short
  exact basis from an arbitrary gadget trapdoor (e.g. via [MG02, Lemma 7.1]
  ToBasis against a mod-q HNF) is possible but was out of scope; the honest
  protocol is unaffected.

### S3.2 Discrete Gaussian CDFs are approximated in the middle regime

The 1-D sampler uses (a) exact windowed summation for s <= 48, (b) the
continuous-Gaussian CDF at half-integer cut points with the first
Euler-Maclaurin correction for 48 < s <= 2^40 (per-point relative error
O(1/s^4), i.e. < 1e-10 at the smallest widths in that regime), and (c) the
same formula in mpmath beyond.  Determinism and explainability are *exact*
for any fixed monotone F; only the distribution's distance from the ideal
discrete Gaussian is affected (bounded by ~dim x per-point error < 1e-6 per
SampleLeft call at toy sizes; a production implementation should use a
constant-time exact sampler).

### S3.3 Floating-point Gram-Schmidt and center computations

GS vectors come from LAPACK QR (float64); Klein centers are exactly-rounded
float sums of exact integer state.  All *integer* state (targets, coefficient
updates, outputs) is exact Python-int arithmetic, so coset membership and all
algebraic identities hold exactly regardless of conditioning.  The float data
affects (i) sampling quality and (ii) cross-platform reproducibility: two
decryptors on different BLAS builds could in principle derive different (still
correct) v from the same tape, which would break their agreement.  Within a
process the data is computed once and shared (PreDec = Dec agreement is
guaranteed and tested).  A deployment must pin the numeric stack or implement
fixed-point GS; flagged here as the main engineering gap.

### S3.4 Toy parameters

The presets in `params.toy_params` are calibrated for correctness margins
(sigma1 >= ~4x the observed extended-basis GS norms; decryption error
<= ~1/8 of q/4 at worst across the deep-verification runs) - **not** for
security: n in {4, 8} and sigma1 is far below the provable
18(2 ell t + m~)^{3/2}... bound of Theorem 2 (which forces q ~ 2^60+ even at
n = 4).  `spec_params` implements the provable derivation and
`check_theorem2` verifies it; `scripts/verify_deep.py` prints exactly which
provable constraints each toy preset violates.

### S3.5 Flooding parameter

Theorem 5 / Remark 3 wants sigma_flood = 2^Omega(lambda) for TBIBE
(indistinguishability), and sigma_flood = O(B_td sqrt(q_sig) lambda) suffices
for GPV (Theorem 7).  The threshold toy presets use sigma_flood = 2^14 so the
flooded key still fits a small modulus; the Renyi-divergence security margin
is therefore nominal only.  The samplers accept arbitrary widths (the mpmath
regime handles 2^lambda-scale flooding) if callers set spec-scale parameters.

*Review note:* an adversarial code review of this repository found (and we
fixed) a real bug here: the original CDF inversion used a +-1 fix-up walk
from a float64 initial guess, which deterministically raised for
sigma >~ 2^70 - i.e. exactly at the spec-scale flooding widths this section
describes.  Inversion is now exponential bracketing + bisection with a
full-precision initial guess in the mpmath regime, and the float64 erf CDF
delegates its tails (values within 2^-40 of 0/1) to mpmath so extreme tapes
are not clamped at ~8.3 sd and per-step CDF increments never collapse.
Regression tests: `test_spec_scale_flooding_widths`,
`test_extreme_tape_reaches_true_tails`.

### S3.6 Hash instantiations

All random oracles are SHAKE-256 with domain separation; mod-q outputs take
64 surplus bits (bias < 2^-64).  H (Def. 13) is the binary expansion for
integer identities < 2^d (injective, as the definition requires) and a
d-bit SHAKE truncation for other identity types, where injectivity is only
computational; PreDec/Dec reject colliding indices, as the paper specifies.

### S3.7 Multi-bit variant

Section 5.3 specifies N target vectors u_j and per-bit ct1 components but does
not spell out PreDec; we run one Hsp-derived SampleLeft and one sbk_j per bit
(the natural reading; |sbk| grows by N).  The threshold scheme is implemented
for msg_bits = 1 (the paper's setting); multi-bit TBIBE would need one TGS
session per bit and is rejected with a clear error.

### S3.8 The KP-ABE instantiation is ours

Section 8 is a generic transformation from any KP-ABE.  We instantiate it
BGG+14-style specialized to F_{k,ell}: B_S = sum of indicator EvalF outputs,
f_S(x) = 1 - sum delta_u(x).  This matches the BGG+14 row of Table 1
asymptotically, but the paper does not fix a concrete ABE; our security
inherits BGG+14's argument shape (not re-proven here).

### S3.9 Trilinear maps do not exist

Section 9 is implemented against a simulated generic trilinear group (the
security model the paper itself uses).  The simulation is pedagogically
faithful (the scheme code touches only oracle queries), but there is no
real-world instantiation; the Section 9.4 threshold flow models the
beta = s(alpha + r) sub-protocol of [Wang et al. 60] as an ideal step.

### S3.10 BEAT-MEV sketch choices

Appendix A.4 gives no full algorithms.  Concrete choices here: binary BLMR
chain matrices (required for the m^k noise growth the appendix quotes),
sibling nodes rounded to p' = 2^38 >= 8 m^k p, and exact chunked-Regev
transport of the ephemeral keys (base-2^8 chunks, homomorphic sums decoded
exactly, matching the appendix's "Regev to preserve the key structure").
The committee is single-party in code; thresholdizing Regev is standard [11]
and out of scope.

### S3.11 chi noise convention

The paper's chi is Psi_alpha (a rounded scaled continuous Gaussian); we use
the discrete Gaussian of width sigma_chi = alpha q in the same rho
convention.  The two are statistically close in the regimes used and the
correctness analysis only uses the noise norm bounds.

## 4. Paper observations noted while implementing

1. (S3.1 above) Lemma 31 / Appendix C.1's "first m linearly-independent
   columns" yields a proper sublattice in general; presumably the intent of
   [CHW25, Lemma 7.4] is a genuine basis - worth an erratum or an explicit
   ToBasis step.
2. Definition 15's Hsp signature contains a Z^+ input (sigma) and the paper's
   PreDec passes sigma1; TBIBE additionally passes sidsp through Hsp^th
   (Def. 17).  Note the tape must also depend on the *target* u to support
   the multi-bit variant; we include u (it is listed in Def. 15's domain).
3. Theorem 2's parameter list contains both the C-width and the Hro-width
   under the same symbol m (a bar was evidently lost in typesetting); we
   split them (`m_c`, `m_ro`) and satisfy both constraints.
4. TGS.Comb (S6.1) as written returns sbk in Z_q^m; decryption needs the
   *centered* short representative (guaranteed unique by ||sbk|| <= B_th <
   q/2); we make the lift explicit.
5. The bound "||y'||_inf <= ceil(log q)" above Theorem 4's proof appears to
   be a typo for ||y'||_inf <= 1 (y' = G^{-1}(c') is binary); the stated
   B_td/B_th are correct either way (they only need ||T_C y'|| <=
   sqrt(m m~) log q).

## 5. Security review of the code (prototype status)

- **Non-goals**: constant-time execution, side-channel resistance, key
  zeroization, cross-platform bit-reproducibility, production parameters.
  None are provided; Python/NumPy cannot provide the first two.
- **Randomness**: long-term secrets use `secrets` / `os.urandom`
  (TrapGen streams, TGS Shamir coefficients and PRF seeds, trilinear alpha,
  BEAT-MEV keys).  Deterministic tapes (`BitStream` over SHAKE-256) are used
  exactly where the paper requires determinism (Hsp -> SampleLeft) or where
  seeds are explicit test fixtures.  `numpy.default_rng` seeded from
  `secrets` is used for Shamir coefficients (uniform mod q via
  rejection-free 64-surplus-bit reduction would be preferable; bias is
  < 2^-64 with the current integers path since rng.integers(0, q) is exact).
- **Secret handling**: T_C, Shamir shares, and PRF seeds live in ordinary
  arrays/dicts; nothing is wiped.  The `Derivation` cache stores only public
  data (crs-derived matrices and SampleLeft outputs, which the paper itself
  makes public via the random tape).
- **Oracle domain separation**: every hash has a distinct prefix; Hsp^th and
  Hsp are separate domains, so a threshold transcript tape can never collide
  with a non-threshold tape.
- **Input validation**: batch size, index collisions, tape length, sigma vs
  GS-norm floor, share consistency (sidsp, c, C sbk = c), signature norm
  bound, punctured-point evaluation are all checked and tested.
- **Known sharp edges**: (i) `PreimageSampler._ext_cache` grows with distinct
  M1 matrices (bounded in practice by the derivation cache's 8-entry cap);
  (ii) ExplainSL's CRT recovery covers |z| < p1 p2 / 2 ~ 2^61 - beyond that
  it raises (never silently mis-explains); (iii) the sublattice path can
  produce Klein integer state of hundreds of bits, which is exact but slow -
  an adversarially chosen identity set could make derivation expensive
  (denial-of-service consideration only); (iv) highly structured label sets
  (e.g. sequential identities 1..ell whose H() labels are unit vectors) make
  the Lemma 31 block I - T G^{-1}(A) singular over R, pushing derivation onto
  the leftmost-rank-profile fallback: ~10x slower, correctness and all tests
  unaffected (measured: 12.3 s vs 1.0 s per tiny-preset batch).  Random /
  hash-derived tags - the intended usage - avoid it with overwhelming
  probability, but an adversary controlling tags can force the slow path
  (again a DoS consideration only).
