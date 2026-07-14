# Parameters

Two regimes are provided; understanding the split is essential to
interpreting anything this repository does.

## 1. Provable parameters (`params.spec_params`)

`spec_params(lam, ell, d)` implements the derivation of Theorems 2/3 with the
concrete slack convention `omega(f) := f * log2(lambda)`, iterating the
modulus to a fixed point:

- n = lambda, m_ro = m_c = 3 n ceil(log q), m~ = n (floor(log q) + 1),
  t = m~ (d + 1), d = 2 lambda by default;
- sigma1 = max(18 (2 ell t + m~)^{3/2} log((2 ell t + m~) lambda) loglog q,
               2 m^{3/2} omega(log(m + t)));
- sigma2 = sqrt(m_c) omega(sqrt(log n));
- q = next_prime around (sigma1 mu^{3/2} + sigma2 sqrt(m_c) sqrt(mu))
  omega(log mu), mu = max(t, m_ro);
- alpha q = sigma_chi from Theorem 2's alpha bound;
- threshold: sigma_flood = 2^lambda (Remark 3) and B_th, B_td from Theorem 4
  via `params.threshold_bounds`.

`params.check_theorem2` re-checks the whole constraint system; the deep
verification run exercises it at lambda in {80, 128, 192}.  Typical outcome at
lambda = 128, ell = 512: log2 q = 100, |pk| ~ hundreds of MiB, |ct| ~ MiB
scale, |sbk| ~ tens of KiB (see benchmarks/RESULTS.md, "Theorem-2 sizes").
These parameters are *not runnable* by this prototype - and that is expected:
they exist so the constraint system itself is testable and the asymptotics
reportable.

## 2. Toy parameters (`params.toy_params`, `params.toy_threshold_params`)

| preset | n | d | ell | log2 q | sigma1 | sigma2 | sigma_chi |
|---|---|---|---|---|---|---|---|
| tiny | 4 | 4 | 2 | 28 | 4000 | 400 | 6 |
| small | 4 | 6 | 4 | 29 | 6000 | 450 | 6 |
| medium | 8 | 6 | 4 | 30 | 14000 | 900 | 6 |

Threshold variants add one modulus bit and sigma_flood = 2^14.

Calibration methodology (`scripts/calibrate.py`):

- sigma1 is set >= ~4x the maximum Gram-Schmidt norm of the extended basis
  observed over many random batches (the run-time check in `bibe.derive`
  raises if a batch ever exceeds it);
- sigma2 >= ~4x the C-basis GS norm;
- q is chosen so the worst observed decryption error stays below ~1/8 of
  q/4 (deep verification: max 7.8% of q/4 over all trials);
- the int64 fast path requires q < 2^31 (beyond that the code falls back to
  exact big-int arithmetic in the few places it matters, but the mod-q
  matmuls would need the object backend - not enabled in presets).

**Toy parameters provide no security.**  They intentionally violate exactly
the two Theorem-2 constraints that exist for the security proof
(sigma1's smudging-scale lower bound, and the alpha upper bound); the deep
verification report prints the violated set.  n = 4 lattices are trivially
breakable regardless.

## 3. What would production parameters look like?

Per Theorem 2 at lambda = 128: n = 128, log q ~ 100, m~ ~ 12,928,
t = m~ (d + 1) with d = 256 -> t ~ 3.3M columns; the SampleLeft dimension for
a batch of ell = 512 is ell t + m~ + ell m ~ 1.7e9.  A practical deployment
would need the ring/module setting (the paper's constructions are stated over
plain Z_q lattices), hardware acceleration, and a streaming Klein sampler -
all far outside a NumPy prototype's scope.  The asymptotic win the scheme
buys - |sbk| polylog in ell versus ell ciphertexts - is visible in the
Theorem-2 size table produced by the benchmarks.
