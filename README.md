# blt25 - Threshold Batch IBE without Epochs (implementation)

A complete research-prototype implementation of

> Dan Boneh, Evan Laufer, Ertem Nusret Tas,
> **"Threshold Batch Identity-based Encryption without Epochs"**
> (Stanford University).

Everything in the paper is implemented: the explainable-SampleLeft machinery
(Section 4, Appendix C), the lattice batch-IBE scheme (Section 5), the
two-round threshold Gaussian preimage sampler and threshold batch IBE
(Section 6), threshold GPV signatures (Section 7), the KP-ABE-based batch IBE
(Section 8, with a concrete BGG+14-style ABE for the membership class), the
trilinear-map scheme in a simulated generic group (Section 9, including the
partial-fraction threshold of 9.4), the lattice BEAT-MEV sketch from the BLMR
key-homomorphic PRF (Appendix A.4), and the Waters-Wee-Wu shifted
multi-preimage trapdoor sampler with d-bit labels (Appendix B, with the
sparse indicator-function evaluation that keeps Expand polynomial).

> **SECURITY WARNING.**  Research prototype.  The runnable parameter presets
> are toy-sized and provide **no** cryptographic security; the code is not
> constant-time and does not manage secrets.  See PARAMS.md and AUDIT.md.

## Layout

```
blt25/
  modq.py       Z_q linear algebra (overflow-safe chunked int64 + exact paths)
  bits.py       deterministic SHAKE-256 bit streams (random tapes)
  gadget.py     gadget matrix G, G^{-1}, basis of Lambda^perp(G)
  gaussian.py   explainable 1-D discrete Gaussian sampler (3 width regimes)
  trapdoor.py   TrapGen, Ajtai bases (Lemma 31 + exact structural basis),
                DetExtBasis, Klein sampler, SamplePre/SampleLeft/ExplainSL
  www24.py      WWW24 sampler with d-bit labels: EvalF/EvalFX (indicators),
                StructTrapGen, Gen/Expand/ExpandLocal/SampleMultPre/GenProg
  hashes.py     H, Hro, Hsp, Hsp^th, Htr, HGPV, PRF (SHAKE-256, domain-sep)
  params.py     Theorem-2/3/4 derivation + checker; toy presets
  bibe.py       Section 5 batch IBE (+ multi-bit variant of 5.3)
  tgs.py        Section 6.1 threshold Gaussian preimage sampling
  tbibe.py      Section 6.2 threshold batch IBE
  gpv.py        Section 7 threshold GPV signatures
  abe.py        Section 8 KP-ABE -> BIBE (BGG+14-style instantiation)
  trilinear.py  Section 9 (simulated generic trilinear group)
  beatmev.py    Appendix A.4 (BLMR puncturable PRF + chunked-Regev transport)
tests/          84 tests: identities, determinism, explainability, statistics,
                end-to-end + negative tests for every scheme
benchmarks/     bench.py -> RESULTS.md (timings, sizes, scaling)
scripts/        calibrate.py, verify_deep.py -> VERIFICATION.md
AUDIT.md        paper-conformance map, documented deviations, security review
PARAMS.md       provable vs toy parameters
```

## Quick start

```bash
pip install numpy mpmath pytest
python3 -m pytest tests/ -q          # ~40 s
python3 scripts/verify_deep.py 16    # deep verification -> VERIFICATION.md
python3 benchmarks/bench.py          # -> benchmarks/RESULTS.md
```

### Batch IBE (Section 5)

```python
from blt25 import bibe
from blt25.params import toy_params

p = toy_params("tiny")                    # NOT SECURE - demo sizes
pk, sk = bibe.setup(p)
ids = (3, 9)                              # identity tags (any distinct ints < 2^d)
cts = [bibe.encrypt(pk, r, bit) for r, bit in zip(ids, [1, 0])]
sbk = bibe.pre_dec(sk, ids)               # short pre-decryption key
assert bibe.decrypt(pk, sbk, cts) == [1, 0]
```

### Threshold batch IBE (Section 6), 3-of-4

```python
from blt25 import tbibe
from blt25.params import toy_threshold_params

p = toy_threshold_params("tiny")
pk, parties, _ = tbibe.setup(p, N=4, tau=3)
ids = (5, 11)
cts = [tbibe.encrypt(pk, r, b) for r, b in zip(ids, [1, 0])]
sid, act = 1, (1, 2, 4)
r1o = {i: tbibe.predec_one(parties[i-1], sid, act, ids) for i in act}
r1 = {i: o.msg for i, o in r1o.items()}
r2 = {i: tbibe.predec_two(parties[i-1], r1o[i].state, sid, act, ids, r1)
      for i in act}
pdk = tbibe.combine(pk, sid, act, ids, r1, r2)
assert tbibe.decrypt(pk, pdk, cts) == [1, 0]
```

### Threshold GPV signatures (Section 7)

```python
from blt25 import gpv
from blt25.params import toy_threshold_params

pk, parties = gpv.setup(toy_threshold_params("tiny"), N=4, tau=3)
msg, sid, act = b"hello", 1, (1, 2, 3)
r1, st = {}, {}
for i in act:
    r1[i], st[i] = gpv.sign_one(parties[i-1], sid, act, msg)
r2 = {i: gpv.sign_two(parties[i-1], st[i], sid, act, msg, r1) for i in act}
sig = gpv.combine(pk, sid, act, msg, r1, r2)
assert gpv.verify(pk, msg, sig)
```

## What the scheme buys (and what the benchmarks show)

For an encrypted mempool holding `ell` ciphertexts, publishing per-ciphertext
decryption material costs `ell x |ct|`-scale bandwidth; this scheme publishes
one short `sbk` (plus, in the threshold case, tau syndromes/shares) whose
size is polylog in `ell` - with **no epochs**: any number of batches can be
opened under one public key, and tags never coordinate.  The benchmark report
includes both the measured toy sizes and the Theorem-2 (provable) size table;
at lambda = 128, ell = 4096 the pre-decryption key is ~5 orders of magnitude
smaller than the batch it opens.

## Fidelity and deviations

The implementation follows the paper algorithm-by-algorithm (the conformance
map in AUDIT.md lists every object).  Deviations are documented in
AUDIT.md section 3; the two worth knowing about before reading code:

1. **Lemma 31 as literally stated can yield a proper sublattice** of
   Lambda^perp(A) (measured index ~2^28 on a small instance).  For
   TrapGen-shaped matrices we construct a provably exact basis instead; for
   the WWW24 batch matrix we keep the paper-literal selection (correctness,
   determinism and explainability are unaffected; the distribution-level
   claim of the security proof is what would need the exact basis).
2. **Toy parameters** are calibrated for correctness margins, not security.

## License / provenance

Implementation written from the paper's text alone (no reference code
existed); pdf -> `paper.txt` extraction used for fidelity checks.
