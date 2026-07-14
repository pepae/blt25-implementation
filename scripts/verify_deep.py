"""Deep verification run: many-batch decryption reliability, sampler
statistics at scale, basis-exactness diagnostics, and the Theorem-2/3/4
constraint system.  Slower than the unit tests; produces VERIFICATION.md.

Usage: python3 scripts/verify_deep.py [trials]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt25 import bibe, tbibe, gpv                                  # noqa: E402
from blt25.bits import stream_from_bytes                            # noqa: E402
from blt25.gaussian import sample_z, sd_of                          # noqa: E402
from blt25.params import (check_theorem2, spec_params,              # noqa: E402
                          threshold_bounds, toy_params,
                          toy_threshold_params)
from blt25.trapdoor import check_exactness                          # noqa: E402

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
report = ["# Deep verification report", ""]


def log(s=""):
    print(s, flush=True)
    report.append(s)


# ---------------------------------------------------------------------------
log("## 1. BIBE decryption reliability (tiny preset)")
p = toy_params("tiny")
pk, sk = bibe.setup(p, seed=b"verify-deep")
rng = np.random.default_rng(2026)
errs, gss, exact_flags, fails = [], [], [], 0
t0 = time.time()
for trial in range(TRIALS):
    ids = tuple(int(x) for x in rng.choice(2 ** p.d, size=p.ell, replace=False))
    bits = [int(b) for b in rng.integers(0, 2, size=p.ell)]
    bibe._derivation_cache.clear()
    der = bibe.derive(pk, ids)
    gss.append(der.exp.sampler().max_gs)
    exact_flags.append(check_exactness(der.exp.sampler().ajtai, p.q,
                                       p.n * len(ids)))
    sbk = bibe.pre_dec(sk, ids)
    cts = [bibe.encrypt(pk, r, b) for r, b in zip(ids, bits)]
    msgs, es = bibe.decrypt(pk, sbk, cts, return_error=True)
    errs.extend(es)
    if msgs != bits:
        fails += 1
log(f"- {TRIALS} random batches x {p.ell} ciphertexts: "
    f"**{fails} decryption failures**")
log(f"- error magnitudes: max {max(errs):,} = {max(errs)/(p.q//4):.4f} of q/4; "
    f"mean {np.mean(errs):,.0f}")
log(f"- extended-basis GS norms: max {max(gss):.0f} "
    f"(sigma1 = {p.sigma1:.0f}, margin {p.sigma1/max(gss):.2f}x)")
log(f"- Lemma-31 selection produced an exact Lambda^perp basis in "
    f"{sum(exact_flags)}/{len(exact_flags)} batches "
    f"(sublattice otherwise - documented deviation, AUDIT.md S3)")
log(f"- wall time {time.time()-t0:.0f}s")
log()

# ---------------------------------------------------------------------------
log("## 2. 1-D sampler statistics at scale")
for sigma in (10.0, 300.0, 2.0 ** 20):
    st = stream_from_bytes(b"deep%d" % int(math.log2(sigma) * 100))
    nsamp = 20_000 if sigma < 2 ** 16 else 2_000
    xs = np.array([sample_z(0.0, sigma, st) for _ in range(nsamp)],
                  dtype=np.float64)
    s = sd_of(sigma)
    zmean, zstd = xs.mean() / s, xs.std() / s
    log(f"- sigma = 2^{math.log2(sigma):.1f}: n={nsamp}, mean/s = {zmean:+.4f} "
        f"(tol {4/math.sqrt(nsamp):.4f}), std/s = {zstd:.4f} (tol 5%)")
    assert abs(zmean) < 4 / math.sqrt(nsamp) and abs(zstd - 1) < 0.05
log()

# ---------------------------------------------------------------------------
log("## 3. TBIBE + GPV reliability (tiny threshold preset)")
pth = toy_threshold_params("tiny")
N, tau = 4, 3
pk_t, parties, _ = tbibe.setup(pth, N, tau, seed=b"verify-th")
th_fails = 0
for trial in range(max(3, TRIALS // 4)):
    ids = tuple(int(x) for x in rng.choice(2 ** pth.d, size=pth.ell,
                                           replace=False))
    bits = [int(b) for b in rng.integers(0, 2, size=pth.ell)]
    act = tuple(sorted(rng.choice(np.arange(1, N + 1), size=tau,
                                  replace=False).tolist()))
    sid = 5000 + trial
    r1o = {i: tbibe.predec_one(parties[i - 1], sid, act, ids) for i in act}
    r1 = {i: o.msg for i, o in r1o.items()}
    r2 = {i: tbibe.predec_two(parties[i - 1], r1o[i].state, sid, act, ids, r1)
          for i in act}
    pdk = tbibe.combine(pk_t, sid, act, ids, r1, r2)
    cts = [tbibe.encrypt(pk_t, r, b) for r, b in zip(ids, bits)]
    if pdk is None or tbibe.decrypt(pk_t, pdk, cts) != bits:
        th_fails += 1
    bibe._derivation_cache.clear()
log(f"- {max(3, TRIALS//4)} random (batch, act) threshold pre-decryptions: "
    f"**{th_fails} failures**")
b = threshold_bounds(pth, N, tau)
log(f"- B_th = {b['B_th']:.3g} vs q/2 = {pth.q/2:.3g}")

pk_g, parties_g = gpv.setup(pth, N, tau, seed=b"verify-gpv")
gpv_fails = 0
for trial in range(max(3, TRIALS // 4)):
    msg = b"verify-%d" % trial
    act = tuple(sorted(rng.choice(np.arange(1, N + 1), size=tau,
                                  replace=False).tolist()))
    r1, st1 = {}, {}
    for i in act:
        m1, s1 = gpv.sign_one(parties_g[i - 1], trial, act, msg)
        r1[i], st1[i] = m1, s1
    r2 = {i: gpv.sign_two(parties_g[i - 1], st1[i], trial, act, msg, r1)
          for i in act}
    sig = gpv.combine(pk_g, trial, act, msg, r1, r2)
    if sig is None or not gpv.verify(pk_g, msg, sig) \
            or gpv.verify(pk_g, msg + b"x", sig):
        gpv_fails += 1
log(f"- {max(3, TRIALS//4)} threshold GPV sign/verify cycles: "
    f"**{gpv_fails} failures**")
log()

# ---------------------------------------------------------------------------
log("## 4. Theorem 2/3/4 constraint system (provable parameters)")
for lam, ell in [(80, 64), (128, 512), (128, 4096), (192, 1024)]:
    sp = spec_params(lam, ell)
    checks = check_theorem2(sp)
    bad = [k for k, v in checks.items() if not v[0]]
    status = "all satisfied" if not bad else f"VIOLATED: {bad}"
    log(f"- lambda={lam}, ell={ell}: log2 q = {sp.q.bit_length()}, "
        f"{len(checks)} constraints -> **{status}**")
    assert not bad
tb = threshold_bounds(spec_params(128, 512), N=16, tau=11)
log(f"- threshold bounds (lambda=128, ell=512, 11-of-16): "
    f"B_th = 2^{math.log2(tb['B_th']):.1f}, "
    f"q required >= 2^{math.log2(tb['q_required']):.1f}")
log()

# ---------------------------------------------------------------------------
log("## 5. Toy-vs-provable gap (honesty check)")
pt = toy_params("tiny")
checks = check_theorem2(pt)
violated = [k for k, v in checks.items() if not v[0]]
log(f"- tiny preset intentionally violates {len(violated)} provable "
    f"constraints: {violated}")
log("- (expected: toy parameters trade the security-proof constraints for "
    "runnability; correctness constraints all hold empirically, section 1)")
log()

Path(__file__).resolve().parents[1].joinpath("VERIFICATION.md").write_text(
    "\n".join(report) + "\n")
print("\nwrote VERIFICATION.md")
