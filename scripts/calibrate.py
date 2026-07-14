"""Calibrate toy parameter presets: measure extended-basis GS norms and
decryption error margins across random batches."""
import sys, time, math
import numpy as np
sys.path.insert(0, '.')
from blt25 import bibe, www24
from blt25.params import toy_params
from blt25.bits import stream_from_bytes, random_stream
from blt25.hashes import hash_index, hash_ro

name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
p = toy_params(name)
print(f"preset={name}: n={p.n} d={p.d} ell={p.ell} q={p.q} (2^{p.q.bit_length()}) "
      f"mt={p.mt} t={p.t} m_ro={p.m_ro} sigma1={p.sigma1} sigma2={p.sigma2}")

pk, sk = bibe.setup(p, seed=b"calib")
max_gs_seen, err_seen = 0.0, 0
rng = np.random.default_rng(7)
for trial in range(trials):
    ids = [int(x) for x in rng.choice(2**p.d, size=p.ell, replace=False)]
    t0 = time.time()
    der = bibe.derive(pk, ids)
    gs = der.exp.sampler().max_gs
    max_gs_seen = max(max_gs_seen, gs)
    sbk = bibe.pre_dec(sk, ids)
    cts = [bibe.encrypt(pk, r, 1 if i % 2 else 0) for i, r in enumerate(ids)]
    msgs, errs = bibe.decrypt(pk, sbk, cts, return_error=True)
    ok = msgs == [1 if i % 2 else 0 for i in range(len(ids))]
    err_seen = max(err_seen, max(errs))
    print(f"  trial {trial}: gs={gs:.0f} decrypt_ok={ok} max|err|={max(errs)} "
          f"q/4={p.q//4} margin={p.q//4/max(1,max(errs)):.1f}x  ({time.time()-t0:.1f}s)")
    bibe._derivation_cache.clear()
print(f"SUMMARY {name}: max_gs={max_gs_seen:.0f} (sigma1={p.sigma1}, "
      f"ratio={p.sigma1/max_gs_seen:.2f}x)  max_err={err_seen} "
      f"(q/4={p.q//4}, margin={p.q//4/max(1,err_seen):.1f}x)")
