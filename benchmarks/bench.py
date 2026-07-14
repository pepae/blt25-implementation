"""Benchmark the BLT25 implementation.

Measures wall-clock times and object sizes for every operation of the BIBE,
TBIBE, and threshold-GPV schemes at the toy presets, plus the batch-size
scaling of pre-decryption/decryption, and prints the Theorem-2 (provable)
size table for reference.

Usage:  python3 benchmarks/bench.py [--presets tiny,small] [--medium]
Writes benchmarks/RESULTS.md and benchmarks/results.json.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt25 import bibe, gpv, tbibe          # noqa: E402
from blt25.params import (spec_params, threshold_bounds, toy_params,  # noqa: E402
                          toy_threshold_params)

OUT_MD = Path(__file__).parent / "RESULTS.md"
OUT_JSON = Path(__file__).parent / "results.json"


def timeit(fn, repeat=3):
    ts = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def rep_ids(count: int, d: int, salt: int = 0) -> tuple:
    """Representative identity tags: distinct pseudorandom ints < 2^d (as
    hash-derived tags would be).  Highly structured tags (e.g. 1..ell, whose
    labels are unit vectors) can make the Lemma-31 fast path singular and
    trigger the ~10x slower rank-profile fallback; that degenerate case is
    reported separately."""
    import numpy as np
    rng = np.random.default_rng(1234 + salt)
    ids = rng.choice(2 ** d, size=count, replace=False)
    return tuple(int(x) for x in ids)


def vec_bytes(v, per_entry_bits):
    return (len(v) * per_entry_bits + 7) // 8


def bench_bibe(name: str, results: dict):
    p = toy_params(name)
    row = {"preset": name, "n": p.n, "d": p.d, "ell": p.ell,
           "log2q": p.q.bit_length(),
           "pk_bytes": p.pk_bytes(), "ct_bytes": p.ct_bytes(),
           "sbk_bytes": p.sbk_bytes()}
    (pk, sk), row["setup_s"] = timeit(lambda: bibe.setup(p, seed=b"bench"), 1)
    ids = rep_ids(p.ell, p.d)
    _, row["enc_s"] = timeit(lambda: bibe.encrypt(pk, ids[0], 1), 5)
    cts = [bibe.encrypt(pk, r, i % 2) for i, r in enumerate(ids)]

    def cold_predec():
        bibe._derivation_cache.clear()
        return bibe.pre_dec(sk, ids)
    sbk, row["predec_cold_s"] = timeit(cold_predec, 1)
    _, row["predec_warm_s"] = timeit(lambda: bibe.pre_dec(sk, ids), 2)

    def cold_dec():
        bibe._derivation_cache.clear()
        return bibe.decrypt(pk, sbk, cts, return_error=True)
    (msgs, errs), row["dec_cold_s"] = timeit(cold_dec, 1)
    (msgs, errs), row["dec_warm_s"] = timeit(
        lambda: bibe.decrypt(pk, sbk, cts, return_error=True), 2)
    assert msgs == [i % 2 for i in range(len(ids))], "benchmark decrypt failed"
    row["max_err_over_q4"] = max(errs) / (p.q // 4)
    row["sbk_norm"] = float(np.linalg.norm(sbk.sbks[0].astype(np.float64)))
    row["sbk_actual_bytes"] = vec_bytes(
        sbk.sbks[0], max(1, math.ceil(math.log2(2 * abs(int(
            np.abs(sbk.sbks[0]).max())) + 2))))
    row["ell_ct_bytes"] = p.ell * p.ct_bytes()
    results["bibe"].append(row)
    return p, pk, sk


def bench_scaling(name: str, results: dict):
    """Pre-decryption / decryption scaling in the batch size ell' <= ell."""
    p = toy_params(name)
    pk, sk = bibe.setup(p, seed=b"bench-scale")
    for ellp in range(1, p.ell + 1):
        ids = rep_ids(ellp, p.d, salt=ellp)
        bibe._derivation_cache.clear()
        t0 = time.perf_counter()
        sbk = bibe.pre_dec(sk, ids)
        t_pd = time.perf_counter() - t0
        cts = [bibe.encrypt(pk, r, 1) for r in ids]
        bibe._derivation_cache.clear()
        t0 = time.perf_counter()
        msgs = bibe.decrypt(pk, sbk, cts)
        t_dec = time.perf_counter() - t0
        assert msgs == [1] * ellp
        results["scaling"].append({
            "preset": name, "ell_prime": ellp,
            "predec_cold_s": t_pd, "dec_cold_s": t_dec,
            "sbk_bytes": p.sbk_bytes(), "batch_ct_bytes": ellp * p.ct_bytes()})


def bench_threshold(name: str, N: int, tau: int, results: dict):
    p = toy_threshold_params(name)
    row = {"preset": name, "N": N, "tau": tau, "log2q": p.q.bit_length(),
           "sigma_flood": p.sigma_flood}
    (pk, parties, _sk), row["setup_s"] = timeit(
        lambda: tbibe.setup(p, N, tau, seed=b"bench-th"), 1)
    ids = (5, 11)
    cts = [tbibe.encrypt(pk, r, i % 2) for i, r in enumerate(ids)]
    act = tuple(range(1, tau + 1))
    sid = 1

    t0 = time.perf_counter()
    r1out = {i: tbibe.predec_one(parties[i - 1], sid, act, ids) for i in act}
    row["predec_r1_s"] = (time.perf_counter() - t0) / tau
    r1 = {i: o.msg for i, o in r1out.items()}
    bibe._derivation_cache.clear()
    t0 = time.perf_counter()
    r2 = {i: tbibe.predec_two(parties[i - 1], r1out[i].state, sid, act, ids, r1)
          for i in act}
    row["predec_r2_total_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pdk = tbibe.combine(pk, sid, act, ids, r1, r2)
    row["combine_s"] = time.perf_counter() - t0
    assert pdk is not None
    t0 = time.perf_counter()
    msgs = tbibe.decrypt(pk, pdk, cts)
    row["dec_warm_s"] = time.perf_counter() - t0
    assert msgs == [0, 1]
    row["B_th"] = threshold_bounds(p, N, tau)["B_th"]
    row["sbk_norm"] = float(np.linalg.norm(pdk.sbks[0].astype(np.float64)))
    # per-party communication: round1 (e_i, m_i), round2 sbk_i
    lg = p.q.bit_length()
    row["comm_r1_bytes"] = ((p.n + p.m_c) * lg + 7) // 8
    row["comm_r2_bytes"] = (p.m_c * lg + 7) // 8
    results["tbibe"].append(row)


def bench_gpv(name: str, N: int, tau: int, results: dict):
    p = toy_threshold_params(name)
    row = {"preset": name, "N": N, "tau": tau}
    (pk, parties), row["setup_s"] = timeit(
        lambda: gpv.setup(p, N, tau, seed=b"bench-gpv"), 1)
    msg = b"benchmark message"
    act = tuple(range(1, tau + 1))
    t0 = time.perf_counter()
    r1, st = {}, {}
    for i in act:
        m1, s = gpv.sign_one(parties[i - 1], 1, act, msg)
        r1[i], st[i] = m1, s
    row["sign_r1_total_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    r2 = {i: gpv.sign_two(parties[i - 1], st[i], 1, act, msg, r1) for i in act}
    row["sign_r2_total_s"] = time.perf_counter() - t0
    sig, row["combine_s"] = timeit(
        lambda: gpv.combine(pk, 1, act, msg, r1, r2), 1)
    ok, row["verify_s"] = timeit(lambda: gpv.verify(pk, msg, sig), 5)
    assert ok
    row["sig_bytes"] = vec_bytes(sig.sig, max(
        1, math.ceil(math.log2(2 * abs(int(np.abs(sig.sig).max())) + 2)))) + 32
    results["gpv"].append(row)


def spec_size_table(results: dict):
    for lam, ell in [(128, 64), (128, 512), (128, 4096)]:
        p = spec_params(lam, ell)
        results["spec"].append({
            "lambda": lam, "ell": ell, "log2q": p.q.bit_length(),
            "n": p.n, "t": p.t,
            "pk_MiB": p.pk_bytes() / 2**20,
            "ct_KiB": p.ct_bytes() / 2**10,
            "sbk_KiB": p.sbk_bytes() / 2**10,
            "ell_ct_MiB": ell * p.ct_bytes() / 2**20})


def render(results: dict) -> str:
    L = ["# BLT25 benchmark results", "",
         "Times are wall-clock seconds (median), single-threaded Python/NumPy.",
         "Toy parameters: NOT cryptographically secure (see PARAMS.md).", ""]
    L.append("## BIBE (Section 5)\n")
    L.append("| preset | n | d | ell | log2 q | setup | enc | predec (cold/warm) | dec (cold/warm) | pk | ct | sbk | ell x ct | err/q4 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results["bibe"]:
        L.append("| {preset} | {n} | {d} | {ell} | {log2q} | {setup_s:.2f}s | "
                 "{enc_s:.3f}s | {predec_cold_s:.2f}/{predec_warm_s:.2f}s | "
                 "{dec_cold_s:.2f}/{dec_warm_s:.2f}s | {pk_kib:.0f} KiB | "
                 "{ct_kib:.1f} KiB | {sbk_b} B | {lct_kib:.0f} KiB | "
                 "{max_err_over_q4:.3f} |".format(
                     pk_kib=r["pk_bytes"] / 1024, ct_kib=r["ct_bytes"] / 1024,
                     sbk_b=r["sbk_actual_bytes"],
                     lct_kib=r["ell_ct_bytes"] / 1024, **r))
    L.append("\nIdentity tags are representative pseudorandom values; highly "
             "structured tags (e.g. sequential 1..ell giving unit-vector "
             "labels) can push per-batch derivation onto the ~10x slower "
             "rank-profile fallback (correctness unaffected).\n")
    L.append("\n## Batch-size scaling (cold derivations)\n")
    L.append("| preset | ell' | predec | dec | sbk | batch ct |")
    L.append("|---|---|---|---|---|---|")
    for r in results["scaling"]:
        L.append("| {preset} | {ell_prime} | {predec_cold_s:.2f}s | "
                 "{dec_cold_s:.2f}s | {sbk_bytes} B | {b:.0f} KiB |".format(
                     b=r["batch_ct_bytes"] / 1024, **r))
    L.append("\n## TBIBE (Section 6, two-round threshold pre-decryption)\n")
    L.append("| preset | N | tau | log2 q | setup | r1/party | r2 total | combine | dec | comm r1+r2/party | ||sbk|| / B_th |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results["tbibe"]:
        L.append("| {preset} | {N} | {tau} | {log2q} | {setup_s:.2f}s | "
                 "{predec_r1_s:.3f}s | {predec_r2_total_s:.2f}s | "
                 "{combine_s:.3f}s | {dec_warm_s:.2f}s | {c1}+{c2} KiB | "
                 "{ratio:.3f} |".format(
                     c1=round(r["comm_r1_bytes"] / 1024, 1),
                     c2=round(r["comm_r2_bytes"] / 1024, 1),
                     ratio=r["sbk_norm"] / r["B_th"], **r))
    L.append("\n## Threshold GPV signatures (Section 7)\n")
    L.append("| preset | N | tau | setup | sign r1 | sign r2 | combine | verify | sig |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results["gpv"]:
        L.append("| {preset} | {N} | {tau} | {setup_s:.2f}s | "
                 "{sign_r1_total_s:.3f}s | {sign_r2_total_s:.3f}s | "
                 "{combine_s:.3f}s | {verify_s:.4f}s | {sig_bytes} B |".format(**r))
    L.append("\n## Theorem-2 (provable) parameter sizes, for reference\n")
    L.append("These are the sizes the paper's proofs require at lambda = 128; "
             "far beyond what this prototype executes.\n")
    L.append("| lambda | ell | log2 q | t | pk | ct | sbk | ell x ct |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results["spec"]:
        L.append("| {lambda} | {ell} | {log2q} | {t} | {pk_MiB:.0f} MiB | "
                 "{ct_KiB:.0f} KiB | {sbk_KiB:.0f} KiB | {ell_ct_MiB:.0f} MiB |"
                 .format(**r))
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="tiny,small")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--tau", type=int, default=3)
    args = ap.parse_args()
    presets = args.presets.split(",")
    results = {"bibe": [], "scaling": [], "tbibe": [], "gpv": [], "spec": []}
    for name in presets:
        print(f"[bench] BIBE {name} ...", flush=True)
        bench_bibe(name, results)
        print(f"[bench] scaling {name} ...", flush=True)
        bench_scaling(name, results)
        print(f"[bench] TBIBE {name} ...", flush=True)
        bench_threshold(name, args.N, args.tau, results)
        print(f"[bench] GPV {name} ...", flush=True)
        bench_gpv(name, args.N, args.tau, results)
    spec_size_table(results)
    OUT_MD.write_text(render(results))
    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f"[bench] wrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
