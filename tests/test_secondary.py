"""Tests for the secondary constructions: ABE-based BIBE (Section 8),
trilinear-map BIBE in the simulated GGM (Section 9), lattice BEAT-MEV
(Appendix A.4), and the parameter derivations (Theorems 2/3/4)."""

from dataclasses import replace

import numpy as np
import pytest

from blt25 import abe, beatmev, trilinear
from blt25.modq import next_prime
from blt25.params import (check_theorem2, spec_params, threshold_bounds,
                          toy_params, toy_threshold_params)

# ---------------------------------------------------------------------------
# Section 8: KP-ABE and the generic transformation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def abe_keys(tiny):
    p = replace(tiny, ell=3)
    return abe.setup(p, seed=b"abe-tests")


def test_abe_membership_policy(abe_keys):
    mpk, msk = abe_keys
    S = [(0, 1, 0, 0), (1, 1, 0, 1), (0, 0, 1, 1)]
    fkey = abe.keygen(msk, S)
    for x in S:
        for mu in (0, 1):
            ct = abe.encrypt(mpk, x, mu)
            assert abe.decrypt(mpk, fkey, ct) == mu
    ct_out = abe.encrypt(mpk, (1, 0, 0, 0), 1)
    assert abe.decrypt(mpk, fkey, ct_out) is None      # f_S(x) != 0


def test_abe_bibe_wrapper(abe_keys):
    mpk, msk = abe_keys
    p = mpk.params
    pk, sk = abe.bibe_setup(p, seed=b"abe-bibe")
    ids = [3, 9, 14]
    bits = [1, 0, 1]
    cts = [abe.bibe_encrypt(pk, r, b) for r, b in zip(ids, bits)]
    fk = abe.bibe_pre_dec(sk, ids)
    assert abe.bibe_decrypt(pk, fk, cts) == bits
    # key size is independent of the batch (single SampleLeft preimage)
    assert fk.sk.shape == (p.m_c + p.mt,)


def test_abe_set_size_cap(abe_keys):
    mpk, msk = abe_keys
    with pytest.raises(AssertionError):
        abe.keygen(msk, [(0, 0, 0, i % 2) for i in range(2)] +
                   [(1, 1, 1, 0), (1, 1, 0, 0)])       # 4 > ell = 3


# ---------------------------------------------------------------------------
# Section 9: trilinear-map BIBE (simulated GGM)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tri():
    p = next_prime(1 << 61)
    grp = trilinear.TrilinearGroup(p, seed=b"tri-tests")
    pk, alpha = trilinear.setup(grp, ell=4)
    return grp, pk, alpha


def test_trilinear_roundtrip(tri):
    grp, pk, alpha = tri
    ids = [12345, 999, 31337, 42]
    msgs = [b"m%d-payload" % i for i in range(4)]
    cts = [trilinear.encrypt(pk, r, m) for r, m in zip(ids, msgs)]
    sbk = trilinear.pre_dec(pk, alpha, ids)
    assert trilinear.decrypt(pk, sbk, cts) == msgs


def test_trilinear_sbk_is_single_element(tri):
    grp, pk, alpha = tri
    sbk = trilinear.pre_dec(pk, alpha, [1, 2, 3, 4])
    assert isinstance(sbk, bytes) and len(sbk) == 16   # one G3 label


def test_trilinear_wrong_key_fails(tri):
    grp, pk, alpha = tri
    ids = [11, 22, 33, 44]
    cts = [trilinear.encrypt(pk, r, b"secret!!") for r in ids]
    sbk_other = trilinear.pre_dec(pk, alpha, [55, 66, 77, 88])
    out = trilinear.decrypt(pk, sbk_other, cts)
    assert out[0] != b"secret!!"


def test_trilinear_threshold_partial_fractions(tri):
    grp, pk, alpha = tri
    ids = [12, 34, 56, 78]
    shares = trilinear.shamir_share_alpha(alpha, N=5, tau=3, p=grp.p)
    direct = trilinear.pre_dec(pk, alpha, ids)
    for act in [(1, 2, 3), (2, 4, 5), (1, 3, 5)]:
        got = trilinear.threshold_pre_dec(pk, {i: shares[i] for i in act},
                                          act, ids)
        assert got == direct


# ---------------------------------------------------------------------------
# Appendix A.4: BEAT-MEV from the BLMR PRF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bm():
    return beatmev.BeatMev(seed=b"bm-tests")


def test_blmr_key_homomorphism(bm):
    import secrets
    p = bm.p
    k1 = np.array([secrets.randbelow(p.q) for _ in range(p.m)], dtype=object)
    k2 = np.array([secrets.randbelow(p.q) for _ in range(p.m)], dtype=object)
    idx = 6
    lhs = bm.prf.eval((k1 + k2) % p.q, idx)
    rhs = (bm.prf.eval(k1, idx) + bm.prf.eval(k2, idx)) % p.p
    diff = (lhs - rhs) % p.p
    diff = np.minimum(diff, p.p - diff)
    assert diff.max() <= 2                              # BLMR error in {0,1,2}


def test_blmr_puncture_eval(bm):
    import secrets
    p = bm.p
    sk = np.array([secrets.randbelow(p.q) for _ in range(p.m)], dtype=object)
    star = 5
    punc = bm.prf.puncture(sk, star)
    for idx in (0, 3, 9, 15):
        full = bm.prf.eval(sk, idx)
        approx = bm.prf.punc_eval(punc, idx)
        diff = (full - approx) % p.p
        diff = np.minimum(diff, p.p - diff)
        assert diff.max() <= 4, idx                     # p' absorbs the noise
    with pytest.raises(AssertionError):
        bm.prf.punc_eval(punc, star)


def test_beatmev_roundtrip(bm):
    msgs = [[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 1, 1]]
    cts = [bm.encrypt(i, m) for i, m in zip([3, 7, 12], msgs)]
    sbk = bm.pre_dec(cts)
    assert bm.decrypt(sbk, cts, nbits=4) == msgs


def test_beatmev_distinct_indices_required(bm):
    cts = [bm.encrypt(3, [1]), bm.encrypt(3, [0])]
    with pytest.raises(AssertionError):
        bm.pre_dec(cts)


# ---------------------------------------------------------------------------
# Parameters: Theorem 2/3/4 constraint system
# ---------------------------------------------------------------------------

def test_spec_params_satisfy_theorem2():
    p = spec_params(lam=128, ell=512)
    checks = check_theorem2(p)
    failures = {k: v for k, v in checks.items() if not v[0]}
    assert not failures, failures


def test_spec_params_sizes_scale_polylog_in_ell():
    a = spec_params(lam=128, ell=64)
    b = spec_params(lam=128, ell=4096)
    # |sbk| grows at most polylog in ell (Table 1): allow a generous factor
    assert b.sbk_bytes() < 4 * a.sbk_bytes()
    # and is far below ell * |ct| (the succinctness win)
    assert b.sbk_bytes() * 100 < 4096 * b.ct_bytes()


def test_toy_params_flag_and_bounds():
    for name in ("tiny", "small"):
        p = toy_params(name)
        assert p.toy
        b = threshold_bounds(toy_threshold_params(name), 4, 3)
        assert b["B_th"] < toy_threshold_params(name).q / 2
