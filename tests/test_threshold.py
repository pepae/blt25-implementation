"""Tests for the threshold Gaussian sampler (Section 6.1), TBIBE (Section 6.2)
and threshold GPV signatures (Section 7)."""

import numpy as np
import pytest

from blt25 import bibe, gpv, tbibe, tgs
from blt25.bits import stream_from_bytes
from blt25.modq import matmul_q
from blt25.params import threshold_bounds
from blt25.trapdoor import trapgen

N_PARTIES, TAU = 4, 3
IDS = (5, 11)


@pytest.fixture(scope="module")
def tgs_setup(tiny_th):
    p = tiny_th
    C, T_C = trapgen(p.n, p.q, p.m_c, stream_from_bytes(b"tgs"))
    keys = tgs.setup(T_C, N_PARTIES, TAU, p.q)
    return p, C, T_C, keys


def run_tgs(p, C, keys, act, sid, c):
    r1, st = {}, {}
    for i in act:
        msg, s = tgs.round_one(keys[i - 1], sid, act, C, p.q, p.sigma_flood)
        r1[i], st[i] = msg, s
    r2 = {i: tgs.round_two(keys[i - 1], st[i], sid, act, C, p.q, c, r1)
          for i in act}
    return r1, r2, tgs.combine(act, C, p.q, c, r1, r2)


def test_shamir_reconstruction(tgs_setup):
    p, C, T_C, keys = tgs_setup
    for act in [(1, 2, 3), (2, 3, 4), (1, 3, 4)]:
        lams = tgs.lagrange_at_zero(act, p.q)
        rec = np.zeros_like(T_C)
        for i in act:
            rec = (rec + lams[i] * keys[i - 1].T_share) % p.q
        assert (rec == np.asarray(T_C) % p.q).all()


def test_fewer_than_tau_shares_do_not_reconstruct(tgs_setup):
    p, C, T_C, keys = tgs_setup
    act = (1, 2)
    lams = tgs.lagrange_at_zero(act, p.q)
    rec = np.zeros_like(T_C)
    for i in act:
        rec = (rec + lams[i] * keys[i - 1].T_share) % p.q
    assert not (rec == np.asarray(T_C) % p.q).all()


def test_tgs_combine_and_norm(tgs_setup):
    """Combined key satisfies C sbk = c and ||sbk|| <= B_th (Theorem 4/6)."""
    p, C, T_C, keys = tgs_setup
    c = (np.arange(p.n, dtype=np.int64) * 991) % p.q
    for sid, act in [(1, (1, 2, 3)), (2, (2, 3, 4)), (3, (1, 3, 4))]:
        r1, r2, sbk = run_tgs(p, C, keys, act, sid, c)
        assert sbk is not None
        assert (matmul_q(C, (sbk % p.q).reshape(-1, 1), p.q).reshape(-1)
                == c % p.q).all()
        b = threshold_bounds(p, N_PARTIES, TAU)
        assert float(np.linalg.norm(sbk.astype(np.float64))) <= b["B_th"]


def test_tgs_mask_cancellation(tgs_setup):
    """sum m_i = sum m*_i: the combiner output equals the unmasked sum."""
    p, C, T_C, keys = tgs_setup
    act, sid = (1, 2, 4), 77
    c = np.ones(p.n, dtype=np.int64)
    r1, r2, sbk = run_tgs(p, C, keys, act, sid, c)
    masks_row = sum(int(x) for i in act for x in r1[i].mask_row) % p.q
    # recompute column masks (PRF sessions are sub-indexed: sid' = sid<<20 | sub)
    from blt25.hashes import prf
    from blt25.tgs import _sub_sid
    col_total = 0
    for i in act:
        for j in act:
            if j != i:
                col_total += int(prf(keys[i - 1].seeds_col[j],
                                     _sub_sid(sid, 0), p.m_c, p.q).sum())
    assert masks_row % p.q == col_total % p.q


def test_tgs_combine_rejects_corruption(tgs_setup):
    p, C, T_C, keys = tgs_setup
    act, sid = (1, 2, 3), 99
    c = np.ones(p.n, dtype=np.int64)
    r1, r2, _ = run_tgs(p, C, keys, act, sid, c)
    r2[2].sbk_share[0] = (r2[2].sbk_share[0] + 1) % p.q
    assert tgs.combine(act, C, p.q, c, r1, r2) is None


# ---------------------------------------------------------------------------
# TBIBE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tbibe_setup(tiny_th):
    pk, parties, sk = tbibe.setup(tiny_th, N_PARTIES, TAU, seed=b"tb-tests")
    return pk, parties, sk


def run_tbibe_predec(pk, parties, sid, act, ids):
    r1out = {i: tbibe.predec_one(parties[i - 1], sid, act, ids) for i in act}
    r1 = {i: o.msg for i, o in r1out.items()}
    r2 = {i: tbibe.predec_two(parties[i - 1], r1out[i].state, sid, act, ids, r1)
          for i in act}
    return r1, r2, tbibe.combine(pk, sid, act, ids, r1, r2)


def test_tbibe_roundtrip(tbibe_setup):
    pk, parties, _ = tbibe_setup
    cts = [tbibe.encrypt(pk, r, b) for r, b in zip(IDS, [1, 0])]
    r1, r2, pdk = run_tbibe_predec(pk, parties, 1001, (1, 2, 4), IDS)
    assert pdk is not None
    msgs, errs = tbibe.decrypt(pk, pdk, cts, return_error=True)
    assert msgs == [1, 0]
    assert max(errs) < pk.params.q // 4


def test_tbibe_other_active_set(tbibe_setup):
    pk, parties, _ = tbibe_setup
    cts = [tbibe.encrypt(pk, r, b) for r, b in zip(IDS, [0, 1])]
    _, _, pdk = run_tbibe_predec(pk, parties, 1002, (2, 3, 4), IDS)
    assert pdk is not None and tbibe.decrypt(pk, pdk, cts) == [0, 1]


def test_tbibe_matches_nonthreshold_derivation(tbibe_setup):
    """The threshold pdk satisfies C sbk = c for the same c that Dec derives:
    threshold and non-threshold decryption see the same batch commitment."""
    pk, parties, sk = tbibe_setup
    _, _, pdk = run_tbibe_predec(pk, parties, 1003, (1, 3, 4), IDS)
    der = bibe.derive(pk, IDS, sidsp=pdk.sidsp)
    q = pk.params.q
    lhs = matmul_q(pk.C, (pdk.sbks[0] % q).reshape(-1, 1), q).reshape(-1)
    assert (lhs == der.cs[0] % q).all()


def test_tbibe_combiner_rejects_mixed_transcripts(tbibe_setup):
    """Round-2 messages from different first-round transcripts (different
    sidsp) must be rejected by the combiner."""
    pk, parties, _ = tbibe_setup
    act = (1, 2, 4)
    r1a, r2a, _ = run_tbibe_predec(pk, parties, 2001, act, IDS)
    r1b, r2b, _ = run_tbibe_predec(pk, parties, 2002, act, IDS)
    mixed = dict(r2a)
    mixed[4] = r2b[4]
    assert tbibe.combine(pk, 2001, act, IDS, r1a, mixed) is None


def test_tbibe_share_tamper_detected(tbibe_setup):
    pk, parties, _ = tbibe_setup
    act = (1, 2, 3)
    r1, r2, _ = run_tbibe_predec(pk, parties, 2003, act, IDS)
    r2[2].sbk_share.sbk_share[5] = (r2[2].sbk_share.sbk_share[5] + 3) % pk.params.q
    assert tbibe.combine(pk, 2003, act, IDS, r1, r2) is None


# ---------------------------------------------------------------------------
# Threshold GPV signatures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpv_setup(tiny_th):
    return gpv.setup(tiny_th, N_PARTIES, TAU, seed=b"gpv-tests")


def run_sign(pk, parties, sid, act, msg):
    r1, st = {}, {}
    for i in act:
        m1, s = gpv.sign_one(parties[i - 1], sid, act, msg)
        r1[i], st[i] = m1, s
    r2 = {i: gpv.sign_two(parties[i - 1], st[i], sid, act, msg, r1) for i in act}
    return gpv.combine(pk, sid, act, msg, r1, r2)


def test_gpv_sign_verify(gpv_setup):
    pk, parties = gpv_setup
    msg = b"the quick brown fox"
    for sid, act in [(1, (1, 2, 3)), (2, (2, 3, 4))]:
        sig = run_sign(pk, parties, sid, act, msg)
        assert sig is not None and gpv.verify(pk, msg, sig)


def test_gpv_rejections(gpv_setup):
    pk, parties = gpv_setup
    msg = b"message"
    sig = run_sign(pk, parties, 9, (1, 3, 4), msg)
    assert gpv.verify(pk, msg, sig)
    assert not gpv.verify(pk, b"other message", sig)
    bad = gpv.Signature(sidsp=sig.sidsp, sig=sig.sig.copy())
    bad.sig[0] += 1
    assert not gpv.verify(pk, msg, bad)
    # sidsp binding: same sig under a different transcript tag fails
    assert not gpv.verify(pk, msg, gpv.Signature(sidsp=b"\x00" * 32, sig=sig.sig))
    # norm bound: a huge multiple of q added to one coordinate still satisfies
    # the mod-q equation but must fail the norm check
    huge = gpv.Signature(sidsp=sig.sidsp, sig=sig.sig.copy())
    huge.sig[0] += pk.params.q
    assert not gpv.verify(pk, msg, huge)


def test_gpv_signature_norm(gpv_setup):
    pk, parties = gpv_setup
    sig = run_sign(pk, parties, 11, (1, 2, 4), b"norm test")
    nrm = float(np.linalg.norm(sig.sig.astype(np.float64)))
    assert 0 < nrm <= pk.B_th
