"""Tests for the shifted multi-preimage trapdoor sampler with d-bit labels
(Appendix B): Definition 24 properties."""

import math

import numpy as np
import pytest

from blt25 import www24
from blt25.bits import random_stream, stream_from_bytes
from blt25.gadget import gadget_matrix
from blt25.modq import matmul_q, next_prime

Q = next_prime(1 << 20)
N, D = 4, 4
US = [(0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 1)]


@pytest.fixture(scope="module")
def crs():
    return www24.gen(N, Q, D, stream_from_bytes(b"crs-tests"))


@pytest.fixture(scope="module")
def expanded(crs):
    return www24.expand(crs, US)


def test_lemma29_identity(crs):
    """(B - x^T (x) G) H_{B,u,x} = B_u - delta_u(x) G, H entries in {-1,0,1}."""
    G = gadget_matrix(N, Q)
    for u in [(0, 1, 0, 1), (1, 1, 1, 1), (0, 0, 0, 0)]:
        Bu, chain = www24.eval_f(crs, u)
        for x in [(0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 1), u]:
            H = www24.eval_fx(crs, u, x, chain)
            assert set(np.unique(H)) <= {-1, 0, 1}
            Bx = (crs.B - www24._label_shift(crs, x)) % Q
            lhs = matmul_q(Bx, H % Q, Q)
            delta = 1 if x == u else 0
            assert (lhs == (Bu - delta * G) % Q).all()


def test_struct_trapgen_relation(crs, expanded):
    """D_ell T = I_ell (x) G with ||T||_inf <= 1 (Lemma 30 / eq. (3))."""
    exp = expanded
    assert np.abs(exp.T).max() <= 1
    Gln = gadget_matrix(len(US) * N, Q)
    assert (matmul_q(exp.D, exp.T % Q, Q) == Gln % Q).all()


def test_local_expansion(crs, expanded):
    """ExpandLocal(crs, u_i) = A_i from Expand, and it reads only crs (local)."""
    for i, u in enumerate(US):
        assert (www24.expand_local(crs, u) == expanded.A_list[i]).all()


def test_sample_mult_pre_correctness(expanded):
    """A_i pi_i = t_i + c for every i (Definition 24 correctness)."""
    exp = expanded
    ps = exp.sampler()
    sigma = ps.max_gs * 3.5
    st = stream_from_bytes(b"smp")
    targets = [st.uniform_mod_vec(N, Q) for _ in US]
    pis, c = www24.sample_mult_pre(exp, targets, sigma, st)
    for i in range(len(US)):
        lhs = matmul_q(exp.A_list[i], (pis[i] % Q).reshape(-1, 1), Q).reshape(-1)
        assert (lhs == (targets[i] + c) % Q).all()


@pytest.mark.stat
def test_preimage_distribution_smoke(expanded):
    """pi_i coordinates ~ Gaussian width sigma; the shift c is close to
    uniform (bin test)."""
    exp = expanded
    ps = exp.sampler()
    sigma = ps.max_gs * 3.5
    st = stream_from_bytes(b"dist")
    coords, shifts = [], []
    for trial in range(4):
        targets = [st.uniform_mod_vec(N, Q) for _ in US]
        pis, c = www24.sample_mult_pre(exp, targets, sigma, st)
        coords.extend(np.concatenate(pis).astype(np.float64).tolist())
        shifts.extend((np.asarray(c) / Q).tolist())
    s = sigma / math.sqrt(2 * math.pi)
    arr = np.array(coords)
    assert abs(arr.mean()) < 5 * s / math.sqrt(arr.size)
    assert abs(arr.std() / s - 1) < 0.2
    hist, _ = np.histogram(shifts, bins=4, range=(0, 1))
    assert hist.min() >= 1                     # crude spread check


def test_gen_prog_perfect(crs):
    """Perfect somewhere programmability (Definition 24): Expand on the
    programmed CRS reproduces A* exactly at the programmed label."""
    st = stream_from_bytes(b"prog")
    A_star = st.uniform_mod_mat(N, crs.t, Q)
    u_prog = (1, 1, 0, 0)
    crs2 = www24.gen_prog(N, Q, D, u_prog, A_star)
    assert (www24.expand_local(crs2, u_prog) == A_star % Q).all()
    exp2 = www24.expand(crs2, [(0, 0, 0, 0), u_prog, (1, 0, 0, 1)])
    assert (exp2.A_list[1] == A_star % Q).all()
    Gln = gadget_matrix(3 * N, Q)
    assert (matmul_q(exp2.D, exp2.T % Q, Q) == Gln % Q).all()


def test_gen_prog_crs_distribution(crs):
    """crs from GenProg(u, uniform A*) is itself uniform: entry-wise bin test
    (the map A* -> crs is a bijection: shift by u^T (x) G)."""
    st = stream_from_bytes(b"progdist")
    vals = []
    for _ in range(3):
        A_star = st.uniform_mod_mat(N, crs.t, Q)
        crs2 = www24.gen_prog(N, Q, D, (1, 0, 1, 1), A_star)
        vals.extend((np.concatenate([crs2.A, crs2.B], axis=1).reshape(-1)
                     / Q).tolist())
    hist, _ = np.histogram(vals, bins=8, range=(0, 1))
    expect = len(vals) / 8
    assert (np.abs(hist - expect) < 6 * math.sqrt(expect) + 20).all()


def test_distinct_labels_required(crs):
    with pytest.raises(AssertionError):
        www24.struct_trapgen(crs, [(0, 0, 0, 0), (0, 0, 0, 0)])
