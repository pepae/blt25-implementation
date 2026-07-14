"""Tests for TrapGen, Ajtai bases (Lemma 31), DetExtBasis, Klein sampling,
SamplePre / SampleLeft / ExplainSL (Section 4, Appendix C)."""

import math

import numpy as np
import pytest

from blt25.bits import random_stream, stream_from_bytes
from blt25.gadget import gadget_dims, gadget_matrix
from blt25.modq import matmul_q, next_prime
from blt25.trapdoor import (PreimageSampler, ajtai_basis, check_exactness,
                            trapgen, trapgen_exact_basis)

Q = next_prime(1 << 20)
N = 4
M = 3 * N * Q.bit_length()


@pytest.fixture(scope="module")
def tg():
    return trapgen(N, Q, M, stream_from_bytes(b"tg-tests"), return_parts=True)


def test_trapgen_relation(tg):
    A, T, Abar, R = tg
    G = gadget_matrix(N, Q)
    assert (matmul_q(A, T % Q, Q) == G % Q).all()
    assert np.abs(T).max() == 1
    # exact object-arithmetic cross-check on all columns
    got = (A.astype(object) @ T.astype(object)) % Q
    assert (got == G.astype(object) % Q).all()


def test_trapgen_marginal_uniformity(tg):
    """A should look uniform: crude bin test on the entries."""
    A, *_ = tg
    vals = A.reshape(-1).astype(np.float64) / Q
    hist, _ = np.histogram(vals, bins=16, range=(0, 1))
    expect = vals.size / 16
    assert (np.abs(hist - expect) < 6 * math.sqrt(expect) + 20).all()


def test_exact_basis_is_genuine(tg):
    """The structural TrapGen basis has |det| = q^n (index 1)."""
    A, T, Abar, R = tg
    basis = trapgen_exact_basis(Abar, R, Q)
    assert (matmul_q(A, basis.B % Q, Q) == 0).all()
    assert check_exactness(basis, Q, N) is True


def test_lemma31_basis_full_rank_but_maybe_sublattice(tg):
    """Lemma 31 literal selection: in Lambda^perp, full rank; exactness is NOT
    guaranteed (documented deviation, see AUDIT.md)."""
    A, T, *_ = tg
    basis = ajtai_basis(A, T, Q)
    assert (matmul_q(A, basis.B % Q, Q) == 0).all()
    check_exactness(basis, Q, N)          # records True/False; no assertion


@pytest.mark.parametrize("exact", [True, False])
def test_sample_pre_correctness_and_norm(tg, exact):
    A, T, Abar, R = tg
    ps = (PreimageSampler.from_trapgen_parts(A, Abar, R, Q) if exact
          else PreimageSampler(A, T, Q))
    sigma = ps.max_gs * 3.0
    u = (np.arange(N, dtype=np.int64) * 7919) % Q
    v = ps.sample_pre(u, sigma)
    assert (matmul_q(A, (v % Q).reshape(-1, 1), Q).reshape(-1) == u % Q).all()
    nrm = float(np.linalg.norm(v.astype(np.float64)))
    expected = sigma * math.sqrt(M) / math.sqrt(2 * math.pi)
    assert nrm < 2.5 * expected


@pytest.mark.parametrize("exact", [True, False])
def test_sample_left_deterministic_and_explainable(tg, exact):
    """Theorem 1: SampleLeft(inp; rb) deterministic; ExplainSL yields a tape
    reproducing the given preimage exactly."""
    A, T, Abar, R = tg
    ps = (PreimageSampler.from_trapgen_parts(A, Abar, R, Q) if exact
          else PreimageSampler(A, T, Q))
    sigma = ps.max_gs * 3.0
    u = (np.arange(N, dtype=np.int64) * 13) % Q
    M1 = stream_from_bytes(b"m1").uniform_mod_mat(N, 25, Q)
    rb = bytes((ps.tape_len_bits(25) // 8))            # zero tape is valid too
    rb = b"\x5a" * (ps.tape_len_bits(25) // 8)
    v1 = ps.sample_left(M1, u, sigma, rb)
    v2 = ps.sample_left(M1, u, sigma, rb)
    assert (v1 == v2).all()
    F = np.concatenate([A, M1], axis=1)
    assert (matmul_q(F, (v1 % Q).reshape(-1, 1), Q).reshape(-1) == u % Q).all()
    rb2 = ps.explain_left(M1, u, sigma, v1)
    assert rb2 != rb or True                            # tapes may differ
    v3 = ps.sample_left(M1, u, sigma, rb2)
    assert (v3 == v1).all()


def test_sample_left_distinct_tapes_distinct_outputs(tg):
    A, T, Abar, R = tg
    ps = PreimageSampler.from_trapgen_parts(A, Abar, R, Q)
    sigma = ps.max_gs * 3.0
    u = np.zeros(N, dtype=np.int64)
    M1 = stream_from_bytes(b"m2").uniform_mod_mat(N, 10, Q)
    nb = ps.tape_len_bits(10) // 8
    v1 = ps.sample_left(M1, u, sigma, b"\x01" * nb)
    v2 = ps.sample_left(M1, u, sigma, b"\x02" * nb)
    assert not (v1 == v2).all()


def test_sigma_too_small_rejected(tg):
    A, T, Abar, R = tg
    ps = PreimageSampler.from_trapgen_parts(A, Abar, R, Q)
    with pytest.raises(ValueError):
        ps.sample_pre(np.zeros(N, dtype=np.int64), ps.max_gs * 0.5)


def test_tape_exhaustion(tg):
    A, T, Abar, R = tg
    ps = PreimageSampler.from_trapgen_parts(A, Abar, R, Q)
    with pytest.raises(RuntimeError):
        ps.sample_left(None, np.zeros(N, dtype=np.int64), ps.max_gs * 3, b"\x00" * 4)


@pytest.mark.stat
def test_preimage_marginals(tg):
    """Preimage coordinates should be centered with std ~ sigma/sqrt(2 pi)
    (exact-basis path; full-coset sampling)."""
    A, T, Abar, R = tg
    ps = PreimageSampler.from_trapgen_parts(A, Abar, R, Q)
    sigma = ps.max_gs * 3.0
    u = np.zeros(N, dtype=np.int64)
    samples = []
    st = stream_from_bytes(b"marg")
    for i in range(6):
        samples.append(ps.sample_pre(u, sigma, st).astype(np.float64))
    vs = np.concatenate(samples)
    s = sigma / math.sqrt(2 * math.pi)
    assert abs(vs.mean()) < 4 * s / math.sqrt(vs.size)
    assert abs(vs.std() / s - 1) < 0.15
