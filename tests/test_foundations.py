"""Tests for modq, bits, gadget: exact arithmetic and algebraic identities."""

import numpy as np
import pytest

from blt25.bits import BitStream, stream_from_bytes
from blt25.gadget import (g_inverse, g_perp_basis, gadget_dims, gadget_len,
                          gadget_matrix)
from blt25.modq import (center_lift, det_mod_p, is_prime, matmul_q, next_prime,
                        rank_profile_mod_p, solve_mod_q)


def test_primes():
    assert is_prime(2) and is_prime(3) and is_prime(2_147_483_647)
    assert not is_prime(1) and not is_prime(2_147_483_646)
    q = next_prime(1 << 27)
    assert is_prime(q) and q > (1 << 27)


@pytest.mark.parametrize("q", [next_prime(1 << 20), next_prime(1 << 28),
                               next_prime(1 << 30)])
def test_matmul_q_matches_object_arithmetic(q):
    """Chunked int64 matmul must agree with exact big-int arithmetic even when
    naive int64 accumulation would overflow (regression for the trapgen bug)."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, q, size=(7, 600), dtype=np.int64)
    b = rng.integers(0, q, size=(600, 5), dtype=np.int64)
    got = matmul_q(a, b, q)
    want = (a.astype(object) @ b.astype(object)) % q
    assert (got == want.astype(np.int64)).all()
    # signed small right operand with declared bound
    r = rng.choice([-1, 1], size=(600, 5)).astype(np.int64)
    got2 = matmul_q(a, r, q, max_abs_b=1)
    want2 = (a.astype(object) @ r.astype(object)) % q
    assert (got2 == want2.astype(np.int64)).all()


def test_solve_mod_q_particular_solution():
    q = next_prime(1 << 20)
    rng = np.random.default_rng(2)
    A = rng.integers(0, q, size=(6, 40), dtype=np.int64)
    X = rng.integers(0, q, size=(40, 3), dtype=np.int64)
    B = matmul_q(A, X, q)
    sol = solve_mod_q(A, B, q)
    assert (matmul_q(A, sol, q) == B).all()
    # determinism
    sol2 = solve_mod_q(A, B, q)
    assert (sol == sol2).all()


def test_rank_profile_and_det():
    p = 2_147_483_647
    rng = np.random.default_rng(3)
    W = rng.integers(-50, 50, size=(8, 20), dtype=np.int64)
    W[:, 3] = 2 * W[:, 1] + 5 * W[:, 2]      # dependent column
    cols = rank_profile_mod_p(W, p, 8)
    assert len(cols) == 8 and 3 not in cols
    sq = W[:, cols[:8]]
    assert det_mod_p(sq, p) != 0


def test_bitstream_determinism_and_uniformity():
    s1, s2 = stream_from_bytes(b"abc"), stream_from_bytes(b"abc")
    assert [s1.take_bits(11) for _ in range(200)] == \
           [s2.take_bits(11) for _ in range(200)]
    s3 = stream_from_bytes(b"abd")
    assert [stream_from_bytes(b"abc").take_bits(64)] != [s3.take_bits(64)]
    # uniform_mod bias smoke test
    q = 97
    vals = [stream_from_bytes(bytes([i, j])).take_uniform_mod(q)
            for i in range(30) for j in range(30)]
    counts = np.bincount(vals, minlength=q)
    assert counts.max() < 5 * (900 / q)


@pytest.mark.parametrize("n,qexp", [(2, 8), (4, 20), (8, 28)])
def test_gadget_identities(n, qexp):
    q = next_prime(1 << qexp)
    G = gadget_matrix(n, q)
    mt = gadget_dims(n, q)
    assert G.shape == (n, mt)
    rng = np.random.default_rng(4)
    X = rng.integers(0, q, size=(n, 9), dtype=np.int64)
    bits = g_inverse(X, q)
    assert set(np.unique(bits)) <= {0, 1}
    assert (matmul_q(G, bits, q, max_abs_b=1) == X).all()
    # S is a basis of Lambda^perp(G): G S = 0 and per-block |det| = q
    S = g_perp_basis(n, q)
    assert (matmul_q(G, S % q, q) == 0).all()
    k1 = gadget_len(q)
    blk = S[:k1, :k1].astype(object)
    # exact integer determinant via fraction-free elimination
    def bareiss(a):
        a = [[int(x) for x in row] for row in a]
        nn, sign, prev = len(a), 1, 1
        for k in range(nn - 1):
            if a[k][k] == 0:
                for r in range(k + 1, nn):
                    if a[r][k]:
                        a[k], a[r] = a[r], a[k]
                        sign = -sign
                        break
                else:
                    return 0
            for i in range(k + 1, nn):
                for j in range(k + 1, nn):
                    a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
            prev = a[k][k]
        return sign * a[nn - 1][nn - 1]
    assert abs(bareiss(blk)) == q


def test_center_lift():
    q = 101
    x = np.array([0, 1, 50, 51, 100], dtype=np.int64)
    lifted = center_lift(x, q)
    assert list(lifted) == [0, 1, 50, -50, -1]
