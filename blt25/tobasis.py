"""Exact short bases of q-ary lattices via MG02 ToBasis.

Fixes the Lemma 31 sublattice deviation (AUDIT.md S3.1): given the short
full-rank set S produced by the literal Lemma 31 selection, produce a genuine
basis R of Lambda^perp(A) whose Gram-Schmidt norms are dominated by S's,

    ||r~_i|| <= ||s~_i||   for every i,

so all sigma calibrations remain valid while the sampler's support becomes
the full coset - which is what the security proof's simulation (Lemma 18,
Hybrids 5/6) requires of the honest sampler.

Method ([MG02, Lemma 7.1] made cheap for q-ary lattices):

1. Echelon basis.  Lambda^perp(A) contains q Z^M, so it has the free
   "q-ary echelon" basis B: RREF A over F_q; for each non-pivot column f a
   kernel vector with 1 at f and the (negated) RREF solution at the pivot
   coordinates; plus q e_j for each pivot column j.  det(B) = +- q^rank -
   a true basis with entries < q, computed by one small mod-q elimination.

2. Coefficients.  Q := B^{-1} S is integral because S is in the lattice; the
   echelon structure makes this a copy for the non-pivot rows and an exactly
   divisible-by-q expression for the pivot rows (asserted).

3. Triangularization.  Row-HNF Q = W H (W unimodular, H upper triangular
   with positive diagonal) via FLINT.  Then R := S H^{-1} = B W is a basis of
   the same lattice, and S = R H with H upper triangular gives the nested
   spans  span(s_1..s_i) = span(r_1..r_i), hence s~_i = H_ii r~_i and
   ||r~_i|| = ||s~_i|| / H_ii <= ||s~_i||.

4. Verification (all enforced): R integral, A R = 0 (mod q),
   |det R| = q^rank modulo two 31-bit primes, and the GS domination.

FLINT (python-flint) is required for steps 2 (rank profile speed) and 3; if
it is unavailable the caller falls back to the literal Lemma 31 behaviour
with a warning (see trapdoor.ajtai_basis).
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .modq import inv_mod

try:
    import flint as _flint
    HAVE_FLINT = True
except ImportError:                       # pragma: no cover
    _flint = None
    HAVE_FLINT = False


def _to_fmpz(M: np.ndarray | list):
    return _flint.fmpz_mat([[int(x) for x in row] for row in M])


def nmod_rref_pivots(M: np.ndarray, p: int):
    """(pivot column indices, rref rows as int64) of M over F_p via FLINT
    nmod_mat (fast, C).  Falls back to a NumPy elimination if FLINT is
    missing."""
    if HAVE_FLINT:
        nm = _flint.nmod_mat([[int(x) % p for x in row] for row in M], p)
        rr, rank = nm.rref()
        rows = [[int(rr[i, j]) for j in range(nm.ncols())] for i in range(rank)]
        pivots = []
        for r in rows:
            nz = next(j for j, v in enumerate(r) if v != 0)
            pivots.append(nz)
        return pivots, np.array(rows, dtype=np.int64)
    # NumPy fallback (slower)
    from .modq import rank_profile_mod_p
    Mp = np.asarray(M, dtype=np.int64) % p
    n = Mp.shape[0]
    pivots = rank_profile_mod_p(Mp.copy(), p, min(Mp.shape))
    # re-run a plain RREF to get the reduced rows
    A = Mp.copy()
    row = 0
    for col in pivots:
        nz = np.nonzero(A[row:, col])[0]
        piv = row + int(nz[0])
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        A[row] = (A[row] * inv_mod(int(A[row, col]), p)) % p
        others = np.nonzero(A[:, col])[0]
        others = others[others != row]
        if others.size:
            A[others] = (A[others] - np.outer(A[others, col], A[row])) % p
        row += 1
    return pivots, A[:row]


def qary_echelon_basis(A: np.ndarray, q: int):
    """The q-ary echelon basis of Lambda^perp(A) (A in Z_q^{r' x M}, full row
    rank r = rank over F_q).  Returns (B (M x M int64), pivots, free)."""
    A = np.asarray(A, dtype=np.int64) % q
    M = A.shape[1]
    pivots, rr = nmod_rref_pivots(A, q)
    r = len(pivots)
    free = [j for j in range(M) if j not in set(pivots)]
    B = np.zeros((M, M), dtype=np.int64)
    # kernel vectors: one per free column
    for k, f in enumerate(free):
        B[f, k] = 1
        for i, pcol in enumerate(pivots):
            B[pcol, k] = (-int(rr[i, f])) % q
    # q e_j for pivot columns
    for k, pcol in enumerate(pivots):
        B[pcol, len(free) + k] = q
    return B, pivots, free


def coefficients_wrt_echelon(S: np.ndarray, pivots, free, q: int,
                             ech_rr_rows: np.ndarray) -> np.ndarray:
    """Q with B Q = S for the echelon basis B, exact (object dtype).

    Row structure of B (permuted to [free; pivots]):
        [[ I,        0   ]
         [ Xmat,   q I_r ]]
    so Q[free rows] = S[free rows] and
    Q[pivot rows] = (S[pivot rows] - Xmat S[free rows]) / q  (exactly)."""
    Sf = np.asarray(S, dtype=object)
    Mdim = Sf.shape[0]
    r = len(pivots)
    X = np.zeros((r, len(free)), dtype=object)
    for k, f in enumerate(free):
        for i in range(r):
            X[i, k] = (-int(ech_rr_rows[i, f])) % q
    S_free = Sf[free, :]
    S_piv = Sf[pivots, :]
    num = S_piv - X @ S_free
    Q = np.zeros_like(Sf)
    Q[free, :] = S_free
    for a in range(r):
        for b in range(Sf.shape[1]):
            v = int(num[a, b])
            if v % q != 0:
                raise ArithmeticError("S is not in Lambda^perp (q-divisibility "
                                      "failed) - cannot take coefficients")
            Q[pivots[a], b] = v // q
    return Q


def to_basis(A: np.ndarray, q: int, S: np.ndarray):
    """MG02 ToBasis for the q-ary lattice Lambda^perp(A).

    Input: S (M x M int64), a full-rank set of lattice vectors (columns).
    Output: (R, H_diag) - R a genuine basis (int64) with GS domination
    ||r~_i|| <= ||s~_i||; H_diag the HNF diagonal (H_ii > 1 marks the
    directions where S was index-deficient).  Raises if FLINT is missing.
    """
    if not HAVE_FLINT:
        raise RuntimeError("python-flint is required for exact ToBasis")
    A = np.asarray(A, dtype=np.int64) % q
    M = S.shape[0]
    _, pivots_and_rr = None, None
    pivots, rr = nmod_rref_pivots(A, q)
    free = [j for j in range(M) if j not in set(pivots)]
    Q = coefficients_wrt_echelon(S, pivots, free, q, rr)
    Qf = _to_fmpz(Q)
    H = Qf.hnf()                                   # row HNF: H = U Q, H upper-tri
    # R^T solves H^T R^T = S^T  (over Q; result must be integral)
    Ht = _flint.fmpq_mat(H.transpose())
    St = _flint.fmpq_mat(_to_fmpz(np.asarray(S, dtype=object).T))
    Rt = Ht.solve(St)
    R = np.zeros((M, M), dtype=object)
    for i in range(M):
        for j in range(M):
            v = Rt[j, i]
            p_, q_ = v.p, v.q
            if int(q_) != 1:
                raise ArithmeticError("ToBasis produced a non-integral basis "
                                      "(H not a valid triangularization?)")
            R[i, j] = int(p_)
    Hdiag = [int(H[i, i]) for i in range(M)]
    return R, Hdiag


def verify_basis(A: np.ndarray, q: int, R: np.ndarray, S: np.ndarray,
                 gs_S_norms: np.ndarray | None = None) -> dict:
    """Check R is a genuine short basis: A R = 0 (mod q), |det R| = q^rank
    (mod two 31-bit primes), and GS(R) <= GS(S) pointwise (when provided)."""
    from .modq import det_mod_p, matmul_q
    from .trapdoor import gram_schmidt
    out = {}
    Rq = (np.asarray(R, dtype=object) % q).astype(np.int64)
    out["in_lattice"] = not matmul_q(A % q, Rq, q).any()
    pivots, _ = nmod_rref_pivots(A % q, q)
    r = len(pivots)
    ok = True
    for p in (2_147_483_647, 2_147_483_629):
        Rp = (np.asarray(R, dtype=object) % p).astype(np.int64)
        d = det_mod_p(Rp, p)
        t = pow(q % p, r, p)
        if d not in (t, (-t) % p):
            ok = False
    out["det_is_q_pow_rank"] = ok
    Ri = np.array([[int(x) for x in row] for row in R], dtype=np.int64)
    gs, n2 = gram_schmidt(Ri)
    out["max_gs"] = float(np.sqrt(n2.max()))
    if gs_S_norms is not None:
        out["gs_dominated"] = bool((np.sqrt(n2) <= gs_S_norms * (1 + 1e-9)).all())
    return out
