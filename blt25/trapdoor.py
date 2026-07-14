"""Gadget trapdoors and discrete Gaussian preimage sampling.

Implements, following the paper:

- ``trapgen``      : Lemma 1 / Lemma 23 (MP12-style).  A = [Abar | G - Abar R],
                     T_A = [R; I], so A T_A = G and ||T_A||_inf = 1.
- ``ajtai_basis``  : Lemma 31 ([CHW25, Lemma 7.4], from [MP12]).  From a gadget
                     trapdoor T_A build the short full-rank integer matrix
                     W = [I_M - T_A G^{-1}(A) | T_A S] whose columns lie in
                     Lambda^perp(A), and select M linearly independent columns.
- ``det_ext_basis``: DetExtBasis from Appendix C.1 - deterministic Peikert basis
                     extension of T'_A to F = [A | M1].
- ``KleinSampler`` : GPV/Klein randomized nearest-plane over an (extended)
                     Ajtai basis, driven by an explainable 1-D sampler so that
                     SamplePre / SampleLeft are deterministic functions of the
                     random tape and admit Explain algorithms (Definition 12,
                     Theorem 1).

Determinism notes (see AUDIT.md): the Gram-Schmidt vectors are computed in
float64 with a fixed algorithm; Klein's per-coordinate centers are ratios of
exactly-rounded float sums (math.fsum).  Two runs in the same process/platform
produce identical outputs given the same tape; cross-platform bit-exactness
would additionally require a fixed software float pipeline, which a production
implementation should provide.  All integer state (targets, preimages) is kept
exact; overflow guards raise rather than silently wrap.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from . import config
from .bits import BitStream, random_stream, stream_from_bytes
from .gadget import g_inverse, g_perp_basis, gadget_dims, gadget_matrix
from .gaussian import NBITS, explain_z, sample_z
from .modq import center_lift, det_mod_p, matmul_q, rank_profile_mod_p, solve_mod_q

_P1 = 2_147_483_647          # 2^31 - 1, Mersenne prime
_P2 = 2_147_483_629          # prime < 2^31
_P3 = 2_147_483_587          # prime < 2^31 (third CRT modulus for ExplainSL)


# ----------------------------------------------------------------------------
# TrapGen (Lemma 1)
# ----------------------------------------------------------------------------

def trapgen(n: int, q: int, m: int, stream: BitStream | None = None,
            return_parts: bool = False):
    """Sample (A, T_A) with A in Z_q^{n x m}, A T_A = G_n mod q, ||T_A||_inf = 1.

    Requires m >= mtilde + (n+1) ceil(log q) + slack for the leftover hash
    lemma; m = 3 n ceil(log q) (the paper's setting) satisfies this.
    With ``return_parts`` also returns (Abar, R) enabling the structural exact
    Lambda^perp basis below.
    """
    stream = stream or random_stream()
    mt = gadget_dims(n, q)
    if m < mt + (n + 1) * (q.bit_length()):
        warnings.warn(f"trapgen: m={m} too small for statistical closeness "
                      f"(need ~{mt + (n + 1) * q.bit_length()})")
    mbar = m - mt
    G = gadget_matrix(n, q)
    Abar = stream.uniform_mod_mat(n, mbar, q)
    R = stream.pm1_matrix(mbar, mt)
    # keep R signed (|entries| = 1) so the declared bound is honest
    right = (G - matmul_q(Abar, R, q, max_abs_b=1)) % q
    A = np.concatenate([Abar, right], axis=1)
    T = np.concatenate([R, np.eye(mt, dtype=np.int64)], axis=0)
    assert (matmul_q(A, T % q, q) == G % q).all()
    if return_parts:
        return A, T, Abar, R
    return A, T


def trapgen_exact_basis(Abar: np.ndarray, R: np.ndarray, q: int) -> AjtaiBasis:
    """Exact short basis of Lambda^perp(A) for A = [Abar | G - Abar R].

    Derivation: with w := x - R y the constraint A (x; y) = 0 becomes
    Abar w + G y = 0 (mod q), and [[I, R], [0, I]] is unimodular.  The lattice
    {(w, y) : Abar w + G y = 0} has the evident basis
    [[I, 0], [-G^{-1}(Abar), S]] of determinant +-det(S) = +-q^n, so

        B_A = [[I, R], [0, I]] @ [[I, 0], [-G^{-1}(Abar), S]]
            = [[I - R G^{-1}(Abar), R S], [-G^{-1}(Abar), S]]

    is a genuine basis of Lambda^perp(A) (index 1), unlike the literal
    Lemma 31 selection (see AUDIT.md).
    """
    n, mbar = Abar.shape
    GinvAbar = g_inverse(Abar % q, q)          # (mt x mbar) bits
    S = g_perp_basis(n, q)                     # (mt x mt)
    top = np.concatenate([np.eye(mbar, dtype=np.int64) - R @ GinvAbar, R @ S], axis=1)
    bot = np.concatenate([-GinvAbar, S], axis=1)
    B = np.concatenate([top, bot], axis=0).astype(np.int64)
    gs, norms2 = gram_schmidt(B)
    return AjtaiBasis(B=B, cols=list(range(B.shape[0])),
                      exact_basis=True, used_fallback=False,
                      gs=gs, norms2=norms2)


# ----------------------------------------------------------------------------
# Ajtai basis from a gadget trapdoor (Lemma 31)
# ----------------------------------------------------------------------------

@dataclass
class AjtaiBasis:
    """Short full-rank T'_A with A T'_A = 0 (mod q), plus diagnostics.

    ``exact_basis``: True if |det| = q^{n'} (a genuine basis of Lambda^perp);
    False if verified otherwise; None if not checked (default - the check is a
    diagnostic, see ``check_exactness``)."""
    B: np.ndarray            # (M x M) int64, columns in Lambda^perp(A)
    cols: list               # which columns of W were selected
    exact_basis: bool | None
    used_fallback: bool      # leftmost-rank-profile fallback was needed
    gs: np.ndarray | None = None       # float64 Gram-Schmidt vectors of B
    norms2: np.ndarray | None = None   # squared GS norms
    # arbitrary-precision GS data for genuinely dense bases (exact ToBasis
    # output whose GS norms span ~log2(index) bits - far beyond float64):
    mp_prec: int | None = None         # working precision in bits
    gs_mp: list | None = None          # list of columns, each a list of mpf
    norms2_mp: list | None = None      # squared GS norms as mpf


def _leftmost_rank_profile(W: np.ndarray, M: int) -> list:
    """Leftmost M linearly independent columns of W: FLINT nmod RREF when
    available (fast C), NumPy elimination otherwise."""
    from . import tobasis
    if tobasis.HAVE_FLINT:
        pivots, _ = tobasis.nmod_rref_pivots(W % _P1, _P1)
        if len(pivots) >= M:
            return list(pivots[:M])
        pivots, _ = tobasis.nmod_rref_pivots(W % _P2, _P2)
        return list(pivots[:M])
    cols = rank_profile_mod_p(W, _P1, M)
    if len(cols) < M:
        cols = rank_profile_mod_p(W, _P2, M)
    return cols


def _maybe_exact_basis(A: np.ndarray, q: int, cand: AjtaiBasis) -> AjtaiBasis:
    """Upgrade the Lemma-31 full-rank set to a genuine Lambda^perp basis via
    MG02 ToBasis (blt25.tobasis) according to config.EXACT_WWW24_BASIS.

    The returned basis R satisfies ||r~_i|| <= ||s~_i|| pointwise, so all
    sigma calibrations against the literal set remain valid."""
    from . import tobasis
    M = cand.B.shape[0]
    mode = config.exact_basis_mode()
    if mode is False:
        return cand
    if mode == "auto" and not (tobasis.HAVE_FLINT
                               and M <= config.exact_basis_max_dim()):
        return cand
    if not tobasis.HAVE_FLINT:
        raise RuntimeError("BLT25_EXACT_BASIS requested but python-flint is "
                           "not installed")
    R, Hdiag = tobasis.to_basis(A, q, cand.B)
    if all(h == 1 for h in Hdiag):
        cand.exact_basis = True          # the literal set was already a basis
        return cand
    maxabs = max(abs(int(x)) for row in R for x in row)
    if maxabs >= (1 << 62):
        raise ArithmeticError("ToBasis output exceeds the int64 backend "
                              "(unexpectedly long basis vectors)")
    B = np.array([[int(x) for x in row] for row in R], dtype=np.int64)
    assert not matmul_q(A, B % q, q, max_abs_b=q - 1).any()
    # The genuine basis has GS norms r~_i = s~_i / H_ii: its dynamic range is
    # ~log2(prod H_ii) = log2(index) bits.  Beyond ~2^12 that is unrepresentable
    # in float64 (Klein centers would be noise - observed as 100x-inflated
    # samples), so switch the GS data and Klein centers to mpmath at a
    # precision derived from the H diagonal.
    hbits = max(int(h).bit_length() for h in Hdiag)
    if hbits > 700:
        raise NotImplementedError("index defect beyond the supported mp "
                                  "precision budget (sigma_i would overflow "
                                  "float64)")
    if hbits <= 12:
        gs, norms2 = gram_schmidt(B)
        if norms2.max() > float(np.max(cand.norms2)) * (1 + 1e-6):
            raise ArithmeticError("ToBasis violated GS domination (bug)")
        return AjtaiBasis(B=B, cols=cand.cols, exact_basis=True,
                          used_fallback=cand.used_fallback, gs=gs, norms2=norms2)
    import mpmath as mp
    prec = 160 + hbits
    gs_mp, norms2_mp = gram_schmidt_mp(B, prec)
    with mp.workprec(prec):
        if max(norms2_mp) > float(np.max(cand.norms2)) * (1 + 1e-6):
            raise ArithmeticError("ToBasis violated GS domination (bug)")
    return AjtaiBasis(B=B, cols=cand.cols, exact_basis=True,
                      used_fallback=cand.used_fallback, gs=None, norms2=None,
                      mp_prec=prec, gs_mp=gs_mp, norms2_mp=norms2_mp)


def ajtai_basis(A: np.ndarray, T: np.ndarray, q: int) -> AjtaiBasis:
    """Lemma 31: short full-rank T'_A in Lambda^perp(A) from a gadget trapdoor
    T (A T = G).  Selects the leftmost R-linearly-independent M columns of
    [I - T G^{-1}(A) | T S]; Gram-Schmidt is computed here and reused by the
    samplers (GS breakdown doubles as the singularity test).

    Per config.EXACT_WWW24_BASIS the full-rank set is then upgraded to a
    genuine basis of Lambda^perp(A) (see blt25.tobasis and AUDIT.md S3.1)."""
    nprime, M = A.shape
    mtp = gadget_dims(nprime, q)
    assert T.shape == (M, mtp)
    GinvA = g_inverse(A % q, q)                      # (mtp x M), {0,1}
    # W1 = I - T G^{-1}(A): plain integer product, |entries| <= 1 + mtp
    W1 = np.eye(M, dtype=np.int64) - T.astype(np.int64) @ GinvA
    assert not matmul_q(A, W1 % q, q, max_abs_b=q - 1).any(), \
        "Lemma 31 block not in Lambda^perp(A)"
    cand = None
    try:
        gs, norms2 = gram_schmidt(W1)
        cand = AjtaiBasis(B=W1, cols=list(range(M)), exact_basis=None,
                          used_fallback=False, gs=gs, norms2=norms2)
    except ArithmeticError:
        pass
    if cand is None:
        S = g_perp_basis(nprime, q)                  # (mtp x mtp), small
        W2 = T @ S                                   # |entries| <= mtp * max|S|
        W = np.concatenate([W1, W2], axis=1)
        cols = _leftmost_rank_profile(W, M)
        assert len(cols) == M, "Lemma 31 matrix not full rank"
        B = W[:, cols].astype(np.int64)
        assert not matmul_q(A, B % q, q, max_abs_b=q - 1).any()
        gs, norms2 = gram_schmidt(B)
        cand = AjtaiBasis(B=B, cols=cols, exact_basis=None, used_fallback=True,
                          gs=gs, norms2=norms2)
    return _maybe_exact_basis(A, q, cand)


def check_exactness(basis: AjtaiBasis, q: int, nprime: int) -> bool:
    """Diagnostic: is Lambda(B) = Lambda^perp(A) exactly (index 1)?  Verified
    via |det B| = q^{n'} modulo two 31-bit primes."""
    exact = True
    for p in (_P1, _P2):
        dd = det_mod_p(basis.B, p)
        target = pow(q % p, nprime, p)
        if dd % p not in (target, (-target) % p):
            exact = False
            break
    basis.exact_basis = exact
    return exact


# ----------------------------------------------------------------------------
# DetExtBasis (Appendix C.1)
# ----------------------------------------------------------------------------

@dataclass
class ExtBasis:
    """Extended basis for F = [A | M1] in block-triangular form.

    Full basis (never materialized) is  [[Bp, X], [0, I_{m1}]] where Bp is the
    Ajtai basis of Lambda^perp(A) and X holds deterministic solutions of
    A x = -M1[:, i] (mod q), center-lifted.  Its Gram-Schmidt orthogonalization
    is [[GS(Bp), 0], [0, I]] exactly, since Bp is nonsingular over R.
    """
    Bp: np.ndarray           # (M x M)
    X: np.ndarray            # (M x m1)
    gs: np.ndarray | None    # (M x M) float64 GS vectors of Bp
    norms2: np.ndarray | None  # (M,) float64 squared GS norms
    max_gs: float
    mp_prec: int | None = None   # arbitrary-precision GS (dense exact bases)
    gs_mp: list | None = None
    norms2_mp: list | None = None

    @property
    def M(self) -> int:
        return self.Bp.shape[0]

    @property
    def m1(self) -> int:
        return self.X.shape[1]


def gram_schmidt(B: np.ndarray):
    """Gram-Schmidt orthogonalization (float64).

    Default backend: Householder QR (LAPACK).  For B = Q R, the Gram-Schmidt
    vector of column i is exactly q_i * R_ii, so gs = Q diag(R) and
    norms2_i = R_ii^2.  Deterministic within a process; across platforms it
    depends on the BLAS.

    Portable backend (config.PORTABLE_GS / BLT25_PORTABLE_GS=1): classical GS
    with re-orthogonalization built only from elementwise IEEE-754 products
    and math.fsum (exactly-rounded sums) - bit-identical on any IEEE platform,
    at ~30-100x the cost.  Use for heterogeneous multi-party deployments.

    Raises ArithmeticError if B is singular; near-zero pivots are resolved
    *exactly* by a determinant test modulo two 31-bit primes (float tolerance
    alone cannot distinguish a skewed-but-regular integer basis from a
    singular one).
    """
    if config.portable_gs():
        return _gram_schmidt_portable(B)
    Bf = B.astype(np.float64)
    M = Bf.shape[1]
    Q, R = np.linalg.qr(Bf, mode="reduced")
    diag = np.diagonal(R).copy()
    # Distinguish "skewed but regular" from "singular" only when the diagonal
    # is truly tiny (float noise scale); the exact mod-p determinant test is
    # O(M^3) and must stay off the hot path.
    if np.abs(diag).min(initial=np.inf) < 1e-9 * max(1.0, np.abs(diag).max()):
        if det_mod_p(B, _P1) == 0 and det_mod_p(B, _P2) == 0:
            raise ArithmeticError("Gram-Schmidt broke down (singular basis)")
    if (diag == 0.0).any():
        raise ArithmeticError("QR produced a zero diagonal entry")
    gs = Q * diag
    norms2 = diag * diag
    return gs, norms2


def gram_schmidt_mp(B: np.ndarray, prec: int):
    """Modified Gram-Schmidt in mpmath at `prec` bits, for bases whose GS
    norms span more dynamic range than float64 (dense exact bases: the ratio
    max/min GS norm is ~ the lattice index of the literal Lemma-31 set,
    hundreds of bits).  Returns (columns, norms2) as mpf lists.  O(M^3) mpf
    operations - use only at small dimensions."""
    import mpmath as mp
    Mrows, M = B.shape
    with mp.workprec(prec):
        cols = [[mp.mpf(int(B[r, i])) for r in range(Mrows)] for i in range(M)]
        norms2 = [mp.mpf(0)] * M
        for i in range(M):
            v = cols[i]
            for _ in range(2):
                for j in range(i):
                    num = mp.fsum(a * b for a, b in zip(v, cols[j]))
                    if num:
                        cji = num / norms2[j]
                        v = [a - cji * b for a, b in zip(v, cols[j])]
            n2 = mp.fsum(a * a for a in v)
            if n2 <= 0:
                raise ArithmeticError("mp Gram-Schmidt broke down")
            cols[i] = v
            norms2[i] = n2
        return cols, norms2


def _gram_schmidt_portable(B: np.ndarray):
    """BLAS-free deterministic GS: every inner product is an exactly-rounded
    fsum of elementwise IEEE products, every update elementwise - no
    reassociation anywhere, so results are bit-identical across platforms."""
    Bf = B.astype(np.float64)
    Mrows, M = Bf.shape
    Q = np.zeros_like(Bf)
    norms2 = np.zeros(M)
    for i in range(M):
        v = Bf[:, i].copy()
        for _ in range(2):
            for j in range(i):
                num = math.fsum((v * Q[:, j]).tolist())
                if num != 0.0:
                    v = v - (num / norms2[j]) * Q[:, j]
        n2 = math.fsum((v * v).tolist())
        col2 = math.fsum((Bf[:, i] * Bf[:, i]).tolist())
        if n2 <= 1e-18 * (col2 + 1.0):
            if det_mod_p(B, _P1) == 0 and det_mod_p(B, _P2) == 0:
                raise ArithmeticError("Gram-Schmidt broke down (singular basis)")
        Q[:, i] = v
        norms2[i] = n2
    return Q, norms2


def det_ext_basis(A: np.ndarray, M1: np.ndarray | None, ajtai: AjtaiBasis,
                  q: int) -> ExtBasis:
    """DetExtBasis(T'_A, [A | M1]) from Appendix C.1 (deterministic)."""
    Bp = ajtai.B
    M = Bp.shape[0]
    if M1 is None or M1.shape[1] == 0:
        X = np.zeros((M, 0), dtype=np.int64)
    else:
        sol = solve_mod_q(A, (-M1) % q, q)           # A sol = -M1 mod q
        X = center_lift(sol, q).astype(np.int64)
    if ajtai.mp_prec is not None:
        import mpmath as mp
        with mp.workprec(ajtai.mp_prec):
            mx = float(mp.sqrt(max(ajtai.norms2_mp)))
        return ExtBasis(Bp=Bp, X=X, gs=None, norms2=None, max_gs=mx,
                        mp_prec=ajtai.mp_prec, gs_mp=ajtai.gs_mp,
                        norms2_mp=ajtai.norms2_mp)
    if ajtai.gs is None:
        ajtai.gs, ajtai.norms2 = gram_schmidt(Bp)
    gs, norms2 = ajtai.gs, ajtai.norms2
    return ExtBasis(Bp=Bp, X=X, gs=gs, norms2=norms2,
                    max_gs=float(np.sqrt(norms2.max())))


# ----------------------------------------------------------------------------
# Nearest-plane reduction and the Klein sampler over an ExtBasis
# ----------------------------------------------------------------------------

def _center_of(c: list, gs_col: np.ndarray, norm2: float) -> float:
    """<c, gs_col> / norm2 with an exactly-rounded float sum.  The Python-int
    entries of c convert to float64 with correct rounding; precision of the
    center only affects sampling quality, never coset membership."""
    prods = [float(a) * b for a, b in zip(c, gs_col.tolist())]
    return math.fsum(prods) / norm2


def _centers_ctx(ext):
    """(center(c, i), sigma_i(sigma, i)) closures for an ExtBasis: float64
    fsum arithmetic normally; mpmath at ext.mp_prec for dense exact bases
    whose GS dynamic range exceeds float64 (see AUDIT.md).  Centers may then
    be mpf of any magnitude - the 1-D sampler shifts them internally."""
    if getattr(ext, "mp_prec", None) is None:
        def center(c, i):
            return _center_of(c, ext.gs[:, i], ext.norms2[i])

        def sigma_i(sigma, i):
            return sigma / math.sqrt(ext.norms2[i])
        return center, sigma_i
    import mpmath as mp
    prec = ext.mp_prec

    def center(c, i):
        with mp.workprec(prec):
            col = ext.gs_mp[i]
            num = mp.fsum(mp.mpf(a) * b for a, b in zip(c, col))
            return num / ext.norms2_mp[i]

    def sigma_i(sigma, i):
        with mp.workprec(prec):
            return float(mp.mpf(sigma) / mp.sqrt(ext.norms2_mp[i]))
    return center, sigma_i


def _round_center(v) -> int:
    try:
        import mpmath as mp
        if isinstance(v, mp.mpf):
            return int(mp.nint(v))
    except ImportError:                      # pragma: no cover
        pass
    return int(np.round(v))


def _nearest_plane(ext: ExtBasis, c_top) -> list:
    """Babai nearest-plane reduction of c_top against Lambda(Bp).

    Deterministic (round-half-even on float64 centers).  Integer state is kept
    exact in Python ints, so arbitrarily skewed bases cannot overflow; the
    float centers only affect which coset representative is returned, never
    coset membership.  Returns the reduced vector as a list of ints.
    """
    c = [int(x) for x in c_top]
    Bp = ext.Bp
    center, _ = _centers_ctx(ext)
    for i in range(ext.M - 1, -1, -1):
        k = _round_center(center(c, i))
        if k:
            col = Bp[:, i].tolist()
            c = [a - k * b for a, b in zip(c, col)]
    return c


def klein_sample_ext(ext: ExtBasis, w: np.ndarray, sigma: float,
                     stream: BitStream) -> np.ndarray:
    """Sample v ~ D over the coset w + Lambda(full basis), width sigma.

    ``w`` must have zeros in its last m1 coordinates (callers arrange this);
    the coset is then handled with the block-triangular basis
    [[Bp, X], [0, I]].  Consumes NBITS per coordinate from ``stream``.
    Returns integer v (length M + m1) with v == w (mod the basis lattice).

    The running Klein target c is kept exact (Python ints); v_top = -c_final
    identically, so the output is exact whatever the basis conditioning.
    """
    M, m1 = ext.M, ext.m1
    assert w.shape == (M + m1,) and not np.any(w[M:]), "w must vanish on the M1 block"
    if sigma < ext.max_gs:
        raise ValueError(f"sigma={sigma:.3g} below max GS norm {ext.max_gs:.3g}; "
                         "Klein sampling quality undefined")
    if sigma < ext.max_gs * math.sqrt(math.log2(M + m1 + 2)):
        warnings.warn("sigma below max_gs * sqrt(log dim): sampling quality is "
                      "marginal", stacklevel=2)
    # reduce the target: replaces w by a short representative of its coset
    w_top = _nearest_plane(ext, w[:M])
    c = [-x for x in w_top]                # Klein target (exact ints)
    zs_bot = np.zeros(m1, dtype=np.int64)
    # identity block first (Klein processes columns in reverse order); centers
    # are 0 there because w vanishes on that block => z_i ~ D_{Z, sigma, 0}
    for j in range(m1 - 1, -1, -1):
        zs_bot[j] = sample_z(0.0, sigma, stream)
    # their effect on the top target: c -= X @ z_bot (exact via Python ints)
    if m1:
        xz = _xz_product(ext.X, zs_bot)
        c = [a - b for a, b in zip(c, xz)]
    center, sig_i = _centers_ctx(ext)
    for i in range(M - 1, -1, -1):
        ci = center(c, i)
        si = sig_i(sigma, i)
        z = sample_z(ci, si, stream)
        if z:
            col = ext.Bp[:, i].tolist()
            c = [a - z * b for a, b in zip(c, col)]
    # v_top = w_top + sum z_i b_i = w_top + ((-w_top) - c_final) = -c_final
    vals = [-x for x in c] + [int(x) for x in zs_bot]
    if max(abs(x) for x in vals) < (1 << 62):
        return np.array(vals, dtype=np.int64)
    return np.array(vals, dtype=object)


def _xz_product(X: np.ndarray, z: np.ndarray) -> list:
    """X @ z exactly, as a list of Python ints."""
    if X.shape[1] == 0:
        return [0] * X.shape[0]
    mx = int(np.abs(X).max(initial=0))
    mz = int(np.abs(z).max(initial=0))
    if mx and mz and mx * mz * X.shape[1] < (1 << 62):
        return [int(x) for x in (X @ z)]
    zl = [int(v) for v in z]
    return [sum(int(a) * b for a, b in zip(row, zl)) for row in X]


_EXTRA_PRIMES: list[int] = []


def _crt_primes_for_ext(ext, sigma: float, M: int) -> tuple:
    """Enough 31-bit CRT primes to recover Klein coefficients exactly: their
    magnitude is at most ~|center| + 14 sigma_i with sigma_i = sigma/min_gs,
    plus slack for the nearest-plane residual."""
    from .modq import is_prime
    import mpmath as mp
    if getattr(ext, "mp_prec", None) is None:
        min_n2 = float(np.min(ext.norms2))
        bits = int(max(0.0, math.log2(max(sigma, 1.0) / max(math.sqrt(min_n2), 1e-300)))) + 40
    else:
        with mp.workprec(ext.mp_prec):
            smax = mp.mpf(sigma) / mp.sqrt(min(ext.norms2_mp))
            bits = int(mp.log(smax, 2)) + 40
    bits += M.bit_length() + 8
    primes = [_P1, _P2, _P3]
    total = sum(p.bit_length() - 1 for p in primes)
    cand = _P3 - 2 if not _EXTRA_PRIMES else _EXTRA_PRIMES[-1] - 2
    primes += _EXTRA_PRIMES
    total += sum(p.bit_length() - 1 for p in _EXTRA_PRIMES)
    while total < bits:
        while not is_prime(cand):
            cand -= 2
        _EXTRA_PRIMES.append(cand)
        primes.append(cand)
        total += cand.bit_length() - 1
        cand -= 2
    # trim to what this call needs (deterministic prefix)
    need, chosen, acc = bits, [], 0
    for p in primes:
        chosen.append(p)
        acc += p.bit_length() - 1
        if acc >= need and len(chosen) >= 3:
            break
    return tuple(chosen)


def klein_explain_ext(ext: ExtBasis, w: np.ndarray, sigma: float,
                      v: np.ndarray, rng: BitStream | None = None) -> bytes:
    """ExplainSL core: find a tape rb with klein_sample_ext(...; rb) == v.

    Recovers the Klein coefficients of v exactly, then replays the sampling
    loop calling explain_z per coordinate.  Raises ExplainError (negligible
    probability) if some coordinate's tape interval is empty.
    """
    M, m1 = ext.M, ext.m1
    rng = rng or random_stream()
    w_top = _nearest_plane(ext, w[:M])
    zs_bot = np.array([int(x) for x in v[M:]], dtype=np.int64)
    xz = _xz_product(ext.X, zs_bot) if m1 else [0] * M
    rhs_l = [int(a) - b - c for a, b, c in zip(v[:M], w_top, xz)]
    rhs = np.array(rhs_l, dtype=object)
    # solve Bp z_top = rhs exactly: CRT over enough 31-bit primes to cover the
    # coefficient range (3 primes ~ 2^92 for float-geometry bases; dense exact
    # bases can need coefficients ~ sigma / min_gs, so the list is sized from
    # the GS data), verify over Z
    zs_top = None
    try:
        primes = _crt_primes_for_ext(ext, sigma, M)
        sols = []
        for p in primes:
            rhs_p = np.array([r % p for r in rhs_l], dtype=np.int64)
            sols.append(solve_mod_q(ext.Bp % p, rhs_p, p))
        prod = 1
        for p in primes:
            prod *= p
        cand = [0] * M
        for i in range(M):
            # incremental CRT over the three residues
            val, mod = int(sols[0][i]), primes[0]
            for p, sol in zip(primes[1:], sols[1:]):
                k = ((int(sol[i]) - val) * pow(mod, -1, p)) % p
                val, mod = val + mod * k, mod * p
            val %= prod
            if val > prod // 2:
                val -= prod
            cand[i] = val
        # exact verification over Z (Python ints)
        chk = [sum(int(a) * b for a, b in zip(row, cand)) for row in ext.Bp]
        if chk == rhs_l:
            zs_top = cand
    except (ValueError, OverflowError):
        pass
    if zs_top is None:
        raise ArithmeticError("could not recover integer Klein coefficients "
                              "(v not in the sampled coset, or precision loss)")
    # replay, emitting tape bits in consumption order (identical arithmetic to
    # klein_sample_ext)
    c = [-x for x in w_top]
    chunks: list[int] = []
    for j in range(m1 - 1, -1, -1):
        chunks.append(explain_z(int(zs_bot[j]), 0.0, sigma, rng=rng))
    if m1:
        c = [a - b for a, b in zip(c, xz)]
    center, sig_i = _centers_ctx(ext)
    for i in range(M - 1, -1, -1):
        ci = center(c, i)
        si = sig_i(sigma, i)
        chunks.append(explain_z(int(zs_top[i]), ci, si, rng=rng))
        if zs_top[i]:
            col = ext.Bp[:, i].tolist()
            z = int(zs_top[i])
            c = [a - z * b for a, b in zip(c, col)]
    total = 0
    for ch in chunks:
        total = (total << NBITS) | ch
    nbytes = ((M + m1) * NBITS + 7) // 8
    return total.to_bytes(nbytes, "big")


class _TapeStream(BitStream):
    """A BitStream that replays a fixed byte string (the tape rb), then raises
    if exhausted.  Used to make SampleLeft(inp; rb) a deterministic function."""

    def __init__(self, tape: bytes):
        self._tape = tape
        self._pos = 0
        self.bits_consumed = 0

    def take_bits(self, k: int) -> int:
        if self._pos + k > 8 * len(self._tape):
            raise RuntimeError("random tape exhausted")
        out = 0
        need = k
        while need > 0:
            byte, bit = divmod(self._pos, 8)
            avail = 8 - bit
            grab = min(need, avail)
            cur = self._tape[byte]
            cur >>= avail - grab
            cur &= (1 << grab) - 1
            out = (out << grab) | cur
            self._pos += grab
            need -= grab
        self.bits_consumed += k
        return out


# ----------------------------------------------------------------------------
# Public samplers: SamplePre / SampleLeft / ExplainSL
# ----------------------------------------------------------------------------

class PreimageSampler:
    """Caches the expensive per-(A, T_A) data (Ajtai basis + GS) and exposes
    SamplePre / SampleLeft / ExplainSL over it.

    - ``sample_pre(u, sigma[, stream])``:  v with A v = u  (mod q).
    - ``sample_left(M1, u, sigma, rb)``:   v with [A | M1] v = u (mod q),
      deterministic function of the tape ``rb``.
    - ``explain_left(M1, u, sigma, v)``:   tape rb' with sample_left(...) == v.
    """

    def __init__(self, A: np.ndarray, T: np.ndarray | None, q: int,
                 basis: AjtaiBasis | None = None):
        self.A = A % q
        self.q = q
        self.ajtai = basis if basis is not None else ajtai_basis(self.A, T, q)
        self._ext_cache: dict[bytes, ExtBasis] = {}
        self._base_ext = det_ext_basis(self.A, None, self.ajtai, q)

    @classmethod
    def from_trapgen_parts(cls, A: np.ndarray, Abar: np.ndarray, R: np.ndarray,
                           q: int) -> "PreimageSampler":
        """Sampler over the *exact* Lambda^perp basis (index 1) available for
        TrapGen-structured matrices A = [Abar | G - Abar R]."""
        return cls(A, None, q, basis=trapgen_exact_basis(Abar, R, q))

    # -- helpers ------------------------------------------------------------
    def _ext_for(self, M1: np.ndarray | None) -> ExtBasis:
        if M1 is None or M1.shape[1] == 0:
            return self._base_ext
        key = M1.tobytes()
        ext = self._ext_cache.get(key)
        if ext is None:
            base = self._base_ext
            ext = ExtBasis(Bp=base.Bp,
                           X=center_lift(solve_mod_q(self.A, (-M1) % self.q, self.q),
                                         self.q).astype(np.int64),
                           gs=base.gs, norms2=base.norms2, max_gs=base.max_gs,
                           mp_prec=base.mp_prec, gs_mp=base.gs_mp,
                           norms2_mp=base.norms2_mp)
            self._ext_cache[key] = ext
        return ext

    def _coset_rep(self, u: np.ndarray, m1: int) -> np.ndarray:
        """Deterministic w with [A | M1] w = u (mod q) and w = 0 on the M1
        block (so all coset arithmetic stays on the well-conditioned side)."""
        w_top = center_lift(solve_mod_q(self.A, u % self.q, self.q), self.q)
        return np.concatenate([w_top.astype(np.int64),
                               np.zeros(m1, dtype=np.int64)])

    @property
    def max_gs(self) -> float:
        return self._base_ext.max_gs

    # -- API ----------------------------------------------------------------
    def sample_pre(self, u: np.ndarray, sigma: float,
                   stream: BitStream | None = None) -> np.ndarray:
        """SamplePre(A, T_A, u, sigma) -> v with A v = u mod q."""
        stream = stream or random_stream()
        ext = self._base_ext
        w = self._coset_rep(u, 0)
        v = klein_sample_ext(ext, w, sigma, stream)
        assert (matvec_q_int(self.A, v, self.q) == u % self.q).all()
        return v

    def tape_len_bits(self, m1: int) -> int:
        return (self._base_ext.M + m1) * NBITS

    def sample_left(self, M1: np.ndarray | None, u: np.ndarray, sigma: float,
                    rb: bytes) -> np.ndarray:
        """SampleLeft([A | M1], T_A, u, sigma; rb) - deterministic in rb."""
        ext = self._ext_for(M1)
        w = self._coset_rep(u, ext.m1)
        stream = _TapeStream(rb)
        v = klein_sample_ext(ext, w, sigma, stream)
        F = self.A if ext.m1 == 0 else np.concatenate([self.A, M1 % self.q], axis=1)
        assert (matvec_q_int(F, v, self.q) == u % self.q).all(), "F v != u"
        return v

    def explain_left(self, M1: np.ndarray | None, u: np.ndarray, sigma: float,
                     v: np.ndarray, rng: BitStream | None = None) -> bytes:
        """ExplainSL: find rb' with sample_left(M1, u, sigma, rb') == v."""
        ext = self._ext_for(M1)
        w = self._coset_rep(u, ext.m1)
        return klein_explain_ext(ext, w, sigma, v, rng=rng)


def matvec_q_int(A: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """A @ v mod q where v is a (possibly negative, possibly large) integer
    vector; overflow-safe by pre-reducing v mod q."""
    return matmul_q(A % q, (v % q).reshape(-1, 1), q).reshape(-1)


def sample_pre(A: np.ndarray, T: np.ndarray, q: int, u: np.ndarray,
               sigma: float, stream: BitStream | None = None) -> np.ndarray:
    """One-shot SamplePre (builds and discards the sampler)."""
    return PreimageSampler(A, T, q).sample_pre(u, sigma, stream)
