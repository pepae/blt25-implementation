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

from .bits import BitStream, random_stream, stream_from_bytes
from .gadget import g_inverse, g_perp_basis, gadget_dims, gadget_matrix
from .gaussian import NBITS, explain_z, sample_z
from .modq import center_lift, det_mod_p, matmul_q, rank_profile_mod_p, solve_mod_q

_P1 = 2_147_483_647          # 2^31 - 1, Mersenne prime
_P2 = 2_147_483_629          # prime < 2^31
_INT_GUARD = 1 << 55         # integer-magnitude guard for int64 paths


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


def ajtai_basis(A: np.ndarray, T: np.ndarray, q: int) -> AjtaiBasis:
    """Lemma 31: short full-rank T'_A in Lambda^perp(A) from a gadget trapdoor
    T (A T = G).  Selects the leftmost R-linearly-independent M columns of
    [I - T G^{-1}(A) | T S]; Gram-Schmidt is computed here and reused by the
    samplers (GS breakdown doubles as the singularity test)."""
    nprime, M = A.shape
    mtp = gadget_dims(nprime, q)
    assert T.shape == (M, mtp)
    GinvA = g_inverse(A % q, q)                      # (mtp x M), {0,1}
    # W1 = I - T G^{-1}(A): plain integer product, |entries| <= 1 + mtp
    W1 = np.eye(M, dtype=np.int64) - T.astype(np.int64) @ GinvA
    assert not matmul_q(A, W1 % q, q, max_abs_b=q - 1).any(), \
        "Lemma 31 block not in Lambda^perp(A)"
    try:
        gs, norms2 = gram_schmidt(W1)
        return AjtaiBasis(B=W1, cols=list(range(M)), exact_basis=None,
                          used_fallback=False, gs=gs, norms2=norms2)
    except ArithmeticError:
        pass
    S = g_perp_basis(nprime, q)                      # (mtp x mtp), small
    W2 = T @ S                                       # |entries| <= mtp * max|S|
    W = np.concatenate([W1, W2], axis=1)
    cols = rank_profile_mod_p(W, _P1, M)
    if len(cols) < M:
        cols = rank_profile_mod_p(W, _P2, M)
    assert len(cols) == M, "Lemma 31 matrix not full rank"
    B = W[:, cols].astype(np.int64)
    assert not matmul_q(A, B % q, q, max_abs_b=q - 1).any()
    gs, norms2 = gram_schmidt(B)
    return AjtaiBasis(B=B, cols=cols, exact_basis=None, used_fallback=True,
                      gs=gs, norms2=norms2)


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
    gs: np.ndarray           # (M x M) float64 GS vectors of Bp
    norms2: np.ndarray       # (M,) float64 squared GS norms
    max_gs: float

    @property
    def M(self) -> int:
        return self.Bp.shape[0]

    @property
    def m1(self) -> int:
        return self.X.shape[1]


def gram_schmidt(B: np.ndarray):
    """Gram-Schmidt orthogonalization via Householder QR (float64, LAPACK).

    For a QR factorization B = Q R (R with positive-or-negative diagonal), the
    Gram-Schmidt vector of column i is exactly q_i * R_ii, so gs = Q diag(R)
    and norms2_i = R_ii^2.  Much faster and more stable than classical GS.

    Raises ArithmeticError if B is singular; near-zero R_ii are resolved
    *exactly* by a determinant test modulo two 31-bit primes (float tolerance
    alone cannot distinguish a skewed-but-regular integer basis from a
    singular one).

    Determinism: identical within a process (results are cached and shared by
    PreDec/Dec); across platforms it depends on the BLAS - see AUDIT.md.
    """
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


def _nearest_plane(ext: ExtBasis, c_top) -> list:
    """Babai nearest-plane reduction of c_top against Lambda(Bp).

    Deterministic (round-half-even on float64 centers).  Integer state is kept
    exact in Python ints, so arbitrarily skewed bases cannot overflow; the
    float centers only affect which coset representative is returned, never
    coset membership.  Returns the reduced vector as a list of ints.
    """
    c = [int(x) for x in c_top]
    Bp, gs, norms2 = ext.Bp, ext.gs, ext.norms2
    for i in range(ext.M - 1, -1, -1):
        k = int(np.round(_center_of(c, gs[:, i], norms2[i])))
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
    for i in range(M - 1, -1, -1):
        ci = _center_of(c, ext.gs[:, i], ext.norms2[i])
        si = sigma / math.sqrt(ext.norms2[i])
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
    # solve Bp z_top = rhs exactly: solve mod two 31-bit primes, CRT-combine
    # (covers |z| < p1*p2/2 ~ 2^61), verify over Z
    zs_top = None
    try:
        rhs1 = np.array([r % _P1 for r in rhs_l], dtype=np.int64)
        rhs2 = np.array([r % _P2 for r in rhs_l], dtype=np.int64)
        z1 = solve_mod_q(ext.Bp % _P1, rhs1, _P1)
        z2 = solve_mod_q(ext.Bp % _P2, rhs2, _P2)
        p1p2 = _P1 * _P2
        inv_p1 = pow(_P1, -1, _P2)
        cand = [0] * M
        for i in range(M):
            a1, a2 = int(z1[i]), int(z2[i])
            k = ((a2 - a1) * inv_p1) % _P2
            val = (a1 + _P1 * k) % p1p2
            if val > p1p2 // 2:
                val -= p1p2
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
    for i in range(M - 1, -1, -1):
        ci = _center_of(c, ext.gs[:, i], ext.norms2[i])
        si = sigma / math.sqrt(ext.norms2[i])
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
            ext = ExtBasis(Bp=self._base_ext.Bp,
                           X=center_lift(solve_mod_q(self.A, (-M1) % self.q, self.q),
                                         self.q).astype(np.int64),
                           gs=self._base_ext.gs,
                           norms2=self._base_ext.norms2,
                           max_gs=self._base_ext.max_gs)
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
