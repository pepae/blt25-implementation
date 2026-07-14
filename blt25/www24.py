"""Shifted multi-preimage trapdoor sampler with d-bit labels.

Implements Appendix B of the paper: the Waters-Wee-Wu [62, Construction 4.6]
sampler adapted to d-bit labels via sparse indicator-function evaluation
(Lemma 29, Lemma 30), so that trapdoor generation runs in time
poly(n, m, ell, d, log q) instead of 2^d * poly(...).

Objects
-------
crs = [A | B],  A in Z_q^{n x m~},  B in Z_q^{n x m~ d},  t = m~ (d+1).

ExpandLocal(crs, u) = A_u := [A | B - u^T (x) G]  in Z_q^{n x t}.

Expand(crs, (u_i)) additionally builds

    D_ell := [ diag(A_{u_1}, ..., A_{u_ell}) | col(G, ..., G) ]
             in Z_q^{ell n x (ell t + m~)},

and a gadget trapdoor T with D_ell T = I_ell (x) G = G_{ell n},
||T||_inf <= 1, assembled from StructTrapGen (Lemma 30) with the EvalF /
EvalFX algorithms for indicator functions delta_u (Lemma 29).

SampleMultPre(td, t_1..t_ell) samples col(pi_1, .., pi_ell, c-hat) as a
Gaussian preimage of col(t_i) under D_ell and outputs the shift c := -G c-hat
with A_{u_i} pi_i = t_i + c for every i (Definition 24 correctness).

GenProg(u, A*) programs the CRS so that ExpandLocal(crs, u) = A*
(perfect somewhere programmability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bits import BitStream, random_stream
from .gadget import g_inverse, gadget_dims, gadget_len, gadget_matrix
from .modq import matmul_q
from .trapdoor import PreimageSampler

Label = tuple[int, ...]  # d bits, each 0/1


@dataclass(frozen=True)
class CRS:
    n: int
    q: int
    d: int
    A: np.ndarray  # (n, m~)
    B: np.ndarray  # (n, m~ d)

    @property
    def mt(self) -> int:
        return gadget_dims(self.n, self.q)

    @property
    def t(self) -> int:
        return self.mt * (self.d + 1)

    def b_block(self, r: int) -> np.ndarray:
        """B_r, the r-th (1-indexed) m~-column block of B."""
        mt = self.mt
        return self.B[:, (r - 1) * mt: r * mt]


def gen(n: int, q: int, d: int, stream: BitStream | None = None) -> CRS:
    """Gen(1^lambda, 1^ell, 1^d): crs = [A | B] uniform.  (The CRS does not
    depend on ell; transparent setup.)"""
    stream = stream or random_stream()
    mt = gadget_dims(n, q)
    A = stream.uniform_mod_mat(n, mt, q)
    B = stream.uniform_mod_mat(n, mt * d, q)
    return CRS(n=n, q=q, d=d, A=A, B=B)


def _label_shift(crs: CRS, u: Label) -> np.ndarray:
    """u^T (x) G  in Z_q^{n x m~ d}."""
    G = gadget_matrix(crs.n, crs.q)
    blocks = [(G if bit else np.zeros_like(G)) for bit in u]
    return np.concatenate(blocks, axis=1)


def expand_local(crs: CRS, u: Label) -> np.ndarray:
    """ExpandLocal: A_u = [A | B - u^T (x) G]  (deterministic, local)."""
    assert len(u) == crs.d
    Bu = (crs.B - _label_shift(crs, u)) % crs.q
    return np.concatenate([crs.A, Bu], axis=1)


def gen_prog(n: int, q: int, d: int, u: Label, A_star: np.ndarray) -> CRS:
    """GenProg: crs such that ExpandLocal(crs, u) = A_star (perfectly)."""
    mt = gadget_dims(n, q)
    assert A_star.shape == (n, mt * (d + 1))
    A = A_star[:, :mt] % q
    Bp = A_star[:, mt:] % q
    crs0 = CRS(n=n, q=q, d=d, A=A, B=Bp)
    B = (Bp + _label_shift(crs0, u)) % q
    return CRS(n=n, q=q, d=d, A=A, B=B)


# ----------------------------------------------------------------------------
# EvalF / EvalFX for indicator functions (Lemma 29)
# ----------------------------------------------------------------------------

def eval_f(crs: CRS, u: Label):
    """EvalF(B, delta_u) -> (B_u, R_chain).

    F^(0) = G; for r = 1..d:  R^(r-1) = G^{-1}(F^(r-1)),
    F^(r) = B_r R^(r-1) if u_r = 1 else F^(r-1) - B_r R^(r-1).
    Returns B_u = F^(d) and the chain (R^(0), ..., R^(d-1)) reused by EvalFX.
    """
    n, q = crs.n, crs.q
    F = gadget_matrix(n, q) % q
    chain = []
    for r in range(1, crs.d + 1):
        R = g_inverse(F, q)                       # (m~ x m~), bits
        chain.append(R)
        BrR = matmul_q(crs.b_block(r), R, q, max_abs_b=1)
        F = BrR if u[r - 1] == 1 else (F - BrR) % q
    return F % q, chain


def eval_fx(crs: CRS, u: Label, x: Label, chain=None) -> np.ndarray:
    """EvalFX(B, delta_u, x) -> H in {-1,0,1}^{m~ d x m~} with
    (B - x^T (x) G) H = B_u - delta_u(x) G   (mod q)."""
    if chain is None:
        _, chain = eval_f(crs, u)
    d, mt = crs.d, crs.mt
    H = np.zeros((mt * d, mt), dtype=np.int64)
    for r in range(1, d + 1):
        mu = 1
        for s in range(r + 1, d + 1):
            if x[s - 1] != u[s - 1]:
                mu = 0
                break
        if mu == 0:
            continue
        eta = 1 if u[r - 1] == 1 else -1
        H[(r - 1) * mt: r * mt, :] = eta * chain[r - 1]
    return H


# ----------------------------------------------------------------------------
# StructTrapGen (Lemma 30) and Expand
# ----------------------------------------------------------------------------

def struct_trapgen(crs: CRS, us: list[Label]) -> np.ndarray:
    """T' in {-1,0,1}^{(ell m~ d + m~) x ell m~} with D'_ell T' = I_ell (x) G,
    where D'_ell = [diag(B - u_i^T (x) G) | col(G)].

    Block layout (Lemma 30):
        T'[(i-block), (j-block)] = -H_{B, u_j, u_i}
        T'[bottom,   (j-block)] = G^{-1}(B_{u_j})
    """
    ell = len(us)
    assert len(set(us)) == ell, "labels must be distinct"
    mt, d, q = crs.mt, crs.d, crs.q
    Bus, chains = [], []
    for u in us:
        Bu, ch = eval_f(crs, u)
        Bus.append(Bu)
        chains.append(ch)
    T = np.zeros((ell * mt * d + mt, ell * mt), dtype=np.int64)
    for j in range(ell):
        for i in range(ell):
            H = eval_fx(crs, us[j], us[i], chain=chains[j])
            T[i * mt * d: (i + 1) * mt * d, j * mt: (j + 1) * mt] = -H
        T[ell * mt * d:, j * mt: (j + 1) * mt] = g_inverse(Bus[j], q)
    return T


@dataclass
class Expanded:
    """Output of Expand: matrices A_i, the joint matrix D (eq. (3)), and its
    gadget trapdoor T (D T = G_{ell n}, ||T||_inf <= 1)."""
    crs: CRS
    us: list[Label]
    A_list: list[np.ndarray]     # each (n x t)
    D: np.ndarray                # (ell n x (ell t + m~))
    T: np.ndarray                # ((ell t + m~) x ell m~)

    _sampler: PreimageSampler | None = None

    @property
    def ell(self) -> int:
        return len(self.us)

    def sampler(self) -> PreimageSampler:
        if self._sampler is None:
            self._sampler = PreimageSampler(self.D, self.T, self.crs.q)
        return self._sampler


def expand(crs: CRS, us: list[Label]) -> Expanded:
    """Expand(1^lambda, 1^ell, 1^d, crs, (u_i)): deterministic."""
    ell = len(us)
    n, q, mt, t = crs.n, crs.q, crs.mt, crs.t
    A_list = [expand_local(crs, u) for u in us]
    G = gadget_matrix(n, q)
    # D = [diag(A_i) | col(G)]
    D = np.zeros((ell * n, ell * t + mt), dtype=np.int64)
    for i, Ai in enumerate(A_list):
        D[i * n:(i + 1) * n, i * t:(i + 1) * t] = Ai
        D[i * n:(i + 1) * n, ell * t:] = G
    Tp = struct_trapgen(crs, us)
    # assemble T for D's natural column order:
    #   rows of block i's A-part (m~) -> 0
    #   rows of block i's B-part (m~ d) -> T' rows for block i
    #   final G-part rows (m~) -> T' bottom rows
    T = np.zeros((ell * t + mt, ell * mt), dtype=np.int64)
    md = mt * crs.d
    for i in range(ell):
        T[i * t + mt: (i + 1) * t, :] = Tp[i * md: (i + 1) * md, :]
    T[ell * t:, :] = Tp[ell * md:, :]
    return Expanded(crs=crs, us=list(us), A_list=A_list, D=D, T=T)


def sample_mult_pre(exp: Expanded, targets: list[np.ndarray], sigma: float,
                    stream: BitStream | None = None):
    """SampleMultPre(td, t_1..t_ell) -> (pi_1..pi_ell, c) with
    A_i pi_i = t_i + c (mod q) for all i, c = -G c-hat."""
    crs, ell, t = exp.crs, exp.ell, exp.crs.t
    assert len(targets) == ell
    tvec = np.concatenate([np.asarray(ti, dtype=np.int64) % crs.q for ti in targets])
    v = exp.sampler().sample_pre(tvec, sigma, stream)
    pis = [v[i * t:(i + 1) * t] for i in range(ell)]
    chat = v[ell * t:]
    G = gadget_matrix(crs.n, crs.q)
    c = (-matmul_q(G, (chat % crs.q).reshape(-1, 1), crs.q).reshape(-1)) % crs.q
    return pis, c
