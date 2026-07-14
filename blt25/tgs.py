"""Threshold Gaussian preimage sampling (Section 6.1).

Two-round protocol for sampling a discrete Gaussian over the coset
Lambda^c(C) = {v : C v = c mod q} when the gadget trapdoor T_C is tau-of-N
Shamir-shared, using the masking technique of threshold Raccoon and noise
flooding:

  Setup     : Shamir-share T_C entry-wise over Z_q; deal pairwise PRF seeds.
  Round one : party i samples p_i ~ D_{Z^m, sigma_flood}, publishes the
              exposed syndrome e_i = C p_i and row mask
              m_i = sum_{j != i} PRF(seed_{i,j}, sid).
  Round two : all parties compute c' = c - sum_j e_j and y' = G^{-1}(c'),
              publish  sbk_i = lambda_{i,act} T_{C,i} y' + p_i + m*_i
              with column mask m*_i = sum_{j != i} PRF(seed_{j,i}, sid).
  Combine   : sbk = sum_i (sbk_i - m_i);  check C sbk = c.

Correctness: masks cancel, Shamir reconstructs T_C y', and
C sbk = G y' + sum e_i = c (Theorem 4).  ||sbk|| <= B_th =
sqrt(m mt) log q + sqrt(tau) sqrt(m) sigma_flood w(sqrt(log lambda)).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from .bits import random_stream
from .gadget import g_inverse
from .gaussian import sample_dgauss_vec
from .hashes import prf
from .modq import matmul_q


@dataclass
class TGSPartyKey:
    index: int                       # party index i in [1..N]
    N: int
    tau: int
    T_share: np.ndarray              # (m x mt) Shamir share of T_C over Z_q
    seeds_row: dict[int, bytes]      # seed_{i,j} for j != i   (this party = i)
    seeds_col: dict[int, bytes]      # seed_{j,i} for j != i


@dataclass
class TGSRoundOneMsg:
    party: int
    e: np.ndarray                    # syndrome C p_i mod q
    mask_row: np.ndarray             # m_i


@dataclass
class TGSRoundOneState:
    party: int
    sid: int
    p: np.ndarray                    # kept secret for round two


@dataclass
class TGSRoundTwoMsg:
    party: int
    sbk_share: np.ndarray            # lambda_i T_{C,i} y' + p_i + m*_i


def lagrange_at_zero(act: tuple[int, ...], q: int) -> dict[int, int]:
    """lambda_{i,act} = prod_{j in act, j != i} j / (j - i)  mod q."""
    lams = {}
    for i in act:
        num, den = 1, 1
        for j in act:
            if j == i:
                continue
            num = (num * j) % q
            den = (den * (j - i)) % q
        lams[i] = (num * pow(den, -1, q)) % q
    return lams


def setup(T_C: np.ndarray, N: int, tau: int, q: int,
          kappa_bytes: int = 32) -> list[TGSPartyKey]:
    """TGS.Setup: Shamir-share T_C (entries centered in Z_q) and deal seeds.

    Since ||T_C||_inf <= 1 and q > 2, the centered representative of each
    entry mod q is the entry itself, so sharing over Z_q loses nothing.
    """
    m, mt = T_C.shape
    rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(8), "big"))
    # coefficient tensors: degree tau-1 polynomial per entry, constant = T_C
    coeffs = [np.asarray(T_C, dtype=np.int64) % q]
    for _ in range(tau - 1):
        coeffs.append(rng.integers(0, q, size=(m, mt), dtype=np.int64))
    shares = []
    for i in range(1, N + 1):
        acc = np.zeros((m, mt), dtype=np.int64)
        xpow = 1
        for c in coeffs:
            acc = (acc + c * xpow) % q
            xpow = (xpow * i) % q
        shares.append(acc)
    seeds = {(i, j): secrets.token_bytes(kappa_bytes)
             for i in range(1, N + 1) for j in range(1, N + 1) if i != j}
    keys = []
    for i in range(1, N + 1):
        keys.append(TGSPartyKey(
            index=i, N=N, tau=tau, T_share=shares[i - 1],
            seeds_row={j: seeds[(i, j)] for j in range(1, N + 1) if j != i},
            seeds_col={j: seeds[(j, i)] for j in range(1, N + 1) if j != i}))
    return keys


def round_one(key: TGSPartyKey, sid: int, act: tuple[int, ...],
              C: np.ndarray, q: int, sigma_flood: float):
    """TGS.RoundOne: publish (e_i, m_i), keep p_i."""
    assert key.index in act
    m = C.shape[1]
    p = sample_dgauss_vec(m, sigma_flood, random_stream())
    e = matmul_q(C, (p % q).reshape(-1, 1), q).reshape(-1)
    mask = np.zeros(m, dtype=np.int64)
    for j in act:
        if j != key.index:
            mask = (mask + prf(key.seeds_row[j], sid, m, q)) % q
    return (TGSRoundOneMsg(party=key.index, e=e, mask_row=mask),
            TGSRoundOneState(party=key.index, sid=sid, p=p))


def round_two(key: TGSPartyKey, state: TGSRoundOneState, sid: int,
              act: tuple[int, ...], C: np.ndarray, q: int, c: np.ndarray,
              round1: dict[int, TGSRoundOneMsg]) -> TGSRoundTwoMsg:
    """TGS.RoundTwo: publish sbk_i = lambda_i T_{C,i} y' + p_i + m*_i."""
    assert state.sid == sid and set(round1) == set(act)
    n, m = C.shape
    e_sum = np.zeros(n, dtype=np.int64)
    for j in act:
        e_sum = (e_sum + round1[j].e) % q
    c_prime = (np.asarray(c, dtype=np.int64) - e_sum) % q
    y_prime = g_inverse(c_prime, q)                  # binary, deterministic
    lam = lagrange_at_zero(act, q)[key.index]
    ty = matmul_q(key.T_share, y_prime.reshape(-1, 1), q,
                  max_abs_b=1).reshape(-1)
    share = (lam * ty) % q
    share = (share + state.p) % q
    for j in act:
        if j != key.index:
            share = (share + prf(key.seeds_col[j], sid, m, q)) % q
    return TGSRoundTwoMsg(party=key.index, sbk_share=share)


def combine(act: tuple[int, ...], C: np.ndarray, q: int, c: np.ndarray,
            round1: dict[int, TGSRoundOneMsg],
            round2: dict[int, TGSRoundTwoMsg]) -> np.ndarray | None:
    """TGS.Comb: sbk = sum_i (sbk_i - m_i); verify C sbk = c mod q.

    Returns the *centered* short representative of sbk (the honest value is
    short; reduction mod q then centered lift recovers it exactly whenever
    ||sbk||_inf < q/2, which Theorem 4's parameters guarantee)."""
    m = C.shape[1]
    acc = np.zeros(m, dtype=np.int64)
    for i in act:
        acc = (acc + round2[i].sbk_share - round1[i].mask_row) % q
    if not (matmul_q(C, acc.reshape(-1, 1), q).reshape(-1)
            == np.asarray(c, dtype=np.int64) % q).all():
        return None
    centered = np.where(acc > q // 2, acc - q, acc)
    return centered
