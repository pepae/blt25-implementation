"""Batch IBE from key-policy ABE (Section 8), with a concrete BGG+14-style
KP-ABE instantiation for the membership function class

    F_{k,ell} = { f_S : {0,1}^k -> {0,1},  f_S(x) = 0  iff  x in S,  |S| <= ell }.

KP-ABE construction (specializing [BGG+14] to F_{k,ell} via the indicator
machinery of Appendix B):

  Setup   : mpk = (A0 with trapdoor, B in Z_q^{n x m~ k}, u).
  Enc     : dual-Regev under [A0 | B - x^T (x) G]:
              ct0 = u^T s + e0 + mu floor(q/2)
              ct1 = A0^T s + e1
              ct2 = (B - x^T (x) G)^T s + R~^T e1   (R~ <- {+-1})
  KeyGen  : B_S := sum_{u in S} EvalF(B, delta_u); since the labels in S are
            distinct, f_S(x) = 1 - sum_u delta_u(x), and
              (B - x^T (x) G) H_S = B_S - (1 - f_S(x)) G,
            for H_S := sum_u EvalFX(B, delta_u, x) with ||H_S||_inf <= |S|.
            The key is sk_f <- SampleLeft(A0, B_S - G, T_A0, u, sigma) so that
            [A0 | B_S - G] sk_f = u.
  Dec     : when f_S(x) = 0, transform ct2' := H_S^T ct2, so (ct1, ct2') is a
            dual-Regev ciphertext under [A0 | B_S - G]; round
            ct0 - sk_f^T (ct1, ct2').

The generic BIBE-from-ABE transformation (Section 8.2) then sets the
attribute x := H(r) and S := {H(r_i)}: the ABE key for f_S *is* the
pre-decryption key.  Its size is (m_c + m~) log q - independent of ell up to
the log q growth - matching the BGG+14 row of Table 1.

The paper notes it is unclear how to thresholdize this construction; no
threshold variant is provided (that is the point of Sections 5-6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import www24
from .bits import BitStream, random_stream, stream_from_bytes
from .gadget import gadget_dims, gadget_matrix
from .gaussian import sample_dgauss_vec
from .hashes import hash_index
from .modq import matmul_q
from .params import Params
from .trapdoor import PreimageSampler, trapgen


@dataclass
class ABEPublicKey:
    params: Params            # d is interpreted as the attribute length k
    A0: np.ndarray            # (n x m_c)
    B: np.ndarray             # (n x m~ k)
    u: np.ndarray             # (n,)


@dataclass
class ABESecretKey:
    mpk: ABEPublicKey
    Abar: np.ndarray
    R: np.ndarray


@dataclass
class ABECiphertext:
    x: tuple[int, ...]        # attribute
    ct0: int
    ct1: np.ndarray
    ct2: np.ndarray


@dataclass
class ABEFunctionKey:
    S: tuple[tuple[int, ...], ...]   # the membership set (sorted labels)
    sk: np.ndarray                   # (m_c + m~,)


def _crs_view(mpk: ABEPublicKey) -> www24.CRS:
    """Reuse the Appendix-B EvalF/EvalFX code: wrap B in a CRS whose A part is
    unused by the evaluation algorithms."""
    p = mpk.params
    mt = gadget_dims(p.n, p.q)
    return www24.CRS(n=p.n, q=p.q, d=p.d, A=np.zeros((p.n, mt), dtype=np.int64),
                     B=mpk.B)


def setup(params: Params, seed: bytes | None = None):
    """ABE.Setup(1^lambda, F_{k,ell})."""
    stream = stream_from_bytes(seed) if seed is not None else random_stream()
    A0, _T, Abar, R = trapgen(params.n, params.q, params.m_c, stream,
                              return_parts=True)
    mt = gadget_dims(params.n, params.q)
    B = stream.uniform_mod_mat(params.n, mt * params.d, params.q)
    u = stream.uniform_mod_vec(params.n, params.q)
    mpk = ABEPublicKey(params=params, A0=A0, B=B, u=u)
    return mpk, ABESecretKey(mpk=mpk, Abar=Abar, R=R)


def encrypt(mpk: ABEPublicKey, x: tuple[int, ...], mu: int,
            stream: BitStream | None = None) -> ABECiphertext:
    """ABE.Enc(mpk, x, mu) for mu in {0,1}."""
    p = mpk.params
    q = p.q
    assert len(x) == p.d and mu in (0, 1)
    stream = stream or random_stream()
    mt = gadget_dims(p.n, q)
    s = stream.uniform_mod_vec(p.n, q)
    e0 = sample_dgauss_vec(1, p.sigma_chi, stream)[0]
    e1 = sample_dgauss_vec(p.m_c, p.sigma_chi, stream)
    crs = _crs_view(mpk)
    Bx = (mpk.B - www24._label_shift(crs, x)) % q
    Rt = stream.pm1_matrix(p.m_c, mt * p.d)
    e2 = Rt.T @ e1
    ct0 = (int(np.dot(mpk.u % q, s) % q) + int(e0) + mu * (q // 2)) % q
    ct1 = (matmul_q(mpk.A0.T, s.reshape(-1, 1), q).reshape(-1) + e1) % q
    ct2 = (matmul_q(Bx.T, s.reshape(-1, 1), q).reshape(-1) + e2) % q
    return ABECiphertext(x=tuple(x), ct0=ct0, ct1=ct1, ct2=ct2)


def _bs_and_hs(mpk: ABEPublicKey, S, x: tuple[int, ...] | None):
    """B_S = sum EvalF, and (if x given) H_S = sum EvalFX."""
    crs = _crs_view(mpk)
    q = mpk.params.q
    B_S = None
    H_S = None
    for u_lab in S:
        Bu, chain = www24.eval_f(crs, u_lab)
        B_S = Bu if B_S is None else (B_S + Bu) % q
        if x is not None:
            H = www24.eval_fx(crs, u_lab, x, chain)
            H_S = H if H_S is None else H_S + H
    return B_S, H_S


def keygen(msk: ABESecretKey, S, sigma: float | None = None,
           stream: BitStream | None = None) -> ABEFunctionKey:
    """ABE.KeyGen(msk, f_S): sk with [A0 | B_S - G] sk = u."""
    mpk = msk.mpk
    p = mpk.params
    q = p.q
    S = tuple(sorted(tuple(lab) for lab in S))
    assert len(set(S)) == len(S) <= p.ell, "S must be distinct and |S| <= ell"
    B_S, _ = _bs_and_hs(mpk, S, None)
    G = gadget_matrix(p.n, q)
    M1 = (B_S - G) % q
    sampler = PreimageSampler.from_trapgen_parts(mpk.A0, msk.Abar, msk.R, q)
    sigma = sigma or p.sigma2 * 4
    stream = stream or random_stream()
    # randomized SampleLeft (key generation needs no determinism): fresh tape
    rb_bits = sampler.tape_len_bits(M1.shape[1])
    rb = bytes(stream.take_bits(8) for _ in range((rb_bits + 7) // 8))
    sk = sampler.sample_left(M1, mpk.u, sigma, rb)
    return ABEFunctionKey(S=S, sk=sk)


def decrypt(mpk: ABEPublicKey, fkey: ABEFunctionKey,
            ct: ABECiphertext) -> int | None:
    """ABE.Dec(sk_f, (x, ct)): rounds when f_S(x) = 0, else returns None."""
    p = mpk.params
    q = p.q
    if tuple(ct.x) not in fkey.S:          # f_S(x) != 0
        return None
    _, H_S = _bs_and_hs(mpk, fkey.S, tuple(ct.x))
    # ct2' = H_S^T ct2 (mod q); |H_S| <= ell so the int64 product is safe
    ct2p = matmul_q(H_S.T % q, ct.ct2.reshape(-1, 1), q,
                    max_abs_a=q - 1).reshape(-1)
    vec = np.concatenate([ct.ct1, ct2p])
    inner = (int(ct.ct0) - int(matmul_q((fkey.sk % q).reshape(1, -1),
                                        vec.reshape(-1, 1), q)[0, 0])) % q
    return 1 if abs(inner - q // 2) < q // 4 else 0


# ----------------------------------------------------------------------------
# Section 8.2: the generic BIBE-from-ABE transformation
# ----------------------------------------------------------------------------

@dataclass
class ABEBIBECiphertext:
    r: object
    ct: ABECiphertext


def bibe_setup(params: Params, seed: bytes | None = None):
    """BIBE.Setup := ABE.Setup (Section 8.2)."""
    return setup(params, seed)


def bibe_encrypt(pk: ABEPublicKey, r, m: int,
                 stream: BitStream | None = None) -> ABEBIBECiphertext:
    """BIBE.Enc: attribute := H(r)."""
    x = hash_index(r, pk.params.d)
    return ABEBIBECiphertext(r=r, ct=encrypt(pk, x, m, stream))


def bibe_pre_dec(sk: ABESecretKey, identities) -> ABEFunctionKey:
    """BIBE.PreDec := ABE.KeyGen(msk, f_S) for S = {H(r_i)}."""
    d = sk.mpk.params.d
    S = [hash_index(r, d) for r in identities]
    if len(set(S)) != len(S):
        raise ValueError("index collision among identities")
    return keygen(sk, S)


def bibe_decrypt(pk: ABEPublicKey, fkey: ABEFunctionKey,
                 cts: list[ABEBIBECiphertext]) -> list[int | None]:
    """BIBE.Dec := ABE.Dec per ciphertext."""
    return [decrypt(pk, fkey, c.ct) for c in cts]
