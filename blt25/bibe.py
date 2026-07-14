"""The batch IBE scheme of Section 5 (epochless, LWE-based).

Algorithms (paper notation in brackets):

- ``setup``    [BIBE.Setup]   pk = (crs_samp, C, u), sk = (pk, T_C).
- ``encrypt``  [BIBE.Enc]     dual-Regev ciphertext under F = [A_r | Hro(r)].
- ``pre_dec``  [BIBE.PreDec]  short pre-decryption key sbk with C sbk = G c-hat.
- ``decrypt``  [BIBE.Dec]     recomputes the same SampleLeft output (via the
                              Hsp random tape) and rounds.

Multi-bit messages (Section 5.3, first variant): the public key carries
``msg_bits`` target vectors u_1..u_N; each bit is encrypted under its own u_j
with the same ephemeral s, and PreDec produces one (sbk_j, v_j) pair per bit.

Batches of size ell' <= ell are supported unchanged (Remark 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import www24
from .bits import BitStream, random_stream, stream_from_bytes
from .gadget import gadget_matrix
from .gaussian import sample_dgauss_vec
from .hashes import hash_index, hash_ro, hash_sp, hash_sp_th
from .modq import matmul_q
from .params import Params
from .trapdoor import PreimageSampler, trapgen


@dataclass
class PublicKey:
    params: Params
    crs: www24.CRS
    C: np.ndarray                 # (n x m_c)
    us: np.ndarray                # (msg_bits x n) target vectors u_j

    def fingerprint(self) -> bytes:
        import hashlib
        h = hashlib.sha256()
        h.update(self.C.tobytes())
        h.update(self.crs.A.tobytes())
        h.update(self.crs.B.tobytes())
        h.update(self.us.tobytes())
        return h.digest()


@dataclass
class SecretKey:
    pk: PublicKey
    T_C: np.ndarray               # gadget trapdoor: C T_C = G (Section 6 shares this)
    Abar: np.ndarray              # TrapGen parts enabling the exact
    R: np.ndarray                 # Lambda^perp(C) basis (see AUDIT.md)


@dataclass
class Ciphertext:
    r: object                     # identity tag
    ct1: np.ndarray               # (msg_bits,)
    ct2: np.ndarray               # (m_c,)
    ct3: np.ndarray               # (t + m_ro,)


@dataclass
class PreDecKey:
    identities: tuple
    sbks: list[np.ndarray]        # one short vector per message bit

    def norm(self) -> float:
        return max(float(np.linalg.norm(s.astype(np.float64))) for s in self.sbks)


def setup(params: Params, seed: bytes | None = None):
    """BIBE.Setup(1^lambda, 1^ell) -> (pk, sk)."""
    stream = stream_from_bytes(seed) if seed is not None else random_stream()
    crs = www24.gen(params.n, params.q, params.d, stream)
    C, T_C, Abar, R = trapgen(params.n, params.q, params.m_c, stream,
                              return_parts=True)
    us = stream.uniform_mod_mat(params.msg_bits, params.n, params.q)
    pk = PublicKey(params=params, crs=crs, C=C, us=us)
    return pk, SecretKey(pk=pk, T_C=T_C, Abar=Abar, R=R)


def _chi_vec(k: int, params: Params, stream: BitStream) -> np.ndarray:
    """LWE noise chi = discrete Gaussian of width sigma_chi (rho convention)."""
    return sample_dgauss_vec(k, params.sigma_chi, stream)


def encrypt(pk: PublicKey, r, message: int,
            stream: BitStream | None = None) -> Ciphertext:
    """BIBE.Enc(pk, r, m).  ``message`` is an integer in [0, 2^msg_bits)."""
    p = pk.params
    if not 0 <= message < (1 << p.msg_bits):
        raise ValueError("message out of range")
    stream = stream or random_stream()
    q = p.q
    s = stream.uniform_mod_vec(p.n, q)
    h_idx = hash_index(r, p.d)
    Ar = www24.expand_local(pk.crs, h_idx)                     # (n x t)
    Hr = hash_ro(r, p.n, p.m_ro, q)                            # (n x m_ro)
    eps1 = _chi_vec(p.msg_bits, p, stream)
    eps2 = _chi_vec(p.m_c, p, stream)
    eps3_pre = _chi_vec(p.t, p, stream)
    Rt = stream.pm1_matrix(p.t, p.m_ro)                        # R~ in {-1,1}^{t x m_ro}
    eps3 = np.concatenate([eps3_pre, Rt.T @ eps3_pre])
    F = np.concatenate([Ar, Hr], axis=1)                       # (n x (t + m_ro))
    half_q = q // 2
    bits = np.array([(message >> j) & 1 for j in range(p.msg_bits)], dtype=np.int64)
    ct1 = (matmul_q(pk.us, s.reshape(-1, 1), q).reshape(-1) + eps1 + bits * half_q) % q
    ct2 = (matmul_q(pk.C.T, s.reshape(-1, 1), q).reshape(-1) + eps2) % q
    ct3 = (matmul_q(F.T, s.reshape(-1, 1), q).reshape(-1) + eps3) % q
    return Ciphertext(r=r, ct1=ct1, ct2=ct2, ct3=ct3)


# ----------------------------------------------------------------------------
# The public SampleLeft derivation shared by PreDec and Dec
# ----------------------------------------------------------------------------

@dataclass
class Derivation:
    """Publicly computable per-batch data: the WWW24 expansion, and for every
    message bit j the SampleLeft output split into (v_i, c-hat, c)."""
    identities: tuple
    exp: www24.Expanded
    v_blocks: list[list[np.ndarray]]   # [bit j][batch i] -> v_i = (v_i^1, v_i^2)
    chats: list[np.ndarray]            # [bit j] -> c-hat
    cs: list[np.ndarray]               # [bit j] -> c = G c-hat mod q


_derivation_cache: dict[bytes, Derivation] = {}


def derive(pk: PublicKey, identities, sidsp: bytes | None = None) -> Derivation:
    """Common PreDec/Dec computation (deterministic).

    When ``sidsp`` is given, the random tape is drawn from the domain-separated
    oracle Hsp^th (threshold scheme, Definition 17); otherwise from Hsp
    (Definition 15).
    """
    p = pk.params
    identities = tuple(identities)
    if len(identities) > p.ell:
        raise ValueError(f"batch too large ({len(identities)} > ell={p.ell})")
    key = pk.fingerprint() + repr(identities).encode() + (sidsp or b"")
    cached = _derivation_cache.get(key)
    if cached is not None:
        return cached
    idx = [hash_index(r, p.d) for r in identities]
    if len(set(idx)) != len(idx):
        raise ValueError("index collision: H(r_i) not pairwise distinct")
    exp = www24.expand(pk.crs, idx)
    ell = len(identities)
    q = p.q
    Hros = [hash_ro(r, p.n, p.m_ro, q) for r in identities]
    Ahat = np.zeros((ell * p.n, ell * p.m_ro), dtype=np.int64)
    for i, Hr in enumerate(Hros):
        Ahat[i * p.n:(i + 1) * p.n, i * p.m_ro:(i + 1) * p.m_ro] = Hr
    ps = exp.sampler()
    if p.sigma1 < ps.max_gs:
        raise ValueError(
            f"sigma1={p.sigma1} below the extended-basis GS norm {ps.max_gs:.1f}; "
            "increase sigma1 in the parameter set")
    G = gadget_matrix(p.n, q)
    t, mt = p.t, p.mt
    v_blocks, chats, cs = [], [], []
    rho = ps.tape_len_bits(ell * p.m_ro)
    for j in range(p.msg_bits):
        u = pk.us[j]
        h = np.tile(u, ell)
        if sidsp is None:
            rb = hash_sp(exp.D, exp.T, identities, u, int(p.sigma1), rho)
        else:
            rb = hash_sp_th(exp.D, exp.T, identities, u, int(p.sigma1), sidsp, rho)
        v = ps.sample_left(Ahat, h, p.sigma1, rb)
        v1 = [v[i * t:(i + 1) * t] for i in range(ell)]
        chat = v[ell * t: ell * t + mt]
        v2 = [v[ell * t + mt + i * p.m_ro: ell * t + mt + (i + 1) * p.m_ro]
              for i in range(ell)]
        v_blocks.append([np.concatenate([v1[i], v2[i]]) for i in range(ell)])
        chats.append(chat)
        cs.append(matmul_q(G, (chat % q).reshape(-1, 1), q).reshape(-1))
    der = Derivation(identities=identities, exp=exp, v_blocks=v_blocks,
                     chats=chats, cs=cs)
    if len(_derivation_cache) > 8:
        _derivation_cache.clear()
    _derivation_cache[key] = der
    return der


def pre_dec(sk: SecretKey, identities,
            stream: BitStream | None = None) -> PreDecKey:
    """BIBE.PreDec(sk, (r_i)) -> sbk with C sbk_j = G c-hat_j."""
    pk = sk.pk
    p = pk.params
    der = derive(pk, identities)
    sampler = PreimageSampler.from_trapgen_parts(pk.C, sk.Abar, sk.R, p.q)
    sbks = []
    for j in range(p.msg_bits):
        sbk = sampler.sample_pre(der.cs[j], p.sigma2, stream)
        sbks.append(sbk)
    return PreDecKey(identities=der.identities, sbks=sbks)


def _round_bit(x: int, q: int) -> int:
    """round(x): 1 iff |x - floor(q/2)| < floor(q/4)  (Section 5, Dec)."""
    return 1 if abs(int(x) - q // 2) < q // 4 else 0


def dot_q(a: np.ndarray, b: np.ndarray, q: int) -> int:
    """<a, b> mod q, overflow-safe."""
    return int(matmul_q((a % q).reshape(1, -1), (b % q).reshape(-1, 1), q)[0, 0])


def decrypt(pk: PublicKey, sbk: PreDecKey, cts: list[Ciphertext],
            return_error: bool = False):
    """BIBE.Dec(pk, sbk, (ct_i)) -> messages (and optional error magnitudes)."""
    p = pk.params
    q = p.q
    identities = tuple(ct.r for ct in cts)
    if identities != sbk.identities:
        raise ValueError("ciphertext identities do not match the pre-decryption key")
    der = derive(pk, identities)
    msgs, errs = [], []
    for i, ct in enumerate(cts):
        message = 0
        worst = 0
        for j in range(p.msg_bits):
            vi = der.v_blocks[j][i]
            inner = (int(ct.ct1[j])
                     - dot_q(sbk.sbks[j], ct.ct2, q)
                     - dot_q(vi, ct.ct3, q)) % q
            bit = _round_bit(inner, q)
            message |= bit << j
            err = (inner - bit * (q // 2)) % q
            err = err if err <= q // 2 else err - q
            worst = max(worst, abs(err))
        msgs.append(message)
        errs.append(worst)
    return (msgs, errs) if return_error else msgs
