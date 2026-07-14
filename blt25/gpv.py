"""Two-round threshold GPV signatures (Section 7).

The first-round transcript fixes the hash target:

    sidsp = Htr(sid, act, (e_i)_{i in act}),   c = HGPV(m, sidsp),

and the second round runs TGS.RoundTwo on target c.  The final signature is
sigma = (sidsp, sigma~) with sigma~ = TGS.Comb(...); verification checks

    C sigma~ = HGPV(m, sidsp)  (mod q)   and   ||sigma~||_2 <= B_th,

where B_th = sqrt(m mt) log q + sqrt(tau) sqrt(m) sigma_flood w(sqrt(log lam))
(Theorem 6).  TS-UF-1 security reduces to ISIS_{q,n,m,B_th} + PRF security
(Theorem 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import tgs
from .bits import stream_from_bytes, random_stream
from .gadget import gadget_dims
from .hashes import hash_gpv, hash_tr
from .modq import matmul_q
from .params import Params, threshold_bounds
from .trapdoor import trapgen


@dataclass
class GPVPublicKey:
    params: Params
    N: int
    tau: int
    C: np.ndarray
    B_th: float


@dataclass
class GPVPartyKey:
    index: int
    pk: GPVPublicKey
    tgs_key: tgs.TGSPartyKey


@dataclass
class Signature:
    sidsp: bytes
    sig: np.ndarray               # sigma~ (centered short vector)


def setup(params: Params, N: int, tau: int, seed: bytes | None = None):
    """TSIG.Setup: pk = C; T_C Shamir-shared via TGS.Setup."""
    if params.sigma_flood <= 0:
        raise ValueError("params.sigma_flood must be set for threshold GPV")
    stream = stream_from_bytes(seed) if seed is not None else random_stream()
    C, T_C = trapgen(params.n, params.q, params.m_c, stream)
    B_th = threshold_bounds(params, N, tau)["B_th"]
    pk = GPVPublicKey(params=params, N=N, tau=tau, C=C, B_th=B_th)
    keys = tgs.setup(T_C, N, tau, params.q, params.kappa_prf)
    return pk, [GPVPartyKey(index=k.index, pk=pk, tgs_key=k) for k in keys]


def sign_one(party: GPVPartyKey, sid: int, act: tuple[int, ...], message: bytes):
    """TSIG.SignOne = TGS.RoundOne."""
    p = party.pk.params
    return tgs.round_one(party.tgs_key, sid, act, party.pk.C, p.q,
                         p.sigma_flood)


def sign_two(party: GPVPartyKey, state: tgs.TGSRoundOneState, sid: int,
             act: tuple[int, ...], message: bytes,
             round1: dict[int, tgs.TGSRoundOneMsg]):
    """TSIG.SignTwo: derive c = HGPV(m, sidsp) and emit the TGS share."""
    p = party.pk.params
    sidsp = hash_tr(sid, act, [round1[j].e for j in sorted(round1)])
    c = hash_gpv(message, sidsp, p.n, p.q)
    share = tgs.round_two(party.tgs_key, state, sid, act, party.pk.C, p.q, c,
                          round1)
    return sidsp, share


def combine(pk: GPVPublicKey, sid: int, act: tuple[int, ...], message: bytes,
            round1: dict[int, tgs.TGSRoundOneMsg],
            round2: dict[int, tuple[bytes, tgs.TGSRoundTwoMsg]]) -> Signature | None:
    """TSIG.Comb."""
    p = pk.params
    sidsps = {s for s, _ in round2.values()}
    if len(sidsps) != 1:
        return None
    sidsp = next(iter(sidsps))
    c = hash_gpv(message, sidsp, p.n, p.q)
    shares = {i: sh for i, (_, sh) in round2.items()}
    sig = tgs.combine(act, pk.C, p.q, c, round1, shares)
    if sig is None:
        return None
    return Signature(sidsp=sidsp, sig=sig)


def verify(pk: GPVPublicKey, message: bytes, signature: Signature) -> bool:
    """TSIG.Ver: C sigma~ = HGPV(m, sidsp) mod q and ||sigma~|| <= B_th."""
    p = pk.params
    q = p.q
    sig = np.asarray(signature.sig, dtype=np.int64)
    norm = math.sqrt(float(np.dot(sig.astype(np.float64), sig.astype(np.float64))))
    if norm > pk.B_th:
        return False
    lhs = matmul_q(pk.C, (sig % q).reshape(-1, 1), q).reshape(-1)
    target = hash_gpv(message, signature.sidsp, p.n, q)
    return bool((lhs == target % q).all())
