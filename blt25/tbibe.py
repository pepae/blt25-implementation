"""Threshold batch IBE (Section 6.2).

Wraps the BIBE scheme of Section 5 with the two-round threshold Gaussian
preimage sampler of Section 6.1.  The SampleLeft random tape is drawn from the
domain-separated oracle Hsp^th whose extra input is the transcript tag

    sidsp = Htr(sid, act, (e_i)_{i in act}),

binding the target c = G c-hat to the first-round syndromes (Definitions
16/17) - the simulator must know the syndromes before programming c.

Message flow per pre-decryption query (sid, act, batch):

  PredecOne  : party i -> (e_i, m_i)             [TGS round one]
  PredecTwo  : party i -> (m_i, sbk_i, sidsp, c) [derives c via Hsp^th tape,
                                                  TGS round two]
  Comb       : checks all sidsp/c equal, runs TGS.Comb, outputs
               pdk = (act, sidsp, sbk)
  Dec        : recomputes the derivation with sidsp, checks C sbk = c,
               then decrypts as in Section 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bibe, tgs
from .hashes import hash_tr
from .modq import matmul_q
from .params import Params


@dataclass
class TBIBEPartyKey:
    index: int
    pk: bibe.PublicKey
    tgs_key: tgs.TGSPartyKey


@dataclass
class PredecOneOut:
    party: int
    msg: tgs.TGSRoundOneMsg
    state: tgs.TGSRoundOneState   # secret; kept by the party


@dataclass
class PredecTwoOut:
    party: int
    mask_row: np.ndarray          # m_i (echoed for the combiner)
    sbk_share: tgs.TGSRoundTwoMsg
    sidsp: bytes
    cs: list[np.ndarray]          # target c per message bit


@dataclass
class ThresholdPreDecKey:
    act: tuple[int, ...]
    sidsp: bytes
    identities: tuple
    sbks: list[np.ndarray]        # combined short keys, one per message bit


def setup(params: Params, N: int, tau: int, seed: bytes | None = None):
    """TBIBE.Setup: BIBE.Setup + TGS.Setup(T_C).  Returns (pk, party keys)."""
    pk, sk = bibe.setup(params, seed)
    keys = tgs.setup(sk.T_C, N, tau, params.q, params.kappa_prf)
    parties = [TBIBEPartyKey(index=k.index, pk=pk, tgs_key=k) for k in keys]
    return pk, parties, sk       # sk returned for test cross-checks only


encrypt = bibe.encrypt          # TBIBE.Enc == BIBE.Enc


def predec_one(party: TBIBEPartyKey, sid: int, act: tuple[int, ...],
               identities) -> PredecOneOut:
    """TBIBE.PredecOne: TGS round one (independent of the batch)."""
    p = party.pk.params
    msg, state = tgs.round_one(party.tgs_key, sid, act, party.pk.C, p.q,
                               p.sigma_flood)
    return PredecOneOut(party=party.index, msg=msg, state=state)


def predec_two(party: TBIBEPartyKey, state: tgs.TGSRoundOneState, sid: int,
               act: tuple[int, ...], identities,
               round1: dict[int, tgs.TGSRoundOneMsg]) -> PredecTwoOut:
    """TBIBE.PredecTwo: derive the batch targets via Hsp^th (bound to the
    round-one transcript) and emit the TGS round-two share."""
    pk = party.pk
    p = pk.params
    if p.sigma_flood <= 0:
        raise ValueError("params.sigma_flood must be set for the threshold scheme")
    sidsp = hash_tr(sid, act, [round1[j].e for j in sorted(round1)])
    der = bibe.derive(pk, identities, sidsp=sidsp)
    shares = []
    for j in range(p.msg_bits):
        shares.append(tgs.round_two(party.tgs_key, state, sid, act, pk.C,
                                    p.q, der.cs[j], round1))
    # message bits share one round-1 (p_i is reused across bits only if the
    # caller runs one bit per sid; for msg_bits > 1 we require msg_bits = 1
    # or independent sids per bit - enforced here for safety.
    if p.msg_bits != 1:
        raise NotImplementedError(
            "threshold pre-decryption is specified for 1-bit messages; run "
            "one TGS session per message bit for the multi-bit variant")
    return PredecTwoOut(party=party.index, mask_row=round1[party.index].mask_row,
                        sbk_share=shares[0], sidsp=sidsp, cs=der.cs)


def combine(pk: bibe.PublicKey, sid: int, act: tuple[int, ...], identities,
            round1: dict[int, tgs.TGSRoundOneMsg],
            round2: dict[int, PredecTwoOut]) -> ThresholdPreDecKey | None:
    """TBIBE.Comb: check sidsp/c consistency, then TGS.Comb."""
    p = pk.params
    sidsps = {r2.sidsp for r2 in round2.values()}
    if len(sidsps) != 1:
        return None
    sidsp = next(iter(sidsps))
    c_refs = [tuple(int(x) for x in r2.cs[0]) for r2 in round2.values()]
    if len(set(c_refs)) != 1:
        return None
    c = round2[next(iter(round2))].cs[0]
    shares = {i: r2.sbk_share for i, r2 in round2.items()}
    sbk = tgs.combine(act, pk.C, p.q, c, round1, shares)
    if sbk is None:
        return None
    return ThresholdPreDecKey(act=act, sidsp=sidsp,
                              identities=tuple(identities), sbks=[sbk])


def decrypt(pk: bibe.PublicKey, pdk: ThresholdPreDecKey,
            cts: list[bibe.Ciphertext], return_error: bool = False):
    """TBIBE.Dec: recompute the sidsp-bound derivation, verify C sbk = c,
    then decrypt as in Section 5."""
    p = pk.params
    q = p.q
    identities = tuple(ct.r for ct in cts)
    if identities != pdk.identities:
        raise ValueError("ciphertexts do not match the pre-decryption key")
    der = bibe.derive(pk, identities, sidsp=pdk.sidsp)
    for j, sbk in enumerate(pdk.sbks):
        lhs = matmul_q(pk.C, (sbk % q).reshape(-1, 1), q).reshape(-1)
        if not (lhs == der.cs[j] % q).all():
            raise ValueError("pre-decryption key fails the C sbk = c check")
    msgs, errs = [], []
    for i, ct in enumerate(cts):
        message, worst = 0, 0
        for j in range(p.msg_bits):
            vi = der.v_blocks[j][i]
            inner = (int(ct.ct1[j])
                     - bibe.dot_q(pdk.sbks[j], ct.ct2, q)
                     - bibe.dot_q(vi, ct.ct3, q)) % q
            bit = bibe._round_bit(inner, q)
            message |= bit << j
            err = (inner - bit * (q // 2)) % q
            err = err if err <= q // 2 else err - q
            worst = max(worst, abs(err))
        msgs.append(message)
        errs.append(worst)
    return (msgs, errs) if return_error else msgs
