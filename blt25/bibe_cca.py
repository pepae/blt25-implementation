"""CCA2 / anti-replication wrapper for the batch IBE scheme (Section 1.2).

The paper's deployment blueprint: an encryptor "can generate a key pair for a
signature scheme, use the public key as the identity tag, and post the
encrypted transaction along with a signature on the encrypted transaction.
The nodes will only include a transaction in a block if it has a valid
signature with respect to its identity tag" - defeating the replicated-tag
attack of Section 1.2, and (via the standard CHK/BCHK transforms the paper
cites [17, 15, 16]) lifting CPA to CCA-style non-malleability: any mauling of
the ciphertext invalidates the one-time signature, and forging under a fresh
tag places the adversary outside the challenge identity.

Instantiation: Lamport one-time signatures over SHAKE-256 (hash-based, so the
wrapper stays plausibly post-quantum like the scheme itself).  The identity
tag is the first d bits of SHAKE-256(vk); PreDec/Dec reject batches whose
tags collide (the paper's abort rule), and `validate` enforces both the
signature and the tag binding.

The formal CCA2 claim for the batch setting follows the paper's cited
transforms; this module supplies the mechanics (see AUDIT.md).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from . import bibe
from .hashes import _ser


def _H(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big") + p)
    return h.digest()


# ----------------------------------------------------------------------------
# Lamport one-time signatures (256-bit messages, SHA3-256)
# ----------------------------------------------------------------------------

def ots_keygen() -> tuple[list[list[bytes]], list[list[bytes]]]:
    sk = [[secrets.token_bytes(32) for _ in range(256)] for _ in range(2)]
    vk = [[_H(b"ots-leaf", s) for s in row] for row in sk]
    return sk, vk


def ots_vk_bytes(vk) -> bytes:
    return _H(b"ots-vk", b"".join(vk[0]) + b"".join(vk[1]))


def ots_sign(sk, message: bytes) -> list[bytes]:
    digest = _H(b"ots-msg", message)
    sig = []
    for i in range(256):
        bit = (digest[i // 8] >> (i % 8)) & 1
        sig.append(sk[bit][i])
    return sig


def ots_verify(vk, message: bytes, sig: list[bytes]) -> bool:
    if len(sig) != 256:
        return False
    digest = _H(b"ots-msg", message)
    for i in range(256):
        bit = (digest[i // 8] >> (i % 8)) & 1
        if _H(b"ots-leaf", sig[i]) != vk[bit][i]:
            return False
    return True


# ----------------------------------------------------------------------------
# The wrapper
# ----------------------------------------------------------------------------

@dataclass
class CCACiphertext:
    vk_digest: bytes             # the identity tag source
    vk: list                     # full OTS verification key
    ct: bibe.Ciphertext
    sig: list[bytes]


def tag_of_vk(vk_digest: bytes, d: int) -> int:
    """Identity tag = first d bits of the vk digest (as an int < 2^d)."""
    v = int.from_bytes(vk_digest, "big")
    return v & ((1 << d) - 1)


def _ct_bytes(ct: bibe.Ciphertext) -> bytes:
    from .hashes import _encode_identity
    return _ser(_encode_identity(ct.r), ct.ct1, ct.ct2, ct.ct3)


def encrypt(pk: bibe.PublicKey, message: int) -> CCACiphertext:
    """EncCCA: fresh OTS keypair; tag := H(vk); sign the whole ciphertext."""
    sk, vk = ots_keygen()
    vkd = ots_vk_bytes(vk)
    r = tag_of_vk(vkd, pk.params.d)
    ct = bibe.encrypt(pk, r, message)
    sig = ots_sign(sk, _ct_bytes(ct))
    return CCACiphertext(vk_digest=vkd, vk=vk, ct=ct, sig=sig)


def validate(pk: bibe.PublicKey, cct: CCACiphertext) -> bool:
    """The mempool/node-side check: tag really binds to vk, and the OTS
    signature covers the exact ciphertext.  Reject before inclusion."""
    if ots_vk_bytes(cct.vk) != cct.vk_digest:
        return False
    if cct.ct.r != tag_of_vk(cct.vk_digest, pk.params.d):
        return False
    return ots_verify(cct.vk, _ct_bytes(cct.ct), cct.sig)


def pre_dec(sk: bibe.SecretKey, ccts: list[CCACiphertext]) -> bibe.PreDecKey:
    """PreDec over validated wrapped ciphertexts (rejects invalid ones and
    colliding tags, per the Section 1.2 inclusion rule)."""
    pk = sk.pk
    for cct in ccts:
        if not validate(pk, cct):
            raise ValueError("invalid OTS-wrapped ciphertext in batch")
    return bibe.pre_dec(sk, tuple(c.ct.r for c in ccts))


def decrypt(pk: bibe.PublicKey, sbk: bibe.PreDecKey,
            ccts: list[CCACiphertext]) -> list[int]:
    for cct in ccts:
        if not validate(pk, cct):
            raise ValueError("invalid OTS-wrapped ciphertext in batch")
    return bibe.decrypt(pk, sbk, [c.ct for c in ccts])
