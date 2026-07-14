"""The scheme's hash functions (Definitions 13-17) and the PRF, instantiated
with SHAKE-256 with domain separation.

- H      : I -> {0,1}^d          (Definition 13; injective index map)
- Hro    : I -> Z_q^{n x m}      (Definition 14; random oracle)
- Hsp    : (A, T_A, (r_i), u, sigma) -> {0,1}^rho          (Definition 15)
- Hsp_th : ... x sidsp -> {0,1}^rho                        (Definition 17)
- Htr    : (sid, act, (e_i)) -> {0,1}^lambda_tr            (Definition 16)
- HGPV   : (msg, sidsp) -> Z_q^n                           (Section 7)
- PRF    : {0,1}^kappa x Z -> Z_q^m                        (Section 6.1)

Random-oracle outputs over Z_q use 64 surplus bits per element, so the mod-q
bias is < 2^-64.  All serializations are canonical (length-prefixed).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .bits import BitStream

LAMBDA_TR_BYTES = 32  # lambda_tr = 256 bits


def _ser(*items) -> bytes:
    """Canonical length-prefixed serialization of ints, bytes, str, arrays."""
    out = bytearray()
    for it in items:
        if isinstance(it, (int, np.integer)):
            b = int(it).to_bytes(17, "big", signed=True)
            out += b"i" + b
        elif isinstance(it, bytes):
            out += b"b" + len(it).to_bytes(8, "big") + it
        elif isinstance(it, str):
            eb = it.encode()
            out += b"s" + len(eb).to_bytes(8, "big") + eb
        elif isinstance(it, np.ndarray):
            arr = np.ascontiguousarray(it, dtype=np.int64)
            shape = ",".join(map(str, arr.shape)).encode()
            out += (b"a" + len(shape).to_bytes(4, "big") + shape
                    + len(arr.tobytes()).to_bytes(8, "big") + arr.tobytes())
        elif isinstance(it, (tuple, list)):
            out += b"l" + len(it).to_bytes(8, "big")
            out += _ser(*it)
        elif it is None:
            out += b"n"
        else:
            raise TypeError(f"cannot serialize {type(it)}")
    return bytes(out)


def _xof(domain: bytes, payload: bytes, nbytes: int) -> bytes:
    h = hashlib.shake_256()
    h.update(b"blt25/" + domain + b"/")
    h.update(payload)
    return h.digest(nbytes)


def _xof_stream(domain: bytes, payload: bytes) -> BitStream:
    return BitStream(_xof(domain, payload, 32))


def _encode_identity(r) -> bytes:
    if isinstance(r, (int, np.integer)):
        return b"int" + int(r).to_bytes(32, "big", signed=False)
    if isinstance(r, bytes):
        return b"byt" + r
    if isinstance(r, str):
        return b"str" + r.encode()
    raise TypeError(f"unsupported identity type {type(r)}")


def hash_index(r, d: int) -> tuple[int, ...]:
    """H : I -> {0,1}^d.  For integer identities r < 2^d this is the (injective)
    binary expansion; otherwise the first d bits of SHAKE-256(r) (injectivity
    then holds only computationally / with the birthday bound; PreDec rejects
    colliding indices per the specification)."""
    if isinstance(r, (int, np.integer)) and 0 <= int(r) < (1 << d):
        v = int(r)
        return tuple((v >> i) & 1 for i in range(d))
    digest = _xof(b"H-index", _encode_identity(r), (d + 7) // 8)
    bits = []
    for i in range(d):
        bits.append((digest[i // 8] >> (i % 8)) & 1)
    return tuple(bits)


def hash_ro(r, n: int, m: int, q: int) -> np.ndarray:
    """Hro : I -> Z_q^{n x m} (random oracle)."""
    stream = _xof_stream(b"Hro", _encode_identity(r))
    return stream.uniform_mod_mat(n, m, q)


def _hsp_payload(A: np.ndarray, T: np.ndarray, identities, u: np.ndarray,
                 sigma) -> bytes:
    ids = [_encode_identity(r) for r in identities]
    return _ser(A, T, ids, u, int(sigma))


def hash_sp(A: np.ndarray, T: np.ndarray, identities, u: np.ndarray,
            sigma, rho_bits: int) -> bytes:
    """Hsp (Definition 15): the SampleLeft random tape, rho_bits long."""
    payload = _hsp_payload(A, T, identities, u, sigma)
    return _xof(b"Hsp", payload, (rho_bits + 7) // 8)


def hash_sp_th(A: np.ndarray, T: np.ndarray, identities, u: np.ndarray,
               sigma, sidsp: bytes, rho_bits: int) -> bytes:
    """Hsp^th (Definition 17): domain-separated by the transcript tag sidsp."""
    payload = _hsp_payload(A, T, identities, u, sigma) + _ser(sidsp)
    return _xof(b"Hsp-th", payload, (rho_bits + 7) // 8)


def hash_tr(sid: int, act: tuple[int, ...], syndromes: list[np.ndarray]) -> bytes:
    """Htr (Definition 16): transcript tag binding sid, act and the round-1
    syndromes."""
    payload = _ser(int(sid), tuple(int(a) for a in act), list(syndromes))
    return _xof(b"Htr", payload, LAMBDA_TR_BYTES)


def hash_gpv(message: bytes, sidsp: bytes, n: int, q: int) -> np.ndarray:
    """HGPV : M x {0,1}^lambda_tr -> Z_q^n (Section 7)."""
    stream = _xof_stream(b"HGPV", _ser(message, sidsp))
    return stream.uniform_mod_vec(n, q)


def prf(seed: bytes, sid: int, m: int, q: int) -> np.ndarray:
    """PRF : {0,1}^kappa x Z -> Z_q^m (Section 6.1), SHAKE-256 instantiation."""
    stream = _xof_stream(b"PRF", _ser(seed, int(sid)))
    return stream.uniform_mod_vec(m, q)
