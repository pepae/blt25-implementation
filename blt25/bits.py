"""Deterministic bit streams used as random tapes.

The paper treats randomized algorithms as deterministic functions of their
input plus a random tape rb (e.g. ``SampleLeft(inp; rb)``).  We realize tapes
as SHAKE-256 output streams: a tape is fully determined by its seed, so two
parties seeding a tape with the same bytes (e.g. the output of the random
oracle Hsp) run the sampler identically.

``BitStream`` also records how many bits were consumed, which lets us report
the effective randomness length rho.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


class BitStream:
    """Deterministic bit stream backed by SHAKE-256 on a seed."""

    _BLOCK = 1 << 16  # bytes squeezed per refill

    def __init__(self, seed: bytes):
        self._seed = bytes(seed)
        self._buf = b""
        self._bitpos = 0        # consumed bits within _buf
        self._blocks = 0        # refill counter
        self.bits_consumed = 0

    def _refill(self) -> None:
        h = hashlib.shake_256()
        h.update(b"blt25/bitstream")
        h.update(len(self._seed).to_bytes(8, "little"))
        h.update(self._seed)
        h.update(self._blocks.to_bytes(8, "little"))
        self._buf = h.digest(self._BLOCK)
        self._blocks += 1
        self._bitpos = 0

    def take_bits(self, k: int) -> int:
        """Next k bits as an integer in [0, 2**k) (big-endian within the stream)."""
        out = 0
        need = k
        while need > 0:
            if self._bitpos >= 8 * len(self._buf):
                self._refill()
            avail = 8 * len(self._buf) - self._bitpos
            grab = min(need, avail)
            byte0, bit0 = divmod(self._bitpos, 8)
            byte1 = (self._bitpos + grab + 7) // 8
            chunk = int.from_bytes(self._buf[byte0:byte1], "big")
            width = 8 * (byte1 - byte0)
            chunk >>= width - (bit0 + grab)
            chunk &= (1 << grab) - 1
            out = (out << grab) | chunk
            self._bitpos += grab
            need -= grab
        self.bits_consumed += k
        return out

    def take_uniform_mod(self, q: int) -> int:
        """Uniform value in [0, q) with negligible bias (64 surplus bits)."""
        width = q.bit_length() + 64
        return self.take_bits(width) % q

    def uniform_mod_vec(self, n: int, q: int) -> np.ndarray:
        return np.array([self.take_uniform_mod(q) for _ in range(n)], dtype=np.int64)

    def uniform_mod_mat(self, rows: int, cols: int, q: int) -> np.ndarray:
        return self.uniform_mod_vec(rows * cols, q).reshape(rows, cols)

    def pm1_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Uniform matrix over {-1, +1}."""
        flat = np.empty(rows * cols, dtype=np.int64)
        # squeeze bits in bulk: one bit per entry
        for i in range(rows * cols):
            flat[i] = 1 if self.take_bits(1) else -1
        return flat.reshape(rows, cols)


def random_stream() -> BitStream:
    """Fresh stream seeded from the OS CSPRNG (non-deterministic use sites)."""
    return BitStream(os.urandom(32))


def stream_from_bytes(seed: bytes) -> BitStream:
    return BitStream(seed)
