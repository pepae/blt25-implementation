"""Lattice BEAT-MEV via the BLMR key-homomorphic PRF (Appendix A.4).

The appendix sketches (no full algorithms are given) a plausibly post-quantum
variant of BEAT-MEV [26] built from:

  * the BLMR key-homomorphic PRF  F(sk, x) = round_{q->p}( sk . prod_i A_{x_i} )
    with A_0, A_1 uniform *binary* m x m matrices (row-vector convention, so
    GGM-tree nodes extend by right-multiplication);
  * GGM-style puncturing: the punctured key holds the sibling nodes
    u_{x_1..x_{j-1}, 1-x_j} along the path of x, each rounded to an
    intermediate modulus p' = O(m^k p) that absorbs the error amplification
    m^{k-j} of continuing the product from a rounded node;
  * an additively homomorphic encryption of the ephemeral PRF keys - here
    Regev encryption of base-2^8 chunks (exactly reassembled after
    homomorphic summation), the scheme the appendix names for preserving the
    key structure and enabling thresholdization [11].

Flow (batch of ell ciphertexts with pairwise distinct indices i in [2^k]):

  Enc(i, m):   k_i <- Z_q^m; gamma_i = encode(m) + F(k_i, i) over Z_p;
               publish (gamma_i, punctured key k*_i, Regev cts of k_i).
  PreDec(B):   homomorphically sum the Regev cts over j in B, decrypt:
               sbk = sum_j k_j  (mod q).   |sbk| = one PRF key.
  Dec:         m_i ~ gamma_i - F(sbk, i) + sum_{j != i} PuncEval(k*_j, i),
               by key homomorphism F(sum k_j, i) = sum_j F(k_j, i) + small.

This module follows the sketch's structure with toy parameters; the
threshold split of the Regev secret key (standard, [11]) is out of scope.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from .bits import BitStream, random_stream, stream_from_bytes
from .gaussian import sample_dgauss_vec
from .modq import matmul_q, next_prime


@dataclass(frozen=True)
class BMParams:
    k: int = 4                 # index bits (2^k indices)
    m: int = 24                # PRF dimension
    q: int = next_prime(1 << 42)
    p: int = 1 << 12           # PRF output modulus
    pp: int = 1 << 38          # p' for rounded inner nodes (>= 8 m^k p)
    # Regev transport
    n_reg: int = 32
    q_reg: int = next_prime(1 << 26)
    chunk_bits: int = 8
    reg_cols: int = 512
    sigma_reg: float = 4.0

    @property
    def n_chunks(self) -> int:
        return -(-self.q.bit_length() // self.chunk_bits)


def _round_to(v: np.ndarray, q_from: int, q_to: int) -> np.ndarray:
    """Coordinate-wise round_{q_from -> q_to}(v) = floor(q_to * v / q_from + 1/2)."""
    v = np.asarray(v, dtype=object) % q_from
    return np.array([int((q_to * int(x) * 2 + q_from) // (2 * q_from)) % q_to
                     for x in v], dtype=np.int64)


def _bits(i: int, k: int) -> tuple[int, ...]:
    return tuple((i >> j) & 1 for j in range(k))


class BLMR:
    """BLMR key-homomorphic PRF with binary chain matrices (row-vector form)."""

    def __init__(self, params: BMParams, seed: bytes = b"blmr-public"):
        self.p = params
        st = stream_from_bytes(seed)
        self.A = [np.array([[st.take_bits(1) for _ in range(params.m)]
                            for _ in range(params.m)], dtype=np.int64)
                  for _ in range(2)]

    def _chain(self, sk_row: np.ndarray, bits: tuple[int, ...]) -> np.ndarray:
        """sk . A_{x_1} . A_{x_2} ... A_{x_j}  (mod q)."""
        q = self.p.q
        u = np.asarray(sk_row, dtype=np.int64) % q
        for b in bits:
            u = matmul_q(u.reshape(1, -1), self.A[b], q, max_abs_b=1).reshape(-1)
        return u

    def eval(self, sk: np.ndarray, index: int) -> np.ndarray:
        """F(sk, index) in Z_p^m."""
        u = self._chain(sk, _bits(index, self.p.k))
        return _round_to(u, self.p.q, self.p.p)

    def puncture(self, sk: np.ndarray, index: int) -> dict:
        """Punctured key at `index`: sibling nodes along the GGM path, rounded
        to p'."""
        x = _bits(index, self.p.k)
        nodes = {}
        for j in range(self.p.k):
            sib = x[:j] + (1 - x[j],)
            nodes[j] = _round_to(self._chain(sk, sib), self.p.q, self.p.pp)
        return {"index": index, "nodes": nodes}

    def punc_eval(self, pkey: dict, index: int) -> np.ndarray:
        """Evaluate F(sk, index) from the punctured key; index != punctured."""
        pp, q, p, k = self.p.pp, self.p.q, self.p.p, self.p.k
        x_star = _bits(pkey["index"], k)
        x = _bits(index, k)
        assert x != x_star, "cannot evaluate at the punctured point"
        j = next(i for i in range(k) if x[i] != x_star[i])
        node = pkey["nodes"][j]                      # prefix x[:j] + (x_j,)
        u = _round_to(node, pp, q)                   # lift back to Z_q (approx)
        for b in x[j + 1:]:
            u = matmul_q(u.reshape(1, -1), self.A[b], q, max_abs_b=1).reshape(-1)
        return _round_to(u, q, p)


# ----------------------------------------------------------------------------
# Regev transport of the ephemeral keys (additively homomorphic, exact
# after chunk reassembly)
# ----------------------------------------------------------------------------

class RegevTransport:
    def __init__(self, params: BMParams, stream: BitStream | None = None):
        self.p = params
        st = stream or random_stream()
        n, M, q = params.n_reg, params.reg_cols, params.q_reg
        self.sk = st.uniform_mod_vec(n, q)
        self.A = st.uniform_mod_mat(n, M, q)
        e = sample_dgauss_vec(M, params.sigma_reg, st)
        self.b = (matmul_q(self.sk.reshape(1, -1), self.A, q).reshape(-1) + e) % q
        # message scale: chunk sums stay below 2^{chunk_bits} * ell_max = 2^{cb+4}
        self.msg_space = 1 << (params.chunk_bits + 4)
        self.scale = q // self.msg_space

    def enc_chunk(self, mu: int, stream: BitStream | None = None):
        st = stream or random_stream()
        q, M = self.p.q_reg, self.p.reg_cols
        r = np.array([st.take_bits(1) for _ in range(M)], dtype=np.int64)
        c1 = matmul_q(self.A, r.reshape(-1, 1), q).reshape(-1)
        c2 = (int(np.dot(self.b, r) % q) + mu * self.scale) % q
        return c1, c2

    def add(self, ct_a, ct_b):
        q = self.p.q_reg
        return (ct_a[0] + ct_b[0]) % q, (ct_a[1] + ct_b[1]) % q

    def dec_chunk(self, ct) -> int:
        q = self.p.q_reg
        c1, c2 = ct
        inner = (c2 - int(matmul_q(self.sk.reshape(1, -1), c1.reshape(-1, 1),
                                   q)[0, 0])) % q
        return int((inner + self.scale // 2) // self.scale) % self.msg_space

    # -- threshold decryption ([11] Bendlin-Damgard style) -------------------
    def share_secret(self, N: int, tau: int) -> list[np.ndarray]:
        """tau-of-N Shamir shares of the Regev secret key (party i gets
        share evaluated at x = i)."""
        import secrets
        q, n = self.p.q_reg, self.p.n_reg
        coeffs = [self.sk % q]
        rng = random_stream()
        for _ in range(tau - 1):
            coeffs.append(rng.uniform_mod_vec(n, q))
        shares = []
        for i in range(1, N + 1):
            acc = np.zeros(n, dtype=np.int64)
            xpow = 1
            for cvec in coeffs:
                acc = (acc + cvec * xpow) % q
                xpow = (xpow * i) % q
            shares.append(acc)
        return shares

    def partial_dec_chunk(self, share: np.ndarray, party: int,
                          act: tuple[int, ...], ct,
                          flood_width: float = 256.0) -> int:
        """Party's partial decryption: lambda_i <s_i, c1> + e_flood.  The
        Lagrange coefficient multiplies *before* flooding so the combined
        error is only the sum of tau small flooding terms."""
        from .tgs import lagrange_at_zero
        q = self.p.q_reg
        c1, _ = ct
        lam = lagrange_at_zero(act, q)[party]
        inner = int(matmul_q(share.reshape(1, -1), c1.reshape(-1, 1), q)[0, 0])
        e = int(sample_dgauss_vec(1, flood_width, random_stream())[0])
        return (lam * inner + e) % q

    def combine_partials_chunk(self, ct, partials: dict[int, int]) -> int:
        q = self.p.q_reg
        _, c2 = ct
        inner = (int(c2) - sum(int(v) for v in partials.values())) % q
        return int((inner + self.scale // 2) // self.scale) % self.msg_space


@dataclass
class BMCiphertext:
    index: int
    gamma: np.ndarray            # Z_p^m masked message
    punc: dict                   # punctured PRF key
    key_cts: list                # Regev cts of the chunks of k_i


class BeatMev:
    """The batch encryption scheme of Appendix A.4 (single-party committee;
    thresholdizing the Regev key is standard [11] and out of scope)."""

    def __init__(self, params: BMParams | None = None, seed: bytes = b"bm"):
        self.p = params or BMParams()
        self.prf = BLMR(self.p, seed=seed + b"/prf")
        self.transport = RegevTransport(self.p, stream_from_bytes(seed + b"/reg"))

    def encode(self, msg_bits: list[int]) -> np.ndarray:
        assert len(msg_bits) <= self.p.m
        v = np.zeros(self.p.m, dtype=np.int64)
        v[:len(msg_bits)] = np.array(msg_bits, dtype=np.int64) * (self.p.p // 2)
        return v

    def decode(self, v: np.ndarray, nbits: int) -> list[int]:
        p = self.p.p
        out = []
        for x in v[:nbits]:
            x = int(x) % p
            out.append(1 if abs(x - p // 2) < p // 4 else 0)
        return out

    def encrypt(self, index: int, msg_bits: list[int]) -> BMCiphertext:
        q = self.p.q
        k_i = np.array([secrets.randbelow(q) for _ in range(self.p.m)],
                       dtype=object).astype(object)
        k_i = np.array([int(x) for x in k_i], dtype=object)
        gamma = (self.encode(msg_bits) + self.prf.eval(k_i, index)) % self.p.p
        punc = self.prf.puncture(k_i, index)
        cb, nc = self.p.chunk_bits, self.p.n_chunks
        key_cts = []
        for coord in k_i:
            chunks = [(int(coord) >> (cb * c)) & ((1 << cb) - 1)
                      for c in range(nc)]
            key_cts.append([self.transport.enc_chunk(mu) for mu in chunks])
        return BMCiphertext(index=index, gamma=gamma, punc=punc,
                            key_cts=key_cts)

    def pre_dec(self, cts: list[BMCiphertext]) -> np.ndarray:
        """sbk = sum_j k_j (mod q), recovered from the homomorphic sums."""
        assert len({ct.index for ct in cts}) == len(cts), "indices must differ"
        m, nc, cb, q = self.p.m, self.p.n_chunks, self.p.chunk_bits, self.p.q
        sbk = np.zeros(m, dtype=object)
        for coord in range(m):
            for c in range(nc):
                acc = cts[0].key_cts[coord][c]
                for ct in cts[1:]:
                    acc = self.transport.add(acc, ct.key_cts[coord][c])
                s = self.transport.dec_chunk(acc)
                sbk[coord] = (int(sbk[coord]) + (s << (cb * c))) % q
        return sbk

    def pre_dec_threshold(self, cts: list[BMCiphertext],
                          shares: list[np.ndarray],
                          act: tuple[int, ...]) -> np.ndarray:
        """Threshold variant of pre_dec: the committee's Regev key is
        tau-of-N Shamir-shared (self.transport.share_secret) and each active
        party contributes flooded partial decryptions per chunk - the
        thresholdization route Appendix A.4 points to via [11]."""
        assert len({ct.index for ct in cts}) == len(cts), "indices must differ"
        m, nc, cb, q = self.p.m, self.p.n_chunks, self.p.chunk_bits, self.p.q
        sbk = np.zeros(m, dtype=object)
        for coord in range(m):
            for c in range(nc):
                acc = cts[0].key_cts[coord][c]
                for ct in cts[1:]:
                    acc = self.transport.add(acc, ct.key_cts[coord][c])
                partials = {i: self.transport.partial_dec_chunk(
                    shares[i - 1], i, act, acc) for i in act}
                s = self.transport.combine_partials_chunk(acc, partials)
                sbk[coord] = (int(sbk[coord]) + (s << (cb * c))) % q
        return sbk

    def decrypt(self, sbk: np.ndarray, cts: list[BMCiphertext],
                nbits: int) -> list[list[int]]:
        p = self.p.p
        out = []
        for i, ct in enumerate(cts):
            acc = (np.asarray(ct.gamma, dtype=np.int64)
                   - self.prf.eval(sbk, ct.index)) % p
            for j, other in enumerate(cts):
                if j == i:
                    continue
                acc = (acc + self.prf.punc_eval(other.punc, ct.index)) % p
            out.append(self.decode(acc, nbits))
        return out
