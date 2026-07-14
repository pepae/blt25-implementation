"""Batch IBE from a trilinear map (Section 9), over a *simulated* generic
trilinear group.

No cryptographic trilinear map is known; the paper proves security only in
Shoup's generic group model (Theorem 9).  This module therefore implements the
scheme's algorithms against a GGM oracle simulation (`TrilinearGroup`): group
elements are opaque random labels, the oracle knows the discrete logs, and the
scheme code touches elements only through labeling / group-op / map / hash
queries - exactly the interface of Figure 1.  It demonstrates and tests the
algebra (including the optimal parameter sizes: sbk is a single G3 element and
ct adds one G1 element) but provides NO security outside the simulation.

Also included: the threshold pre-decryption of Section 9.4 via partial
fractions,

    1/prod(alpha + r_i) = sum_i beta_i / (alpha + r_i),
    beta_i = prod_{j != i} 1/(r_j - r_i),

with the per-identity Boneh-Boyen-style two-round evaluation: an ephemeral s
is additively shared, beta = s (alpha + r_i) is published in the clear (the
joint computation of beta is the [Wang et al.] sub-protocol, which we model as
an ideal functionality over the Shamir shares), and each party publishes
P_k = (s_k / beta) G3 so that sum_k P_k = (1/(alpha + r_i)) G3.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from .modq import is_prime


class TrilinearGroup:
    """Generic-group simulation of G1 x G2 x G3 -> GT, all of prime order p.

    Labels are 16-byte opaque strings.  The simulation tracks discrete logs
    internally (a real adversary never sees them; scheme code below only uses
    the oracle interface)."""

    GROUPS = (1, 2, 3, "T")

    def __init__(self, p: int, seed: bytes | None = None):
        assert is_prime(p)
        self.p = p
        self._fwd = {g: {} for g in self.GROUPS}   # exponent -> label
        self._rev = {g: {} for g in self.GROUPS}   # label -> exponent
        self._ctr = 0
        self._seed = seed or secrets.token_bytes(16)

    def _fresh_label(self, g) -> bytes:
        self._ctr += 1
        return hashlib.shake_256(
            b"tri-label" + self._seed + str(g).encode()
            + self._ctr.to_bytes(8, "big")).digest(16)

    def label(self, g, x: int) -> bytes:
        """Labeling query: L_g(x)."""
        x %= self.p
        lab = self._fwd[g].get(x)
        if lab is None:
            lab = self._fresh_label(g)
            self._fwd[g][x] = lab
            self._rev[g][lab] = x
        return lab

    def op(self, g, l1: bytes, l2: bytes, a1: int, a2: int) -> bytes | None:
        """Group operation: L_g(a1 x1 + a2 x2)."""
        x1 = self._rev[g].get(l1)
        x2 = self._rev[g].get(l2)
        if x1 is None or x2 is None:
            return None
        return self.label(g, (a1 * x1 + a2 * x2) % self.p)

    def scalar(self, g, l1: bytes, a: int) -> bytes | None:
        """Convenience: L_g(a x1)."""
        one = self.label(g, 0)
        return self.op(g, l1, one, a, 0)

    def map3(self, l1: bytes, l2: bytes, l3: bytes) -> bytes | None:
        """Trilinear map query: LT(x1 x2 x3)."""
        x1 = self._rev[1].get(l1)
        x2 = self._rev[2].get(l2)
        x3 = self._rev[3].get(l3)
        if None in (x1, x2, x3):
            return None
        return self.label("T", (x1 * x2 * x3) % self.p)

    def hash_to_bytes(self, lt_label: bytes, nbytes: int) -> bytes:
        """The one-way function H : GT -> {0,1}^{|M|} (random oracle)."""
        if lt_label not in self._rev["T"]:
            raise ValueError("hash query on a non-element")
        return hashlib.shake_256(b"tri-H" + lt_label).digest(nbytes)


@dataclass
class TriPublicKey:
    grp: TrilinearGroup
    ell: int
    A1: bytes                  # alpha G1
    Bs: list[bytes]            # [G2, alpha G2, ..., alpha^{ell-1} G2]
    G1: bytes
    G3: bytes


@dataclass
class TriCiphertext:
    r: int
    ct1: bytes                 # t (alpha + r) G1
    ct2: bytes                 # m XOR H(t GT)


def setup(grp: TrilinearGroup, ell: int):
    """BIBE.Setup: sk = alpha; pk = (alpha G1, (alpha^i G2)_{i<ell})."""
    p = grp.p
    alpha = secrets.randbelow(p - 1) + 1
    A1 = grp.label(1, alpha)
    Bs = [grp.label(2, pow(alpha, i, p)) for i in range(ell)]
    pk = TriPublicKey(grp=grp, ell=ell, A1=A1, Bs=Bs,
                      G1=grp.label(1, 1), G3=grp.label(3, 1))
    return pk, alpha


def encrypt(pk: TriPublicKey, r: int, message: bytes) -> TriCiphertext:
    """BIBE.Enc: ct1 = t A1 + r t G1; ct2 = m XOR H(t GT)."""
    grp = pk.grp
    p = grp.p
    t = secrets.randbelow(p - 1) + 1
    ct1 = grp.op(1, pk.A1, pk.G1, t, (r * t) % p)
    mask = grp.hash_to_bytes(grp.label("T", t), len(message))
    ct2 = bytes(a ^ b for a, b in zip(message, mask))
    return TriCiphertext(r=r % p, ct1=ct1, ct2=ct2)


def pre_dec(pk: TriPublicKey, alpha: int, identities: list[int]) -> bytes:
    """BIBE.PreDec: sbk = (1 / prod_i (alpha + r_i)) G3."""
    grp = pk.grp
    p = grp.p
    prod = 1
    for r in identities:
        term = (alpha + r) % p
        if term == 0:
            raise ValueError("alpha + r = 0 (degenerate identity)")
        prod = (prod * term) % p
    return grp.label(3, pow(prod, -1, p))


def _fi_coeffs(identities: list[int], i: int, p: int) -> list[int]:
    """Coefficients of f_i(X) = prod_{j != i} (X + r_j), degree ell-1,
    little-endian (constant first)."""
    coeffs = [1]
    for j, r in enumerate(identities):
        if j == i:
            continue
        new = [0] * (len(coeffs) + 1)
        for k, c in enumerate(coeffs):
            new[k] = (new[k] + c * r) % p        # * (X + r): constant part
            new[k + 1] = (new[k + 1] + c) % p    # X part
        coeffs = new
    return coeffs


def decrypt(pk: TriPublicKey, sbk: bytes,
            cts: list[TriCiphertext]) -> list[bytes]:
    """BIBE.Dec: m_i = ct2_i XOR H(ct1_i . f_i(alpha) G2 . sbk)."""
    grp = pk.grp
    p = grp.p
    ids = [ct.r for ct in cts]
    out = []
    for i, ct in enumerate(cts):
        coeffs = _fi_coeffs(ids, i, p)           # degree ell-1 -> uses B_0..B_{ell-1}
        acc = None
        for k, c in enumerate(coeffs):
            if c == 0:
                continue
            term = pk.Bs[k]
            acc = grp.scalar(2, term, c) if acc is None else \
                grp.op(2, acc, term, 1, c)
        lt = grp.map3(ct.ct1, acc, sbk)
        mask = grp.hash_to_bytes(lt, len(ct.ct2))
        out.append(bytes(a ^ b for a, b in zip(ct.ct2, mask)))
    return out


# ----------------------------------------------------------------------------
# Section 9.4: thresholdizing the pre-decryption key via partial fractions
# ----------------------------------------------------------------------------

def shamir_share_alpha(alpha: int, N: int, tau: int, p: int) -> dict[int, int]:
    coeffs = [alpha] + [secrets.randbelow(p) for _ in range(tau - 1)]
    shares = {}
    for i in range(1, N + 1):
        acc, xp = 0, 1
        for c in coeffs:
            acc = (acc + c * xp) % p
            xp = (xp * i) % p
        shares[i] = acc
    return shares


def threshold_pre_dec(pk: TriPublicKey, shares: dict[int, int],
                      act: tuple[int, ...], identities: list[int]) -> bytes:
    """Two-round threshold pre-decryption (Section 9.4).

    For each identity r_i the quorum computes (1/(alpha+r_i)) G3 as in the
    Boneh-Boyen threshold protocol: additively share an ephemeral s, publish
    beta = s (alpha + r_i) in the clear (the beta computation from the Shamir
    shares is the two-round [Wang et al.] sub-protocol; modeled here as an
    ideal step over the shares), publish P_k = (s_k / beta) G3, and sum.  The
    partial-fraction coefficients beta_i then combine the per-identity
    elements into sbk = sum_i beta_i (1/(alpha+r_i)) G3.
    """
    grp = pk.grp
    p = grp.p
    lam = {}
    for i in act:
        num, den = 1, 1
        for j in act:
            if j != i:
                num = (num * j) % p
                den = (den * (j - i)) % p
        lam[i] = (num * pow(den, -1, p)) % p
    alpha = sum(lam[i] * shares[i] for i in act) % p   # ideal-step recombination

    per_identity: list[bytes] = []
    for r in identities:
        s_shares = {k: secrets.randbelow(p - 1) + 1 for k in act}
        s = sum(s_shares.values()) % p
        beta = (s * (alpha + r)) % p                   # published in the clear
        if beta == 0:
            raise ValueError("degenerate ephemeral; retry")
        parts = [grp.label(3, (sk * pow(beta, -1, p)) % p)
                 for sk in s_shares.values()]
        acc = parts[0]
        for lab in parts[1:]:
            acc = grp.op(3, acc, lab, 1, 1)
        per_identity.append(acc)                       # (1/(alpha+r)) G3

    # partial fractions: 1/prod(alpha+r_i) = sum_i beta_i/(alpha+r_i)
    sbk = None
    for i, r_i in enumerate(identities):
        beta_i = 1
        for j, r_j in enumerate(identities):
            if j != i:
                beta_i = (beta_i * pow((r_j - r_i) % p, -1, p)) % p
        term = grp.scalar(3, per_identity[i], beta_i)
        sbk = term if sbk is None else grp.op(3, sbk, term, 1, 1)
    return sbk
