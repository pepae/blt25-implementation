"""The gadget matrix G_n, bit decomposition G^{-1}, and a basis of Lambda^perp(G).

Following Section 3.4.1 / A.2.2 of the paper:

    G_n := I_n (x) g^T  in Z_q^{n x m~},   g^T := [1, 2, ..., 2^{floor(log q)}],
    m~  := n * (floor(log q) + 1).

``g_inverse`` is the deterministic binary decomposition satisfying
G * G^{-1}(X) = X (mod q) with entries in {0, 1}.

``g_perp_basis`` returns the standard basis S of Lambda^perp(g^T) from
[MP12, Theorem 4.1] (cited as Lemma 21): the (k+1) x (k+1) matrix with
columns (2, -1, 0, ...), (0, 2, -1, ...), ..., and last column the binary
digits of q; block-diagonally repeated n times it is a basis of
Lambda^perp(G_n) with ||S|| <= max(sqrt(5), sqrt(k+1)).
"""

from __future__ import annotations

import numpy as np


def gadget_len(q: int) -> int:
    """k + 1 where k = floor(log2 q); the length of g."""
    return (q.bit_length() - 1) + 1


def gadget_dims(n: int, q: int) -> int:
    """m~ = n * (floor(log q) + 1)."""
    return n * gadget_len(q)


def gadget_vector(q: int) -> np.ndarray:
    k1 = gadget_len(q)
    return np.array([1 << i for i in range(k1)], dtype=np.int64)


def gadget_matrix(n: int, q: int) -> np.ndarray:
    """G_n in Z_q^{n x m~} (entries are the powers of two, already < q... except
    2^k may exceed q? No: 2^k <= q-1 < q since k = floor(log2 q))."""
    g = gadget_vector(q) % q
    return np.kron(np.eye(n, dtype=np.int64), g.reshape(1, -1))


def g_inverse(X: np.ndarray, q: int) -> np.ndarray:
    """Binary decomposition: G_n * g_inverse(X) = X (mod q), entries in {0,1}.

    X is (n x c) canonical mod-q; the result is (m~ x c).
    Also accepts vectors (n,) returning (m~,).
    """
    vec = X.ndim == 1
    if vec:
        X = X.reshape(-1, 1)
    n, c = X.shape
    k1 = gadget_len(q)
    Xc = np.asarray(X, dtype=np.int64) % q
    out = np.zeros((n * k1, c), dtype=np.int64)
    for b in range(k1):
        out[b::k1, :] = (Xc >> b) & 1
    return out.reshape(-1) if vec else out


def g_perp_basis(n: int, q: int) -> np.ndarray:
    """Basis S of Lambda^perp(G_n): block diag of n copies of S_k.

    S_k columns: for j < k: 2*e_j - e_{j+1}; last column: binary digits of q.
    g^T S_k = 0 mod q by construction (exact 0 for the first k columns, and
    q = 0 mod q for the last).
    """
    k1 = gadget_len(q)
    S = np.zeros((k1, k1), dtype=np.int64)
    for j in range(k1 - 1):
        S[j, j] = 2
        S[j + 1, j] = -1
    for i in range(k1):
        S[i, k1 - 1] = (q >> i) & 1
    return np.kron(np.eye(n, dtype=np.int64), S)
