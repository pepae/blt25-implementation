"""Z_q matrix and vector arithmetic.

All Z_q values are stored as NumPy ``int64`` arrays reduced to the canonical
range [0, q).  Products of two canonical values fit in an int64 as long as
q < 2**31; matrix products additionally accumulate over the inner dimension,
so ``matmul_q`` splits the inner dimension into chunks small enough that the
partial sums cannot overflow, reducing mod q between chunks.

Integer (non mod-q) matrices such as trapdoors and Gaussian preimages are held
as plain int64 arrays (their entries are small by construction); products of
those with mod-q data go through the same overflow-safe path.
"""

from __future__ import annotations

import numpy as np

INT64_MAX = np.iinfo(np.int64).max  # 2**63 - 1


def _chunk_size(max_abs_a: int, max_abs_b: int) -> int:
    """Largest inner-dimension chunk such that chunk * max_a * max_b < 2**62."""
    prod = max(1, max_abs_a) * max(1, max_abs_b)
    return max(1, (1 << 62) // prod)


def matmul_q(a: np.ndarray, b: np.ndarray, q: int,
             max_abs_a: int | None = None, max_abs_b: int | None = None) -> np.ndarray:
    """(a @ b) mod q with int64 overflow protection.

    ``max_abs_a`` / ``max_abs_b`` are bounds on |entries|; default assumes
    canonical mod-q values in [0, q).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if max_abs_a is None:
        max_abs_a = q - 1
    if max_abs_b is None:
        max_abs_b = q - 1
    if max(1, max_abs_a) * max(1, max_abs_b) >= (1 << 62):
        raise ValueError("entries too large for int64 backend; use object arithmetic")
    k = a.shape[-1]
    step = _chunk_size(max_abs_a, max_abs_b)
    if step >= k:
        return (a @ b) % q
    acc = None
    for s in range(0, k, step):
        part = (a[..., s:s + step] @ b[s:s + step, ...]) % q
        acc = part if acc is None else (acc + part) % q
    return acc


def matvec_q(a: np.ndarray, v: np.ndarray, q: int, **kw) -> np.ndarray:
    return matmul_q(a, v.reshape(-1, 1), q, **kw).reshape(-1)


def center_lift(x: np.ndarray | int, q: int):
    """Map canonical [0, q) representatives to centered ones in (-q/2, q/2]."""
    if isinstance(x, (int, np.integer)):
        x = int(x) % q
        return x - q if x > q // 2 else x
    x = np.asarray(x, dtype=np.int64) % q
    return np.where(x > q // 2, x - q, x)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    n = max(2, n + 1)
    if n % 2 == 0:
        n += 1
    while not is_prime(n):
        n += 2
    return n


def inv_mod(a: int, q: int) -> int:
    return pow(int(a) % q, -1, q)


def solve_mod_q(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    """Deterministically solve A X = B (mod q) for prime q.

    A is (n x m) with m >= n and rank n (mod q); B is (n x k).  Returns a
    particular solution X (m x k) with free variables set to 0, computed by
    Gaussian elimination with the fixed pivot rule "first row with a nonzero
    entry in the leftmost unused column".  Deterministic so that every party
    derives the same solution.  Raises ValueError if the system is inconsistent.
    """
    A = np.asarray(A, dtype=np.int64) % q
    B = np.asarray(B, dtype=np.int64) % q
    single = B.ndim == 1
    if single:
        B = B.reshape(-1, 1)
    n, m = A.shape
    M = np.concatenate([A, B], axis=1).astype(np.int64)
    piv_cols: list[int] = []
    row = 0
    for col in range(m):
        if row >= n:
            break
        # find pivot in this column at or below `row`
        nz = np.nonzero(M[row:, col])[0]
        if nz.size == 0:
            continue
        p = row + int(nz[0])
        if p != row:
            M[[row, p]] = M[[p, row]]
        inv = inv_mod(int(M[row, col]), q)
        M[row] = (M[row] * inv) % q
        # eliminate this column from all other rows (overflow-safe: entries < q < 2**31)
        others = np.nonzero(M[:, col])[0]
        others = others[others != row]
        if others.size:
            M[others] = (M[others] - np.outer(M[others, col], M[row])) % q
        piv_cols.append(col)
        row += 1
    # consistency: rows beyond `row` must have zero RHS
    if row < n and np.any(M[row:, m:] % q):
        raise ValueError("inconsistent linear system mod q")
    X = np.zeros((m, B.shape[1]), dtype=np.int64)
    for r, c in enumerate(piv_cols):
        X[c] = M[r, m:]
    return X.reshape(-1) if single else X


def rank_profile_mod_p(W: np.ndarray, p: int, target_rank: int) -> list[int]:
    """Indices of the first `target_rank` linearly independent columns of W mod p.

    Uses fraction-free row-echelon Gaussian elimination over F_p with the fixed
    "leftmost nonzero column" pivot rule, so the selection is deterministic and
    equals the leftmost rank profile mod p.  p must be prime and < 2**31.
    """
    W = np.asarray(W, dtype=np.int64) % p
    n, m = W.shape
    cols: list[int] = []
    row = 0
    for col in range(m):
        if row >= n or len(cols) >= target_rank:
            break
        nz = np.nonzero(W[row:, col])[0]
        if nz.size == 0:
            continue
        piv = row + int(nz[0])
        if piv != row:
            W[[row, piv]] = W[[piv, row]]
        inv = inv_mod(int(W[row, col]), p)
        W[row, col:] = (W[row, col:] * inv) % p
        below = np.nonzero(W[row + 1:, col])[0]
        if below.size:
            idx = below + row + 1
            W[idx, col:] = (W[idx, col:] - np.outer(W[idx, col], W[row, col:])) % p
        cols.append(col)
        row += 1
    return cols


def det_mod_p(M: np.ndarray, p: int) -> int:
    """det(M) mod p via Gaussian elimination over F_p (p prime, < 2**31)."""
    M = np.asarray(M, dtype=np.int64) % p
    n = M.shape[0]
    assert M.shape == (n, n)
    det = 1
    for col in range(n):
        nz = np.nonzero(M[col:, col])[0]
        if nz.size == 0:
            return 0
        piv = col + int(nz[0])
        if piv != col:
            M[[col, piv]] = M[[piv, col]]
            det = (-det) % p
        det = (det * int(M[col, col])) % p
        inv = inv_mod(int(M[col, col]), p)
        M[col, col:] = (M[col, col:] * inv) % p
        below = np.nonzero(M[col + 1:, col])[0]
        if below.size:
            idx = below + col + 1
            M[idx, col:] = (M[idx, col:] - np.outer(M[idx, col], M[col, col:])) % p
    return det % p
