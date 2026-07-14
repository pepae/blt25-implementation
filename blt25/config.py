"""Runtime configuration switches (also settable via environment variables).

EXACT_WWW24_BASIS ('auto' | True | False, env BLT25_EXACT_BASIS=auto/1/0)
    Whether `trapdoor.ajtai_basis` upgrades the literal Lemma-31 full-rank set
    to a genuine Lambda^perp basis via MG02 ToBasis (blt25.tobasis; needs
    python-flint).  'auto' enables it when flint is importable and the lattice
    dimension is at most EXACT_BASIS_MAX_DIM (the HNF step is the cost).
    True forces it (raises if flint is missing); False keeps the literal
    Lemma-31 behaviour (AUDIT.md S3.1 deviation).

EXACT_BASIS_MAX_DIM (int, env BLT25_EXACT_BASIS_MAX_DIM)
    Dimension cutoff for 'auto'.

PORTABLE_GS (bool, env BLT25_PORTABLE_GS=1)
    Use the BLAS-free, exactly-specified float64 Gram-Schmidt (elementwise
    IEEE products + math.fsum) so that the derivation is bit-identical across
    platforms/BLAS builds, at ~30-100x the LAPACK cost.  Required for
    multi-party deployments where decryptors run heterogeneous stacks.
"""

from __future__ import annotations

import os

# Exact-basis mode is OFF by default: the genuine Lambda^perp basis of these
# q-ary lattices is extremely dense (GS norms spanning ~log2(index) bits, with
# indexes of hundreds of bits), so correct Klein sampling over it requires
# mpmath Gram-Schmidt and centers at matching precision - O(M^3) mpf work,
# minutes already at M ~ 400 (plus the flint HNF: M=378 -> 3.3 s,
# M=1232 -> ~5 min).  Enable for fidelity runs; 'auto' turns it on when flint
# is available AND the dimension is below the cutoff.
EXACT_WWW24_BASIS: object = False
EXACT_BASIS_MAX_DIM: int = 400
PORTABLE_GS: bool = False


def _env_bool(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


def exact_basis_mode() -> object:
    v = os.environ.get("BLT25_EXACT_BASIS")
    if v is None:
        return EXACT_WWW24_BASIS
    if v == "auto":
        return "auto"
    return v not in ("0", "false", "False", "")


def exact_basis_max_dim() -> int:
    return int(os.environ.get("BLT25_EXACT_BASIS_MAX_DIM", EXACT_BASIS_MAX_DIM))


def portable_gs() -> bool:
    return _env_bool("BLT25_PORTABLE_GS", PORTABLE_GS)
