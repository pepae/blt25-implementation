"""Explainable 1-D discrete Gaussian sampling (Section 4 building block).

The paper's SampleLeft must be a *deterministic* function of its input and a
random tape rb, and must come with an ExplainSL algorithm that, given a vector
v in the support, finds a tape rb' with SampleLeft(inp; rb') = v.  Following
Lu-Waters [49] and Champion-Hsieh-Wu [28, Sec. 7], the primitive underneath is
a 1-D discrete Gaussian sampler over Z that

  * consumes a fixed number of tape bits (NBITS per coordinate), interpreting
    them as a dyadic point x in (0,1), and outputs z = F^{-1}(x) for a fixed
    monotone CDF F approximating the discrete Gaussian D_{Z, sigma, c}; and
  * can be "explained": given z, output fresh uniform bits x' in the interval
    [F(z-1), F(z)), so that re-running the sampler on x' returns z.

Determinism and explainability hold *exactly* for any fixed monotone F; the
quality of F only affects how close the output distribution is to the ideal
discrete Gaussian.  We use:

  - width convention rho_sigma(x) = exp(-pi x^2 / sigma^2), i.e. the standard
    deviation is s = sigma / sqrt(2 pi)  (matches the paper / [GPV08]);
  - s <= 48:      exact windowed summation of rho (float64 + fsum);
  - s in (48, 2^40]: continuous-Gaussian CDF at half-integer cut points with
    the first Euler-Maclaurin correction term,
        F(z) = Phi(t) + t phi(t) / (24 s^2),  t = (z + 1/2 - c)/s,
    accurate to O(1/s^4) per point (float64);
  - s > 2^40:     same formula evaluated in mpmath (huge flooding widths).

The per-point statistical error of the middle regime is ~ 1/(24 s^2)^2-scale;
at the toy parameters used in this repo it contributes < 1e-6 total variation
per SampleLeft call.  See AUDIT.md for the accounting.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist

import mpmath as mp
import numpy as np

from .bits import BitStream, random_stream

NBITS = 128                 # tape bits consumed per coordinate
_EXACT_S_MAX = 48.0         # exact summation below this s.d.
_MP_S_MIN = float(1 << 40)  # mpmath above this s.d.
_WINDOW_SD = 10.5           # exact-mode window half-width, in s.d. units
_SQRT2PI = math.sqrt(2.0 * math.pi)
_ND = NormalDist()


class ExplainError(RuntimeError):
    """Raised when the CDF interval of the requested value contains no dyadic
    tape point (probability ~ 2^-NBITS / mass(z)) or the value has ~zero mass."""


def sd_of(sigma: float) -> float:
    """Standard deviation s for width parameter sigma (rho convention)."""
    return float(sigma) / _SQRT2PI


# ----------------------------------------------------------------------------
# CDF implementations.  All return P[X <= z] for X ~ (approx) D_{Z, sigma, c}.
# ----------------------------------------------------------------------------

class _ExactCDF:
    """Windowed exact CDF for small s: normalized cumulative rho over
    [z0 - R, z0 + R], z0 = round(c), R = ceil(WINDOW_SD * s) + 2."""

    def __init__(self, c: float, sigma: float):
        s = sd_of(sigma)
        self.z0 = int(round(c))
        self.R = int(math.ceil(_WINDOW_SD * max(s, 1.0))) + 2
        self.lo = self.z0 - self.R
        self.hi = self.z0 + self.R
        zs = np.arange(self.lo, self.hi + 1, dtype=np.float64)
        logs = -math.pi * (zs - c) ** 2 / (float(sigma) ** 2)
        # subtract max for stability; normalization is by the exact total below
        w = np.exp(logs - logs.max())
        self._cums = np.cumsum(w)
        self._total = Fraction(float(self._cums[-1]))

    def F(self, z: int) -> Fraction:
        if z < self.lo:
            return Fraction(0)
        if z >= self.hi:
            return Fraction(1)
        return Fraction(float(self._cums[z - self.lo])) / self._total


class _Erf64CDF:
    """Float64 erf-based CDF with first Euler-Maclaurin correction."""

    def __init__(self, c: float, sigma: float):
        self.c = float(c)
        self.s = sd_of(sigma)

    def F(self, z: int) -> Fraction:
        t = (z + 0.5 - self.c) / self.s
        if t <= -38.0:
            return Fraction(0)
        if t >= 38.0:
            return Fraction(1)
        phi_cdf = 0.5 * math.erfc(-t / math.sqrt(2.0))
        phi_pdf = math.exp(-0.5 * t * t) / _SQRT2PI
        val = phi_cdf + t * phi_pdf / (24.0 * self.s * self.s)
        val = min(max(val, 0.0), 1.0)
        return Fraction(val)


def _mpf_to_fraction(v) -> Fraction:
    """Exact dyadic value of an mpf as a Fraction."""
    sign, man, exp, _ = mp.mpf(v)._mpf_
    man = int(man)
    if sign:
        man = -man
    e = int(exp)
    if e >= 0:
        return Fraction(man << e)
    return Fraction(man, 1 << (-e))


class _ErfMpCDF:
    """mpmath erf-based CDF for huge s (arbitrary magnitude, deterministic)."""

    def __init__(self, c, sigma, prec_bits: int):
        self.prec = prec_bits
        with mp.workprec(self.prec):
            self.c = mp.mpf(c)
            self.s = mp.mpf(sigma) / mp.sqrt(2 * mp.pi)

    def F(self, z: int) -> Fraction:
        with mp.workprec(self.prec):
            t = (mp.mpf(z) + mp.mpf(0.5) - self.c) / self.s
            if t <= -40:
                return Fraction(0)
            if t >= 40:
                return Fraction(1)
            phi_cdf = mp.erfc(-t / mp.sqrt(2)) / 2
            phi_pdf = mp.e ** (-t * t / 2) / mp.sqrt(2 * mp.pi)
            val = phi_cdf + t * phi_pdf / (24 * self.s * self.s)
            val = min(max(val, mp.mpf(0)), mp.mpf(1))
            return _mpf_to_fraction(val)


def _make_cdf(c: float, sigma: float):
    s = sd_of(sigma)
    if s <= _EXACT_S_MAX:
        return _ExactCDF(c, sigma)
    if s <= _MP_S_MIN:
        return _Erf64CDF(c, sigma)
    prec = NBITS + int(math.log2(max(s, 2))) + 48
    return _ErfMpCDF(c, sigma, prec)


# ----------------------------------------------------------------------------
# Sampling / explaining
# ----------------------------------------------------------------------------

def _initial_guess(x: Fraction, c: float, sigma: float) -> int:
    s = sd_of(sigma)
    xf = float(x)
    xf = min(max(xf, 1e-300), 1.0 - 1e-16)
    try:
        t = _ND.inv_cdf(xf)
    except Exception:
        t = 0.0
    t = min(max(t, -45.0), 45.0)
    if s > _MP_S_MIN:
        with mp.workprec(NBITS + int(math.log2(s)) + 48):
            z = mp.floor(mp.mpf(c) + (mp.mpf(sigma) / mp.sqrt(2 * mp.pi)) * t)
            return int(z)
    return int(math.floor(c + s * t))


def sample_z(c: float, sigma: float, stream: BitStream, nbits: int = NBITS) -> int:
    """Sample z ~ D_{Z, sigma, c} (approx) deterministically from `stream`.

    Consumes exactly `nbits` bits.  Output is z = min{ w : x < F(w) } for
    x = (bits + 1/2) / 2^nbits.
    """
    xi = stream.take_bits(nbits)
    x = Fraction(2 * xi + 1, 2 ** (nbits + 1))
    cdf = _make_cdf(c, sigma)
    z = _initial_guess(x, c, sigma)
    # fix-up: establish invariant F(z-1) <= x < F(z)
    guard = 0
    while x >= cdf.F(z):
        z += 1
        guard += 1
        if guard > 10_000:
            raise RuntimeError("CDF inversion diverged (F too flat?)")
    while x < cdf.F(z - 1):
        z -= 1
        guard += 1
        if guard > 20_000:
            raise RuntimeError("CDF inversion diverged")
    return z


def explain_z(z: int, c: float, sigma: float, nbits: int = NBITS,
              rng: BitStream | None = None) -> int:
    """Return tape bits xi (as an int of `nbits` bits) such that
    sample_z would output z, chosen uniformly among all such tapes."""
    cdf = _make_cdf(c, sigma)
    Fzm1, Fz = cdf.F(z - 1), cdf.F(z)
    # x = (2 xi + 1) / 2^{nbits+1} must satisfy  Fzm1 <= x < Fz
    # xi >= (Fzm1 * 2^{nbits+1} - 1) / 2   -> xi_lo = ceil(...)
    # xi <  (Fz   * 2^{nbits+1} - 1) / 2   -> xi_hi = ceil(...) - 1
    scale = Fraction(2 ** (nbits + 1))
    lo_num = Fzm1 * scale - 1
    hi_num = Fz * scale - 1
    xi_lo = -((-lo_num.numerator) // (2 * lo_num.denominator)) if lo_num > 0 else 0
    xi_hi = -((-hi_num.numerator) // (2 * hi_num.denominator)) - 1
    xi_hi = min(xi_hi, 2 ** nbits - 1)
    xi_lo = max(xi_lo, 0)
    if xi_hi < xi_lo:
        raise ExplainError(f"empty tape interval for z={z} (mass too small)")
    rng = rng or random_stream()
    span = xi_hi - xi_lo + 1
    return xi_lo + (rng.take_bits(span.bit_length() + 64) % span)


def sample_dgauss_vec(n: int, sigma: float, stream: BitStream | None = None,
                      c: np.ndarray | None = None, nbits: int = NBITS) -> np.ndarray:
    """iid vector sample from D_{Z^n, sigma, c} (approx).  Returns int64 when
    the entries fit, else an object array of Python ints."""
    stream = stream or random_stream()
    out = []
    for i in range(n):
        ci = 0.0 if c is None else float(c[i])
        out.append(sample_z(ci, sigma, stream, nbits))
    if all(abs(v) < (1 << 62) for v in out):
        return np.array(out, dtype=np.int64)
    return np.array(out, dtype=object)
