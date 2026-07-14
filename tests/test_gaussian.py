"""Tests for the explainable 1-D discrete Gaussian sampler."""

import math
from fractions import Fraction

import numpy as np
import pytest

from blt25.bits import stream_from_bytes
from blt25.gaussian import (NBITS, _make_cdf, explain_z, sample_dgauss_vec,
                            sample_z, sd_of)

REGIMES = [
    ("exact", 25.0, 0.3),
    ("exact-neg-center", 60.0, -12.7),
    ("erf64", 500.0, 3.9),
    ("erf64-large", 2.0 ** 20, 1000.5),
    ("mpmath-huge", 2.0 ** 45, 7.0),
]


@pytest.mark.parametrize("name,sigma,c", REGIMES)
def test_determinism(name, sigma, c):
    a = [sample_z(c, sigma, stream_from_bytes(b"s" + name.encode()))
         for _ in range(1)]
    b = [sample_z(c, sigma, stream_from_bytes(b"s" + name.encode()))
         for _ in range(1)]
    assert a == b


@pytest.mark.parametrize("name,sigma,c", REGIMES)
def test_explain_roundtrip(name, sigma, c):
    """Definition 12 explainability at the 1-D level: the explained tape must
    reproduce the value through the CDF-interval invariant."""
    st = stream_from_bytes(b"e" + name.encode())
    cdf = _make_cdf(c, sigma)
    for _ in range(8):
        z = sample_z(c, sigma, st)
        xi = explain_z(z, c, sigma)
        x = Fraction(2 * xi + 1, 2 ** (NBITS + 1))
        assert cdf.F(z - 1) <= x < cdf.F(z)


def test_cdf_monotone_and_bounded():
    for sigma, c in [(25.0, 0.3), (500.0, -3.2), (2.0 ** 20, 11.0)]:
        cdf = _make_cdf(c, sigma)
        s = sd_of(sigma)
        zs = [int(c + k * s) for k in range(-8, 9)]
        vals = [cdf.F(z) for z in zs]
        assert all(Fraction(0) <= v <= Fraction(1) for v in vals)
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


@pytest.mark.stat
def test_moments_match():
    """Mean/variance of the sampler match the discrete Gaussian.  The bit
    streams are deterministic, so this test cannot flake."""
    for sigma in (12.0, 300.0):
        st = stream_from_bytes(b"mom%d" % int(sigma))
        n = 4000
        v = sample_dgauss_vec(n, sigma, st).astype(np.float64)
        s = sd_of(sigma)
        assert abs(v.mean()) < 4 * s / math.sqrt(n)
        assert abs(v.std() / s - 1) < 0.05


@pytest.mark.stat
def test_chi_square_small_sigma():
    """Frequency test against the exact pmf for a small width."""
    sigma, c = 10.0, 0.0
    s = sd_of(sigma)
    st = stream_from_bytes(b"chi2")
    n = 6000
    xs = [sample_z(c, sigma, st) for _ in range(n)]
    lo, hi = -12, 12
    obs = np.zeros(hi - lo + 1)
    for x in xs:
        if lo <= x <= hi:
            obs[x - lo] += 1
    zs = np.arange(lo, hi + 1, dtype=np.float64)
    pmf = np.exp(-math.pi * zs ** 2 / sigma ** 2)
    pmf /= pmf.sum()
    exp = pmf * n
    mask = exp > 8
    chi2 = float(((obs[mask] - exp[mask]) ** 2 / exp[mask]).sum())
    dof = int(mask.sum()) - 1
    # generous bound: mean dof, sd sqrt(2 dof); allow 6 sigma
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)


def test_tail_cut():
    """Values stay within ~14 standard deviations (tape has 128 bits)."""
    st = stream_from_bytes(b"tail")
    sigma = 40.0
    s = sd_of(sigma)
    for _ in range(500):
        z = sample_z(0.0, sigma, st)
        assert abs(z) < 14 * s
