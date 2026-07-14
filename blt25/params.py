"""Parameter selection.

Two regimes:

* ``spec_params``: the provable parameter derivation of Theorems 2/3 (and 4
  for the threshold scheme) with explicit slack functions for the omega(.)
  factors.  These parameters are what the paper's proofs need; the resulting
  moduli/dimensions are far too large to run the scheme on a laptop, but the
  derivation is used by the test-suite to check the constraint system and to
  report asymptotic sizes.

* ``toy_params``: small parameter presets for exercising and benchmarking the
  implementation.  They keep every *algebraic* invariant of the scheme (all
  correctness identities hold, decryption succeeds with large margins) but
  provide NO cryptographic security: n is tiny, sigma1 is calibrated to the
  measured Gram-Schmidt norms instead of the provable bound, and the flooding
  parameter is far below 2^Omega(lambda).  See PARAMS.md.

The omega(.) slack convention used throughout:  omega(f) := f * log2(lambda),
which is omega(f) asymptotically for any fixed polylog and keeps the formulas
concrete and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .gadget import gadget_dims
from .modq import next_prime


@dataclass(frozen=True)
class Params:
    lam: int              # security parameter (nominal in toy mode)
    ell: int              # maximum batch size
    d: int                # index length (bits); |I| <= 2^d for injective H
    n: int                # lattice dimension
    q: int                # modulus (prime)
    m_ro: int             # width of Hro(r)  (the paper's m)
    m_c: int              # width of C       (the paper's m-bar)
    sigma1: float         # SampleLeft Gaussian width
    sigma2: float         # pre-decryption key width
    sigma_chi: float      # LWE noise width (alpha*q in the rho convention)
    msg_bits: int = 1     # N of Section 5.3 (multi-bit variant)
    # threshold-scheme extras (Section 6)
    sigma_flood: float = 0.0
    kappa_prf: int = 32   # PRF seed length in bytes
    toy: bool = True      # True => NOT cryptographically secure

    @property
    def mt(self) -> int:
        """m~ = n (floor(log q) + 1)."""
        return gadget_dims(self.n, self.q)

    @property
    def t(self) -> int:
        """t = m~ (d + 1)."""
        return self.mt * (self.d + 1)

    @property
    def logq(self) -> int:
        return self.q.bit_length()

    # -- transparency: byte sizes (packed at ceil(log q) bits per entry) ------
    def pk_bytes(self) -> int:
        crs = self.n * self.t * self.logq            # [A | B]
        c = self.n * self.m_c * self.logq
        u = self.msg_bits * self.n * self.logq
        return (crs + c + u + 7) // 8

    def ct_bytes(self) -> int:
        bits = (self.msg_bits + self.m_c + self.t + self.m_ro) * self.logq
        return (bits + 7) // 8

    def sbk_bytes(self) -> int:
        per = math.ceil(math.log2(max(2.0, 12 * self.sigma2)))
        return (self.msg_bits * self.m_c * per + 7) // 8


def _omega(f: float, lam: int) -> float:
    return f * math.log2(max(4, lam))


def spec_params(lam: int, ell: int, d: int | None = None,
                msg_bits: int = 1) -> Params:
    """Theorem 2/3 parameter derivation (iterated to a fixed point on log q).

    d defaults to 2*lam (index space large enough for a negligible collision
    probability among poly many random tags).
    """
    if d is None:
        d = 2 * lam
    n = lam
    logq = 40.0
    q = 1 << 40
    for _ in range(64):
        mt = n * (int(logq) + 1)
        t = mt * (d + 1)
        m = 3 * n * math.ceil(logq)
        dim = 2 * ell * t + mt
        sigma1 = max(
            18.0 * dim ** 1.5 * math.log2(dim * lam) * math.log2(max(2.0, math.log2(q))),
            2.0 * m ** 1.5 * _omega(math.log2(m + t), lam),
        )
        mu = max(t, m)
        sigma2 = math.sqrt(m) * _omega(math.sqrt(math.log2(n)), lam)
        q_target = sigma1 * mu ** 1.5 + sigma2 * math.sqrt(m) * mu ** 0.5
        q_target *= _omega(math.log2(mu), lam)
        new_logq = math.log2(q_target) + 1
        if abs(new_logq - logq) < 0.5:
            q = next_prime(int(q_target))
            logq = math.log2(q)
            break
        logq = new_logq
        q = int(2 ** logq)
    mt = n * (q.bit_length() - 1 + 1)
    m = 3 * n * q.bit_length()
    mu = max(mt * (d + 1), m)
    alpha_inv = (mu * sigma1 + math.sqrt(m) * sigma2) * _omega(math.log2(mu), lam)
    sigma_chi = q / alpha_inv  # alpha * q
    return Params(lam=lam, ell=ell, d=d, n=n, q=q, m_ro=m, m_c=m,
                  sigma1=sigma1, sigma2=sigma2, sigma_chi=sigma_chi,
                  msg_bits=msg_bits, sigma_flood=2.0 ** lam, toy=False)


def check_theorem2(p: Params) -> dict[str, tuple[bool, float, float]]:
    """Programmatic check of the Theorem 2/3 constraint system.
    Returns {constraint: (satisfied, lhs, rhs)}."""
    lam, n, q, d, ell = p.lam, p.n, p.q, p.d, p.ell
    mt, t, m, mbar = p.mt, p.t, p.m_ro, p.m_c
    dim = 2 * ell * t + mt
    mu = max(t, m)
    out = {}
    out["n >= lambda"] = (n >= lam, n, lam)
    out["m_ro >= 3 n ceil(log q)"] = (m >= 3 * n * q.bit_length(), m, 3 * n * q.bit_length())
    out["m_c >= 3 n ceil(log q)"] = (mbar >= 3 * n * q.bit_length(), mbar, 3 * n * q.bit_length())
    out["t = mt (d+1)"] = (t == mt * (d + 1), t, mt * (d + 1))
    s1_req = max(18.0 * dim ** 1.5 * math.log2(dim * lam) * math.log2(max(2.0, math.log2(q))),
                 2.0 * m ** 1.5 * _omega(math.log2(m + t), lam))
    out["sigma1 >= max(18 dim^1.5 log(dim lam) loglog q, 2 m^1.5 w(log(m+t)))"] = (
        p.sigma1 >= s1_req, p.sigma1, s1_req)
    s2_req = math.sqrt(mbar) * _omega(math.sqrt(math.log2(n)), lam)
    out["sigma2 >= sqrt(m_c) w(sqrt(log n))"] = (p.sigma2 >= s2_req, p.sigma2, s2_req)
    q_req = p.sigma1 * mu ** 1.5 + p.sigma2 * math.sqrt(mbar) * math.sqrt(mu)
    out["q = Omega(sigma1 mu^1.5 + sigma2 sqrt(m_c) sqrt(mu))"] = (q >= q_req, q, q_req)
    alpha = p.sigma_chi / q
    a_req = 1.0 / ((mu * p.sigma1 + math.sqrt(mbar) * p.sigma2) * _omega(math.log2(mu), lam))
    out["alpha < 1/((mu sigma1 + sqrt(m_c) sigma2) w(log mu))"] = (alpha <= a_req, alpha, a_req)
    out["q > 2 sqrt(n)/alpha"] = (q > 2 * math.sqrt(n) / max(alpha, 1e-300), q,
                                  2 * math.sqrt(n) / max(alpha, 1e-300))
    return out


def threshold_bounds(p: Params, N: int, tau: int) -> dict[str, float]:
    """Theorem 4 quantities: B_td, B_th and the threshold correctness bounds."""
    mbar, mt, q = p.m_c, gadget_dims(p.n, p.q), p.q
    B_td = mbar ** 1.5 * math.ceil(math.log2(q))
    B_th = (math.sqrt(mbar * mt) * math.log2(q)
            + math.sqrt(tau) * math.sqrt(mbar) * p.sigma_flood
            * _omega(math.sqrt(math.log2(p.lam)), p.lam))
    mu = max(p.t, p.m_ro)
    return {
        "B_td": B_td,
        "B_th": B_th,
        "q_required": p.sigma1 * mu ** 1.5 + B_th * math.sqrt(mu),
        "alpha_bound": 1.0 / ((mu * p.sigma1 + B_th) * _omega(math.log2(mu), p.lam)),
    }


# ----------------------------------------------------------------------------
# Toy presets (calibrated empirically; see scripts/calibrate.py)
# ----------------------------------------------------------------------------

def toy_params(name: str = "tiny") -> Params:
    """Toy presets.  NOT SECURE - for correctness testing and benchmarking.

    sigma1 must exceed the maximum Gram-Schmidt norm of the extended basis for
    the batch matrix [A | A-hat]; the values below were calibrated with a
    >= 2x margin over the maxima observed across many random batches, and the
    schemes re-check the bound at run time.
    """
    presets = {
        # n=4, d=4, ell<=2: fastest; unit tests
        "tiny": Params(lam=4, ell=2, d=4, n=4, q=next_prime(1 << 27),
                       m_ro=3 * 4 * 28, m_c=3 * 4 * 28,
                       sigma1=4000.0, sigma2=400.0, sigma_chi=6.0),
        # n=4, d=6, ell<=4: small batches
        "small": Params(lam=4, ell=4, d=6, n=4, q=next_prime(1 << 28),
                        m_ro=3 * 4 * 29, m_c=3 * 4 * 29,
                        sigma1=6000.0, sigma2=450.0, sigma_chi=6.0),
        # n=8, d=6, ell<=4: benchmark upper end (minutes per derivation)
        "medium": Params(lam=8, ell=4, d=6, n=8, q=next_prime(1 << 29),
                         m_ro=3 * 8 * 30, m_c=3 * 8 * 30,
                         sigma1=14000.0, sigma2=900.0, sigma_chi=6.0),
    }
    p = presets[name]
    return p


def toy_threshold_params(name: str = "tiny", sigma_flood: float = 2.0 ** 14) -> Params:
    """Threshold toy preset: same as toy_params but with a flooding width and a
    modulus enlarged (one extra bit) to absorb the flooded key norm
    ||sbk|| <= B_th (Theorem 4)."""
    base = toy_params(name)
    q = next_prime(base.q << 1)
    return replace(base, q=q, m_ro=3 * base.n * q.bit_length(),
                   m_c=3 * base.n * q.bit_length(), sigma_flood=sigma_flood)
