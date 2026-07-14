"""Round-2 fixes: exact WWW24 basis (ToBasis), portable GS, object mod-q
backend + spec-scale flooding, RO-compressed pk, multi-bit TBIBE,
threshold-Regev BEAT-MEV, CCA2/anti-replication wrapper."""

import math
from dataclasses import replace

import numpy as np
import pytest

from blt25 import bibe, bibe_cca, config, gpv, tbibe, tgs, tobasis, www24
from blt25.bits import stream_from_bytes
from blt25.modq import matmul_q, next_prime
from blt25.params import Params, toy_params, toy_threshold_params
from blt25.trapdoor import trapgen

needs_flint = pytest.mark.skipif(not tobasis.HAVE_FLINT,
                                 reason="python-flint not installed")


# ---------------------------------------------------------------------------
# Exact Lambda^perp basis for the WWW24 matrix (AUDIT S3.1 fix)
# ---------------------------------------------------------------------------

def _dense_instance(tag: bytes):
    """Small WWW24 instance whose exact basis is genuinely dense (huge
    per-direction index defects): the full mp pipeline is exercised."""
    q = next_prime(1 << 10)
    crs = www24.gen(2, q, 2, stream_from_bytes(tag))
    return q, www24.expand(crs, [(0, 1), (1, 0)])


@needs_flint
@pytest.mark.slow
def test_www24_exact_basis_genuine():
    """With exact mode on, the sampler's basis is a genuine basis of
    Lambda^perp(D) (|det| = q^rank, in-lattice) held with mp GS data."""
    old = config.EXACT_WWW24_BASIS
    config.EXACT_WWW24_BASIS = True
    try:
        q, exp = _dense_instance(b"r2-exact")
        ps = exp.sampler()
        assert ps.ajtai.exact_basis is True
        assert ps.ajtai.mp_prec is not None      # dense geometry detected
        rep = tobasis.verify_basis(exp.D, q, ps.ajtai.B, ps.ajtai.B)
        assert rep["in_lattice"] and rep["det_is_q_pow_rank"]
    finally:
        config.EXACT_WWW24_BASIS = old


@needs_flint
@pytest.mark.slow
def test_exact_basis_sampling_quality_and_explain():
    """The regression that motivated the mp pipeline: on the dense genuine
    basis, float64 GS produced ~100x inflated samples.  With mp GS the norms
    match sigma sqrt(M) / sqrt(2 pi), determinism holds, and ExplainSL
    round-trips through the dynamically sized CRT (coefficients here span
    hundreds of bits)."""
    import math
    old = config.EXACT_WWW24_BASIS
    config.EXACT_WWW24_BASIS = True
    try:
        q, exp = _dense_instance(b"r2-quality")
        ps = exp.sampler()
        M = ps.ajtai.B.shape[0]
        sigma = ps.max_gs * 4
        u = stream_from_bytes(b"u").uniform_mod_vec(4, q)
        norms = []
        for t in range(4):
            v = ps.sample_pre(u, sigma, stream_from_bytes(b"q%d" % t))
            norms.append(float(np.linalg.norm(np.asarray(v, dtype=np.float64))))
        expected = sigma * math.sqrt(M) / math.sqrt(2 * math.pi)
        ratio = np.mean(norms) / expected
        assert 0.7 < ratio < 1.4, (ratio, norms)
        # determinism + explain on the mp path
        rb = stream_from_bytes(b"tape").take_bits(
            ps.tape_len_bits(0)).to_bytes(ps.tape_len_bits(0) // 8, "big")
        v = ps.sample_left(None, u, sigma, rb)
        assert (ps.sample_left(None, u, sigma, rb) == v).all()
        rb2 = ps.explain_left(None, u, sigma, v)
        assert (ps.sample_left(None, u, sigma, rb2) == v).all()
    finally:
        config.EXACT_WWW24_BASIS = old


def test_exact_basis_mode_off_by_default():
    """Default configuration keeps the literal Lemma-31 behaviour (the dense
    genuine basis needs the opt-in mp pipeline; see AUDIT.md S3.1)."""
    q = next_prime(1 << 20)
    crs = www24.gen(2, q, 3, stream_from_bytes(b"r2-off"))
    exp = www24.expand(crs, [(0, 1, 0), (1, 0, 1)])
    assert exp.sampler().ajtai.exact_basis is not True


# ---------------------------------------------------------------------------
# Portable deterministic Gram-Schmidt
# ---------------------------------------------------------------------------

def test_portable_gs_matches_and_is_deterministic():
    from blt25.trapdoor import _gram_schmidt_portable, gram_schmidt
    rng = np.random.default_rng(5)
    B = rng.integers(-50, 50, size=(60, 60)).astype(np.int64)
    gs1, n1 = _gram_schmidt_portable(B)
    gs2, n2 = _gram_schmidt_portable(B)
    assert (gs1 == gs2).all() and (n1 == n2).all()      # bit-identical
    gs3, n3 = gram_schmidt(B)                            # QR reference
    assert np.allclose(n1, n3, rtol=1e-9)
    assert np.allclose(np.abs(gs1), np.abs(gs3), atol=1e-6 * np.abs(gs3).max())


def test_portable_gs_via_config_smoke():
    old = config.PORTABLE_GS
    config.PORTABLE_GS = True
    try:
        q = next_prime(1 << 16)
        A, T = trapgen(3, q, 3 * 3 * q.bit_length(), stream_from_bytes(b"pgs"))
        from blt25.trapdoor import PreimageSampler
        ps = PreimageSampler(A, T, q)
        u = np.arange(3, dtype=np.int64)
        v = ps.sample_pre(u, ps.max_gs * 4)
        assert (matmul_q(A, (v % q).reshape(-1, 1), q).reshape(-1) == u % q).all()
    finally:
        config.PORTABLE_GS = old


# ---------------------------------------------------------------------------
# Object backend: spec-scale flooding for threshold GPV (Theorem 7 regime)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gpv_at_theorem7_scale_flooding():
    """Threshold GPV with sigma_flood ~ B_td sqrt(q_sig) lambda (the Theorem 7
    setting) - requires q > 2^31, exercising the object mod-q backend
    end-to-end.  This removes the 'flooding is nominal only' caveat for the
    signature scheme."""
    lam_toy, qsig = 32, 16
    q = next_prime(1 << 45)
    n = 2
    m_c = 3 * n * q.bit_length()
    # B_td = m^{3/2} ceil(log q); sigma_flood = B_td sqrt(qsig) lam
    B_td = m_c ** 1.5 * math.ceil(math.log2(q))
    sigma_flood = B_td * math.sqrt(qsig) * lam_toy
    p = Params(lam=lam_toy, ell=2, d=4, n=n, q=q, m_ro=m_c, m_c=m_c,
               sigma1=1.0, sigma2=1.0, sigma_chi=6.0,
               sigma_flood=sigma_flood)
    pk, parties = gpv.setup(p, N=4, tau=3, seed=b"spec-flood")
    assert pk.B_th < q / 2, "modulus too small for the flooded norm"
    msg = b"theorem-7 scale"
    act = (1, 2, 4)
    r1, st = {}, {}
    for i in act:
        m1, s1 = gpv.sign_one(parties[i - 1], 1, act, msg)
        r1[i], st[i] = m1, s1
    r2 = {i: gpv.sign_two(parties[i - 1], st[i], 1, act, msg, r1) for i in act}
    sig = gpv.combine(pk, 1, act, msg, r1, r2)
    assert sig is not None and gpv.verify(pk, msg, sig)
    assert not gpv.verify(pk, msg + b"!", sig)
    # the run genuinely exercised the beyond-int64 regime: q over 2^31 and a
    # flooded norm bound beyond 2^32 (impossible under the old backend)
    assert q > 2 ** 31 and pk.B_th > 2 ** 32
    assert sigma_flood == B_td * math.sqrt(qsig) * lam_toy


def test_object_backend_matmul_and_solve():
    from blt25.modq import solve_mod_q
    q = next_prime(1 << 45)
    rng = np.random.default_rng(9)
    a = np.array([[int(x) for x in row] for row in
                  rng.integers(0, 1 << 45, size=(4, 12))], dtype=object) % q
    b = np.array([[int(x) for x in row] for row in
                  rng.integers(0, 1 << 45, size=(12, 3))], dtype=object) % q
    got = matmul_q(a, b, q)
    want = (a @ b) % q
    assert all(int(x) == int(y) for x, y in
               zip(np.ravel(got), np.ravel(want)))
    X = solve_mod_q(a, want[:, 0], q)
    assert all(int(x) == int(y) for x, y in
               zip(matmul_q(a, X.reshape(-1, 1), q).reshape(-1), want[:, 0]))


# ---------------------------------------------------------------------------
# Remark 2: RO-compressed public key
# ---------------------------------------------------------------------------

def test_compressed_pk_roundtrip(tiny):
    pk, sk = bibe.setup(tiny, seed=b"r2-compress", compress_pk=True)
    assert pk.pk_seed is not None
    # a decryptor reconstructs the full pk from (seed, C) alone
    pk2 = bibe.expand_compressed_pk(tiny, pk.pk_seed, pk.C)
    assert (pk2.crs.A == pk.crs.A).all() and (pk2.crs.B == pk.crs.B).all()
    assert (pk2.us == pk.us).all()
    ids = (6, 13)
    cts = [bibe.encrypt(pk2, r, b) for r, b in zip(ids, [1, 0])]
    sbk = bibe.pre_dec(sk, ids)
    assert bibe.decrypt(pk2, sbk, cts) == [1, 0]
    bibe._derivation_cache.clear()


# ---------------------------------------------------------------------------
# Multi-bit threshold TBIBE (one TGS sub-session per bit)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_multibit_tbibe(tiny_th):
    p = replace(tiny_th, msg_bits=2)
    pk, parties, _ = tbibe.setup(p, 4, 3, seed=b"r2-mb")
    ids = (5, 11)
    cts = [tbibe.encrypt(pk, r, m) for r, m in zip(ids, [3, 1])]
    sid, act = 7, (1, 3, 4)
    r1o = {i: tbibe.predec_one(parties[i - 1], sid, act, ids) for i in act}
    r1 = {i: o.msgs for i, o in r1o.items()}
    r2 = {i: tbibe.predec_two(parties[i - 1], r1o[i].states, sid, act, ids, r1)
          for i in act}
    pdk = tbibe.combine(pk, sid, act, ids, r1, r2)
    assert pdk is not None and len(pdk.sbks) == 2
    assert tbibe.decrypt(pk, pdk, cts) == [3, 1]
    bibe._derivation_cache.clear()


# ---------------------------------------------------------------------------
# Threshold-Regev BEAT-MEV
# ---------------------------------------------------------------------------

def test_beatmev_threshold_committee():
    from blt25.beatmev import BeatMev
    bm = BeatMev(seed=b"r2-bm")
    msgs = [[1, 0, 1, 1], [0, 1, 0, 0]]
    cts = [bm.encrypt(i, m) for i, m in zip([2, 9], msgs)]
    shares = bm.transport.share_secret(N=5, tau=3)
    for act in [(1, 2, 3), (2, 4, 5)]:
        sbk = bm.pre_dec_threshold(cts, shares, act)
        assert bm.decrypt(sbk, cts, nbits=4) == msgs


# ---------------------------------------------------------------------------
# CCA2 / anti-replication wrapper (Section 1.2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cca_setup(tiny):
    return bibe.setup(tiny, seed=b"r2-cca")


def _two_distinct(pk):
    for _ in range(50):
        a = bibe_cca.encrypt(pk, 1)
        b = bibe_cca.encrypt(pk, 0)
        if a.ct.r != b.ct.r:
            return a, b
    raise RuntimeError("tag space too small (toy d)")


def test_cca_roundtrip(cca_setup):
    pk, sk = cca_setup
    a, b = _two_distinct(pk)
    assert bibe_cca.validate(pk, a) and bibe_cca.validate(pk, b)
    sbk = bibe_cca.pre_dec(sk, [a, b])
    assert bibe_cca.decrypt(pk, sbk, [a, b]) == [1, 0]
    bibe._derivation_cache.clear()


def test_cca_rejects_mauled_ciphertext(cca_setup):
    """Any change to the ciphertext body invalidates the OTS - the CHK-style
    non-malleability mechanism."""
    pk, sk = cca_setup
    a, b = _two_distinct(pk)
    a.ct.ct2[3] = (int(a.ct.ct2[3]) + 1) % pk.params.q
    assert not bibe_cca.validate(pk, a)
    with pytest.raises(ValueError):
        bibe_cca.pre_dec(sk, [a, b])


def test_cca_rejects_replicated_tag(cca_setup):
    """The Section 1.2 attack: reusing an honest tag without the OTS secret
    key cannot produce a valid wrapped ciphertext."""
    pk, sk = cca_setup
    a, b = _two_distinct(pk)
    # adversary reuses a's tag/vk on its own ciphertext body
    forged_ct = bibe.encrypt(pk, a.ct.r, 1)
    forged = bibe_cca.CCACiphertext(vk_digest=a.vk_digest, vk=a.vk,
                                    ct=forged_ct, sig=a.sig)
    assert not bibe_cca.validate(pk, forged)


def test_cca_rejects_wrong_tag_binding(cca_setup):
    pk, sk = cca_setup
    a, b = _two_distinct(pk)
    # signature valid, but tag does not match H(vk)
    swapped = bibe_cca.CCACiphertext(vk_digest=b.vk_digest, vk=b.vk,
                                     ct=a.ct, sig=a.sig)
    assert not bibe_cca.validate(pk, swapped)
