"""End-to-end tests for the Section 5 batch IBE scheme."""

import numpy as np
import pytest
from dataclasses import replace

from blt25 import bibe
from blt25.bits import stream_from_bytes
from blt25.hashes import hash_sp
from blt25.modq import matmul_q
from blt25.params import toy_params

IDS = (3, 9)          # a fixed batch, reused so the derivation cache warms up
BITS = [1, 0]


@pytest.fixture(scope="module")
def keys(tiny):
    return bibe.setup(tiny, seed=b"bibe-tests")


@pytest.fixture(scope="module")
def batch(keys):
    pk, sk = keys
    cts = [bibe.encrypt(pk, r, b) for r, b in zip(IDS, BITS)]
    sbk = bibe.pre_dec(sk, IDS)
    return cts, sbk


def test_roundtrip(keys, batch):
    pk, _ = keys
    cts, sbk = batch
    msgs, errs = bibe.decrypt(pk, sbk, cts, return_error=True)
    assert msgs == BITS
    assert max(errs) < pk.params.q // 4


def test_all_bit_patterns(keys):
    pk, sk = keys
    for bits in ([0, 0], [1, 1], [0, 1]):
        cts = [bibe.encrypt(pk, r, b) for r, b in zip(IDS, bits)]
        sbk = bibe.pre_dec(sk, IDS)
        assert bibe.decrypt(pk, sbk, cts) == bits


def test_smaller_batch(keys):
    """Remark 1: batches of size < ell work unchanged."""
    pk, sk = keys
    cts = [bibe.encrypt(pk, 7, 1)]
    sbk = bibe.pre_dec(sk, (7,))
    assert bibe.decrypt(pk, sbk, cts) == [1]


def test_predec_key_relation(keys, batch):
    """C sbk = G c-hat (the defining relation of the pre-decryption key)."""
    pk, _ = keys
    _, sbk = batch
    der = bibe.derive(pk, IDS)
    q = pk.params.q
    lhs = matmul_q(pk.C, (sbk.sbks[0] % q).reshape(-1, 1), q).reshape(-1)
    assert (lhs == der.cs[0] % q).all()


def test_dec_equals_predec_derivation(keys):
    """PreDec and Dec must derive the identical SampleLeft output; this is the
    determinism property that makes the short sbk sufficient."""
    pk, _ = keys
    bibe._derivation_cache.clear()
    d1 = bibe.derive(pk, IDS)
    bibe._derivation_cache.clear()
    d2 = bibe.derive(pk, IDS)
    for a, b in zip(d1.v_blocks[0], d2.v_blocks[0]):
        assert (a == b).all()
    assert (d1.chats[0] == d2.chats[0]).all()


def test_identity_binding(keys, batch):
    """Decrypting with a key for a different batch raises."""
    pk, sk = keys
    cts, _ = batch
    other = bibe.pre_dec(sk, (5, 12))
    with pytest.raises(ValueError):
        bibe.decrypt(pk, other, cts)


def test_out_of_batch_ciphertext_does_not_decrypt(keys, batch):
    """A ciphertext under an identity outside the batch must not decrypt
    correctly under the batch key even if we force the tag to collide."""
    pk, sk = keys
    _, sbk = batch
    outside = bibe.encrypt(pk, 14, 1)
    forged = bibe.Ciphertext(r=IDS[0], ct1=outside.ct1, ct2=outside.ct2,
                             ct3=outside.ct3)
    cts = [forged, bibe.encrypt(pk, IDS[1], 0)]
    msgs, errs = bibe.decrypt(pk, sbk, cts, return_error=True)
    # the forged slot decodes to garbage: its error term is essentially
    # uniform, far above the honest bound
    assert errs[0] > pk.params.q // 64


def test_index_collision_rejected(keys):
    pk, sk = keys
    with pytest.raises(ValueError):
        bibe.pre_dec(sk, (3, 3))


def test_batch_too_large(keys):
    pk, sk = keys
    with pytest.raises(ValueError):
        bibe.pre_dec(sk, (1, 2, 3))


def test_hsp_tape_binds_inputs(keys):
    """Different target vector u => different Hsp tape (Definition 15)."""
    pk, _ = keys
    der = bibe.derive(pk, IDS)
    rho = 8 * 100
    a = hash_sp(der.exp.D, der.exp.T, IDS, pk.us[0], 4000, rho)
    b = hash_sp(der.exp.D, der.exp.T, IDS, (pk.us[0] + 1) % pk.params.q, 4000, rho)
    assert a != b


def test_multibit(tiny):
    p = replace(tiny, msg_bits=3)
    pk, sk = bibe.setup(p, seed=b"multibit")
    cts = [bibe.encrypt(pk, r, m) for r, m in zip(IDS, [5, 2])]
    sbk = bibe.pre_dec(sk, IDS)
    assert bibe.decrypt(pk, sbk, cts) == [5, 2]
    assert len(sbk.sbks) == 3


def test_sizes_reported(tiny):
    assert tiny.pk_bytes() > 0
    assert tiny.ct_bytes() > 0
    assert tiny.sbk_bytes() > 0
    # succinctness sanity: sbk is much smaller than ell ciphertexts
    assert tiny.sbk_bytes() < tiny.ct_bytes()
