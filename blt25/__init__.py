"""blt25: implementation of "Threshold Batch Identity-based Encryption without Epochs"
by Dan Boneh, Evan Laufer, and Ertem Nusret Tas (Stanford University).

Modules
-------
- ``modq``      : Z_q matrix/vector arithmetic (overflow-safe NumPy backend).
- ``bits``      : deterministic bit streams (SHAKE-256 XOF) used as random tapes.
- ``gadget``    : gadget matrix G_n, bit decomposition G^{-1}, basis of Lambda^perp(G).
- ``gaussian``  : explainable 1-D discrete Gaussian sampler (Lu-Waters / CHW style).
- ``trapdoor``  : TrapGen, Ajtai bases (Lemma 31), DetExtBasis, Klein sampler,
                  SamplePre / SampleLeft / ExplainSL  (Section 4, Appendix C).
- ``www24``     : shifted multi-preimage trapdoor sampler with d-bit labels
                  (Section 3.5, Appendix B; Waters-Wee-Wu construction).
- ``hashes``    : the hash functions H, Hro, Hsp, Hsp_th, Htr, HGPV and the PRF.
- ``params``    : parameter derivation per Theorems 2/3/4 + toy presets.
- ``bibe``      : the batch IBE scheme of Section 5 (incl. multi-bit variant).
- ``tgs``       : threshold Gaussian preimage sampling (Section 6.1).
- ``tbibe``     : threshold batch IBE (Section 6.2).
- ``gpv``       : two-round threshold GPV signatures (Section 7).
- ``abe``       : KP-ABE (BGG+14-style, indicator-sum policies) and the generic
                  ABE -> BIBE transformation (Section 8).
- ``trilinear`` : batch IBE from a trilinear map, simulated generic group model
                  (Section 9).  NOT a real cryptographic instantiation.
- ``beatmev``   : lattice BEAT-MEV sketch via the BLMR key-homomorphic PRF
                  (Appendix A.4).

SECURITY WARNING: this is a research prototype accompanying the paper.  The
bundled parameter presets are *toy* parameters chosen so that the algebra and
protocol logic can be exercised and benchmarked on a laptop; they provide **no**
cryptographic security.  See PARAMS.md / AUDIT.md for details.
"""

__all__ = [
    "modq", "bits", "gadget", "gaussian", "trapdoor", "www24", "hashes",
    "params", "bibe", "tgs", "tbibe", "gpv", "abe", "trilinear", "beatmev",
]
__version__ = "0.1.0"
