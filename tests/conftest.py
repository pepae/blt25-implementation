import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt25 import bibe
from blt25.params import toy_params, toy_threshold_params


@pytest.fixture(scope="session")
def tiny():
    return toy_params("tiny")


@pytest.fixture(scope="session")
def tiny_th():
    return toy_threshold_params("tiny")


@pytest.fixture(scope="session")
def bibe_keys(tiny):
    """One BIBE keypair shared across the session (setup is cheap; the
    derivation cache inside blt25.bibe amortizes repeated batches)."""
    return bibe.setup(tiny, seed=b"test-suite")
