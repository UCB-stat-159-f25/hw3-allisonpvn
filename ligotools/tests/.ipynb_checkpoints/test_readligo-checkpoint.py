import os
import numpy as np
import pytest
from ligotools import readligo as rl


def test_readligo_api_surface():
    import ligotools
    assert hasattr(ligotools, "readligo")
    rl = ligotools.readligo
    assert callable(rl) or hasattr(rl, "loaddata")

def test_readligo_missing_file_contract():
    bad = os.path.join(os.path.dirname(__file__), "this_file_should_not_exist_12345.hdf5")

    if callable(rl):
        try:
            out = rl(bad, "H1")
        except FileNotFoundError:
            return
        assert out == (None, None, None)
    else:
        try:
            out = rl.loaddata(bad, "H1")
        except Exception:
            return
        assert out == (None, None, None)
