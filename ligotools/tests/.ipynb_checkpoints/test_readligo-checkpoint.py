import os
import numpy as np
from ligotools.readligo import readligo


def _data_path(fname):
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(repo_root, "data", fname)


def test_readligo_returns_expected_shapes():
    fname = _data_path("H-H1_LOSC_4_V2-1126259446-32.hdf5")
    strain, time, chan_dict = readligo(fname)

    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert strain.ndim == 1
    assert time.ndim == 1
    assert len(strain) == len(time)
    assert isinstance(chan_dict, dict)


def test_readligo_sampling_rate_is_reasonable():
    fname = _data_path("H-H1_LOSC_4_V2-1126259446-32.hdf5")
    strain, time, _ = readligo(fname)

    dt = time[1] - time[0]
    fs = 1.0 / dt
    assert 4090 < fs < 4100   # loose but should catch errors


def test_readligo_raises_for_missing_file():
    bad_fname = _data_path("this_file_does_not_exist.hdf5")
    try:
        readligo(bad_fname)
        assert False, "readligo should fail on missing file"
    except (IOError, OSError, FileNotFoundError):
        pass