import numpy as np
from pathlib import Path
from scipy.io import wavfile
from ligotools.utils import whiten, write_wavfile, reqshift

def _flat_interp_psd(fs, n):
    """Helper: return a callable interp_psd(f) ~ 1 over [0, fs/2]."""
    f = np.linspace(0, fs/2, n//2 + 1)
    psd = np.ones_like(f)
    # simple linear interpolator implemented via numpy.interp inside whiten
    return lambda freq: np.interp(freq, f, psd, left=1.0, right=1.0)

def test_whiten_shape_and_scale():
    rng = np.random.default_rng(0)
    fs = 4096
    dt = 1.0 / fs
    x = rng.normal(0, 1, fs * 2)  # 2 seconds
    interp_psd = _flat_interp_psd(fs, x.size)
    w = whiten(x, interp_psd, dt)
    assert w.shape == x.shape

    v_in = np.var(x)
    v_out = np.var(w)
    expected = 2 * dt * v_in

    assert np.isclose(v_out, expected, rtol=0.25, atol=0)

def test_reqshift_and_write_wavfile(tmp_path):
    fs = 4096
    t = np.arange(fs) / fs
    x = np.sin(2*np.pi*440*t)
    y = reqshift(x, fshift=100, sample_rate=fs)
    # Basic frequency-content sanity: peak should move roughly by ~100 Hz
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    fx = np.fft.rfftfreq(x.size, 1/fs)
    f_peak_x = fx[np.argmax(np.abs(X))]
    f_peak_y = fx[np.argmax(np.abs(Y))]
    assert abs((f_peak_y - f_peak_x) - 100) < 15.0  # allow leakage tolerance

    out = tmp_path / "test.wav"
    write_wavfile(out, fs, y)
    assert out.exists()
    fs2, y2 = wavfile.read(out)
    assert fs2 == fs
    assert len(y2) == len(y)