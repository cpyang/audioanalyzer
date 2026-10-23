"""Spectral engine tests: calibration, frequency accuracy, floor estimation (FR-3)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from audioanalyzer import synth
from audioanalyzer.config import Config
from audioanalyzer.spectral import (SpectralEngine, candidate_tones, floor_curve, interp_peak,
                                    local_maxima, mask_below, neighborhood_floor, power_from_db)
from audioanalyzer.spectral import aggregate

FS = 16000


def _sine(freq: float, amp: float = 1.0, seconds: float = 5.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * math.pi * freq * t)


def test_amplitude_reference_is_peak_dbfs():
    """A full-scale sine at a bin centre reads 0.0 dBFS (FR-3.3 convention)."""
    eng = SpectralEngine(Config(), FS)
    n, db = eng.block_db(_sine(1000.0).astype(np.float32))     # 1000/7.8125 = 128 exactly
    assert n > 0
    assert db[5, 128] == pytest.approx(0.0, abs=0.02)


def test_rms_and_amplitude_reference_differ_by_3_01_db():
    """Documents the 3.01 dB relationship so nobody mixes the two conventions."""
    x = _sine(1000.0)
    eng = SpectralEngine(Config(), FS)
    _, db = eng.block_db(x.astype(np.float32))
    rms_dbfs = 20 * math.log10(float(np.sqrt(np.mean(x ** 2))))
    assert db[5, 128] - rms_dbfs == pytest.approx(3.0103, abs=0.02)


def test_half_bin_scalloping_and_interpolation():
    """Worst-case offset loses ~1.42 dB in the bin; interpolation recovers freq exactly."""
    eng = SpectralEngine(Config(), FS)
    _, db = eng.block_db(_sine(1003.90625).astype(np.float32))   # bin 128.5
    row = np.nan_to_num(db[5], neginf=-400.0)
    k = int(np.argmax(row[1:row.size - 1]) + 1)
    delta, lvl = interp_peak(row, k)
    assert row[k] == pytest.approx(-1.42, abs=0.15)              # Hann scalloping, as theory
    assert eng.freqs[k] + float(delta[0]) * eng.df == pytest.approx(1003.90625, abs=0.05)


@pytest.mark.parametrize("freq,level", [(1000.0, 0.0), (3000.0, -20.0), (50.0, -45.0),
                                        (7000.0, -60.0), (7813.0, -80.0)])
def test_frequency_accuracy_across_band(freq, level):
    eng = SpectralEngine(Config(), FS)
    _, db = eng.block_db(_sine(freq, 10 ** (level / 20.0)).astype(np.float32))
    row = np.nan_to_num(db[5], neginf=-400.0)
    idx = local_maxima(row, min_bin=2)
    best = idx[np.argmax(row[idx])]
    delta, _ = interp_peak(row, best)
    assert eng.freqs[best] + float(delta[0]) * eng.df == pytest.approx(freq, abs=0.6)


def test_dc_offset_produces_no_candidates():
    """AC-1: the -38 dBFS DC offset must never appear as a tone.

    Before the per-frame DC notch, that offset leaked to -108 dBFS in bin 2 -- 12 dB above the
    stop-band around it, i.e. a convincing phantom "tone" at 15.6 Hz in any quiet recording.
    Removing the frame's weighted mean drives the DC region to exact zero, and A6 still reports
    the offset from the time domain.
    """
    cfg = Config()
    eng = SpectralEngine(cfg, FS)
    _, db = eng.block_db(np.full(5 * FS, -0.012237, dtype=np.float32))
    row = np.nan_to_num(db[3], neginf=-400.0)
    assert row[1:].max() < cfg.min_level_dbfs          # residue -163 dBFS, 43 dB below the floor
    # a tone next to the offset must keep its exact level despite the notch
    _, db2 = eng.block_db((np.sin(2 * np.pi * 1000.0 * np.arange(5 * FS) / FS)
                           - 0.012237).astype(np.float32))
    assert db2[5, 128] == pytest.approx(0.0, abs=0.02)


def test_mean_spectrum_of_plain_floor_offers_no_candidates(tmp_path: Path):
    """R1 at the level that matters: stage 1 must propose nothing on a tone-free floor.

    A single *frame* is a different proposition -- one FFT bin is exponentially distributed, so
    its raw tone-to-noise reaches 18 dB on a tone-free floor (hence ``presence_smooth_s``).
    Stage 1 reads the time-averaged spectrum, where that noise averages away.
    """
    cfg = Config()
    path = tmp_path / "20260101T000000.flac"
    synth.write(path, synth.rumble(FS, 90.0, seed=11), FS)
    agg, eng = aggregate(path, cfg)
    lo = int(cfg.min_freq / eng.df)
    s = np.nan_to_num(mask_below(agg.mean_db, lo), neginf=-400.0, nan=-400.0)[lo:]
    assert float(np.max(s - floor_curve(s, cfg.neighbor_k))) < 6.0
    assert len(candidate_tones(agg, cfg, eng)) == 0


def test_power_from_db_matches_direct_power():
    """Interior bins must agree; DC/Nyquist carry a one-sided convention we never analyse."""
    eng = SpectralEngine(Config(), FS)
    x = _sine(1000.0, 0.1).astype(np.float32)
    _, db = eng.block_db(x)
    n, spec = eng.block_spectrum(x)
    direct = (spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2)
    direct *= (2.0 / eng.win_sum) ** 2
    got = power_from_db(db)
    assert np.allclose(got[:, 1:-1], direct[:, 1:-1], rtol=1e-5, atol=1e-20)
    # the two convention-dependent bins are excluded from detection by min_freq/Nyquist masks
    assert np.allclose(got[:, 1], direct[:, 1], rtol=1e-5)


def test_neighborhood_floor_is_unbiased_for_smooth_spectrum():
    """For a monotone smooth spectrum the centred median equals the centre value."""
    f = np.linspace(10.0, 7900.0, 1025)
    s = -40.0 - 60.0 * np.log10(f / f[0])
    floor = neighborhood_floor(s[None, :], 32)
    assert np.allclose(floor[0, 40:985], s[40:985], atol=0.02)


def test_adaptive_floor_beats_the_one_sided_window():
    """Regression for the bias that made 15-140 Hz look 6-24 dB above its own floor.

    A fixed +-32 window centred on bin 8, with bins <3 masked, holds 37 values that are all
    *above* the knee -- so its median sits 12 dB below the plateau and every plateau bin looks
    like a tone.  Narrowing the window symmetrically (floor_curve) puts the median back on the
    centre value.
    """
    f = np.linspace(0.0, 8000.0, 1025)
    s = np.where(f < 150.0, -54.0, -54.0 - 40.0 * np.log10(np.maximum(f, 150.0) / 150.0))
    s[0] = -38.0                                       # DC spike, as measured
    masked = np.nan_to_num(mask_below(s, 3), nan=-400.0, neginf=-400.0)
    assert floor_curve(masked, 32, first_valid=1)[8] == pytest.approx(-54.0, abs=0.6)
    one_sided = neighborhood_floor(masked, 32)
    assert abs(one_sided[8] - (-54.0)) > 2.0          # the biased version, for the record


def test_no_runtime_warnings_on_silence():
    eng = SpectralEngine(Config(), FS)
    _, db = eng.block_db(np.zeros(3 * FS, dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = np.nan_to_num(db[1], neginf=-400.0)
        neighborhood_floor(row, 32)
