"""Seeding known.toml from always-present sound (FR-7.3).

The property that matters most is the negative one: a floor that is merely *tilted* must produce
no candidates.  The reference corpus floor falls ~5.5 dB/octave, and the first version of the ridge
trend used a window median, which reads a slope as a bump near a knee.
"""

import numpy as np
import pytest
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.seed import SeedCandidate, _ridge_peaks, promote, seed_candidates, write_candidates
from audioanalyzer.synth import add_tone, rumble

FS = 16000


class _Agg:
    """Just enough of an Aggregate for _ridge_peaks."""

    def __init__(self, spec):
        self._spec = np.asarray(spec, dtype=float)

    def percentile(self, q):
        return self._spec


class _Eng:
    def __init__(self, freqs):
        self.freqs = np.asarray(freqs, dtype=float)


def _spectrum(freqs, level_at=lambda f: -60.0 - 5.5 * np.log2(np.maximum(f, 1.0) / 20.0)):
    s = level_at(np.asarray(freqs, dtype=float))
    s[0] = -38.0                        # a DC offset, which must never look like a feature
    s[1] = np.nan                       # masked bin
    return s


def test_a_pure_power_law_floor_produces_no_ridges():
    f = np.linspace(0.0, 8000.0, 1025)
    agg, eng = _Agg(_spectrum(f)), _Eng(f)
    got = _ridge_peaks(agg, eng, Config(), ridge_db=4.0, octaves=1.0)
    assert got == [], [(g["freq_hz"], g["over_floor_db"]) for g in got]


def test_a_real_bump_on_a_tilted_floor_is_found():
    f = np.linspace(0.0, 8000.0, 1025)

    def lvl(x):
        base = -60.0 - 5.5 * np.log2(np.maximum(x, 1.0) / 20.0)
        # ~0.4 octave wide, like the measured 125 Hz feature.  A bump only 1-2 bins wide would be
        # removed by the 3-bin median on purpose: single-bin spikes are spectral noise, not
        # permanent structure (that filter is what stopped the anti-alias knee becoming 16 rows).
        return base + 9.0 * np.exp(-0.5 * ((x - 1000.0) / 40.0) ** 2)

    agg, eng = _Agg(_spectrum(f, lvl)), _Eng(f)
    got = _ridge_peaks(agg, eng, Config(), ridge_db=4.0, octaves=1.0)
    assert len(got) == 1, got
    assert abs(got[0]["freq_hz"] - 1000.0) < 20.0
    assert 6.0 <= got[0]["over_floor_db"] <= 11.0, got[0]
    assert got[0]["band_hz"][0] < 1000.0 < got[0]["band_hz"][1]


def test_band_edge_is_not_reported_as_a_feature():
    """A one-sided fit window at 20 Hz would call the entire LF shelf a ridge."""
    f = np.linspace(0.0, 8000.0, 1025)
    agg, eng = _Agg(_spectrum(f)), _Eng(f)
    low = [g for g in _ridge_peaks(agg, eng, Config(), ridge_db=1.0,
                                   octaves=1.0) if g["freq_hz"] < 3.0 * Config().min_freq]
    assert low == [], low


def _hours(tmp_path, n, tones=(), steady_level=-46.0):
    paths = []
    for i in range(n):
        x = rumble(FS, 40.0, seed=200 + i)
        for fc in tones:
            x = add_tone(x, FS, fc, steady_level, 0.0, 40.0)
        p = tmp_path / f"20261023T{i:02d}0002.flac"
        sf.write(p, x, FS, subtype="PCM_16")
        paths.append(p)
    return paths


def test_an_antialias_knee_is_not_a_source():
    """What the first seeder actually produced on a real day: 16 entries, 6242-7203 Hz, -97 dBFS.

    Above the corner the spectrum is flat and low-SNR, so every bin is a strict local maximum of
    the raw curve and reads as +4..9 dB against a straight line fitted across the knee.  A knee has
    no crest in the residual, which is what distinguishes it from a feature.
    """
    f = np.linspace(0.0, 8000.0, 1025)

    def lvl(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 6000.0, -60.0 - 5.5 * np.log2(np.maximum(x, 1.0) / 20.0),
                        -97.0 + 0.6 * np.sin(x / 37.0))     # flat, wiggling HF floor

    agg, eng = _Agg(_spectrum(f, lvl)), _Eng(f)
    got = _ridge_peaks(agg, eng, Config(), ridge_db=4.0, octaves=1.0)
    assert [g for g in got if g["freq_hz"] > 5000.0] == [], \
        [(g["freq_hz"], round(g["over_floor_db"], 1)) for g in got]


def test_a_smooth_shelf_is_not_a_source():
    """Same failure class: a monotone slope change has no crest to point at."""
    f = np.linspace(0.0, 8000.0, 1025)

    def lvl(x):
        x = np.asarray(x, dtype=float)
        return -55.0 - 3.0 * np.log2(np.maximum(x, 20.0) / 20.0) \
            * (1.0 + 1.4 * (x > 150.0))                     # knee at 150 Hz, no bump

    agg, eng = _Agg(_spectrum(f, lvl)), _Eng(f)
    assert _ridge_peaks(agg, eng, Config(), ridge_db=4.0, octaves=1.0) == []


def test_a_line_present_in_every_file_becomes_a_candidate(tmp_path):
    paths = _hours(tmp_path, 3, tones=(315.0,))
    cands = seed_candidates(paths, Config(), min_file_frac=0.6)
    line = [c for c in cands if c.kind == "line" and abs(c.freq_hz - 315.0) < 3.0]
    assert line, [(c.kind, c.freq_hz) for c in cands]
    assert line[0].n_files == 3 and line[0].files_total == 3
    assert line[0].duty_pct == pytest.approx(100.0, abs=6.0), "a steady tone is 'all the time'"
    assert line[0].always_score > 0.8


def test_a_one_off_noise_is_not_offered_as_background(tmp_path):
    """The filter that makes this a *known-noise* tool rather than a findings dump."""
    paths = _hours(tmp_path, 4)
    x = add_tone(rumble(FS, 40.0, seed=99), FS, 1777.0, -44.0, 0.0, 40.0)
    sf.write(tmp_path / "20261023T090002.flac", x, FS, subtype="PCM_16")
    cands = seed_candidates(paths + [tmp_path / "20261023T090002.flac"], Config(),
                            min_file_frac=0.6)
    assert not [c for c in cands if abs(c.freq_hz - 1777.0) < 3.0], \
        "1 of 5 files is an incident, not site background"


def test_no_fundamental_is_invented(tmp_path):
    """A generated library that claimed a 5 Hz machine would poison every later report."""
    c = SeedCandidate(freq_hz=100.0, kind="line", n_files=5, files_total=5, level_dbfs=-50.0,
                      over_floor_db=9.0, duty_pct=97.0, harmonics=[1])
    assert c.freq_hz == 100.0                     # f0 is the observed line, always
    assert "AUTO-CANDIDATE" in c.note()


def test_candidates_file_is_valid_toml_and_promotion_is_human_gated(tmp_path):
    paths = _hours(tmp_path, 2, tones=(250.0,))
    cands = seed_candidates(paths, Config(), min_file_frac=0.5)
    assert cands
    cf = write_candidates(tmp_path / "cand.toml", cands)
    from audioanalyzer._toml import tomllib
    doc = tomllib.loads(cf.read_text())
    assert len(doc["source"]) == len(cands)
    live = tmp_path / "known.toml"
    names = promote(cf, live, [1], {1: "CRH-3 fan"})
    assert names == ["CRH-3 fan"]
    assert "CRH-3 fan" in live.read_text()
    assert promote(cf, live, [1]) == []           # already there: promoting is idempotent
    with pytest.raises(IndexError):
        promote(cf, live, [999])


def test_seeding_is_silent_when_nothing_is_always_there(tmp_path):
    paths = _hours(tmp_path, 2)                   # rumble only, no injected line
    cands = [c for c in seed_candidates(paths, Config(), min_file_frac=0.99) if c.kind == "line"]
    assert cands == [], [(c.freq_hz, c.n_files) for c in cands]


def test_silence_comes_with_an_explanation(tmp_path):
    """"No candidates" must be distinguishable from a broken seeder, without --verbose."""
    paths = _hours(tmp_path, 3, tones=(222.0,))
    st: dict = {}
    seed_candidates(paths, Config(), min_file_frac=0.99, stats=st)   # threshold nothing can meet
    assert st["files_total"] == 3
    assert st["examined"]["line"] >= 1, "the tone was seen; it was the agreement that failed"
    assert st["best_file_frac"]["line"] <= 1.0
    assert st["per_file_counts"]["lines"], "per-file counts are what prove the probe ran"
