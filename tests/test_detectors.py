"""Tonal detector tests against a realistic, measured noise floor (AC-2, AC-5, AC-9).

Every fixture here is ``synth.rumble`` — a spectrum fitted to the mean spectrum actually
measured on the corpus — so detectors face a steep, non-white floor rather than the white
noise that would let a bad threshold look fine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from audioanalyzer import synth
from audioanalyzer.config import Config
from audioanalyzer.scan import scan_file

FS = 16000


def _fixture(tmp_path: Path, seconds: float, tones=(), name: str = "20260101T000000.flac",
            seed: int = 5) -> Path:
    x = synth.rumble(FS, seconds, seed=seed)
    for spec in tones:
        freq, level, t0, t1 = spec
        x = synth.add_tone(x, FS, freq, level, t0, t1)
    path = tmp_path / name
    synth.write(path, x, FS)
    return path


def test_no_detections_on_plain_floor(tmp_path: Path):
    """Precision-first: a realistic floor with no tone must yield nothing (AC-4 FP budget)."""
    path = _fixture(tmp_path, 120.0)
    res = scan_file(path, Config())
    assert res.detections == [], [(d.freq_hz, d.tone_noise_db) for d in res.detections]


def test_steady_tone_above_floor_is_found(tmp_path: Path):
    """A1 at 7 kHz, -70 dBFS over a -96 dBFS floor: found, correct, and *steady*."""
    path = _fixture(tmp_path, 300.0, tones=[(7000.0, -70.0, 0.0, None)])
    res = scan_file(path, Config())
    hits = [d for d in res.detections if abs(d.freq_hz - 7000.0) < 1.5]
    assert len(hits) == 1, [(d.freq_hz, d.tone_noise_db) for d in res.detections]
    d = hits[0]
    assert d.cls == "A1"
    assert d.tone_noise_db > 20.0
    assert d.level_dbfs == pytest.approx(-70.0, abs=1.5)
    assert d.persistence_pct > 95.0
    assert d.n_episodes == 1
    assert d.drift_class == "stable"
    assert d.level_std_db < 2.0


@pytest.mark.parametrize("freq", [50.0, 62.5, 120.0, 1000.0, 3500.0, 6900.0, 7600.0])
def test_tone_found_across_the_band(tmp_path: Path, freq):
    """Detection must not depend on where in 0-8 kHz the tone sits.

    The tone is injected 12 dB above the *local* floor rather than at a fixed dBFS, because
    this corpus's rumble is -53 dBFS per bin at 50 Hz -- a -60 dBFS tone down there is below
    the noise, and no honest detector should claim to find it.
    """
    level = synth._db_at(freq) + 12.0
    path = _fixture(tmp_path, 180.0, tones=[(freq, level, 0.0, None)])
    res = scan_file(path, Config())
    hits = [d for d in res.detections if abs(d.freq_hz - freq) < 1.0]
    assert hits, f"no detection at {freq} Hz; got {[(d.freq_hz, d.tone_noise_db) for d in res.detections]}"
    best = max(hits, key=lambda d: d.tone_noise_db)
    assert best.tone_noise_db >= 6.0
    assert best.level_dbfs == pytest.approx(level, abs=2.5)


def test_nyquist_guard_drops_the_resampling_line(tmp_path: Path):
    """R1/R3: a 44.1k→16k SoX resample leaves a coherent line *at* Nyquist.

    It scores phase coherence ~0.97 — the same score a real motor whine scores — so no
    detector gate can tell them apart.  The honest answer is to not claim anything in the
    last 2% of the band, and to say so, rather than to report an artefact of the capture
    chain as a machine on site.
    """
    freq = 7950.0
    level = synth._db_at(freq) + 12.0
    path = _fixture(tmp_path, 120.0, tones=[(freq, level, 0.0, None)])
    res = scan_file(path, Config())
    assert not [d for d in res.detections if d.freq_hz > 16000 / 2 * 0.97], \
        "near-Nyquist line reported as a tone"
    # it is there in the audio, so the exclusion is the guard and not a missed detection
    shown = scan_file(path, Config(nyquist_guard=0.9999))
    assert [d for d in shown.detections if abs(d.freq_hz - freq) < 3.0], \
        "guard test is vacuous: the tone was not detected even with the guard open"


def test_intermittent_tone_becomes_a2_with_episodes(tmp_path: Path):
    """On 20 s / off 20 s -> A2, three episodes, ~1/3 duty (FR-4.2)."""
    tone = [(4000.0, -60.0, s, s + 20.0) for s in (0.0, 40.0, 80.0)]
    path = _fixture(tmp_path, 120.0, tones=tone)
    res = scan_file(path, Config())
    hits = [d for d in res.detections if abs(d.freq_hz - 4000.0) < 1.5]
    assert len(hits) == 1
    d = hits[0]
    assert d.cls == "A2"
    assert d.n_episodes == 3, d.extra["episodes"]
    # 3 x 20 s on / 20 s off over 120 s = 50% duty; temporal smoothing adds the few tenths
    assert 45.0 <= d.duty_pct <= 60.0, d.duty_pct


def test_emerging_tone_found_by_block_means(tmp_path: Path):
    """A tone that starts at 300 s is invisible in the whole-file mean; block means find it."""
    path = _fixture(tmp_path, 600.0, tones=[(2500.0, -65.0, 300.0, None)])
    res = scan_file(path, Config())
    hits = [d for d in res.detections if abs(d.freq_hz - 2500.0) < 1.5]
    assert len(hits) == 1
    assert hits[0].t_start_s > 295.0, hits[0].extra.get("episodes")


def test_slow_drift_reports_slope(tmp_path: Path):
    """A motor that speeds up by 6 Hz/min is exactly what drift detection is for (FR-4.3)."""
    secs, slope = 300.0, 0.02                       # Hz/s -> 1.2 Hz/min -> 6 Hz total
    t = np.arange(int(FS * secs)) / FS
    x = synth.rumble(FS, secs, seed=9) + 10 ** (-60.0 / 20.0) * np.sin(
        2 * math.pi * (1000.0 * t + 0.5 * slope * t * t))
    path = tmp_path / "20260101T000000.flac"
    synth.write(path, x, FS)
    dets = scan_file(path, Config()).detections
    hits = [d for d in dets if 995.0 <= d.freq_hz <= 1010.0]
    assert hits, [(d.freq_hz, d.tone_noise_db) for d in dets]
    d = max(hits, key=lambda x_: x_.tone_noise_db)
    assert d.drift_hz_per_min == pytest.approx(1.2, abs=0.7)
    assert d.drift_r2 > 0.8
    assert d.drift_class == "drifting"


def test_fast_sweep_is_not_reported_as_a_steady_tone(tmp_path: Path):
    """Documented limitation, tested so it cannot regress into a silent false claim.

    A tone sweeping 60 Hz in 60 s is broadband in any time average, so A1/A2 must not present
    it as a steady line.  Sweeps are A4/report territory (M3+); the honest outcome here is
    "not a tonal detection".
    """
    secs = 60.0
    t = np.arange(int(FS * secs)) / FS
    x = synth.rumble(FS, secs, seed=21) + 10 ** (-60.0 / 20.0) * np.sin(
        2 * math.pi * (1000.0 * t + 0.5 * 1.0 * t * t))       # 1000 -> 1030 Hz
    path = tmp_path / "20260101T000000.flac"
    synth.write(path, x, FS)
    dets = scan_file(path, Config()).detections
    steady = [d for d in dets if d.cls == "A1" and 990.0 <= d.freq_hz <= 1040.0
              and d.persistence_pct > 90.0]
    assert not steady, [(d.freq_hz, d.cls, d.persistence_pct) for d in steady]


def test_harmonic_stack_is_not_reported_as_one_wide_feature(tmp_path: Path):
    """50/100/150 Hz tones are separate features; A5 groups them into a source in M2."""
    tones = [(f, synth._db_at(f) + 15.0, 0.0, None) for f in (50.0, 100.0, 150.0)]
    path = _fixture(tmp_path, 180.0, tones=tones)
    res = scan_file(path, Config())
    found = [d.freq_hz for d in res.detections if d.tone_noise_db >= 6.0]
    for target in (50.0, 100.0, 150.0):
        assert any(abs(f - target) < 1.2 for f in found), f"missing {target} Hz in {found}"


def test_threshold_sweep_is_monotonic(tmp_path: Path):
    """AC-9: relaxing the gate can only add detections, never remove them."""
    path = _fixture(tmp_path, 240.0, tones=[(7000.0, synth._db_at(7000.0) + 28.0, 0.0, None),
                                           (3000.0, synth._db_at(3000.0) + 14.0, 0.0, None),
                                           (1200.0, synth._db_at(1200.0) + 7.0, 0.0, None)])
    counts = []
    for gate in (3.0, 6.0, 12.0, 25.0):
        cfg = Config(tone_noise_db=gate)
        counts.append(len(scan_file(path, cfg).detections))
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] >= 3 and counts[-1] <= 1


def test_dc_offset_never_becomes_a_tone(tmp_path: Path):
    """AC-1 on a realistic signal: the -38 dBFS DC offset yields no bin-0 detection."""
    path = _fixture(tmp_path, 120.0)
    res = scan_file(path, Config())
    assert all(d.freq_hz >= 20.0 for d in res.detections)
    kinds = {f["kind"] for f in res.media.quality_flags(Config())}
    assert "dc_offset" in kinds


def test_recall_flag_adds_detections(tmp_path: Path):
    """§10 decision 3: precision-first by default, --recall relaxes it."""
    path = _fixture(tmp_path, 240.0, tones=[(5500.0, synth._db_at(5500.0) + 5.0, 0.0, None)])
    strict = scan_file(path, Config())
    from audioanalyzer.config import recall_preset
    relaxed = scan_file(path, recall_preset(Config()))
    assert len(strict.detections) == 0, [d.tone_noise_db for d in strict.detections]
    assert len(relaxed.detections) >= 1


def test_synth_floor_matches_measured_corpus_envelope():
    """The fixture is only meaningful if it reproduces the measured spectrum (§2)."""
    x = synth.rumble(FS, 120.0, seed=13)
    measured = {312.5: -75.8, 625.0: -81.3, 1250.0: -86.9, 2500.0: -92.3, 5000.0: -96.4}
    for freq, want in measured.items():
        assert synth._bin_level(x, FS, freq) == pytest.approx(want, abs=1.5), freq
    assert abs(synth._bin_level(x, FS, 62.5) - (-53.5)) < 2.5
    assert 20 * math.log10(abs(float(x.mean()))) == pytest.approx(-38.25, abs=0.5)
