"""A3/A4 tests: transients and impulse trains against synthetic ground truth (FR-4.4/4.5)."""

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.detectors.periodicity import detect_periodicity, periodicity
from audioanalyzer.detectors.sources import group_sources
from audioanalyzer.detectors.transients import band_edges, detect_transients
from audioanalyzer.ingest import probe
from audioanalyzer.synth import add_click, rumble
from audioanalyzer.spectral import aggregate

FS = 16000


def _write(tmp_path: Path, name: str, x: np.ndarray, cfg: Config):
    p = tmp_path / name
    sf.write(p, x, FS, subtype="PCM_16")
    agg, eng = aggregate(p, cfg)
    return p, agg, eng


def test_click_train_found_with_right_period(tmp_path):
    """Ten clicks 7.5 s apart must read as A4 events *and* as a 7.5 s A3 train (FR-4.5)."""
    cfg = Config()
    x = rumble(FS, 80.0, seed=11)
    times = [5.0, 12.5, 20.0, 27.5, 35.0, 42.5, 50.0, 57.5, 65.0, 72.5]
    for t in times:
        x = add_click(x, FS, t, level_dbfs=-20.0, ring_hz=900.0)
    p, agg, eng = _write(tmp_path, "train.flac", x, cfg)
    dets, onsets = detect_transients(p, cfg, agg, eng, media=probe(p, cfg))
    a4 = [d for d in dets if d.cls == "A4"]
    assert len(a4) >= len(times) - 2, f"found {len(a4)} of {len(times)} clicks"
    got = sorted(d.t_start_s for d in a4)
    for t in times[:len(got)]:
        assert any(abs(t - g) < 0.5 for g in got), f"click at {t} s mis-dated: {got}"

    trains = detect_periodicity(onsets, cfg, band_edges(eng, cfg), agg.duration_s, str(p))
    assert trains, f"no periodicity reported from a 10-click train: {onsets}"
    best = min(trains, key=lambda d: abs(d.extra["period_s"] - 7.5))
    assert abs(best.extra["period_s"] - 7.5) < 0.4, best.extra
    assert best.extra["n_onsets"] >= 8
    assert re.match(r"^\d", f"{best.extra['rpm']:.1f}")      # rpm-equivalent is reported


def test_isolated_impulse_not_called_periodic():
    """FR-4.5: two or three hits are not a train — period_min_repeats must hold the line."""
    assert periodicity(np.array([10.0, 17.0, 40.0]), Config(), 60.0) == []
    assert periodicity(np.array([1.0, 2.0]), Config(), 60.0) == []


def test_clipped_frame_is_not_an_event(tmp_path):
    """FR-4.4: clipping is broadband by construction and must not read as a transient."""
    cfg = Config()
    x = rumble(FS, 40.0, seed=13)
    i = int(20.0 * FS)
    x[i:i + 64] = 1.0                                        # 4 ms of flat-top
    p, agg, eng = _write(tmp_path, "clip.flac", x, cfg)
    info = probe(p, cfg)
    assert info.clipped_samples >= 60, info.clipped_samples
    dets, _ = detect_transients(p, cfg, agg, eng, media=info, detail=False)
    assert not [d for d in dets if abs(d.t_start_s - 20.0) < 0.8], \
        "clipped sample reported as a transient"
    dets2, _ = detect_transients(p, cfg, agg, eng, media=None, detail=False)
    assert [d for d in dets2 if abs(d.t_start_s - 20.0) < 0.8], \
        "without the exclusion it should show up — proves the gate is what silenced it"


def test_quiet_recording_yields_no_events(tmp_path):
    """AC-3 on the transient side: corpus-shaped rumble alone must not produce a single A4."""
    cfg = Config()
    p, agg, eng = _write(tmp_path, "quiet.flac", rumble(FS, 90.0, seed=14), cfg)
    dets, _ = detect_transients(p, cfg, agg, eng, media=probe(p, cfg), detail=False)
    assert dets == [], f"{len(dets)} false transients on clean rumble"


def test_event_signature_never_publishes_the_dc_skirt_as_a_frequency(tmp_path):
    """Bin 1 of a 16 kHz/2048 FFT is 7.81 Hz: an ordinary impact whose loudest riser is there has
    no resolvable ring, and must keep its band centre rather than claim a sub-audible frequency."""
    cfg = Config()
    x = rumble(FS, 40.0, seed=19)
    for k in range(6):                      # low-frequency thumps, energy concentrated at DC
        i = int((4.0 + k * 5.0) * FS)
        n = 300
        x[i:i + n] += 0.35 * np.hanning(n) * np.sin(2 * np.pi * 9.0 * np.arange(n) / FS)
    p, agg, eng = _write(tmp_path, "thumps.flac", x, cfg)
    ev, _ = detect_transients(p, cfg, agg, eng, media=probe(p, cfg))
    assert ev, "low-frequency thumps should be detected as events"
    for d in ev:
        assert d.freq_hz >= 2 * FS / 2048, (d.freq_hz, d.extra.get("signature"))


def test_transients_do_not_form_sources(tmp_path):
    """FR-5.1: sources group A1/A2 tone tracks only — an A4 `freq_hz` is a band label.

    Grouping events merged every impact in a day into one "source" with thousands of identical
    harmonics, which is what the first whole-day run printed before this rule existed.
    """
    cfg = Config()
    x = rumble(FS, 60.0, seed=44)
    for t in (5.0, 15.0, 25.0, 35.0, 45.0, 55.0):
        x = add_click(x, FS, t, level_dbfs=-18.0, ring_hz=700.0, ring_ms=40.0)
    p = tmp_path / "clicks.flac"
    sf.write(p, x, FS, subtype="PCM_16")
    agg, eng = aggregate(p, cfg)
    ev, _ = detect_transients(p, cfg, agg, eng, media=probe(p, cfg))
    assert len(ev) >= 4, [d.cls for d in ev]
    srcs = group_sources(ev, cfg)
    assert srcs == [], f"events must not form sources: {[(s.f0_hz, s.n_harmonics) for s in srcs]}"
    assert all(d.source_id == "" for d in ev)


def test_event_signature_names_the_ringing_frequency(tmp_path):
    """FR-4.4 evidence: a bar ringing at 1.2 kHz must report ~1.2 kHz in its signature."""
    cfg = Config()
    x = rumble(FS, 60.0, seed=15)
    for t in (8.0, 16.0, 24.0, 32.0, 40.0, 48.0):
        x = add_click(x, FS, t, level_dbfs=-18.0, ring_hz=1200.0, ring_ms=60.0)
    p, agg, eng = _write(tmp_path, "ring.flac", x, cfg)
    dets, onsets = detect_transients(p, cfg, agg, eng, media=probe(p, cfg))
    a4 = [d for d in dets if d.cls == "A4"]
    assert a4, "no ring events found"
    sigs = [d for d in a4 if d.extra.get("signature")]
    assert sigs, f"no event got a spectral signature: {[d.extra for d in a4[:2]]}"
    tops = [d.extra["signature"][0]["freq_hz"] for d in sigs]
    assert any(abs(f - 1200.0) < 80.0 for f in tops), f"signature peaks {tops} miss 1200 Hz"
