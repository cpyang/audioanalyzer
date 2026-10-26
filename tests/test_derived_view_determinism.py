"""The derived views must be a function of the findings, not of the order they arrived in (NFR-6).

Two bugs live here. `group_sources` seeds greedy clustering from pool[0], so the source table used to
change with input order — which the scan hid (one order) and merge/split/report exposed (another).
And the artifact loader silently dropped any row key it did not recognise, so an artifact-derived
report lost each impact's `signature` and `nearest` while the scan's own report kept them.
"""

import json
import random

import pytest

from audioanalyzer.artifact import load_run
from audioanalyzer.config import Config
from audioanalyzer.detectors.sources import group_sources
from audioanalyzer.daymap import _from_json
from audioanalyzer.models import Detection
from audioanalyzer.report import detections_json


def _tone(file, t, freq, cls="A2", duty=50.0, conf=0.4, episodes=None):
    d = Detection(cls=cls, file=file, freq_hz=freq, t_start_s=t, t_end_s=t + 8.0,
                  duration_s=8.0, level_dbfs=-60.0, peak_dbfs=-52.0, tone_noise_db=12.0,
                  duty_pct=duty, confidence=conf)
    if episodes is not None:
        d.extra["episodes"] = episodes
    return d


def _table(dets):
    return sorted((round(s.f0_hz, 6), s.n_harmonics,
                   tuple(sorted(round(m.freq_hz, 4) for m in s.members)))
                  for s in group_sources(dets, Config()))


def test_source_table_does_not_depend_on_input_order():
    dets = [_tone("20261023T000002.flac", 600.0 + i * 300, f,
                  episodes=[(600.0 + i * 300, 90.0)], duty=20.0 + i)
            for i, f in enumerate((65.37, 554.95, 24.48, 26.77, 69.45, 1318.5))]
    base = _table(dets)
    assert base, "no sources clustered at all; the fixture is not exercising the linker"
    for seed in range(8):
        shuffled = list(dets)
        random.Random(seed).shuffle(shuffled)
        assert _table(shuffled) == base, f"clustering changed with input order (seed {seed})"


def test_source_table_is_stable_across_a_round_trip(tmp_path):
    """What the scan records must be what a report re-derives from the scan's own file."""
    dets = [_tone("20261023T000002.flac", 600.0 + i * 300, f,
                  episodes=[(600.0 + i * 300, 90.0)], duty=25.0 + i)
            for i, f in enumerate((65.37, 554.95, 24.48, 69.45))]
    group_sources(dets, Config())                       # annotate, as the scan does
    rows = detections_json(dets)
    media = {"path": "20261023T000002.flac", "format": "FLAC", "subtype": "PCM_16",
             "channels": 1, "sample_rate": 16000, "frames": 4000, "duration_s": 2.5,
             "size_bytes": 4096, "capture_start": "2026-10-23T00:00:02+08:00", "error": None}
    doc = {"tool": {"version": "test", "config": Config().to_dict()},
           "files": [{"file": media["path"], "media": media,
                      "analysis": {}, "quality_flags": [], "detections": rows}]}
    (tmp_path / "s.json").write_text(json.dumps(doc))
    assert _table(load_run(tmp_path / "s.json").dets) == _table(dets)


def test_loader_keeps_row_keys_it_does_not_recognise():
    """`signature` and `nearest` are per-detection extras with no field and no whitelist entry; the
    loader used to fall off the end of its branch chain and lose them."""
    d = _tone("x.flac", 10.0, 1000.0)
    d.extra["signature"] = {"rise_db": 3.5, "bands_hot": [2, 3]}
    d.extra["nearest"] = {"freq_hz": 1005.0, "dt_s": 0.25}
    row = detections_json([d])[0]
    back = _from_json(row)
    assert back.extra["signature"] == d.extra["signature"], "impact signature lost on reload"
    assert back.extra["nearest"] == d.extra["nearest"], "nearest-neighbour link lost on reload"
    assert detections_json([back])[0] == row, "load -> save is not a fixed point"
