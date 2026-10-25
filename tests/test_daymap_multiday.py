"""The multi-day day map (FR-8.2): hour alignment, height control, artifact-only rendering.

The point of the stacked layout is one geometric claim — the same hour of different days must sit
at the same x — so that is what is asserted, by parsing the painted rects, not by trusting a legend.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import soundfile as sf

from audioanalyzer.cli import main
from audioanalyzer.config import Config
from audioanalyzer.daymap import build_day_map, load_artifact
from audioanalyzer.models import Detection
from audioanalyzer.synth import add_tone, rumble

TZ = timezone(timedelta(hours=8))
FS = 16000


def det(freq, t0, dur, cls="A2", ttn=12.0, z=None, level=-50.0, conf=0.8):
    d = Detection(cls=cls, freq_hz=freq, file="x.flac", level_dbfs=level, tone_noise_db=ttn,
                  coherence=0.9, duty_pct=100.0, confidence=conf)
    d.t_start_wall = t0.isoformat()
    d.t_end_wall = (t0 + timedelta(seconds=dur)).isoformat()
    if z is not None:
        d.extra["z_peak"] = z
    return d


def _rects(svg):
    """(x, y, w, h, fill) of every painted cell."""
    return [(float(a), float(b), float(c), float(d), f) for a, b, c, d, f in
            re.findall(r"<rect x='([\d.]+)' y='([\d.]+)' width='([\d.]+)' height='([\d.]+)' "
                       r"fill='(#[0-9a-f]{6})'>", svg)]


def _panels(svg):
    """Distinct cell y-values, i.e. one entry per (panel, lane)."""
    return sorted({y for _, y, _, _, _ in _rects(svg)})


def test_the_same_hour_of_different_days_lands_on_the_same_x():
    """Hour alignment, measured. Day 1 has a 03:00 event, day 2 an 08:00 event."""
    cfg = Config()
    d1 = datetime(2026, 10, 23, tzinfo=TZ)
    d2 = datetime(2026, 10, 24, tzinfo=TZ)
    dets = [det(500.0, d1 + timedelta(hours=3), 60.0),
            det(500.0, d2 + timedelta(hours=8), 60.0)]
    svg = build_day_map(dets, {"gaps": []}, cfg, by_day=True)
    r = _rects(svg)
    top0 = min(y for _, y, *_ in r)
    top1 = max(y for _, y, *_ in r)
    xs0 = {x for x, y, *_ in r if y == top0}
    xs1 = {x for x, y, *_ in r if y == top1}
    assert xs0 == {96.0 + 180, 96.0 + 181}, "day 1's 03:00 must start at column 180"
    assert xs1 == {96.0 + 480, 96.0 + 481}, "day 2's 08:00 must start at column 480"
    assert top1 > top0, "the two days must not share one row"


def test_panels_are_a_constant_pitch_apart():
    """Stacked panels must repeat at a fixed offset or the hour ruler stops meaning anything."""
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(500.0, base + timedelta(days=i, hours=4), 3600.0) for i in range(4)]
    svg = build_day_map(dets, {"gaps": []}, cfg, by_day=True, lanes=16, lane_h=13.0)
    ys = _panels(svg)
    assert len(ys) == 4, "one frequency paints one lane per panel"
    step = round(ys[1] - ys[0], 3)
    assert step == pytest.approx(16 * 13.0 + 16.0 + 26.0)          # body + tone strip + gap
    assert all(round(b - a, 3) == step for a, b in zip(ys, ys[1:]))


def test_lane_height_and_lane_count_change_the_geometry():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(500.0, base + timedelta(days=i, hours=4), 3600.0) for i in range(2)]
    h = lambda s: float(re.search(r"viewBox='0 0 \d+ (\d+)'", s).group(1))
    small = build_day_map(dets, {"gaps": []}, cfg, by_day=True, lanes=16, lane_h=13.0)
    tall = build_day_map(dets, {"gaps": []}, cfg, by_day=True, lanes=16, lane_h=26.0)
    assert h(tall) - h(small) == pytest.approx(2 * (26.0 - 13.0) * 16, abs=2)
    assert _panels(tall) != _panels(small)
    # 2 day-panels × 1 painted lane each; more lanes must lengthen the canvas, not add rows here
    many = build_day_map(dets, {"gaps": []}, cfg, by_day=True, lanes=24, lane_h=13.0)
    assert len(_panels(many)) == 2
    assert h(many) - h(small) == pytest.approx(2 * 8 * 13.0, abs=2)   # 8 extra lanes × 2 panels


def test_a_multi_day_map_stays_openable():
    """Per-cell tooltips tripled the file; with 30 days that is a megabyte of unused markup."""
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(500.0, base + timedelta(days=i, hours=h), 60.0)
            for i in range(30) for h in range(0, 24, 2)]
    stacked = build_day_map(dets, {"gaps": []}, cfg, by_day=True)
    assert "<title>" not in stacked
    single = build_day_map(dets[:40], {"gaps": []}, cfg, by_day=False)
    assert "<title>" in single, "one-day maps keep their tooltips"


def test_load_artifact_round_trips_json_with_z_and_coverage(tmp_path):
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(128.0, base + timedelta(hours=1), 0.4, cls="A4", ttn=0.0, z=31.5),
            det(65.0, base + timedelta(hours=2), 900.0, cls="A2", ttn=9.0)]
    import json
    payload = {"files": [{"detections": [
        {"cls": d.cls, "freq_hz": d.freq_hz, "t_start_wall": d.t_start_wall,
         "t_end_wall": d.t_end_wall, "tone_noise_db": d.tone_noise_db,
         "level_dbfs": d.level_dbfs, "confidence": d.confidence, "duration_s": d.duration_s,
         **({"z_peak": d.extra["z_peak"]} if "z_peak" in d.extra else {})} for d in dets]}],
        "coverage": {"gaps": [{"at": (base + timedelta(hours=3)).isoformat(), "gap_s": 20.0,
                               "after": "a.flac"}]},
        "tool": {"config": {"transient_sigma": 7.5}}}
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(payload))
    got, cov, cfg2 = load_artifact(p)
    assert len(got) == 2
    assert got[0].cls == "A4" and got[0].extra["z_peak"] == 31.5, "impact severity must survive"
    assert cov["gaps"][0]["gap_s"] == 20.0
    assert cfg2.transient_sigma == 7.5, "the run's own config must be restored, not defaulted"
    svg = build_day_map(got, cov, cfg2, by_day=False)
    assert "NOT RECORDED: 20.0" in svg


def test_load_artifact_from_csv_loses_coverage_and_says_so(tmp_path):
    import csv as _csv
    from audioanalyzer.report import write_csv
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(400.0, base + timedelta(hours=5), 300.0)]
    p = tmp_path / "d.csv"
    write_csv(dets, p)
    got, cov, _ = load_artifact(p)
    assert len(got) == 1 and got[0].freq_hz == pytest.approx(400.0)
    assert got[0].t_start_wall.startswith("2026-10-23T05")
    assert cov == {}


def test_daymap_command_renders_without_touching_audio(tmp_path, monkeypatch):
    """FR-8.2: re-drawing the map must not re-analyse. No FLAC exists in this test at all."""
    import json
    base = datetime(2026, 10, 23, tzinfo=TZ)
    payload = {"files": [{"detections": [
        {"cls": "A2", "freq_hz": f, "t_start_wall": (base + timedelta(days=d, hours=3)).isoformat(),
         "t_end_wall": (base + timedelta(days=d, hours=3, minutes=5)).isoformat(),
         "tone_noise_db": 10.0, "level_dbfs": -50.0, "confidence": 0.7, "duration_s": 300.0}
        for d in range(3) for f in (200.0, 900.0)]}],
        "coverage": {"gaps": []}, "tool": {"config": {}}}
    src = tmp_path / "scan.json"
    src.write_text(json.dumps(payload))
    out = tmp_path / "m.svg"
    from audioanalyzer.scan import scan_file                          # noqa: F401  (guard target)
    import audioanalyzer.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_expand_paths",
                        lambda *a, **k: pytest.fail("daymap re-analysed audio"))
    assert main(["daymap", str(src), "-o", str(out), "--by-day", "--lane-h", "22"]) == 0
    svg = out.read_text()
    assert len(_panels(svg)) == 3 * 2, "three day-panels × two frequencies"
    assert "2026-10-23" in svg and "2026-10-25" in svg, "each panel needs its date"


def test_daymap_command_fails_loudly_on_a_bad_artifact(tmp_path):
    (tmp_path / "junk.json").write_text("{not json")
    assert main(["daymap", str(tmp_path / "junk.json"), "-o", str(tmp_path / "x.svg")]) == 2
    (tmp_path / "empty.json").write_text('{"files": [], "coverage": {}}')
    assert main(["daymap", str(tmp_path / "empty.json"), "-o", str(tmp_path / "x.svg")]) == 1


def test_single_day_input_is_unchanged_by_the_new_default():
    """by_day auto-selects, so a one-day corpus must stay on the wall-clock layout it had before."""
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(500.0, base + timedelta(hours=4), 120.0),
            det(700.0, base + timedelta(hours=9), 120.0)]
    svg = build_day_map(dets, {"gaps": []}, cfg)               # no by_day argument
    assert "x = wall clock" in svg
    ys = _panels(svg)
    from audioanalyzer.daymap import _LANE_H
    assert max(ys) - min(ys) == pytest.approx(_LANE_H), "one panel: adjacent lanes, no stacked pitch"
    xs = sorted({x for x, *_ in _rects(svg)})
    assert 96.0 in xs, "x is measured from the first detection, as it always was"
    assert 96.0 + 300 in xs                                    # 04:00 -> 09:00 is 300 min
