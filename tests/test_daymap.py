"""The day map (FR-8.2): marks, not pixels — and why that choice is load-bearing.

The alternative (a day-long spectrogram) was measured and rejected: at 900 columns a day is 96 s
per pixel, so the 15.26 s tone that was the strongest finding of the reference day occupies 0.16 of
a pixel.  These tests pin the property that motivated the design — a short event must appear.
"""

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import soundfile as sf

from audioanalyzer.cli import main
from audioanalyzer.config import Config
from audioanalyzer.daymap import build_day_map, _colour
from audioanalyzer.models import Detection
from audioanalyzer.synth import add_tone, rumble

TZ = timezone(timedelta(hours=8))
FS = 16000


def det(freq, t0, dur, cls="A2", ttn=12.0, z=None, level=-50.0, conf=0.8, name=""):
    d = Detection(cls=cls, freq_hz=freq, file="x.flac", level_dbfs=level, tone_noise_db=ttn,
                  coherence=0.9, duty_pct=100.0 if dur > 60 else 0.0, confidence=conf,
                  known_name=name)
    d.t_start_wall = t0.isoformat()
    d.t_end_wall = (t0 + timedelta(seconds=dur)).isoformat()
    if z is not None:
        d.extra["z_peak"] = z
    return d


def test_a_fifteen_second_event_actually_appears():
    """The whole justification for marks-over-pixels, asserted rather than argued."""
    cfg = Config()
    base = datetime(2026, 10, 23, 0, 0, tzinfo=TZ)
    short = det(1318.5, base + timedelta(hours=13, minutes=30), 15.26, ttn=29.4)
    svg = build_day_map([short], {"gaps": []}, cfg)
    rects = re.findall(r"<rect x='([\d.]+)' y='[\d.]+' width='1.0' height='[\d.]+'", svg)
    assert len(rects) >= 1, "a 15 s event vanished from the day map"
    assert "1318.5Hz" in svg, "top tone tracks must be labelled, not just coloured"


def test_long_and_short_events_share_the_page_legibly():
    cfg = Config()
    base = datetime(2026, 10, 23, 0, 0, tzinfo=TZ)
    dets = [det(65.1, base, 3000.0, ttn=9.0),
            det(1318.5, base + timedelta(hours=13, minutes=30), 15.26, ttn=29.4)]
    svg = build_day_map(dets, {"gaps": []}, cfg)
    assert svg.count("<rect") >= 50                    # the sustained tone tiles its minutes
    assert "65.1Hz" in svg and "1318.5Hz" in svg


def test_unrecorded_time_is_drawn_not_ignored():
    """R10: a blank stretch must never read as 'nothing happened'."""
    cfg = Config()
    base = datetime(2026, 10, 23, 0, 0, tzinfo=TZ)
    dets = [det(417.0, base + timedelta(hours=5), 600.0)]
    cov = {"gaps": [{"at": (base + timedelta(hours=7, minutes=30)).isoformat(),
                     "gap_s": 26.7, "after": "20261023T070002.flac"}]}
    svg = build_day_map(dets, cov, cfg)
    assert "url(#dm-norecorded)" in svg
    assert "NOT RECORDED: 26.7 s" in svg
    assert "dm-norecorded'" in svg and "pattern" in svg


def test_standalone_svg_declares_namespace_and_inline_does_not():
    """A .svg file needs xmlns; inline HTML must not gain an http:// reference (FR-8.2)."""
    cfg = Config()
    d = [det(500.0, datetime(2026, 10, 23, 1, 0, tzinfo=TZ), 120.0)]
    alone = build_day_map(d, {"gaps": []}, cfg, standalone=True)
    inline = build_day_map(d, {"gaps": []}, cfg, standalone=False)
    assert "http://www.w3.org/2000/svg" in alone
    assert "http" not in inline


def test_colour_is_monotonic_in_severity():
    """Rec.601 luma must increase with severity — the first ramp failed this at the top end."""
    vs = [0, 6, 12, 18, 26, 34]
    lum = [sum(int(_colour(v)[i:i + 2], 16) * w for i, w in ((1, .299), (3, .587), (5, .114)))
           for v in vs]
    assert all(b > a for a, b in zip(lum, lum[1:])), [f"{v:+.1f}" for v in
                                                      np.diff(lum)]


def test_map_says_so_when_there_is_no_clock():
    cfg = Config()
    d = Detection(cls="A2", freq_hz=100.0, file="x.flac", tone_noise_db=8.0)   # undated
    svg = build_day_map([d], {"gaps": []}, cfg)
    assert "Day map unavailable" in svg and "not evidence" not in svg


def test_impact_lane_uses_z_and_tone_lane_uses_over_floor():
    """A4's freq_hz is a band centre; its severity is z, not tone-to-noise."""
    cfg = Config()
    base = datetime(2026, 10, 23, 0, 0, tzinfo=TZ)
    ev = det(128.0, base + timedelta(hours=2), 0.5, cls="A4", ttn=0.0, z=31.0)
    svg = build_day_map([ev], {"gaps": []}, cfg)
    assert "31.0 over floor" in svg or "31.0" in svg


def test_report_embeds_the_day_map_for_a_real_selection(tmp_path):
    cfg = Config()
    for h in (1, 2):
        x = add_tone(rumble(FS, 30.0, seed=300 + h), FS, 400.0 + h, -45.0, 0.0, 30.0)
        sf.write(tmp_path / f"20261023T{h:02d}0002.flac", x, FS, subtype="PCM_16")
    out = tmp_path / "r.html"
    assert main(["scan", str(tmp_path), "--report", str(out), "--quiet"]) == 0
    text = out.read_text()
    assert "Day map" in text and "viewBox" in text
    assert "http://" not in text                       # still self-contained
