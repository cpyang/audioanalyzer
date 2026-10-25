"""Day-map view filters (FR-8.2b): a frequency window and a severity cut.

Both knobs exist so a map can be *read*, but a filter that silently removes data is worse than no
filter — it turns "I hid that" into "it wasn't there". So the tests below are less about the knobs
working than about the knobs being unable to lie: dropped detections never reappear in a lane they
do not belong to, filtering never changes the colour of a cell it did not touch, and a map filtered
empty says so on its face.
"""

import re
from datetime import datetime, timedelta, timezone

import pytest

from audioanalyzer.cli import main
from audioanalyzer.config import Config
from audioanalyzer.daymap import _LANE_H, build_day_map
from audioanalyzer.models import Detection

TZ = timezone(timedelta(hours=8))


def det(freq, t0, dur=60.0, cls="A2", ttn=12.0, z=None, conf=0.8):
    d = Detection(cls=cls, freq_hz=freq, file="x.flac", level_dbfs=-50.0, tone_noise_db=ttn,
                  coherence=0.9, duty_pct=80.0, confidence=conf)
    d.t_start_wall = t0.isoformat()
    d.t_end_wall = (t0 + timedelta(seconds=dur)).isoformat()
    if z is not None:
        d.extra["z_peak"] = z
    return d


def _cells(svg):
    """{(x, y): fill} for painted cells only — 1 px wide, one lane tall."""
    return {((float(a), float(b))): f for a, b, c, d, f in
            re.findall(r"<rect x='([\d.]+)' y='([\d.]+)' width='([\d.]+)' height='([\d.]+)' "
                       r"fill='(#[0-9a-f]{6})'>", svg)
            if float(c) == 1.0 and abs(float(d) - _LANE_H) < 0.51}


def _lane_of(svg, y):
    rows = sorted({yy for _, yy in _cells(svg)})
    return rows.index(y)


def test_a_low_cut_drops_sub_band_sound_instead_of_stacking_it_in_lane_zero():
    """The bug this guards: np.clip() on the lane index would paint every low sound in lane 0."""
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(30.0, base, ttn=25.0), det(45.0, base + timedelta(minutes=1), ttn=25.0),
            det(2000.0, base + timedelta(minutes=2), ttn=25.0)]
    svg = build_day_map(dets, {"gaps": []}, cfg, freq_lo=1000.0, lanes=8, lane_h=18.0)
    cells = _cells(svg)
    # Anti-clamp, computed: with a 1 kHz window the lanes are geomspace(1000, 7600, 9), so 2000 Hz
    # belongs to a specific lane — and the 30/45 Hz detections must not appear anywhere, least of
    # all in lane 0, which is exactly what np.clip on the index would have produced.
    import numpy as np
    edges = np.geomspace(1000.0, 7600.0, 9)
    lane = int(np.searchsorted(edges, 2000.0, side="right") - 1)
    assert lane > 0, "test setup: 2000 Hz should not be in the bottom lane"
    assert {y for _, y in cells} == {42.0 + lane * 18.0}, "out-of-band sound leaked into a lane"
    assert "1000-7600 Hz" in svg


def test_a_high_cut_behaves_the_same_way():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(6000.0, base), det(700.0, base + timedelta(minutes=1))]
    svg = build_day_map(dets, {"gaps": []}, cfg, freq_hi=1000.0, lanes=8)
    assert len(_cells(svg)) == 1


def test_the_window_is_stated_on_the_map_and_in_the_accessibility_label():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    svg = build_day_map([det(500.0, base)], {"gaps": []}, cfg, freq_lo=100.0, freq_hi=2000.0,
                        min_severity=20.0)
    assert "100-2000 Hz" in svg
    assert "≥20 over floor" in svg
    assert "100-2000 Hz" in re.search(r"aria-label='([^']*)'", svg).group(1)


def test_the_cut_hides_quiet_cells_without_recolouring_the_ones_it_keeps():
    """Colour is absolute dB-over-floor. A view filter must not re-scale what it leaves on screen,
    or turning the cut up would make the same event look louder."""
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(500.0, base, ttn=21.0), det(1500.0, base, ttn=5.0)]
    plain = _cells(build_day_map(dets, {"gaps": []}, cfg, lanes=8))
    cut = _cells(build_day_map(dets, {"gaps": []}, cfg, lanes=8, min_severity=20.0))
    assert len(plain) == 2 and len(cut) == 1             # the 5 dB track is hidden
    assert all(plain[k] == v for k, v in cut.items()), "a filter changed a kept cell's colour"
    assert set(cut) <= set(plain), "the filter invented a position"


def test_the_map_counts_what_it_filtered_out():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(30.0, base), det(500.0, base + timedelta(minutes=1), ttn=3.0),
            det(900.0, base + timedelta(minutes=2), ttn=25.0)]
    st = {}
    svg = build_day_map(dets, {"gaps": []}, cfg, freq_lo=100.0, min_severity=10.0, stats=st)
    assert st["dropped_freq"] == 1 and st["dropped_sev"] == 1
    assert "filtered out: 1 outside the band, 1 below the cut" in svg


def test_drop_counts_are_totals_not_per_panel_sums():
    """A 4-panel map once reported 54,784 detections dropped from a 14,660-detection file.

    The band/severity tests run before the day test inside _grid_for, so every panel counted every
    out-of-band detection. The headline count has to be a count OF DETECTIONS, or the map's own
    honesty note becomes the thing that misleads.
    """
    cfg = Config()
    dets = [det(30.0, datetime(2026, 10, d + 1, 3, tzinfo=TZ)) for d in range(4)]
    st = {}
    build_day_map(dets, {"gaps": []}, cfg, by_day=True, freq_lo=1000.0, stats=st)
    assert st["dropped_freq"] == 4 == st["detections"], st
    assert st["cells"] == 0


def test_a_map_filtered_empty_says_so_instead_of_looking_like_a_quiet_site():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    dets = [det(30.0, base), det(40.0, base + timedelta(minutes=5))]
    svg = build_day_map(dets, {"gaps": []}, cfg, freq_lo=1000.0)
    assert not _cells(svg)
    assert "EMPTY AFTER FILTERS" in svg and "not a" in svg.lower() or "view setting" in svg


def test_an_impossible_window_is_an_error_not_a_blank_page():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    with pytest.raises(ValueError):
        build_day_map([det(500.0, base)], {"gaps": []}, cfg, freq_lo=4000.0, freq_hi=100.0)


def test_an_event_above_nyquist_is_dropped_not_wrapped():
    cfg = Config()
    base = datetime(2026, 10, 23, tzinfo=TZ)
    svg = build_day_map([det(9000.0, base)], {"gaps": []}, cfg, lanes=8)
    assert len(_cells(svg)) == 0, "a detection above the analysis ceiling wrapped into a real lane"


def test_cli_window_and_cut_report_their_effect(tmp_path, capsys):
    import json
    base = datetime(2026, 10, 23, tzinfo=TZ)
    rows = [{"cls": "A2", "freq_hz": f, "tone_noise_db": t, "level_dbfs": -50.0, "confidence": 0.7,
             "t_start_wall": (base + timedelta(minutes=i)).isoformat(),
             "t_end_wall": (base + timedelta(minutes=i, seconds=60)).isoformat(),
             "duration_s": 60.0}
            for i, (f, t) in enumerate([(30.0, 25.0), (600.0, 4.0), (900.0, 25.0)])]
    payload = {"files": [{"detections": rows}],
               "coverage": {"gaps": []}, "tool": {"config": {}}}
    src = tmp_path / "scan.json"
    src.write_text(json.dumps(payload))
    assert main(["daymap", str(src), "-o", str(tmp_path / "m.svg"), "--map-lo", "100",
                 "--map-cut", "10"]) == 0
    out = capsys.readouterr().out
    assert "dropped 1 outside the band, 1 under the cut" in out
    assert "1 of 3 detections drawn" in out


def test_cli_rejects_an_inverted_window_before_any_analysis(tmp_path):
    rc = main(["daymap", str(tmp_path / "nope.json"), "--map-lo", "4000", "--map-hi", "100"])
    assert rc == 2                       # unreadable artifact is reported first, and cheaply
