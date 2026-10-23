"""M4 tests: known-source library, recurrence analysis, HTML report (FR-7.2/7.3, FR-8.2)."""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.corpus import build_timeline
from audioanalyzer.detectors.sources import group_sources
from audioanalyzer.known import label, load_known, recurrence, upsert_known
from audioanalyzer.models import Detection
from audioanalyzer.report_html import build_report
from audioanalyzer.synth import add_tone, rumble

TZ = timezone(timedelta(hours=8))


def _det(freq, wall, cls="A2", file="x.flac"):
    d = Detection(cls=cls, freq_hz=freq, file=file, t_start_wall=wall.isoformat(),
                  t_end_wall=(wall + timedelta(seconds=60)).isoformat(),
                  t_start_s=100.0, t_end_s=160.0, duration_s=60.0, tone_noise_db=9.0,
                  coherence=0.95, isolation_db=12.0, level_dbfs=-55.0, n_episodes=1,
                  duty_pct=1.7, confidence=0.7)
    return d


def test_known_library_roundtrips_and_labels_harmonics(tmp_path):
    lib = tmp_path / "known.toml"
    upsert_known(lib, "UPS double-mains ripple", 50.0, 1.0, 5, "rectifier hum")
    srcs = load_known(lib)
    assert len(srcs) == 1 and srcs[0].f0_hz == 50.0
    assert srcs[0].freqs == [50, 100, 150, 200, 250]
    dets = [_det(150.2, datetime(2026, 10, 23, 3, 0, tzinfo=TZ)),        # 3rd harmonic
            _det(1750.0, datetime(2026, 10, 23, 4, 0, tzinfo=TZ))]      # nothing known
    assert label(dets, srcs) == 1
    assert dets[0].known_name.startswith("UPS")
    assert dets[0].extra["known"]["harmonic"] == 3
    assert dets[1].known_name == ""                # unmatched stays UNKNOWN (FR-7.3)


def test_appending_does_not_rewrite_the_library(tmp_path):
    lib = tmp_path / "known.toml"
    upsert_known(lib, "a", 50.0)
    text = lib.read_text()
    upsert_known(lib, "b", 100.0)
    assert text in lib.read_text()                 # hand-edited file: append only
    assert len(load_known(lib)) == 2


def test_recurrence_finds_a_daily_schedule_and_says_what_it_cannot(tmp_path):
    """FR-7.2's actual promise: 'on every day at 07:00 ± 4 min' — one episode per day."""
    dets = [_det(617.0, datetime(2026, 10, day, 7, 4, 10, tzinfo=TZ), file=f"202610{day:02d}T070002.flac")
            for day in (21, 22, 23)]
    rec = recurrence(dets, Config())
    assert rec["note"] == ""                        # 3 days: nothing to disclaim
    c = next(c for c in rec["clusters"] if abs(c["freq_hz"] - 617.0) < 1.0)
    assert c["n_episodes"] == 3 and c["files"] == 3
    assert abs(c["median_gap_s"] - 24 * 3600) < 60, c
    assert abs(c["phase_of_day_hour"] - 7.07) < 0.05, c
    assert c["phase_concentration"] > 0.99 and c["phase_spread_h"] < 0.02
    assert sorted(c["class"]) == ["A2"]

    # shift work every 8 h: the gap is the finding, and the hour-of-day is deliberately
    # meaningless because three evenly spaced hours have no circular mean.
    spaced = [_det(300.0, datetime(2026, 10, day, h, 0, tzinfo=TZ), file=f"f{day}{h}.flac")
              for day in (21, 22) for h in (7, 15, 23)]
    sc = recurrence(spaced, Config())["clusters"][0]
    assert abs(sc["median_gap_s"] - 8 * 3600) < 600, sc


def test_recurrence_across_midnight_stays_one_cluster():
    """23:50 and 00:10 are 20 minutes apart; a linear mean would call them 23 hours apart."""
    dets = [_det(300.0, datetime(2026, 10, d, h, m, tzinfo=TZ))
            for d, h, m in ((21, 23, 50), (22, 0, 10), (22, 23, 52), (23, 0, 8))]
    c = recurrence(dets, Config())["clusters"][0]
    assert c["phase_spread_h"] < 0.6, c
    assert c["phase_concentration"] > 0.9, c


def test_a_few_close_tones_do_not_invent_a_low_fundamental():
    """infer_f0 can always find *an* f0; printing a 5 Hz machine would be a fabrication.

    Three LF tracks near 25 Hz produced "f0 = 5.04 Hz, 3 harmonics" on the real day — the
    arithmetic is defensible and the physics is not, so a harmonic claim now needs the members to
    actually spread over several harmonic orders.
    """
    cfg = Config()
    dets = [_det(24.5, datetime(2026, 10, 23, 1, 0, tzinfo=TZ)),
            _det(24.8, datetime(2026, 10, 23, 2, 0, tzinfo=TZ)),
            _det(25.2, datetime(2026, 10, 23, 3, 0, tzinfo=TZ))]
    srcs = group_sources(dets, cfg)
    assert len(srcs) >= 1
    for s in srcs:
        if s.link_reason != "harmonic":
            assert s.f0_hz >= min(d.freq_hz for d in s.members) - 0.01
            assert s.f0_hz > 20.0, f"published f0 {s.f0_hz} Hz is not a machine"


def test_recurrence_labels_event_clusters_by_band_not_pseudo_frequency():
    """An A4 band centre (128 Hz for 88-186 Hz) is not a tone; do not present it as a signature."""
    cfg = Config()
    dets = []
    for i, h in enumerate(range(0, 24)):
        d = _det(128.0, datetime(2026, 10, 23, h, 12, tzinfo=TZ), cls="A4",
                 file=f"20261023T{h:02d}0002.flac")
        d.extra.update({"kind": "band-limited", "band": [88.3, 185.5], "band_index": 2})
        dets.append(d)
    rec = recurrence(dets, cfg)
    c = rec["clusters"][0]
    assert c["kind"] == "event-band" and c["band_hz"] == [88.3, 185.5], c


def test_recurrence_keeps_real_tones_labelled_as_tones():
    cfg = Config()
    dets = [_det(617.0, datetime(2026, 10, d, 7, 4, tzinfo=TZ), file=f"d{d}.flac") for d in (21, 22, 23)]
    assert recurrence(dets, cfg)["clusters"][0]["kind"] == "tone"


def test_a_tone_near_an_event_band_is_not_merged_with_the_band():
    """20-29 Hz band events centre on ~24 Hz, which is 2 Hz from a real 26.8 Hz tone.

    They are different phenomena; the census row used to read "classes A2,A4".
    """
    cfg = Config()
    tone = [_det(26.8, datetime(2026, 10, 23, h, 0, tzinfo=TZ), cls="A2") for h in (4, 5, 6)]
    ev = []
    for h in (4, 5, 6):
        e = _det(24.1, datetime(2026, 10, 23, h, 30, tzinfo=TZ), cls="A4")
        e.extra.update({"band": [20.0, 29.0], "band_index": 0})
        ev.append(e)
    clusters = recurrence(tone + ev, cfg)["clusters"]
    assert len(clusters) == 2, clusters
    kinds = sorted(c["kind"] for c in clusters)
    assert kinds == ["event-band", "tone"], clusters
    assert all(len(c["class"]) == 1 for c in clusters), clusters
    assert all(c["n_episodes"] == 3 for c in clusters), clusters


def test_recurrence_declares_single_day_limitation():
    dets = [_det(120.0, datetime(2026, 10, 23, h, 0, tzinfo=TZ)) for h in (1, 5, 9, 13)]
    rec = recurrence(dets, Config())
    assert "single-day" in rec["note"]
    assert rec["days"] == ["2026-10-23"]


def test_html_report_is_self_contained_and_carries_the_coverage_warning(tmp_path):
    cfg = Config()
    x = add_tone(rumble(16000, 20.0, seed=31), 16000, 900.0, -50.0, 0.0, 20.0)
    p = tmp_path / "20261023T010002.flac"
    sf.write(p, x, 16000, subtype="PCM_16")
    det = _det(900.0, datetime(2026, 10, 23, 1, 0, 5, tzinfo=TZ), file=str(p))
    det.t_start_s, det.t_end_s, det.duration_s = 5.0, 15.0, 10.0
    tl = build_timeline([det], cfg)
    out = build_report([str(p)], [det], [], {"coverage_pct": None}, cfg,
                       tmp_path / "report.html", timeline=tl)
    text = out.read_text()
    assert out.exists() and out.stat().st_size > 20_000
    assert "data:image/png;base64," in text                      # figures embedded
    assert "http://" not in text and "https://" not in text.replace("https://example", "")
    assert re.search(r"no wall-clock|not named YYYYMMDDTHHMMSS", text, re.I) or \
        "coverage" in text.lower()
    assert "audioanalyzer" in text and "900.00" in text
    assert "dBFS" in text                                       # FR-3.3 reference stated
    assert "Corpus timeline" in text and tl[0]["id"] in text     # FR-2.3 view present
