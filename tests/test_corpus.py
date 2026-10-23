"""FR-2.3: an event that straddles a chunk boundary is reported once, not twice."""

from datetime import datetime, timedelta, timezone

import pytest

from audioanalyzer.config import Config
from audioanalyzer.corpus import build_timeline
from audioanalyzer.models import Detection

TZ = timezone(timedelta(hours=8))


def det(freq, hh, mm, dur=600.0, cls="A2", idx=0):
    """One per-file detection. ``idx`` keeps filenames distinct: two chunks can start in the
    same hour, and reusing a name would silently under-count n_files."""
    t = datetime(2026, 10, 23, hh, mm, tzinfo=TZ)
    return Detection(cls=cls, freq_hz=freq, file=f"20261023T{hh:02d}{mm:02d}0{idx}.flac",
                     t_start_s=0.0, duration_s=dur, level_dbfs=-50.0, tone_noise_db=9.0,
                     coherence=0.95, duty_pct=100.0,
                     t_start_wall=t.isoformat(), t_end_wall=(t + timedelta(seconds=dur)).isoformat())


def test_one_tone_across_the_hour_seam_is_one_feature():
    """The 06:50-07:14 whine lives in four files; the corpus must say "one", not "four"."""
    cfg = Config()
    dets = [det(417.3, 6, 50, 600.0, idx=0), det(417.5, 6, 58, 120.0, idx=1),
            det(417.4, 7, 0, 600.0, idx=2), det(417.2, 7, 10, 240.0, idx=3)]
    tl = build_timeline(dets, cfg)
    assert len(tl) == 1, tl
    e = tl[0]
    assert e["n_files"] == 4 and e["straddles_boundary"] and e["n_detections"] == 4
    assert e["t_start_wall"][11:19] == "06:50:00" and e["t_end_wall"][11:19] == "07:14:00"
    assert e["active_s"] == pytest.approx(sum(d.duration_s for d in dets), abs=1e-6)
    assert len({d.timeline_entry for d in dets}) == 1


def test_gap_s_must_cover_the_measured_blackout():
    """26.7 s was the worst measured inter-chunk gap (§2.2); a smaller horizon re-splits it."""
    # 3570 s of audio in a 3600 s hour = a 30 s blackout at the seam, near the measured worst.
    dets = [det(1234.0, 6, 0, 3570.0, idx=0), det(1234.2, 7, 0, 60.0, idx=1)]
    assert len(build_timeline(dets, Config())) == 1
    assert len(build_timeline(dets, Config(), gap_s=10.0)) == 2


def test_repeated_each_hour_is_not_merged_into_one_long_feature():
    cfg = Config()
    dets = [det(900.0, 3, 5, 60.0, idx=0), det(900.2, 3, 45, 60.0, idx=1)]
    tl = build_timeline(dets, cfg)
    assert len(tl) == 2
    assert all(e["n_detections"] == 1 for e in tl)


def test_transients_stay_atomic_and_unmerged():
    """A4's freq_hz is a band label; merging labels would invent one 'source' per band."""
    cfg = Config()
    dets = [det(617.0, 1, 0, 20.0, cls="A4", idx=0), det(617.1, 1, 0, 600.0, idx=1),
            det(617.0, 1, 20, 20.0, cls="A4", idx=2)]
    tl = build_timeline(dets, cfg)
    assert len(tl) == 3, [(e["class"], e["n_detections"]) for e in tl]
    tone = next(e for e in tl if e["class"] == "A2")
    assert tone["n_detections"] == 1


def test_untimed_detections_are_left_alone():
    cfg = Config()
    d = det(500.0, 4, 0)
    d.t_start_wall = d.t_end_wall = ""
    assert build_timeline([d], cfg) == []
    assert d.timeline_entry == ""


def test_far_apart_frequencies_never_merge():
    cfg = Config()
    dets = [det(500.0, 2, 0), det(503.0, 2, 5)]
    assert len(build_timeline(dets, cfg)) == 2
