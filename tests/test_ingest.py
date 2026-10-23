"""Ingest / data-quality tests (REQUIREMENTS.md FR-1, FR-2.1, A6, AC-1)."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.ingest import MediaInfo, capture_start_from_name, probe

SR = 16000


def _write(path: Path, x: np.ndarray, subtype: str = "PCM_16") -> Path:
    sf.write(str(path), x, SR, subtype=subtype)
    return path


def test_media_info_matches_synthetic_ground_truth(tmp_path: Path):
    """Known DC/RMS/clip values must come back from the streaming reducer."""
    rng = np.random.default_rng(7)
    n = 20 * SR
    noise = rng.normal(0.0, 10 ** (-50.0 / 20.0), n)          # -50 dBFS broadband
    noise += 0.01                                             # DC offset = -40 dBFS
    # hard clip a 10 ms square burst: 160 samples pinned at full scale
    noise[5 * SR:5 * SR + 160] = -1.0
    path = _write(tmp_path / "20260101T000000.flac", noise)

    info = probe(path, Config(block_s=2.0))
    y, _ = sf.read(str(path), dtype="float64")   # decoded reference the reducer must reproduce

    assert info.sample_rate == SR
    assert info.channels == 1
    assert info.frames == n
    assert abs(info.duration_s - 20.0) < 1e-6
    assert info.bits == 16
    assert info.nyquist == 8000.0
    # the injected -1.0 burst shifts the true mean by 160*(-1.01)/n = -0.0005, so the
    # analytic 0.01 is not the ground truth; the decoded signal is.
    assert info.dc_offset == pytest.approx(float(y.mean()), abs=1e-12)
    assert abs(20 * math.log10(abs(info.dc_offset)) + 40.0) < 0.6
    assert info.peak_dbfs == pytest.approx(0.0, abs=1e-6)
    assert info.clipped_samples >= 160
    assert info.full_scale_run_samples >= 160
    assert info.rms_dbfs == pytest.approx(
        20 * math.log10(float(np.sqrt(np.mean(y ** 2)))), abs=1e-9)
    assert len(info.block_rms_dbfs) == math.ceil(n / (2.0 * SR))


def test_block_size_does_not_change_results(tmp_path: Path):
    """FR-1.2: streaming must not alter any aggregate, and runs may straddle blocks."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 10 ** (-45.0 / 20.0), 37 * SR) + 0.005
    x[10 * SR:10 * SR + 400] = 1.0                 # a run spanning a 30 s block edge
    path = _write(tmp_path / "20260101T120000.flac", x)

    a = probe(path, Config(block_s=2.0))
    b = probe(path, Config(block_s=30.0))
    assert a.frames == b.frames
    assert a.clipped_samples == b.clipped_samples == 400
    assert a.full_scale_run_samples == b.full_scale_run_samples == 400
    assert a.rms_dbfs == pytest.approx(b.rms_dbfs, abs=1e-9)
    assert a.dc_offset == pytest.approx(b.dc_offset, abs=1e-12)


def test_quality_flags_fire(tmp_path: Path):
    """A6 (FR-4.6): DC offset and clipping must be reported, silence must be measured."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 10 ** (-30.0 / 20.0), 8 * SR)
    x += 10 ** (-20.0 / 20.0)                      # -20 dBFS DC offset
    path = _write(tmp_path / "20260101T000000.flac", x)
    kinds = {f["kind"] for f in probe(path, Config()).quality_flags(Config())}
    assert "dc_offset" in kinds

    quiet = np.zeros(8 * SR)
    qpath = _write(tmp_path / "20260101T000010.flac", quiet)
    qinfo = probe(qpath, Config())
    assert qinfo.silence_s == pytest.approx(8.0, abs=0.2)
    assert {f["kind"] for f in qinfo.quality_flags(Config())} == {"silence"}


def test_clean_signal_has_no_flags(tmp_path: Path):
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 10 ** (-40.0 / 20.0), 12 * SR)
    path = _write(tmp_path / "20260101T000000.flac", x)
    assert probe(path, Config()).quality_flags(Config()) == []


def test_filename_to_wall_clock(tmp_path: Path):
    """FR-2.1 / AC-7: filename is capture start in host-local time."""
    dt = capture_start_from_name("20261023T070002.flac", tz="Asia/Taipei")
    assert dt == datetime(2026, 10, 23, 7, 0, 2, tzinfo=dt.tzinfo)
    assert dt.utcoffset().total_seconds() == 8 * 3600

    assert capture_start_from_name("recording_final.flac", tz="Asia/Taipei") is None
    assert capture_start_from_name("20261023T070002.flac").tzinfo is not None


def test_wall_clock_mapping_on_mediainfo():
    dt = capture_start_from_name("20261023T070002.flac", tz="Asia/Taipei")
    info = MediaInfo(path="x", sample_rate=SR, channels=1, frames=SR, duration_s=1.0,
                     subtype="PCM_16", format="FLAC", size_bytes=0,
                     capture_start=dt, tz="Asia/Taipei")
    assert info.wall_clock(0).isoformat() == "2026-10-23T07:00:02+08:00"
    assert info.wall_clock(3599.5).isoformat() == "2026-10-23T08:00:01.500000+08:00"


def test_nonexistent_file_is_decode_error(tmp_path: Path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        probe(tmp_path / "nope.flac", Config())


def test_clip_level_is_top_pcm_code():
    """Regression: the clip threshold must equal the top signed-PCM code, not 1-2**-B.

    An earlier version used ``1 - 2**-16`` = 0.99998474, which is *above* the largest
    positive s16 value (0.99996948) and therefore never counted a clipped sample.
    """
    from audioanalyzer.ingest import _clip_level

    assert _clip_level("PCM_16", 16) == 32767.0 / 32768.0 == 1.0 - 2.0 ** -15
    assert _clip_level("PCM_24", 24) == 1.0 - 2.0 ** -23
    assert _clip_level("FLOAT", 32) == 1.0
    # the top code must be detected, one LSB below it must not
    assert 32767.0 / 32768.0 >= _clip_level("PCM_16", 16)
    assert 32766.0 / 32768.0 < _clip_level("PCM_16", 16)


def test_negative_full_scale_counts(tmp_path: Path):
    """-32768 maps to -1.0 and must also count as clipped."""
    x = np.zeros(4 * SR)
    x[100:120] = -1.0
    path = _write(tmp_path / "20260101T000000.flac", x)
    info = probe(path, Config())
    assert info.clipped_samples == 20
    assert info.full_scale_run_samples == 20


def test_digital_silence_does_not_count_every_sample_as_at_max(tmp_path: Path):
    """FR-1.5: an all-zero file has peak 0.0, so 'samples equal to the peak' = every sample.

    The first implementation reported at-max = 480000 for a 30 s silent file, which reads as
    "the whole file clipped".  at-max is a full-scale statement, so it is 0 unless something
    actually reached full scale.
    """
    path = _write(tmp_path / "20260101T000000.flac", np.zeros(30 * SR, dtype=np.float32))
    info = probe(path, Config())
    assert info.peak_dbfs == float("-inf")
    assert info.peak_count == 0
    assert info.clipped_samples == 0
    assert math.isinf(info.rms_dbfs) and info.rms_dbfs < 0
    flags = {f["kind"] for f in info.quality_flags(Config())}
    assert "silence" in flags              # A6 says so instead of staying quiet


@pytest.mark.parametrize("seconds", [0.010, 0.05, 0.127, 0.128, 0.5])
def test_files_shorter_than_a_window_are_analysed_or_declared(tmp_path: Path, seconds):
    """FR-1.5: sub-window, exactly-one-window and slightly-over files must all exit cleanly."""
    x = np.zeros(max(1, int(seconds * SR)), dtype=np.float32)
    if x.size > 2048:
        x[::7] = 0.01
    path = _write(tmp_path / "20260101T000000.flac", x)
    info = probe(path, Config())
    assert abs(info.duration_s - seconds) < 1e-4


def test_stereo_file_analyses_both_channels(tmp_path: Path):
    """FR-1.1: multi-channel is analysed per channel then merged, tagged with `channel`."""
    x = np.zeros((SR * 3, 2), dtype=np.float32)
    path = tmp_path / "20260101T000000.flac"
    sf.write(str(path), x, SR, subtype="PCM_16")
    info = probe(path, Config())
    assert info.channels == 2
    assert info.duration_s == pytest.approx(3.0, abs=1e-4)
