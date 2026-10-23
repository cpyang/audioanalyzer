"""FR-6 tests: reference floor profiles and compare mode."""

from pathlib import Path

import numpy as np
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.profile import FloorProfile, compare
from audioanalyzer.synth import add_tone, rumble

FS = 16000


def _write(tmp_path: Path, name: str, x: np.ndarray) -> Path:
    p = tmp_path / name
    sf.write(p, x, FS, subtype="PCM_16")
    return p


def test_profile_roundtrips_and_names_the_reference_tones(tmp_path):
    cfg = Config()
    ref = _write(tmp_path, "ref.flac", rumble(FS, 30.0, seed=21))
    prof = FloorProfile.from_files([ref], cfg)
    prof.save(tmp_path / "baseline.npz")
    back = FloorProfile.load(tmp_path / "baseline.npz")
    assert np.allclose(back.mean_db, prof.mean_db, atol=1e-6)
    assert back.sample_rate == FS and back.nperseg == cfg.nperseg and back.hop == cfg.nperseg - cfg.noverlap
    assert abs(back.duration_s - 30.0) < 0.1
    # the reference's own floor must not conjure tones out of nothing
    assert back.tone_candidates(cfg).size == 0


def test_compare_finds_a_tone_that_was_not_there_before(tmp_path):
    """FR-6.2: the whole point — 'something started making noise'."""
    cfg = Config()
    base = rumble(FS, 30.0, seed=22)
    ref = _write(tmp_path, "ref.flac", base)
    today = base.copy()
    today = add_tone(today, FS, 1750.0, -52.0, 0.0, 30.0)
    _write(tmp_path, "today.flac", today)
    prof = FloorProfile.from_files([ref], cfg)
    res = compare(tmp_path / "today.flac", prof, cfg)
    assert res.gone_tones == []
    assert any(abs(d.freq_hz - 1750.0) < 8.0 for d in res.new_tones), \
        f"new tone not reported: {[d.freq_hz for d in res.new_tones]}"
    assert all(d.extra.get("source") == "compare-new" for d in res.new_tones)


def test_compare_of_identical_audio_reports_nothing_new(tmp_path):
    cfg = Config()
    a = _write(tmp_path, "a.flac", rumble(FS, 24.0, seed=23))
    prof = FloorProfile.from_files([a], cfg)
    res = compare(a, prof, cfg)
    assert res.new_tones == [] and res.gone_tones == []
    assert np.max(np.abs(res.band_file_db - res.band_ref_db)) < 0.05


def test_compare_refuses_a_mismatched_analysis_grid(tmp_path):
    """A profile built at another window size would compare different bins; that is not a
    quiet-but-honest failure mode, it is a wrong number, so it must be an error."""
    cfg = Config()
    a = _write(tmp_path, "a.flac", rumble(FS, 12.0, seed=24))
    prof = FloorProfile.from_files([a], cfg)
    try:
        compare(a, prof, Config(mode="hr"))
    except ValueError as exc:
        assert "grid" in str(exc)
    else:
        raise AssertionError("grid mismatch was not caught")


def test_profile_from_several_files_accumulates_all_the_audio(tmp_path):
    cfg = Config()
    files = [_write(tmp_path, f"r{i}.flac", rumble(FS, 12.0, seed=30 + i)) for i in range(3)]
    prof = FloorProfile.from_files(files, cfg)
    assert abs(prof.duration_s - 36.0) < 0.3
    # no double counting: the merged histogram's total equals the sum over files
    per_file = [FloorProfile.from_files([f], cfg).model.blocks[0][1].counts.sum() for f in files]
    assert int(prof.model.blocks[0][1].counts.sum()) == int(sum(per_file))
