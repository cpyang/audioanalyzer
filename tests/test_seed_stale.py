"""Stale-candidate safety (FR-7.3b).

A whole-day seed with the pre-fix ridge estimator wrote 16 bogus entries to
``out/known.candidates.toml``. The fixed estimator found nothing and returned *before writing*, so
the poisoned file stayed on disk one ``--promote`` away from the live library. These tests pin the
two properties that make that impossible: a seed run always rewrites the candidate file, and
``promote`` indices refer to that same file.
"""

from pathlib import Path

import pytest
import soundfile as sf

from audioanalyzer.cli import main
from audioanalyzer.config import Config
from audioanalyzer.seed import SeedCandidate, promote, write_candidates
from audioanalyzer.synth import add_tone, rumble

FS = 16000


def _cand(freq, kind="line", n=2, total=2, duty=100.0):
    return SeedCandidate(freq_hz=freq, kind=kind, n_files=n, files_total=total,
                         level_dbfs=-50.0, over_floor_db=9.0, duty_pct=duty)


def test_a_new_seed_run_replaces_the_previous_list(tmp_path):
    cf = tmp_path / "cand.toml"
    write_candidates(cf, [_cand(100.0), _cand(200.0)])
    assert cf.read_text().count("[[source]]") == 2
    write_candidates(cf, [_cand(300.0)])
    text = cf.read_text()
    assert text.count("[[source]]") == 1, "old candidates survived a new run"
    assert "100.000" not in text and "200.000" not in text


def test_an_empty_seed_run_clears_a_stale_list(tmp_path):
    """The exact incident: fix lands, run finds nothing, file never rewritten, junk persists."""
    cf = tmp_path / "cand.toml"
    write_candidates(cf, [_cand(6375.0, kind="ridge", n=24, total=24, duty=None)])
    write_candidates(cf, [])                                    # what a zero-candidate run does
    assert cf.read_text().count("[[source]]") == 0
    live = tmp_path / "known.toml"
    with pytest.raises(IndexError):                             # nothing left to promote
        promote(cf, live, [1])


def test_append_mode_is_opt_in_and_indexable(tmp_path):
    from audioanalyzer.known import load_known
    cf = tmp_path / "cand.toml"
    write_candidates(cf, [_cand(100.0)])
    write_candidates(cf, [_cand(300.0)], append=True)
    assert len(load_known(cf)) == 2                            # parse, don't grep: the header
    live = tmp_path / "known.toml"                             # contains a commented [[source]]
    assert promote(cf, live, [2], {2: "Chiller 2 bearing"}) == ["Chiller 2 bearing"]
    assert [round(k.f0_hz) for k in load_known(live)] == [300]


def test_cli_seed_rewrites_a_poisoned_file(tmp_path):
    """End to end: seed a directory whose sounds are not always-present, over a dirty file."""
    d = tmp_path / "rec"
    d.mkdir()
    for i in range(3):
        x = add_tone(rumble(FS, 30.0, seed=400 + i), FS, 880.0 + i, -45.0, 0.0, 30.0)
        sf.write(d / f"20261023T{i:02d}0002.flac", x, FS, subtype="PCM_16")
    cf = tmp_path / "cand.toml"
    write_candidates(cf, [_cand(6375.0, kind="ridge", n=24, total=24, duty=None)])
    assert cf.read_text().count("[[source]]") == 1

    # --min-file-frac above what the tone achieves -> zero candidates -> must clear the file
    rc = main(["known", "--seed", str(d), "--candidates", str(cf), "--min-file-frac", "1.5"])
    assert rc == 0
    assert cf.read_text().count("[[source]]") == 0, "stale candidates survived a real run"


def test_promote_is_idempotent_and_reports_nothing_silently(tmp_path):
    from audioanalyzer.known import load_known
    cf = tmp_path / "cand.toml"
    write_candidates(cf, [_cand(120.0), _cand(240.0)])
    live = tmp_path / "known.toml"
    assert len(promote(cf, live, [1, 2])) == 2
    assert promote(cf, live, [1, 2]) == []                     # already present
    assert sorted(round(k.f0_hz) for k in load_known(live)) == [120, 240]
