"""Parallel scanning and the fail-fast environment checks.

Both exist because of a specific field report: a 24-file directory ran 38 minutes, wrote CSV and
JSON, and only then died on ``import matplotlib`` inside ``--report``.  The lesson is that an
environment problem must cost seconds, not the whole run — and that a directory-wide speed-up may
not reorder a single byte of output.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioanalyzer import cli
from audioanalyzer.cli import _resolve_jobs, main
from audioanalyzer.synth import add_tone, rumble

FS = 16000


def _dir(tmp_path, n=3, seconds=26.0, freq=880.0):
    """Fixtures long enough to be detectable: a tone must hold for >= min_tone_s (10 s), so a
    6-second file yields zero rows and every 'parallel == serial' assertion below would pass
    vacuously on two empty CSVs."""
    d = tmp_path / "rec"
    d.mkdir()
    for i in range(n):
        x = add_tone(rumble(FS, seconds, seed=100 + i), FS, freq + i, -45.0, 0.0, seconds)
        sf.write(d / f"20261023T{i:02d}0002.flac", x, FS, subtype="PCM_16")
    return d


def _strip_timing(text: str) -> str:
    """Wall-clock elapsed per file is legitimately nondeterministic; the order is not."""
    import re
    return re.sub(r"\d+\.\d+ s wall", "X s wall", text)


def test_parallel_output_is_byte_identical_to_serial(tmp_path):
    """NFR-6 has to survive the process pool: workers finish out of order, the CSV may not."""
    d = _dir(tmp_path, n=4)
    a, b = tmp_path / "serial.csv", tmp_path / "par.csv"
    assert main(["scan", str(d), "--jobs", "1", "--csv", str(a), "--quiet"]) == 0
    assert main(["scan", str(d), "--jobs", "4", "--csv", str(b), "--quiet"]) == 0
    rows = list(csv.DictReader(a.open()))
    assert len(rows) == 4, f"fixture produced {len(rows)} rows; the comparison would be vacuous"
    assert a.read_bytes() == b.read_bytes()


def test_parallel_stdout_order_matches_serial(tmp_path, capsys):
    d = _dir(tmp_path, n=3)
    main(["scan", str(d), "--jobs", "1"])
    capsys.readouterr()
    main(["scan", str(d), "--jobs", "3"])
    out_par = _strip_timing(capsys.readouterr().out)
    main(["scan", str(d), "--jobs", "1"])
    out_ser = _strip_timing(capsys.readouterr().out)
    assert out_par == out_ser
    # the sequence of file names is the point: workers finish in arbitrary order
    seq = [Path(l.strip()).name for l in out_ser.splitlines()
           if l.strip().endswith(".flac")]
    assert seq == [f"20261023T{i:02d}0002.flac" for i in range(3)], seq
    seq_par = [Path(l.strip()).name for l in out_par.splitlines()
               if l.strip().endswith(".flac")]
    assert seq_par == seq, "parallel run must emit per-file blocks in input order"


def test_json_corpus_views_are_identical_under_parallelism(tmp_path):
    """A5/timeline/recurrence aggregate over the whole selection, so they must not see order drift."""
    import json
    d = _dir(tmp_path, n=4)
    j1, j2 = tmp_path / "1.json", tmp_path / "2.json"
    main(["scan", str(d), "--jobs", "1", "--json", str(j1), "--quiet"])
    main(["scan", str(d), "--jobs", "4", "--json", str(j2), "--quiet"])
    a, b = json.loads(j1.read_text()), json.loads(j2.read_text())
    assert [f["media"]["path"] for f in a["files"]] == [f["media"]["path"] for f in b["files"]]
    assert a["sources"] == b["sources"] and a["timeline"] == b["timeline"]


def test_resolve_jobs_rules():
    assert _resolve_jobs(0, 1) == 1                      # nothing to parallelise
    assert _resolve_jobs(1, 24) == 1                     # explicit serial
    assert _resolve_jobs(8, 3) == 3                      # never more workers than files
    assert _resolve_jobs(0, 24) >= 1                     # auto is always at least 1
    auto = _resolve_jobs(0, 24)
    import os
    assert auto <= max(1, (os.cpu_count() or 1))


def test_a_missing_report_extra_exits_before_any_analysis(tmp_path, monkeypatch):
    """The bug this test pins: 38 minutes of work, then ModuleNotFoundError at the last import."""
    d = _dir(tmp_path, n=2)
    called = []
    monkeypatch.setattr(cli, "scan_file",
                        lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)   # pretend: not installed
    rc = main(["scan", str(d), "--report", str(tmp_path / "r.html")])
    assert rc == 1, "missing optional dependency is a usage/environment error (FR-8.4)"
    assert called == [], "nothing may be analysed when the report cannot be written"
    assert not (tmp_path / "r.html").exists()


def test_report_runs_when_the_extra_is_present(tmp_path):
    pytest.importorskip("matplotlib")
    d = _dir(tmp_path, n=2)
    out = tmp_path / "r.html"
    assert main(["scan", str(d), "--report", str(out), "--quiet"]) == 0   # FR-8.4: findings ⇒ 0
    assert out.exists() and out.stat().st_size > 10_000


def test_undecodable_input_fails_before_the_pool_starts(tmp_path):
    """A directory with one bad file must cost seconds, not the good files' analysis time."""
    bad = tmp_path / "20261023T010002.flac"
    bad.write_bytes(b"not a flac file at all")
    sf.write(tmp_path / "20261023T020002.flac", rumble(FS, 6.0, seed=7), FS, subtype="PCM_16")
    seen: list[str] = []
    real = cli.scan_file
    cli.scan_file = lambda *a, **k: (seen.append(str(a[0])), real(*a, **k))[1]
    try:
        rc = main(["scan", str(tmp_path), "--csv", str(tmp_path / "x.csv"), "--quiet"])
    finally:
        cli.scan_file = real
    assert rc == 2
    assert seen == [], "preflight must reject the bad file before any file is analysed"


def test_doctor_reports_the_environment_without_scanning(tmp_path, capsys):
    d = _dir(tmp_path, n=2)
    assert main(["doctor", str(d)]) == 0
    out = capsys.readouterr().out
    assert "numpy" in out and "libsndfile" in out
    assert "worker(s)" in out, "the pool sizing must be stated, not left to guesswork"


def test_doctor_flags_a_missing_report_extra(monkeypatch, capsys):
    real = importlib.util.find_spec
    monkeypatch.setattr("importlib.util.find_spec",
                        lambda name: None if name == "matplotlib" else real(name))
    assert main(["doctor"]) == 0                 # still usable: CSV/JSON work without it
    out = capsys.readouterr().out
    assert "matplotlib" in out and "PROBLEM" in out


def test_doctor_exits_1_when_a_core_dependency_is_absent(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr("importlib.util.find_spec",
                        lambda name: None if name == "numpy" else real(name))
    assert main(["doctor"]) == 1


def test_toml_shim_works_on_this_interpreter(tmp_path):
    """config.py is in every import chain, so a bare `import tomllib` would break 3.10 wholly."""
    from audioanalyzer._toml import tomllib
    assert tomllib.loads("[a]\nb = 1") == {"a": {"b": 1}}
    p = tmp_path / "c.toml"
    p.write_text("tone_noise_db = 4.5\n")             # Config keys are flat, not sectioned
    from audioanalyzer.config import Config
    assert Config.from_toml(p).tone_noise_db == 4.5
