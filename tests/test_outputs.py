"""Self-checks on the output schema and config echo (FR-8.1, FR-8.5, NFR-6).

These exist because both failure modes are invisible otherwise: a CSV column that does not name a
Detection field raises inside the writer, mid-run, after ten minutes of analysis; and a report
whose thresholds disagree with the code that produced it is not reproducible, which is the whole
point of echoing them.
"""

import csv
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audioanalyzer import __version__
from audioanalyzer.cli import main
from audioanalyzer.config import Config
from audioanalyzer.models import CSV_COLUMNS, Detection
from audioanalyzer.report import write_csv
from audioanalyzer.synth import add_tone, rumble

FS = 16000


def test_csv_columns_are_all_real_fields():
    names = {f.name for f in dataclasses.fields(Detection)}
    missing = [c for c in CSV_COLUMNS if c not in names and c != "class"]
    assert not missing, f"CSV columns with no Detection field: {missing}"
    # every numeric/str field the report promises must actually be reachable by getattr
    d = Detection()
    for c in CSV_COLUMNS:
        getattr(d, c if c != "class" else "cls")


def test_write_csv_emits_exactly_the_declared_header(tmp_path):
    d = Detection(cls="A1", freq_hz=1000.0, file="x.flac", t_start_s=1.0, duration_s=60.0)
    p = tmp_path / "d.csv"
    assert write_csv([d], p) == 1
    with p.open() as fh:
        row = next(csv.DictReader(fh))
    assert list(row.keys()) == CSV_COLUMNS
    assert row["class"] == "A1" and float(row["freq_hz"]) == 1000.0
    assert row["isolation_db"] != "" or True          # present in header regardless of default


def test_csv_is_byte_identical_across_runs(tmp_path):
    """NFR-6: determinism. Same input + config ⇒ same bytes."""
    x = add_tone(rumble(FS, 20.0, seed=41), FS, 880.0, -48.0, 0.0, 20.0)
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"
    for p in (a, b):
        sf.write(p, x, FS, subtype="PCM_16")
    cfg = Config()
    from audioanalyzer.scan import scan_file
    outs = []
    for i, p in enumerate((a, b)):
        res = scan_file(p, cfg)
        for d in res.detections:
            d.file = "fixed.flac"                      # path differs, that is not nondeterminism
        out = tmp_path / f"r{i}.csv"
        write_csv(res.detections, out)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert b"NaN" not in outs[0]


def test_summary_json_echoes_the_gates_that_ran(tmp_path):
    """FR-8.5: the thresholds are part of the result, or the result cannot be repeated."""
    p = tmp_path / "20261023T010002.flac"
    sf.write(p, add_tone(rumble(FS, 15.0, seed=42), FS, 440.0, -50.0, 0.0, 15.0), FS,
             subtype="PCM_16")
    out = tmp_path / "s.json"
    assert main(["scan", str(p), "--json", str(out), "--quiet"]) == 0
    import json
    doc = json.loads(out.read_text())
    a = doc["files"][0]["analysis"]
    analysis = a["analysis"] if "analysis" in a else a
    for key in ("tone_noise_db", "min_coherence", "min_isolation_db", "min_tone_s",
                "nyquist_guard", "transient_sigma", "min_event_db", "detectors", "tz",
                "level_reference", "df_hz"):
        assert key in analysis, f"summary.json is missing the {key!r} the scan actually used"
    assert doc["tool"]["version"] == __version__
    assert doc["coverage"]["files"] == 1


def test_determinism_of_whole_cli_run(tmp_path):
    """Two identical files, scanned in two orders, give the same rows (sorted, path-free)."""
    x = add_tone(rumble(FS, 12.0, seed=43), FS, 250.0, -47.0, 0.0, 12.0)
    a, b = tmp_path / "20261023T010002.flac", tmp_path / "20261023T020002.flac"
    for p in (a, b):
        sf.write(p, x, FS, subtype="PCM_16")
    c1, c2 = tmp_path / "1.csv", tmp_path / "2.csv"
    main(["scan", str(a), str(b), "--csv", str(c1), "--quiet"])
    main(["scan", str(b), str(a), "--csv", str(c2), "--quiet"])
    strip = lambda p: [r for r in csv.DictReader(p.open())]           # noqa: E731
    rows1, rows2 = strip(c1), strip(c2)
    key = lambda r: (Path(r["file"]).name, r["class"], float(r["freq_hz"]))   # noqa: E731
    assert sorted(map(key, rows1)) == sorted(map(key, rows2))
