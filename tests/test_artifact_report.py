"""Reports built from artifacts, never from audio (FR-8.7, FR-8.8).

The acceptance test for the whole feature is blunt: scan, DELETE the recordings, rebuild the report.
If it comes out with the same tables and the same evidence pictures, the JSON really is the record
and the FLAC really is optional.
"""

import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from audioanalyzer.artifact import identity, load_run, merge_runs, scanned_set
from audioanalyzer.cli import main
from audioanalyzer.synth import add_tone, rumble

FS = 16000


@pytest.fixture
def run(capsys):
    """Call the CLI and hand back (rc, stdout, stderr) — the assertions below are about what the
    tool TELLS the operator, so capturing per call is the point, not a convenience."""
    def _run(argv):
        rc = main(argv)
        cap = capsys.readouterr()
        return rc, cap.out, cap.err
    return _run


def _night(tmp_path, tag, hours, tone, seed):
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
    for i, hh in enumerate(hours):
        x = add_tone(rumble(FS, 26.0, seed=seed + i), FS, tone + i, -45.0, 0.0, 26.0)
        sf.write(d / f"20261023T{hh:02d}0002.flac", x, FS, subtype="PCM_16")
    return d


def _scan(run, d, out_json, tmp_path, extra=()):
    rc, out, err = run(["scan", str(d), "--json", str(out_json), "--quiet",
                        "--figures-dir", str(tmp_path / "figs"), *extra])
    assert rc == 0, err
    return out


def test_report_builds_after_the_recordings_are_gone(tmp_path, run):
    d = _night(tmp_path, "rec", [0, 1, 2], 880.0, 900)
    j = tmp_path / "scan.json"
    _scan(run, d, j, tmp_path)
    rc, out, _ = run(["scan", str(d), "--report", str(tmp_path / "scan-report.html"), "--quiet",
                      "--figures-dir", str(tmp_path / "figs")])
    assert rc == 0
    html_scan = (tmp_path / "scan-report.html").read_text()
    shutil.rmtree(d)                                      # the audio is simply gone
    rc, out, err = run(["report", str(j), "-o", str(tmp_path / "r.html"),
                        "--figures", str(tmp_path / "figs")])
    assert rc == 0, err
    html_art = (tmp_path / "r.html").read_text()
    assert "NO AUDIO WAS READ" in out
    assert "evidence figure(s) missing" not in html_art, "cards lost their evidence"
    assert html_art.count("<img") == html_scan.count("<img") > 0, \
        f"artifact report has {html_art.count('<img')} images, scan report {html_scan.count('<img')}"


def test_report_of_two_nights_adds_up_without_double_counting(tmp_path, run):
    a = _night(tmp_path, "n1", [0, 1], 880.0, 10)
    b = _night(tmp_path, "n2", [10, 11], 1400.0, 20)
    ja, jb = tmp_path / "a.json", tmp_path / "b.json"
    _scan(run, a, ja, tmp_path)
    _scan(run, b, jb, tmp_path)
    ra, rb = load_run(ja), load_run(jb)
    m = merge_runs([ra, rb])
    assert len(m["dets"]) == len(ra.dets) + len(rb.dets), "disjoint nights must simply add up"
    assert m["stats"]["conflicts"] == 0
    rc, out, _ = run(["report", str(ja), str(jb), "-o", str(tmp_path / "m.html"),
                      "--figures", str(tmp_path / "figs")])
    assert rc == 0
    assert "2 artifact(s)" in out


def test_merge_is_order_independent(tmp_path, run):
    a = _night(tmp_path, "o1", [3], 880.0, 31)
    b = _night(tmp_path, "o2", [9], 1700.0, 32)
    ja, jb = tmp_path / "a.json", tmp_path / "b.json"
    _scan(run, a, ja, tmp_path)
    _scan(run, b, jb, tmp_path)
    assert run(["report", str(ja), str(jb), "-o", str(tmp_path / "m1.html"), "--quiet"])[0] == 0
    assert run(["report", str(jb), str(ja), "-o", str(tmp_path / "m2.html"), "--quiet"])[0] == 0
    assert (tmp_path / "m1.html").read_bytes() == (tmp_path / "m2.html").read_bytes(), \
        "NFR-6 (byte-identical output) must hold across artifacts too"


def test_merging_one_artifact_is_lossless(tmp_path, run):
    """The regression this feature shipped with: identity used key(), which buckets the start to a
    minute, so detections from ONE run collided with each other and 47 % of the census disappeared
    from a report that looked perfectly self-consistent."""
    d = _night(tmp_path, "lossless", [0, 1], 880.0, 95)
    j = tmp_path / "s.json"
    _scan(run, d, j, tmp_path)
    r = load_run(j)
    m = merge_runs([r])
    assert len(m["dets"]) == len(r.dets), \
        f"merging one artifact lost {len(r.dets) - len(m['dets'])} of {len(r.dets)} detections"
    assert m["stats"]["conflicts"] == 0


def test_same_minute_same_band_detections_are_not_duplicates(tmp_path):
    """Impacts routinely share a minute AND a band centre, so a coarse identity is not merely wrong
    in the aggregate -- it deletes real findings."""
    from audioanalyzer.models import Detection
    a = Detection(file="x.flac", cls="A4", freq_hz=93.75, t_start_s=12.0, duration_s=0.4)
    b = Detection(file="x.flac", cls="A4", freq_hz=93.75, t_start_s=47.5, duration_s=0.3)
    assert identity(a, 1000) != identity(b, 1000)
    assert identity(a, 1000) != identity(a, 2000), "different recording, same name"


def test_rescanning_the_same_directory_does_not_double_the_census(tmp_path, run):
    d = _night(tmp_path, "same", [0, 1], 880.0, 40)
    main(["scan", str(d), "--json", str(tmp_path / "run1.json"), "--quiet"])
    main(["scan", str(d), "--json", str(tmp_path / "run2.json"), "--quiet"])
    m = merge_runs([load_run(tmp_path / "run1.json"), load_run(tmp_path / "run2.json")])
    once = len(load_run(tmp_path / "run1.json").dets)
    assert len(m["dets"]) == once, f"merge doubled the census: {len(m['dets'])} vs {once}"


def test_path_spelling_cannot_split_a_detection():
    """Detection.file stores the path AS TYPED, so identity must ignore the directory part."""
    from audioanalyzer.models import Detection
    d = Detection(file="/data/site/20261023T070002.flac", freq_hz=100.0, t_start_s=5.0)
    e = Detection(file="20261023T070002.flac", freq_hz=100.0, t_start_s=5.0)
    assert identity(d, 57357653) == identity(e, 57357653)
    assert identity(d, 57357653) != identity(d, 999), "frame count must distinguish same-named files"


def test_different_thresholds_are_refused_not_blended(tmp_path, run):
    d = _night(tmp_path, "gate", [4], 880.0, 50)
    assert run(["scan", str(d), "--json", str(tmp_path / "a.json"), "--quiet"])[0] == 0
    assert run(["scan", str(d), "--json", str(tmp_path / "b.json"), "--quiet",
                "--min-tone-s", "20"])[0] == 0
    out = tmp_path / "m.html"
    rc, _, err = run(["report", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "-o", str(out)])
    assert rc == 1
    assert "different thresholds" in err and "--allow-mixed" in err
    assert not out.exists(), "a refused merge must not leave a half-truth behind"
    rc, so, _ = run(["report", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                    "-o", str(out), "--allow-mixed"])
    assert rc == 0 and "mixed thresholds merged" in so


def test_incremental_scans_only_what_is_new(tmp_path, run):
    """FR-8.8 with no state file: the previous artifact is the memory."""
    d = _night(tmp_path, "night1", [0, 1, 2], 880.0, 60)
    assert run(["scan", str(d), "--json", str(tmp_path / "scan-1.json"), "--quiet"])[0] == 0
    covered = scanned_set([load_run(tmp_path / "scan-1.json")])
    assert len(covered) == 3
    night2 = tmp_path / "night2"
    night2.mkdir()
    for hh in (0, 1):
        x = add_tone(rumble(FS, 26.0, seed=77 + hh), FS, 660.0 + hh, -45.0, 0.0, 26.0)
        sf.write(night2 / f"20261024T{hh:02d}0002.flac", x, FS, subtype="PCM_16")
    j2 = tmp_path / "scan-2.json"
    rc, out, err = run(["scan", str(d), str(night2), "--json", str(j2), "--incremental"])
    assert rc == 0, err
    names = [Path(f["media"]["path"]).name for f in json.loads(j2.read_text())["files"]]
    assert len(names) == 2, f"expected only the two new chunks, got {names}"
    assert all((n, None) not in {(b, None) for b, _ in covered} for n in names)
    assert not any(n in {b for b, _ in covered} for n in names), "--incremental re-scanned old files"
    assert "skipping 3 of 5" in out


def test_incremental_with_no_history_says_so(tmp_path, run, capsys):
    d = _night(tmp_path, "fresh", [7], 990.0, 70)
    rc, out, err = run(["scan", str(d), "--json", str(tmp_path / "only.json"), "--incremental",
                        "--quiet"])
    assert rc == 0
    assert "no prior artifacts" in err


def test_a_non_artifact_is_rejected_clearly(tmp_path, run):
    bogus = tmp_path / "x.json"
    bogus.write_text(json.dumps({"hello": "world"}))
    rc, _, err = run(["report", str(bogus), "-o", str(tmp_path / "o.html")])
    assert rc == 2
    assert "not a scan artifact" in err


def test_load_run_restores_the_config_the_run_used(tmp_path, run):
    d = _night(tmp_path, "cfg", [5], 880.0, 80)
    j = tmp_path / "c.json"
    assert run(["scan", str(d), "--json", str(j), "--quiet", "--min-tone-s", "13"])[0] == 0
    assert load_run(j).cfg.min_tone_s == 13.0, "a merged report must inherit the run's thresholds"
