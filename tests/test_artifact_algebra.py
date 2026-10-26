"""Artifact algebra: merge (FR-8.9) and split (FR-8.9).

These are the tests that catch the failure mode this feature actually risks — not "did it write a
file" but "did the new artifact silently stop saying something the old one said". A merged artifact
that loses a file's quality_flags loses the record that the audio was clipped; one that serializes
detections through the CSV row builder loses z_peak, and the day map then colours impacts with the
wrong number. Both reload fine and both pass a count assertion.
"""

import json
from pathlib import Path

import pytest
import soundfile as sf

from audioanalyzer.artifact import load_run
from audioanalyzer.cli import main
from audioanalyzer.synth import add_click, add_tone, rumble

FS = 16000


@pytest.fixture
def run(capsys):
    def _run(argv):
        rc = main(argv)
        cap = capsys.readouterr()
        return rc, cap.out, cap.err
    return _run


def _rec(tmp_path, name, *, tone=None, click=None, secs=120, seed=7):
    """One recording: rumble floor, an optional tone, an optional impact."""
    x = rumble(FS, secs, seed=seed)
    if tone:
        x = add_tone(x, FS, tone, -45.0, 0.0, secs - 2.0)
    if click:
        x = add_click(x, FS, click, -20.0)
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / name), x, FS, subtype="PCM_16")
    return tmp_path / name


def _scan(run, d, j, tmp_path):
    rc, out, err = run(["scan", str(d), "--json", str(j), "--quiet",
                        "--figures-dir", str(tmp_path / "figs")])
    assert rc == 0, err
    return j


def _keys(doc):
    """Same tuple identity() builds from a Detection, computed from a saved row instead — so a test
    can compare what is IN two files rather than what a loader made of them."""
    return {(Path(f["media"]["path"]).name, r["cls"], round(float(r["freq_hz"]), 4),
             round(float(r["t_start_s"]), 6))
            for f in doc["files"] for r in f["detections"]}


# ------------------------------------------------------------------------------ merge

def test_merge_combines_two_nights_into_one_artifact(tmp_path, run):
    a, b = tmp_path / "n1", tmp_path / "n2"
    for name in ("20261023T000002.flac", "20261023T010002.flac"):
        _rec(a, name, tone=880.0)
    for name in ("20261024T000002.flac", "20261024T010002.flac"):
        _rec(b, name, tone=1400.0)
    ja, jb = _scan(run, a, tmp_path / "a.json", tmp_path), _scan(run, b, tmp_path / "b.json", tmp_path)

    rc, out, err = run(["merge", str(ja), str(jb), "-o", str(tmp_path / "all.json")])
    assert rc == 0, err
    m = load_run(tmp_path / "all.json")
    assert {Path(i.path).name for i in m.infos} == {
        "20261023T000002.flac", "20261023T010002.flac",
        "20261024T000002.flac", "20261024T010002.flac"}
    assert _keys(json.loads((tmp_path / "all.json").read_text())) == \
        _keys(json.loads(ja.read_text())) | _keys(json.loads(jb.read_text()))


def test_merge_is_idempotent_and_never_double_counts(tmp_path, run):
    a = tmp_path / "n"
    _rec(a, "20261023T000002.flac", tone=880.0)
    j = _scan(run, a, tmp_path / "s.json", tmp_path)
    once = len(load_run(j).dets)
    run(["merge", str(j), "-o", str(tmp_path / "m1.json")])
    run(["merge", str(tmp_path / "m1.json"), str(j), "-o", str(tmp_path / "m2.json")])
    m2 = json.loads((tmp_path / "m2.json").read_text())
    assert sum(len(f["detections"]) for f in m2["files"]) == once, "re-merging duplicated the census"


def test_merged_artifact_keeps_every_source_files_metadata(tmp_path, run):
    """R17: analysis and quality_flags must survive the merge verbatim."""
    a, b = tmp_path / "a", tmp_path / "b"
    _rec(a, "20261023T000002.flac", tone=880.0)
    _rec(b, "20261024T000002.flac", tone=1400.0, seed=11)
    ja, jb = _scan(run, a, tmp_path / "a.json", tmp_path), _scan(run, b, tmp_path / "b.json", tmp_path)
    run(["merge", str(ja), str(jb), "-o", str(tmp_path / "m.json")])
    src = {f["media"]["path"]: f
           for doc in (json.loads(ja.read_text()), json.loads(jb.read_text()))
           for f in doc["files"]}
    got = {f["media"]["path"]: f for f in json.loads((tmp_path / "m.json").read_text())["files"]}
    assert set(got) == set(src)
    for k in src:
        assert got[k]["analysis"] == src[k]["analysis"], f"lost analysis block for {k}"
        assert got[k]["quality_flags"] == src[k]["quality_flags"], f"lost quality_flags for {k}"


def test_merged_artifact_keeps_extra_fields_that_drive_the_plots(tmp_path, run):
    """An impact's z_peak lives in Detection.extra; a row-builder that flattens to CSV columns drops
    it, and the day map then colours that impact by the fallback severity instead of its peak."""
    a = tmp_path / "a"
    _rec(a, "20261023T000002.flac", click=30.0)
    j = _scan(run, a, tmp_path / "s.json", tmp_path)
    run(["merge", str(j), "-o", str(tmp_path / "m.json")])
    before = [x for f in json.loads(j.read_text())["files"] for x in f["detections"]]
    after = [x for f in json.loads((tmp_path / "m.json").read_text())["files"] for x in f["detections"]]
    assert after, "no detections to compare"
    assert {k for x in before for k in x} == {k for x in after for k in x}, \
        "merged artifact has a different key set than the scan artifact"
    peaked = [x for x in before if "z_peak" in x]
    if peaked:
        assert all("z_peak" in y for y in after), "z_peak dropped by the merge"


def test_merge_refuses_to_clobber_an_existing_artifact(tmp_path, run):
    a = tmp_path / "a"
    _rec(a, "20261023T000002.flac", tone=880.0)
    j = _scan(run, a, tmp_path / "s.json", tmp_path)
    out = tmp_path / "m.json"
    run(["merge", str(j), "-o", str(out)])
    keep = out.read_text()
    rc, _, err = run(["merge", str(j), "-o", str(out)])
    assert rc != 0 and "--force" in err, err
    assert out.read_text() == keep, "an existing artifact was overwritten without --force"
    assert run(["merge", str(j), "-o", str(out), "--force"])[0] == 0


def test_merge_of_different_thresholds_needs_an_explicit_opt_in(tmp_path, run):
    a = tmp_path / "a"
    _rec(a, "20261023T000002.flac", tone=880.0)
    j1 = _scan(run, a, tmp_path / "s1.json", tmp_path)
    rc, _, _ = run(["scan", str(a), "--json", str(tmp_path / "s2.json"), "--quiet",
                    "--min-prominence", "9.0", "--figures-dir", str(tmp_path / "figs")])
    assert rc == 0
    rc, _, err = run(["merge", str(j1), str(tmp_path / "s2.json"), "-o", str(tmp_path / "m.json")])
    assert rc != 0, "merged two different gate settings and said nothing"
    rc, out, _ = run(["merge", str(j1), str(tmp_path / "s2.json"), "-o", str(tmp_path / "m.json"),
                      "--allow-mixed"])
    assert rc == 0 and "mixed" in out.lower()
    assert "provenance" in json.loads((tmp_path / "m.json").read_text())


# ------------------------------------------------------------------------------ split

def _threeday(tmp_path, run):
    """24h of hourly recordings across two days and two months, one scan artifact."""
    d = tmp_path / "corpus"
    for stamp in ("20261023T000002", "20261023T010002", "20261024T000002", "20261101T000002"):
        _rec(d, f"{stamp}.flac", tone=880.0 + len(stamp), seed=hash(stamp) % 1000)
    return _scan(run, d, tmp_path / "all.json", tmp_path)


def test_split_by_day_covers_everything_exactly_once(tmp_path, run):
    j = _threeday(tmp_path, run)
    whole = json.loads(j.read_text())
    total = sum(len(f["detections"]) for f in whole["files"])
    rc, out, err = run(["split", str(j), "--by", "day", "--dir", str(tmp_path / "parts")])
    assert rc == 0, err
    parts = sorted((tmp_path / "parts").glob("scan-*.json"))
    assert [p.name for p in parts] == ["scan-20261023.json", "scan-20261024.json",
                                       "scan-20261101.json"], [p.name for p in parts]
    seen = []
    for p in parts:
        seen += [x for f in json.loads(p.read_text())["files"] for x in f["detections"]]
    assert len(seen) == total, f"split wrote {len(seen)} of {total} detections"
    ids = []
    for p in parts:
        for f in json.loads(p.read_text())["files"]:
            ids += [(Path(f["media"]["path"]).name, r["cls"], r["t_start_s"])
                    for r in f["detections"]]
    assert len(ids) == len(set(ids)), "a detection landed in two buckets"


def test_split_by_hour_and_month_use_their_own_bucket_names(tmp_path, run):
    j = _threeday(tmp_path, run)
    run(["split", str(j), "--by", "hour", "--dir", str(tmp_path / "h"), "--prefix", "scan"])
    names = sorted(p.name for p in (tmp_path / "h").glob("*.json"))
    assert names == [f"scan-{k}.json" for k in
                     ("20261023T00", "20261023T01", "20261024T00", "20261101T00")], names
    run(["split", str(j), "--by", "month", "--dir", str(tmp_path / "m")])
    names = sorted(p.name for p in (tmp_path / "m").glob("*.json"))
    assert names == ["scan-2026-10.json", "scan-2026-11.json"], names


def test_merge_of_split_parts_reproduces_the_original_set(tmp_path, run):
    j = _threeday(tmp_path, run)
    run(["split", str(j), "--by", "day", "--dir", str(tmp_path / "parts")])
    parts = sorted((tmp_path / "parts").glob("scan-*.json"))
    rc, out, err = run(["merge", *[str(p) for p in parts], "-o", str(tmp_path / "back.json")])
    assert rc == 0, err
    assert _keys(json.loads((tmp_path / "back.json").read_text())) == \
        _keys(json.loads(j.read_text())), "split then merge changed what was found"


def test_each_part_describes_itself_not_the_whole_run(tmp_path, run):
    """Views are recomputed per artifact; a part must not carry the parent's coverage span."""
    j = _threeday(tmp_path, run)
    run(["split", str(j), "--by", "day", "--dir", str(tmp_path / "parts")])
    whole = json.loads(j.read_text())
    p0 = json.loads((tmp_path / "parts" / "scan-20261023.json").read_text())
    assert len(p0["files"]) < len(whole["files"])
    assert p0["coverage"]["audio_s"] < whole["coverage"]["audio_s"], \
        "a day part claims the whole corpus's recorded seconds"
    assert all(Path(s["first_file"]).name.startswith("202610") if "first_file" in s else True
               for s in p0["sources"])


def test_split_refuses_to_clobber_and_names_its_buckets(tmp_path, run):
    j = _threeday(tmp_path, run)
    outdir = tmp_path / "parts"
    rc, _, err = run(["split", str(j), "--by", "day", "--dir", str(outdir)])
    assert rc == 0, err
    keep = (outdir / "scan-20261023.json").read_text()
    rc, _, err = run(["split", str(j), "--by", "day", "--dir", str(outdir)])
    assert rc != 0 and "--force" in err, err
    assert (outdir / "scan-20261023.json").read_text() == keep
    assert run(["split", str(j), "--by", "day", "--dir", str(outdir), "--force"])[0] == 0


def test_split_and_merge_never_open_audio(tmp_path, run, monkeypatch):
    """The artifacts are the record: after a scan, the FLACs can go and the algebra still works."""
    import audioanalyzer.ingest as ingest
    d = tmp_path / "corpus"
    _rec(d, "20261023T000002.flac", tone=880.0)
    _rec(d, "20261024T000002.flac", tone=1400.0, seed=3)
    j = _scan(run, d, tmp_path / "s.json", tmp_path)
    (tmp_path / "away").mkdir()
    for f in d.iterdir():
        f.rename(tmp_path / "away" / f.name)
    rc, _, err = run(["split", str(j), "--by", "day", "--dir", str(tmp_path / "parts")])
    assert rc == 0, err
    parts = sorted((tmp_path / "parts").glob("scan-*.json"))
    rc, out, err = run(["merge", *[str(p) for p in parts], "-o", str(tmp_path / "m.json")])
    assert rc == 0, err
    assert len(load_run(tmp_path / "m.json").dets) == len(load_run(j).dets)


def test_split_granularity_rejects_nonsense(tmp_path, run):
    d = tmp_path / "c"
    _rec(d, "20261023T000002.flac", tone=880.0)
    j = _scan(run, d, tmp_path / "s.json", tmp_path)
    with pytest.raises(SystemExit) as exc:          # argparse refuses before main() sees a value
        run(["split", str(j), "--by", "week", "--dir", str(tmp_path / "p")])
    assert exc.value.code != 0
