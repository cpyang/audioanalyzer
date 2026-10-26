"""Reading, merging and reporting on previous runs WITHOUT touching audio (FR-8.7, FR-8.8).

Why this is possible at all: every corpus view in the tool already takes detections plus per-file
media metadata. `coverage_ledger` needs only `capture_start`/`duration_s`, both of which the JSON
stores; timeline, sources, recurrence and the day map need only detections. The FLAC was never
required by the analysis — only by the figures, which FR-8.2c caches at scan time.

The two rules that keep a merge honest:

* **Identity is (basename, frames, class, frequency, start time), never the stored path.**
  `Detection.file` holds the path exactly as typed, so scanning `~/src/audioanalyzer` and then `./`
  describes the same recording under two strings. Path-keyed dedupe would double every detection in
  the merged census, the timeline and the recurrence table — and the report would still look
  internally consistent. A coarse key (for example one that buckets the start time to a minute) is
  worse: it collides detections from a single run with each other, which cost 47 % of one census
  before this was fixed.
* **Corpus views are recomputed, never merged.** Two `sources` blocks each contain a feature that
  straddles their run boundary; concatenating them counts it twice and invents episodes. The merged
  detection set is the only input to the derived views.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Config
from .ingest import MediaInfo
from .models import Detection
from .report import detections_json

# Fields whose disagreement between two artifacts means they did not see the same audio the same way.
_MATERIAL = ("cls", "freq_hz", "level_dbfs", "tone_noise_db", "t_start_s", "duration_s")
# Config keys that gate a detection: differing values make confidences non-comparable.
_GATING = ("tone_noise_db", "min_prominence_db", "min_coherence", "min_isolation_db",
           "min_persistence_pct", "min_tone_s", "mode", "min_freq", "max_freq",
           "nyquist_guard", "transient_sigma", "min_event_db")


@dataclass
class Run:
    """One scan artifact: its detections, the files it covered, and how it was configured."""
    path: str
    dets: list[Detection] = field(default_factory=list)
    infos: list[MediaInfo] = field(default_factory=list)
    cfg: Config = field(default_factory=Config)
    version: str = ""
    blocks: list[dict] = field(default_factory=list)   # raw files[] entries (R17: lossless)
    coverage: dict = field(default_factory=dict)

    @property
    def basename_frames(self) -> set[tuple[str, int]]:
        return {(Path(i.path).name, int(i.frames)) for i in self.infos}

    def block_of(self, d: Detection) -> dict | None:
        """The raw files[] entry a detection came from, matched by media path."""
        want = Path(str(d.file)).name
        for b in self.blocks:
            if Path(str((b.get("media") or {}).get("path", b.get("file", "")))).name == want:
                return b
        return None


def load_run(path: str | Path) -> Run:
    """Recover a Run from `scan.json`. Raises ValueError if it is not a scan artifact."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or "files" not in doc:
        raise ValueError(f"{p} is not a scan artifact (no 'files' block). Pass the --json output of "
                         f"a scan, not the CSV.")
    from .daymap import _from_json          # single implementation of "dict -> Detection"
    cfg_doc = (doc.get("tool") or {}).get("config") or {}
    known = {f for f in Config.__dataclass_fields__}
    cfg = Config(**{k: v for k, v in cfg_doc.items() if k in known})
    infos = [_info(m) for m in ((f.get("media") or {}) for f in doc["files"] if isinstance(f, dict)) if m]
    dets: list[Detection] = []
    for f in doc["files"]:
        if isinstance(f, dict):
            dets.extend(_from_json(r) for r in (f.get("detections") or []))
    dets.extend(_from_json(r) for r in (doc.get("detections") or []))
    run = Run(path=str(p), dets=dets, infos=infos, cfg=cfg,
              version=str((doc.get("tool") or {}).get("version", "?")),
              blocks=[b for b in doc["files"] if isinstance(b, dict)],
              coverage=doc.get("coverage") or {})
    return run


def _info(m: dict) -> MediaInfo:
    """MediaInfo from a JSON media block. `capture_start` must come back as a datetime, or the
    coverage ledger compares it against a string and raises (a crash this path used to have)."""
    fields = {f for f in MediaInfo.__dataclass_fields__}
    kw = {k: v for k, v in m.items() if k in fields}
    cs = m.get("capture_start")
    if isinstance(cs, str) and cs:
        try:
            kw["capture_start"] = datetime.fromisoformat(cs)
        except ValueError:
            kw["capture_start"] = None
    elif "capture_start" in kw:
        kw["capture_start"] = None
    return MediaInfo(**kw)


def identity(d: Detection, frames: int | None) -> tuple:
    """(basename, frames, class, frequency, start) — the merge's idea of "the same detection".

    Deliberately NOT `key()`. That helper buckets the start time to a whole minute for cross-run
    diffing, and using it here made detections from a SINGLE artifact collide with each other:
    measured, merging one 5,023-detection run came out 2,634-strong, because two impacts in the same
    minute at the same band centre are distinct findings, and 47 % of the census quietly vanished
    with nothing in the report disagreeing with itself. Merge identity must therefore be exact.
    NFR-6 guarantees a re-scan reproduces the same floats, so exactness costs nothing across runs.
    """
    return (Path(str(d.file)).name, frames if frames is not None else Path(str(d.file)).name,
            d.cls, round(float(d.freq_hz), 4), round(float(d.t_start_s), 6))


def merge_runs(runs: list[Run], *, allow_mixed: bool = False) -> dict:
    """Combine runs into one detection set + file list, with the checks FR-8.7 demands.

    Returns {dets, infos, cfg, provenance, stats, conflicts, gating_diff}.
    Raises ValueError on a threshold disagreement unless allow_mixed.
    """
    if not runs:
        raise ValueError("no artifacts to merge")
    # frame counts by basename, so identity does not depend on how the path was typed
    frames_by_base: dict[str, int | None] = {}
    for r in runs:
        for i in r.infos:
            frames_by_base.setdefault(Path(i.path).name, int(i.frames))
    by_id: dict[tuple, Detection] = {}
    conflicts = 0
    for r in runs:
        for d in r.dets:
            k = identity(d, frames_by_base.get(Path(str(d.file)).name))
            prev = by_id.get(k)
            if prev is None:
                by_id[k] = d
                continue
            if any(getattr(prev, f, None) != getattr(d, f, None) for f in _MATERIAL):
                conflicts += 1        # same identity, different substance: report it, keep first
    # file records, deduped by (basename, frames). Sorted LAST, by capture time, so the merged
    # output cannot depend on the order artifacts were named in — the first-seen order did, which
    # made `report a.json b.json` and `report b.json a.json` differ by a filename in the table.
    seen: set[tuple[str, int]] = set()
    infos: list[MediaInfo] = []
    for r in runs:
        for i in r.infos:
            key = (Path(i.path).name, int(i.frames))
            if key in seen:
                continue
            seen.add(key)
            infos.append(i)
    infos.sort(key=lambda i: (i.capture_start is None,
                              i.capture_start or datetime.min.replace(tzinfo=None),
                              Path(i.path).name))
    gating_diff = sorted({k for k in _GATING
                          for a, b in ((runs[0].cfg.to_dict(), r.cfg.to_dict()) for r in runs[1:])
                          if a.get(k) != b.get(k)})
    if gating_diff and not allow_mixed:
        raise ValueError(
            "artifacts were produced under different thresholds: "
            + ", ".join(f"{k} ({runs[0].cfg.to_dict()[k]} vs "
                        f"{next(r.cfg.to_dict()[k] for r in runs[1:] if r.cfg.to_dict()[k] != runs[0].cfg.to_dict()[k])})"
                        for k in gating_diff)
            + " — a single confidence column across different gates is a false comparison. "
              "Re-scan under one config, or pass --allow-mixed to merge and label per section.")
    dets = sorted(by_id.values(), key=_order_key)
    return {"dets": dets, "infos": infos, "cfg": runs[0].cfg,
            "provenance": [{"path": r.path, "version": r.version, "files": len(r.infos),
                            "detections": len(r.dets)} for r in runs],
            "stats": {"runs": len(runs), "unique": len(dets), "conflicts": conflicts,
                      "gating_diff": gating_diff},
            "conflicts": conflicts, "gating_diff": gating_diff}


def _order_key(d: Detection):
    """Deterministic order independent of argument order (NFR-6 across artifacts)."""
    return (d.t_start_wall or "zzz", Path(str(d.file)).name, round(float(d.freq_hz), 4), d.cls)


def scanned_set(runs: list[Run]) -> set[tuple[str, int]]:
    """What has already been analysed — derived from the artifacts, which is why there is no state
    file to rot (FR-8.8)."""
    out: set[tuple[str, int]] = set()
    for r in runs:
        out |= r.basename_frames
    return out


# --------------------------------------------------------------------- artifact algebra (FR-8.9)

def _blocks_for(runs: list[Run], dets: list[Detection]) -> list[dict]:
    """Rebuild files[] entries for a detection set, carrying each source block's metadata verbatim.

    The per-file `analysis` and `quality_flags` are copied, never recomputed: they record what the
    decode actually did (DC offset, clipping, sample counts), and re-deriving them would need the
    audio. Detections are re-pointed at their block so a file's rows stay grouped (R17).
    """
    owner: dict[tuple, Run] = {}
    for r in runs:
        for d in r.dets:
            owner.setdefault(identity(d, None), r)
    grouped: dict[str, list[Detection]] = {}
    src_of: dict[str, dict] = {}
    for d in dets:
        r = owner.get(identity(d, None))
        src = r.block_of(d) if r is not None else None
        key = (src or {}).get("media", {}).get("path", str(d.file))
        grouped.setdefault(key, []).append(d)
        if src and key not in src_of:
            src_of[key] = src
    blocks = []
    for key in sorted(grouped):
        src = src_of.get(key, {})
        blocks.append({
            "file": key,
            "media": src.get("media") or {"path": key},
            "analysis": src.get("analysis") or {},
            "quality_flags": src.get("quality_flags") or [],
            # detections_json, NOT Detection.to_row(): to_row emits only the CSV columns and drops
            # `extra`, which carries z_peak — the number that colours an impact in the day map. A
            # merged artifact built that way would have reloaded fine and drawn the wrong picture.
            "detections": detections_json(grouped[key]),
        })
    return blocks


def artifact_doc(runs: list[Run], dets: list[Detection], infos: list[MediaInfo], cfg: Config,
                 *, provenance: list[dict] | None = None) -> dict:
    """A complete, re-loadable artifact for a merged or split detection set.

    Views are recomputed here rather than carried over, which is what makes a part's `sources` list
    describe that part (FR-8.9).
    """
    from .corpus import build_timeline
    from .detectors.sources import group_sources
    from .ingest import coverage_ledger
    from .known import recurrence
    from .report import tool_header
    timeline = build_timeline(dets, cfg)
    sources = group_sources(dets, cfg)
    coverage = coverage_ledger(infos, cfg.gap_warn_s)
    doc = {"tool": tool_header(cfg), "coverage": coverage, "recurrence": recurrence(dets, cfg),
           "timeline": timeline, "sources": [s.to_dict() for s in sources],
           "files": _blocks_for(runs, dets)}
    if provenance is not None:
        doc["provenance"] = provenance
    return doc


def bucket_of(dt: datetime, by: str) -> str:
    """Chronologically sortable bucket label: hour 20261023T07, day 20261023, month 2026-10."""
    if by == "hour":
        return f"{dt:%Y%m%dT%H}"
    if by == "day":
        return f"{dt:%Y%m%d}"
    if by == "month":
        return f"{dt:%Y-%m}"
    raise ValueError(f"unknown split granularity {by!r} (use hour, day or month)")


def split_plan(run: Run, by: str) -> tuple[dict[str, list[Detection]], list[str]]:
    """Group a run's detections into buckets by their RECORDING's capture time.

    Returns (buckets, notes). Recording-atomic by design: `t_start_s` is file-relative and the
    evidence figures seek into that file at that offset, so a detection cannot be cut across two
    artifacts without making its own row and its picture disagree. A recording that straddles a
    boundary therefore stays whole in its start bucket and is named in the notes.
    """
    starts: dict[str, datetime] = {}
    for i in run.infos:
        if i.capture_start is not None:
            starts[Path(i.path).name] = i.capture_start
    buckets: dict[str, list[Detection]] = {}
    straddlers: dict[str, set[str]] = {}
    for d in run.dets:
        base = Path(str(d.file)).name
        anchor = starts.get(base)
        if anchor is None:
            # no capture time (unnamed file): bucket by the detection's own wall clock, else dump
            t = d.t_start_wall
            anchor = datetime.fromisoformat(t) if t else None
            if anchor is None:
                buckets.setdefault("undated", []).append(d)
                continue
        b = bucket_of(anchor, by)
        buckets.setdefault(b, []).append(d)
        end = d.t_end_wall or d.t_start_wall
        if end and d.t_start_wall:
            try:
                if bucket_of(datetime.fromisoformat(end), by) != b:
                    straddlers.setdefault(b, set()).add(base)
            except ValueError:
                pass
    notes = [f"bucket {k}: {', '.join(sorted(bases))} straddle(s) the boundary, kept whole in that "
             f"bucket" for k, bases in sorted(straddlers.items())]
    return buckets, notes


def write_artifact(doc: dict, path: str | Path, *, force: bool = False) -> Path:
    """Write an artifact, refusing to clobber one (FR-8.9: these files are --incremental's memory)."""
    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(f"{p} already exists — pass --force to overwrite an artifact")
    from .report import write_json
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(doc, p)
    return p
