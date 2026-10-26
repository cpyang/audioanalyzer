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

    @property
    def basename_frames(self) -> set[tuple[str, int]]:
        return {(Path(i.path).name, int(i.frames)) for i in self.infos}


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
              version=str((doc.get("tool") or {}).get("version", "?")))
    run.coverage = doc.get("coverage") or {}      # type: ignore[attr-defined]
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
