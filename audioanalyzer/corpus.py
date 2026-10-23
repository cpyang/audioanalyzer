"""Corpus-level timeline: merge detections that straddle a chunk boundary (FR-2.3, FR-7.1).

The recorder writes one file per hour, so a machine that runs from 06:52 to 07:14 is *two*
detections in the per-file output — same frequency, same level, contiguous in wall clock across a
~10 s blackout.  Reported twice, that reads as "the event happened twice", which is the exact
misleading inference this tool exists to prevent.

Merging happens only in the corpus view.  The per-file CSV keeps one row per detection (FR-8.1),
because those rows are what you join against a BMS log by file; ``timeline_entry`` ties them back
to the merged statement.

Two tolerances are deliberate:

* **Frequency.** ``max(recurrence_tol_hz, recurrence_tol_pct)`` — looser than the tone tracker's
  own precision, because chunk-to-chunk the interpolated peak of a real tone wanders by a
  fraction of a bin: the chunks start at different phases of it.
* **Time.** ``gap_s`` must exceed the largest measured inter-chunk blackout (26.7 s, §2.2) or a
  straddling tone splits again; 90 s keeps that headroom for filename jitter while staying short
  enough that a machine which genuinely runs twice an hour remains two features.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .config import Config
from .models import Detection


def _same_feature(a: Detection, b: Detection, cfg: Config) -> bool:
    if a.cls in ("A3", "A4") or b.cls in ("A3", "A4"):
        return False                      # events are not tracks; they are already atomic
    centre = 0.5 * (a.freq_hz + b.freq_hz)
    return abs(a.freq_hz - b.freq_hz) <= max(cfg.recurrence_tol_hz,
                                            cfg.recurrence_tol_pct / 100.0 * centre)


def build_timeline(dets: list[Detection], cfg: Config, gap_s: float = 90.0) -> list[dict]:
    """Merge per-file detections into wall-clock features (FR-2.3)."""
    for d in dets:
        d.timeline_entry = ""
    timed = sorted((d for d in dets if d.t_start_wall), key=lambda d: d.t_start_wall)
    if not timed:
        return []

    tol = max(cfg.recurrence_tol_hz, 1.0)
    horizon = timedelta(seconds=gap_s)
    live: dict[int, list[dict]] = {}          # freq bucket -> entries that can still grow
    entries: list[dict] = []

    for d in timed:
        t0 = datetime.fromisoformat(d.t_start_wall)
        t1 = datetime.fromisoformat(d.t_end_wall or d.t_start_wall)
        target = _find_open(live, d, t0, horizon, tol, cfg)
        if target is None:
            target = {"cls": d.cls, "freq_hz": d.freq_hz, "t_start": t0, "t_end": t1,
                      "active": timedelta(seconds=max(d.duration_s, 0.0)),
                      "files": {d.file}, "n_detections": 1, "levels": [d.level_dbfs],
                      "ttn": [d.tone_noise_db], "freqs": [d.freq_hz], "_members": [d]}
            entries.append(target)
            b0 = int(d.freq_hz // tol)
            for b in (b0 - 1, b0, b0 + 1):
                live.setdefault(b, []).append(target)
        else:
            target["t_end"] = max(target["t_end"], t1)
            target["active"] += timedelta(seconds=max(d.duration_s, 0.0))
            target["files"].add(d.file)
            target["n_detections"] += 1
            target["levels"].append(d.level_dbfs)
            target["ttn"].append(d.tone_noise_db)
            target["freqs"].append(d.freq_hz)
            target["_members"].append(d)
        _prune(live, t0 - horizon)

    out = []
    for i, e in enumerate(sorted(entries, key=lambda e: e["t_start"])):
        eid = f"tl-{i:04d}"
        for d in e["_members"]:
            d.timeline_entry = eid
        span = (e["t_end"] - e["t_start"]).total_seconds()
        out.append({
            "id": eid, "class": e["cls"],
            "freq_hz": round(sum(e["freqs"]) / len(e["freqs"]), 2),
            "t_start_wall": e["t_start"].isoformat(), "t_end_wall": e["t_end"].isoformat(),
            "span_s": round(span, 3),
            "active_s": round(e["active"].total_seconds(), 3),
            "duty_pct": round(100.0 * e["active"].total_seconds() / max(span, 1e-9), 2),
            "n_files": len(e["files"]), "n_detections": e["n_detections"],
            "level_dbfs": round(sum(e["levels"]) / len(e["levels"]), 2),
            "tone_noise_db": round(sum(e["ttn"]) / len(e["ttn"]), 2),
            "straddles_boundary": len(e["files"]) > 1,
            "files": sorted(Path(f).name for f in e["files"]),
        })
    return out


def _find_open(live: dict[int, list[dict]], d: Detection, t0: datetime, horizon: timedelta,
               tol: float, cfg: Config) -> dict | None:
    b0 = int(d.freq_hz // tol)
    for b in (b0 - 1, b0, b0 + 1):
        for e in live.get(b, ()):
            if t0 - horizon > e["t_end"] or t0 < e["t_start"]:
                continue
            if _same_feature(d, e["_members"][0], cfg):
                return e
    return None


def _prune(live: dict[int, list[dict]], older_than: datetime) -> None:
    """Drop entries that can no longer grow, so the scan stays linear in detections."""
    for b in list(live):
        keep = [e for e in live[b] if e["t_end"] >= older_than]
        if keep:
            live[b] = keep
        else:
            del live[b]
