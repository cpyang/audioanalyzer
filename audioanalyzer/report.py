"""CSV / JSON writers (REQUIREMENTS.md FR-8.1, FR-8.5).

Deterministic: rows are sorted by (t_start_s, freq_hz) and column order is fixed, so two runs
on the same input produce byte-identical files (NFR-6).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .models import CSV_COLUMNS, Detection


def write_csv(dets: list[Detection], path: str | Path) -> int:
    rows = sorted(dets, key=lambda d: (round(d.t_start_s, 6), round(d.freq_hz, 6), d.cls))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for d in rows:
            w.writerow(_fmt(d.to_row()))
    return len(rows)


def _fmt(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = "" if v == -999.0 else f"{v:.4f}".rstrip("0").rstrip(".") if v == v else ""
        else:
            out[k] = v
    return out


def write_json(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def detections_json(dets: list[Detection]) -> list[dict]:
    return [{k: v for k, v in asdict(d).items() if k != "extra"} | d.extra
            for d in sorted(dets, key=lambda d: (round(d.t_start_s, 6), round(d.freq_hz, 6)))]


def tool_header(cfg) -> dict:
    return {"name": "audioanalyzer", "version": __version__,
            "level_reference": "amplitude dBFS per-bin; RMS dBFS for whole-file fields",
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)}
