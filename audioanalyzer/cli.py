"""Command line surface (REQUIREMENTS.md FR-9).

M0 scope: ``scan`` performs ingest + A6 data-quality reporting and prints/writes MediaInfo.
Detectors (A1-A5), profile/compare, periodicity and the HTML report arrive in M1-M4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from . import __version__
from .config import Config, recall_preset
from .ingest import probe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audioanalyzer",
        description="Scan long-duration FLAC recordings for abnormal frequencies.",
    )
    p.add_argument("--version", action="version", version=f"audioanalyzer {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="analyse one or more recordings")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("-c", "--config", type=Path, default=None, help="TOML config file")
    s.add_argument("--mode", choices=["standard", "hr"], default="standard")
    s.add_argument("--min-prominence", type=float, default=None, dest="min_prominence_db")
    s.add_argument("--min-tone-s", type=float, default=None, dest="min_tone_s")
    s.add_argument("--recall", action="store_true", help="relax thresholds, triage everything")
    s.add_argument("--top", type=int, default=None)
    s.add_argument("--json", type=Path, default=None, help="write detections/summary JSON")
    s.add_argument("--csv", type=Path, default=None, help="write detections CSV")
    s.add_argument("--report", type=Path, default=None, help="write HTML report into DIR")
    s.add_argument("--tz", default=None)
    s.add_argument("--min-freq", type=float, default=None, dest="min_freq")
    s.add_argument("--max-freq", type=float, default=None, dest="max_freq")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--progress", action="store_true")

    v = sub.add_parser("version", help="print version")
    v.add_argument("--json", action="store_true")

    c = sub.add_parser("config", help="show the effective configuration")
    c.add_argument("--dump", action="store_true")

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_toml(args.config) if getattr(args, "config", None) else Config()
    for key in ("min_prominence_db", "min_tone_s", "top", "min_freq", "max_freq"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if getattr(args, "tz", None):
        cfg.tz = args.tz
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    if getattr(args, "recall", False):
        cfg = recall_preset(cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(json.dumps({"version": __version__}) if args.json else f"audioanalyzer {__version__}")
        return 0

    if args.command == "config":
        cfg = config_from_args(args)
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return 0

    cfg = config_from_args(args)
    paths: list[Path] = []
    for pattern in args.files:
        if pattern.is_dir():
            paths += sorted(pattern.glob("*.flac")) + sorted(pattern.glob("*.wav"))
        elif any(ch in str(pattern) for ch in "*?["):
            paths += sorted(Path().glob(str(pattern)))
        else:
            paths.append(pattern)

    summaries = []
    for path in paths:
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2
        try:
            info = probe(path, cfg, progress=args.progress)
        except (OSError, sf.SoundFileError) as exc:  # unreadable/undecodable -> exit 2
            print(f"error: cannot decode {path}: {exc}", file=sys.stderr)
            return 2
        flags = info.quality_flags(cfg)
        if not args.quiet:
            print(_render(info, flags, cfg))
        summaries.append({"media": info.to_dict(), "quality_flags": flags,
                          "detectors": "not-implemented-yet (M1)"})

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"tool": {"name": "audioanalyzer", "version": __version__},
                                         "config": cfg.to_dict(), "files": summaries},
                                        indent=2, default=str), encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.json}")
    return 0


def _render(info, flags: list[dict], cfg: Config) -> str:
    lines = [
        f"{info.path}",
        f"  format      {info.format}/{info.subtype} ({info.bits}-bit), {info.channels} ch, "
        f"{info.sample_rate} Hz, Nyquist {info.nyquist:.0f} Hz",
        f"  duration    {info.duration_s:.6f} s ({info.duration}), {info.frames} frames, "
        f"{info.size_bytes} bytes",
        f"  capture     {info.capture_start.isoformat() if info.capture_start else 'unknown (name not YYYYMMDDTHHMMSS)'} [{info.tz}]",
        f"  tags        {info.tags or '-'}",
        f"  level       peak {info.peak_dbfs:+.3f} dBFS (at-max {info.peak_count}, "
        f"|x|>=1LSB {info.clipped_samples}, flat-top {1000 * info.full_scale_run_s:.1f} ms), "
        f"RMS {info.rms_dbfs:.3f} dBFS, DC {info.dc_offset:+.6f} "
        f"({'' if info.dc_offset_dbfs is None else f'{info.dc_offset_dbfs:.2f} dBFS'})",
        f"  blocks      {len(info.block_rms_dbfs)} x {info.block_seconds:.1f} s RMS",
        f"  flags       {flags or 'none'}",
        f"  detectors   A1/A2 tonal: pending M1 (precision-first, "
        f">={cfg.min_prominence_db:g} dB over baseline, >={cfg.min_tone_s:g} s)",
    ]
    return "\n".join(lines)
