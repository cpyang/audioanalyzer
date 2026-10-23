"""Command line surface (REQUIREMENTS.md FR-9).

M0 scope: ``scan`` performs ingest + A6 data-quality reporting and prints/writes MediaInfo.
Detectors (A1-A5), profile/compare, periodicity and the HTML report arrive in M1-M4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .config import Config, recall_preset
from .corpus import build_timeline
from .detectors.sources import annotate_neighbours, group_sources
from .ingest import coverage_ledger
from .known import load_known, label, recurrence, upsert_known
from .models import Detection
from .profile import FloorProfile, compare
from .report import detections_json, tool_header, write_csv, write_json
from .scan import scan_file


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error; FR-8.4 reserves 2 for unreadable input, so 1 here."""

    def error(self, message):                      # pragma: no cover - exits
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
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
    s.add_argument("--class", dest="klass", choices=["tonal", "transient", "all"], default="all",
                   help="restrict which detectors contribute to the output (FR-9)")
    s.add_argument("--known", type=Path, default=None,
                   help="label detections against a known-source TOML (FR-7.3)")
    s.add_argument("--recall", action="store_true", help="relax thresholds, triage everything")
    s.add_argument("--no-events", action="store_true",
                   help="skip A3/A4 transient + periodicity analysis (tonal only)")
    s.add_argument("--top", type=int, default=None)
    s.add_argument("--json", type=Path, default=None, help="write detections/summary JSON")
    s.add_argument("--csv", type=Path, default=None, help="write detections CSV")
    s.add_argument("--report", type=Path, default=None, help="write HTML report into DIR")
    s.add_argument("--tz", default=None)
    s.add_argument("--min-freq", type=float, default=None, dest="min_freq")
    s.add_argument("--max-freq", type=float, default=None, dest="max_freq")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--progress", action="store_true")
    s.add_argument("--fail-on", type=float, default=None, dest="fail_on",
                   help="exit 3 if any detection's confidence reaches this value (FR-8.4)")

    v = sub.add_parser("version", help="print version")
    v.add_argument("--json", action="store_true")

    pr = sub.add_parser("profile", help="build a reference noise-floor profile (FR-6.1)")
    pr.add_argument("files", nargs="+", type=Path)
    pr.add_argument("-o", "--out", type=Path, default=Path("baseline.npz"))
    pr.add_argument("--mode", choices=["standard", "hr"], default="standard")
    pr.add_argument("--quiet", action="store_true")

    cp = sub.add_parser("compare", help="diff recordings against a floor profile (FR-6.2)")
    cp.add_argument("files", nargs="+", type=Path)
    cp.add_argument("--profile", "--baseline", dest="profile", type=Path, required=True)
    cp.add_argument("--mode", choices=["standard", "hr"], default="standard")
    cp.add_argument("--json", type=Path, default=None)
    cp.add_argument("--tz", default=None)
    cp.add_argument("--quiet", action="store_true")

    c = sub.add_parser("config", help="show the effective configuration")
    c.add_argument("--dump", action="store_true")

    k = sub.add_parser("known", help="show or extend the known-source library (FR-7.3)")
    k.add_argument("--list", action="store_true")
    k.add_argument("--add", nargs="+", metavar=("NAME", "F0_HZ"), default=None,
                   help="append a source: NAME F0_HZ [TOL_PCT] [HARMONICS]")
    k.add_argument("-f", "--file", type=Path, default=Path("known.toml"))

    pe = sub.add_parser("periodic", help="impulse-train analysis only (FR-9)")
    pe.add_argument("files", nargs="+", type=Path)
    pe.add_argument("--json", type=Path, default=None)
    pe.add_argument("--csv", type=Path, default=None)
    pe.add_argument("--report", type=Path, default=None)
    pe.add_argument("--tz", default=None)
    pe.add_argument("--top", type=int, default=None)
    pe.add_argument("--quiet", action="store_true")
    pe.set_defaults(command="scan", klass="transient", mode="standard", recall=False,
                    no_events=False, config=None, min_prominence_db=None, min_tone_s=None,
                    min_freq=None, max_freq=None, fail_on=None, progress=False)

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
    if getattr(args, "no_events", False):
        cfg.enable_events = False
    if getattr(args, "klass", "all") == "tonal":
        cfg.enable_events = False
    elif getattr(args, "klass", "all") == "transient":
        cfg.enable_tonal = False
    if getattr(args, "recall", False):
        cfg = recall_preset(cfg)
    return cfg


def _expand_paths(patterns):
    paths: list[Path] = []
    for pattern in patterns:
        if pattern.is_dir():
            paths += sorted(pattern.glob("*.flac")) + sorted(pattern.glob("*.wav"))
        elif any(ch in str(pattern) for ch in "*?["):
            paths += sorted(Path().glob(str(pattern)))
        else:
            paths.append(pattern)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(json.dumps({"version": __version__}) if args.json else f"audioanalyzer {__version__}")
        return 0

    if args.command == "profile":
        cfg = Config()
        if getattr(args, "mode", None):
            cfg.mode = args.mode
        paths = _expand_paths(args.files)
        try:
            prof = FloorProfile.from_files(paths, cfg)
        except (OSError, sf.SoundFileError) as exc:
            print(f"error: cannot build profile: {exc}", file=sys.stderr)
            return 2
        prof.save(args.out)
        if not args.quiet:
            print(f"profile      {len(paths)} reference recording(s), "
                  f"{prof.duration_s:.1f} s of audio")
            print(f"  written    {args.out}")
            print(f"  floor      p50 {np.median(prof.percentile(0.50)):.1f} dBFS median over bins, "
                  f"{prof.tone_candidates(cfg).size} tone(s) in the reference itself")
        return 0

    if args.command == "compare":
        cfg = Config()
        if getattr(args, "mode", None):
            cfg.mode = args.mode
        if getattr(args, "tz", None):
            cfg.tz = args.tz
        try:
            prof = FloorProfile.load(args.profile)
        except (OSError, ValueError, KeyError) as exc:
            print(f"error: cannot read profile {args.profile}: {exc}", file=sys.stderr)
            return 2
        out = []
        for path in _expand_paths(args.files):
            try:
                res = compare(path, prof, cfg)
            except (OSError, sf.SoundFileError) as exc:
                print(f"error: cannot decode {path}: {exc}", file=sys.stderr)
                return 2
            except ValueError as exc:                      # grid mismatch (FR-6 honesty rule)
                print(f"error: {path}: {exc}", file=sys.stderr)
                return 1
            out.append({"file": str(path), **res.to_dict()})
            if not args.quiet:
                print(_render_compare(path, res))
        if args.json:
            write_json({"tool": tool_header(cfg), "profile": {"path": str(args.profile),
                        "files": prof.files, "created_utc": prof.created_utc},
                        "comparisons": out}, args.json)
        return 0

    if args.command == "config":
        cfg = config_from_args(args)
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return 0

    if args.command == "known":
        srcs = load_known(args.file)
        if args.add:
            name, f0 = args.add[0], float(args.add[1])
            tol = float(args.add[2]) if len(args.add) > 2 else 1.0
            harm = int(args.add[3]) if len(args.add) > 3 else 5
            p = upsert_known(args.file, name, f0, tol, harm)
            print(f"known        added {name!r} f0={f0:g} Hz to {p}")
            return 0
        for s in srcs:
            print(f"  {s.f0_hz:10.2f} Hz  x{len(s.freqs):<3} {s.name}"
                  + (f"  — {s.note}" if s.note else ""))
        if not srcs:
            print(f"  (no known sources in {args.file})")
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

    summaries, all_dets, infos = [], [], []
    for path in paths:
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2
        try:
            res = scan_file(path, cfg, progress=args.progress)
        except (OSError, sf.SoundFileError) as exc:  # unreadable/undecodable -> exit 2
            print(f"error: cannot decode {path}: {exc}", file=sys.stderr)
            return 2
        flags = res.media.quality_flags(cfg)
        all_dets += res.detections
        infos.append(res.media)
        summaries.append({"media": res.media.to_dict(), "quality_flags": flags,
                          "analysis": res.summary(),
                          "detections": detections_json(res.detections)})
        if not args.quiet:
            print(_render(res, flags, cfg))

    # A5 grouping runs over the whole selection, not per file: a machine that whines through
    # six hourly chunks is one source, and that is the row an engineer walks toward (FR-5.1).
    annotate_neighbours(all_dets)                     # FR-4.7
    timeline = build_timeline(all_dets, cfg)          # FR-2.3
    sources = group_sources(all_dets, cfg)
    known = load_known(getattr(args, "known", None))
    n_labelled = label(all_dets, known)               # FR-7.3
    coverage = coverage_ledger(infos, cfg.gap_warn_s)
    rec = recurrence(all_dets, cfg)                   # FR-7.2
    if len(paths) > 1 and not args.quiet:
        print(_render_corpus(paths, sources, coverage, cfg, rec, n_labelled, timeline))

    if args.csv:
        n = write_csv(all_dets, args.csv)
        if not args.quiet:
            print(f"wrote {args.csv} ({n} rows)")
    if args.json:
        write_json({"tool": tool_header(cfg), "coverage": coverage,
                    "recurrence": rec, "timeline": timeline,
                    "sources": [s.to_dict() for s in sources], "files": summaries}, args.json)
        if not args.quiet:
            print(f"wrote {args.json}")
    if args.report:
        from .report_html import build_report
        out = Path(args.report)
        out = out / "report.html" if out.is_dir() or str(args.report).endswith("/") else out
        media = {s["media"]["path"]: s["media"] for s in summaries}
        p = build_report([str(x) for x in paths], all_dets, sources, coverage, cfg, out,
                         media=media, timeline=timeline)
        if not args.quiet:
            print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB, self-contained)")
    if args.fail_on and _max_severity(all_dets) >= args.fail_on:
        return 3
    return 0


def _render_corpus(paths, sources, coverage: dict, cfg: Config, rec: dict | None = None,
                   n_labelled: int = 0, timeline: list | None = None) -> str:
    lines = ["", f"corpus        {len(paths)} recording(s)"]
    if coverage.get("coverage_pct") is not None:
        lines.append(
            f"  coverage    {coverage['audio_s']:.1f} s audio in "
            f"{coverage['wall_s']:.1f} s wall = {coverage['coverage_pct']:.3f}% "
            f"({coverage['missing_s']:.1f} s NOT RECORDED across {coverage['n_gaps']} gap(s), "
            f"largest {coverage['max_gap_s']:.1f} s)")
    elif coverage.get("untimed"):
        lines.append(f"  coverage    undatable: {len(coverage['untimed'])} file(s) not named "
                     f"YYYYMMDDTHHMMSS -> no wall-clock gaps can be computed")
    lines.append(f"  sources     {len(sources)} noise source(s) after A5 grouping of "
                 f"{sum(s.n_harmonics for s in sources)} tone(s)")
    if sources:
        lines.append(f"    {'f0 Hz':>10}{'harm':>6}{'duty%':>8}{'eps':>5}  members")
        for s in sources[: cfg.top_n]:
            members = " ".join(f"{m.freq_hz:.0f}" for m in
                               sorted(s.members, key=lambda m: m.freq_hz)[:8])
            lines.append(f"    {s.f0_hz:10.2f}{s.n_harmonics:6d}{s.duty_pct:8.1f}"
                         f"{s.n_episodes:5d}  {members}")
    if timeline:
        cross = [e for e in timeline if e["n_detections"] > 1 and e["class"] in ("A1", "A2")]
        lines.append(f"  timeline    {len(timeline)} wall-clock feature(s) from "
                     f"{sum(e['n_detections'] for e in timeline)} per-file detection(s)"
                     + (f"; {len(cross)} of them continue across a chunk boundary" if cross else ""))
        for e in sorted(cross, key=lambda x: -x["n_detections"])[:10]:
            lines.append(f"    {e['id']}  {e['class']} {e['freq_hz']:8.2f} Hz  "
                         f"{e['t_start_wall'][5:16]}→{e['t_end_wall'][11:19]}  "
                         f"active {e['active_s']:7.1f} s over {e['n_files']} file(s)")
    if n_labelled:
        lines.append(f"  known       {n_labelled} detection(s) matched a known-source library entry")
    if rec and rec.get("clusters"):
        lines.append(f"  recurrence  {len(rec['clusters'])} recurring signature(s) over "
                     f"{len(rec.get('days', []))} day(s)"
                     + (f" — NOTE: {rec['note']}" if rec.get("note") else ""))
        lines.append(f"    {'freq Hz':>10}{'kind':>13}{'eps':>6}{'files':>7}{'median gap s':>14}"
                     f"{'jitter s':>10}{'hour':>7}{'spread h':>10}  classes")
        for c in rec["clusters"][: cfg.top_n]:
            lines.append(f"    {c['freq_hz']:10.2f}{c.get('kind','tone'):>13}{c['n_episodes']:6d}"
                         f"{c['files']:7d}"
                         f"{(c['median_gap_s'] if c['median_gap_s'] is not None else float('nan')):14.1f}"
                         f"{(c['gap_jitter_s'] if c['gap_jitter_s'] is not None else float('nan')):10.1f}"
                         f"{c['phase_of_day_hour']:7.2f}{c['phase_spread_h']:10.2f}  "
                         f"{','.join(c['class'])}{'  ' + c['known_name'] if c['known_name'] else ''}")
    return "\n".join(lines)


def _render_compare(path, res) -> str:
    lines = [f"{path}",
             f"  vs profile  {res.coverage_ref_s:.1f} s reference / {res.coverage_file_s:.1f} s here",
             f"  new tones   {len(res.new_tones)}"
             + (": " + ", ".join(f"{d.freq_hz:.1f} Hz (+{d.tone_noise_db:.1f} dB over its floor)"
                                 for d in res.new_tones[:6]) if res.new_tones else "")]
    if res.gone_tones:
        lines.append(f"  gone tones  {len(res.gone_tones)}: "
                     + ", ".join(f"{g['freq_hz']:.1f}" for g in res.gone_tones[:6]))
    lines.append(f"    {'band Hz':>14}{'reference':>11}{'this file':>11}{'delta dB':>10}")
    for b in res.to_dict()["bands"]:
        lines.append(f"    {b['lo_hz']:6.0f}-{b['hi_hz']:<7.0f}{b['ref_dbfs']:11.1f}"
                     f"{b['file_dbfs']:11.1f}{b['delta_db']:+10.1f}")
    return "\n".join(lines)


def _max_severity(dets: list[Detection]) -> float:
    return max([d.confidence for d in dets], default=0.0)


def _render(res, flags: list[dict], cfg: Config) -> str:
    info = res.media
    lines = [
        f"{info.path}",
        f"  format      {info.format}/{info.subtype} ({info.bits}-bit), {info.channels} ch, "
        f"{info.sample_rate} Hz, Nyquist {info.nyquist:.0f} Hz",
        f"  duration    {info.duration_s:.6f} s ({info.duration}), {info.frames} frames, "
        f"{info.size_bytes} bytes",
        f"  capture     {info.capture_start.isoformat() if info.capture_start else 'unknown (name not YYYYMMDDTHHMMSS)'} [{info.tz}]",
        f"  level       peak {_fmt_db(info.peak_dbfs)} dBFS (at-max {info.peak_count}, "
        f"top-code {info.clipped_samples}, flat-top {1000 * info.full_scale_run_s:.1f} ms), "
        f"RMS {_fmt_db(info.rms_dbfs)} dBFS, DC {info.dc_offset:+.6f}"
        f"{'' if info.dc_offset_dbfs is None else f' ({info.dc_offset_dbfs:.2f} dBFS)'}",
        f"  analysis    {res.agg.n_frames} frames, df {res.engine.df:.4f} Hz, "
        f"mode {cfg.mode}, {0 if res.candidates is None else len(res.candidates)} "
        f"candidate bins, {res.elapsed_s:.2f} s wall",
        f"  flags       {flags or 'none'}",
    ]
    tonal = [d for d in res.detections if d.cls in ("A1", "A2")]
    events = [d for d in res.detections if d.cls == "A4"]
    trains = [d for d in res.detections if d.cls == "A3"]
    if not tonal and not events:
        lines.append("  tonal       none above threshold "
                     f"(ttn >= {cfg.tone_noise_db:g} dB, >= {cfg.min_tone_s:g} s, "
                     f"coherence >= {cfg.min_coherence:g}, isolation >= {cfg.min_isolation_db:g} dB)")
        return "\n".join(lines)
    lines.append(f"  tonal       {len(tonal)} tone track(s), top {min(len(tonal), cfg.top_n)}:")
    if tonal:
        lines.append(f"    {'class':5}{'freq Hz':>11}{'ttn dB':>8}{'coh':>6}{'iso':>6}{'lvl dBFS':>10}"
                     f"{'sig dB':>8}{'start':>10}{'dur s':>9}{'pers%':>7}{'eps':>5}  drift")
        for d in tonal[: cfg.top_n]:
            lines.append(f"    {d.cls:5}{d.freq_hz:11.2f}{d.tone_noise_db:8.1f}{d.coherence:6.2f}"
                         f"{d.isolation_db:6.1f}{d.level_dbfs:10.1f}{d.level_std_db:8.1f}"
                         f"{_t_hms(d.t_start_s):>10}{d.duration_s:9.1f}{d.persistence_pct:7.1f}"
                         f"{d.n_episodes:5d}  {d.drift_class}({d.drift_hz_per_min:+.2f} Hz/min)")
    if events:
        loudest = max(events, key=lambda e: e.coherence)
        nb = len({tuple(e.extra.get("band", ())) for e in events})
        lines.append(f"  events      {len(events)} transient(s) in {nb} band(s); "
                     f"peak z {loudest.coherence:.1f} at {_t_hms(loudest.t_start_s)} "
                     f"({loudest.extra.get('kind', '?')}, "
                     f"{loudest.extra.get('rise_db', 0):.0f} dB rise)")
        for e in sorted(events, key=lambda e: -e.coherence)[:3]:
            sig = " ".join(f"{s['freq_hz']:.0f}Hz+{s['rise_db']:.0f}dB"
                           for s in e.extra.get("signature", []))
            b = e.extra.get("band", [0, 0])
            lines.append(f"    {_t_hms(e.t_start_s):>9} z{e.coherence:5.1f} "
                         f"{b[0]:7.0f}-{b[1]:<7.0f}{e.duration_s * 1000:7.0f} ms  {sig or e.extra.get('kind', '')}")
    for d in trains[:5]:
        jit = d.extra["jitter_s"]
        lines.append(f"  periodic    {d.extra['period_s']:.3f} s train in band "
                     f"{d.extra['band'][0]:.0f}-{d.extra['band'][1]:.0f} Hz: {d.extra['n_onsets']} hits, "
                     f"{d.extra['interval_fraction'] * 100:.0f}% of intervals, "
                     f"jitter {'n/a' if jit is None else f'{jit * 1000:.0f} ms'} "
                     f"({d.extra['rpm']:.1f} rpm-equivalent)")
    return "\n".join(lines)


def _fmt_db(v: float) -> str:
    """-inf renders as '-inf', not '+-inf' or 'nan' (all-zero files reach it, FR-1.5)."""
    return "-inf" if v == float("-inf") else (f"{v:+.3f}" if v == v else "n/a")


def _t_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
