"""Command line surface (REQUIREMENTS.md FR-9).

M0 scope: ``scan`` performs ingest + A6 data-quality reporting and prints/writes MediaInfo.
Detectors (A1-A5), profile/compare, periodicity and the HTML report arrive in M1-M4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must happen before numpy/OpenBLAS loads (i.e. before the imports below), because the library
# reads its thread count once, at init, and a pool of workers each running N BLAS threads only
# thrashes shared cache: 6 files measured 120 s serial, 83.7 s at 6 default-threaded workers, and
# 68.6 s with one thread each — faster *and* using 32 % less total CPU.  setdefault, so an explicit
# `OMP_NUM_THREADS=4 audioanalyzer …` still wins.  Cost to a single-file scan: 14.6 -> 15.4 s.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

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
    s.add_argument("-J", "--jobs", type=int, default=0, metavar="N",
                   help="files to analyse in parallel; 0 = auto (min of cores/2 and free RAM/260MB)")
    s.add_argument("--fail-on", type=float, default=None, dest="fail_on",
                   help="exit 3 if any detection's confidence reaches this value (FR-8.4)")

    v = sub.add_parser("version", help="print version")
    v.add_argument("--json", action="store_true")

    d = sub.add_parser("doctor", help="check this environment BEFORE a long run (deps, threads, sizing)")
    d.add_argument("files", nargs="*", type=Path, default=[],
                   help="optional: the files/dir you are about to scan, to size the pool")
    d.add_argument("--json", type=Path, default=None)

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


_PER_WORKER_CORES = 2        # measured knee: throughput saturates near cores/2 (see _resolve_jobs)
_PER_WORKER_MEM_MB = 260.0   # measured: 210-225 MB peak per 16 kHz hour, plus slack
# 24 hour-long files in 269.6 s at auto sizing on a 12-core box.  Machine-specific by construction;
# doctor quotes it so a "how long will this take" question has a number attached, not a shrug.
_SECONDS_PER_FILE_PARALLEL = 11.2


def _mem_available_mb() -> float:
    """MemAvailable where the kernel exposes it; otherwise assume nothing is free (=> 1 job)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _resolve_jobs(requested: int, n_files: int) -> int:
    """How many files to analyse at once.  ``0`` = auto.

    Auto is ``cores/2``, and that number is measured rather than reasoned.  On a 12-core box, six
    hour-long files took 120 s serial, 83.7 s with six default-threaded workers, and 68.6 s with
    six workers whose BLAS threads are pinned (so pinning, not more workers, is what paid).  Going
    wider buys nothing: the whole day ran 269.6 s at 6 workers (576 % CPU) and 267.9 s at 12
    (962 % CPU) — same wall clock, 1.7× the CPU.  Throughput saturates on memory bandwidth at
    roughly half the cores, so oversubscribing spends electricity and cache to move nothing.

    Free RAM is the other bound: one worker measured 210-225 MB for an hour of 16 kHz, and the
    pool must not size itself into an OOM on a shared machine.
    """
    if n_files < 2:
        return 1
    if requested > 0:
        return min(requested, n_files)
    import os
    cores = max(1, (os.cpu_count() or 1) // _PER_WORKER_CORES)
    mem = max(1, int(_mem_available_mb() // _PER_WORKER_MEM_MB))
    return max(1, min(cores, mem, n_files, 16))


def _require_optional(module: str, extra: str, flag: str) -> int:
    """Check an optional dependency BEFORE the expensive work, not after it.

    On the reported machine a 24-file directory ran for 38 minutes, wrote CSV and JSON, and only
    then died on ``import matplotlib`` inside ``--report`` — the traceback was the first thing the
    user saw that the environment was incomplete.  Exit 1 (FR-8.4: usage/environment), with the
    command that fixes it.
    """
    import importlib.util
    if importlib.util.find_spec(module) is not None:
        return 0
    print(f"error: {flag} needs the '{module}' package, which is not installed here.\n"
          f"  fix the environment:  python -m pip install '.[{extra}]'\n"
          f"  or run without {flag} to get CSV/JSON only (they need no optional packages).",
          file=sys.stderr)
    return 1


def _preflight_readable(path: Path) -> int:
    """Open the header only — cheap, and it turns a 38-minute failure into an instant one."""
    try:
        with sf.SoundFile(str(path)):
            pass
    except (OSError, sf.SoundFileError) as exc:
        print(f"error: cannot decode {path}: {exc}", file=sys.stderr)
        return 2
    return 0


def _scan_worker(payload):
    """One file, in a child process.  Module-level so it pickles; renders its own text block."""
    path_str, cfg, progress = payload
    try:
        res = scan_file(Path(path_str), cfg, progress=progress)
    except (OSError, sf.SoundFileError) as exc:
        return {"path": path_str, "error": f"error: cannot decode {path_str}: {exc}"}
    except Exception as exc:                                # a worker crash must not hang the run
        return {"path": path_str,
                "error": f"error: worker failed on {path_str}: {type(exc).__name__}: {exc}"}
    flags = res.media.quality_flags(cfg)
    return {"path": path_str, "error": None, "media": res.media, "flags": flags,
            "detections": res.detections, "analysis": res.summary(),
            "json_dets": detections_json(res.detections),
            "text": _render(res, flags, cfg)}


def _scan_all(paths, cfg, *, jobs: int, quiet: bool, progress: bool):
    """Scan a selection, serially or across processes, and assemble it in *input* order.

    Ordering matters twice over: the CSV must be byte-identical to a serial run (NFR-6) and the
    per-file blocks must land in the same stdout sequence whichever path produced them.  With
    workers, results are buffered and printed in input order as soon as the next one is ready to
    show, while a one-line completion counter goes to stderr so a long run is not silent.
    """
    n = len(paths)
    results: list[dict | None] = [None] * n
    if jobs > 1 and n > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_scan_worker, (str(p), cfg, False)): i for i, p in enumerate(paths)}
            print(f"scanning {n} file(s) with {jobs} worker(s)", file=sys.stderr, flush=True)
            done = 0
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:                    # pragma: no cover - child killed
                    results[i] = {"path": str(paths[i]),
                                  "error": f"error: worker lost {paths[i]}: {exc}"}
                done += 1
                if not quiet:
                    tail = " FAILED" if results[i]["error"] else ""
                    print(f"  [{done}/{n}] {Path(results[i]['path']).name}{tail}",
                          file=sys.stderr, flush=True)
        return _assemble(results, quiet=quiet, stream=False)
    stop = n
    for i, path in enumerate(paths):
        results[i] = _scan_worker((str(path), cfg, progress))
        if results[i]["error"]:
            stop = i + 1                            # serial keeps the old fail-fast behaviour
            break
    return _assemble(results[:stop], quiet=quiet, stream=True)


def _assemble(results, *, quiet: bool, stream: bool):
    summaries, all_dets, infos, errors = [], [], [], []
    for r in results:
        if r is None:
            continue
        if r["error"]:
            errors.append(r["error"])
            continue
        all_dets += r["detections"]
        infos.append(r["media"])
        summaries.append({"media": r["media"].to_dict(), "quality_flags": r["flags"],
                          "analysis": r["analysis"], "detections": r["json_dets"]})
        if not quiet and stream:
            print(r["text"])
    if not stream and not quiet:
        for r in results:
            if r and not r["error"]:
                print(r["text"])
    return summaries, all_dets, infos, errors


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


def _doctor(files: list[Path], out_json: Path | None = None) -> int:
    """Report what this environment can and cannot do, before a run costs an hour to find out.

    Written because a clone on a fresh machine ran a whole day, then died on a missing optional
    package.  A scan is expensive; this check is not, and it answers the three questions that
    actually bite: is every import present, are BLAS threads pinned, and how many workers will
    the machine afford.
    """
    import importlib.util
    import multiprocessing

    n_files = 0
    for pattern in files:
        if pattern.is_dir():
            n_files += len(sorted(pattern.glob("*.flac")) + sorted(pattern.glob("*.wav")))
        elif pattern.exists():
            n_files += 1

    mods: dict[str, str] = {}
    problems: list[str] = []
    for name, role in (("numpy", "required"), ("scipy", "required"), ("soundfile", "required"),
                       ("matplotlib", "required by --report only"), ("pytest", "tests only")):
        spec = importlib.util.find_spec(name)
        if spec is None:
            mods[name] = "MISSING"
            if role.startswith("required") and "only" not in role:
                problems.append(f"{name} is not installed — install with:  pip install -e '.[report]'")
            elif name == "matplotlib":
                problems.append("matplotlib is not installed — `--report` will exit 1 (scan/CSV/JSON "
                                "still work); install with:  pip install -e '.[report]'")
        else:
            try:
                mods[name] = __import__(name).__version__
            except Exception:                            # pragma: no cover
                mods[name] = "importable, version unknown"
    try:
        import soundfile as _sf
        mods["libsndfile"] = _sf.__libsndfile_version__
    except Exception:                                    # pragma: no cover
        problems.append("could not read the bundled libsndfile version")

    py = sys.version_info
    if py < (3, 10):
        problems.append(f"python {py.major}.{py.minor} is below the supported floor (3.10)")
    elif py >= (3, 13):
        problems.append(f"python {py.major}.{py.minor} has no verified wheels for this stack "
                        "(§11 R9): use 3.12")
    if sys.version_info < (3, 11) and importlib.util.find_spec("tomli") is None:
        problems.append("python <3.11 needs the `tomli` backport for config/known TOML")

    threads = {k: os.environ.get(k, "unset") for k in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    mem = _mem_available_mb()
    jobs = _resolve_jobs(0, max(n_files, 2))
    doc = {
        "tool": __version__, "python": sys.version.split()[0], "modules": mods,
        "blas_threads": threads,
        "cpu_count": multiprocessing.cpu_count(),
        "mem_available_mb": round(mem),
        "per_worker_mb_estimate": _PER_WORKER_MEM_MB,
        "auto_jobs_for_this_machine": jobs,
        "files_found": n_files,
        "problems": problems,
    }
    if out_json:
        Path(out_json).write_text(json.dumps(doc, indent=2) + "\n")
    print(f"audioanalyzer {__version__} on python {doc['python']}")
    for k, v in mods.items():
        print(f"  {'OK ' if v != 'MISSING' else '!! '}{k:12s} {v}")
    print(f"  cpu(s)      {doc['cpu_count']}   BLAS threads pinned to {threads['OMP_NUM_THREADS']}")
    print(f"  memory      {doc['mem_available_mb']} MB available "
          f"(~{_PER_WORKER_MEM_MB:.0f} MB per worker)")
    if files:
        est = n_files * _SECONDS_PER_FILE_PARALLEL
        print(f"  your scan   {n_files} file(s) → {jobs} worker(s), "
              f"≈{int(est // 60)} m {int(est % 60):02d} s at this box's measured throughput "
              f"({_SECONDS_PER_FILE_PARALLEL} s/file at auto sizing)")
    for p in problems:
        print(f"  PROBLEM     {p}")
    # Exit 1 only when the tool cannot scan at all; a missing --report extra is a warning.
    return 1 if any(m == "MISSING" for k, m in mods.items()
                    if k in ("numpy", "scipy", "soundfile")) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(json.dumps({"version": __version__}) if args.json else f"audioanalyzer {__version__}")
        return 0

    if args.command == "doctor":
        return _doctor(list(args.files), args.json)

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

    if args.report:                                   # FR-8.4: fail before the expensive work
        rc = _require_optional("matplotlib", "report", "--report")
        if rc:
            return rc
    # Missing or undecodable input is cheap to discover and expensive to discover late: a 24-file
    # directory used to burn the whole run before erroring on the file that could not be opened.
    for path in paths:
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2
        rc = _preflight_readable(path)
        if rc:
            return rc

    jobs = _resolve_jobs(getattr(args, "jobs", 0), len(paths))
    summaries, all_dets, infos, errors = _scan_all(paths, cfg, jobs=jobs,
                                                   quiet=args.quiet, progress=args.progress)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 2

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
        try:
            from .report_html import build_report
        except ImportError as exc:                # pragma: no cover - broken/partial install
            print(f"error: --report could not load ({exc}); CSV/JSON above are complete",
                  file=sys.stderr)
            return 1
        out = Path(args.report)
        out = out / "report.html" if out.is_dir() or str(args.report).endswith("/") else out
        media = {s["media"]["path"]: s["media"] for s in summaries}
        try:
            p = build_report([str(x) for x in paths], all_dets, sources, coverage, cfg, out,
                             media=media, timeline=timeline)
        except Exception as exc:
            print(f"error: report generation failed after the analysis ({type(exc).__name__}: "
                  f"{exc}); the CSV and JSON are complete and valid", file=sys.stderr)
            return 1
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
