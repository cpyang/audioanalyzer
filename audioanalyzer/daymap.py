"""The day map: "what happened, when", on one page (REQUIREMENTS.md FR-8.2).

Why this exists instead of a day-long spectrogram — measured on the reference corpus:

* **95.8 % of a day spectrogram's image variance is the noise floor's LF→HF slope** (-59.7 dBFS at
  20-125 Hz falling to -94.2 at 2-6 kHz).  That slope is microphone/preamp response, not site
  behaviour, and it would eat essentially the whole colour budget.  The residual that means
  anything spans only -3.1..+12.5 dB.
* **The resolution cannot work.**  The strongest tone finding of the whole day lasts **15.26 s**;
  giving it even one pixel across 24 h needs 5,661 columns, and the 900-column overview figure
  gives 96 s per pixel.  Two of the day's three mid-band tones are sub-pixel at day scale, so a
  day spectrogram would look flat and read as "nothing happened" - which is a false negative
  drawn in colour.

So the map draws **marks, not pixels**: a mark at a time position is scale-free, and a 15-second
event is as legible as a 40-minute one.  Cells are computed from *detections*, so generating the
map costs nothing beyond the scan; the spectrogram stays where it is useful, as ±30 s evidence
crops per detection.  Because it needs only detections, it can also be rebuilt from a previous
run's CSV or JSON (``audioanalyzer daymap``) — see load_artifact.

Two layouts, chosen by ``by_day``:

* ``by_day=False`` — one panel, x = wall clock from the first detection.  Right for a single day.
* ``by_day=True`` — one panel per calendar day, **x = hour of that day**, panels stacked and
  sharing the same horizontal ruler, so a feature at 03:00 lines up vertically across days and a
  difference *between* days is a shape difference in the same column.  This is what makes a
  multi-night survey readable: day-over-day comparison stops requiring you to remember where
  yesterday's 03:00 was.
"""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import Config
from .models import Detection

_BG = "#111418"
_FG = "#cfd6dd"
_GRID = "#2a3138"
_LANE_H = 18.0                      # pixels per frequency lane; --lane-h overrides
_LEFT = 96.0
_TOP = 42.0
_GAP_Y = 26.0                      # vertical space between day panels (by_day layout)
_DAY_LABEL_W = 0.0
_HOURS_FONTSIZE = 8.0

# severity (dB over the local floor, or robust z for impacts) -> colour.
# Monotonic in *luminance*, not just in hue: the first ramp ended in red (222,88,58), whose Rec.601
# luma is 147 — dimmer than the amber below it at 160.  A map where the loudest cell is the faintest
# one is worse than no map, so the top end goes to white-hot instead (a familiar convention, and
# strictly increasing: 29 -> 64 -> 110 -> 160 -> 211 -> 245).
_RAMP = [(0.0, (24, 30, 38)), (6.0, (32, 74, 96)), (12.0, (48, 140, 128)),
         (18.0, (206, 156, 62)), (26.0, (240, 214, 120)), (34.0, (250, 246, 230))]


def _colour(v: float) -> str:
    if not np.isfinite(v):
        return _BG
    xs = [p[0] for p in _RAMP]
    r = np.interp(v, xs, [p[1][0] for p in _RAMP])
    g = np.interp(v, xs, [p[1][1] for p in _RAMP])
    b = np.interp(v, xs, [p[1][2] for p in _RAMP])
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _severity(d: Detection) -> float:
    """How loud this is *relative to what normally sits there* — the only comparable unit."""
    if d.cls in ("A3", "A4"):
        return float(d.extra.get("z_peak", d.tone_noise_db))
    return float(d.tone_noise_db)


def _wall(d: Detection) -> datetime | None:
    return datetime.fromisoformat(d.t_start_wall) if d.t_start_wall else None


def _midnight(t: datetime) -> datetime:
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------- artifact loading

_EXTRA_FIELDS = ("z_peak", "rise_db", "band", "band_index", "bands_hot", "edges_hz", "frames",
                 "kind")


def load_artifact(path: str | Path) -> tuple[list[Detection], dict, Config]:
    """Recover (detections, coverage, config) from a previous run's JSON or CSV — no audio needed.

    JSON is the faithful route: it carries ``coverage`` (so the unrecorded hatching survives) and
    the config the run actually used, plus ``z_peak``, which is what colours an impact.  A CSV has
    no coverage block and no extras, so the map comes out without hatching and impacts fall back to
    tone-to-noise — usable, but the JSON is the artifact to keep.
    """
    p = Path(path)
    cfg = Config()
    if p.suffix.lower() == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        cov = doc.get("coverage", {}) or {}
        cfg_d = ((doc.get("tool") or {}).get("config")) or {}
        known = {f.name for f in cfg.__dataclass_fields__.values()}     # type: ignore[attr-defined]
        cfg = Config(**{k: v for k, v in cfg_d.items() if k in known})
        raw: list[dict] = []
        for fd in doc.get("files", []) or []:
            if isinstance(fd, dict):
                raw.extend(fd.get("detections", []) or [])
        raw.extend(doc.get("detections", []) or [])
        dets = [_from_json(r) for r in raw]
        return dets, cov, cfg
    with p.open(newline="", encoding="utf-8") as fh:
        dets = [_from_csv(r) for r in csv.DictReader(fh)]
    return dets, {}, cfg


def _from_json(r: dict) -> Detection:
    d = Detection()
    for k, v in r.items():
        if k == "cls":
            d.cls = v
        elif k in _EXTRA_FIELDS:
            d.extra[k] = v
        elif k == "class":
            d.cls = v
        elif hasattr(d, k):
            setattr(d, k, v)
        else:
            # Not a field, not a known extra: still something the run measured. Falling off the end of
            # this chain used to discard it silently — `nearest` and `signature` were lost this way, so
            # an artifact-derived report dropped an impact's signature while the scan's own report
            # kept it. Anything unrecognised goes to `extra`; detections_json flattens those back out,
            # which makes load -> save -> load a fixed point.
            d.extra[k] = v
    return d


def _from_csv(r: dict) -> Detection:
    d = Detection()
    num = {"freq_hz", "freq_std_hz", "bandwidth_hz", "t_start_s", "t_end_s", "duration_s",
           "level_dbfs", "peak_dbfs", "level_std_db", "tone_noise_db", "prominence_db",
           "coherence", "isolation_db", "persistence_pct", "longest_episode_s", "duty_pct",
           "drift_hz_per_min", "drift_r2", "source_f0_hz", "nn_freq_hz", "nn_ratio",
           "confidence"}
    ints = {"channel", "n_bins", "n_episodes", "n_harmonics"}
    for k, v in r.items():
        if v in (None, ""):
            continue
        try:
            if k in num:
                v = float(v)
            elif k in ints:
                v = int(float(v))
        except (TypeError, ValueError):
            continue
        if k == "class":
            d.cls = v
        elif hasattr(d, k):
            setattr(d, k, v)
    return d


# --------------------------------------------------------------------------- painting

def _grid_for(dets, edges, lanes, origin, ncol, bucket_s, by_day, day_start=None,
              min_sev=None):
    """Max severity per (lane, time bucket). ``by_day`` measures x from that day's midnight.

    Returns (grid, dropped) where dropped counts what the filters removed.

    Out-of-range detections are **dropped, never clamped**.  ``np.clip`` on the lane index is the
    tempting bug: with a 1-4 kHz window it would pile every sub-1 kHz sound in the file into the
    lowest visible lane and paint a hot band that does not exist.  A filter must be able to say
    "nothing here", which is only useful if it cannot lie about where the something went.
    """
    grid = np.full((lanes, ncol), -np.inf)
    dropped = {"freq": 0, "sev": 0}
    day0 = day_start if by_day else origin
    for d in dets:
        t = _wall(d)
        if t is None:
            continue
        band = int(np.searchsorted(edges, d.freq_hz, side="right") - 1)
        if band < 0 or band >= lanes:            # outside the drawn window: drop, do not clamp
            dropped["freq"] += 1
            continue
        v = _severity(d)
        if min_sev is not None and v < min_sev:
            dropped["sev"] += 1
            continue
        a = (t - day0).total_seconds()
        te = _end(d, t)
        b = (te - day0).total_seconds()
        if by_day:
            if not (day0 <= t < day0 + timedelta(hours=24)):
                continue                       # belongs to another panel
            a, b = max(a, 0.0), min(b, 86400.0)
        if not np.isfinite(a) or b <= a:
            continue
        i0 = max(0, int(a // bucket_s))
        i1 = min(ncol, int(np.ceil(b / bucket_s)) + 1)
        grid[band, i0:max(i0 + 1, i1)] = np.maximum(grid[band, i0:max(i0 + 1, i1)], v)
    return grid, dropped


def _end(d: Detection, t: datetime) -> datetime:
    if d.t_end_wall:
        try:
            return datetime.fromisoformat(d.t_end_wall)
        except ValueError:
            pass
    if d.duration_s:
        return t + timedelta(seconds=float(d.duration_s))
    return t + timedelta(seconds=1.0)


def build_day_map(dets: list[Detection], coverage: dict, cfg: Config, *,
                  lanes: int = 16, bucket_s: float = 60.0, lane_h: float = _LANE_H,
                  by_day: bool | None = None, top_labels: int = 8, cell_titles: bool = True,
                  freq_lo: float | None = None, freq_hi: float | None = None,
                  min_severity: float | None = None, stats: dict | None = None,
                  standalone: bool = False) -> str:
    """Return an SVG (inline in HTML, or standalone with ``standalone=True``).

    ``by_day`` defaults to True when the detections span more than one calendar day.

    ``freq_lo``/``freq_hi`` window the frequency axis (detections outside them are dropped, not
    folded into the edge lanes), and ``min_severity`` hides anything quieter than that many dB over
    the local floor — dB for tones, robust z for impacts, the same number the colour ramp uses. Both
    are **view filters**: the caption and the file both record what was filtered out, because a map
    that silently drops data is indistinguishable from a site that silently makes none.
    """
    timed = [d for d in dets if d.t_start_wall]
    if not timed:
        return ("<p class='meta'>Day map unavailable: the recordings carry no wall-clock time, "
                "so nothing can be placed on a timeline. Name files "
                "<code>YYYYMMDDTHHMMSS.flac</code> or pass <code>--tz</code>.</p>")
    nyq = 0.98 * 8000.0                                   # standard-mode analysis ceiling
    lo = float(cfg.min_freq if freq_lo is None else freq_lo)
    hi = float(min(nyq, 7600.0) if freq_hi is None else min(freq_hi, nyq))
    if not (0.0 < lo < hi):
        raise ValueError(f"day map frequency window must satisfy 0 < low < high, got {lo}-{hi} Hz")
    edges = np.geomspace(lo, hi, lanes + 1)
    starts = [_wall(d) for d in timed]
    t0 = min(starts)

    days = sorted({t.date() for t in starts})
    if by_day is None:
        by_day = len(days) > 1
    if by_day:
        ncol = int(np.ceil(86400.0 / bucket_s))
        panels = [(f"{d:%Y-%m-%d}", _midnight(datetime(d.year, d.month, d.day, tzinfo=t0.tzinfo)))
                  for d in days]
    else:
        span_s = max(60.0, max((_end(d, _wall(d)) - t0).total_seconds() for d in timed))
        # A blackout AFTER the last detection is still part of the picture (R10): the canvas has to
        # reach it, or the one thing the map exists to say goes unsaid. Measured: before this, a gap
        # later than every detection was dropped, and the old test passed only because the rect was
        # written at an x beyond the viewBox.
        for g in (coverage or {}).get("gaps", []) or []:
            gt = _parse(g.get("at"))
            if gt is not None and gt.tzinfo is not None:
                span_s = max(span_s, (gt - t0).total_seconds() + float(g.get("gap_s", 0.0)))
        ncol = int(np.ceil(span_s / bucket_s))
        panels = [("", t0)]

    body_h = lanes * lane_h
    tone_h = 16.0 if by_day else 0.0          # per-panel track strip; labels go once, at the bottom
    npanel = len(panels)
    pitch = body_h + tone_h + _GAP_Y          # top-to-top distance between day panels
    width = int(_LEFT + ncol + 18)
    height = int(_TOP + (npanel - 1) * pitch + body_h + 46 + (16 if by_day else 0))
    ns = "xmlns='http://www.w3.org/2000/svg' " if standalone else ""
    parts = [f"<svg {ns}"
             f"viewBox='0 0 {width} {height}' width='100%' "
             f"style='background:{_BG};border-radius:4px' role='img' "
             f"aria-label='Day map: detections by frequency band over "
             f"{'hour of day, one panel per day' if by_day else 'time'}, "
             f"{lo:.0f}-{hi:.0f} Hz"
             f"{'' if min_severity is None else f', {min_severity:g} dB over floor and above'}'>"]
    parts.append("<defs><pattern id='dm-norecorded' width='6' height='6' "
                 "patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
                 "<rect width='6' height='6' fill='#1b1f24'/>"
                 "<line x1='0' y1='0' x2='0' y2='6' stroke='#6b7682' stroke-width='2.2'/>"
                 "</pattern></defs>")

    kinds = {d.cls for d in timed}
    track_tags: dict[float, str] = {}
    n_drop = {"freq": 0, "sev": 0}
    painted = 0
    for pi, (dlab, origin) in enumerate(panels):
        top = _TOP + pi * pitch
        grid, dropped = _grid_for(timed, edges, lanes, t0, ncol, bucket_s, by_day,
                                  day_start=origin, min_sev=min_severity)
        if pi == 0:
            # Take the totals from one pass only. The band/severity tests run before the day test, so
            # panel 0's counts ARE the global counts; accumulating across panels would multiply the
            # headline number by the number of days (measured: 4 panels reported 54,784 dropped from
            # 14,660 detections).
            n_drop.update(dropped)
        painted += int(np.isfinite(grid).sum())

        # --- cells. No per-cell <title> in the multi-day layout: it tripled the file size for a
        # tooltip that repeats itself once per day, and a 30-day map needs to stay openable.
        titles = cell_titles and not by_day
        for r in range(lanes):
            y = top + r * lane_h
            row = np.flatnonzero(np.isfinite(grid[r]))
            for c in row:
                v = float(grid[r, c])
                ttl = (f"<title>lane {edges[r]:.0f}-{edges[r + 1]:.0f} Hz, "
                       f"{'hour ' + format_time(c * bucket_s) if by_day else f't+{c * bucket_s / 60:.0f} min'}, "
                       f"{v:.1f} over floor</title>") if titles else ""
                parts.append(f"<rect x='{_LEFT + c:.1f}' y='{y:.1f}' width='1.0' "
                             f"height='{lane_h:.1f}' fill='{_colour(v)}'>{ttl}</rect>")

        # --- hour grid. In by_day mode every panel gets the SAME ruler, which is the point.
        step = 3600.0 if (by_day or ncol * bucket_s <= 6 * 3600) else 6 * 3600.0
        t = 0.0
        while t <= ncol * bucket_s:
            x = _LEFT + t / bucket_s
            parts.append(f"<line x1='{x:.1f}' y1='{top:.1f}' x2='{x:.1f}' y2='{top + body_h:.1f}' "
                         f"stroke='{_GRID}' stroke-width='0.5'/>")
            lab = format_time(t) if by_day else f"{(origin + timedelta(seconds=t)):%H:%M}"
            ty = (_TOP - 6) if pi == 0 else (top - 4 if by_day else _TOP - 6)
            parts.append(f"<text x='{x:.1f}' y='{ty:.1f}' fill='{_FG}' "
                         f"font-size='{_HOURS_FONTSIZE if pi == 0 or not by_day else 7}' "
                         f"font-family='ui-monospace,monospace'>{lab}</text>")
            t += step
        parts.append(f"<rect x='{_LEFT:.1f}' y='{top:.1f}' width='{ncol}' height='{body_h:.1f}' "
                     f"fill='none' stroke='{_GRID}' stroke-width='0.6'/>")

        # --- coverage holes for THIS panel ("not recorded" is part of the picture; FR-2.4, R10)
        for g in (coverage or {}).get("gaps", []) or []:
            gt = _parse(g.get("at"))
            if gt is None:
                continue
            if by_day and gt.date() != origin.date():
                continue
            at = (gt - origin).total_seconds()
            if by_day:
                at = at % 86400.0
            if not np.isfinite(at) or at < 0 or at > ncol * bucket_s:
                continue
            w = max(1.0, float(g.get("gap_s", 5.0)) / bucket_s)
            parts.append(f"<rect x='{_LEFT + at:.1f}' y='{top:.1f}' width='{w:.1f}' "
                         f"height='{body_h:.1f}' fill='url(#dm-norecorded)' opacity='0.9'>"
                         f"<title>NOT RECORDED: {float(g.get('gap_s', 0.0)):.1f} s after "
                         f"{html.escape(str(g.get('after', '?')))}</title></rect>")

        # --- lane labels (per panel in the stacked layout, so each block is self-describing)
        for r in range(lanes):
            y = top + r * lane_h + lane_h * 0.78
            parts.append(f"<text x='{_LEFT - 6:.1f}' y='{y:.1f}' text-anchor='end' fill='{_FG}' "
                         f"font-size='{min(7.0, 0.55 * lane_h):.1f}' "
                         f"font-family='ui-monospace,monospace'>"
                         f"{edges[r]:.0f}-{edges[r + 1]:.0f}Hz</text>")

        # --- the day label sits left of its own panel
        if by_day:
            parts.append(f"<text x='{_LEFT - 6:.1f}' y='{top + 10:.1f}' text-anchor='end' "
                         f"fill='#8f9aa6' font-size='8.5' font-family='ui-monospace,monospace' "
                         f"transform='rotate(-90 {_LEFT - 6:.1f} {top + 10:.1f})'>{dlab}</text>")

        # --- tone tracks: a labelled mark per active stretch. In the stacked layout the marks go
        # in a strip under their own panel and the labels are written once, at the bottom — the
        # same track is usually present every day, so repeating its name 30 times is noise.
        tones = sorted([d for d in timed
                        if d.cls in ("A1", "A2") and lo <= d.freq_hz <= hi
                        and _in_panel(d, origin, by_day)],
                       key=lambda d: -float(d.confidence or 0.0))
        seen: list[float] = []
        row = 0
        for d in tones:
            if any(abs(d.freq_hz - f) < max(2.0, 0.005 * f) for f in seen):
                continue
            if len(seen) >= top_labels:
                break
            seen.append(d.freq_hz)
            t = _wall(d)
            mod = 86400.0 if by_day else None
            a = (t - origin).total_seconds()
            b = (_end(d, t) - origin).total_seconds()
            if mod:
                a, b = a % mod, b % mod
            if b < a:
                b = a + bucket_s
            x0, x1 = _LEFT + a / bucket_s, _LEFT + max(b, a + bucket_s) / bucket_s
            y = (top + body_h + 8.0) if by_day else (height - 26 + (row % 2) * 11)
            parts.append(f"<line x1='{x0:.1f}' y1='{y - 4:.1f}' x2='{x0:.1f}' "
                         f"y2='{top + body_h:.1f}' stroke='#5b6672' stroke-width='0.4'/>")
            parts.append(f"<rect x='{x0:.1f}' y='{y - 1.5:.1f}' width='{max(1.0, x1 - x0):.1f}' "
                         f"height='2.5' fill='#7fd1c8'/>")
            tag = (f"{d.freq_hz:.1f}Hz {d.level_dbfs:.0f}dBFS duty"
                   f"{'n/a' if d.duty_pct is None else f'{d.duty_pct:.0f}%'}"
                   f"{' — ' + html.escape(d.known_name) if d.known_name else ''}")
            if not by_day:
                parts.append(f"<text x='{x0 + 2:.1f}' y='{y:.1f}' fill='{_FG}' font-size='7' "
                             f"font-family='ui-monospace,monospace'>{tag}</text>")
                row += 1
            else:
                track_tags.setdefault(round(d.freq_hz, 1), tag)

    if by_day and track_tags:
        y = height - 12
        parts.append(f"<text x='{_LEFT - 6:.1f}' y='{y:.1f}' fill='{_FG}' font-size='7' "
                     f"font-family='ui-monospace,monospace'>tone tracks: "
                     + " · ".join(track_tags.values()) + "</text>")

    # --- legend: states the filters and what they removed, so the view cannot be misread as the record
    if stats is not None:
        stats.update(detections=len(timed), cells=painted, dropped_freq=n_drop["freq"],
                     dropped_sev=n_drop["sev"], drawn=len(timed) - n_drop["freq"] - n_drop["sev"],
                     freq_lo=lo, freq_hi=hi,
                     min_severity=min_severity, panels=len(panels), lanes=lanes)
    filt = f"{lo:.0f}-{hi:.0f} Hz"
    if min_severity is not None:
        filt += f", ≥{min_severity:g} over floor (z for impacts)"
    drop = ""
    if n_drop["freq"] or n_drop["sev"]:
        drop = (f" — filtered out: {n_drop['freq']:,} outside the band, "
                f"{n_drop['sev']:,} below the cut")
    parts.append(f"<text x='{_LEFT - 6:.1f}' y='{_TOP - 26:.1f}' fill='{_FG}' font-size='8' "
                 f"font-family='ui-monospace,monospace'>day map — max dB over the local floor "
                 f"(z for impacts), {filt}, {bucket_s / 60:.0f}-min buckets, "
                 f"{'one panel per day, aligned on hour of day' if by_day else 'x = wall clock'}, "
                 f"{', '.join(sorted(kinds))} present{drop}</text>")
    if painted == 0:
        # Everything was filtered away.  A blank canvas here is the exact false negative this tool
        # exists to avoid, so say it in the middle of the picture, not in a caption nobody reads.
        parts.append(f"<text x='{width / 2:.0f}' y='{_TOP + body_h / 2:.0f}' text-anchor='middle' "
                     f"fill='#e8a33a' font-size='11' font-family='ui-monospace,monospace'>"
                     f"EMPTY AFTER FILTERS: {len(timed):,} dated detections, 0 cells drawn"
                     f" ({n_drop['freq']:,} outside {lo:.0f}-{hi:.0f} Hz, "
                     f"{n_drop['sev']:,} under {min_severity if min_severity is not None else 0:g})"
                     f"</text>")
        parts.append(f"<text x='{width / 2:.0f}' y='{_TOP + body_h / 2 + 16:.0f}' "
                     f"text-anchor='middle' fill='{_FG}' font-size='9' "
                     f"font-family='ui-monospace,monospace'>this is a view setting, not a "
                     f"measurement of the site — widen the band or lower the cut</text>")
    for i, v in enumerate([4, 8, 12, 18, 26, 34]):
        x = _LEFT + 300 + i * 26
        parts.append(f"<rect x='{x:.1f}' y='{_TOP - 32:.1f}' width='22' height='8' "
                     f"fill='{_colour(v)}'/><text x='{x:.1f}' y='{_TOP - 18:.1f}' fill='{_FG}' "
                     f"font-size='6.5' font-family='monospace'>{v}</text>")
    x = _LEFT + 300 + 6 * 26 + 12
    parts.append(f"<rect x='{x:.1f}' y='{_TOP - 32:.1f}' width='22' height='8' "
                 f"fill='url(#dm-norecorded)'/><text x='{x:.1f}' y='{_TOP - 18:.1f}' fill='{_FG}' "
                 f"font-size='6.5' font-family='monospace'>not recorded</text>")
    parts.append("</svg>")
    return "".join(parts)


def _in_panel(d: Detection, origin: datetime, by_day: bool) -> bool:
    if not by_day:
        return True
    t = _wall(d)
    return t is not None and origin <= t < origin + timedelta(hours=24)


def _parse(s):
    try:
        return datetime.fromisoformat(s) if isinstance(s, str) else None
    except ValueError:
        return None


def format_time(secs: float) -> str:
    """Hour-of-day tick label: 00:00, 03:00, … (by_day layout)."""
    return f"{int(secs // 3600) % 24:02d}:00"
