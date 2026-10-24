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
crops per detection.
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta

import numpy as np

from .config import Config
from .models import Detection

_BG = "#111418"
_FG = "#cfd6dd"
_GRID = "#2a3138"
_LANE_H = 13.0
_LEFT = 96.0
_TOP = 42.0
_HOURS_FONTSIZE = 8.0

# severity (dB over the local floor, or robust z for impacts) -> colour.
# Monotonic in *luminance*, not just in hue: the first ramp ended in red (222,88,58), whose Rec.601
# luma is 147 — dimmer than the amber below it at 160.  A map where the loudest cell is the faintest
# one is worse than no map, so the top end goes to white-hot instead (a familiar convention, and
# strictly increasing: 29 -> 64 -> 111 -> 160 -> 211 -> 245).
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


def build_day_map(dets: list[Detection], coverage: dict, cfg: Config, *,
                  lanes: int = 16, bucket_s: float = 60.0,
                  top_labels: int = 8, standalone: bool = False) -> str:
    """Return an SVG (inline in HTML, or standalone with ``standalone=True``)."""
    timed = [d for d in dets if d.t_start_wall]
    if not timed:
        return ("<p class='meta'>Day map unavailable: the recordings carry no wall-clock time, "
                "so nothing can be placed on a timeline. Name files "
                "<code>YYYYMMDDTHHMMSS.flac</code> or pass <code>--tz</code>.</p>")
    t0 = min(datetime.fromisoformat(d.t_start_wall) for d in timed)
    nyq = 0.98 * 8000.0                                   # standard-mode analysis ceiling
    edges = np.geomspace(cfg.min_freq, min(nyq, 7600.0), lanes + 1)
    span_s = max(60.0, max((datetime.fromisoformat(d.t_end_wall or d.t_start_wall) - t0)
                           .total_seconds() for d in timed))
    ncol = int(np.ceil(span_s / bucket_s))
    width = int(_LEFT + ncol + 18)
    height = int(_TOP + lanes * _LANE_H + 46)

    grid = np.full((lanes, ncol), -np.inf)
    kinds: set[str] = set()
    for d in timed:
        a = (datetime.fromisoformat(d.t_start_wall) - t0).total_seconds()
        b = (datetime.fromisoformat(d.t_end_wall or d.t_start_wall) - t0).total_seconds()
        if not np.isfinite(a):
            continue
        lo, hi = max(0.0, a), max(a + bucket_s, b)
        band = int(np.clip(np.searchsorted(edges, d.freq_hz, side="right") - 1, 0, lanes - 1))
        i0, i1 = int(lo // bucket_s), min(ncol, int(np.ceil(hi / bucket_s)) + 1)
        v = _severity(d)
        grid[band, max(0, i0):max(i0 + 1, i1)] = np.maximum(
            grid[band, max(0, i0):max(i0 + 1, i1)], v)
        kinds.add(d.cls)

    ns = "xmlns='http://www.w3.org/2000/svg' " if standalone else ""
    parts = [f"<svg {ns}"
             f"viewBox='0 0 {width} {height}' width='100%' "
             f"style='background:{_BG};border-radius:4px' role='img' "
             f"aria-label='Day map: detections by frequency band over time'>"]
    parts.append("<defs><pattern id='dm-norecorded' width='6' height='6' "
                 "patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
                 "<rect width='6' height='6' fill='#1b1f24'/>"
                 "<line x1='0' y1='0' x2='0' y2='6' stroke='#6b7682' stroke-width='2.2'/>"
                 "</pattern></defs>")
    body_h = lanes * _LANE_H

    # --- cells ---
    for r in range(lanes):
        y = _TOP + r * _LANE_H
        for c in range(ncol):
            v = grid[r, c]
            if not np.isfinite(v):
                continue
            parts.append(f"<rect x='{_LEFT + c:.1f}' y='{y:.1f}' width='1.0' height='{_LANE_H:.1f}' "
                         f"fill='{_colour(float(v))}'>"
                         f"<title>lane {edges[r]:.0f}-{edges[r + 1]:.0f} Hz, "
                         f"t+{c * bucket_s / 60:.0f} min, {float(v):.1f} over floor</title></rect>")

    # --- hour grid + labels ---
    step = 3600.0 if span_s <= 6 * 3600 else 6 * 3600.0
    t = 0.0
    while t <= span_s:
        x = _LEFT + t / bucket_s
        parts.append(f"<line x1='{x:.1f}' y1='{_TOP:.1f}' x2='{x:.1f}' "
                     f"y2='{_TOP + body_h:.1f}' stroke='{_GRID}' stroke-width='0.5'/>")
        parts.append(f"<text x='{x:.1f}' y='{_TOP - 6:.1f}' fill='{_FG}' "
                     f"font-size='{_HOURS_FONTSIZE}' font-family='ui-monospace,monospace'>"
                     f"{(t0 + timedelta(seconds=t)).strftime('%H:%M')}"
                     f"</text>")
        t += step

    # --- coverage holes: "not recorded" is part of the picture (FR-2.4, R10) ---
    for g in coverage.get("gaps", []) or []:
        try:
            at = (datetime.fromisoformat(g["at"]) - t0).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(at):
            continue
        x = _LEFT + at / bucket_s
        w = max(1.0, float(g.get("gap_s", 5.0)) / bucket_s)
        parts.append(f"<rect x='{x:.1f}' y='{_TOP:.1f}' width='{w:.1f}' height='{body_h:.1f}' "
                     f"fill='url(#dm-norecorded)' opacity='0.9'>"
                     f"<title>NOT RECORDED: {float(g.get('gap_s', 0.0)):.1f} s after "
                     f"{html.escape(str(g.get('after', '?')))}</title></rect>")

    # --- lane labels ---
    for r in range(lanes):
        y = _TOP + r * _LANE_H + _LANE_H * 0.78
        parts.append(f"<text x='{_LEFT - 6:.1f}' y='{y:.1f}' text-anchor='end' fill='{_FG}' "
                     f"font-size='7' font-family='ui-monospace,monospace'>"
                     f"{edges[r]:.0f}-{edges[r + 1]:.0f}Hz</text>")

    # --- top tone tracks, labelled (a mark is readable where a band of pixels is not) ---
    tones = [d for d in timed if d.cls in ("A1", "A2")]
    tones.sort(key=lambda d: -float(d.confidence or 0.0))
    seen: list[tuple[float, float]] = []
    row = 0
    for d in tones:
        if any(abs(d.freq_hz - f) < max(2.0, 0.005 * f) for f, _ in seen):
            continue
        if len(seen) >= top_labels:
            break
        seen.append((d.freq_hz, row))
        a = (datetime.fromisoformat(d.t_start_wall) - t0).total_seconds()
        b = (datetime.fromisoformat(d.t_end_wall or d.t_start_wall) - t0).total_seconds()
        x0, x1 = _LEFT + a / bucket_s, _LEFT + max(b, a + bucket_s) / bucket_s
        y = height - 26 + (row % 2) * 11
        parts.append(f"<line x1='{x0:.1f}' y1='{y - 4:.1f}' x2='{x0:.1f}' "
                     f"y2='{_TOP + body_h:.1f}' stroke='#5b6672' stroke-width='0.4'/>")
        parts.append(f"<rect x='{x0:.1f}' y='{y - 1.5:.1f}' width='{max(1.0, x1 - x0):.1f}' "
                     f"height='2.5' fill='#7fd1c8'/>")
        parts.append(f"<text x='{x0 + 2:.1f}' y='{y:.1f}' fill='{_FG}' font-size='7' "
                     f"font-family='ui-monospace,monospace'>{d.freq_hz:.1f}Hz "
                     f"{d.level_dbfs:.0f}dBFS duty{'n/a' if d.duty_pct is None else f'{d.duty_pct:.0f}%'}"
                     f"{' — ' + html.escape(d.known_name) if d.known_name else ''}</text>")
        row += 1

    # --- legend ---
    lx = _LEFT
    parts.append(f"<text x='{lx - 6:.1f}' y='{_TOP - 26:.1f}' fill='{_FG}' font-size='8' "
                 f"font-family='ui-monospace,monospace'>day map — max dB over the local floor "
                 f"(z for impacts), {bucket_s / 60:.0f}-min buckets, "
                 f"{', '.join(sorted(kinds))} present</text>")
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
