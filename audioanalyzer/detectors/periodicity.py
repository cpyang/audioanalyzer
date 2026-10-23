"""A3: periodic impulse trains (REQUIREMENTS.md FR-4.5).

A single click says "something hit".  The same click every 1.42 s says "a fan blade, a bearing
race, a pump stroke" — the repetition rate *is* the machine's identity, and it is the one number
in this whole tool that maps directly onto a nameplate (blades x rpm / 60).

So the detector takes A4's onset times per band and autocorrelates the impulse series over
0.5 s .. 3600 s.  Autocorrelation of event times rather than of the signal itself, because a
bearing knocks for 40 ms out of every 1400; correlating samples would spend all its effort on
the 40 ms and see nothing of the rhythm.

Two guards, both learned from the corpus rather than invented:

* A period is only reported if it survives ``period_min_repeats`` recurrences.  Two clicks 1 s
  apart look like a 1 Hz train otherwise, and the report would fill with coincidences.
* The reported confidence is peak-over-mean of the autocorrelation, not peak-over-noise-floor.
  A 6-blade fan on a 50 Hz supply and its own 3rd harmonic both score high; the number is there
  to rank candidates for a human, not to decide for them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..config import Config
from ..models import Detection

_EPS = 1e-12


def periodicity(onsets_s: np.ndarray, cfg: Config, duration_s: float
                ) -> list[tuple[float, float, float, int]]:
    """Return (period_s, confidence, jitter_s, n_repeats) for one band's onsets.

    Pairwise-difference histogram: with n onsets this is n^2 pairs, capped so a day of
    footsteps cannot cost gigabytes.  Bin width follows the shortest period we can resolve.
    """
    t = np.sort(np.asarray(onsets_s, dtype=np.float64))
    t = t[np.isfinite(t)]
    if t.size < cfg.period_min_repeats:
        return []
    if t.size > cfg.periodicity_max_onsets:
        t = t[: cfg.periodicity_max_onsets]
    lo, hi = cfg.comb_min_period_s, min(cfg.comb_max_period_s, 0.45 * (t[-1] - t[0]))
    if hi <= lo:
        return []
    step = max(cfg.comb_period_resolution_s, lo / 200.0)
    # Upper-triangle of the pairwise-difference matrix, row by row: n^2/2 floats but never the
    # full n x n matrix, which for 4000 onsets would be 128 MB.
    d = np.concatenate([t[i + 1:] - t[i] for i in range(t.size - 1)])
    edges = np.arange(lo, hi + step, step)
    hist, _ = np.histogram(d, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    # Boxcar the histogram before counting.  Each onset carries up to +-32 ms of frame-grid
    # quantisation, so a difference of two of them spreads a true 7.5 s period over ~0.1 s of
    # lag axis: without this the 9 supporting pairs land in 13 different bins and a clean
    # impulse train scores as noise (found by the 7.5 s train in tests/test_transients.py).
    w = max(1, int(round(cfg.period_smooth_s / step)))
    hist_s = uniform_filter1d(hist.astype(np.float64), size=w, mode="nearest") * w
    total = max(int(d.size), 1)
    mean = float(hist_s.mean()) + _EPS
    out = []
    for i in _local_maxima(hist_s):
        n_pairs = int(round(hist_s[i]))
        if n_pairs < cfg.period_min_pairs:
            continue
        # A fraction of the inter-onset intervals, not a peak/mean ratio: with 16 onsets the
        # background of the histogram is near zero, so peak/mean ran to 6740 and ranked noise
        # above evidence.  "8 of 120 intervals are ~1.02 s" is a claim a person can check.
        frac = n_pairs / total
        if frac < cfg.period_min_fraction:
            continue
        near = d[np.abs(d - centers[i]) <= max(step, 0.02 * centers[i])]
        jitter = float(np.std(near)) if near.size >= 3 else float("nan")
        n_rep = int(round((t[-1] - t[0]) / centers[i])) if centers[i] > 0 else 0
        if n_rep < cfg.period_min_repeats or near.size < cfg.period_min_repeats:
            continue
        out.append((float(centers[i]), frac, jitter, n_rep, float(hist_s[i] / mean)))
    out.sort(key=lambda r: (-r[1], r[0]))
    return [(p, f, j, n) for p, f, j, n, _pm in out[: cfg.period_max_report]]


def _local_maxima(h: np.ndarray) -> np.ndarray:
    return np.flatnonzero((h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:])) + 1


def detect_periodicity(onsets: dict[int, np.ndarray], cfg: Config, band_edges: np.ndarray,
                       duration_s: float, path: str = "") -> list[Detection]:
    """A3 detections from A4 onsets, one per band that shows a train (FR-4.5)."""
    dets: list[Detection] = []
    for band, t in sorted(onsets.items()):
        if band + 1 >= len(band_edges):
            continue
        for period, frac, jitter, n_rep in periodicity(t, cfg, duration_s):
            f_lo, f_hi = float(band_edges[band]), float(band_edges[band + 1])
            d = Detection(
                cls="A3", freq_hz=round(1.0 / period, 4),
                bandwidth_hz=round(abs(1.0 / max(period - (jitter if np.isfinite(jitter)
                                                           else period * 0.05), 1e-9)
                                        - 1.0 / period), 4),
                t_start_s=float(t.min()), t_end_s=float(t.max()),
                duration_s=float(t.max() - t.min()),
                tone_noise_db=round(10.0 * math.log10(max(frac, _EPS)), 2),
                coherence=round(frac, 3),
                persistence_pct=round(100.0 * n_rep * cfg.max_event_s / max(duration_s, 1e-9), 2),
                n_episodes=int(n_rep), confidence=float(min(0.99, frac * 2.0)),
                extra={"kind": "impulse-train", "period_s": round(period, 4),
                       "jitter_s": None if not np.isfinite(jitter) else round(jitter, 4),
                       "band": [f_lo, f_hi], "band_index": int(band),
                       "interval_fraction": round(frac, 3),
                       "n_onsets": int(t.size), "rpm": round(60.0 / period, 2)},
            )
            d.file = path
            dets.append(d)
    return dets
