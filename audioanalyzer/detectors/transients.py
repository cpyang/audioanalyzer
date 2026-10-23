"""A4: transient / impulse detection (REQUIREMENTS.md FR-4.4).

The tonal detectors look for energy that is *narrow in frequency and long in time*.  A4 is the
opposite corner: energy that is *short in time*, whether or not it is narrow in frequency.  On
this corpus that distinction matters more than it usually would — the recordings are almost
perfectly flat in level (whole-file RMS -37.9 dBFS, p99 of 10 s blocks -35.8 dBFS), so nothing
about a hammer blow shows up in a loudness trace.  It only shows up against the local
distribution of the neighbouring minutes.

Method, in the order the code runs it:

1. One streaming pass accumulates per-frame *band* energy (log-spaced bands) and per-frame
   clipping.  Memory is (frames x bands), a few MB regardless of file length.
2. Each series is compared to its own rolling median and rolling MAD over ``z_window_s`` on
   either side.  Median/MAD rather than mean/std because an impulse train contaminates the very
   statistics meant to find it; and local rather than global because a site that is quiet at
   03:00 and busy at 14:00 must not have 03:00 footsteps drowned by the 14:00 baseline, nor
   14:00 machinery excused by 03:00.
3. A frame is an event only if BOTH the robust z clears ``transient_sigma`` AND the absolute
   rise clears ``min_event_db``.  Z alone is meaningless in a dead-quiet stretch: 6 sigma of a
   0.1 dB spread is 0.6 dB, and a change in the weather would be an "event".
4. Frames containing a clipped sample are excluded outright (FR-4.4).  Flat-topped samples are
   broadband by construction — the loudest "transient" in these files is the recording, not the
   room.
5. The strongest events then get a targeted spectral look: what rang, not just when.

Onset times are also handed to A3 (periodicity), which is how a bearing or a fan blade turns
into a number instead of a hunch.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from ..config import Config
from ..models import Detection
from ..spectral import SpectralEngine

_EPS = 1e-30


def band_edges(eng: SpectralEngine, cfg: Config) -> np.ndarray:
    """Log-spaced band edges covering the analysis range (FR-4.4)."""
    hi_nyq = 0.95 * eng.fs / 2.0
    lo = cfg.min_freq
    hi = min(cfg.max_freq if cfg.max_freq else hi_nyq, hi_nyq)
    return np.geomspace(lo, hi, max(2, cfg.transient_bands) + 1)


def detect_transients(path: str | Path, cfg: Config, agg, eng: SpectralEngine,
                      media=None, detail: bool = True) -> tuple[list[Detection], dict]:
    """Return ``(A4 detections, onsets_per_band)``; onsets feed A3 (FR-4.5)."""
    if agg.n_frames == 0:
        return [], {}
    edges = band_edges(eng, cfg)
    nb = len(edges) - 1
    bin_band = np.digitize(eng.freqs, edges) - 1            # 0..nb-1 inside, -1/hi outside
    bin_band = np.clip(bin_band, 0, nb - 1)
    inside = (eng.freqs >= edges[0]) & (eng.freqs <= edges[-1])
    nframes = int(agg.n_frames)
    frame_t = (np.arange(nframes) * eng.hop + eng.nperseg / 2.0) / eng.fs
    dt = eng.hop / eng.fs

    band_db = np.full((nframes, nb), -400.0)
    clipped = np.zeros(nframes, dtype=bool)
    seg = max(1, int(math.ceil(eng.nperseg / eng.hop)))
    seen = 0
    for data in eng.iter_blocks(path):
        n, spec = eng.block_spectrum(data)
        if n == 0:
            continue
        db = eng.to_db(np.abs(spec))
        lin = np.where(inside, 10.0 ** (db / 10.0), 0.0)
        sums = np.zeros((n, nb))
        for b in range(nb):
            cols = np.flatnonzero((bin_band == b) & inside)
            if cols.size:
                sums[:, b] = lin[:, cols].sum(axis=1)
        band_db[seen:seen + n] = 10.0 * np.log10(sums + _EPS)
        if media is not None and getattr(media, "clipped_samples", 0):
            # Per hop-segment clip counts, then summed over the ``seg`` segments a frame spans:
            # exact for the frame window, and it never materialises a strided view of the file.
            x = np.asarray(data, dtype=np.float32)
            counts = _clip_per_segment(x, eng.hop, media.clip_level)
            # A frame starting at hop-segment k spans segments k..k+seg-1, so the sum must be
            # LEFT-aligned on the frame. scipy's filters window from a centred origin (and
            # refuses the shift that would align it), so this is an explicit cumsum difference:
            # exact, O(n), and it flags the frames that actually contain the clipped samples.
            cs = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
            hi = np.minimum(np.arange(n) + seg, counts.size)
            c = cs[hi] - cs[np.minimum(np.arange(n), counts.size)]
            clipped[seen:seen + n] = c[:n] > 0
        seen += n
    total_db = 10.0 * np.log10(np.power(10.0, band_db / 10.0).sum(axis=1) + _EPS)

    decim = max(1, int(round(0.25 * eng.frames_per_second())))
    win = max(3, 2 * int(round(cfg.z_window_s / (decim * dt))) + 1)
    med_b, mad_b = _rolling(band_db, decim, win)
    med_t, mad_t = _rolling(total_db[:, None], decim, win)
    med_t, mad_t = med_t[:, 0], mad_t[:, 0]           # both, or rise_t/mad_t broadcasts to n²

    rise_b, rise_t = band_db - med_b, total_db - med_t
    z_b = rise_b / (1.4826 * mad_b + cfg.mad_floor_db)
    z_t = rise_t / (1.4826 * mad_t + cfg.mad_floor_db)
    good = ~clipped
    hot_b = (z_b >= cfg.transient_sigma) & (rise_b >= cfg.min_event_db) & good[:, None]
    hot_t = (z_t >= cfg.transient_sigma) & (rise_t >= cfg.min_event_db) & good
    any_hot = hot_b.any(axis=1) | hot_t
    if not any_hot.any():
        return [], {b: np.zeros(0) for b in range(nb)}

    events = _frames_to_events(np.flatnonzero(any_hot), frame_t, dt,
                              cfg.event_merge_s, cfg.max_event_s, cfg.max_events)
    dets: list[Detection] = []
    onsets: dict[int, list[float]] = {}
    for t0, t1, idx in events:
        hits = np.flatnonzero(hot_b[idx].any(axis=0))
        if hits.size == 0:
            hits = np.array([int(np.argmax(z_b[idx].max(axis=0)))])
        primary = int(hits[int(np.argmax(z_b[idx].max(axis=0)[hits]))])
        z_peak = float(max(z_b[np.ix_(idx, hits)].max(), z_t[idx].max()))
        wide = hits.size >= max(2, math.ceil(0.6 * nb))
        d = Detection(
            cls="A4",
            freq_hz=float(math.sqrt(edges[primary] * edges[primary + 1])),
            t_start_s=float(t0), t_end_s=float(min(t1, frame_t[idx][-1] + dt)),
            duration_s=float(frame_t[idx][-1] - t0 + dt),
            level_dbfs=float(band_db[idx, primary].max()),
            tone_noise_db=float(np.median(z_b[idx, primary])),
            coherence=z_peak, persistence_pct=100.0 if wide else 0.0, n_episodes=1,
            duty_pct=0.0, confidence=float(min(0.99, z_peak / 20.0)),
            extra={"kind": "broadband-impact" if wide else "band-limited",
                   "band_index": int(primary),
                   "band": [round(float(edges[primary]), 1), round(float(edges[primary + 1]), 1)],
                   "z_peak": round(z_peak, 2), "rise_db": round(float(rise_b[idx, primary].max()), 1),
                   "bands_hot": [int(b) for b in hits], "frames": int(idx.size),
                   "edges_hz": [round(float(edges[b]), 1) for b in range(nb + 1)]},
        )
        d.file = str(path)
        dets.append(d)
        for b in hits:
            onsets.setdefault(int(b), []).append(float(t0))

    if detail and dets:
        _attach_signatures(path, cfg, eng, agg, dets, events)
    return dets, {k: np.asarray(v) for k, v in onsets.items()}


def _clip_per_segment(x: np.ndarray, hop: int, clip_level: float) -> np.ndarray:
    n = int(math.ceil(x.size / hop))
    pad = n * hop - x.size
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    return (np.abs(x.reshape(n, hop)) >= clip_level).sum(axis=1).astype(np.int32)


def _rolling(series: np.ndarray, decim: int, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling median and MAD, computed on a decimated copy and re-expanded (FR-4.4).

    A 60 s window is 1875 frames; running median_filter at frame resolution on 111k frames x
    8 bands costs minutes for no gain in accuracy, because the baseline is supposed to be slow.
    """
    n = series.shape[0]
    tail = (decim - n % decim) % decim
    s = series if not tail else np.concatenate([series, np.full((tail, series.shape[1]), np.nan)])
    dec = np.nanmedian(s.reshape(-1, decim, series.shape[1]), axis=1)
    # mode='reflect', not 'nearest': at the array edge 'nearest' duplicates the last decimated
    # sample across the whole window, so the MAD collapses and z inflates — 9 of 13 false
    # transients in calibration landed in the final 15 s of the file for exactly this reason.
    med = median_filter(dec, size=(win, 1), mode="reflect")
    mad = median_filter(np.abs(dec - med), size=(win, 1), mode="reflect")
    return (np.repeat(med, decim, axis=0)[:n], np.repeat(mad, decim, axis=0)[:n])


def _frames_to_events(idx: np.ndarray, frame_t: np.ndarray, dt: float, merge_s: float,
                      max_s: float, max_events: int) -> list[tuple[float, float, np.ndarray]]:
    """Group flagged frames, splitting at gaps > merge_s; cap by peak z (cheapest first pass)."""
    out: list[tuple[float, float, np.ndarray]] = []
    start = 0
    for i in range(1, idx.size + 1):
        gap = i == idx.size or (frame_t[idx[i]] - frame_t[idx[i - 1]]) > merge_s
        if gap or (i < idx.size and frame_t[idx[i]] - frame_t[idx[start]] > max_s):
            out.append((float(frame_t[idx[start]]), float(frame_t[idx[i - 1]] + dt),
                        idx[start:i]))
            start = i
    if max_events and len(out) > max_events:
        out.sort(key=lambda e: -(e[1] - e[0]))
        out = out[:max_events]
        out.sort(key=lambda e: e[0])
    return out


def _attach_signatures(path, cfg: Config, eng: SpectralEngine, agg,
                       dets: list[Detection], events) -> None:
    """Targeted look at the strongest events: what rang, not just when (FR-4.4).

    The event's mean spectrum is compared to that bin's own p50 over the recording, so a
    signature line means *this frequency rose like this while the machinery was hitting*.  Only
    the top ``event_detail_top`` by peak z are detailed.

    Each one is decoded separately by seeking to its frame range.  Sweeping the file once instead
    sounds cheaper until events spread out: the union of 40 event windows across an hour is the
    whole hour, which turned a 2-pass scan into a 3-pass one and put the day over the NFR-4
    budget.  40 seeks of a few hundred milliseconds cost under a second in total.
    """
    order = sorted(range(len(dets)), key=lambda j: -dets[j].coherence)[: cfg.event_detail_top]
    chosen = [(j, events[j]) for j in order if j < len(events)]
    if not chosen:
        return
    base = agg.percentile(0.50)
    import soundfile as _sf
    with _sf.SoundFile(str(path)) as fh:
        n_total = fh.frames
        for j, (t0, t1, idx) in chosen:
            first = int(idx[0]) * eng.hop
            last = min(int(idx[-1]) * eng.hop + eng.nperseg, n_total)
            if last - first < eng.nperseg:
                continue
            fh.seek(first)
            spec = eng.block_spectrum(fh.read(last - first, dtype="float32"))[1]
            if spec.shape[0] == 0:
                continue
            d = dets[j]
            db = eng.to_db(np.abs(spec).mean(axis=0))
            rise = db - base
            order_bins = np.argsort(np.nan_to_num(rise, neginf=-999.0))[::-1][: 16]
            # A riser at or below cfg.min_freq is the analysis filter's skirt, not a ring: a 9 Hz
            # thump put its whole signature at 15.6/23.4/31.2 Hz, and publishing that as an event's
            # frequency claimed a sub-audible tone where FR-1.4 declares nothing resolvable.  Same
            # floor band_edges() uses, so signature and band agree about what is nameable.
            lo_hz, hi_nyq = cfg.min_freq, 0.95 * eng.fs / 2.0
            tops = [(float(eng.freqs[k]), float(rise[k]), float(db[k])) for k in order_bins
                    if np.isfinite(rise[k]) and rise[k] >= cfg.min_event_db
                    and lo_hz <= eng.freqs[k] <= hi_nyq][:3]
            d.extra["signature"] = [{"freq_hz": round(f, 2), "rise_db": round(r, 1),
                                     "level_dbfs": round(l, 1)} for f, r, l in tops]
            if tops:
                d.freq_hz = round(tops[0][0], 2)
                d.level_dbfs = round(tops[0][2], 1)
