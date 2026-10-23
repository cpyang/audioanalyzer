"""A1/A2 tonal detector, stage 2 (REQUIREMENTS.md FR-4.1/4.2/4.3/4.7).

Stage 1 (:func:`audioanalyzer.spectral.candidate_tones`) proposes candidate bins from the
time-averaged spectrum.  This stage streams the file a second time and measures, for those
bins only, *when* each tone is present: per-frame presence, episodes, persistence, level
stability and frequency drift.  Cost scales with the candidate count, not with bins x frames.

Candidate bins within ``merge_bin_gap`` of each other are grouped into one *feature* first.
Without that, a single line that straddles a bin boundary, drifts by a bin, or is just the
knee of the rumble gets reported several times -- which is how six phantom "tones" at
63/70/85/94/102/109 Hz (one 7.8 Hz apart, i.e. adjacent bins) once came from one smooth hump.

Two prominence measures are reported and answer different questions:

``tone_noise_db``  spectral  level - median of the +-``neighbor_k`` bins, per frame
    -> sees a tone present for the *whole* recording.  This is the A1 gate.
``prominence_db``  temporal  level - that bin's own p50 over its baseline block
    -> sees a tone that *starts*, or comes and goes.  Reported; used by ``compare`` (FR-6).

Gating A1 on the temporal measure alone would make an ever-present tone its own baseline and
report nothing -- the trap this design exists to avoid.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..config import Config
from ..models import Detection
from ..spectral import MIN_FLOOR_K, SpectralEngine


def detect_tonal(path: str | Path, cfg: Config, agg, eng: SpectralEngine,
                 cands: np.ndarray) -> list[Detection]:
    """Return A1/A2 detections from ``cands`` (a ``candidate_tones`` record array)."""
    if int(len(cands)) == 0 or agg.n_frames == 0:
        return []
    groups = _group_adjacent(cands, cfg)
    ng, nframes = len(groups), int(agg.n_frames)
    k = cfg.neighbor_k
    frame_t = (np.arange(nframes) * eng.hop + eng.nperseg / 2.0) / eng.fs
    max_bin = eng.nbins - 2

    present = np.zeros((ng, nframes), dtype=bool)
    freq_of = np.full((ng, nframes), np.nan, dtype=np.float32)
    lvl_of = np.full((ng, nframes), np.nan, dtype=np.float32)
    ttn_of = np.full((ng, nframes), np.nan, dtype=np.float32)
    coh_of = np.full((ng, nframes), np.nan, dtype=np.float32)
    iso_of = np.full((ng, nframes), np.nan, dtype=np.float32)
    bin_of = np.zeros((ng, nframes), dtype=np.int16)
    min_bin = max(1, int(math.floor(cfg.min_freq / eng.df)))
    win_c = max(3, int(round(cfg.coherence_smooth_s * eng.fs / eng.hop)))

    seen = 0
    for data in eng.iter_blocks(path):
        n, spec = eng.block_spectrum(data)          # one FFT per block, reused for dB + phase
        if n == 0:
            continue
        db = eng.to_db(np.abs(spec))
        dbr = np.array(db, dtype=np.float64, copy=True)
        dbr[:, :min_bin] = np.nan                       # DC/rumble excluded from every floor
        rows = np.arange(n)
        t_frames = (seen + rows) * eng.hop / eng.fs
        for gi, g in enumerate(groups):
            cols_g = np.asarray(g["bins"], dtype=np.int64)
            sub = db[:, cols_g]
            pick = np.argmax(sub, axis=1) if cols_g.size > 1 else np.zeros(n, dtype=np.int64)
            gb = cols_g[pick]                           # global bin carrying the peak, per frame
            own = sub[rows, pick] if cols_g.size > 1 else sub[:, 0]
            centre = int((int(cols_g[0]) + int(cols_g[-1])) // 2)
            kk = max(MIN_FLOOR_K, min(k, centre - min_bin, eng.nbins - 1 - centre))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                floor = np.nanmedian(dbr[:, centre - kk:centre + kk + 1], axis=-1)
            floor = np.nan_to_num(floor, nan=-400.0)
            ttn = own - floor
            # Phase coherence: de-rotate by the tone frequency, then average the complex
            # coefficient. A locked tone keeps |mean| ~ mean|X|; band energy drifts in phase and
            # collapses toward 1/sqrt(N_eff).  This is what separates a 65 Hz motor whine from
            # the +-5 dB wobble of the rumble plateau it sits in.
            #
            # A small offset bank is essential, not paranoia: stage 1 knows f only to the
            # interpolated peak (~0.1-0.5 Hz), and over a 1 s window a 0.5 Hz error already
            # rotates the phase by pi, which would destroy coherence for a perfectly real tone.
            # Sweeping +-2 offsets of 1/L recovers it, and the max is what we threshold.
            mag = np.abs(spec[:, centre])
            # Instantaneous narrowbandness: a tone leaves its neighbours at the floor, while a
            # gust, a door, or the rumble knee lifting raises +-30 Hz together.  Stage 1's width
            # test only sees this in the time-average, so a brief broadband LF lift can still
            # look like a candidate; checking it per frame is what separates them (FR-4.1).
            nb = [b for b in range(centre - cfg.isolation_span_bins, centre + cfg.isolation_span_bins + 1)
                  if min_bin <= b < spec.shape[1] and abs(b - centre) >= cfg.isolation_gap_bins]
            nb_mag = np.median(np.abs(spec[:, nb]), axis=1) if nb else np.zeros_like(mag)
            iso_db = 20.0 * np.log10(np.maximum(mag, 1e-30) / np.maximum(nb_mag, 1e-30))
            denom = np.maximum(uniform_filter1d(mag, size=win_c, mode="nearest"), 1e-30)
            coh = np.zeros(n, dtype=np.float64)
            for off in (_coherence_offsets(cfg)):
                y = spec[:, centre] * np.exp(-2j * math.pi * (g["freq"] + off) * t_frames)
                cy = (uniform_filter1d(y.real, size=win_c, mode="nearest")
                      + 1j * uniform_filter1d(y.imag, size=win_c, mode="nearest"))
                np.maximum(coh, np.abs(cy) / denom, out=coh)
            y1 = np.nan_to_num(own, neginf=-400.0)
            y0 = np.nan_to_num(db[rows, np.clip(gb - 1, 1, max_bin)], neginf=-400.0)
            y2 = np.nan_to_num(db[rows, np.clip(gb + 1, 1, max_bin)], neginf=-400.0)
            den = y0 - 2.0 * y1 + y2
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(den != 0.0, 0.5 * (y0 - y2) / den, 0.0)
            d = np.where(np.isfinite(d) & (np.abs(d) <= 1.0), d, 0.0)
            present[gi, seen:seen + n] = False           # decided after temporal smoothing
            freq_of[gi, seen:seen + n] = eng.freqs[gb] + d * eng.df
            lvl_of[gi, seen:seen + n] = y1
            ttn_of[gi, seen:seen + n] = ttn
            coh_of[gi, seen:seen + n] = np.clip(coh, 0.0, 1.0)
            iso_of[gi, seen:seen + n] = iso_db
            bin_of[gi, seen:seen + n] = gb.astype(np.int16)
        seen += n
    sl = slice(0, seen)
    present, freq_of, lvl_of, ttn_of, coh_of, bin_of = (a[:, sl] for a in
                                                       (present, freq_of, lvl_of, ttn_of,
                                                        coh_of, bin_of))
    iso_of = iso_of[:, sl]

    # Presence is decided on a *time-smoothed* tone-to-noise, never the raw per-frame value.
    # A single FFT bin is exponentially distributed, so it exceeds its local floor by 6 dB in
    # ~25% of frames with no tone present at all (measured: 17.9 dB peaks on a tone-free
    # realistic floor).  Smoothing power over presence_smooth_s frames suppresses that; the
    # cost is smeared on/off edges, which is the right trade for tones (>= min_tone_s, FR-4.1)
    # and is precisely why short events are A4's job instead.
    win = max(1, int(round(cfg.presence_smooth_s * eng.fs / eng.hop)))
    for gi in range(len(groups)):
        ttn_of[gi] = _smooth_ttn(ttn_of[gi], win)
        iso_of[gi] = uniform_filter1d(np.nan_to_num(iso_of[gi], neginf=-400.0), size=win,
                                      mode="nearest")
        # Three criteria must hold together: enough level above the local floor, a phase-locked
        # tone, and an instantaneously isolated one.  A rumble wobble can satisfy the first;
        # only a real oscillator satisfies all three.
        present[gi] = ((ttn_of[gi] >= cfg.tone_noise_db)
                       & (coh_of[gi] >= cfg.min_coherence)
                       & (iso_of[gi] >= cfg.min_isolation_db))

    frame_dt = eng.hop / eng.fs
    p50 = np.stack([h.percentile(0.50) for _, h in agg.baseline.blocks]) if agg.baseline else None
    out: list[Detection] = []
    for gi, g in enumerate(groups):
        idx = np.flatnonzero(present[gi])
        if cfg.debug_groups:
            import sys
            t_all = frame_t
            print(f"dbg g{gi} bins={g['bins']} f={g['freq']:.2f} ttn1={g['ttn']:.1f} "
                  f"active={idx.size}/{present[gi].size}"
                  f" span_q1..q3={np.percentile(t_all[idx], [25, 75]).round(0) if idx.size else '-'}"
                  f" coh_med={np.nanmedian(coh_of[gi][idx]) if idx.size else float('nan'):.2f}"
                  f" iso_med={np.nanmedian(iso_of[gi][idx]) if idx.size else float('nan'):.1f}"
                  f" ttn_med={np.median(ttn_of[gi][idx]) if idx.size else float('nan'):.1f}",
                  file=sys.stderr)
        if idx.size == 0:
            continue
        t0 = float(frame_t[idx[0]])
        t1 = float(frame_t[idx[-1]] + frame_dt)
        span = t1 - t0
        if span < cfg.min_tone_s:
            continue
        episodes = [e for e in _episodes(idx, frame_dt, cfg.gap_tolerance_s)
                    if e[1] - e[0] >= cfg.min_tone_s]
        # Only qualifying occurrences are reported.  Counting single-frame stragglers as
        # "episodes" made an intermittent tone look like it fired 51 times when it had fired
        # once for 10 minutes; brief pips are A4's business, not A1/A2's.
        if not episodes:
            continue
        active_s = sum(b - a for a, b in episodes)
        t_first, t_last = episodes[0][0], episodes[-1][1]
        persistence = 100.0 * active_s / max(t_last - t_first, 1e-9)
        longest = max(b - a for a, b in episodes)
        duty = 100.0 * active_s / max(agg.duration_s, 1e-9)
        t = frame_t[idx]
        keep = (t >= t_first) & (t <= t_last)             # drop strays outside real episodes
        idx, t = idx[keep], t[keep]
        if t.size < 3:
            continue
        f = np.where(np.isfinite(freq_of[gi, idx]), freq_of[gi, idx], g["freq"]).astype(np.float64)
        lvl = np.where(np.isfinite(lvl_of[gi, idx]), lvl_of[gi, idx], -400.0).astype(np.float64)
        ttn = np.where(np.isfinite(ttn_of[gi, idx]), ttn_of[gi, idx], 0.0).astype(np.float64)
        slope, r2 = _drift(t, f)
        coh_active = np.nan_to_num(coh_of[gi, idx], nan=0.0).astype(np.float64)
        prom = _temporal_prominence(t, lvl, bin_of[gi, idx], p50, cfg.baseline_block_s)
        lvl_std = float(np.std(lvl))
        out.append(Detection(
            cls=("A1" if len(episodes) <= 1 and duty >= 90.0
                 and lvl_std <= cfg.steady_level_std_db else "A2"),
            freq_hz=float(np.median(f)), freq_std_hz=float(np.std(f)),
            bandwidth_hz=float(max(eng.df, np.ptp(f))), n_bins=len(g["bins"]),
            t_start_s=float(t_first), t_end_s=float(t_last),
            duration_s=float(t_last - t_first),
            level_dbfs=float(np.median(lvl)), peak_dbfs=float(np.max(lvl)),
            level_std_db=lvl_std, tone_noise_db=float(np.median(ttn)), coherence=float(np.median(coh_active)),
            isolation_db=float(np.median(np.nan_to_num(iso_of[gi, idx], neginf=-400.0))),
            prominence_db=float(np.median(prom)) if prom.size else 0.0,
            persistence_pct=float(persistence), n_episodes=len(episodes),
            longest_episode_s=float(longest),
            duty_pct=float(100.0 * active_s / max(agg.duration_s, 1e-9)),
            drift_hz_per_min=float(slope * 60.0), drift_r2=float(r2),
            drift_class=_drift_class(slope * 60.0, r2, cfg),
            confidence=_confidence(float(np.median(ttn)), float(np.median(coh_active)),
                                   duty, lvl_std, cfg),
            extra={"episodes": [[round(a, 3), round(b, 3)] for a, b in episodes[:64]],
                   "n_frames": int(idx.size), "mean_spectrum_dbfs": float(g["mean_db"]),
                   "candidate_bins": [int(b) for b in g["bins"]],
                   "first_block": int(g["first_block"])},
        ))
    out.sort(key=lambda d: (-round(d.tone_noise_db, 3), d.freq_hz))
    return out


# ---------------------------------------------------------------- helpers

def _coherence_offsets(cfg: Config) -> tuple[float, ...]:
    """Frequency offsets (Hz) swept when measuring coherence.

    Spacing is 1/L, the frequency resolution of the smoothing window, so the bank spans the
    uncertainty in the stage-1 frequency estimate without blurring the tone.
    """
    L = max(cfg.coherence_smooth_s, 1e-3)
    return tuple(k / L for k in (-cfg.coherence_offset_bank, 0, cfg.coherence_offset_bank))


def _smooth_ttn(ttn: np.ndarray, win: int) -> np.ndarray:
    """Moving power-average of tone-to-noise (dB) over *win* frames.

    Averaging in the power domain, not the dB domain: the mean of log-magnitudes is a
    geometric mean and under-weights the peaks that carry the tone.
    """
    if win <= 1 or ttn.size == 0:
        return ttn.astype(np.float64)
    p = np.power(10.0, np.nan_to_num(ttn, nan=-400.0, neginf=-400.0) / 10.0)
    s = uniform_filter1d(p, size=win, mode="nearest")
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(s, out=np.full_like(s, -400.0), where=s > 0)


def _group_adjacent(cands: np.ndarray, cfg: Config) -> list[dict]:
    """Merge candidate bins within ``merge_bin_gap`` into one feature group (FR-4.7)."""
    order = np.argsort(cands["bin"], kind="stable")
    groups: list[dict] = []
    for pos in order:
        b = int(cands["bin"][pos])
        rec = {"ttn": float(cands["ttn"][pos]), "freq": float(cands["freq"][pos]),
               "mean_db": float(cands["mean_db"][pos]),
               "first_block": int(cands["first_block"][pos])}
        if groups and b - groups[-1]["bins"][-1] <= cfg.merge_bin_gap:
            prev = groups[-1]
            prev["bins"].append(b)
            if rec["ttn"] > prev["ttn"]:                # group identity = strongest member
                for key in ("ttn", "freq", "mean_db", "first_block"):
                    prev[key] = rec[key]
        else:
            groups.append({"bins": [b], **rec})
    return groups


def _episodes(idx: np.ndarray, frame_dt: float, gap_s: float) -> list[tuple[float, float]]:
    """Split active frame indices into episodes, allowing gaps up to ``gap_s`` (FR-4.2)."""
    if idx.size == 0:
        return []
    jumps = np.flatnonzero(np.diff(idx) * frame_dt > gap_s)
    starts = np.concatenate(([0], jumps + 1))
    ends = np.concatenate((jumps, [idx.size - 1]))
    return [(float(idx[i]) * frame_dt - 0.5 * frame_dt,
             float(idx[j]) * frame_dt + 1.5 * frame_dt) for i, j in zip(starts, ends)]


def _drift(t: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Least-squares slope (Hz/s) and r^2 (FR-4.3)."""
    if t.size < 3 or np.ptp(t) <= 0:
        return 0.0, 0.0
    tc, fc = t - t.mean(), f - f.mean()
    sxx = float(tc @ tc)
    if sxx <= 0:
        return 0.0, 0.0
    slope = float(tc @ fc) / sxx
    resid = fc - slope * tc
    tot = float(fc @ fc)
    r2 = 1.0 - float(resid @ resid) / tot if tot > 0 else 0.0
    return slope, max(0.0, min(1.0, r2))


def _drift_class(hz_per_min: float, r2: float, cfg: Config) -> str:
    if r2 < 0.5 or abs(hz_per_min) < cfg.drift_stable_hz_per_min:
        return "stable"
    return "drifting" if abs(hz_per_min) < 50.0 else "sweeping"


def _temporal_prominence(t: np.ndarray, levels: np.ndarray, bins: np.ndarray,
                         p50: np.ndarray | None, block_s: float) -> np.ndarray:
    """level(t) minus the p50 of the bin that actually carried the peak at t (FR-3.5)."""
    if p50 is None or p50.size == 0:
        return np.zeros(levels.shape, dtype=np.float64)
    bi = np.clip((t // block_s).astype(np.int64), 0, p50.shape[0] - 1)
    return levels - p50[bi, bins.astype(np.int64)]


def _confidence(ttn: float, coherence: float, duty: float, level_std: float,
                cfg: Config) -> float:
    """Severity for A1/A2, so `--fail-on` can gate a scheduled check (FR-8.4).

    A *product of gate margins* rather than a blend of weights: each factor is 1.0 once the
    evidence is comfortably past the threshold that admitted it and falls toward 0 as it
    approaches that threshold, so any score can be explained factor by factor ("0.55: coherence
    0.82, tone-to-noise only 9 dB").  The previous form — 0.5/0.3/0.2 weights over unrelated
    quantities — produced 0.86 for a marginal line and ignored coherence entirely, which made
    `--fail-on` both arbitrary and nearly inert.
    """
    margin = min(1.0, ttn / (2.0 * cfg.tone_noise_db))
    presence = min(1.0, 0.5 + duty / 100.0)
    stability = max(0.4, 1.0 - level_std / (4.0 * cfg.steady_level_std_db))
    return round(float(np.clip(coherence * margin * presence * stability, 0.0, 1.0)), 3)
