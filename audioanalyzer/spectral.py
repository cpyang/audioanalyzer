"""STFT engine, level calibration, and time-averaged spectral aggregation.

Why the design is two-stage rather than per-frame peak-picking
--------------------------------------------------------------
White noise makes ~1/3 of all bins a "local maximum", so on real audio a per-frame peak
picker emits ~280 candidates *per frame* -- 15.6 M records for one hour, 2 GB of RAM and
112 s of wall time.  Averaging in power first suppresses the noise floor by sqrt(N_frames)
while a tone present even intermittently survives, so tones become single obvious spikes in
the mean spectrum.  Stage 1 therefore aggregates; stage 2 (``detectors.tonal``) measures
persistence for the handful of candidate bins only.

Level convention (FR-3.3): **amplitude-referenced dBFS** -- a full-scale sine at a bin centre
reads 0.0 dBFS.  Its RMS-referenced value would be -3.01 dBFS.  Whole-file ``MediaInfo``
levels are RMS-referenced; every output labels which is which.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import get_window

from .baseline import BaselineModel
from .config import Config

DC_TERM = 1.0
AC_TERM = 2.0


@dataclass
class Aggregate:
    """Stage-1 output: everything needed to propose tone candidates."""

    sample_rate: int
    nperseg: int
    hop: int
    freqs: np.ndarray
    df: float
    nbins: int
    n_frames: int = 0
    duration_s: float = 0.0
    mean_db: np.ndarray = field(default_factory=lambda: np.empty(0))
    block_db: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    block_t0: list[float] = field(default_factory=list)
    baseline: BaselineModel | None = None

    def percentile(self, q: float, t: float = 0.0) -> np.ndarray:
        assert self.baseline is not None
        return self.baseline.p50(t) if q == 0.5 else self.baseline.block_for(t).percentile(q)


class SpectralEngine:
    """Streaming short-time Fourier analysis on one file."""

    def __init__(self, cfg: Config, sample_rate: int):
        self.cfg = cfg
        self.fs = int(sample_rate)
        self.nperseg = cfg.hr_nperseg if cfg.mode == "hr" else cfg.nperseg
        # Guard against absurd windows for the rate, not against windows longer than a second:
        # nperseg 16384 at 16 kHz is 1.024 s of audio and 0.98 Hz/bin, which is exactly what
        # --mode hr is for (FR-3.2).  Clamping at nperseg > fs quietly made hr 3.9 Hz/bin.
        if self.nperseg > 16 * self.fs:
            self.nperseg = int(2 ** math.floor(math.log2(max(8, 16 * self.fs))))
        if cfg.mode == "hr":
            self.hop = min(cfg.hr_hop, self.nperseg - 1)
            self.noverlap = self.nperseg - self.hop
        else:
            self.noverlap = min(cfg.noverlap, self.nperseg - 1)
            self.hop = self.nperseg - self.noverlap
        self.window = get_window("hann", self.nperseg, fftbins=False)
        self.win_sum = float(self.window.sum())
        self.freqs = np.fft.rfftfreq(self.nperseg, 1.0 / self.fs)
        self.df = float(self.freqs[1] - self.freqs[0])
        self.nbins = int(self.freqs.size)
        self.block_samples = max(self.nperseg,
                                 int(round(cfg.block_s * self.fs / self.hop)) * self.hop
                                 + (self.nperseg - self.hop))
        # Frames advance by exactly ``hop``; the extra (nperseg - hop) samples are the window's
        # overlap, read again from the previous chunk rather than skipped (see iter_blocks).
        self.frames_per_block = max(1, self.block_samples // self.hop
                                    - (self.nperseg - self.hop) // self.hop)
        self.chunk_advance = self.frames_per_block * self.hop

    # ---- level scale -------------------------------------------------------
    def to_db(self, mag: np.ndarray) -> np.ndarray:
        """Magnitude spectrum -> amplitude-referenced dBFS.

        ``a_k = 2|X_k| / sum(w)`` (DC/Nyquist factor 1).  A unit sine puts ``sum(w)/2`` in
        its bin, so this returns exactly 1.0 -> 0 dBFS.
        """
        out = (AC_TERM * np.asarray(mag, dtype=np.float64)) / self.win_sum
        out[..., 0] *= DC_TERM / AC_TERM
        if self.nperseg % 2 == 0:
            out[..., -1] *= DC_TERM / AC_TERM
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(out, out=np.full(out.shape, -np.inf, dtype=np.float64),
                                   where=out > 0)

    # ---- block iteration ---------------------------------------------------
    def frames_in_block(self, n_samples: int) -> int:
        return 0 if n_samples < self.nperseg else 1 + (n_samples - self.nperseg) // self.hop

    def block_db(self, data: np.ndarray) -> tuple[int, np.ndarray]:
        """(n_frames, nbins) dB matrix for one chunk of samples.

        The per-frame weighted mean is removed first: a persistent DC offset (this corpus has
        -38.25 dBFS of it) otherwise leaks -70 dB down into bins 1-3 through the window, and in
        a quiet recording that leakage is a 15-23 Hz "tone" that does not exist.  The offset is
        still measured and reported in the time domain by ``MediaInfo``/A6 (FR-4.6).
        """
        x = np.asarray(data, dtype=np.float32)
        n = self.frames_in_block(x.size)
        if n == 0:
            return 0, np.empty((0, self.nbins), dtype=np.float64)
        frames = sliding_frames(x[:n * self.hop + self.nperseg - self.hop],
                                self.nperseg, self.hop)[:n]
        w = self.window.astype(np.float32)
        dc = (frames @ w) / self.win_sum                     # float32: a float64 copy here costs
        frames = (frames - dc[:, None]) * w                  # ~60 MB per block and 40% of runtime
        return n, self.to_db(np.abs(np.fft.rfft(frames, axis=1)))

    def block_spectrum(self, data: np.ndarray) -> tuple[int, np.ndarray]:
        """(n_frames, nbins) complex STFT with the same DC notch as :meth:`block_db`."""
        x = np.asarray(data, dtype=np.float32)
        n = self.frames_in_block(x.size)
        if n == 0:
            return 0, np.empty((0, self.nbins), dtype=np.complex64)
        frames = sliding_frames(x[:n * self.hop + self.nperseg - self.hop],
                                self.nperseg, self.hop)[:n]
        w = self.window.astype(np.float32)
        dc = (frames @ w) / self.win_sum
        frames = (frames - dc[:, None]) * w
        return n, np.fft.rfft(frames, axis=1)


    def iter_blocks(self, path: str | Path) -> Iterator[np.ndarray]:
        """Yield successive analysis chunks; channel-mean for multi-channel (FR-1.1).

        Each chunk is ``frames_per_block * hop + (nperseg - hop)`` samples: the hop-spaced
        frames plus the window overlap re-read from the previous chunk.  Reading *sequential*
        blocks of that size instead — which is what ``SoundFile.blocks`` does — silently loses
        ``(nperseg - hop)`` samples per block, so frame *i* no longer starts at sample
        ``i * hop``: on this corpus that was 96 ms per 30 s block, 0.32% of the timeline, and
        11.4 s of accumulated error by the end of an hour.  A tone injected at 300-900 s came
        back dated 299.0 s for 598.3 s, which is how it was caught (FR-1.2, FR-2.1).
        """
        with sf.SoundFile(str(path)) as fh:
            pos = 0
            while True:
                fh.seek(pos)
                raw = fh.read(self.block_samples, dtype="float32", always_2d=True)
                if raw.shape[0] == 0:
                    return
                x = raw[:, 0] if raw.shape[1] == 1 else raw.T.mean(axis=0).astype(np.float32)
                yield x
                if x.size < self.block_samples:            # short read: end of file
                    return
                pos += self.chunk_advance

    def frames_per_second(self) -> float:
        return self.fs / self.hop


def power_from_db(db: np.ndarray) -> np.ndarray:
    """One-sided power from amplitude-dBFS: ``(10**(db/20))**2``.

    Used so stage 1 can average in the power domain (a true PSD average, which is what makes
    a time-averaged tone spike survive) without running the FFT twice.
    """
    with np.errstate(under="ignore", over="ignore"):
        p = np.power(10.0, np.nan_to_num(db, neginf=-400.0) / 10.0)
    return np.where(np.isfinite(p), p, 0.0)


def sliding_frames(x: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    """(n_frames, nperseg) view of *x*, no copy."""
    if x.size < nperseg:
        x = np.pad(x, (0, nperseg - x.size))
    return np.lib.stride_tricks.sliding_window_view(x, nperseg)[::hop]


def aggregate(path: str | Path, cfg: Config, progress: bool = False) -> tuple[Aggregate, SpectralEngine]:
    """Stage 1: stream the file once, accumulating mean/block spectra + per-bin histograms."""
    with sf.SoundFile(str(path)) as fh:
        sr, n_frames_total, chans = int(fh.samplerate), int(fh.frames), int(fh.channels)
    eng = SpectralEngine(cfg, sr)
    duration = n_frames_total / sr
    nblocks = max(1, int(math.ceil(duration / cfg.baseline_block_s)))
    agg = Aggregate(sample_rate=sr, nperseg=eng.nperseg, hop=eng.hop, freqs=eng.freqs,
                    df=eng.df, nbins=eng.nbins, duration_s=duration,
                    block_t0=[i * cfg.baseline_block_s for i in range(nblocks)],
                    baseline=BaselineModel(nbins=eng.nbins, block_s=cfg.baseline_block_s,
                                           sample_rate=sr,
                                           hist_cfg=(cfg.hist_db_lo, cfg.hist_db_hi,
                                                     cfg.hist_db_bw * (2.0 if cfg.mode == "hr" else 1.0))))
    assert agg.baseline is not None
    sum_p = np.zeros(eng.nbins, dtype=np.float64)
    block_p = np.zeros((nblocks, eng.nbins), dtype=np.float64)
    block_n = np.zeros(nblocks, dtype=np.int64)
    seen = 0
    for _data in eng.iter_blocks(path):
        n, db = eng.block_db(_data)
        if n == 0:
            continue
        # power is exactly recoverable from the dB matrix: 10**(db/10).  Recomputing the
        # FFT here would double stage-1 cost for no gain.
        p = power_from_db(db)
        t_frames = (np.arange(seen, seen + n) * eng.hop + eng.nperseg / 2.0) / sr
        sum_p += p.sum(axis=0)
        bi = np.clip((t_frames // cfg.baseline_block_s).astype(np.int64), 0, nblocks - 1)
        for b in range(nblocks):
            rows = np.flatnonzero(bi == b)
            if rows.size:
                block_p[b] += p[rows].sum(axis=0)
                block_n[b] += rows.size
        agg.baseline.update(t_frames, db)
        seen += n
        if progress:
            print(f"  stage1 {100.0 * min(1.0, seen * eng.hop / max(1, n_frames_total)):3.0f}%",
                  end="\r", flush=True)
    if progress:
        print()
    agg.n_frames = seen
    with np.errstate(divide="ignore", invalid="ignore"):
        agg.mean_db = 10.0 * np.log10(np.where(seen, sum_p / max(seen, 1), 0.0))
        blk = np.where(block_p > 0, block_p, np.nan)
        agg.block_db = 10.0 * np.log10(blk / np.maximum(block_n[:, None], 1))
        agg.block_db = np.nan_to_num(agg.block_db, nan=-400.0, neginf=-400.0)
    return agg, eng


MIN_FLOOR_K = 3


def floor_curve(spec: np.ndarray, k_max: int, first_valid: int = 1) -> np.ndarray:
    """Per-bin local floor from a *symmetric* window, narrowed near the masked edge.

    A fixed +-k window is one-sided for bins within k of ``first_valid``: its median then
    lands k bins *above* the centre, so on a spectrum with a knee the low band reads 10-25 dB
    "above the floor" with no tone present at all.  Narrowing the window to ``min(k_max,
    b - first_valid)`` keeps it centred and therefore unbiased for a monotone spectrum, at the
    cost of a noisier estimate in the lowest few bins.
    """
    s = np.nan_to_num(np.asarray(spec, dtype=np.float64), neginf=-400.0, nan=-400.0)
    n = s.size
    out = np.empty(n, dtype=np.float64)
    for b in range(n):
        kk = max(MIN_FLOOR_K, min(k_max, b - first_valid, n - 1 - b))
        out[b] = float(np.median(s[max(b - kk, 0):b + kk + 1]))
    return out


def adaptive_floor_at(spec: np.ndarray, bin_: int, k_max: int, first_valid: int = 1) -> float:
    """Scalar :func:`floor_curve` for one bin (stage 2 hot path)."""
    n = spec.shape[-1]
    kk = max(MIN_FLOOR_K, min(k_max, int(bin_) - first_valid, n - 1 - int(bin_)))
    return float(np.median(spec[..., max(int(bin_) - kk, 0):int(bin_) + kk + 1], axis=-1))


def mask_below(spec_db: np.ndarray, min_bin: int) -> np.ndarray:
    """Set bins below ``min_bin`` (the DC/rumble region) to NaN so they cannot enter a
    neighbourhood median.

    This matters more than it looks: a persistent DC offset of -38 dBFS sitting in bin 0 will
    be *replicated* by edge padding into every window that reaches bin 0, corrupting the floor
    estimate for the first few hundred Hz.  ``min_bin`` derives from ``Config.min_freq``.
    """
    s = np.array(spec_db, dtype=np.float64, copy=True)
    s[..., :max(1, int(min_bin))] = np.nan
    return s


def neighborhood_floor(spec_db: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Median of the +-k bin neighbourhood: the local broadband floor (PSDL-style).

    NaNs (excluded bins, digital silence) are ignored rather than padded with edge values, so
    the DC spike cannot leak into low-frequency windows.  A genuine narrowband tone is 1 of
    ``2k+1`` values and cannot move the median, while a broadband rise does -- exactly the
    behaviour wanted, because broadband energy is not an A1 tone (FR-4.1, risk R1).  For a
    smooth monotone spectrum the centred median equals the centre value, so the measure is
    unbiased.
    """
    s = np.asarray(spec_db, dtype=np.float64)
    if k <= 0:
        return s.copy()
    axis = s.ndim - 1 if axis == -1 else axis
    pad_width = [(0, 0)] * s.ndim
    pad_width[axis] = (k, k)
    pad = np.pad(s, pad_width, mode="constant", constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1, axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices near the edges
        out = np.nanmedian(view, axis=-1)
    return np.nan_to_num(out, nan=-400.0)


def interp_peak(spec_db: np.ndarray, idx: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-parabolic interpolation on a 1-D dB spectrum -> (delta_bins, peak_level_db).

    Accepts a scalar or an array index.  The input is already a log domain, so a ``log()``
    here would be a double-log bug.
    """
    i = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    n = spec_db.shape[-1]
    y0 = np.nan_to_num(spec_db[np.maximum(i - 1, 0)], neginf=-400.0)
    y1 = np.nan_to_num(spec_db[i], neginf=-400.0)
    y2 = np.nan_to_num(spec_db[np.minimum(i + 1, n - 1)], neginf=-400.0)
    denom = y0 - 2.0 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom != 0.0, 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.where(np.isfinite(delta) & (np.abs(delta) <= 1.0), delta, 0.0)
    return delta, y1 - 0.25 * (y0 - y2) * delta


def local_maxima(spec_db: np.ndarray, min_bin: int = 1) -> np.ndarray:
    """Indices of strict local maxima in a 1-D dB spectrum."""
    s = np.nan_to_num(np.asarray(spec_db, dtype=np.float64), neginf=-400.0)
    cand = np.zeros(s.shape, dtype=bool)
    cand[1:-1] = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    cand[:max(1, min_bin)] = False
    return np.flatnonzero(cand)


def candidate_tones(agg: Aggregate, cfg: Config, eng: SpectralEngine) -> np.ndarray:
    """Candidate tone bins: structured as a record array, sorted by tone-to-noise desc.

    A bin qualifies if it is a peak in the whole-file mean spectrum, or in any single
    baseline block mean (so a tone that started halfway through the hour is still found),
    at least ``tone_noise_db`` above its spectral neighbourhood and no wider than
    ``max_tone_bins`` (FR-4.1).
    """
    k = cfg.neighbor_k
    lo_bin = int(math.floor(cfg.min_freq / eng.df))
    hi_hz = min(cfg.max_freq if cfg.max_freq else eng.fs / 2.0 * cfg.nyquist_guard,
                eng.fs / 2.0 * cfg.nyquist_guard)
    hi_bin = int(math.ceil(min(hi_hz, eng.freqs[-1] - eng.df) / eng.df))
    hi_bin = max(lo_bin + 2, min(hi_bin, eng.nbins - 1))

    specs = [(0, agg.mean_db)] + [(i + 1, b) for i, b in enumerate(agg.block_db)]
    best: dict[int, tuple[float, float, int]] = {}
    for si, full in specs:
        s = np.nan_to_num(mask_below(full, lo_bin)[lo_bin:hi_bin], neginf=-400.0, nan=-400.0)
        floor = floor_curve(s, k, first_valid=1)
        ttn = s - floor
        for i in local_maxima(s, min_bin=1):
            if ttn[i] < cfg.tone_noise_db or s[i] < cfg.min_level_dbfs:
                continue
            gbin = int(i) + lo_bin
            if gbin < 1 or gbin >= eng.nbins - 1:
                continue
            delta, _lvl = interp_peak(s, int(i))
            t, prev = float(ttn[i]), best.get(gbin)
            if prev is None or t > prev[0]:
                best[gbin] = (t, float(eng.freqs[gbin] + float(delta[0]) * eng.df), si)
    mean_s = np.nan_to_num(mask_below(agg.mean_db, lo_bin), neginf=-400.0, nan=-400.0)
    mean_floor = floor_curve(mean_s, k, first_valid=1)
    rows = []
    for gbin, (ttn, freq, si) in best.items():
        # Bandwidth = run of bins within `tone_width_db` of the tone's OWN peak.  Measuring it
        # against "floor + half the excess" looked equivalent and was not: on this corpus the
        # noise floor falls ~5.5 dB per octave, so a +-250 Hz window underestimates the local
        # floor by ~8 dB, the half-height line lands too low, and an ordinary 3-bin tone measures
        # 4 bins wide and gets rejected (an 880 Hz tone at -40 dBFS vanished this way).  FR-4.1
        # says "confined to <=3 adjacent bins", which is a statement about the tone, not the floor.
        half = mean_s[gbin] - cfg.tone_width_db
        w, j = 1, gbin - 1
        while j >= 0 and mean_s[j] >= half and w <= cfg.max_tone_bins + 1:
            w += 1
            j -= 1
        j = gbin + 1
        while j < mean_s.size and mean_s[j] >= half and w <= cfg.max_tone_bins + 1:
            w += 1
            j += 1
        if w > cfg.max_tone_bins:
            continue
        rows.append((gbin, freq, ttn, float(agg.mean_db[gbin]), w, si))
    rows.sort(key=lambda r: (-r[2], r[1]))
    rows = rows[: cfg.max_candidates]
    dtype = np.dtype([("bin", "i4"), ("freq", "f8"), ("ttn", "f4"), ("mean_db", "f8"),
                      ("width", "i2"), ("first_block", "i2")])
    return np.array(rows, dtype=dtype)
