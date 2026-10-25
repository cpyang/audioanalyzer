"""Figures for the HTML report (REQUIREMENTS.md FR-8.2) — matplotlib, Agg backend, no display.

Two views per detection, because they answer different questions:

* the **spectrogram crop** (±30 s, ±1 kHz) says whether a line is really a line, or the edge of
  a broad hump, and whether it starts when the report says it starts;
* the **waveform zoom** (±0.5 s) says whether it is modulation or an impulse — an impact looks
  like a clipped spike with ringing, a tone looks like a thickened band of constant width.

Crops are decoded directly (seek + read), never the whole file: an hour at 16 kHz float is
229 MB, and a report with 40 figures must not need that.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import colormaps
from matplotlib.figure import Figure

from .config import Config
from .spectral import SpectralEngine


def _fig_to_png_b64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#111418")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cmap() -> Colormap:
    return colormaps["magma"].copy()


def crop_grid(path: str | Path, t_center: float, f_center: float, cfg: Config, *,
              span_s: float = 30.0, span_hz: float = 1000.0,
              nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, freqs_in_band, dB[bins, frames]) around a point.

    Split out of `spectrogram_crop` so the geometry can be asserted numerically. A PNG that renders
    is not evidence the picture is right — the bug this function exists to prevent produced a valid
    image path that raised before ever reaching a file, and its cousin (axes swapped) would have
    produced a plausible-looking image of the wrong thing.
    """
    with sf.SoundFile(str(path)) as fh:
        sr = fh.samplerate
        total = fh.frames
        hop = max(1, nperseg // 4)
        t0 = max(0.0, t_center - span_s / 2.0)
        n = int(min(total - int(t0 * sr), span_s * sr))
        fh.seek(int(t0 * sr))
        x = fh.read(n, dtype="float32")
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < nperseg + hop:
        return np.array([]), np.array([]), np.zeros((0, 0))
    frames = 1 + (x.size - nperseg) // hop
    idx = np.arange(nperseg)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(nperseg)
    seg = (x[idx] - x[idx].mean(axis=1, keepdims=True)) * win
    spec = np.abs(np.fft.rfft(seg, axis=1)) / (win.sum() / 2.0)
    db = 20.0 * np.log10(np.maximum(spec, 1e-9))
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
    fband = (freqs >= max(0.0, f_center - span_hz)) & (freqs <= f_center + span_hz)
    if fband.sum() < 3:
        fband = np.ones(freqs.size, dtype=bool)
    t = (np.arange(frames) * hop + nperseg / 2.0) / sr + t0
    return t, freqs[fband], db[:, fband].T


def spectrogram_crop(path: str | Path, t_center: float, f_center: float, cfg: Config,
                     span_s: float = 30.0, span_hz: float = 1000.0,
                     nperseg: int = 1024,
                     db_range: tuple[float, float] | None = None) -> str:
    """dB-colourmap spectrogram around (t_center, f_center), y in Hz (FR-8.2).

    The unused ``fs=`` parameter was dropped here: nothing in the tool ever passed it, and a
    parameter that a helper quietly ignores is where the next wrong-axis bug hides.
    """
    t, fb, z = crop_grid(path, t_center, f_center, cfg, span_s=span_s, span_hz=span_hz,
                         nperseg=nperseg)
    if z.size == 0:
        return ""
    fig = Figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(111)
    vmin, vmax = db_range or (np.percentile(z, 5), np.percentile(z, 99.5))
    ax.pcolormesh(t, fb, z, shading="gouraud", cmap=_cmap(), vmin=vmin, vmax=vmax)
    ax.set_ylim(fb[[0, -1]])
    ax.set_xlim(t[[0, -1]])
    ax.set_ylabel("Hz")
    ax.tick_params(colors="#cfd6dd", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#3a4048")
    ax.set_facecolor("#111418")
    fig.patch.set_facecolor("#111418")
    ax.set_title(f"{Path(str(path)).name}  t={t_center:.1f}s  f={f_center:.1f}Hz",
                 fontsize=8, color="#cfd6dd", loc="left")
    fig.tight_layout(pad=0.4)
    return _fig_to_png_b64(fig)


def waveform_zoom(path: str | Path, t_center: float, cfg: Config, span_s: float = 1.0,
                  fs: int | None = None) -> str:
    """Time waveform around an event: impulse and modulation do not look the same (FR-8.2)."""
    with sf.SoundFile(str(path)) as fh:
        sr = fs or fh.samplerate
        total = fh.frames
        t0 = max(0.0, t_center - span_s / 2.0)
        n = int(min(total - int(t0 * sr), span_s * sr))
        fh.seek(int(t0 * sr))
        x = fh.read(n, dtype="float32")
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        return ""
    t = np.arange(x.size) / sr + t0
    fig = Figure(figsize=(7.2, 1.5))
    ax = fig.add_subplot(111)
    ax.plot(t, x, lw=0.5, color="#7fd1ff")
    ax.set_ylim(-1.02, 1.02)
    ax.axhline(0.0, color="#3a4048", lw=0.6)
    ax.tick_params(colors="#cfd6dd", labelsize=7)
    ax.set_facecolor("#111418")
    fig.patch.set_facecolor("#111418")
    ax.set_title(f"waveform  t={t0:.3f}..{t0 + n / sr:.3f}s", fontsize=8, color="#cfd6dd",
                 loc="left")
    fig.tight_layout(pad=0.4)
    return _fig_to_png_b64(fig)


def overview_spectrogram(path: str | Path, cfg: Config, height_s: float = 3.0,
                         log_freq: bool = True) -> str:
    """Whole-recording spectrogram, log frequency — how a human reads it (FR-8.2)."""
    with sf.SoundFile(str(path)) as fh:
        sr, total = fh.samplerate, fh.frames
    duration = total / sr
    # Fixed output grid: coarse for a long file, and the window is chosen so the *number of
    # columns* stays bounded rather than the sample count.
    cols = 900
    hop = max(1, int(total / cols))
    nperseg = max(256, int(2 ** np.ceil(np.log2(max(hop * 4, sr / 200.0)))))
    nperseg = min(nperseg, 8192, total)
    win = np.hanning(nperseg)
    rows = 1 + max(0, (total - nperseg)) // hop
    mag = np.zeros((int(np.fft.rfftfreq(nperseg, 1.0 / sr).size), rows), dtype=np.float32)
    with sf.SoundFile(str(path)) as fh:
        for k in range(rows):
            fh.seek(k * hop)
            x = fh.read(nperseg, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if x.size < nperseg:
                x = np.concatenate([x, np.zeros(nperseg - x.size, dtype=np.float32)])
            mag[:, k] = np.abs(np.fft.rfft((x - x.mean()) * win)) / (win.sum() / 2.0)
    db = 20.0 * np.log10(np.maximum(mag, 1e-9))
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
    # Colour by dB over an adaptive *frequency* floor, not by absolute dBFS.  On absolute scale the
    # 47 dB LF->HF slope of the noise floor eats the whole colour budget: measured on the reference
    # hour, 95.8 % of the image variance was that slope and the informative residual spanned only
    # -3.1..+12.5 dB, so a real 26 dB-over-floor tone rendered as background.  Percentile windows
    # made it worse: the bottom 2 % of cells are numerical dust from masked bins (vmin came out at
    # -122 dBFS, 28 dB below the genuine floor).
    from .spectral import mask_below, neighborhood_floor
    first = max(1, int(np.ceil(cfg.min_freq / max(1e-6, freqs[1] - freqs[0]))))
    floor = neighborhood_floor(mask_below(db, first), cfg.neighbor_k, axis=0)
    show = np.nan_to_num(db - floor, nan=np.nan, posinf=0.0, neginf=-40.0)
    fig = Figure(figsize=(11.0, height_s))
    ax = fig.add_subplot(111)
    t = (np.arange(rows) * hop + nperseg / 2.0) / sr
    keep = freqs >= cfg.min_freq
    if log_freq:
        ax.set_yscale("log")
    ax.pcolormesh(t, freqs[keep], show[keep], shading="gouraud", cmap=_cmap(),
                  vmin=-4.0, vmax=24.0)
    ax.set_xlim(t[[0, -1]])
    ax.set_ylim((freqs[keep][1] if log_freq else freqs[keep][0], freqs[keep][-1]))
    ax.set_ylabel("Hz (log)\ncolour: dB over\nlocal floor" if log_freq else "Hz\ncolour: dB over\nlocal floor")
    ax.set_xlabel("s from file start")
    ax.tick_params(colors="#cfd6dd", labelsize=7)
    ax.set_facecolor("#111418")
    fig.patch.set_facecolor("#111418")
    fig.tight_layout(pad=0.5)
    return _fig_to_png_b64(fig)
