"""Synthetic test signals with *known* ground truth (NFR-8, AC-4/AC-5).

Detector math is tested against signals whose tone frequencies, levels, episodes and event
times are chosen here, so a failure is a detector bug rather than a difference of opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy.signal import get_window

from .spectral import sliding_frames

SR_DEFAULT = 16000


@dataclass
class Truth:
    tones: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    sample_rate: int = SR_DEFAULT


def add_tone(x: np.ndarray, fs: int, freq: float, level_dbfs: float,
            t0: float = 0.0, t1: float | None = None, phase: float = 0.0) -> np.ndarray:
    """Add a sine at amplitude-referenced ``level_dbfs`` over [t0, t1) (FR-3.3 convention)."""
    t1 = x.size / fs if t1 is None else t1
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    i1 = min(i1, x.size)
    if i1 <= i0:
        return x
    n = np.arange(i1 - i0)
    a = 10.0 ** (level_dbfs / 20.0)
    x[i0:i1] += a * np.sin(2 * math.pi * freq * n / fs + phase)
    return x


def gaussian_noise(fs: int, seconds: float, psd_dbfs_hz: float, rng=None) -> np.ndarray:
    """White noise with a given one-sided PSD in dBFS/Hz (so bin levels are predictable)."""
    rng = rng or np.random.default_rng(0)
    n = int(fs * seconds)
    sigma = math.sqrt(10.0 ** (psd_dbfs_hz / 10.0) * fs / 2.0)
    return rng.normal(0.0, sigma, n)


def add_click(x: np.ndarray, fs: int, t: float, level_dbfs: float,
              width_samples: int = 8, ring_hz: float | None = None,
              ring_ms: float = 40.0) -> np.ndarray:
    """Add an impulse.  With ``ring_hz`` it rings like a struck bar instead of an ADC spike.

    A pure 8-sample sign flip has a flat spectrum, which no real impact does; the damped sine
    version is what a knock on a duct or a machine guard looks like, and it is what the A4
    spectral-signature report is tested against (FR-4.4).
    """
    i = int(round(t * fs))
    if i >= x.size:
        return x
    amp = 10.0 ** (level_dbfs / 20.0)
    if ring_hz is None:
        i1 = min(i + width_samples, x.size)
        x[i:i1] = np.sign(x[i:i1] + 1.0) * amp
        return x
    n = int(fs * ring_ms / 1000.0)
    i1 = min(i + n, x.size)
    k = np.arange(i1 - i)
    env = np.exp(-k / (0.15 * fs * ring_ms / 1000.0 * 10.0))
    x[i:i1] += amp * env * np.sin(2.0 * np.pi * ring_hz * k / fs)
    return x


# Measured mean-spectrum envelope of 20261023T070002.flac (amplitude dBFS *per bin* at
# df = 7.8125 Hz, i.e. nperseg 2048 @ 16 kHz).  Smoothed/monotoned; these are data, not
# assumptions -- see REQUIREMENTS.md §2. If nperseg changes, convert to PSD: dB/Hz = dB - 10log10(df).
_RUMBLE_ANCHORS_HZ_DB = [(9.0, -57.0), (25.0, -55.5), (62.5, -53.5), (95.0, -55.5),
                         (125.0, -59.4), (141.0, -62.0), (148.0, -66.8), (172.0, -67.5),
                         (203.0, -69.8), (234.0, -69.7), (312.5, -75.8),
                         (625.0, -81.3), (1250.0, -86.9), (2500.0, -92.3), (5000.0, -96.4),
                         (7900.0, -100.0)]


def rumble(fs: int, seconds: float, knee_hz: float = 150.0, dc_offset: float = -0.01224,
           seed: int = 0) -> np.ndarray:
    """Noise shaped like this corpus, calibrated to the *measured* mean spectrum.

    The shaping curve is a log-log interpolation through :data:`_RUMBLE_ANCHORS_HZ_DB`, i.e.
    a plateau of about -53 dBFS near 60 Hz rolling off to -96 dBFS at 5 kHz -- decidedly not
    white, and not a clean power law either.  Plus the measured -38.25 dBFS DC offset.

    Detectors are tested against this rather than the real recordings so the repository stays
    audio-free (``.gitignore``) while still facing a realistic floor (NFR-8, R1).
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    ax = np.array([a[0] for a in _RUMBLE_ANCHORS_HZ_DB], dtype=np.float64)
    ay = np.array([a[1] for a in _RUMBLE_ANCHORS_HZ_DB], dtype=np.float64)
    target = np.interp(np.log10(np.clip(f, ax[0] * 0.5, ax[-1])), np.log10(ax), ay)
    shape = np.power(10.0, target / 20.0)
    shape[0] = 0.0                                    # DC added separately below
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n)
    ref = _bin_level(x, fs, 1000.0)                   # cancel the white-noise/PSD constant
    x *= 10.0 ** ((_db_at(1000.0) - ref) / 20.0)
    return x + dc_offset


def _db_at(freq: float) -> float:
    ax = np.array([a[0] for a in _RUMBLE_ANCHORS_HZ_DB])
    ay = np.array([a[1] for a in _RUMBLE_ANCHORS_HZ_DB])
    return float(np.interp(np.log10(max(freq, 1e-3)), np.log10(ax), ay))


def _bin_level(x: np.ndarray, fs: int, freq: float, nperseg: int = 2048) -> float:
    """Amplitude-referenced dBFS of the bin nearest *freq*, averaged over the signal."""
    w = get_window("hann", nperseg, fftbins=False)
    frames = sliding_frames(x.astype(np.float32), nperseg, nperseg // 4)
    if frames.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(frames * w.astype(np.float32), axis=1)).mean(axis=0)
    k = int(round(freq / (fs / nperseg)))
    k = min(max(k, 1), mag.size - 2)
    return float(20.0 * math.log10(max(2.0 * mag[k] / w.sum(), 1e-12)))


def synth(duration_s: float = 60.0, fs: int = SR_DEFAULT, psd_dbfs_hz: float = -60.0,
          tones: list[tuple[float, float, float, float]] | None = None,
          events: list[tuple[float, float]] | None = None,
          seed: int = 0) -> tuple[np.ndarray, Truth]:
    """Build a signal. ``tones``: (freq, level_dbfs, t0, t1); ``events``: (t, level_dbfs)."""
    rng = np.random.default_rng(seed)
    x = gaussian_noise(fs, duration_s, psd_dbfs_hz, rng)
    truth = Truth(duration_s=duration_s, sample_rate=fs)
    for freq, level, t0, t1 in (tones or []):
        x = add_tone(x, fs, freq, level, t0, t1)
        truth.tones.append({"freq_hz": freq, "level_dbfs": level,
                            "t0": t0, "t1": duration_s if t1 is None else t1})
    for t, level in (events or []):
        x = add_click(x, fs, t, level)
        truth.events.append({"t": t, "level_dbfs": level})
    return x, truth


def write(path, x: np.ndarray, fs: int = SR_DEFAULT, subtype: str = "PCM_16") -> None:
    peak = float(np.max(np.abs(x))) if x.size else 1.0
    if peak > 0.99:                       # keep synthetic files away from the clipper
        x = x * (0.99 / peak)
    sf.write(str(path), x, fs, subtype=subtype, format="FLAC")
