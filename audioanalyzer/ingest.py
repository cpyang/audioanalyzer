"""Block-based audio ingest and media metadata (REQUIREMENTS.md FR-1, FR-2, A6).

Never materialises the whole file: decodes fixed-size blocks through ``soundfile``
(libsndfile handles FLAC natively, no ffmpeg at runtime) and reduces each block into
streaming statistics.  Peak memory therefore does not scale with file length (NFR-5).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

from .config import Config

# 20261023T070002.flac  ->  capture start, host-local (§10 decision 3)
_NAME_TS = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})T(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})")


@dataclass
class MediaInfo:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    subtype: str
    format: str
    size_bytes: int
    tags: dict[str, str] = field(default_factory=dict)
    capture_start: datetime | None = None
    tz: str | None = None
    # streaming measurements
    dc_offset: float = 0.0
    peak_abs: float = 0.0
    peak_count: int = 0
    rms_dbfs: float = -math.inf
    block_rms_dbfs: list[float] = field(default_factory=list)
    block_seconds: float = 0.0
    clipped_samples: int = 0
    clip_level: float = 0.0
    full_scale_run_samples: int = 0
    non_finite: int = 0
    silence_s: float = 0.0

    # ---- derived properties -------------------------------------------------
    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def bits(self) -> int:
        return {"PCM_16": 16, "PCM_24": 24, "PCM_S24LE": 24, "PCM_32": 32,
                "PCM_8": 8, "FLOAT": 32, "DOUBLE": 64,
                "PCM_U8": 8}.get(self.subtype, 0)

    @property
    def duration(self) -> str:
        """Duration as H:MM:SS (human form of ``duration_s``)."""
        return _hms(self.duration_s)

    @property
    def rms_db(self) -> float:
        return self.rms_dbfs

    def wall_clock(self, t_offset_s: float) -> datetime | None:
        """File offset -> absolute time (FR-2.1)."""
        if self.capture_start is None:
            return None
        return self.capture_start + timedelta(seconds=float(t_offset_s))

    def quality_flags(self, cfg: Config) -> list[dict]:
        """A6 data-quality anomalies (FR-4.6)."""
        flags: list[dict] = []
        if self.dc_offset_dbfs is not None and self.dc_offset_dbfs > cfg.dc_offset_max_dbfs:
            flags.append({"kind": "dc_offset", "value": self.dc_offset,
                          "level_dbfs": round(self.dc_offset_dbfs, 2),
                          "threshold_dbfs": cfg.dc_offset_max_dbfs})
        if self.peak_dbfs > -0.1:
            flags.append({"kind": "clipping",
                          "definition": f"|x| >= {self.clip_level:.8f} (top PCM code)",
                          "clipped_samples": self.clipped_samples,
                          "peak_count_at_max": self.peak_count,
                          "longest_flat_top_s": round(self.full_scale_run_s, 4),
                          "peak_dbfs": round(self.peak_dbfs, 3)})
        if self.non_finite:
            flags.append({"kind": "non_finite", "count": self.non_finite})
        if self.silence_s > cfg.silence_max_s:
            flags.append({"kind": "silence", "seconds": round(self.silence_s, 2),
                          "threshold_s": cfg.silence_max_s})
        return flags

    @property
    def dc_offset_dbfs(self) -> float | None:
        return 20.0 * math.log10(abs(self.dc_offset)) if abs(self.dc_offset) > 0 else None

    @property
    def peak_dbfs(self) -> float:
        return 20.0 * math.log10(self.peak_abs) if self.peak_abs > 0 else -math.inf

    @property
    def full_scale_run_s(self) -> float:
        """Longest run of consecutive full-scale samples, in seconds (clipping evidence)."""
        return self.full_scale_run_samples / self.sample_rate if self.sample_rate else 0.0

    def to_dict(self) -> dict:
        d = {
            "path": self.path, "format": self.format, "subtype": self.subtype,
            "sample_rate": self.sample_rate, "channels": self.channels,
            "frames": self.frames, "duration_s": round(self.duration_s, 6),
            "duration": _hms(self.duration_s), "size_bytes": self.size_bytes,
            "bits": self.bits, "nyquist_hz": self.nyquist,
            "capture_start": self.capture_start.isoformat() if self.capture_start else None,
            "tz": self.tz, "tags": self.tags,
            "dc_offset": round(self.dc_offset, 8),
            "dc_offset_dbfs": None if self.dc_offset_dbfs is None else round(self.dc_offset_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 3), "peak_count": self.peak_count,
            "clipped_samples": self.clipped_samples, "rms_dbfs": round(self.rms_dbfs, 3),
            "clip_level": round(self.clip_level, 7),
            "full_scale_run_s": round(self.full_scale_run_s, 4),
            "non_finite": self.non_finite, "silence_s": round(self.silence_s, 2),
            "block_seconds": self.block_seconds,
        }
        return d


def _clip_level(subtype: str, bits: int) -> float:
    """Magnitude at (or beyond) which a sample is considered hard-clipped.

    For signed integer PCM of *B* bits the largest positive representable value is
    ``(2**(B-1) - 1) / 2**(B-1)`` = ``1 - 2**(1-B)`` — *not* ``1 - 2**-B``, which sits
    half an LSB above the top code and would never match a clipped sample.  ``-1.0``
    (``-2**(B-1)``) is also >= this level, so both polarities are counted.
    For float formats only ``|x| >= 1.0`` is real clipping.
    """
    if subtype.startswith("FLOAT") or subtype == "DOUBLE":
        return 1.0
    if subtype.startswith("PCM_U"):
        return 1.0
    if bits < 8:
        return 1.0 - 1e-6
    return 1.0 - 2.0 ** (1 - bits)


def _read_tags(fh) -> dict[str, str]:
    """Vendor/chain provenance tags (FR-1.4).

    ``soundfile`` exposes Vorbis comments only inside the ``extra_info`` text blob, so
    parse them; keep the raw blob as ``extra_info`` so nothing is silently dropped.
    """
    tags: dict[str, str] = {}
    for attr in ("lines", "caption", "history"):
        value = getattr(fh, attr, None)
        if value:
            tags[attr] = str(value)
    try:
        blob = fh.extra_info or ""
    except Exception:  # pragma: no cover - very old soundfile
        blob = ""
    if blob:
        tags["extra_info"] = blob
        section = None
        for line in blob.splitlines():
            stripped = line.strip()
            if stripped.lower().endswith("metadata"):
                section = stripped.split()[0].lower()
                continue
            if section and " : " in stripped:
                key, _, val = stripped.partition(" : ")
                if key.strip().lower() != "vendor":
                    tags.setdefault(key.strip().lower(), val.strip())
    return tags


def capture_start_from_name(path: str | Path, tz: str | None = None) -> datetime | None:
    """Parse the ``YYYYMMDDTHHMMSS`` filename convention into an aware datetime."""
    m = _NAME_TS.match(Path(path).name)
    if not m:
        return None
    naive = datetime(int(m["y"]), int(m["m"]), int(m["d"]),
                     int(m["H"]), int(m["M"]), int(m["S"]))
    if tz:
        return naive.replace(tzinfo=ZoneInfo(tz))
    return naive.astimezone()


def probe(path: str | Path, cfg: Config | None = None,
          progress: bool = False) -> MediaInfo:
    """Decode *path* in blocks and return its :class:`MediaInfo` (FR-1.3)."""
    cfg = cfg or Config()
    path = Path(path)
    with sf.SoundFile(str(path)) as fh:
        info = MediaInfo(
            path=str(path), sample_rate=int(fh.samplerate), channels=int(fh.channels),
            frames=int(fh.frames), duration_s=float(fh.frames) / float(fh.samplerate),
            subtype=str(fh.subtype), format=str(fh.format), size_bytes=path.stat().st_size,
            tags=_read_tags(fh),
            tz=cfg.tz,
        )
        block = max(1, int(round(cfg.block_s * info.sample_rate)))
        info.block_seconds = block / info.sample_rate
        info.capture_start = capture_start_from_name(path, cfg.tz)

        clip_level = _clip_level(info.subtype, info.bits)
        sum_sq = 0.0
        peak, peak_count = 0.0, 0
        n_nonfinite = 0
        n_clipped = 0
        run_carry = 0
        longest_run = 0
        sum_all = 0.0
        longest_silent_run = 0
        cur_silent_run = 0
        done = 0
        for blk in fh.blocks(blocksize=block, dtype="float64", always_2d=True):
            if info.channels > 1:
                data = blk.T[0]
                # channel-0 stats only for the summary; per-channel handled by detectors
            else:
                data = blk[:, 0]
            finite = np.isfinite(data)
            n_nonfinite += int(data.size - np.count_nonzero(finite))
            if not finite.all():
                data = np.where(finite, data, 0.0)
            a = np.abs(data)
            bmax = float(a.max()) if a.size else 0.0
            if bmax > peak:
                peak, peak_count = bmax, int(np.count_nonzero(a == bmax))
            elif bmax == peak:
                peak_count += int(np.count_nonzero(a == bmax))
            at_fs = a >= clip_level
            n_clipped += int(np.count_nonzero(at_fs))
            if at_fs.any():
                padded = np.concatenate(([False], at_fs, [False]))
                edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
                runs = edges[1::2] - edges[0::2]
                if at_fs[0]:
                    longest_run = max(longest_run, run_carry + int(runs[0]))
                run_carry = int(runs[-1]) if at_fs[-1] else 0
                longest_run = max(longest_run, int(runs.max()))
            sum_sq += float(np.dot(data, data))
            sum_all += float(data.sum())
            block_rms = math.sqrt(float(np.dot(data, data)) / data.size) if data.size else 0.0
            info.block_rms_dbfs.append(20.0 * math.log10(block_rms) if block_rms > 0 else -math.inf)
            # block-level silence for the A6 silence check
            if block_rms < 10.0 ** (-90.0 / 20.0):
                cur_silent_run += data.size
                longest_silent_run = max(longest_silent_run, cur_silent_run)
            else:
                cur_silent_run = 0
            done += data.size
            if progress:
                print(f"  decode {100.0 * done / max(1, info.frames):.0f}%", end="\r", flush=True)
        if progress:
            print()
        info.clipped_samples = n_clipped
        info.clip_level = clip_level
        info.full_scale_run_samples = longest_run
        info.peak_abs = peak
        info.peak_count = peak_count
        info.non_finite = n_nonfinite
        info.dc_offset = sum_all / max(1, info.frames)
        info.rms_dbfs = (20.0 * math.log10(math.sqrt(sum_sq / info.frames))
                         if sum_sq > 0 else -math.inf)
        info.silence_s = longest_silent_run / info.sample_rate
    return info


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
