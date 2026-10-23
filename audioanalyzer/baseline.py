"""Per-bin noise-floor baseline, accumulated without holding frames (FR-3.5, FR-3.6, FR-6).

Two orthogonal floors are needed and they answer different questions:

``temporal`` (this module)
    percentile of each *bin* over time.  Answers "is this bin louder than it usually is?"
    -> intermittent / emerging tones (A2) and ``compare`` against a reference profile (FR-6).

``spectral`` (:func:`audioanalyzer.spectral.neighbor_median`)
    median of the neighbouring *bins* in one frame.  Answers "does this bin stick out from
    what surrounds it right now?" -> steady tones (A1).

A tone present for the whole recording is its own temporal median, so the temporal floor
alone cannot see it; that is why A1 is gated on tone-to-noise, not on temporal prominence.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BinHistogram:
    """Fixed-width log-domain histogram per magnitude bin; memory independent of length."""

    nbins: int
    lo: float = -160.0
    hi: float = 0.0
    bw: float = 0.5
    counts: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.nb = max(8, int(math.ceil((self.hi - self.lo) / self.bw))) + 1
        if self.counts is None:
            self.counts = np.zeros((self.nbins, self.nb), dtype=np.int64)
        self._colbase = (np.arange(self.nbins, dtype=np.int64) * self.nb).reshape(1, self.nbins)

    # ---- accumulate --------------------------------------------------------
    def update_fast(self, db: np.ndarray) -> None:
        """Add a (frames, nbins) dB matrix; -inf/NaN land in the lowest bucket."""
        x = np.nan_to_num(db, nan=self.lo, neginf=self.lo, posinf=self.hi)
        idx = np.rint((x - self.lo) / self.bw).astype(np.int64)
        np.clip(idx, 0, self.nb - 1, out=idx)
        flat = np.bincount((idx + self._colbase).ravel(), minlength=self.nbins * self.nb)
        self.counts += flat.reshape(self.nbins, self.nb)

    update = update_fast

    # ---- queries -----------------------------------------------------------
    @property
    def total(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    def percentile(self, q: float) -> np.ndarray:
        """Per-bin percentile in dB, by linear interpolation inside the bucket."""
        cum = np.cumsum(self.counts, axis=1)
        tot = np.maximum(cum[:, -1:], 1)
        target = q * tot
        i = np.argmax(cum >= target, axis=1)
        below = np.where(i > 0, cum[np.arange(self.nbins), np.maximum(i - 1, 0)], 0)
        within = np.maximum(cum[np.arange(self.nbins), i] - below, 1)
        frac = (target[:, 0] - below) / within
        return self.lo + (i + frac.clip(0.0, 1.0)) * self.bw

    def merge(self, other: "BinHistogram") -> None:
        if (other.nbins, other.nb, other.lo, other.bw) != (self.nbins, self.nb, self.lo, self.bw):
            raise ValueError("incompatible histograms")
        self.counts += other.counts

    def to_npz(self) -> dict[str, np.ndarray]:
        return {"counts": self.counts, "lo": np.array([self.lo]),
                "bw": np.array([self.bw]), "nb": np.array([self.nb])}

    @classmethod
    def from_npz(cls, d: dict[str, np.ndarray]) -> "BinHistogram":
        lo, bw, nb = float(d["lo"][0]), float(d["bw"][0]), int(d["nb"][0])
        return cls(nbins=int(d["counts"].shape[0]), lo=lo, hi=lo + (nb - 1) * bw, bw=bw,
                   counts=d["counts"].astype(np.int64))


@dataclass
class BaselineModel:
    """Time-blocked collection of :class:`BinHistogram` (FR-3.6)."""

    nbins: int
    block_s: float
    sample_rate: int
    hist_cfg: tuple[float, float, float]
    blocks: list[tuple[float, BinHistogram]] = field(default_factory=list, repr=False)
    total_frames: int = 0
    duration_s: float = 0.0

    def block_for(self, t: float) -> BinHistogram:
        i = min(max(int(t // self.block_s), 0), len(self.blocks) - 1)
        return self.blocks[i][1]

    def _append_block(self) -> None:
        self.blocks.append((len(self.blocks) * self.block_s,
                            BinHistogram(self.nbins, *self.hist_cfg)))

    def ensure_block(self, t: float) -> BinHistogram:
        i = max(int(t // self.block_s), 0)
        while len(self.blocks) <= i:
            self._append_block()
        return self.blocks[i][1]

    def update(self, t_frames: np.ndarray, db: np.ndarray) -> None:
        """Route each frame's spectrum into the baseline block covering its time (FR-3.6)."""
        if db.shape[0] == 0:
            return
        self.ensure_block(float(t_frames[-1]))
        idx = np.clip((t_frames // self.block_s).astype(np.int64), 0, len(self.blocks) - 1)
        order = np.argsort(idx, kind="stable")
        bounds = np.searchsorted(idx[order], np.arange(len(self.blocks) + 1))
        for b in range(len(self.blocks)):
            rows = order[bounds[b]:bounds[b + 1]]
            if rows.size:
                self.blocks[b][1].update_fast(db[rows])
        self.total_frames += int(db.shape[0])

    # ---- convenience views -------------------------------------------------
    def p50(self, t: float = 0.0) -> np.ndarray:
        return self.block_for(t).percentile(0.50)

    def p90(self, t: float = 0.0) -> np.ndarray:
        return self.block_for(t).percentile(0.90)

    def global_percentile(self, q: float) -> np.ndarray:
        tot = BinHistogram(self.nbins, *self.hist_cfg)
        for _, h in self.blocks:
            tot.merge(h)
        return tot.percentile(q)

    # ---- persistence (FR-6.1) ---------------------------------------------
    def save(self, path: str | Path) -> None:
        data: dict[str, np.ndarray] = {
            "nbins": np.array([self.nbins]), "block_s": np.array([self.block_s]),
            "sample_rate": np.array([self.sample_rate]), "hist_cfg": np.array(self.hist_cfg),
            "n_blocks": np.array([len(self.blocks)]),
        }
        for i, (_, h) in enumerate(self.blocks):
            data[f"counts_{i}"] = h.counts.astype(np.int32)
        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        with np.load(str(path)) as d:
            cfg = (float(d["hist_cfg"][0]), float(d["hist_cfg"][1]), float(d["hist_cfg"][2]))
            m = cls(nbins=int(d["nbins"][0]), block_s=float(d["block_s"][0]),
                    sample_rate=int(d["sample_rate"][0]), hist_cfg=cfg)
            for i in range(int(d["n_blocks"][0])):
                lo, hi, bw = cfg
                m.blocks.append((i * m.block_s,
                                 BinHistogram(m.nbins, lo, hi, bw, counts=d[f"counts_{i}"].astype(np.int64))))
            return m
