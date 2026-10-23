"""Baseline floor profiles and compare mode (REQUIREMENTS.md FR-6).

``profile`` answers "what does normal sound like here" and ``compare`` answers the question that
actually gets asked at 07:00: *"did something start making noise?"*  A tone that is present in
today's recording and absent from the reference profile is the single most valuable line in the
whole report, and it cannot be found by any threshold applied to one file in isolation — the
site's own permanent machinery would bury it.

The profile is a per-bin histogram (see :mod:`audioanalyzer.baseline`) plus the mean spectrum,
so it survives being built from recordings of different lengths, and it stays small: a few MB of
integer counts, never audio.

Two honesty rules, both learned the hard way in this project:

* A profile is only valid for the sample rate / window it was built with.  Loading a 16 kHz
  profile against a 48 kHz recording would compare different frequency grids bin by bin and
  report silence as an anomaly, so the check is on load, not on first use.
* ``compare`` reports the coverage of *both* sides.  If the reference night has a 40-minute hole
  and today's recording has a tone inside it, "new" is the wrong word — it is unmeasured.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baseline import BaselineModel, BinHistogram
from .config import Config
from .models import Detection
from .spectral import Aggregate, SpectralEngine, aggregate, candidate_tones, mask_below, floor_curve


@dataclass
class FloorProfile:
    """A reference noise floor: per-bin percentiles plus the mean spectrum they came from."""

    sample_rate: int
    nperseg: int
    hop: int
    mean_db: np.ndarray = field(repr=False)
    model: BaselineModel = field(repr=False)
    files: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    created_utc: str = ""
    cfg_echo: dict = field(default_factory=dict)

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_files(cls, paths, cfg: Config, progress: bool = False) -> "FloorProfile":
        paths = [str(p) for p in paths]
        if not paths:
            raise ValueError("profile needs at least one reference recording")
        model: BaselineModel | None = None
        sum_p: np.ndarray | None = None
        seen = 0
        eng0: SpectralEngine | None = None
        dur = 0.0
        for p in paths:
            agg, eng = aggregate(p, cfg, progress=progress)
            if eng0 is None:
                eng0 = eng
                # A single accumulation block: a profile is one floor, not a time-blocked one.
                # Starting from the first file's own block list would double-count it in
                # global_percentile(), because its blocks would still be present alongside the
                # merged copy.
                model = BaselineModel(nbins=eng.nbins, block_s=max(1.0, 1e6),
                                      sample_rate=eng.fs, hist_cfg=(cfg.hist_db_lo, cfg.hist_db_hi,
                                                                   cfg.hist_db_bw))
                model._append_block()
                sum_p = np.zeros(eng.nbins)
            else:
                if (eng.nperseg, eng.hop, eng.fs) != (eng0.nperseg, eng0.hop, eng0.fs):
                    raise ValueError(f"{p}: analysis grid differs from the profile's first file")
            assert agg.baseline is not None and model is not None
            _merge_into(model, agg.baseline)
            assert agg.mean_db is not None and sum_p is not None
            sum_p += np.where(np.isfinite(agg.mean_db), power_of_db(agg.mean_db), 0.0) * agg.n_frames
            seen += int(agg.n_frames)
            dur += agg.duration_s
        assert eng0 is not None and model is not None and sum_p is not None
        mean_db = 10.0 * np.log10(np.where(seen, sum_p / max(seen, 1), 0.0) + 1e-30)
        model.duration_s = dur
        return cls(sample_rate=eng0.fs, nperseg=eng0.nperseg, hop=eng0.hop, mean_db=mean_db,
                   model=model, files=paths, duration_s=dur,
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   cfg_echo={"tone_noise_db": cfg.tone_noise_db, "min_freq": cfg.min_freq,
                             "max_freq": cfg.max_freq, "neighbor_k": cfg.neighbor_k,
                             "min_level_dbfs": cfg.min_level_dbfs, "mode": cfg.mode})

    # ---- queries -----------------------------------------------------------
    def freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.nperseg, 1.0 / self.sample_rate)

    def percentile(self, q: float) -> np.ndarray:
        return self.model.global_percentile(q)

    def tone_candidates(self, cfg: Config) -> np.ndarray:
        """Stage-1 candidates *in the profile itself* — the tones that are normal here."""
        eng = SpectralEngine(cfg, self.sample_rate)
        if eng.nperseg != self.nperseg or eng.hop != self.hop:
            raise ValueError("profile analysis grid does not match the current --mode")
        agg = Aggregate(sample_rate=self.sample_rate, nperseg=self.nperseg, hop=self.hop,
                        freqs=self.freqs(), df=eng.df, nbins=self.mean_db.size,
                        n_frames=1, duration_s=self.duration_s, mean_db=self.mean_db,
                        block_db=np.empty((0, self.mean_db.size)), baseline=self.model)
        return candidate_tones(agg, cfg, eng)

    # ---- persistence (FR-6.1) ---------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez_compressed(str(path), freqs=self.freqs(), mean_db=self.mean_db,
                            counts=self.model.blocks[0][1].counts.astype(np.int32),
                            p50=self.percentile(0.50), p90=self.percentile(0.90),
                            meta=np.array([json.dumps({
                                "sample_rate": self.sample_rate, "nperseg": self.nperseg,
                                "hop": self.hop, "files": self.files, "created_utc": self.created_utc,
                                "duration_s": round(self.duration_s, 3), "cfg": self.cfg_echo,
                                "hist_cfg": list(self.model.hist_cfg), "nbins": self.model.nbins,
                                "block_s": self.model.block_s})], dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "FloorProfile":
        with np.load(str(path), allow_pickle=True) as d:
            meta = json.loads(str(d["meta"][0]))
            cfg_t = (meta["hist_cfg"][0], meta["hist_cfg"][1], meta["hist_cfg"][2])
            model = BaselineModel(nbins=meta["nbins"], block_s=meta["block_s"],
                                  sample_rate=meta["sample_rate"], hist_cfg=cfg_t)
            model.blocks = [(0.0, BinHistogram(meta["nbins"], *cfg_t, counts=d["counts"]))] \
                if "counts" in d else model.blocks
            return cls(sample_rate=meta["sample_rate"], nperseg=meta["nperseg"], hop=meta["hop"],
                       mean_db=d["mean_db"], model=model, files=meta["files"],
                       duration_s=meta["duration_s"], created_utc=meta["created_utc"],
                       cfg_echo=meta["cfg"])


def power_of_db(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.nan_to_num(db, neginf=-400.0) / 10.0)


def _merge_into(dst: BaselineModel, src: BaselineModel) -> None:
    """Add *src*'s counts into *dst*'s single accumulated block (FR-6.1)."""
    if src.nbins != dst.nbins or src.hist_cfg != dst.hist_cfg:
        raise ValueError("incompatible baselines (different nbins or histogram config)")
    if not dst.blocks:
        dst._append_block()
    for _, h in src.blocks:
        dst.blocks[0][1].counts += h.counts
    dst.total_frames += src.total_frames
    dst.duration_s += src.duration_s


@dataclass
class CompareResult:
    band_edges: np.ndarray
    band_ref_db: np.ndarray
    band_file_db: np.ndarray
    new_tones: list[Detection]
    gone_tones: list[dict]
    coverage_ref_s: float
    coverage_file_s: float

    def to_dict(self) -> dict:
        rise = self.band_file_db - self.band_ref_db
        return {
            "bands": [{"lo_hz": round(float(self.band_edges[i]), 1),
                       "hi_hz": round(float(self.band_edges[i + 1]), 1),
                       "ref_dbfs": round(float(self.band_ref_db[i]), 2),
                       "file_dbfs": round(float(self.band_file_db[i]), 2),
                       "delta_db": round(float(rise[i]), 2)}
                      for i in range(len(self.band_ref_db))],
            "new_tones": [d.to_dict() for d in self.new_tones],
            "gone_tones": self.gone_tones,
            "coverage": {"reference_s": round(self.coverage_ref_s, 3),
                         "file_s": round(self.coverage_file_s, 3)},
        }


def compare(path: str | Path, prof: FloorProfile, cfg: Config,
            agg=None, eng=None) -> CompareResult:
    """What is different about *path* relative to the reference floor (FR-6.2)."""
    if agg is None or eng is None:
        agg, eng = aggregate(path, cfg)
    if eng.nperseg != prof.nperseg or eng.fs != prof.sample_rate:
        raise ValueError("profile and recording analysed on different grids")
    edges = np.geomspace(cfg.min_freq, 0.95 * eng.fs / 2.0, cfg.transient_bands + 1)
    file_db, ref_db = _band_levels(agg.mean_db, eng, edges), _band_levels(prof.mean_db, eng, edges)

    cur = prof.tone_candidates(cfg)
    mine = candidate_tones(agg, cfg, eng)
    # Two tones "match" when they sit within one analysis bin of each other; anything wider is a
    # different machine, not a level change of the same one.
    tol = max(1.5 * eng.df, 2.0)
    gone = [{"freq_hz": round(float(r["freq"]), 2),
             "tone_noise_db": round(float(r["ttn"]), 1),
             "level_dbfs": round(float(r["mean_db"]), 1)}
            for r in cur
            if not any(abs(float(m["freq"]) - float(r["freq"])) <= tol for m in mine)]
    new = [d for d in _as_detections(mine, eng)
           if not any(abs(d.freq_hz - float(r["freq"])) <= tol for r in cur)]
    return CompareResult(band_edges=edges, band_ref_db=ref_db, band_file_db=file_db,
                         new_tones=new, gone_tones=gone, coverage_ref_s=prof.duration_s,
                         coverage_file_s=agg.duration_s)


def _band_levels(mean_db: np.ndarray, eng: SpectralEngine, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, -400.0)
    p = power_of_db(mean_db)
    for b in range(len(edges) - 1):
        cols = np.flatnonzero((eng.freqs >= edges[b]) & (eng.freqs < edges[b + 1]))
        if cols.size:
            out[b] = 10.0 * math.log10(float(p[cols].sum()) + 1e-30)
    return out


def _as_detections(cands, eng: SpectralEngine) -> list[Detection]:
    """Stage-1 candidates as Detection rows so compare output shares the CSV schema (FR-8.1)."""
    out = []
    for r in cands:
        d = Detection(cls="A2", freq_hz=float(r["freq"]), bandwidth_hz=float(eng.df),
                      level_dbfs=float(r["mean_db"]), tone_noise_db=float(r["ttn"]),
                      extra={"source": "compare-new"})
        out.append(d)
    return out
