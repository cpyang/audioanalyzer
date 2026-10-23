"""Single-file scan pipeline (REQUIREMENTS.md FR-1..FR-4).

Two streaming passes: stage 1 aggregates (mean/block spectra + per-bin histograms), stage 2
measures persistence for the candidate bins stage 1 proposed.  Each pass decodes the file once,
so peak memory is a few tens of MB regardless of file length (NFR-5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .detectors.periodicity import detect_periodicity
from .detectors.tonal import detect_tonal
from .detectors.transients import band_edges, detect_transients
from .ingest import MediaInfo, probe
from .models import Detection
from .spectral import Aggregate, SpectralEngine, aggregate, candidate_tones


@dataclass
class ScanResult:
    path: str
    media: MediaInfo
    agg: Aggregate = field(repr=False)
    engine: SpectralEngine = field(repr=False)
    detections: list[Detection] = field(default_factory=list)
    candidates: object = field(default=None, repr=False)
    elapsed_s: float = 0.0

    def summary(self) -> dict:
        cfg = self.engine.cfg
        return {
            "file": self.path,
            "analysis": {
                "mode": cfg.mode, "nperseg": self.engine.nperseg, "hop": self.engine.hop,
                "df_hz": round(self.engine.df, 6), "frames": self.agg.n_frames,
                "neighbor_k": cfg.neighbor_k, "baseline_blocks": len(self.agg.block_t0),
                "candidates": int(0 if self.candidates is None else len(self.candidates)),
                "level_reference": "amplitude dBFS (FR-3.3)",
                "min_level_dbfs": cfg.min_level_dbfs,
                "tone_noise_db": cfg.tone_noise_db,
                "min_tone_s": cfg.min_tone_s,
                "min_coherence": cfg.min_coherence,        # the three stage-2 gates (FR-4.1)
                "min_isolation_db": cfg.min_isolation_db,
                "presence_smooth_s": cfg.presence_smooth_s,
                "nyquist_guard": cfg.nyquist_guard,
                "transient_sigma": cfg.transient_sigma,
                "min_event_db": cfg.min_event_db,
                "gap_tolerance_s": cfg.gap_tolerance_s,
                "detectors": {"tonal": cfg.enable_tonal, "events": cfg.enable_events},
                "tz": cfg.tz,
            },
            "elapsed_s": round(self.elapsed_s, 3),
            "detection_count": len(self.detections),
        }


def scan_file(path: str | Path, cfg: Config | None = None, progress: bool = False,
              media: MediaInfo | None = None) -> ScanResult:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    info = media or probe(path, cfg)
    agg, eng = aggregate(path, cfg, progress=progress)
    cands = candidate_tones(agg, cfg, eng)
    dets = detect_tonal(path, cfg, agg, eng, cands) if cfg.enable_tonal else []
    if cfg.enable_events:
        ev, onsets = detect_transients(path, cfg, agg, eng, media=info)
        dets += detect_periodicity(onsets, cfg, band_edges(eng, cfg), agg.duration_s, str(path))
        dets += ev
    for d in dets:
        d.file = str(path)
        d.channel = 0 if info.channels == 1 else -1
        if info.capture_start is not None:
            d.t_start_wall = info.wall_clock(d.t_start_s).isoformat()
            d.t_end_wall = info.wall_clock(d.t_end_s).isoformat()
    return ScanResult(path=str(path), media=info, agg=agg, engine=eng, detections=dets,
                      candidates=cands, elapsed_s=time.perf_counter() - t0)
