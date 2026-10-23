"""Configuration and detection thresholds (REQUIREMENTS.md FR-4, §10).

Defaults are precision-first: a detection must clear the adaptive per-bin baseline by
`min_prominence_db` and persist for `min_tone_s` before it is reported.  The ``--recall``
flag relaxes these to a triage-everything profile.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Host-local convention resolved per REQUIREMENTS.md §10 decision 3
# (filename timestamp = capture start in host-local time).
DEFAULT_TZ = "Asia/Taipei"


@dataclass
class Config:
    # --- analysis grid (FR-3.1/3.2) ---
    sample_rate_hint: int = 16000          # informational; actual rate comes from the file
    nperseg: int = 2048                    # 7.8125 Hz/bin at 16 kHz
    noverlap: int = 1536                   # hop = 512 samples = 32 ms @ 16 kHz
    hr_nperseg: int = 16384                # --mode hr -> 0.98 Hz/bin
    block_s: float = 30.0                  # decode/streaming block (FR-1.2)
    baseline_block_s: float = 300.0        # floor re-estimation granularity (FR-3.6)

    # --- tonal detectors A1/A2 (FR-4.1/4.2) ---
    min_prominence_db: float = 6.0
    min_persistence_pct: float = 50.0
    min_tone_s: float = 10.0
    max_tone_bins: int = 3                 # narrowband definition (FR-4.1)
    gap_tolerance_s: float = 2.0           # episode splitter (FR-4.2)
    drift_stable_hz_per_min: float = 0.5   # FR-4.3
    floor_variance_margin_db: float = 3.0  # R1 leakage guard

    # --- transient detector A4 (FR-4.4) ---
    transient_sigma: float = 6.0
    min_event_db: float = 6.0
    z_window_s: float = 60.0
    max_event_s: float = 1.0

    # --- comb / periodicity A3 (FR-4.5) ---
    comb_min_period_s: float = 0.5
    comb_max_period_s: float = 3600.0

    # --- source grouping A5 (FR-5.1) ---
    harmonic_tol_pct: float = 1.0
    cooccurrence_min_jaccard: float = 0.6

    # --- data quality A6 (FR-4.6) ---
    dc_offset_max_dbfs: float = -60.0
    silence_max_s: float = 5.0
    clip_full_scale_tol: float = 1e-4

    # --- reporting ---
    top_n: int = 40
    mode: str = field(default="standard")   # "standard" | "hr"
    tz: str = DEFAULT_TZ
    dbfs_offset: float | None = None        # calibration; None => label outputs dBFS
    exclude_bands: list[tuple[float, float]] = field(default_factory=list)
    min_freq: float = 20.0                  # bin-0 / DC guard (AC-1)
    max_freq: float | None = None           # None => Nyquist

    def resolved(self) -> "Config":
        """Apply --recall relaxation. Precision-first by default (§10 decision 5)."""
        return self

    @property
    def hop(self) -> int:
        return self.nperseg - self.noverlap

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        cfg = cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)}
        for key, value in data.items():
            if key == "exclude_bands":
                value = [tuple(b) for b in value]
            elif key == "mode":
                continue
            elif key not in known:
                raise KeyError(f"unknown config key: {key!r}")
            setattr(cfg, key, value)
        return cfg


RECALL_PRESET = Config(
    min_prominence_db=3.0,
    min_persistence_pct=20.0,
    min_tone_s=2.0,
    transient_sigma=4.0,
    min_event_db=3.0,
)


def recall_preset(base: Config) -> Config:
    """Return a copy of *base* with every threshold relaxed (opt-in triage mode)."""
    cfg = dataclasses.replace(
        base,
        min_prominence_db=min(base.min_prominence_db, RECALL_PRESET.min_prominence_db),
        min_persistence_pct=min(base.min_persistence_pct, RECALL_PRESET.min_persistence_pct),
        min_tone_s=min(base.min_tone_s, RECALL_PRESET.min_tone_s),
        transient_sigma=min(base.transient_sigma, RECALL_PRESET.transient_sigma),
        min_event_db=min(base.min_event_db, RECALL_PRESET.min_event_db),
    )
    return cfg
