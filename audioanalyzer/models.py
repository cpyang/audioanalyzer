"""Shared detection record (REQUIREMENTS.md §3, FR-8.1).

One flat dataclass per detection so CSV column order stays stable (NFR-6).  Level fields are
**amplitude-referenced dBFS** unless the name says otherwise; ``MediaInfo`` levels are
RMS-referenced, which differ by 3.01 dB for a pure tone (§2, FR-3.3).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

CSV_COLUMNS = [
    "file", "channel", "class", "freq_hz", "freq_std_hz", "bandwidth_hz", "n_bins",
    "t_start_s", "t_end_s", "duration_s", "t_start_wall", "t_end_wall",
    "level_dbfs", "peak_dbfs", "level_std_db", "tone_noise_db", "prominence_db", "coherence",
    "isolation_db",
    "persistence_pct", "n_episodes", "longest_episode_s", "duty_pct",
    "drift_hz_per_min", "drift_r2", "drift_class",
    "source_id", "source_f0_hz", "n_harmonics", "nn_freq_hz", "nn_ratio",
    "timeline_entry", "known_name", "confidence",
]


@dataclass
class Detection:
    file: str = ""
    channel: int = 0
    cls: str = "A1"                       # class -> CSV key "class"
    freq_hz: float = 0.0
    freq_std_hz: float = 0.0
    bandwidth_hz: float = 0.0
    n_bins: int = 1
    t_start_s: float = 0.0
    t_end_s: float = 0.0
    duration_s: float = 0.0
    t_start_wall: str = ""
    t_end_wall: str = ""
    level_dbfs: float = -999.0            # median level over the track
    peak_dbfs: float = -999.0             # max level
    level_std_db: float = 0.0             # steady tone ~1-2 dB; fluctuating hump ~6 dB
    tone_noise_db: float = 0.0            # spectral prominence (A1 gate)
    coherence: float = 0.0                # phase-lock 0..1 over active frames (FR-4.1)
    isolation_db: float = 0.0             # narrowband vs immediate neighbours (FR-4.1)
    prominence_db: float = 0.0            # temporal prominence vs per-bin baseline
    persistence_pct: float = 0.0
    n_episodes: int = 1
    longest_episode_s: float = 0.0
    duty_pct: float = 0.0
    drift_hz_per_min: float = 0.0
    drift_r2: float = 0.0
    drift_class: str = ""
    source_id: str = ""
    source_f0_hz: float = 0.0
    n_harmonics: int = 0
    nn_freq_hz: float = 0.0              # FR-4.7 nearest other detection, for eyeballing calls
    nn_ratio: float = 0.0                # higher/lower; ~1.00k suggests one machine
    timeline_entry: str = ""             # FR-2.3 corpus id: same feature across chunk seams
    known_name: str = ""
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = {k: (self.cls if k == "class" else getattr(self, k)) for k in CSV_COLUMNS}
        return d

    def key(self) -> str:
        """Stable identity for cross-run diffing (FR-5.3): rounded f0 + span bucket."""
        return hashlib.sha1(f"{round(self.freq_hz, 1):.1f}|{round(self.t_start_s / 60):d}"
                            .encode()).hexdigest()[:12]