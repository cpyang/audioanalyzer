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
    hr_hop: int = 4096                     # --mode hr -> 4 frames/s (1.024 s window, 4x over)
    block_s: float = 30.0                  # decode/streaming block (FR-1.2)
    baseline_block_s: float = 300.0        # floor re-estimation granularity (FR-3.6)

    # --- tonal detectors A1/A2 (FR-4.1/4.2) ---
    min_prominence_db: float = 6.0         # temporal: above this bin's own baseline
    tone_noise_db: float = 6.0             # spectral: tone-to-noise vs +-neighbor_k median
    min_coherence: float = 0.70            # FR-4.1: phase-locked tone vs incoherent band energy
    coherence_smooth_s: float = 1.0        # window over which the tone phase is averaged
    coherence_offset_bank: int = 2         # sweep f0 +-k/L Hz; stage 1 knows f0 to <1 Hz
    min_isolation_db: float = 4.0          # FR-4.1 narrowband: tone vs local band lifting
    isolation_span_bins: int = 5           # look +-5 bins (+-39 Hz at 16 kHz/2048) ...
    isolation_gap_bins: int = 2            # ... skipping the tone's own skirt
    # --- candidate screening (FR-4.1) ---
    min_level_dbfs: float = -120.0         # 16-bit PCM floor is ~-96 dBFS; below this the FFT
                                           # is arithmetic debris, not signal (R1). Processing
                                           # gain means real tones can still sit under -96.
    neighbor_k: int = 32                   # +-250 Hz at 16 kHz / nperseg 2048 (PSDL-style)
    min_persistence_pct: float = 50.0
    min_tone_s: float = 10.0
    max_tone_bins: int = 3                 # narrowband definition (FR-4.1)
    tone_width_db: float = 3.0             # ... measured down from the tone's own peak
    merge_bin_gap: int = 2                 # FR-4.7: candidate bins this close are one feature
    presence_smooth_s: float = 2.0         # FR-4.1: a raw per-frame bin is Rayleigh-noisy
    gap_tolerance_s: float = 2.0           # episode splitter (FR-4.2)
    drift_stable_hz_per_min: float = 0.5   # FR-4.3
    steady_level_std_db: float = 3.0       # A1 vs A2: a steady tone's level σ is 1-2 dB
    floor_variance_margin_db: float = 3.0  # R1 leakage guard

    # --- streaming baseline histogram (FR-3.5/3.6) ---
    hist_db_lo: float = -160.0
    hist_db_hi: float = 0.0
    hist_db_bw: float = 0.5
    max_candidates: int = 400              # stage-2 cost guard, ranked by mean-spectrum ttn

    # --- transient detector A4 (FR-4.4) ---
    transient_sigma: float = 6.0
    min_event_db: float = 15.0             # FR-4.4, calibrated: corpus frames wobble to 8 dB,
                                           # a real impact (even 15 dB below the LF floor) is 33+
    z_window_s: float = 60.0
    max_event_s: float = 1.0
    transient_bands: int = 8               # log bands spanning min_freq..max_freq (FR-4.4)
    mad_floor_db: float = 0.5              # never call 6 sigma of a 0.1 dB spread an event
    event_merge_s: float = 0.25            # frames closer than this are one event
    max_events: int = 500                  # cap kept events per file (report budget, FR-9)
    event_detail_top: int = 40             # how many get the second, spectral-signature pass

    # --- comb / periodicity A3 (FR-4.5) ---
    comb_min_period_s: float = 0.5
    comb_max_period_s: float = 3600.0
    comb_period_resolution_s: float = 0.001  # histogram step; auto-widened for short periods
    period_min_repeats: int = 4            # two clicks are not a train
    period_min_pairs: int = 4              # intervals supporting the candidate period
    period_smooth_s: float = 0.1           # boxcar on the lag axis: two onsets, +-32 ms each
    period_min_fraction: float = 0.15      # ... as a share of all inter-onset intervals
    period_max_report: int = 3
    periodicity_max_onsets: int = 4000     # O(n^2) pairwise-difference guard

    # --- recurrence across a corpus A5/FR-7.2 ---
    recurrence_tol_hz: float = 2.0         # cluster detections as "the same tone" ...
    recurrence_tol_pct: float = 0.5        # ... whichever tolerance is wider
    recurrence_min_episodes: int = 2

    # --- source grouping A5 (FR-5.1) ---
    harmonic_tol_pct: float = 1.0
    cooccurrence_min_jaccard: float = 0.6

    # --- data quality A6 (FR-4.6) ---
    dc_offset_max_dbfs: float = -60.0
    silence_max_s: float = 5.0
    clip_full_scale_tol: float = 1e-4

    # --- reporting ---
    top_n: int = 40
    enable_events: bool = True             # A3/A4 transients (FR-4.4/4.5); --no-events disables
    enable_tonal: bool = True              # A1/A2 tones; --class transient disables
    gap_warn_s: float = 2.0                # FR-2.4: inter-chunk silence worth listing in the ledger
    debug_groups: bool = False             # per-candidate stage-2 diagnostics on stderr
    mode: str = field(default="standard")   # "standard" | "hr"
    tz: str = DEFAULT_TZ
    dbfs_offset: float | None = None        # calibration; None => label outputs dBFS
    exclude_bands: list[tuple[float, float]] = field(default_factory=list)
    min_freq: float = 20.0                  # bin-0 / DC guard (AC-1)
    max_freq: float | None = None          # None => the Nyquist guard below
    nyquist_guard: float = 0.98            # ignore the top 2% of the band (R1/R3): a 44.1k->16k
                                           # SoX resample leaves a coherent line *at* Nyquist,
                                           # which scores high phase coherence and is not a machine

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
    tone_noise_db=3.0,
    min_prominence_db=3.0,
    min_persistence_pct=20.0,
    min_tone_s=2.0,
    transient_sigma=4.0,
    min_event_db=3.0,
    min_coherence=0.5,
    min_isolation_db=2.0,
)


def recall_preset(base: Config) -> Config:
    """Return a copy of *base* with every threshold relaxed (opt-in triage mode).

    Includes ``tone_noise_db``: it gates both detector stages, so a preset that omitted it
    would leave ``--recall`` almost inert for tonal work.  Same argument for the coherence and
    isolation gates added in M1 — relaxing only the level gate finds nothing new.
    """
    return dataclasses.replace(
        base,
        tone_noise_db=min(base.tone_noise_db, RECALL_PRESET.tone_noise_db),
        min_prominence_db=min(base.min_prominence_db, RECALL_PRESET.min_prominence_db),
        min_persistence_pct=min(base.min_persistence_pct, RECALL_PRESET.min_persistence_pct),
        min_tone_s=min(base.min_tone_s, RECALL_PRESET.min_tone_s),
        transient_sigma=min(base.transient_sigma, RECALL_PRESET.transient_sigma),
        min_event_db=min(base.min_event_db, RECALL_PRESET.min_event_db),
        min_coherence=min(base.min_coherence, RECALL_PRESET.min_coherence),
        min_isolation_db=min(base.min_isolation_db, RECALL_PRESET.min_isolation_db),
    )
