"""A5 source grouping: collapse related tonal detections into one physical source.

Two independent links are used, as required by FR-5.1:

* harmonic  - members sit at ``k * f0`` within ``harmonic_tol_pct``.  Note the fundamental
  itself may be absent (a bearing whine at 3x shaft rate, a PSU at 2x mains), so f0 is
  *inferred* from the observed set rather than assumed to be one of the detections.
* temporal  - members switch on and off together (Jaccard of active time >= threshold).

Each source gets a deterministic id from its rounded f0 (FR-5.3) so two runs on the same
file can be diffed.  Sources, not raw detections, are what an engineer should walk toward:
one machine, one row.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..models import Detection


@dataclass
class Source:
    source_id: str
    f0_hz: float
    f0_uncertainty_hz: float
    members: list[Detection] = field(default_factory=list)
    n_harmonics: int = 0
    harmonic_numbers: list[int] = field(default_factory=list)
    t_first_s: float = 0.0
    t_last_s: float = 0.0
    active_s: float = 0.0
    duty_pct: float = 0.0
    n_episodes: int = 0
    total_ttn_db: float = 0.0
    level_dbfs: float = -999.0
    drift_class: str = ""
    known_name: str = ""
    link_reason: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "members"} | {
            "members": [{"freq_hz": m.freq_hz, "class": m.cls, "tone_noise_db": m.tone_noise_db,
                         "level_dbfs": m.level_dbfs, "detection_key": m.key()}
                        for m in self.members]}


def _intervals(det: Detection) -> list[tuple[float, float]]:
    eps = det.extra.get("episodes") or [[det.t_start_s, det.t_end_s]]
    return [(float(a), float(b)) for a, b in eps]


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    out: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _length(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def _jaccard(iv_a: list[tuple[float, float]], iv_b: list[tuple[float, float]]) -> float:
    """Overlap of active time between two detections (FR-5.1b)."""
    a, b = _union(iv_a), _union(iv_b)
    inter = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            inter += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    union = _length(a) + _length(b) - inter
    return inter / union if union > 0 else 0.0


def _harmonic_number(freq: float, f0: float, tol_pct: float) -> int | None:
    if f0 <= 0:
        return None
    k = freq / f0
    kr = int(round(k))
    if kr < 1 or kr > 64:
        return None
    return kr if abs(k - kr) * f0 <= max(tol_pct / 100.0 * freq, 0.75) else None


def infer_f0(members: list[Detection], tol_pct: float) -> tuple[float, list[int]] | None:
    """Best f0 for a set of frequencies, including the missing-fundamental case.

    Candidate fundamentals are every member frequency divided by 1..8; the winner is the one
    explaining the most members, tie-broken by the smallest f0 (a hum at 100 Hz explains
    100/200/300, and 50 Hz explains them too -- prefer the lower only when it also explains
    the whole set).
    """
    if not members:
        return None
    freqs = [m.freq_hz for m in members]
    best: tuple[float, list[int]] | None = None
    for f in freqs:
        for div in range(1, 9):
            f0 = f / div
            if f0 < 5.0:
                continue
            numbers = []
            for fr in freqs:
                kn = _harmonic_number(fr, f0, tol_pct)
                if kn is None:
                    break
                numbers.append(kn)
            else:
                cand = (f0, numbers)
                if best is None or len(numbers) > len(best[1]) or (
                        len(numbers) == len(best[1]) and f0 < best[0]):
                    best = cand
    return best


def group_sources(dets: list[Detection], cfg: Config) -> list[Source]:
    """Cluster *tonal* detections into sources (FR-5.1). Returns all A1/A2 in some source.

    Only A1/A2 participate.  An A4 event's ``freq_hz`` is its band's geometric centre (a label,
    not a measurement), so feeding events in merged 1178 separate impacts into one "source" with
    1178 harmonics all at 61 Hz — which is exactly what the first whole-day run reported.
    """
    pool = [d for d in dets if d.cls in ("A1", "A2")]
    for d in dets:
        if d.cls not in ("A1", "A2"):
            d.source_id, d.source_f0_hz, d.n_harmonics = "", 0.0, 0
    if not pool:
        return []
    sources: list[Source] = []
    while pool:
        seed = pool[0]
        # candidates: harmonic partners of the seed, or time-correlated partners
        group = [seed]
        for other in pool[1:]:
            temporal = _jaccard(_intervals(seed), _intervals(other)) >= cfg.cooccurrence_min_jaccard
            # Co-occurrence is only evidence when switching behaviour exists to correlate.
            # Two steady tones both active 100% of the hour give Jaccard = 1.0, which would
            # merge every unrelated machine in the plant into a single "source".
            steady_pair = seed.duty_pct >= 90.0 and other.duty_pct >= 90.0
            temporal = temporal and not steady_pair
            harmonic = False
            for f0 in (seed.freq_hz, other.freq_hz,
                       seed.freq_hz / 2.0, other.freq_hz / 2.0):
                if f0 > 5.0 and _harmonic_number(other.freq_hz, f0, cfg.harmonic_tol_pct) is not None \
                        and _harmonic_number(seed.freq_hz, f0, cfg.harmonic_tol_pct) is not None:
                    harmonic = True
                    break
            if harmonic or (temporal and abs(other.freq_hz - seed.freq_hz) > cfg.min_freq):
                group.append(other)
            elif temporal and len(group) == 1:
                group.append(other)
        inferred = infer_f0(group, cfg.harmonic_tol_pct)
        f0_est, numbers = inferred if inferred else (seed.freq_hz, [1] * len(group))
        distinct = sorted(set(numbers))
        # A harmonic f0 is only a claim worth printing when the members actually spread over several
        # harmonic numbers.  Three LF tracks near 25 Hz let infer_f0 propose "5.04 Hz", which is
        # arithmetically true and physically meaningless — no machine on site runs at 300 rpm and
        # reports its 5th through 7th partials with no fundamental.  Below a real stack, report the
        # lowest member and the spread, and say the link was temporal.
        harmonic_claim = len(group) > 1 and len(distinct) >= 3 and max(distinct) >= 3
        if harmonic_claim:
            f0, reason = f0_est, "harmonic"
        else:
            f0 = min(m.freq_hz for m in group)
            reason = ("single" if len(group) == 1
                      else "temporal" if len(group) > 1 else "spectral-only")
        src = Source(
            source_id=_source_id(f0), f0_hz=float(f0),
            f0_uncertainty_hz=float(np.std([m.freq_hz / max(k, 1)
                                           for m, k in zip(group, numbers)])
                                    if harmonic_claim else
                                    max(0.5, (max(m.freq_hz for m in group) - f0) / 2.0)),
            members=group, n_harmonics=len(distinct), harmonic_numbers=distinct,
            t_first_s=min(m.t_start_s for m in group), t_last_s=max(m.t_end_s for m in group),
            active_s=_length(_union([iv for m in group for iv in _intervals(m)])),
            n_episodes=sum(m.n_episodes for m in group),
            total_ttn_db=float(np.mean([m.tone_noise_db for m in group])),
            level_dbfs=float(np.max([m.peak_dbfs for m in group])),
            drift_class=_dominant([m.drift_class for m in group]),
            link_reason=reason,
            )
        src.duty_pct = 100.0 * src.active_s / max(seed.extra.get("span_s", src.t_last_s - src.t_first_s), 1e-9)
        sources.append(src)
        for m in group:
            m.source_id = src.source_id
            m.source_f0_hz = src.f0_hz
            m.n_harmonics = src.n_harmonics
        pool = [m for m in pool if id(m) not in {id(g) for g in group}]
    sources.sort(key=lambda s: (-s.n_harmonics, s.f0_hz))
    return sources


def annotate_neighbours(dets: list[Detection]) -> None:
    """FR-4.7: every detection records its nearest other frequency and the ratio.

    This is how a human checks a call in one glance: 500 Hz with neighbour 1500 Hz at ratio
    3.00 is one machine's second harmonic, not two anomalies.  Ratio is written the same way a
    spectrogram is read -- higher over lower -- so 0.33 and 3.00 never mean different things.
    """
    if len(dets) < 2:
        for d in dets:
            d.nn_freq_hz, d.nn_ratio = 0.0, 0.0
            d.extra["nearest"] = None
        return
    freqs = np.array([d.freq_hz for d in dets], dtype=np.float64)
    for i, d in enumerate(dets):
        others = np.delete(np.abs(freqs - d.freq_hz), i)
        j = int(np.argmin(others)) if others.size else -1
        if j < 0:
            d.nn_freq_hz, d.nn_ratio = 0.0, 0.0
            continue
        nj = j if j < i else j + 1                     # undo the delete() index shift
        lo, hi = sorted((d.freq_hz, dets[nj].freq_hz))
        d.nn_freq_hz, d.nn_ratio = float(dets[nj].freq_hz), float(hi / lo if lo > 0 else 0.0)
        d.extra["nearest"] = {"freq_hz": d.nn_freq_hz, "ratio": round(d.nn_ratio, 3),
                              "same_file": dets[nj].file == d.file}


def _dominant(values: list[str]) -> str:
    uniq, counts = np.unique(np.asarray(values, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(counts))]) if len(uniq) else ""


def _source_id(f0: float) -> str:
    """Stable across runs on the same file: hash of f0 rounded to 0.1 Hz (FR-5.3)."""
    return "src-" + hashlib.sha1(f"{round(f0, 1):.1f}".encode()).hexdigest()[:10]
