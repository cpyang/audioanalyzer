"""Turn "the noise that is always here" into candidate ``known.toml`` entries (FR-7.3).

The question this answers is Conrad's: feed it a recording (or a day of them) and get back the
sounds that are *essentially always present*, so they stop being findings and become site
fingerprint.  Two criteria, because one is not enough:

* **lines** — narrowband tracks.  Reported with their measured duty cycle, which is the honest
  "all the time" number *inside* an hour.
* **floor ridges** — broadband bumps in the median spectrum that stand above a one-octave-smoothed
  trend of the same spectrum.  These are what a spectrogram shows as a permanent band (the LF
  rumble shelf, a transformer whine that is never phase-steady enough to be a tone).  On the
  reference day this matters: **there are zero A1 steady tones in 24 hours**, so a duty-only
  seeder would correctly report "nothing always present" and be useless.

What this deliberately does NOT do:

* **It never infers a fundamental it did not measure.** ``sources.infer_f0`` finds the f0 that best
  explains a set of frequencies, and on three 25 Hz tracks it invented "5.04 Hz" — defensible
  arithmetic, false machine.  A generated library would propagate that into every later report, so
  here every proposed ``f0_hz`` is an observed line and ``harmonics`` counts only harmonics that
  were themselves observed.
* **It writes candidates, not the library.** ``known.toml`` is a hand-edited file, and promoting an
  entry is a claim about the physical site.  So output goes to ``known.candidates.toml`` and
  ``promote()`` is the human-gated step.  (Labelling is annotation only — no code path filters on
  ``known_name`` — so a wrong candidate degrades a label, never hides a finding.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .spectral import aggregate, candidate_tones, local_maxima
from .detectors.tonal import detect_tonal


@dataclass
class SeedCandidate:
    freq_hz: float
    kind: str                                  # "line" | "ridge"
    n_files: int
    files_total: int
    level_dbfs: float                          # median absolute level (ridge: at the peak)
    over_floor_db: float                       # line: tone-to-noise; ridge: above own trend
    duty_pct: float | None = None              # lines only; None for ridges
    coherence: float | None = None             # lines only
    band_hz: tuple[float, float] | None = None  # ridges: the +-octave extent actually measured
    harmonics: list[int] = field(default_factory=lambda: [1])
    seen_in: list[str] = field(default_factory=list)

    @property
    def file_frac(self) -> float:
        return self.n_files / max(1, self.files_total)

    @property
    def always_score(self) -> float:
        """How much this looks like permanent site background rather than an event.

        A line in every file but on for 5 % of each hour is a recurring event, not background, so
        duty discounts it; a ridge needs no discount because it is measured as floor structure.
        """
        d = 1.0 if self.duty_pct is None else max(0.0, min(1.0, self.duty_pct / 100.0))
        return self.file_frac * (0.35 + 0.65 * d)

    def name(self) -> str:
        if self.kind == "ridge":
            lo, hi = self.band_hz or (self.freq_hz, self.freq_hz)
            return f"auto: floor ridge {lo:.0f}-{hi:.0f} Hz"
        return f"auto: {self.freq_hz:.1f} Hz line"

    def note(self) -> str:
        ev = [f"{self.n_files}/{self.files_total} files"]
        if self.duty_pct is not None:
            ev.append(f"duty {self.duty_pct:.1f}%")
        ev.append(f"{self.over_floor_db:.1f} dB over local floor")
        ev.append(f"peak {self.level_dbfs:.1f} dBFS")
        if self.coherence is not None:
            ev.append(f"coherence {self.coherence:.2f}")
        if len(self.harmonics) > 1:
            ev.append("observed harmonics " + "/".join(str(k) for k in self.harmonics))
        return ("AUTO-CANDIDATE - verify by listening before promoting; " + ", ".join(ev))


def _ridge_peaks(agg, eng, cfg: Config, ridge_db: float, octaves: float) -> list[dict]:
    """Contiguous bumps of the median spectrum above a locally fitted power-law trend.

    Three guards, each learned from what the first version produced on the reference day (16
    "always-present sources", all in 6242-7203 Hz, all at -97 dBFS - one anti-alias knee, exploded
    into one entry per noise wiggle):

    * **Smooth first.** In a flat, low-SNR region above the anti-alias corner every bin is a strict
      local maximum of the raw spectrum.  A ~1/12-octave median filter leaves genuine broad
      structure (the 125 Hz bump is ~0.3 octave wide) and removes single-bin wiggles.
    * **Cluster into features.** Neighbouring positive bins are one thing happening, not many.
    * **Refuse anything within 1.5 octaves of an analysis-band edge.** The fit window is truncated
      there, so the fitted line under-reads the level on the flat side of a knee and manufactures a
      "feature" out of the filter response.  That is a property of the equipment, not of the site,
      and it must not become a library entry.
    """
    m = agg.percentile(0.5)
    if not np.isfinite(m).any():
        return []
    df = float(eng.freqs[1] - eng.freqs[0])
    valid = np.isfinite(m) & (eng.freqs >= cfg.min_freq)
    if not valid.any():
        return []
    first = int(np.argmax(valid))
    # A 3-bin median removes isolated single-bin wiggles and nothing else.  A wider kernel looked
    # tempting (the HF junk below is wiggle-driven) but bins are LINEAR here, so a fixed 10-bin
    # kernel is 0.64 octave of smoothing at 125 Hz and erased the one genuine feature.
    from scipy.ndimage import median_filter
    ms = median_filter(np.nan_to_num(m, nan=float(np.nanmedian(m))), size=3, mode="nearest")
    ms[:first] = np.nan
    fit_span_oct = 0.5                    # fit points must reach this far on BOTH sides
    peaks = []
    for b in local_maxima(ms, min_bin=first):
        f = float(eng.freqs[b])
        if f < cfg.min_freq or not np.isfinite(ms[b]):
            continue
        half = max(3, int(round(f * (2.0 ** octaves - 1.0) / df / 2.0)))
        lo, hi = max(first, b - half), min(len(ms), b + half + 1)
        core_lo = max(0, b - int(0.1 * f / df))
        core_hi = min(len(ms), b + int(0.1 * f / df) + 1)
        left = np.array([j for j in range(lo, core_lo) if np.isfinite(ms[j])])
        right = np.array([j for j in range(core_hi, hi) if np.isfinite(ms[j])])
        if left.size < 4 or right.size < 4:
            continue                      # one-sided fit near the band edge invents ridges
        # A straight line through a knee under-reads the level on the flat side, which is how the
        # anti-alias corner above ~6 kHz became 16 "always-present sources" at -97 dBFS.  Requiring
        # the fit to span >= 0.5 octave on both sides rejects the truncated cases (6.5 kHz has only
        # 0.22 octave of spectrum to its right) without disqualifying the upper half of the band.
        if (math.log2(f / max(eng.freqs[first], eng.freqs[left[0]])) < fit_span_oct
                or math.log2(eng.freqs[right[-1]] / f) < fit_span_oct):
            continue
        j = np.concatenate([left, right])
        # Trend = a LOCAL LINEAR FIT in log f, not the window median.  The corpus floor tilts
        # ~5.5 dB/octave, so a median taken across a knee sits below the centre and mis-measures
        # the bump: on the reference hour the 125 Hz feature read +5.1 dB by median and +7.3 dB by
        # fit, with the flanking bins (101.6, 148.4 Hz) at -2.1 and -1.1.  A power-law floor is the
        # honest null hypothesis for "no feature here".
        x = np.log2(eng.freqs[j])
        coef, *_ = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, ms[j], rcond=None)
        resid = ms - (coef[0] + coef[1] * np.log2(np.maximum(eng.freqs, 1e-6)))
        above = float(resid[b])
        if above < ridge_db:
            continue
        # The bump must exist in the RESIDUAL, not merely in the spectrum: a shelf or a knee is
        # monotone through its own centre, so its centre is not a residual maximum.
        w = max(1, int(0.1 * f / df))
        flank = [resid[k] for k in (b - w, b + w, b - 2 * w, b + 2 * w)
                 if 0 <= k < len(resid) and np.isfinite(resid[k])]
        if not flank or max(flank) >= above - 1.0:
            continue                      # no crest: this is a slope, not a feature
        k = b
        while k - 1 >= first and resid[k - 1] >= ridge_db / 2.0:
            k -= 1
        e = b
        while e + 1 < len(resid) and np.isfinite(resid[e + 1]) and resid[e + 1] >= ridge_db / 2.0:
            e += 1
        wide = math.log2(max(eng.freqs[e], 1e-6) / max(eng.freqs[k], 1e-6))
        if wide < cfg.seed_ridge_min_octaves:
            continue                      # narrower than real structure: wiggle, not feature
        peaks.append({"bin": int(b), "freq_hz": f, "over_floor_db": above,
                      "level_dbfs": float(ms[b]),
                      "band_hz": (float(eng.freqs[k]), float(eng.freqs[e])),
                      "width_oct": round(wide, 3)})
    # merge overlapping features (same bump found from neighbouring bins)
    peaks.sort(key=lambda p: p["band_hz"][0])
    merged: list[dict] = []
    for p in peaks:
        if merged and merged[-1]["band_hz"][1] >= p["band_hz"][0]:
            q = merged[-1]
            if p["over_floor_db"] > q["over_floor_db"]:
                q.update(freq_hz=p["freq_hz"], over_floor_db=p["over_floor_db"],
                         level_dbfs=p["level_dbfs"], bin=p["bin"])
            q["band_hz"] = (min(q["band_hz"][0], p["band_hz"][0]),
                            max(q["band_hz"][1], p["band_hz"][1]))
            continue
        merged.append(p)
    return merged


def seed_probe(path: str | Path, cfg: Config) -> dict:
    """One pass over one file: the tonal tracks and the floor ridges it contains."""
    agg, eng = aggregate(path, cfg)
    cands = candidate_tones(agg, cfg, eng)
    dets = detect_tonal(path, cfg, agg, eng, cands)
    lines = [{"freq_hz": float(d.freq_hz), "level_dbfs": float(d.level_dbfs),
              "over_floor_db": float(d.tone_noise_db), "duty_pct": float(d.duty_pct),
              "coherence": float(d.coherence)}
             for d in dets if d.cls in ("A1", "A2")]
    cfg2 = cfg                                       # ridges need no event detection
    return {"path": str(path), "lines": lines,
            "ridges": _ridge_peaks(agg, eng, cfg2, cfg.seed_ridge_db, cfg.seed_ridge_octaves)}


def _group(items: list[dict], tol_hz: float, tol_pct: float) -> list[list[dict]]:
    """Cluster by frequency, sorted, linear in n (same shape as recurrence's grouping)."""
    groups: list[list[dict]] = []
    for it in sorted(items, key=lambda x: x["freq_hz"]):
        placed = False
        for g in groups:
            centre = float(np.median([x["freq_hz"] for x in g]))
            if abs(it["freq_hz"] - centre) <= max(tol_hz, tol_pct / 100.0 * centre):
                g.append(it)
                placed = True
                break
        if not placed:
            groups.append([it])
    return groups


def seed_candidates(paths: list[str | Path], cfg: Config, *, min_file_frac: float = 0.6,
                    ridge_db: float | None = None, octaves: float | None = None,
                    progress=None, stats: dict | None = None) -> list[SeedCandidate]:
    """Collect the always-present sounds across a selection.  ~1 stage-1+2 pass per file.

    `stats`, if given a dict, is filled with what was examined and why each part was rejected, so
    "no candidates" can be told apart from a broken seeder.
    """
    cfg = Config(**{**cfg.to_dict()})
    cfg.enable_events = False                    # background fingerprint, not incidents
    if ridge_db is not None:
        cfg.seed_ridge_db = ridge_db
    if octaves is not None:
        cfg.seed_ridge_octaves = octaves
    probes = []
    for i, p in enumerate(paths):
        if progress:
            progress(i + 1, len(paths), str(p))
        probes.append(seed_probe(p, cfg))
    total = len(probes)
    if total == 0:
        return []

    out: list[SeedCandidate] = []
    if stats is not None:
        stats.update(files_total=total, examined={"line": 0, "ridge": 0},
                     best_file_frac={"line": 0.0, "ridge": 0.0},
                     best_duty={"line": 0.0, "ridge": None}, per_file_counts={
                         "lines": [len(pr["lines"]) for pr in probes],
                         "ridges": [len(pr["ridges"]) for pr in probes]})
    for kind in ("line", "ridge"):
        pool: list[dict] = []
        for pr in probes:
            for it in pr[kind + "s"]:
                pool.append({**it, "path": pr["path"]})
        for g in _group(pool, cfg.recurrence_tol_hz * 2, cfg.recurrence_tol_pct):
            files = {x["path"] for x in g}
            if stats is not None:
                s = stats["examined"]
                s[kind] += 1
                frac = len(files) / total
                if frac > stats["best_file_frac"][kind]:
                    stats["best_file_frac"][kind] = frac
                if kind == "line":
                    d = float(np.median([x["duty_pct"] for x in g])) / 100.0
                    if d > stats["best_duty"]["line"]:
                        stats["best_duty"]["line"] = d
            if len(files) / total < min_file_frac:
                continue                    # not "always here": it is here sometimes
            med = lambda key, dflt=None: float(np.median([x[key] for x in g]))      # noqa: E731
            c = SeedCandidate(
                freq_hz=med("freq_hz"), kind=kind, n_files=len(files), files_total=total,
                level_dbfs=med("level_dbfs"), over_floor_db=med("over_floor_db"),
                duty_pct=(None if kind == "ridge" else round(med("duty_pct"), 1)),
                coherence=(None if kind == "ridge" else round(med("coherence"), 3)),
                band_hz=(tuple(round(float(v), 1) for v in
                               (min(x["band_hz"][0] for x in g),
                                max(x["band_hz"][1] for x in g)))
                         if kind == "ridge" else None),
                seen_in=sorted(Path(f).name for f in files))
            out.append(c)

    # Observed harmonics only: for each candidate, count the *other* candidates that sit at k*f
    # within tolerance.  Nothing is invented; a missing fundamental stays missing and is reported
    # as such in the entry's note.
    freqs = [(c, c.freq_hz) for c in out]
    for c, f0 in freqs:
        ks = []
        for other, fx in freqs:
            if other is c or f0 <= 0:
                continue
            k = fx / f0
            if 2 <= k <= 12 and abs(k - round(k)) * f0 <= max(2.0, cfg.recurrence_tol_pct / 100 * fx):
                ks.append(int(round(k)))
        c.harmonics = [1] + sorted(set(ks))
    out.sort(key=lambda c: (-round(c.always_score, 4), c.freq_hz))
    return out


def write_candidates(path: str | Path, cands: list[SeedCandidate], *,
                     append: bool = False) -> Path:
    """Write the candidate list to a *candidates* file (never the live library).

    **Rewrites by default.** Two reasons, both from an incident: the first whole-day run emitted 16
    bogus entries (one anti-alias knee, per-bin noise wiggles), the fixed run found nothing and
    returned without writing, and the poisoned file stayed on disk one `--promote` away from the
    live library. Appending also breaks the index contract — `promote` indexes file positions while
    the printed table numbers come from the current run, so `--index 1` would promote the wrong
    entry once any stale block was present. A candidate file should always mean "the last seed run".
    Pass ``append=True`` to accumulate across deliberately different site visits.
    """
    p = Path(path)
    header = ("# Candidate known-noise entries, from the MOST RECENT `audioanalyzer known --seed`.\n"
              "# NOTHING HERE IS CONFIRMED. Review, rename with the physical source, then promote:\n"
              "#   audioanalyzer known --promote known.candidates.toml --index 1,3,5 --file known.toml\n"
              "# A candidate's f0_hz is always an observed line; fundamentals are never inferred.\n"
              "# Re-running --seed replaces this file, so the printed indices always match it.\n")
    if append and p.exists():
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\n# ---- appended by a later --seed run ----\n")
    else:
        p.write_text(header, encoding="utf-8")
    with p.open("a", encoding="utf-8") as fh:
        for c in cands:
            fh.write(f"\n[[source]]\nname = {json.dumps(c.name())}\n"
                     f"f0_hz = {c.freq_hz:.3f}\nharmonics = {len(c.harmonics)}\n"
                     f"tol_pct = 0.5\nkind = {json.dumps(c.kind)}\n"
                     f"note = {json.dumps(c.note())}\n")
    return p


def promote(src: str | Path, dst: str | Path, indices: list[int],
            renames: dict[int, str] | None = None) -> list[str]:
    """Move chosen candidates into the live library, optionally renamed to the physical source.

    1-based indices, as printed by the seeding table: a human picks, this does not guess.
    """
    from .known import load_known, upsert_known
    existing = {round(k.f0_hz, 2) for k in load_known(dst)}
    doc = load_known(src)
    done = []
    for i in indices:
        if not 1 <= i <= len(doc):
            raise IndexError(f"candidate index {i} out of range (1..{len(doc)})")
        c = doc[i - 1]
        if round(c.f0_hz, 2) in existing:
            continue
        name = (renames or {}).get(i) or c.name
        upsert_known(dst, name, c.f0_hz, c.tol_pct, c.harmonics, c.note)
        done.append(name)
    return done
