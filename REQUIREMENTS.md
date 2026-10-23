# Audio Anomaly Analyser — Requirements

Status: v0.2 — M0 complete (venv + ingest + A6 verified); M1 plan awaiting approval
Date: 2026-10-24
Owner: (unassigned)
Reference corpus: `/home/cpyang/Audio/20261023T070002.flac`
Related: `@file:20261023T070002.flac`

---

## 1. Purpose

Long-duration unattended audio recordings (1 h FLAC chunks) are being captured for
site surveys. Today there is no way to answer, in minutes, the questions that matter:

* *Which* frequencies in this recording are abnormal (tones, hums, whines, clicks)?
* *When* did they occur, and for how long — as file offsets **and** wall-clock time?
* Do they repeat? At what period? (→ machine cycles, bearings, PSU switching, PMS polling)
* Which physical source do they belong to? (→ group harmonics + co-occurring tones into a "source")

The tool is a **scan-and-report** utility, not a streamer and not a classifier trained on
labelled data. Output is a ranked, machine-readable list of detections plus a human report
that an engineer can eyeball to decide where to walk with a second microphone.

Non-goal restated: we are finding **spectral anomalies against a learned/adaptive baseline**,
not "loud sounds". See §3 for why that distinction is load-bearing on this corpus.

---

## 2. Input data characteristics (measured, not assumed)

Probed with `ffprobe` / `ffmpeg -af astats` and a log-scale `showspectrumpic` on the
reference file; values below are facts about the corpus, and the design depends on them.

### 2.0 Corpus (measured 2026-10-24)

One site, one day, 24 hourly chunks — `20261023T000003.flac` … `20261023T230002.flac`,
1.1 GB total, all 16 000 Hz / mono / PCM_16, no missing hour. Chunk durations span
**3573.33 s … 3595.35 s**, i.e. every chunk is slightly short of the 3600 s its filename
implies. See §2.2 — this is the most consequential finding in this section.

| Property | Measured value | Design consequence |
|---|---|---|
| Container / codec | raw FLAC, `sample_fmt s16`, 16-bit | Lossless; decode via `libsndfile` (bundled in `soundfile` wheel), no ffmpeg needed at runtime |
| Sample rate | 16 000 Hz, mono | **Nyquist = 8 000 Hz.** Nothing above 8 kHz exists. Bin grid must be stated in Hz |
| Duration | 3573.333 s (59 min 33 s), 57 173 333 samples | Streaming/blocking mandatory for memory; whole-file arrays are too big |
| Size / bitrate | 45 963 274 B, ~103 kbps | Decode ≈ a few seconds; I/O is not the bottleneck |
| Tags | `Comment: "Processed by SoX"` | Chain is unknown (resampled from where?). Origin rate + anti-alias filter must be recorded as metadata (§ FR-1.4); aliasing risk §11-R3 |
| Filename | `YYYYMMDDTHHMMSS.flac` = capture start (local) | Timestamp mapping is derivable, not stored → FR-2.3 |
| DC offset | −0.01224 (≈ −38 dBFS), persistent | Bin 0 is **not** a tone; must be excluded/hi-passed or flagged as a data-quality item, never reported as an anomaly |
| Peak level | 0.000 dBFS. 375 samples at \|x\| = 1.0 (= -32768; equals ffmpeg's *Abs Peak count*), 464 at the top PCM code (\|x\| ≥ 32767/32768), longest consecutive flat-top run **424 samples = 26.5 ms** | Genuine hard clipping exists but is rare and localised. Clip detection (A6) must use the **top PCM code**, not "1 LSB below 1.0" — that formula sits above the largest positive s16 value and matches nothing. Clipped frames must not masquerade as broadband anomalies |
| Global RMS | −37.44 dBFS | — |
| RMS in 10 s blocks (358 blocks) | min −38.2 / median **−37.9** / p99 −35.8 / max **−23.1** | **Only one block exceeds median+6 dB (the first, t=0).** Broadband energy is essentially constant over the hour |
| Dynamic range (astats) | 96.3 dB | Headroom exists; anomaly detection must work at ~−60 dBFS per-bin levels, not just at loud levels |
| Visual inspection (log-freq spectrogram, linear-freq 0–180 s) | (a) strong broadband floor confined to roughly DC–500 Hz, steady across the whole hour; (b) a faint **continuous horizontal line near 6.8–7.0 kHz**; (c) a **regular comb of short vertical dashes in 5–6.5 kHz** repeating across the full 59 min; (d) isolated broadband transients, densest around 22–29 min and near the start; (e) second half visibly quieter in mid/high bands than the first | (b) and (c) are **hypotheses from an eye-read of a rendered image, not measurements** — the tool's M1 milestone exists to confirm or refute their exact Hz, level, and period. (e) is invisible in RMS, which is the core motivation for §3 |

### 2.1 The key finding

Anything based on " louder than usual" fails on this corpus: total loudness varies by
~0.5 dB for 59 minutes. The interesting events are either **narrow in frequency**
(one bin above a flat floor) or **short in time** (a click inside a 10 s window). Therefore
detection is defined per frequency bin, against a per-bin baseline learned from the same
recording (or from a reference profile), expressed in dB above that baseline.

### 2.2 The recorder drops audio (measured)

Reconstructing each chunk's end from filename-start + measured duration and comparing to the
next chunk's filename-start gives the true listening blackout between chunks:

| Metric | Measured |
|---|---|
| Inter-chunk gaps | min **5.65 s**, median **10.37 s**, max **26.67 s** (after `20261023T070002.flac`) |
| Total audio vs wall clock | 86 132.3 s of audio covering 86 390.3 s of day = **99.70 %**, **258 s (4 min 18 s) unrecorded** |
| Overlaps (double-covered time) | none — gaps are all positive, so no chunk-to-chunk ambiguity |

Consequences, all binding on the design:

* **A non-detection is not evidence of absence.** Every corpus report must carry a coverage
  ledger (FR-2.2/2.4) and any "no events found" conclusion must be qualified by it.
* A sub-second transient has ~0.3 % chance per event of landing in a blackout; an event whose
  *apparent* period is near 3600 s may be an artifact of the gap, not a real cycle (FR-7.2
  must subtract gap windows before period search, else the hour boundary fakes a harmonic).
* Filename-start times are exact on the hour (±1 s jitter: :02/:03), so the mapping
  offset→wall-clock within a chunk is trustworthy; it is the *between*-chunk regions that are
  blind. 26 s blind on a 59.9 min chunk ≈ 0.74 % duty loss, per hour.
* Worth a separate ticket on the capture side: ~4.3 min/day of missing audio points to
  underruns/SD stalls, not to a rounding convention. This tool cannot fix it, only reveal it.

---

## 3. Definitions — what counts as an "abnormal frequency"

Six detector classes. Each detection carries: `freq_hz`, `bandwidth_hz`, `t_start`, `t_end`,
`duration_s`, `peak_dbfs`, `prominence_db` (above per-bin baseline), `class`, `confidence`.

| ID | Class | Definition | Primary signal |
|---|---|---|---|
| A1 | **Steady tone** | Local spectral peak confined to ≤3 bins, prominence ≥ *P* dB, present in ≥ *D*% of frames over ≥ *T* s | per-bin median baseline |
| A2 | **Intermittent / emerging tone** | Same spectral signature as A1 but appearing in episodes (duty cycle < *D*%), optionally with frequency or level **trend** (drift, ramp, step) | baseline + episode segmentation |
| A3 | **Impulse train / comb** | Repeating broadband or band-limited transients at a stable rate; report **period** (ms and Hz) and jitter | onset series → autocorrelation |
| A4 | **Transient event** | One-off energy burst: ≥ *K* dB above the local (causal-free) noise floor in a window < 1 s. Sub-typed *broadband* (click/pop) vs *band-limited* (whine onset, strike) | frame-wise robust z-score |
| A5 | **Harmonic family / SOURCE** | A *set* of A1/A2 detections at `f0, 2f0, 3f0…` (tolerance ±*tol*%) or a set time-correlated (on/off together). Collapsed into one entity: the **source**, with estimated f0, harmonic count, and the fundamental's stability | post-clustering |
| A6 | **Data-quality anomaly** | Not sound, but invalidates interpretation: DC offset, clipping frames, digital silence, sample-rate/duration mismatch, NaN/Inf, missing-file gaps in a corpus | decode-time checks |

"A normal frequency" = one whose long-run distribution the baseline profile describes
(fan/room noise floor, HVAC broadband). "Abnormal" = deviation from it. Nothing is claimed
about *which physical object* produced a tone except via the optional known-source library
(FR-7.2), which is a user-maintained mapping, not inference.

---

## 4. Scope

**In (v1):** single-file and multi-file (glob/dir) scan of FLAC/WAV; detectors A1–A6;
source grouping; baseline profile + compare; CSV/JSON output; one HTML report with embedded
spectrogram crops per detection; CLI; TOML config; offline, CPU-only.

**Out of v1 (documented, deferred):** streaming/real-time capture; ML classification of
sources (SED models); pitch/beat/MIR features; VAD/speech; stereo/phase analysis; auto gain
control; dashboards or DB backends; GUI.

---

## 5. Functional requirements

### FR-1 Ingest & validation
* 1.1 Read `.flac` (and `.wav`/`.ogg` as `libsndfile` allows) via `soundfile`; support mono
  and multi-channel (analyse each channel independently, then merge, tagging `channel`).
* 1.2 Never load the whole file as one float array: decode in blocks (default 30 s,
  configurable) with the hop/window math preserved across block boundaries.
* 1.3 Emit a `MediaInfo` record per file: path, size, mtime, codec, sample rate, channels,
  bits, exact duration, global peak/RMS/DC, clipped-sample count, NaN/Inf count.
* 1.4 Capture and preserve the SoX/ffmpeg provenance tags when present.
* 1.5 A1–A6 must tolerate: files shorter than one analysis window, all-silence files
  (report "no content", exit 0), and >24 h concatenations.

### FR-2 Timebase
* 2.1 Every detection has `t_offset_s` (from file start, ms precision) and ISO-8601
  `t_wall` derived from the filename timestamp (`YYYYMMDDTHHMMSS`, local TZ, override flag
  `--tz`) with a documented offset convention.
* 2.2 Multi-file corpus: sort by start time, detect and report gaps/overlaps between chunks
  (this is how "the event at 07:41:12" is located across chunk boundaries). Measured on the
  2026-10-23 day: gaps 5.65–26.67 s, 258 s total (§2.2).
* 2.3 An event that straddles a chunk boundary must be reported **once** in corpus mode
  (dedup by frequency + wall-clock overlap), not twice.
* 2.4 **Coverage ledger:** every corpus run emits, per hour and in total, the recorded vs
  elapsed seconds and the list of blackout intervals. Any summary line claiming "no anomalies
  in 03:00–04:00" must be accompanied by that hour's coverage figure.

### FR-3 Spectral engine
* 3.1 STFT, Hann window, `nperseg` default 2048 (**7.8125 Hz/bin** at 16 kHz),
  `noverlap` = 3/4 (`hop` = 512 → 32 ms, 31.25 frames/s). All three are config.
* 3.2 Two resolution modes: `--mode standard` (above, time-faithful) and `--mode hr`
  (`nperseg` 16384 → **0.98 Hz/bin**, 1 s window) for precise tone frequency and **drift**
  measurement — the frequency-accuracy figure of merit for source identification.
* 3.3 Magnitude in dBFS (full scale = 1.0), with configurable reference for dBFS-vs-dBSPL
  (a `dbfs_offset` calibration constant; unset ⇒ dBFS, and the report must say so).
* 3.4 Per-frame spectral peak picking with **sub-bin frequency estimation** (quadratic
  interpolation of log-magnitude around the peak) — required to hit AC-2's ±2 Hz.
* 3.5 Noise floor model: per-bin **median** (robust) over the recording, plus per-bin p90;
  computed in a streaming manner (t-digest or fixed histogram), not by holding all frames.
* 3.6 Time-blocked baseline: the floor is re-estimated per `baseline_block_s` (default 300 s)
  so slow environmental change does not hide (or fake) tones.

### FR-4 Detectors
* 4.1 A1/A2: candidate = bin whose p90 exceeds `floor_p50 + min_prominence_db` (default
  **6.0 dB**) in ≥ `min_persistence_pct` (default **50%**) of frames within ≥ `min_tone_s`
  (default **10 s**), and whose energy is confined to ≤3 adjacent bins.
* 4.2 Episode segmentation per frequency track: merge consecutive frames, split after
  `gap_tolerance_s` (default 2 s) of absence; a tone that never splits is "steady", otherwise
  "intermittent" with a duty cycle and an episode count.
* 4.3 Drift: fit frequency vs time per track; report slope (Hz/min) and `r²`; classify
  `stable` (<0.5 Hz/min), `drifting`, or `stepped` (breakpoint fit).
* 4.4 A4: per-frame broadband and per-band robust z-score against the local
  (±`z_window_s`, default 60 s) distribution; threshold default **6σ** *and* ≥ `min_event_db`
  (default 6 dB) absolute rise. Clipped frames are excluded from A4 (FR-1.3 flags them) so
  clipping cannot read as a broadband transient.
* 4.5 A3: collect A4 onset times in a band, then autocorrelate the onset impulse series over
  0.5 s–3600 s; report best period, its confidence (peak/mean ratio), and jitter (σ of period).
* 4.6 A6: DC offset > −60 dBFS, any clipped sample, >5 s of digital silence, sample-rate
  change, duration/filename mismatch, non-finite samples.
* 4.7 **FP control:** harmonics of *detected* anomalies are reported under their source
  (A5), not as independent anomalies; every detection lists its nearest neighbour frequency
  and the ratio, so a human can see *why* it was called; `--exclude-band Hz:Hz` and
  `--min-freq/--max-freq` suppress known-good bands.

### FR-5 Source grouping
* 5.1 Cluster A1/A2 tracks into sources by (a) harmonic relation to a common f0
  (`harmonic_tol_pct` default 1.0%) and (b) temporal co-occurrence (Jaccard of on-time ≥ 0.6).
* 5.2 Per-source record: id, f0 estimate + uncertainty, member frequencies with prominence,
  harmonic count, first/last seen, total active time, duty cycle, drift class,
  band-span, and a "time fingerprint" for cross-file matching.
* 5.3 Sources must have **stable ids across runs on the same file** (deterministic hash of
  rounded f0 + band) so reports can be diffed after a re-scan.

### FR-6 Baseline profile & compare mode
* 6.1 `profile` command builds a floor model (`baseline.npz`) from one or many reference
  recordings (e.g. a known-quiet night) — the "normal" against which later scans differ.
* 6.2 `compare` reports, per band: dB rise/fall vs profile, and new/disappeared tones.
  A tone present in the file and absent from the profile is the single most valuable output
  for "something started making noise".

### FR-7 Corpus / event-tracking
* 7.1 `scan dir/` or glob → per-file detections plus a merged wall-clock timeline.
* 7.2 **Recurrence analysis** across the corpus: for each source, inter-arrival times of its
  episodes → period, phase-of-day clustering (e.g. "on every day at 07:00 ± 4 min"),
  and day-of-week correlation. This is the "track down the source" payoff.
* 7.3 Known-source library (`known.toml`): name → expected frequency/f0 + tolerance + notes.
  Matches are labelled by name in all outputs; unmatched detections stay "UNKNOWN".
* 7.4 Export a detection CSV/JSON suitable for joining against external event logs
  (PLC/BMS/pump schedules) — the tool must not require those logs, only be joinable.

### FR-8 Outputs
* 8.1 `detections.csv` (flat, one row per detection, stable column order) and
  `detections.jsonl` / `summary.json` (per file: media info, config echo, sources,
  detections, data-quality flags).
* 8.2 `report.html`: self-contained, sortable table (freq, span, time, level, prominence,
  class, source, confidence) + one figure per top-N detection: a dB-colourmap spectrogram
  crop (±30 s, ±1 kHz; y-axis in Hz) and a 1 s waveform zoom, so noise floor vs tone is
  legible. Also one full-recording overview figure (log-freq, matching how a human reads it).
* 8.3 `--stdout` human summary: top-N lines, no HTML needed (for scripts and for chat relays).
* 8.4 Exit codes: 0 clean, 1 usage, 2 unreadable/decode error, 3 detections exceed a
  `--fail-on` severity (makes it usable as a scheduled check).
* 8.5 All outputs record the tool version, the config, and the exact thresholds used
  (a report is not reproducible without them).

### FR-9 CLI (surface, names may change)
```
audioanalyzer scan  FILE...         [-c cfg.toml] [--mode standard|hr] [--min-prominence DB]
                                    [--class tonal|transient|all] [--top N] [--report DIR]
                                    [--csv F] [--json F] [--tz TZ] [--min-freq/--max-freq]
audioanalyzer profile  REF...       -o baseline.npz
audioanalyzer compare  FILE --baseline baseline.npz
audioanalyzer periodic FILE...      [--report DIR]
audioanalyzer known  [--list] [--add NAME F0 TOL]
audioanalyzer version / config --dump
```

---

## 6. Non-functional requirements

| ID | Requirement | Target / basis |
|---|---|---|
| NFR-1 | Language / runtime | Pure Python, ≥3.10. **3.12 chosen** for wheel breadth (host has 3.12 and 3.14; `uv` at `~/.local/bin/uv`) |
| NFR-2 | Dependencies | `numpy`, `scipy`, `soundfile`, `matplotlib` (report only), `tomli`/stdlib TOML. **No `librosa`** (drags numba/llvmlite; no MIR needed). All four verified to have cp312 manylinux wheels (numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2; soundfile 0.14.0 ships `py2.py3-none-manylinux_2_28_x86_64` with bundled libsndfile) |
| NFR-3 | Install | `uv venv --python 3.12` in `/home/cpyang/Audio/.venv`, `uv pip install -e .` — offline-reproducible via `requirements.lock`. No conda, no system-site-packages mutation |
| NFR-4 | Throughput | ≤ 45 s wall and ≤ 1 CPU per 60-min/16 kHz mono file in standard mode; ≤ 4 min in `hr` mode (target: 1000× realtime). Measure with `time`; regression-test it. **Whole 24-chunk day ≤ 12 min standard mode** (the real workload unit). M0 measured: ingest pass = **1.35 s wall, 43 MB RSS** for 3573 s of audio (≈2650× realtime, single pass) |
| NFR-5 | Memory | **< 400 MB peak RSS** per file (full STFT of the reference file would be 111 600 frames × 1025 bins ≈ 460 MB in float32 → must be blocked, per FR-1.2/3.5) |
| NFR-6 | Determinism | Same input + config ⇒ byte-identical CSV. No RNG; sort by (t_start, freq) |
| NFR-7 | Privacy / network | Runs fully offline after install; never uploads audio |
| NFR-8 | Tests | Unit tests on synthetic signals (known tone/click/comb injected ⇒ detector must return the ground truth) + a golden test on the real file. `pytest`, coverage on detector math |
| NFR-9 | Logging / UX | `--progress` block-level percentage; `--quiet`; warnings to stderr so stdout stays pipeable |
| NFR-10 | Extensibility | Detectors behind one interface (`class Detector: feed(frames) → events`) so a future ML detector plugs in without touching ingest |
| NFR-11 | Portability | Must also run on the Pi fleet (`armhf/aarch64`) — no x86-only deps; `soundfile` publishes aarch64 wheels. No ffmpeg at runtime (ffmpeg only for probes/report experiments) |

---

## 7. Acceptance criteria (measurable; reference file unless stated)

| ID | Criterion |
|---|---|
| AC-1 | `scan 20261023T070002.flac` completes within NFR-4/5 and emits CSV+JSON+HTML for all 57.17 M samples with **zero** detections in bin 0 / DC, and reports the measured DC offset (−38.25 dBFS) and the clipping evidence (375 samples at \|x\|=1.0, 464 at the top PCM code, 26.5 ms flat-top run) as A6 items — **implemented and verified in M0** |
| AC-2 | The 6.8–7.0 kHz feature seen in §2 is detected as A1/A2 with frequency accuracy **±2 Hz** in `hr` mode, and its continuity/duty cycle is reported. (This is the M1 confirmation of a visual hypothesis — if it turns out to be a rendering artifact, AC-2 is rewritten from measured truth, not dropped silently) |
| AC-3 | The 5–6.5 kHz comb is characterised by A3 with a period accurate to **±5%** against the true inter-onset interval, and the estimate is reproducible across two independent chunks of the file |
| AC-4 | ≥90% of the transients visible in the 22–29 min window are recovered by A4 with time accuracy ±0.2 s, and a hand-made corpus with 10 injected clicks (known times, −20 dBFS) yields ≥9/10 detections at **≤2 false positives per hour** |
| AC-5 | A synthetic 10-min test file containing tones at 50/100/150/1000 Hz at −45 dBFS over −60 dBFS/Hz noise ⇒ exactly **one** source (f0 = 50 Hz ±0.5 Hz, 4 harmonics), no spurious extra sources |
| AC-6 | Two files of the same site, one with an added 120 Hz tone: `compare` lists it as "NEW tone" and nothing else, i.e. the baseline path works |
| AC-7 | Every detection's wall-clock time is correct to ±1 s against the filename timestamp across a 3-chunk corpus including a mid-corpus gap (FR-2.2) |
| AC-8 | `report.html` opens standalone; hovering/reading any top-20 detection shows its spectrogram crop where the anomaly is visually identifiable by a human who has not seen the code |
| AC-9 | Threshold sweep `--min-prominence {3,6,10}` changes detection counts monotonically and is logged, so FP/FN trade-off is tunable in the field without code edits |

---

## 8. Deliverables & layout

```
/home/cpyang/Audio/
  REQUIREMENTS.md            this document
  .venv/                     uv-managed, python 3.12  (created in M0)
  pyproject.toml             metadata + console-script, optional [dev] extras
  requirements.lock          pinned exact versions for reproducible offline installs
  README.md                  10-line quick start + 1 worked example
  audioanalyzer/
    __init__.py  __main__.py cli.py config.py
    ingest.py       block decoder, MediaInfo, data-quality checks (A6)
    spectral.py     STFT blocks, dB, peak pick + sub-bin freq, streaming floor model
    baseline.py     per-bin p50/p90, time-blocked, save/load .npz profile
    detectors/      tonal.py  episodes.py  transient.py  comb.py  sources.py
    corpus.py       multi-file timeline, dedup, recurrence/periodicity, known.toml
    report.py       matplotlib figures (Agg, no GUI) + HTML assembly
    synth.py        test-signal generator (tones/clicks/combs) for AC-4/5
  tests/            test_spectral.py test_detectors.py test_timebase.py test_golden.py
  out/              per-run artifacts (git-ignored)
```

---

## 9. Milestones

* **M0 (30 min) — DONE 2026-10-24.** venv + skeleton: `uv venv --python 3.12` (CPython 3.12.13),
  deps `numpy 2.5.4 / scipy 1.18.1 / soundfile 0.14.0 (libsndfile 1.2.2) / pytest 9.1.1`,
  `uv pip install -e .`, `audioanalyzer version`, and `scan` emitting full `MediaInfo` + A6
  flags for the reference file. Gate met: FLAC decodes with **no ffmpeg at runtime**;
  measured **1.35 s / 43 MB RSS** for 3573 s of audio. `MediaInfo` cross-checked against
  ffprobe/ffmpeg: duration 3573.333312 s ✓, 57 173 333 frames ✓, RMS −37.445 vs −37.444457 ✓,
  DC −0.012237 vs −0.012238 ✓, at-max sample count 375 vs ffmpeg *Abs Peak count* 375 ✓.
  9 ingest tests green. Two bugs found and fixed by those tests: an invalid
  `SoundFile.extrainfo` reference (→ `extra_info`, parsed for Vorbis comments) and a clip
  threshold of `1-2**-16` that sat above the top s16 code and therefore counted nothing
  (→ `1-2**(1-bits)`; see §2 and R6).
* **M1 (0.5 day)** — ingest + spectral + A6 + **A1/A2 tonal** + CSV. **Pays off immediately:**
  confirms/refutes AC-2 with real numbers and gives the first ranked anomaly list.
* **M2 (0.5 day)** — A5 source grouping + `hr` mode + drift + HTML report with crops (AC-5, AC-8).
* **M3 (0.5 day)** — A4 transients + A3 comb/period + `profile`/`compare` (AC-3, AC-4, AC-6).
* **M4 (0.5 day)** — corpus/wall-clock merge, recurrence analysis, `known.toml`, threshold sweep
  + FP tuning against a labelled 2 h of recordings (AC-7, AC-9), README.
* **M5 (later)** — cross-night trending, optional ML classifier, capture-side integration.

---

## 10. Decisions

Resolved 2026-10-24 (owner: Conrad):

1. **Emphasis — tonal-first (A1/A2/A5).** Matches §2.1: with energy flat to ~0.5 dB over the
   hour, only spectral structure carries information. Transients (A4/A3) land in M3.
2. **Wall-clock — filename = capture start in host-local time (CST, UTC+08:00).**
   Implemented as `tz="Asia/Taipei"` default in `config.py`, overridable with `--tz`;
   verified in `tests/test_ingest.py::test_filename_to_wall_clock`.
3. **False positives — precision-first, `--recall` opt-in.** `Config` holds the strict
   defaults (≥6 dB prominence, ≥10 s persistence); `recall_preset()` relaxes them
   (3 dB / 2 s / 4σ) when explicitly requested.
4. **Corpus shape — measured, no longer open:** 24 hourly chunks = 1 site × 24 h × 1.1 GB,
   continuous except the §2.2 blackouts. So M4 recurrence work is scoped now (one day is
   enough to test phase-of-day and hourly-period artifacts), and the throughput unit of
   work is a whole day, not a file.

Still open (do not block M1 on these):

5. **Calibration** — mic/chain sensitivity unknown ⇒ everything stays in **dBFS** and the
   report labels it as such. `Config.dbfs_offset` exists for when a calibrator reading
   arrives; until then no dB(SPL) claim is made anywhere.
6. **Multi-mic** — current corpus is 100 % mono (§2.0); if stereo ever appears, FR-1.1
   already accepts it but localisation is out of scope (§4).
7. **Named sources** — nothing supplied yet to seed `known.toml`; M4 ships the mechanism
   with an empty file.

---

## 11. Risks & mitigations

| ID | Risk | Mitigation |
|---|---|---|
| R1 | An "anomaly" is just the floor's own variance (FFT leakage from the loud DC–500 Hz band leaking upward and faking high-band tones — Hann sidelobes are −31 dB) | Window selection + leakage estimate in the baseline model; require `prominence > floor_variance + margin`; test with the real floor, not white noise |
| R2 | Mains-hum harmonics (50/60 Hz + comb) dominate the low band and bury real findings | A5 harmonic grouping, per-band suppression, and `known.toml` naming so hum is *stated*, not repeated 40× |
| R3 | 16 kHz decimation without a proper anti-alias filter means sub-Nyquist "tones" could be aliases of >8 kHz switching noise (SoX tag ⇒ chain unknown) | Flag in report; `hr` mode + drift signature (aliases drift with playback rate, true tones do not); recommend recording at 48 kHz for confirmation sweeps |
| R4 | A steady tone is actually a property of the recorder (self-noise, PLL spur) not the site | Baseline profile from a known-quiet/"dummy" recording ⇒ recorder artefacts disappear from `compare` output |
| R5 | 60-min files × many days ⇒ runtime and clutter | Blocked streaming (NFR-5), `--top`, per-run `out/<ts>/` dirs, corpus manifest cache keyed by (path, mtime, size) |
| R6 | Clipping at full scale (464 samples at the top PCM code, one 26.5 ms flat-top run measured) creates broadband spikes read as A4 events | A4 excludes clipped frames; A6 reports them with the run length (FR-4.4) |
| R7 | Threshold defaults tuned on one file don't transfer to a noisier site | All thresholds in config, sweepable (AC-9); `profile` relearns the floor per site |
| R8 | Time-axis misattribution ⇒ engineer walks to the wrong place / wrong shift | AC-7 + explicit `--tz`, chunk-gap detection, both-offset-and-wall-clock in every row |
| R9 | Python 3.14 chosen by accident ⇒ wheel gaps (e.g. scipy/numpy cp314 availability is newer) | Pin `--python 3.12` in M0 and in `pyproject.toml` (`requires-python = ">=3.10,<3.13"` until 3.13 wheels are proven here) |
| R10 | **Capture blackout (measured §2.2):** 258 s/day, up to 26.7 s in one gap, so quiet or brief sources can be missed entirely and an "all clear" report can be wrong | Coverage ledger in every report (FR-2.4); gap-aware period search (FR-7.2); severity wording never says "no anomalies", only "none detected in the 99.7 % covered". Raise a capture-side ticket for the underruns |

---

## 12. Appendix — probe evidence (reproducible)

```bash
export PATH=/home/cpyang/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin:$PATH
cd /home/cpyang/Audio

ffprobe -v error -show_format -show_streams -of json 20261023T070002.flac
#  flac, s16, 16000 Hz, 1 ch, duration 3573.333312, size 45963274, tag Comment="Processed by SoX"

ffmpeg -hide_banner -nostats -i 20261023T070002.flac -ac 1 -af astats -f null - 2>&1 \
  | grep -E "DC offset|Peak level|RMS level|Flat factor|Dynamic range|Peak count|Number of samples"
#  DC offset -0.012238 | Peak 0.000265 dB | RMS -37.444 dB | Flat 23.44 | DR 96.33 | Peak count 401 | 57173333 samples
#  Note: ffmpeg's *Peak count* (401) counts one polarity of its own peak value; *Abs Peak count*
#  (375) is the number of samples at |x| = 1.0, and that is the figure our A6 reproduces.
#  Counting "at or beyond the top PCM code" gives 464. See §2 for why 1-2**-16 is the wrong formula.

# Corpus coverage / blackout ledger (the §2.2 finding) — exact numbers, reproducible
.venv/bin/python -c "
import glob,re,statistics,soundfile as sf
from datetime import datetime,timedelta
R=[]
for p in sorted(glob.glob('*.flac')):
    with sf.SoundFile(p) as f: R.append((datetime.strptime(re.match(r'(\d{8}T\d{6})',p).group(1),'%Y%m%dT%H%M%S'), f.frames/f.samplerate))
R.sort(); g=[(R[i+1][0]-(R[i][0]+timedelta(seconds=R[i][1]))).total_seconds() for i in range(len(R)-1)]
print(len(R),'chunks | gaps min %.2f med %.2f max %.2f | audio %.1f s vs wall %.1f s = %.4f%% coverage'
      %(min(g),statistics.median(g),max(g),sum(d for _,d in R),(R[-1][0]-R[0][0]).total_seconds()+R[-1][1],
        100*sum(d for _,d in R)/((R[-1][0]-R[0][0]).total_seconds()+R[-1][1])))"
#  24 chunks | gaps min 5.65 med 10.37 max 26.67 | audio 86132.3 s vs wall 86390.3 s = 99.7013% coverage

# 10 s-block RMS (shows the loudness-flatness that drives §2.1 and §3)
ffmpeg -hide_banner -nostats -i 20261023T070002.flac -ac 1 \
  -af "asetnsamples=n=160000:p=0,astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=/tmp/rms.txt" -f null -
#  358 blocks: median -37.9 dBFS, p99 -35.8, max -23.1 (at t=0)

# figures used for the visual hypotheses in §2
ffmpeg -i 20261023T070002.flac -lavfi "showspectrumpic=s=1920x640:scale=log:legend=1:saturation=2" -y .probe/overview.png
ffmpeg -t 180 -i 20261023T070002.flac -lavfi "showspectrumpic=s=1920x640:scale=lin:legend=1:saturation=3" -y .probe/first180s.png
```

Caveat carried into M1: items (b)–(d) of §2's last row come from a model reading a rendered
spectrogram. They are hypotheses to verify, not measurements.
