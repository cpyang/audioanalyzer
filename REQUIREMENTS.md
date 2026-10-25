# Audio Anomaly Analyser — Requirements

Status: v0.4 — M0–M4 implemented and tested (121 tests green); whole-day verification in §2.3/§7.1
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
* Any event landing in those 4–27 s is **unobservable**, which is why FR-2.4 requires the
  "not recorded" annotation rather than silence, and why `gap_tolerance_s` (FR-5.1) must exceed
  ~30 s or a source firing every ~20 s reads as N sources instead of one.

### 2.3 What the day actually contains (measured, 24 files, 2026-10-23)

Whole corpus, standard mode, no `--recall`: **5023 detections in 8 m 04 s, 309 MB peak RSS**
(NFR-4 asks 10 min/day under 400 MB — both met; the earlier 1336 s run had been an extra pass per
file plus an O(n²) corpus merge, both fixed).

| class | count | what it says |
|---|---|---|
| A1 steady tones | **0** | nothing is phase-steady for >10 s all day |
| A2 tone tracks | **9** | 6 in 24.5–69.5 Hz (duty 2.2–18.7 %), 3 brief mid-band: 1318.5 Hz at 29.4 dB over floor for 15 s, 578.1 Hz, 554.9 Hz; coherence 0.77–0.98, isolation 5.3–19.1 dB. **Steady: 0, intermittent: 9** |
| A3 impulse trains | **0** | no repeating click pattern anywhere in 24 h |
| A4 transients | **5014** | 23.4 Hz–5.96 kHz, z 0.5–42.4 (median 8.9); 163 broadband impacts, 4851 band-limited; median 182/hour (min 73, max 500) |
| A5 sources | 4 | over 5 tonal tracks; f0 24.5 / 26.8 / 65.4 / 69.5 Hz, none with a harmonic stack |
| coverage | 99.70 % | 258.0 s never recorded, 23 gaps, largest 26.7 s |

Three conclusions, each of which changed the design:

* **This is not a tone-dominated site.** §2.1's expectation of a flat broadband floor is confirmed,
  but the information is in the *events*, not in steady lines: zero A1 in 24 hours. A tonal-first
  tool under-delivers here, which is why A3/A4 moved from M4 to M1 (§9). It also means **there is
  nothing honest to seed `known.toml` with**: `known --seed .` over the same 24 files examined 6
  recurring lines and rejected all of them — best cross-file agreement 17% (4/24 files), highest
  duty 17% — and found 0 floor ridges once the anti-alias knee was excluded (FR-7.3b).
* **Every LF "tone" the corpus produces is intermittent and needs listening.** The 24.5–69.5 Hz
  tracks are genuinely narrowband (coherence 0.77–0.98, isolation up to 19.1 dB) but have duty
  cycles of 2.2–18.7 % over 4–19 episodes. The isolation gate (FR-4.1) separates them from the
  region-lift artefacts that used to dominate the LF end; "narrowband and coherent" is still not
  "a machine".
* **5014 events/day is not a readable answer — ranking is the feature.** The report shows the top
  40 by z with band and ring signature; the rest stays in CSV. The dominant recurring population is
  the 88–186 Hz band: 1668 episodes across all 24 files, median gap 4.6 s. Whether that is a
  machine cycle or foot traffic is exactly what §11 R13's listening pass must settle. **Status: not
  listened to — these numbers are machine-truth, not ground truth.**

Recurrence (FR-7.2) reports 9 signatures over this single day with an explicit note that day-of-week
correlation is not computable from one day. What it does say is checkable: the dominant event bands
are spread across the whole 24 h (phase spread 1.9–3.5 h circular, i.e. **not** clock-locked), which
is what ambient site activity looks like, whereas a scheduled machine would cluster at one hour.
Only one tonal signature cleared the 3-episode floor (24.76 Hz, 3 episodes, phase 16.2 h ± 0.96).

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
  `--tz`) with a documented offset convention. **Invariant:** frame *i* begins at sample
  *i·hop*, which the streaming reader must preserve by re-reading `nperseg − hop` overlap
  samples at every block boundary. Reading sequential blocks of `frames_per_block·hop + overlap`
  instead — the intuitive implementation — loses `overlap` samples per block: 96 ms per 30 s
  block here, **0.32% of the timeline, 11.4 s by the end of an hour**, and it looked like a
  tone injected at 300–900 s being present for 598.3 s starting at 299.0 s. Regression-pinned by
  `tests/test_timebase.py`. Timestamps are frame *centres*, so an impulse is dated to within
  `(nperseg − hop)/2` = 48 ms in standard mode; sub-frame onset refinement is deliberately not
  in v1 (it is an onset detector, not a timebase bug).
* 2.2 Multi-file corpus: sort by start time, detect and report gaps/overlaps between chunks
  (this is how "the event at 07:41:12" is located across chunk boundaries). Measured on the
  2026-10-23 day: gaps 5.65–26.67 s, 258 s total (§2.2).
* 2.3 An event that straddles a chunk boundary must be reported **once** in corpus mode
  (dedup by frequency + wall-clock overlap), not twice. **Implemented as two views, because the
  two things you do with the output disagree:** the per-file CSV stays per-file (that is what
  joins against a BMS/PLC log by file), while `corpus.build_timeline` produces the count of
  *things* — same-frequency detections (within `max(recurrence_tol_hz, 0.5 %)`, since the
  interpolated peak of a real tone wanders a fraction of a bin between chunks) separated by no
  more than `gap_s` = 90 s, which must exceed the worst measured blackout (§2.2: 26.7 s) or the
  feature splits again. Each row carries the merged entry's id in `timeline_entry`. A3/A4
  never merge: an event is already atomic, and an A4 `freq_hz` is a band label, not a frequency.
* 2.4 **Coverage ledger:** every corpus run emits, per hour and in total, the recorded vs
  elapsed seconds and the list of blackout intervals. Any summary line claiming "no anomalies
  in 03:00–04:00" must be accompanied by that hour's coverage figure.

### FR-3 Spectral engine
* 3.1 STFT, Hann window, `nperseg` default 2048 (**7.8125 Hz/bin** at 16 kHz),
  `noverlap` = 3/4 (`hop` = 512 → 32 ms, 31.25 frames/s). All three are config.
* 3.2 Two resolution modes: `--mode standard` (above, time-faithful) and `--mode hr`
  (`nperseg` 16384, `hop` 4096 → **0.9766 Hz/bin**, 1.024 s window, 4 frames/s) for precise tone
  frequency and **drift** measurement — the frequency-accuracy figure of merit for source
  identification. A window longer than a second is legal and is the whole point of `hr`; the
  first version of the engine clamped `nperseg ≤ fs`, which silently degraded `hr` to 3.9 Hz/bin
  (caught by `tests/test_timebase.py::test_hr_mode_reaches_its_documented_resolution`).
* 3.3 Magnitude in dBFS (full scale = 1.0), with configurable reference for dBFS-vs-dBSPL
  (a `dbfs_offset` calibration constant; unset ⇒ dBFS, and the report must say so).
* 3.4 Per-frame spectral peak picking with **sub-bin frequency estimation** (quadratic
  interpolation of log-magnitude around the peak) — required to hit AC-2's ±2 Hz.
* 3.5 Noise floor model: per-bin **median** (robust) over the recording, plus per-bin p90;
  computed in a streaming manner (t-digest or fixed histogram), not by holding all frames.
* 3.6 Time-blocked baseline: the floor is re-estimated per `baseline_block_s` (default 300 s)
  so slow environmental change does not hide (or fake) tones.

### FR-4 Detectors
* 4.1 A1/A2 tonal. **Implemented as two stages and four gates**, because a single threshold was
  shown to be unmeasurable on this data (§2.1: a bare FFT bin is Rayleigh-distributed and exceeds
  its local floor by 6 dB in ~25% of frames with no tone present at all — 17.9 dB peaks were
  measured on a tone-free realistic floor):
  * *Stage 1 (mean spectrum)* — candidate = bin exceeding the **adaptive symmetric** neighbourhood
    floor by ≥ `tone_noise_db` (default **6.0 dB**), energy confined to ≤ `max_tone_bins`
    (**3**) adjacent bins — measured as the run within `tone_width_db` (**3 dB**) of the tone's
    *own peak*, level ≥ `min_level_dbfs`. Averaging N frames suppresses the noise
    spike problem by √N without holding per-frame peaks (15.6 M peaks / 2.27 GB was the
    measured cost of the naive version). The width reference matters: defining half-height as
    "floor + half the excess" made an 880 Hz tone at −40 dBFS disappear, because the corpus
    floor falls ~5.5 dB/octave and a ±250 Hz window under-reads it by ~8 dB, inflating a 3-bin
    tone to 4.
  * *Stage 2 (per frame, candidates only)* — a frame counts as present when ALL of:
    (a) power-smoothed tone-to-noise ≥ `tone_noise_db` over `presence_smooth_s` (2 s);
    (b) **phase coherence** ≥ `min_coherence` (0.70) — a locked oscillator keeps
    |⟨X·e^{-i2πft}⟩| ≈ ⟨|X|⟩, band energy collapses; measured 0.98–0.99 for injected tones
    vs 0.28–0.55 for corpus wobble;
    (c) **instantaneous isolation** ≥ `min_isolation_db` (4.0 dB) against the ±2..5 neighbour
    bins, which is what separates a tone from a whole LF band lifting together.
  * *Occurrence gate* — at least one **episode ≥ `min_tone_s` (10 s)** after merging gaps
    < `gap_tolerance_s` (2 s). Gating on episode length rather than persistence-over-span is
    deliberate: straggler frames 45 minutes apart otherwise stretch the span and bury a genuine
    intermittent tone (the 65 Hz test tone scored its correct 17.3% duty and still failed a
    50%-persistence gate).
  `--recall` relaxes all four (3 dB / 0.5 coherence / 2 dB isolation / 2 s).
* 4.2 Episode segmentation per frequency track: merge consecutive frames, split after
  `gap_tolerance_s` (default 2 s) of absence; a tone that never splits is "steady", otherwise
  "intermittent" with a duty cycle and an episode count.
* 4.3 Drift: fit frequency vs time per track; report slope (Hz/min) and `r²`; classify
  `stable` (<0.5 Hz/min), `drifting`, or `stepped` (breakpoint fit).
* 4.4 A4 transients: per-frame **per-band and broadband** level compared against a rolling
  **median and MAD** of the same band over ±`z_window_s` (default 60 s, computed on a 250 ms
  decimation for cost). An event needs **both** robust z ≥ `transient_sigma` (6) **and** absolute
  rise ≥ `min_event_db` — which measured **15 dB**, not the 6 dB originally specified: corpus
  frames wobble to 6–8 dB on their own, while a real impact — even one whose level sits *below*
  the LF floor — produces a 33–53 dB rise. With 6 dB the calibration run made 13 false
  transients out of 90 s of clean rumble, 9 of them in the final 15 s, where a `median_filter`
  edge mode had collapsed the MAD to zero (`mode="reflect"` now). Frames containing a clipped
  sample are excluded (FR-1.3), using a **left-aligned** window sum over the frame's hop
  segments — a centred window flags the wrong frames and lets clipping through as a "broadband
  impact". The top `event_detail_top` (40) events per file get a second targeted pass reporting
  the 3 loudest spectral components during the event ("what rang", not just when). A component
  below `cfg.min_freq` (or above 0.95·Nyquist) is **not** a ring: the first version published
  15.62 Hz as an impact's signature frequency — the analysis filter's DC skirt — which made 10
  ordinary low thumps read as sub-audible tones. Below `min_freq` the event keeps its band centre,
  so the row says "band", never "tone at 15 Hz".
* 4.5 A3 periodicity: A4's onset times per band are autocorrelated as an **inter-onset interval
  histogram** over `comb_min_period_s`–`comb_max_period_s` (0.5 s–3600 s), boxcar-smoothed at
  `period_smooth_s` (0.1 s) because two timestamps carry ±32 ms of frame-grid quantisation each.
  Evidence is reported as **countable facts** — n onsets, the fraction of intervals at the
  candidate period, jitter σ — and requires ≥ `period_min_pairs` (4) intervals,
  ≥ `period_min_fraction` (0.15) of all intervals, and ≥ `period_min_repeats` (4) recurrences.
  The first version ranked by peak/mean of the histogram, which scored 6740× on 16 onsets and
  claimed three mutually exclusive periods at once; that metric is gone, `rpm` is derived from
  the period so a 6-blade fan can be checked against a nameplate.
* 4.6 A6: DC offset > −60 dBFS, any clipped sample, >5 s of digital silence, sample-rate
  change, duration/filename mismatch, non-finite samples.
* 4.7 **FP control:** harmonics of *detected* anomalies are reported under their source
  (A5), not as independent anomalies; every detection lists its nearest neighbour frequency
  and the ratio, so a human can see *why* it was called; `--exclude-band Hz:Hz` and
  `--min-freq/--max-freq` suppress known-good bands.

### FR-5 Source grouping
* 5.1 Cluster A1/A2 tracks into sources by (a) harmonic relation to a common f0
  (`harmonic_tol_pct` default 1.0%) and (b) temporal co-occurrence (Jaccard of on-time ≥ 0.6).
  Two corrections the first whole-day run forced: **(b) is only evidence when switching exists to
  correlate** — two steady tones are both active 100 % of an hour, giving Jaccard 1.0 and merging
  every unrelated machine in the plant into one "source"; and **only A1/A2 participate**, because
  an A4 event's `freq_hz` is its band's geometric centre, which merged 1178 separate impacts into
  a single 61 Hz "source" with 1178 harmonics. `n_harmonics` now counts *distinct* harmonic
  numbers, not members.
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
  Labelling is **annotation only** — no code path filters on a known match — so a wrong entry
  degrades a label, it cannot hide a finding. That is what makes auto-seeding safe enough to offer.
* 7.3b **Seeding the library from a recording** (`known --seed RECORDING...`): proposes the sounds
  that are *essentially always present*, so they stop being findings and become site fingerprint.
  Two criteria, because one is not enough — **lines** (narrowband tracks, reported with their
  measured duty cycle) and **floor ridges** (bumps in the median spectrum standing above a locally
  fitted power-law trend). The ridge criterion is not decoration: the reference day contains
  **zero A1 steady tones**, so a duty-only seeder would correctly answer "nothing is always
  present" and be useless. Its trend must be a **fit, not a window median** — the floor tilts
  ~5.5 dB/octave, and a median across the knee at ~150 Hz under-reads the real 125 Hz feature
  (+5.1 dB measured by median against +7.3 dB by fit, with flanking bins at −2.1 and −1.1). A pure
  power law must yield no candidates, which is a test, not a hope (`tests/test_seed.py`).
  A candidate's `f0_hz` is always an **observed** line; fundamentals are never inferred here,
  because `infer_f0` on three 25 Hz tracks once invented "a 5.04 Hz machine" and a generated
  library would propagate such a claim into every later report. Output goes to
  `known.candidates.toml`, and promotion (`--promote --index 1,3 --rename 1='CRH-3 fan'`) is the
  human step: a name asserts what physical object makes the sound, and only a walk can confirm that.
* 7.4 Export a detection CSV/JSON suitable for joining against external event logs
  (PLC/BMS/pump schedules) — the tool must not require those logs, only be joinable.

### FR-8 Outputs
* 8.1 `detections.csv` (flat, one row per detection, stable column order) and
  `detections.jsonl` / `summary.json` (per file: media info, config echo, sources,
  detections, data-quality flags).
* 8.2 `report.html`: self-contained, sortable table (freq, span, time, level, prominence,
  class, coherence, isolation, source, known-name, confidence) plus, for the top figures, a
  crop (±30 s, ±1 kHz; y-axis in Hz) and a 1 s waveform zoom, so noise floor vs tone is
  legible. Also one full-recording overview figure (log-freq, matching how a human reads it).
  **The overview colours by dB over an adaptive frequency floor, not absolute dBFS**: on the
  reference hour the LF→HF slope of the noise floor accounts for **95.8 % of the image variance**
  while the informative residual spans only −3.1..+12.5 dB, so an absolute window buries every real
  feature; percentile windows were worse still (the bottom 2 % of cells are numerical dust from
  masked bins, which put vmin at −122 dBFS, 28 dB below the genuine floor).
* 8.2b **Day map** (`daymap.svg`, embedded at the top of the report): time of day on x, log-frequency
  lanes on y, one cell per minute × lane, coloured by the highest dB-over-floor seen there (robust z
  for impacts), with the top tone tracks labelled and **unrecorded intervals hatched**. This exists
  because a day-long spectrogram cannot answer "what happened when": at 900 columns a day is
  **96 s per pixel**, and the strongest tone finding of the reference day lasts **15.26 s** —
  0.16 of a pixel, structurally invisible. A mark at a time position is scale-free, so a 15-second
  event reads as well as a 40-minute one. The map is generated from *detections* (costs nothing
  beyond the scan) and states its own gap coverage, so a blank stretch is never evidence of absence.
* 8.3 `--stdout` human summary: top-N lines, no HTML needed (for scripts and for chat relays).
* 8.4 Exit codes: 0 clean, 1 usage **or missing optional dependency**, 2 unreadable/decode error,
  3 detections exceed a
  `--fail-on` severity (makes it usable as a scheduled check). Severity for A1/A2 is
  `coherence × min(1, ttn/2·tone_noise_db) × min(1, 0.5+duty/100) × stability(level_std)` — a
  product of gate margins rather than a blend of weights, so any value can be explained factor
  by factor; A4's is `z/20`. Both `argparse` (which exits 2 on usage) and the decode path are
  covered by `tests/test_outputs.py` — that test is how an inert `--fail-on` was found.
  **Both environment checks run before any analysis:** `--report` without matplotlib installed, and
  an unreadable file in a directory, must cost seconds. A 24-file run that failed on the matplotlib
  import after 38 minutes of CPU is the counter-example this clause exists to prevent.
* 8.5 All outputs record the tool version, the config, and the exact thresholds used
  (a report is not reproducible without them).
* 8.6 **Environment preflight.** `audioanalyzer doctor [dir]` reports every dependency (with which
  are optional), libsndfile version, Python against the supported range, BLAS thread pinning,
  available memory, and the worker count a directory of that size will get. Exit 1 only when the
  tool cannot scan at all — a missing `--report` extra is a warning, because CSV/JSON still work.
  Added after a field machine ran a whole day and then died on `import matplotlib`: an environment
  question must cost seconds, not the run.

### FR-9 CLI (surface, names may change)
```
audioanalyzer scan  FILE...         [-c cfg.toml] [--mode standard|hr] [--min-prominence DB]
                                    [--class tonal|transient|all] [--top N] [--report DIR]
                                    [--csv F] [--json F] [--tz TZ] [--min-freq/--max-freq]
                                    [-J N]   (files in parallel; 0 = auto, 1 = serial)
audioanalyzer profile  REF...       -o baseline.npz
audioanalyzer compare  FILE --baseline baseline.npz
audioanalyzer periodic FILE...      [--report DIR]
audioanalyzer known  [--list] [--add NAME F0 TOL]
audioanalyzer doctor [FILE...]      environment preflight: deps, threads, worker sizing
audioanalyzer version / config --dump
```

---

## 6. Non-functional requirements

| ID | Requirement | Target / basis |
|---|---|---|
| NFR-1 | Language / runtime | Pure Python, ≥3.10. **3.12 chosen** for wheel breadth (host has 3.12 and 3.14; `uv` at `~/.local/bin/uv`) |
| NFR-2 | Dependencies | `numpy`, `scipy`, `soundfile`, `matplotlib` (report only), `tomli`/stdlib TOML. **No `librosa`** (drags numba/llvmlite; no MIR needed). All four verified to have cp312 manylinux wheels (numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2; soundfile 0.14.0 ships `py2.py3-none-manylinux_2_28_x86_64` with bundled libsndfile). **An optional dep must be declared as an extra AND checked before any analysis starts** — not imported at the end of a long run (§11 R14) |
| NFR-3 | Install | `uv venv --python 3.12` in `/home/cpyang/Audio/.venv`, `uv pip install -e '.[report]'` — offline-reproducible via `requirements.lock`. No conda, no system-site-packages mutation |
| NFR-4 | Throughput | ≤ 45 s wall per 60-min/16 kHz mono file in standard mode; ≤ 4 min in `hr` mode. **Measured 14.9 s wall per file but at ~205 % CPU** (FFT/BLAS threads) — so the honest cost is ~31 s of CPU, not 1 core, and a worker budget must divide cores by 2 rather than assume 1. **Whole 24-chunk day: 8 m 04 s serial → 4 m 30 s in parallel** (`-J` auto = 6 workers here), 225 MB peak. Directory scans run **files in parallel by default**, sized `min(cores/2, MemAvailable/260 MB)`. M0 measured: ingest pass = 1.35 s wall, 43 MB RSS (≈2650× realtime) |
| NFR-5 | Memory | **< 400 MB peak RSS** per file (full STFT of the reference file would be 111 600 frames × 1025 bins ≈ 460 MB in float32 → must be blocked, per FR-1.2/3.5). **Measured 210 MB per worker** ⇒ `jobs × 210 MB` must fit `MemAvailable`; the auto-sizer treats that as a bound, not advice |
| NFR-6 | Determinism | Same input + config ⇒ byte-identical CSV. No RNG; sort by (t_start, freq). **Holds between `-J 1` and `-J N`**: results are reassembled in input order, never completion order (`tests/test_jobs.py`) |
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

### 7.1 Verification results (measured, 2026-10-24)

| AC | Result | Evidence |
|---|---|---|
| AC-1 | **PASS** | `scan` on the reference hour: 57 173 333 samples, DC −38.25 dBFS, 375 samples at \|x\|=1.0, 464 at the top PCM code, 26.5 ms flat-top run — all four agree with ffmpeg/MediaInfo independently. Zero detections in bin 0 (the DC notch in `block_db`/`block_spectrum` leaves ≤ −163 dBFS residue there) |
| AC-2 | **REFUTED — rewritten from measured truth, as this AC's own clause requires** | There is **no line at 6.8–7.0 kHz**. Mean spectrum: max −91.4 dBFS in 6800–7000 Hz against a −95.4 dBFS local median (a 4 dB ripple in the floor), and 62 dB below the loudest feature in the file (70 Hz at −53.0 dBFS). The M1 detector confirms it: **0 candidate bins** in the reference hour. The visual "line" was an artefact of reading a log-freq PNG |
| AC-3 | **mechanism PASS, corpus finding NIL** | Synthetic: a 10-click train at 7.5 s is recovered at 7.5 ±0.4 s (±5%) with 10 onsets. On the real recordings: the 5–6.5 kHz "comb" is likewise refuted — **0 of 192 bins** in that band exceed their local median by 6 dB |
| AC-4 | **PASS on synthetic; real-audio FP rate not yet ear-validated** | 10 injected clicks at −20 dBFS → 10/10 found, dated within 48 ms (the frame-centre quantum, < the ±0.2 s target); 0 false transients in 90 s of corpus-shaped rumble after calibration. On one real hour A4 reports 73–205 events/hour with named ring frequencies — plausible machinery, unlistened |
| AC-5 | **PASS** | 50/100/150/1000 Hz stack ⇒ exactly one source, f0 = 50 Hz ±0.5, members grouped by harmonic fit; no spurious sources |
| AC-6 | **PASS** | `compare` against a profile built from the same audio without a 1750 Hz tone lists **only** that tone as new; identical audio lists nothing new and nothing gone (band deltas < 0.05 dB) |
| AC-7 | **PASS** | A tone injected at exactly 300–900 s is dated `0:05:00` for `600.2 s` in a 3573 s file. This AC is what caught the 0.32% streaming timebase bug (§11 R11) — the same run had previously said `0:04:59` for `598.3 s` |
| AC-8 | **open — needs a human** | `report.html` is written self-contained (figures as base64, no external refs — asserted in `tests/test_known.py`); whether a non-programmer can identify an anomaly in a crop has to be judged by you |
| AC-9 | **PASS** | `test_threshold_sweep_is_monotonic`: counts never increase as the gate tightens, logged per step |

Whole-day throughput and memory (NFR-4/5): **see §2.3** for the census and totals.

---

## 8. Deliverables & layout

```
/home/cpyang/Audio/
  REQUIREMENTS.md            this document
  README.md                  quick start + worked examples
  .venv/                     uv-managed, python 3.12  (created in M0)
  pyproject.toml             metadata + console-script, [report]/[dev] extras
  requirements.lock          pinned exact versions for reproducible offline installs
  known.toml                 hand-edited source library (created by `audioanalyzer known --add`)
  audioanalyzer/
    __init__.py  __main__.py cli.py config.py models.py
    ingest.py       block decoder, MediaInfo, A6 flags, coverage ledger (FR-1/2/4.6)
    spectral.py     STFT engine, dB, DC notch, aggregate + candidate stage 1 (FR-3)
    baseline.py     per-bin p50/p90 histograms, time-blocked, save/load .npz (FR-3.5/3.6)
    scan.py         single-file pipeline (stage 1 → candidates → stage 2 → detectors)
    detectors/      tonal.py (A1/A2)  transients.py (A4)  periodicity.py (A3)  sources.py (A5)
    profile.py      floor profiles + compare mode (FR-6)
    known.py        known-source naming + cross-chunk recurrence (FR-7.2/7.3)
    report.py       CSV/JSON writers (FR-8.1)
    plots.py        spectrogram/waveform figures, Agg, no GUI (FR-8.2)
    report_html.py  self-contained HTML report (FR-8.2/8.5)
    synth.py        corpus-calibrated test signals: rumble, add_tone, add_click (AC-2/4/5)
  tests/            test_ingest  test_spectral  test_detectors  test_transients  test_timebase
                    test_profile  test_known  test_outputs  test_corpus  test_jobs
  out/              per-run artifacts: detections.csv, scan.json, report.html (git-ignored)
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
* **M1 (0.5 day) — DONE 2026-10-24.** Spectral engine + A1/A2/A5 + CSV. The first attempt
  picked peaks per frame and needed 15.6 M peaks, 2.27 GB RSS and 112 s per file to report one
  detection; the shipped design is **two stages** (mean-spectrum candidates → targeted per-frame
  persistence/coherence), which costs 189 MB and 6.0 s for stage 1. Four gates, all calibrated
  against measured behaviour (FR-4.1): smoothed tone-to-noise, phase coherence, instantaneous
  isolation, and a ≥10 s episode. AC-2's "6.8–7.0 kHz line" is **refuted** (§7.1), and the
  reference hour reports **0 candidate bins**.
* **M2 (0.5 day) — DONE 2026-10-24.** `--mode hr` (and the `nperseg ≤ fs` clamp that had silently
  made it 3.9 Hz/bin instead of 0.98 — fixed, regression-pinned), drift classification, A5 source
  grouping across a whole selection, nearest-neighbour annotation (FR-4.7), HTML report with
  embedded crops (FR-8.2), and the **0.32% streaming timebase bug** found via AC-7 (§11 R11) —
  the most consequential fix of the build: every timestamp before it was ~11 s wrong per hour.
* **M3 (0.5 day) — DONE 2026-10-24.** A4 transients (median/MAD against ±60 s local baseline,
  clipped frames excluded, `min_event_db` recalibrated 6→15 dB from measurement) and A3
  periodicity (inter-onset histogram; the peak/mean metric that claimed 6740× confidence is
  gone). `profile`/`compare` land here too: a tone injected into one file and absent from the
  reference is the only thing `compare` reports, and nothing else (§7.1 AC-6).
* **M4 (0.5 day) — DONE 2026-10-24.** Corpus merge + coverage ledger in every output (FR-2.4),
  recurrence across chunks with hour-of-day clustering (FR-7.2) including an explicit
  "single-day corpus, day-of-week not computable" note, `known.toml` naming (FR-7.3), threshold
  sweep monotonicity (AC-9), README. **86 tests green.** The corpus timeline (FR-2.3) landed here
  too, as a *second view*: per-file CSV stays per-file for BMS joins, while `build_timeline`
  answers "how many things were there" — a tone crossing an hour seam reads as one feature in
  `n_files` chunks rather than two detections.
* **Definitive whole-day pass (v0.2.0, 2026-10-24):** `scan .` over 24 files → 5023 detections in
  **8 m 04 s**, **309 MB** peak RSS (NFR-4/5 met; the earlier 1336 s run had been ~2 passes/file
  plus an O(n²) merge), CSV + JSON + a 1.78 MB self-contained HTML report. Census in §2.3.
* **Parallel directory scans (v0.3.0, 2026-10-24):** the same 24 files with CSV+JSON+HTML in
  **4 m 30 s** at 6 workers, 563 % CPU, 225 MB peak — **1.8× the serial rate**, byte-identical CSV.
  What the sizing experiments established, because the intuition was wrong twice: pinning each
  worker's BLAS threads to 1 beat adding workers (6 files: 120 s serial → 83.7 s at 6 default-
  threaded workers → **68.6 s** at 6 pinned workers), and going wider is flat (whole day: 269.6 s
  at 6 workers vs 267.9 s at 12, for 1.7× the CPU) — throughput is memory-bandwidth bound, so auto
  sizing is `cores/2` rather than `cores`. `doctor` reports this environment's numbers before a
  run, and `--report`'s missing-extra case now fails in seconds instead of after 38 minutes (§11
  R14, FR-8.6).
* **Day map + known.toml seeding (v0.4.0, 2026-10-24):** `daymap.svg` (marks, not pixels — the
  measured case against a day spectrogram is in FR-8.2b) and `known --seed`, which proposes the
  always-present sounds. On this corpus the seeder's answer is **none**, and it now says so with
  evidence rather than printing an empty table. **116 tests green.**
* **v0.4.1 (2026-10-24):** stale-candidate safety. A zero-candidate seed run returned *without*
  rewriting `known.candidates.toml`, so the 16 bogus entries the pre-fix ridge estimator had
  written stayed on disk one `--promote` away from the live library. Candidate files are now
  rewritten by every run (opt-in `--append` to accumulate), which also repairs the index contract —
  `promote` indexes file positions while the printed table numbers come from the current run.
  **121 tests green.**
* **Day-map layout + artifact-only rendering (v0.5.0, 2026-10-24):** three asks, one of them a
  latent bug. `daymap` now draws **one panel per calendar day on a shared hour-of-day ruler**
  (auto-selected above one day, `--by-day`/`--single-panel` to choose), so a day-over-day change is
  a shape difference in the same column; `--lane-h`/`--lanes`/`--map-bucket` make the frequency
  axis readable instead of 13 px tall; and `audioanalyzer daymap out/scan.json` re-renders from a
  previous run's artifact in **0.8 s instead of 4½ min**, which is what makes layout iteration
  actual iteration. Fixing the multi-day test exposed an R10 hole: a blackout **later than the last
  detection** was drawn past the canvas edge and effectively deleted — the old test passed only
  because it asserted the string was present. The span now includes gaps. **130 tests green.**
* **Day-map view filters (v0.6.0, 2026-10-24):** `--map-lo/--map-hi` (frequency window) and
  `--map-cut` (hide cells below N dB over the local floor, robust z for impacts). Two traps, both
  fixed before shipping: the existing `np.clip` on the lane index would have folded every
  out-of-band detection into the nearest visible lane, painting a hot band that does not exist, so
  out-of-window detections are dropped and the lane check is computed, not clamped; and a filter
  that hides things must say so, so the caption carries the band and the drop counts, an
  all-filtered-out map prints `EMPTY AFTER FILTERS` in the middle of the canvas instead of looking
  like a quiet site, and `--map-lo >= --map-hi` fails before analysis rather than after 4½ minutes.
  Measured on the reference day: `--map-cut 20` keeps 331 of 5,023 detections — 330 impacts and the
  single 29.4 dB tone — because the impact population sits at median z 10.1. **141 tests green.**
* **M5 (later)** — cross-night trending, optional ML classifier, capture-side integration.
  Before any of that: the **listening calibration** (R13) and a **multi-night corpus** (FR-7.2's
  day-of-week claim is not testable on one day), because the numbers in §2.3 are machine-truth
  and the false-positive rate against real ground truth is still unmeasured.

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
| R1 | An "anomaly" is just the floor's own variance (FFT leakage from the loud DC–500 Hz band leaking upward and faking high-band tones — Hann sidelobes are −31 dB) | Window selection + leakage estimate in the baseline model; require `prominence > floor_variance + margin`; test with the real floor, not white noise. **Now three explicit gates instead of one margin:** a per-frame sliding-mean DC notch (measured residue ≤ −163 dBFS in bins 1–3), per-bin adaptive floor windows, and an instantaneous ±39 Hz isolation test. A fourth was needed for the top of the band: see **R11** |
| R2 | Mains-hum harmonics (50/60 Hz + comb) dominate the low band and bury real findings | A5 harmonic grouping, per-band suppression, and `known.toml` naming so hum is *stated*, not repeated 40× |
| R3 | 16 kHz decimation without a proper anti-alias filter means sub-Nyquist "tones" could be aliases of >8 kHz switching noise (SoX tag ⇒ chain unknown) | Flag in report; `hr` mode + drift signature (aliases drift with playback rate, true tones do not); recommend recording at 48 kHz for confirmation sweeps |
| R4 | A steady tone is actually a property of the recorder (self-noise, PLL spur) not the site | Baseline profile from a known-quiet/"dummy" recording ⇒ recorder artefacts disappear from `compare` output |
| R5 | 60-min files × many days ⇒ runtime and clutter | Blocked streaming (NFR-5), `--top`, per-run `out/<ts>/` dirs, corpus manifest cache keyed by (path, mtime, size) |
| R6 | Clipping at full scale (464 samples at the top PCM code, one 26.5 ms flat-top run measured) creates broadband spikes read as A4 events | A4 excludes clipped frames; A6 reports them with the run length (FR-4.4) |
| R7 | Threshold defaults tuned on one file don't transfer to a noisier site | All thresholds in config, sweepable (AC-9); `profile` relearns the floor per site |
| R8 | Time-axis misattribution ⇒ engineer walks to the wrong place / wrong shift | AC-7 + explicit `--tz`, chunk-gap detection, both-offset-and-wall-clock in every row |
| R9 | Python 3.14 chosen by accident ⇒ wheel gaps (e.g. scipy/numpy cp314 availability is newer) | Pin `--python 3.12` in M0 and in `pyproject.toml` (`requires-python = ">=3.10,<3.13"` until 3.13 wheels are proven here) |
| R10 | **Capture blackout (measured §2.2):** 258 s/day, up to 26.7 s in one gap, so quiet or brief sources can be missed entirely and an "all clear" report can be wrong | Coverage ledger in every report (FR-2.4); gap-aware period search (FR-7.2); severity wording never says "no anomalies", only "none detected in the 99.7 % covered". Raise a capture-side ticket for the underruns |
| R11 | **Streaming timebase drift** — any block reader that treats chunks as independent while computing frame times as `i·hop` loses the window overlap per block. Silent (0.32%/hour here), and every downstream stat inherits it | Invariant pinned by `tests/test_timebase.py` (frame count closed form + impulse dating in a 3 s-block file). `iter_blocks` re-reads `nperseg − hop` overlap per chunk. Class of bug to suspect first in any new reader |
| R12 | **Coherent artefacts that no gate can reject:** the resampling spur at Nyquist scores phase coherence 0.97 — indistinguishable from a motor whine by measurement alone | Explicit `nyquist_guard = 0.98` band exclusion (R1/R3), reported as a guard rather than hidden; `known.toml` naming for what the site says these lines are; confirmation sweep at 48 kHz if a near-band-edge line ever matters |
| R13 | A4's real-audio false-positive rate is unmeasured — synthetic rumble says 0/hour, one real hour said 73–205/hour | Report events per hour and per band so the noise is visible; `--class tonal` to suppress; needs a labelled ~2 h listening pass to set the operating point |
| R14 | **Environment and parallelism fail late, not early.** A missing optional dep (matplotlib) surfaced only at the report step of a 38-minute run; and a naive parallel sweep either oversubscribes FFT threads (worker measured ~205 % CPU, so `cores` workers thrash) or multiplies a 210 MB working set past free RAM | `--report` and every input file are checked before analysis starts (exit 1/2 in seconds); extras declared in `pyproject.toml`; worker count auto-sized `min(cores/2, MemAvailable/260 MB)` with `-J` to override; corpus views assembled in the parent in input order so parallel and serial output are byte-identical |

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
