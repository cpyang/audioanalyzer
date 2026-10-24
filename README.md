# audioanalyzer — abnormal frequency scanner for long recordings

Finds **which frequencies** are abnormal in a recording and **during which time periods**, so an
engineer can walk toward a machine instead of guessing. Built for hourly FLAC survey recordings
(16 kHz mono here), offline, CPU-only, no ffmpeg at runtime.

Design and measured data characteristics: **`REQUIREMENTS.md`** (read §2 first — it explains why
this tool does not use a loudness threshold).

## Install

```bash
cd ~/Audio
uv venv --python 3.12 .venv           # python 3.12: widest wheel coverage (doc §11 R9)
.venv/bin/python -m pip install -e '.[report]'    # or: uv pip install -e '.[report]'
.venv/bin/audioanalyzer version
```

`[report]` pulls matplotlib, needed only by `--report`; scan/CSV/JSON need nothing optional.
**Install it if you plan to write HTML** — the tool now checks up front and exits 1 with the fix
command rather than dying after a 38-minute directory run.

Verify the environment in one line before a long job:

```bash
.venv/bin/python -c "import numpy, scipy, soundfile, matplotlib; print('deps ok')"
```

Audio is never committed: `*.flac` and friends are in `.gitignore`, and the test suite generates
synthetic fixtures calibrated to this corpus's measured floor (`audioanalyzer/synth.py`).

### Checking a machine before a long run

```bash
.venv/bin/audioanalyzer doctor /path/to/recordings
```

Lists every dependency (and which are optional), the bundled libsndfile version, the Python
version against the supported range, BLAS thread pinning, available memory, and how many workers a
directory of that size will actually get on this box. Exit 1 means it cannot scan at all; a missing
`--report` extra is reported as a problem but still exits 0, because CSV/JSON work without it.

On Python 3.10 the `tomli` backport is required and is declared as a conditional dependency;
3.11+ uses stdlib `tomllib`.

## Use

```bash
# one file: ranked anomalies with frequency, level, time span, and evidence
.venv/bin/audioanalyzer scan 20261023T070002.flac

# a whole day (directory or glob): merged timeline, sources, coverage ledger, recurrence
.venv/bin/audioanalyzer scan . --csv out/detections.csv --json out/scan.json --report out/report.html

# parallel across files — the default; `-J 1` forces serial, `-J N` pins the worker count
.venv/bin/audioanalyzer scan . -J 8 --csv out/detections.csv
```

A directory is analysed **in parallel by file**, because each hourly chunk is independent work and
the corpus views (A5 grouping, timeline, recurrence) only need the results afterwards. Measured on
this corpus, 12-core box: **8 m 14 s serial → 4 m 30 s** at the default sizing (1.83×).

Two measurements set the policy, because neither was obvious in advance:

* **Threads matter more than workers.** One file uses ~205 % CPU from BLAS, so 6 default-threaded
  workers on 6 files took 83.7 s while 6 workers with BLAS pinned to 1 thread took **68.6 s** —
  faster *and* using less total CPU. `cli.py` therefore pins `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS`
  to 1 before numpy loads (an explicit `OMP_NUM_THREADS=4` in the environment still wins).
* **More workers is not more speed.** The whole day ran 269.6 s at 6 workers (576 % CPU) and
  267.9 s at 12 (**962 % CPU**) — identical wall clock for 1.7× the energy. Throughput saturates on
  memory bandwidth near half the cores, so auto-sizing is `min(cores/2, MemAvailable/260 MB)` and
  `-J N` overrides it when you know better.

Output is assembled in **input order**, never completion order, so `--csv` is byte-identical
between `-J 1` and `-J 12` (verified on the real 24-file day, `cmp` clean) and only a `[k/n] file`
completion counter goes to stderr, leaving stdout pipeable.

```bash
# more sensitive sweep (triage; expect more false positives)
.venv/bin/audioanalyzer scan . --recall --class tonal

# sub-Hz frequency and drift measurement, for identifying a machine by its tone
.venv/bin/audioanalyzer scan 20261023T070002.flac --mode hr

# "did something start making noise?" — normal first, then compare
.venv/bin/audioanalyzer profile 20261023T0[0-4]*.flac -o out/quiet-night.npz
.venv/bin/audioanalyzer compare 20261023T150003.flac --profile out/quiet-night.npz

# name what you already know (append-only, hand-editable)
.venv/bin/audioanalyzer known --add "UPS double-mains ripple" 50 1.0 5
.venv/bin/audioanalyzer scan . --known known.toml

# or: ask what is always here, and let the tool propose the library (see below)
.venv/bin/audioanalyzer known --seed . --candidates out/known.candidates.toml
```

### Seeding `known.toml` from a recording

Feed it recordings and it reports the sounds that are **essentially always present** — the site's
acoustic fingerprint — as candidate entries, ranked by how permanent they look. On a file with three
tones injected at known levels (the recall fixture), it recovers all three, with the duty cycle
doing the ranking:

```
    #        Hz  kind    files  duty% over dB    dBFS  harmonics
    1   4199.91  line     1/1     100    23.6   -70.9  1
    2   7300.08  line     1/1     100    16.9   -80.8  1
    3     65.08  line     1/1      17    15.0   -45.4  1
```

Two criteria, because one would miss the point: **lines** (narrowband tracks, with their measured
duty cycle) and **floor ridges** (bumps standing above a locally *fitted* power-law trend). The
ridge path needs care, and the reference corpus is the proof: its first output was **16 "always
present sources", all in 6242–7203 Hz, all at −97 dBFS** — one anti-alias filter knee, exploded into
one entry per single-bin noise wiggle, because a straight line fitted across a knee under-reads the
level on the flat side. Three guards fixed it (a 3-bin median to drop isolated spikes, contiguity
clustering, a crest requirement *in the residual* plus a fit window that must reach half an octave
each way). What survives on this site is nothing:

```
no always-present sound found in 24 recording(s): nothing appeared in ≥60% of them.
  examined 6 distinct line(s) and 0 floor ridge(s) across the files
  best agreement seen: line 17% of files, ridge 0% of files (highest steady duty: 17%)
```

That is a real result, not a broken tool, and it agrees with the day census (§2.3): **zero steady
A1 tones in 24 hours**, and the 6 tone tracks that recur at all appear in at most 4 of 24 files
(17%) and never above 17% duty. A site whose noise is event-shaped has nothing honest to put in
`known.toml` yet. When there *is* background hum (mains, a fan that never stops), this is the
command that finds it.

Then, after listening:

```bash
.venv/bin/audioanalyzer known --promote out/known.candidates.toml --index 1,2 \
    --rename 1='CRH-3 fan' --file known.toml
```

Design constraints worth knowing, since they are the difference between a tool and a rumor mill:

* Candidates go to `known.candidates.toml`, **never** into the live library. A name claims what
  physical object makes a sound; only a walk confirms that.
* A candidate's `f0_hz` is always an **observed** line. Fundamentals are never inferred — the
  harmonic f0 fitter once produced "a 5.04 Hz machine" from three 25 Hz tracks, and a generated
  library would have propagated that into every report afterwards.
* Labelling is annotation only. Nothing anywhere filters on a known match, so a mislabelled entry
  cannot hide a finding — it can only mis-name it.

### Day map

`--report` also writes `out/daymap.svg`: time of day across the top, log-frequency lanes down the
side, each minute-and-band cell coloured by the loudest dB-over-floor seen there. Unrecorded
intervals are hatched, and the top tone tracks are labelled with frequency, level and duty.

This is deliberately not a day-long spectrogram. At 900 columns a 24-hour recording gives **96
seconds per pixel**, and the strongest tone finding of the reference day lasts **15.26 seconds** —
a sixth of a pixel, invisible no matter how the colour scale is set. A mark at a time position is
scale-free, so a 15-second event reads as clearly as a 40-minute one. Spectrograms still do the job
they are good at: ±30 s evidence crops beside each detection.

`report.html` is self-contained (figures embedded as base64): sortable table, per-detection
spectrogram + waveform crops, coverage ledger stating what was never recorded.

## What it reports

| Class | Meaning |
|---|---|
| A1 | Steady tone — narrowband, phase-locked, present ≥90 % of the recording |
| A2 | Intermittent / emerging tone, or one with episodes; duty cycle and episode count given |
| A3 | Impulse train — a repetition period (→ rpm-equivalent), from inter-onset autocorrelation |
| A4 | Transient/impact — robust z against the local median ±60 s, with "what rang" spectrum |
| A5 | Source grouping — harmonics and co-switching collapsed into one row per machine |
| A6 | Data quality — DC offset, clipping (with flat-top run length), silence, gaps, undated files |

Every row carries `file`, `channel`, `class`, frequency, `t_start_s` + ISO wall clock, level in
dBFS, tone-to-noise, coherence, isolation, persistence, duty, drift, source id, nearest-neighbour
frequency, and confidence. Thresholds come from `config.py` (or `-c cfg.toml`) and are echoed into
every output, because a report you cannot reproduce is a rumour.

## Tests

```bash
.venv/bin/python -m pytest tests -q        # 116 tests, ~90 s, no audio files needed
```

Ground truth is injected and recovered: a tone added at 300–900 s comes back dated `0:05:00` for
`600.2 s`; a 7.5 s click train is named at ±0.4 s; clipped samples are excluded from transient
counts on purpose (`tests/test_timebase.py` and friends document why).
