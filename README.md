# audioanalyzer — abnormal frequency scanner for long recordings

Finds **which frequencies** are abnormal in a recording and **during which time periods**, so an
engineer can walk toward a machine instead of guessing. Built for hourly FLAC survey recordings
(16 kHz mono here), offline, CPU-only, no ffmpeg at runtime.

Design and measured data characteristics: **`REQUIREMENTS.md`** (read §2 first — it explains why
this tool does not use a loudness threshold).

## Install

```bash
cd ~/src/audioanalyzer
uv venv --python 3.12 --prompt noise .venv   # 3.12 = widest wheel coverage (§11 R9);
#                                             --prompt sets the shell prefix; uv 0.5.9 otherwise
#                                             bakes an empty prompt and bash falls back to ".venv"
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
  physical object makes a sound; only a walk confirms that. Each run **rewrites** that file — a
  stale list from an earlier run must not sit one `--promote` away from the library, and the printed
  indices have to mean what the file contains (`--candidates FILE --append` opts into accumulating).
* A candidate's `f0_hz` is always an **observed** line. Fundamentals are never inferred — the
  harmonic f0 fitter once produced "a 5.04 Hz machine" from three 25 Hz tracks, and a generated
  library would have propagated that into every report afterwards.
* Labelling is annotation only. Nothing anywhere filters on a known match, so a mislabelled entry
  cannot hide a finding — it can only mis-name it.

### Day map

`scan --report` writes `out/daymap.svg` and embeds it in the report. It is a **marks** diagram, not
a spectrogram, for two measured reasons: 95.8 % of a day spectrogram's image variance is the noise
floor's own low-to-high tilt (microphone response, not the site), and at day scale a pixel is 96
seconds wide — the strongest tone of the reference day lasts 15.26 s, so it would occupy 0.16 of a
pixel and read as "nothing happened". A mark at a time position is scale-free.

Three layout controls:

| flag | effect |
|---|---|
| `--lane-h PX` | pixels per frequency lane. Default 18 (was 13); **24–30** if you are reading bands rather than scanning shape |
| `--lanes N` | frequency lanes (geometric 20 Hz → 7.6 kHz). More lanes = finer frequency detail, taller map |
| `--map-bucket S` | seconds per column (default 60) |
| `--map-lo HZ` / `--map-hi HZ` | frequency window. Detections outside it are **dropped**, never folded into the nearest visible lane — a window that clamped would paint every sub-band sound as a hot band at the edge of the view |
| `--map-cut DB` | hide cells under this many dB over the local floor (robust z for impacts — the same number the colour ramp uses) |
| `--by-day` / `--single-panel` | one panel per calendar day with **x aligned on hour of day**, or one long wall-clock panel. Auto-selects `--by-day` when the data spans more than one day |

Both filters are **view settings, not measurements**, so the map says what it hid. The caption
carries the window and the drop counts (`331 of 5,023 detections drawn — 4,692 under the cut`), and
a map filtered down to nothing prints `EMPTY AFTER FILTERS` across the middle rather than handing
you a blank canvas that looks like a quiet site. What that cut reveals about the reference day is
worth stating: the 5,014 impacts run at median z 10.1 (p95 21.3), so `--map-cut 20` keeps 330 of
them plus the one 29.4 dB tone — i.e. the day's texture is individually *below* 20 over the floor,
which is exactly why it reads as texture rather than as events.

The day-aligned layout exists for multi-night surveys. Stacking one panel per day on a shared hour
ruler turns "is Tuesday different?" into a shape comparison in the same column: on a synthetic
three-day test (day B silent 02:00–04:00, day C drifted +90 min), day B's 02–03 h cells drop to
2 and 0 against day A's 105 and 146, and day C's busiest hour moves 18:00 → 20:00 — visible without
having to remember where yesterday's evening sat.

### Reports and maps from artifacts — no re-scanning

`scan.json` is a complete record of a run, so the report and the day map can be rebuilt from it
without opening a single FLAC:

```
audioanalyzer report out/scan-2026-10-23.json -o out/report.html
audioanalyzer report out/scan-2026-*.json -o out/week.html   # merged across nights
audioanalyzer daymap out/scan.json -o out/daymap.svg --by-day
```

Merging artifacts follows two rules, both because a report that looks internally consistent can
still be wrong:

* **Identity is (filename, frame count, class, frequency, start time), never the stored path and
  never a minute-bucketed key.** A path is kept exactly as typed, so scanning `~/src/audioanalyzer`
  and `./` describes the same recording twice under two strings — and the merged census, timeline and
  recurrence would all double without any symptom. A coarse key is worse still: bucketing the start
  time to a minute makes detections from a *single* run collide, and 47 % of one census disappeared
  before it was fixed.
* **Derived views are recomputed, never merged.** Two `sources` blocks each contain a feature that
  straddles their boundary; concatenating them counts it twice and invents episodes.

If the artifacts were produced under **different thresholds**, `report` refuses and explains which
gate differs — one confidence column across different gates is a false comparison. `--allow-mixed`
merges anyway and labels the fact. Argument order cannot change the output: `report a.json b.json`
and `report b.json a.json` are byte-identical.

**The evidence figures come from a cache the scan writes.** Figures are the only part of a report
that needs the audio, so the scan renders them once while it has it — the top `--figure-top` (40)
detections plus one overview per recording, ~1 MB a night — and a later report reads the cache. A
card whose figure is missing says so and names the command that would have made it, because an empty
box beside a detection reads as "nothing interesting in the spectrum". `--no-figures` skips the
cache; `--figures DIR` points a report at one.

This closes a defect that ran the project's whole length: the crop call had a transposed index
(`db[fband]` selects *frames* by a mask built for *bins*), so it raised on 100 % of calls and the
report's `except Exception` turned that into no spectrograms at all — the reference day's report
shipped 12 cards with 12 images, all waveforms. No test could have caught it: the HTML assertions
checked the *absence* of external URLs, which rewards producing fewer images. There is now a test
that a card contains two images, and one that a −45 dBFS sine reads −45 dB in the crop.

### Merging and splitting artifacts

The artifacts are data, so they can be rearranged without touching the audio:

```
audioanalyzer merge out/night-01.json out/night-02.json -o out/two-nights.json
audioanalyzer split out/scan.json --by hour  --dir out/hourly    # scan-20261023T07.json ...
audioanalyzer split out/scan.json --by day   --dir out/daily     # scan-20261023.json ...
audioanalyzer split out/scan.json --by month --dir out/monthly   # scan-2026-10.json ...
```

`merge` takes 2+ artifacts (or 1, to normalise one) and writes a new one; `split` writes one file per
bucket, named `<prefix>-<bucket>.json` so the filenames sort chronologically. Both refuse to
overwrite an existing artifact unless you pass `--force` — these files *are* `--incremental`'s
memory, and clobbering one quietly rewinds what you have already scanned. A split is recording-atomic:
a detection always stays in the file it was measured from, and if a recording straddles a boundary its
bucket says so rather than duplicating the finding in two parts (0 of 5,023 rows in the reference day
straddle a day boundary; hourly chunks make hour-splits a clean 1:1).

Re-running the day through `split --by hour | merge` reproduces it exactly: 5,023 detections, same
identity set, same `coverage.audio_s` (86,132.309 s), and no lost `analysis` or `quality_flags`.
Getting there surfaced three defects that were already in the code, each invisible to the path that
hid it:

* **The JSON rows were serialized before the annotation passes ran.** `nearest`, `label`,
  `source_id`, `timeline_entry` are set after the per-file blocks were frozen, so an artifact recorded
  a *pre-annotation* view of its own detections, and every artifact-derived report re-deriving from it
  disagreed with the CSV of the same run.
* **The loader dropped any row key it did not recognise.** `signature` and `nearest` are neither
  dataclass fields nor in the whitelist, so they fell off the end of a branch chain: an
  artifact-derived report silently lost each impact's waveform signature. Unrecognised keys now go to
  `extra`, which makes load → save → load a fixed point.
* **Source clustering depended on input order.** It seeds greedily from `pool[0]`, and the order a scan
  assembles detections in is not the order they reload from a file — measured on the reference day,
  4 sources recorded against 3 re-derived, including one bogus 65.4 Hz / 555 Hz "temporal" pair that
  existed on only one of the two paths. The pool is now canonically ordered, so all derived views are
  a function of the findings.

### Incremental scanning

```
audioanalyzer scan /data/night-02/ --json out/scan-2026-10-24.json --incremental
```

`--incremental` skips any recording an existing artifact in the output directory already covers —
matched by filename and frame count. **There is no state file**: the artifacts are the memory. A
separate ledger was rejected deliberately, because ledgers rot when files are moved, restored or
re-encoded, and a stale skip is a silent false negative — while an artifact that failed to record a
file is simply an artifact that is not there, which is checkable.

Two rules make that actually work, and the first release got both wrong (symptom: `--incremental`
rescanned all 77 files every time):

* **The file named by `--json` is part of the history.** It must be read *before* it is written.
  Excluding it as "this run's own output" meant the natural `--json out/scan.json` layout could never
  find prior coverage at all.
* **A range run never overwrites the artifact it learned from.** Writing only the new subset to
  `out/scan.json` would delete the record of everything already covered, so the next run would
  re-add it all to a doubled census. Instead each incremental run writes a sibling named by its
  covered span — `out/scan-20261024T0000-20261024T0100.json` — and the command prints the `report`
  line that merges them. `--csv`/`--report` follow the same suffix so a partial run never lands under
  a name that implies completeness.

One JSON per range, never an append to a growing one: a bad run gets deleted, not unpicked out of a
singleton.

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
.venv/bin/python -m pytest tests -q        # 178 tests, ~4 min, no audio files needed
```

Ground truth is injected and recovered: a tone added at 300–900 s comes back dated `0:05:00` for
`600.2 s`; a 7.5 s click train is named at ±0.4 s; clipped samples are excluded from transient
counts on purpose (`tests/test_timebase.py` and friends document why).
