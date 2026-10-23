# audioanalyzer — abnormal frequency scanner for long recordings

Finds **which frequencies** are abnormal in a recording and **during which time periods**, so an
engineer can walk toward a machine instead of guessing. Built for hourly FLAC survey recordings
(16 kHz mono here), offline, CPU-only, no ffmpeg at runtime.

Design and measured data characteristics: **`REQUIREMENTS.md`** (read §2 first — it explains why
this tool does not use a loudness threshold).

## Install

```bash
cd ~/Audio
uv venv --python 3.12 .venv          # python 3.12: widest wheel coverage (doc §11 R9)
.venv/bin/python -m pip install -e . # or: uv pip install -e .
.venv/bin/audioanalyzer version
```

Audio is never committed: `*.flac` and friends are in `.gitignore`, and the test suite generates
synthetic fixtures calibrated to this corpus's measured floor (`audioanalyzer/synth.py`).

## Use

```bash
# one file: ranked anomalies with frequency, level, time span, and evidence
.venv/bin/audioanalyzer scan 20261023T070002.flac

# a whole day (directory or glob): merged timeline, sources, coverage ledger, recurrence
.venv/bin/audioanalyzer scan . --csv out/detections.csv --json out/scan.json --report out/report.html

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
```

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
.venv/bin/python -m pytest tests -q        # 62 tests, ~45 s, no audio files needed
```

Ground truth is injected and recovered: a tone added at 300–900 s comes back dated `0:05:00` for
`600.2 s`; a 7.5 s click train is named at ±0.4 s; clipped samples are excluded from transient
counts on purpose (`tests/test_timebase.py` and friends document why).
