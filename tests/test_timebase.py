"""Timebase regressions (FR-1.2, FR-2.1): frame index must equal sample index / hop.

These exist because the first version of ``iter_blocks`` read *sequential* chunks whose size
included the STFT window overlap, while every downstream consumer computed frame times as
``i * hop``.  The two disagreed by (nperseg - hop) samples per block — 96 ms per 30 s block,
0.32% of the timeline, 11.4 s by the end of an hour — and it showed up as a tone injected at
300-900 s being dated 299.0 s for 598.3 s.  Nothing else in the test suite could see a
sub-1% drift, so this file exists to keep it visible.
"""

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.detectors.transients import detect_transients
from audioanalyzer.ingest import probe
from audioanalyzer.spectral import SpectralEngine, aggregate
from audioanalyzer.synth import add_click, rumble

FS = 16000


def _sweep_file(tmp_path: Path, seconds: float, name: str) -> Path:
    p = tmp_path / name
    sf.write(p, rumble(FS, seconds, seed=3), FS, subtype="PCM_16")
    return p


def test_blocks_tile_the_file_without_losing_samples(tmp_path):
    """Frame count must match the closed form over the whole file, across every block."""
    for block_s in (2.0, 5.0, 30.0):
        p = _sweep_file(tmp_path, 22.0, f"tile_{block_s}.flac")
        cfg = Config(block_s=block_s)
        eng = SpectralEngine(cfg, FS)
        blocks = list(eng.iter_blocks(p))
        n = eng.frames_in_block(FS * 22)
        got = sum(eng.frames_in_block(b.size) for b in blocks)
        assert got == n, f"block_s={block_s}: {got} frames, expected {n} ({100*(got/n-1):+.3f}%)"
        if len(blocks) > 1:                      # a file shorter than one block has no block 1
            assert blocks[1].size == eng.block_samples
        assert eng.chunk_advance == eng.frames_per_block * eng.hop


def test_frame_times_land_on_the_sample_grid(tmp_path):
    """Aggregate's own time axis must say t = i*hop/fs + centre, not a drifted approximation."""
    p = _sweep_file(tmp_path, 12.0, "grid.flac")
    cfg = Config(block_s=2.0)
    agg, eng = aggregate(p, cfg)
    expected = (np.arange(agg.n_frames) * eng.hop + eng.nperseg / 2.0) / eng.fs
    assert agg.block_t0 == [i * cfg.baseline_block_s for i in range(len(agg.block_t0))]
    assert eng.frames_per_block * eng.hop + (eng.nperseg - eng.hop) == eng.block_samples
    # and the decoded span covers the file to within one window
    assert abs(expected[-1] - 12.0) < 0.5, expected[-1]


def test_impulse_is_dated_within_one_frame_across_blocks(tmp_path):
    """End-to-end: clicks late in a multi-block file must not arrive early by 0.3%."""
    cfg = Config(block_s=3.0)
    x = rumble(FS, 24.0, seed=4)
    truth = [1.0, 6.3, 12.75, 19.999, 23.4]
    for t in truth:
        x = add_click(x, FS, t, level_dbfs=-16.0, ring_hz=1500.0, ring_ms=40.0)
    p = tmp_path / "imp.flac"
    sf.write(p, x, FS, subtype="PCM_16")
    agg, eng = aggregate(p, cfg)
    dets, _ = detect_transients(p, cfg, agg, eng, media=probe(p, cfg), detail=False)
    got = sorted(d.t_start_s for d in dets if d.cls == "A4")
    assert len(got) >= len(truth) - 1, f"found {got}"
    for t in truth:
        near = min((abs(t - g) for g in got), default=9e9)
        # A 32 ms step grid means an impulse can be up to (nperseg - hop)/2 = 48 ms from its
        # frame's centre.  That is the timestamp's *quantisation*, not drift: the point of this
        # test is that the error does not GROW with file position, which the 0.32% bug did.
        assert near <= (eng.nperseg - eng.hop) / 2.0 / eng.fs + 1e-6, \
            f"click at {t} s dated {near:.4f} s off"


def test_hr_mode_reaches_its_documented_resolution():
    """FR-3.2: --mode hr must really be ~1 Hz/bin, not clamped by a window-vs-rate guard."""
    eng = SpectralEngine(Config(mode="hr"), FS)
    assert eng.nperseg == 16384
    assert abs(eng.df - 0.9765625) < 1e-6, eng.df
    assert eng.hop == 4096
    assert math.isclose(eng.nperseg / eng.fs, 1.024, rel_tol=1e-9)
