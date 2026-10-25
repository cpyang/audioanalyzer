"""Evidence figures must EXIST, not merely not-crash (FR-8.2/8.2c, requirement R15).

The spectrogram crop raised IndexError on every call for the entire history of this project because
`db[fband]` applied a bin mask to axis 0 (frames), and `_card` swallowed the exception into an empty
string. The HTML tests could not have caught it: they assert the ABSENCE of external URLs, which
rewards producing fewer images. So these tests assert presence and geometry.
"""

import base64
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import soundfile as sf

from audioanalyzer.config import Config
from audioanalyzer.models import Detection
from audioanalyzer.plots import crop_grid, spectrogram_crop, waveform_zoom
from audioanalyzer.report_html import _card
from audioanalyzer.synth import add_tone, rumble

FS = 16000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _sine_file(tmp_path, freq=1500.0, level=-45.0, seconds=6.0):
    t = np.arange(int(seconds * FS)) / FS
    x = level_amp(level) * np.sin(2 * np.pi * freq * t)
    p = tmp_path / "sine.flac"
    sf.write(p, x.astype(np.float32), FS, subtype="PCM_16")
    return p


def level_amp(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def test_the_crop_measures_amplitude_honestly(tmp_path):
    """A -45 dBFS sine must read -45 dB at its own frequency. This is the assertion the bug
    violated -- not "an image appeared", which a blank plot would also satisfy."""
    p = _sine_file(tmp_path, 1500.0, -45.0)
    t, fb, z = crop_grid(p, 3.0, 1500.0, Config(), span_s=6.0)
    row = int(np.argmin(np.abs(fb - 1500.0)))
    assert abs(fb[row] - 1500.0) < 5.0
    assert abs(np.median(z[row]) - (-45.0)) < 1.5, np.median(z[row])


def test_the_crop_is_a_grid_of_bins_by_frames_not_frames_by_bins(tmp_path):
    """The bug's shape: a bin mask applied to axis 0. Frames >> bins at any sane span."""
    p = _sine_file(tmp_path, 2000.0, -50.0, seconds=30.0)
    t, fb, z = crop_grid(p, 15.0, 2000.0, Config(), span_s=30.0, span_hz=1000.0)
    assert z.shape == (len(fb), len(t))
    assert len(t) > 4 * len(fb), "a 30 s crop should have far more frames than bins"


def test_a_tone_only_lights_up_where_the_tone_is(tmp_path):
    """Temporal placement, so the report's time column and the picture agree."""
    x = add_tone(rumble(FS, 60.0, seed=11), FS, 1500.0, -45.0, 20.0, 35.0)
    p = tmp_path / "burst.flac"
    sf.write(p, x.astype(np.float32), FS, subtype="PCM_16")
    t, fb, z = crop_grid(p, 27.0, 1500.0, Config(), span_s=30.0)
    row = int(np.argmin(np.abs(fb - 1500.0)))
    during = np.median(z[row, (t > 22) & (t < 33)])
    before = np.median(z[row, t < 17])
    assert during > before + 25, (during, before)
    assert np.median(z[row - 2:row + 3, (t > 22) & (t < 33)].max(axis=0).mean()) >= during


def test_spectrogram_crop_returns_a_real_png(tmp_path):
    p = _sine_file(tmp_path)
    b64 = spectrogram_crop(p, 3.0, 1500.0, Config(), span_s=6.0)
    assert b64, "empty string means the renderer failed and the caller will hide it"
    assert base64.b64decode(b64)[:8] == PNG_MAGIC


def test_a_card_carries_TWO_images(tmp_path):
    """The regression test R15 asks for: before the fix, cards carried one (the waveform)."""
    x = add_tone(rumble(FS, 20.0, seed=5), FS, 1200.0, -42.0, 5.0, 15.0)
    p = tmp_path / "a.flac"
    sf.write(p, x.astype(np.float32), FS, subtype="PCM_16")
    d = Detection(cls="A2", file=str(p), freq_hz=1200.0, t_start_s=5.0, duration_s=10.0,
                  tone_noise_db=12.0, level_dbfs=-42.0, confidence=0.9)
    html = _card(d, Config())
    assert html.count("<img") == 2, f"card has {html.count('<img')} images; expected spectrogram + waveform"
    assert "spectrogram" in html.lower()
    assert not re.search(r"class='warn'>.*(fail|unavailable)", html), "card reports a render failure"


def test_add_tone_refuses_an_empty_window():
    """t1 is an ABSOLUTE end time. Passing a duration as the 6th argument used to return x
    unchanged, so fixtures silently contained no tone at all."""
    x = rumble(FS, 5.0, seed=1)
    with pytest.raises(ValueError, match="absolute end time"):
        add_tone(x, FS, 1000.0, -40.0, 20.0, 15.0)


def test_card_reports_a_missing_figure_with_the_command_to_make_it(tmp_path):
    """R15's second half: an absent image must never look like an unremarkable spectrum."""
    d = Detection(cls="A2", file=str(tmp_path / "gone.flac"), freq_hz=900.0, t_start_s=1.0,
                  duration_s=8.0, tone_noise_db=9.0, level_dbfs=-50.0, confidence=0.5)
    html = _card(d, Config())
    assert html.count("<img") == 0
    assert "class='warn'" in html
    assert "evidence figure(s) missing" in html
    assert "audioanalyzer scan" in html and "--report" in html, "must name the command that fixes it"
