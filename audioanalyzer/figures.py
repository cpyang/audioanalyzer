"""Evidence figures as an artifact of the scan, not a render of the report (FR-8.2c).

The report needs two PNGs per featured detection. Until now it produced them by seeking into the
FLAC at report-build time, which made the report the one output that could not be rebuilt from
artifacts. Caching them at scan time costs about 25 KB per figure and fixes that: the scan that
already read the audio renders the evidence once, and any later `report` — hours, days, or a week
later, on a machine with no recordings — assembles the same picture.

Identity is deliberately NOT the stored path. `Detection.file` holds the path exactly as typed
(`dir/x.flac` vs `x.flac`), so path-keyed evidence would fragment the same recording across two
names and a merge would find no figures for half its own cards. It is
`sha1(basename | frames | key())`: the frame count distinguishes two different recordings that
share a filename, and `key()` (FR-5.3) is already the stable per-detection identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import soundfile as sf

from .config import Config
from .models import Detection

KINDS = ("spectrogram", "waveform")


def identity(d: Detection, frames: int | None = None) -> str:
    """Cache key for one detection's evidence.

    Uses the EXACT merge identity, not `key()`: two impacts in the same minute at the same band
    centre are distinct findings, and a shared key would make one card display the other's crop --
    the same coarseness that cost 47 % of a census in artifact.merge_runs (see its docstring).
    """
    from .artifact import identity as exact
    tag = exact(d, frames)
    return hashlib.sha1("|".join(str(x) for x in tag).encode()).hexdigest()[:16]


def spans(d: Detection) -> tuple[float, float]:
    """(span_s, span_hz) for the crop — the same rule the report has always used, in one place."""
    span_s = 0.2 if d.cls == "A4" else 30.0
    span_hz = 1000.0 if d.cls != "A3" else max(200.0, 0.2 * d.freq_hz)
    return span_s, span_hz


def render(d: Detection, cfg: Config) -> dict[str, bytes]:
    """Render both figures from the audio. Empty dict when the recording cannot be opened.

    Written out longhand on purpose: the two plotters have different signatures (the spectrogram
    also takes a centre frequency), and a loop over them needs a conditional third argument that
    reads like a trap and type-checks like one too.
    No try/except around the renders beyond the missing-file case: R15 exists because a broad
    except at a call site turned a broken figure into an empty string.
    """
    import base64

    from .plots import spectrogram_crop, waveform_zoom
    span_s, span_hz = spans(d)
    out: dict[str, bytes] = {}
    if not Path(str(d.file)).exists():
        return out
    t_mid = d.t_start_s + d.duration_s / 2.0
    spec = spectrogram_crop(str(d.file), t_mid, d.freq_hz, cfg, span_s=span_s, span_hz=span_hz)
    if spec:
        out["spectrogram"] = base64.b64decode(spec)
    t_wave = d.t_start_s if d.cls == "A4" else t_mid
    wave = waveform_zoom(str(d.file), t_wave, cfg, span_s=0.5 if d.cls == "A4" else 1.0)
    if wave:
        out["waveform"] = base64.b64decode(wave)
    return out


def frames_of(d: Detection) -> int | None:
    """Frame count of the source recording, for the identity. Cheap: header only."""
    try:
        return int(sf.info(str(d.file)).frames)
    except Exception:                       # noqa: BLE001 - absent audio is normal for artifacts
        return None


def overview_identity(path: str | Path, frames: int | None) -> str:
    """Cache key for a whole-file overview: the recording's own identity, no detection involved."""
    return hashlib.sha1(f"{Path(str(path)).name}|{frames if frames is not None else 'na'}"
                        .encode()).hexdigest()[:16]


def overview_figure(path: str | Path, cfg: Config, directory: str | Path | None = None,
                    frames: int | None = None) -> bytes | None:
    """The per-file overview spectrogram, from cache if present, from audio if not (FR-8.2c).

    Cached too rather than left as the one figure an artifact-only report cannot draw: it costs one
    PNG per recording (~30 KB), and without it a merged report would look like the overview section
    had gone missing from the tool rather than from the machine.
    """
    stem = overview_identity(path, frames)
    if directory:
        p = Path(directory) / f"{stem}-overview.png"
        if p.exists():
            return p.read_bytes()
    if not Path(str(path)).exists():
        return None
    from .plots import overview_spectrogram
    import base64
    b64 = overview_spectrogram(str(path), cfg)
    if not b64:
        return None
    png = base64.b64decode(b64)
    if directory:
        p = Path(directory) / f"{stem}-overview.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png)
    return png


def write_cache(dets: list[Detection], cfg: Config, directory: str | Path, *,
                top: int = 40, frames: dict[str, int] | None = None) -> dict:
    """Render and store figures for the `top` most confident detections. Returns stats."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ordered = sorted(dets, key=lambda x: (-float(x.confidence or 0.0), x.freq_hz, x.file))[:top]
    written = skipped = 0
    for det in ordered:
        fr = (frames or {}).get(Path(str(det.file)).name) or frames_of(det)
        stem = identity(det, fr)
        got = render(det, cfg)
        for kind, png in got.items():
            p = d / f"{stem}-{kind}.png"
            if not p.exists():
                p.write_bytes(png)
                written += 1
            else:
                skipped += 1
        # A card with no audio reachable writes nothing; `report` will look in the cache by the
        # same identity and warn if the entry is missing. That is why frames must be known from the
        # artifact's media block rather than asked of the file.
    return {"attempted": len(ordered), "written": written, "reused": skipped,
            "dir": str(d), "total_detections": len(dets)}


def load(directory: str | Path, stem: str) -> dict[str, bytes]:
    out = {}
    d = Path(directory)
    for kind in KINDS:
        p = d / f"{stem}-{kind}.png"
        if p.exists():
            out[kind] = p.read_bytes()
    return out
