"""Self-contained HTML report (REQUIREMENTS.md FR-8.2, FR-8.5).

Everything is inline — CSS, JS, base64 PNGs — so one file can be mailed, dropped in a ticket, or
opened from a USB stick at a site with no network. The alternative (a report plus a directory of
figures) reliably loses half its evidence.

The table is sortable because the first question asked of any of these reports is "which of
these matters most", and that question has a different answer by level, by persistence, or by
how weird the frequency is.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import __version__
from .config import Config
from .models import Detection
from .plots import overview_spectrogram, spectrogram_crop, waveform_zoom

_CSS = """
body{background:#0d1116;color:#c9d1d9;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;color:#8b949e;font-weight:600}
.meta{color:#8b949e;font-size:12px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid #21262d;text-align:right;white-space:nowrap}
th{background:#161b22;position:sticky;top:0;cursor:pointer;user-select:none}
th:hover{color:#58a6ff} td.l,th.l{text-align:left}
tr:hover{background:#161b22}
.pill{padding:1px 6px;border-radius:8px;font-size:11px;background:#21262d;color:#c9d1d9}
.A1{background:#1f6feb33}.A2{background:#9e6a0333}.A3{background:#23863633}.A4{background:#da3633aa;color:#fff}
.A5{background:#6639ba33}.A6{background:#6e768133}
.card{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:12px;margin:10px 0}
.card img{display:block;width:100%;height:auto;border-radius:4px;margin:6px 0}
.grid{display:grid;grid-template-columns:1fr;gap:4px}
.warn{color:#f0883e} .ok{color:#3fb950}
pre{background:#161b22;padding:10px;border-radius:6px;overflow:auto;font-size:11px}
"""

_JS = """
function sortT(th){
  const t=th.closest('table'), i=th.cellIndex, num=!isNaN(parseFloat(t.tBodies[0].rows[0]?.cells[i]?.innerText));
  const rows=[...t.tBodies[0].rows];
  const dir=th.dataset.dir==='d'?1:-1; th.dataset.dir=dir<0?'d':'a';
  rows.sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText;
    return num?(parseFloat(x)-parseFloat(y))*dir:x.localeCompare(y)*dir;});
  t.tBodies[0].append(...rows);
}
document.querySelectorAll('th').forEach(h=>h.onclick=()=>sortT(h));
"""


def build_report(files: list[str], dets: list[Detection], sources: list, coverage: dict,
                 cfg: Config, out_path: str | Path, top_figures: int = 12,
                 media: dict | None = None, timeline: list | None = None) -> Path:
    """Write one self-contained HTML report and return its path (FR-8.2)."""
    out_path = Path(out_path)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>Audio anomaly report — {len(files)} recording(s)</title>",
             f"<style>{_CSS}</style></head><body>",
             f"<h1>Audio anomaly report</h1>",
             f"<div class='meta'>{len(files)} recording(s) · "
             f"{_h(coverage.get('first', ''))} → {_h(coverage.get('last', ''))} · "
             f"audioanalyzer {__version__} · mode {cfg.mode} · "
             f"{'dBFS (uncalibrated — absolute SPL requires dbfs_offset)' if cfg.dbfs_offset is None else 'dB re calibrated SPL'}</div>"]

    parts.append("<h2>Coverage</h2>")
    parts.append(_coverage_block(files, coverage, media or {}, cfg))

    parts.append(f"<h2>Detections ({len(dets)})</h2>")
    parts.append(_table(dets))

    if sources:
        parts.append(f"<h2>Candidate sources ({len(sources)})</h2>")
        parts.append(_source_table(sources))

    if timeline:
        cross = [e for e in timeline if e["n_detections"] > 1 and e["class"] in ("A1", "A2")]
        parts.append(f"<h2>Corpus timeline ({len(timeline)} feature(s), "
                     f"{len(cross)} continuing across a chunk boundary)</h2>")
        parts.append(
            "<p class='meta'>A tone that runs across an hour seam is recorded in two files. "
            "Per-file rows stay per-file (that is what joins against a BMS log); this view says "
            "how many <i>things</i> there were.</p>")
        rows = "".join(
            f"<tr><td class='l'>{_h(e['id'])}</td><td class='l'><span class='pill {e['class']}'>"
            f"{e['class']}</span></td><td>{e['freq_hz']:.2f}</td>"
            f"<td class='l'>{_h(e['t_start_wall'])}</td><td class='l'>{_h(e['t_end_wall'])}</td>"
            f"<td>{e['span_s']:.1f}</td><td>{e['active_s']:.1f}</td><td>{e['n_files']}</td>"
            f"<td>{'yes' if e['straddles_boundary'] else '—'}</td>"
            f"<td class='l'>{_h(', '.join(e['files'][:4]))}</td></tr>"
            for e in sorted(timeline, key=lambda x: -x["n_detections"])[:200])
        parts.append("<table><thead><tr>" + "".join(
            f"<th class='{c}'>{n}</th>" for n, c in
            [("id", "l"), ("class", "l"), ("Hz", ""), ("from", "l"), ("to", "l"), ("span s", ""),
             ("active s", ""), ("files", ""), ("straddles", ""), ("member files", "l")])
            + f"</tr></thead><tbody>{rows}</tbody></table>")

    ordered = sorted(dets, key=lambda d: (-d.confidence, d.freq_hz))[:top_figures]
    if ordered:
        parts.append(f"<h2>Evidence — top {len(ordered)} by confidence</h2>")
        for d in ordered:
            parts.append(_card(d, cfg))

    if files and len(files) <= 3:
        parts.append("<h2>Overview spectrograms</h2>")
        for f in files:
            try:
                png = overview_spectrogram(f, cfg)
            except Exception as exc:                       # noqa: BLE001 - report must still exist
                png = ""
                parts.append(f"<p class='warn'>overview failed for {_h(f)}: {_h(str(exc))}</p>")
            if png:
                parts.append(f"<div class='card'><b>{_h(Path(f).name)}</b>"
                             f"<img src='data:image/png;base64,{png}'></div>")

    parts.append("<h2>Configuration and thresholds</h2>")
    parts.append(f"<pre>{_h(json.dumps(cfg.to_dict(), indent=2, default=str))}</pre>")
    parts.append(f"<script>{_JS}</script></body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def _coverage_block(files, coverage: dict, media: dict, cfg: Config) -> str:
    if coverage.get("coverage_pct") is None:
        return ("<p class='warn'>No wall-clock coverage: files are not named "
                "YYYYMMDDTHHMMSS, so gaps between chunks cannot be computed.</p>")
    cls = "ok" if coverage["coverage_pct"] > 99.9 else "warn"
    rows = "".join(
        f"<tr><td class='l'>{_h(g['after'])}</td><td>{g['gap_s']:.2f}</td>"
        f"<td class='l'>{_h(g['at'])}</td></tr>" for g in coverage.get("gaps", [])[:40])
    return (f"<p class='{cls}'>{coverage['audio_s']:,.1f} s of audio inside "
            f"{coverage['wall_s']:,.1f} s of wall clock = <b>{coverage['coverage_pct']:.3f}%</b>; "
            f"<b>{coverage['missing_s']:,.1f} s were never recorded</b> across "
            f"{coverage['n_gaps']} chunk boundary gap(s), largest {coverage['max_gap_s']:.1f} s. "
            f"Absence of a finding inside a gap means nothing.</p>"
            f"<table><thead><tr><th class='l'>after file</th><th>gap s</th>"
            f"<th class='l'>at</th></tr></thead><tbody>{rows}</tbody></table>")


_COLS = [("class", "l"), ("freq_hz", ""), ("ttn", ""), ("coh", ""), ("iso", ""),
         ("level", ""), ("start", "l"), ("dur", ""), ("duty", ""), ("eps", ""),
         ("drift", "l"), ("file", "l"), ("conf", "")]


def _table(dets: list[Detection]) -> str:
    body = []
    for d in sorted(dets, key=lambda d: (-d.confidence, d.freq_hz)):
        e = d.extra
        drift = e.get("kind") or d.drift_class
        body.append(
            f"<tr><td class='l'><span class='pill {d.cls}'>{d.cls}</span></td>"
            f"<td>{d.freq_hz:.2f}</td><td>{d.tone_noise_db:.1f}</td>"
            f"<td>{d.coherence:.2f}</td><td>{getattr(d, 'isolation_db', 0):.1f}</td>"
            f"<td>{d.level_dbfs:.1f}</td>"
            f"<td class='l'>{_h(d.t_start_wall or _t_hms(d.t_start_s))}</td>"
            f"<td>{d.duration_s:.1f}</td><td>{d.duty_pct:.1f}</td><td>{d.n_episodes}</td>"
            f"<td class='l'>{_h(str(drift))}</td><td class='l'>{_h(Path(d.file).name)}</td>"
            f"<td>{d.confidence:.2f}</td></tr>")
    head = "".join(f"<th class='{c}'>{n}</th>" for n, c in _COLS)
    return (f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def _source_table(sources: list) -> str:
    rows = []
    for s in sorted(sources, key=lambda s: (-s.n_harmonics, s.f0_hz))[:60]:
        members = ", ".join(f"{m.freq_hz:.1f}" for m in sorted(s.members, key=lambda m: m.freq_hz)[:10])
        rows.append(f"<tr><td class='l'>{_h(s.source_id)}</td><td>{s.f0_hz:.2f}</td>"
                    f"<td>{s.f0_uncertainty_hz:.2f}</td><td>{s.n_harmonics}</td>"
                    f"<td>{s.duty_pct:.1f}</td><td>{s.n_episodes}</td>"
                    f"<td class='l'>{_h(s.link_reason)}</td><td class='l'>{_h(members)}</td></tr>")
    head = "".join(f"<th class='{'l' if c == 'l' else ''}'>{n}</th>" for n, c in
                   [("id", "l"), ("f0 Hz", ""), ("±Hz", ""), ("harmonics", ""), ("duty %", ""),
                    ("episodes", ""), ("link", "l"), ("members Hz", "l")])
    return (f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _card(d: Detection, cfg: Config) -> str:
    span_s = 0.2 if d.cls == "A4" else 30.0
    span_hz = 1000.0 if d.cls != "A3" else max(200.0, 0.2 * d.freq_hz)
    t_c = d.t_start_s + d.duration_s / 2.0
    try:
        crop = spectrogram_crop(d.file, t_c, d.freq_hz, cfg, span_s=span_s, span_hz=span_hz)
    except Exception as exc:                               # noqa: BLE001
        crop = ""
    try:
        wave = waveform_zoom(d.file, d.t_start_s + (d.duration_s / 2.0 if d.cls != "A4" else 0.0),
                             cfg, span_s=0.5 if d.cls == "A4" else 1.0)
    except Exception:                                      # noqa: BLE001
        wave = ""
    meta = (f"<b class='pill {d.cls}'>{d.cls}</b> {d.freq_hz:.2f} Hz — tone-to-noise "
            f"{d.tone_noise_db:.1f} dB, coherence {d.coherence:.2f}, level {d.level_dbfs:.1f} dBFS, "
            f"t={_t_hms(d.t_start_s)} for {d.duration_s:.1f} s, {d.n_episodes} episode(s) "
            f"in {_h(Path(d.file).name)}")
    sig = d.extra.get("signature")
    if sig:
        meta += "<br><span class='meta'>rang at " + ", ".join(
            f"{s['freq_hz']:.0f} Hz (+{s['rise_db']:.0f} dB)" for s in sig) + "</span>"
    return (f"<div class='card'>{meta}"
            + (f"<img src='data:image/png;base64,{crop}'>" if crop else "")
            + (f"<img src='data:image/png;base64,{wave}'>" if wave else "") + "</div>")


def _h(s) -> str:
    return html.escape(str(s), quote=True)


def _t_hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
