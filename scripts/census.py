"""Whole-day census from out/detections.csv — the numbers for REQUIREMENTS.md §2.3.

Reads the CSV/JSON the CLI wrote (no re-analysis), and emits Markdown, so the figures quoted in
the document are the artifacts themselves rather than a parallel implementation of them.  Run:

    .venv/bin/python scripts/census.py out/detections.csv out/scan.json
"""
import collections
import csv
import json
import sys
from pathlib import Path

CSV = sys.argv[1] if len(sys.argv) > 1 else "out/detections.csv"
JSON = sys.argv[2] if len(sys.argv) > 2 else "out/scan.json"

rows = list(csv.DictReader(open(CSV)))
doc = json.load(open(JSON))
by = collections.Counter(r["class"] for r in rows)
files = sorted({Path(r["file"]).name for r in rows})

print(f"rows {len(rows)} | by class {dict(by)} | files with findings {len(files)}")

BAND = lambda x: ("<20" if x < 20 else "20-100" if x < 100 else "0.1-0.5k" if x < 500
                  else "0.5-2k" if x < 2000 else "2-6k" if x < 6000 else "6k+")

print("\n== per class ==")
for cls in ("A1", "A2", "A3", "A4"):
    sel = [r for r in rows if r["class"] == cls]
    if not sel:
        print(f"{cls}: none")
        continue
    f = [float(r["freq_hz"]) for r in sel]
    lv = [float(r["level_dbfs"]) for r in sel]
    du = [float(r["duration_s"]) for r in sel]
    bands = collections.Counter(BAND(x) for x in f)
    print(f"{cls}: n={len(sel)} freq {min(f):.1f}..{max(f):.1f} Hz level {min(lv):.1f}.."
          f"{max(lv):.1f} dBFS dur {min(du):.2f}..{max(du):.1f} s bands {dict(bands)}")
    if cls in ("A1", "A2"):
        coh = sorted(float(r["coherence"]) for r in sel)
        iso = sorted(float(r["isolation_db"]) for r in sel)
        print(f"    coherence {coh[0]:.2f}..{coh[-1]:.2f} (med {coh[len(coh)//2]:.2f})"
              f" isolation {iso[0]:.1f}..{iso[-1]:.1f} dB"
              f" steady={sum(1 for r in sel if float(r['duty_pct'])>90)}"
              f" intermittent={sum(1 for r in sel if float(r['duty_pct'])<=90)}")
        top = sorted(sel, key=lambda r: -float(r["tone_noise_db"]))[:6]
        for r in top:
            print(f"    * {r['class']} {float(r['freq_hz']):8.2f} Hz ttn {float(r['tone_noise_db']):5.1f}"
                  f" coh {float(r['coherence']):.2f} lvl {float(r['level_dbfs']):6.1f}"
                  f" duty {float(r['duty_pct']):5.1f}% eps {r['n_episodes']:>3}"
                  f" files {len({Path(x['file']).name for x in sel if abs(float(x['freq_hz'])-float(r['freq_hz']))<3})}")
    if cls == "A4":
        z = sorted(float(r["tone_noise_db"]) for r in sel)
        kinds = collections.Counter("broadband" if float(r["persistence_pct"]) >= 100 else "band"
                                    for r in sel)
        per_hour = collections.Counter(Path(r["file"]).name for r in sel)
        print(f"    z {z[0]:.1f}..{z[-1]:.1f} (med {z[len(z)//2]:.1f}) kinds {dict(kinds)}")
        print(f"    events/hour min {min(per_hour.values())} med "
              f"{sorted(per_hour.values())[len(per_hour)//2]} max {max(per_hour.values())}")
    if cls == "A3":
        for r in sel[:6]:
            print(f"    * period from JSON (see recurrence/timeline sections)")

src = [r for r in rows if r["source_id"]]
ids = collections.Counter(r["source_id"] for r in src)
print(f"\n== A5: {len(ids)} source(s) over {len(src)} tone track(s) ==")
for sid, n in ids.most_common(10):
    r0 = next(r for r in src if r["source_id"] == sid)
    mem = sorted({round(float(r['freq_hz']), 1) for r in src if r["source_id"] == sid})
    print(f"   {sid} f0 {float(r0['source_f0_hz']):8.2f} Hz members {n:3d} "
          f"distinct {len(mem):3d} {mem[:6]}")

tl = doc.get("timeline", [])
cross = [e for e in tl if e["n_detections"] > 1 and e["class"] in ("A1", "A2")]
print(f"\n== timeline: {len(tl)} feature(s); {len(cross)} continue across a chunk boundary ==")
for e in sorted(cross, key=lambda x: -x["n_detections"])[:8]:
    print(f"   {e['id']} {e['class']} {e['freq_hz']:8.2f} Hz  {e['t_start_wall'][5:16]} -> "
          f"{e['t_end_wall'][11:19]}  active {e['active_s']:7.1f} s in {e['n_files']} file(s)")

rec = doc.get("recurrence", {})
print(f"\n== recurrence: {len(rec.get('clusters', []))} cluster(s) over "
      f"{len(rec.get('days', []))} day(s); note: {rec.get('note','') or 'none'} ==")
for c in rec.get("clusters", [])[:6]:
    print(f"   {c['freq_hz']:8.2f} Hz eps {c['n_episodes']:3d} files {c['n_files'] if 'n_files' in c else c['files']:3d}"
          f" median gap {c['median_gap_s']} s phase {c['phase_of_day_hour']:.2f} h "
          f"(spread {c['phase_spread_h']:.2f} h, conc {c['phase_concentration']:.2f})")

cov = doc["coverage"]
print(f"\n== coverage: {cov['coverage_pct']}% of wall; {cov['missing_s']} s missing over "
      f"{cov['n_gaps']} gaps (max {cov['max_gap_s']} s) ==")
n_by_hour = collections.Counter()
for f in doc["files"]:
    n_by_hour[Path(f["media"]["path"]).name[9:11]] = sum(1 for d in f["detections"])
print("   detections per hour:", dict(sorted(n_by_hour.items())))
elapsed = [f.get("elapsed_s") or f["analysis"].get("elapsed_s") for f in doc["files"]]
elapsed = [e for e in elapsed if e]
if elapsed:
    print(f"   per-file wall: {min(elapsed):.1f}-{max(elapsed):.1f} s (sum {sum(elapsed):.0f} s)")
print("   candidates per file:", [f["analysis"].get("candidates")
                                  or f["analysis"].get("analysis", {}).get("candidates")
                                  for f in doc["files"]][:24])
