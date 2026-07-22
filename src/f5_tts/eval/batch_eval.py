"""Batch-eval a directory of TTO-generated wavs against paired ref wavs.

Two modes:

* **flat mode** (legacy, ``--ref-dir`` + ``--gen-text``): pairs
  ``gen_dir/{stem}.wav`` with ``ref_dir/{stem}.wav`` by basename and uses a
  single ``--gen-text`` for every sample. Suitable when all stems share the
  same generated text.
* **manifest mode** (``--manifest``): each row in the JSONL manifest carries
  its own ``ref_wav`` / ``gen_text`` (and optional ``ref_text``), so different
  stems can be evaluated against different reference audio and target texts.
  Built for the ESD emo-change eval (``build_emochange_eval.py``).

In both modes ``eval_metric.evaluate_all`` runs per pair, writing one CSV row
per sample plus a ``.summary.txt`` aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from f5_tts.eval.eval_metric import evaluate_all


SCALAR_COLS = ["wer", "cer", "utmos", "spk_sim",
               "EMO-sim_utt", "EMO-sim_frame",
               "av_sim_utt", "av_sim_chunk", "pcp_score"]
# Lists are serialized to ';'-joined strings before write (see _csv_serialize).
STR_COLS = ["hyp"]


def _csv_serialize(v):
    """Flatten list-valued cells to a ';'-joined string for clean CSVs."""
    if isinstance(v, list):
        if v and isinstance(v[0], float):
            return ";".join(f"{x:.4f}" for x in v)
        return ";".join(str(x) for x in v)
    return v


def _load_manifest(path: Path, gen_dir: Path) -> list[dict]:
    """Load manifest JSONL into per-stem eval jobs.

    Each row must carry ``stem`` and ``gen_text``; ``ref_wav`` is required for
    manifest mode (paths are resolved against the manifest's parent dir if
    they're not absolute). Skips rows whose ``gen_dir/{stem}.wav`` is missing.
    """
    jobs: list[dict] = []
    base = path.parent.resolve()
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k in ("stem", "gen_text", "ref_wav"):
                if k not in row:
                    raise ValueError(f"manifest row {i} missing '{k}': {row}")
            ref_p = Path(row["ref_wav"])
            if not ref_p.is_absolute():
                ref_p = (base / ref_p).resolve()
            row["ref_wav_resolved"] = str(ref_p)
            row["gen_wav_resolved"] = str(gen_dir / f"{row['stem']}.wav")
            jobs.append(row)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch TTO eval (ref vs gen pairs)")
    ap.add_argument("--gen-dir", required=True, help="dir with generated wavs")
    ap.add_argument("--ref-dir", help="(flat mode) dir with reference wavs (basename match)")
    ap.add_argument("--gen-text", help="(flat mode) intended gen text (same for all samples)")
    ap.add_argument("--manifest", help="(manifest mode) JSONL with per-row ref_wav/gen_text")
    ap.add_argument("--out-csv", required=True, help="output CSV path")
    ap.add_argument("--device", default="cuda",
                    help="device for eval models (cuda or cpu)")
    ap.add_argument("--pattern", default="*.wav")
    args = ap.parse_args()

    if (args.manifest is None) == (args.ref_dir is None):
        print("[batch_eval] must pass exactly one of --manifest or --ref-dir",
              file=sys.stderr)
        return 2
    if args.ref_dir is not None and args.gen_text is None:
        print("[batch_eval] --gen-text is required in flat (--ref-dir) mode",
              file=sys.stderr)
        return 2

    gen_dir = Path(args.gen_dir)

    if args.manifest is not None:
        jobs = _load_manifest(Path(args.manifest), gen_dir)
        ref_label = args.manifest
    else:
        gen_paths = sorted(gen_dir.glob(args.pattern))
        if not gen_paths:
            print(f"[batch_eval] no wav in {gen_dir}", file=sys.stderr)
            return 1
        jobs = [{
            "stem": gp.stem,
            "gen_text": args.gen_text,
            "ref_wav_resolved": str(Path(args.ref_dir) / gp.name),
            "gen_wav_resolved": str(gp),
        } for gp in gen_paths]
        ref_label = args.ref_dir

    rows: list[dict] = []
    skipped: list[str] = []
    t_start = time.time()
    for i, job in enumerate(jobs, 1):
        stem = job["stem"]
        gp = Path(job["gen_wav_resolved"])
        rp = Path(job["ref_wav_resolved"])
        if not gp.exists():
            skipped.append(f"{stem} (gen missing)")
            print(f"[{i:02d}/{len(jobs)}] SKIP {stem} (no gen wav)")
            continue
        if not rp.exists():
            skipped.append(f"{stem} (ref missing)")
            print(f"[{i:02d}/{len(jobs)}] SKIP {stem} (no ref wav)")
            continue
        t0 = time.time()
        r = evaluate_all(
            gen_wav=str(gp), ref_wav=str(rp),
            gen_text=job["gen_text"], device=args.device,
        )
        r["stem"] = stem
        r["gen_text"] = job["gen_text"]
        if "ref_text" in job:
            r["ref_text"] = job["ref_text"]
        if "emo_pair" in job:
            r["emo_pair"] = "->".join(job["emo_pair"])
        if "speaker" in job:
            r["speaker"] = job["speaker"]
        rows.append(r)
        dt = time.time() - t0
        text_metric = "wer" if "wer" in r else "cer"
        print(f"[{i:02d}/{len(jobs)}] {stem:30s}  "
              f"{text_metric}={r[text_metric]:.3f}  utmos={r['utmos']:.2f}  "
              f"spk={r['spk_sim']:.3f}  emo_utt={r['EMO-sim_utt']:.3f}  "
              f"emo_frm={r['EMO-sim_frame']:.3f}  "
              f"av_utt={r['av_sim_utt']:.3f}  av_chk={r['av_sim_chunk']:.3f}  "
              f"pcp={r['pcp_score']:.2f}  [{dt:.1f}s]")

    if not rows:
        print("[batch_eval] no pairs evaluated", file=sys.stderr)
        return 1

    # --- write per-sample CSV ---
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # In manifest mode include per-row context columns; harmless empties in flat.
    extra_cols = ["speaker", "emo_pair", "gen_text", "ref_text"]
    cols = ["stem"] + SCALAR_COLS + STR_COLS + extra_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _csv_serialize(r.get(c, "")) for c in cols})

    # --- aggregate summary ---
    def _stats(key):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and r.get(key) is not None]
        if not vals:
            return None
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return m, sd, min(vals), max(vals), len(vals)

    summary_path = out_csv.with_suffix(".summary.txt")
    lines = []
    lines.append(f"eval summary for {args.gen_dir} (n={len(rows)}, "
                 f"skipped={len(skipped)}, wall={time.time()-t_start:.1f}s)")
    lines.append(f"ref     : {ref_label}")
    if args.gen_text is not None:
        lines.append(f"gen_text: {args.gen_text!r}")
    lines.append("")
    header = f"{'metric':<18} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}  {'n':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in SCALAR_COLS:
        s = _stats(k)
        if s is None:
            continue
        m, sd, mn, mx, n = s
        lines.append(f"{k:<18} {m:>10.4f} {sd:>10.4f} {mn:>10.4f} {mx:>10.4f}  {n:>4}")
    if skipped:
        lines.append("")
        lines.append(f"skipped (no ref): {', '.join(skipped)}")
    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary)

    print(f"\nCSV     : {out_csv}")
    print(f"Summary : {summary_path}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
