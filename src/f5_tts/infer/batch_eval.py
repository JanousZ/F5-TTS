"""Batch-eval a directory of TTO-generated wavs against paired ref wavs.

Pairs ``gen_dir/{stem}.wav`` with ``ref_dir/{stem}.wav`` by basename, runs
``eval_metric.evaluate_all`` on each pair, writes one CSV row per sample plus a
``.summary.txt`` with mean / std / min / max of the scalar metrics.

Intended to run right after ``tto.py`` batch generation — refs and gens share
the same stem because tto.py names outputs after the ref file it sampled.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from f5_tts.infer.eval_metric import evaluate_all


SCALAR_COLS = ["wer", "cer", "utmos", "spk_sim",
               "e2v_dtw_jsd", "e2v_frame_jsd_mean",
               "e2v_label_edit_norm", "e2v_top_label_match"]
# Lists are serialized to ';'-joined strings before write (see _csv_serialize).
STR_COLS = ["e2v_gen_label_seq", "e2v_ref_label_seq",
            "e2v_gen_probs_mean", "e2v_ref_probs_mean", "hyp"]


def _csv_serialize(v):
    """Flatten list-valued cells to a ';'-joined string for clean CSVs."""
    if isinstance(v, list):
        if v and isinstance(v[0], float):
            return ";".join(f"{x:.4f}" for x in v)
        return ";".join(str(x) for x in v)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch TTO eval (ref vs gen pairs)")
    ap.add_argument("--gen-dir", required=True, help="dir with generated wavs")
    ap.add_argument("--ref-dir", required=True, help="dir with reference wavs (basename match)")
    ap.add_argument("--gen-text", required=True, help="intended gen text (same for all samples)")
    ap.add_argument("--out-csv", required=True, help="output CSV path")
    ap.add_argument("--device", default="cuda",
                    help="device for eval models (cuda or cpu)")
    ap.add_argument("--pattern", default="*.wav")
    args = ap.parse_args()

    gen_paths = sorted(Path(args.gen_dir).glob(args.pattern))
    if not gen_paths:
        print(f"[batch_eval] no wav in {args.gen_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    skipped: list[str] = []
    t_start = time.time()
    for i, gp in enumerate(gen_paths, 1):
        rp = Path(args.ref_dir) / gp.name
        if not rp.exists():
            skipped.append(gp.name)
            print(f"[{i:02d}/{len(gen_paths)}] SKIP {gp.name} (no matching ref)")
            continue
        t0 = time.time()
        r = evaluate_all(
            gen_wav=str(gp), ref_wav=str(rp),
            gen_text=args.gen_text, device=args.device,
        )
        r["stem"] = gp.stem
        rows.append(r)
        dt = time.time() - t0
        # Pick whichever of wer/cer was returned for this sample.
        text_metric = "wer" if "wer" in r else "cer"
        print(f"[{i:02d}/{len(gen_paths)}] {gp.stem:30s}  "
              f"{text_metric}={r[text_metric]:.3f}  utmos={r['utmos']:.2f}  "
              f"spk={r['spk_sim']:.3f}  e2v_dtw={r['e2v_dtw_jsd']:.3f}  "
              f"[{dt:.1f}s]")

    if not rows:
        print("[batch_eval] no pairs evaluated", file=sys.stderr)
        return 1

    # --- write per-sample CSV ---
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem"] + SCALAR_COLS + STR_COLS
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
    lines.append(f"ref_dir : {args.ref_dir}")
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
