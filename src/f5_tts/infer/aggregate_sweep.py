"""Aggregate ``tto_outputs/*/metrics.summary.txt`` into one Excel-friendly CSV.

One row per TAG directory. Columns expose the config (loss/vad/window/hop/
opt_at/opt_steps/opt_lr) plus per-metric mean/std/min/max — sortable and
filterable in Excel / pandas.

Usage:
    python src/f5_tts/infer/aggregate_sweep.py
    python src/f5_tts/infer/aggregate_sweep.py --root tto_outputs --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


# Matches:  <loss>-<vad>_w<ws>_h<hs>_at<oa>_s<steps>_lr<lr>[_sm<mode>]
# The _sm<mode> suffix is optional — runs from before slide_mode was introduced
# default to "audio" (the legacy implementation). To keep the lr group from
# greedily eating "_smX", we match _lr lazily and consume the optional suffix.
TAG_RE = re.compile(
    r"^(?P<loss>[a-z]+)-(?P<vad>[a-z]+)"
    r"_w(?P<ws>[\d.]+)_h(?P<hs>[\d.]+)"
    r"_at(?P<oa>[\d\-]+)_s(?P<steps>\d+)"
    r"_lr(?P<lr>[\w.\-+]+?)"
    r"(?:_sm(?P<sm>[a-z]+))?$"
)

# Order of metric columns in the CSV (mean / std / min / max all kept).
METRIC_ORDER = (
    "wer", "cer", "utmos", "spk_sim",
    "e2v_dtw_jsd", "e2v_frame_jsd_mean",
    "e2v_label_edit_norm", "e2v_top_label_match",
)


def parse_tag(name: str) -> dict | None:
    m = TAG_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    pts = g["oa"].split("-") if g["oa"] else []
    return {
        "tag": name,
        "loss_mode": g["loss"],
        "vad_level": g["vad"],
        "window_size": float(g["ws"]),
        "hop_size": float(g["hs"]),
        "opt_at": g["oa"],
        "opt_at_npts": len(pts),
        "opt_steps": int(g["steps"]),
        "opt_lr": g["lr"],
        "slide_mode": g.get("sm") or "audio",  # legacy runs default to audio
        "total_iters": len(pts) * int(g["steps"]),
    }


def parse_summary(path: Path) -> dict:
    """Parse the body of metrics.summary.txt — one row per metric.

    Returns a flat dict like {wer_mean, wer_std, wer_min, wer_max, ...,
    n_samples, ref_dir, gen_text}.
    """
    out: dict = {}
    text = path.read_text()
    # n samples / ref_dir / gen_text (header lines)
    m = re.search(r"\(n=(\d+)", text)
    if m: out["n_samples"] = int(m.group(1))
    m = re.search(r"ref_dir\s*:\s*(.+)", text)
    if m: out["ref_dir"] = m.group(1).strip()
    m = re.search(r"gen_text:\s*(.+)", text)
    if m: out["gen_text"] = m.group(1).strip().strip("'\"")
    # metric rows after the dashes
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("----"):
            in_table = True
            continue
        if not in_table or not s:
            continue
        parts = s.split()
        if len(parts) >= 6 and parts[0].replace("_", "").isalnum():
            name = parts[0]
            try:
                mean, std, mn, mx = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            out[f"{name}_mean"] = mean
            out[f"{name}_std"]  = std
            out[f"{name}_min"]  = mn
            out[f"{name}_max"]  = mx
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="tto_outputs",
                    help="dir containing TAG subdirs (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: <root>/_sweep_summary.csv)")
    ap.add_argument("--pattern", default="*",
                    help="glob filter on TAG dir names (default: all)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[aggregate] no such dir: {root}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else (root / "_sweep_summary.csv")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if args.pattern != "*" and not d.match(args.pattern):
            continue
        cfg = parse_tag(d.name)
        if cfg is None:
            skipped.append((d.name, "unparseable TAG"))
            continue
        summary = d / "metrics.summary.txt"
        if not summary.exists():
            skipped.append((d.name, "no metrics.summary.txt"))
            continue
        try:
            metrics = parse_summary(summary)
        except Exception as e:
            skipped.append((d.name, f"parse error: {e}"))
            continue
        rows.append({**cfg, **metrics})

    if not rows:
        print(f"[aggregate] no parseable TAG under {root}", file=sys.stderr)
        return 1

    # Build column order: config → context → mean → std → min → max
    config_cols = [
        "tag", "loss_mode", "vad_level", "slide_mode",
        "window_size", "hop_size",
        "opt_at", "opt_at_npts",
        "opt_steps", "opt_lr", "total_iters",
    ]
    context_cols = ["n_samples", "ref_dir", "gen_text"]
    metric_cols: list[str] = []
    for suffix in ("_mean", "_std", "_min", "_max"):
        metric_cols += [f"{m}{suffix}" for m in METRIC_ORDER]
    cols = config_cols + context_cols + metric_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds BOM so Excel auto-detects encoding for non-ASCII text.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"wrote {out_path}  ({len(rows)} rows, {len(cols)} cols)")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for name, why in skipped:
            print(f"  {name}  ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
