"""Aggregate & plot budget-sweep results.

Reads ``results.csv`` produced by ``run_budget_sweep.py``, computes per-(config,
total_steps) mean/std for each metric, and draws one figure per metric with
baseline as a horizontal reference line.

Usage:
  python src/f5_tts/infer/aggregate_budget_sweep.py \
      --run-dir experiments/budget_sweep/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("VAD_mse", "VAD-MSE (↓)", "lower is better"),
    ("WER",     "WER (↓)",     "lower is better"),
    ("spkSIM",  "spk-SIM (↑)", "higher is better"),
]

CONFIG_ORDER = ["early", "mid", "late"]
CONFIG_COLORS = {"early": "tab:blue", "mid": "tab:orange", "late": "tab:green"}


def _mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan"), 0
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v), n


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """Returns:
        baseline_stats: dict[metric] -> (mean, std, n)
        config_stats: dict[config][total_steps][metric] -> (mean, std, n)
    """
    by_group = defaultdict(lambda: defaultdict(list))  # (config, total) -> metric -> list
    for r in rows:
        cfg = r["config"]
        total = int(r["total_steps"])
        key = (cfg, total)
        for m, *_ in METRICS:
            try:
                by_group[key][m].append(float(r[m]))
            except (KeyError, ValueError):
                pass

    baseline_stats = {}
    config_stats = defaultdict(lambda: defaultdict(dict))
    for (cfg, total), m_vals in by_group.items():
        for m, *_ in METRICS:
            stats = _mean_std(m_vals.get(m, []))
            if cfg == "baseline":
                baseline_stats[m] = stats
            else:
                config_stats[cfg][total][m] = stats
    return baseline_stats, config_stats


def write_aggregate_csv(out_path, baseline_stats, config_stats):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["config", "total_steps"]
        for m, *_ in METRICS:
            header += [f"{m}_mean", f"{m}_std", f"{m}_n"]
        w.writerow(header)

        row = ["baseline", 0]
        for m, *_ in METRICS:
            mean, std, n = baseline_stats.get(m, (float("nan"), float("nan"), 0))
            row += [f"{mean:.6f}", f"{std:.6f}", n]
        w.writerow(row)

        for cfg in CONFIG_ORDER:
            if cfg not in config_stats:
                continue
            for total in sorted(config_stats[cfg].keys()):
                row = [cfg, total]
                for m, *_ in METRICS:
                    mean, std, n = config_stats[cfg][total].get(
                        m, (float("nan"), float("nan"), 0),
                    )
                    row += [f"{mean:.6f}", f"{std:.6f}", n]
                w.writerow(row)


def plot_metric(out_path, metric_key, metric_label, baseline_stats, config_stats):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # baseline horizontal band
    if metric_key in baseline_stats:
        mean, std, n = baseline_stats[metric_key]
        if not math.isnan(mean):
            ax.axhline(mean, color="black", ls="--", lw=1.2,
                       label=f"baseline (n={n}, μ={mean:.3f})")
            if n >= 2 and std > 0:
                ax.axhspan(mean - std, mean + std, color="black", alpha=0.08)

    for cfg in CONFIG_ORDER:
        if cfg not in config_stats:
            continue
        totals = sorted(config_stats[cfg].keys())
        means, stds, ns = [], [], []
        for t in totals:
            stats = config_stats[cfg][t].get(metric_key)
            if stats is None:
                means.append(float("nan"))
                stds.append(0.0)
                ns.append(0)
                continue
            m, s, n = stats
            means.append(m)
            stds.append(s if n >= 2 else 0.0)
            ns.append(n)
        color = CONFIG_COLORS.get(cfg, None)
        ax.errorbar(totals, means, yerr=stds, marker="o", capsize=3,
                    color=color, label=f"{cfg}")

    ax.set_xlabel("total TTO optimization steps (= level × len(opt_at))")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs. total TTO steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Aggregate & plot budget-sweep results")
    p.add_argument("--run-dir", required=True,
                   help="Directory containing results.csv")
    args = p.parse_args()

    csv_path = os.path.join(args.run_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing: {csv_path}")
    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rows = load_rows(csv_path)
    baseline_stats, config_stats = aggregate(rows)

    agg_csv = os.path.join(args.run_dir, "aggregate.csv")
    write_aggregate_csv(agg_csv, baseline_stats, config_stats)
    print(f"wrote {agg_csv}")

    for m_key, m_label, _ in METRICS:
        out = os.path.join(plots_dir, f"{m_key}.png")
        plot_metric(out, m_key, m_label, baseline_stats, config_stats)
        print(f"wrote {out}")

    # quick terminal summary
    print("\n=== summary (mean ± std, n) ===")
    print(f"{'group':<20} {'total':>5}  " + "  ".join(f"{m:>16}" for m, *_ in METRICS))
    mean, std, n = (baseline_stats.get("VAD_mse", (float("nan"),)*3))
    bl_row = ["baseline", "0"]
    for m, *_ in METRICS:
        mn, sd, nn = baseline_stats.get(m, (float("nan"), float("nan"), 0))
        bl_row.append(f"{mn:.3f}±{sd:.3f}(n={nn})")
    print(f"{bl_row[0]:<20} {bl_row[1]:>5}  " + "  ".join(f"{c:>16}" for c in bl_row[2:]))
    for cfg in CONFIG_ORDER:
        if cfg not in config_stats:
            continue
        for total in sorted(config_stats[cfg].keys()):
            cells = []
            for m, *_ in METRICS:
                mn, sd, nn = config_stats[cfg][total].get(
                    m, (float("nan"), float("nan"), 0),
                )
                cells.append(f"{mn:.3f}±{sd:.3f}(n={nn})")
            print(f"{cfg:<20} {total:>5}  " + "  ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
