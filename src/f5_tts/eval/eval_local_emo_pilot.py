"""Evaluate baseline vs local target-span emotion-control outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio

from f5_tts.infer.target_span import crop_waveform
from f5_tts.infer.emotion_vad_target import emotion_vad_target
from f5_tts.infer.vad_loss import GradVADExtractor, precompute_reference_vad


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("stem", "target_audio", "gen_text", "target_span_sec"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing {key}")
            if row["target_span_sec"] is None:
                raise ValueError(f"{path}:{line_no} target_span_sec is empty")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def scalar_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((x.float() - y.float()) ** 2).detach().cpu())


def evaluate_row(
    row: dict,
    *,
    baseline_dir: Path,
    tto_dir: Path,
    vad: GradVADExtractor,
    device: torch.device,
    context_sec: float,
) -> dict:
    stem = row["stem"]
    baseline_path = baseline_dir / f"{stem}.wav"
    tto_path = tto_dir / f"{stem}.wav"
    target_path = Path(row["target_audio"])
    span = tuple(float(x) for x in row["target_span_sec"])

    baseline, baseline_sr = load_mono(baseline_path)
    tto, tto_sr = load_mono(tto_path)
    target, target_sr = load_mono(target_path)
    baseline = baseline.to(device)
    tto = tto.to(device)
    target = target.to(device)

    if row.get("vad_target_mode") == "emotion_proto":
        target_vad = emotion_vad_target(
            row["emotion"],
            alpha=float(row.get("emotion_vad_alpha", 1.0)),
            device=device,
            dtype=next(vad.parameters()).dtype,
        )
    else:
        target_vad = precompute_reference_vad(
            vad, target.squeeze(0), in_sr=target_sr, embeddings=False, utter=True,
        ).squeeze(0)
    baseline_crop = crop_waveform(
        baseline, span, sample_rate=baseline_sr, context_sec=context_sec,
    )
    tto_crop = crop_waveform(
        tto, span, sample_rate=tto_sr, context_sec=context_sec,
    )
    baseline_target_vad = vad(
        baseline_crop.squeeze(0), in_sr=baseline_sr, embeddings=False, utter=True,
    ).squeeze(0)
    tto_target_vad = vad(
        tto_crop.squeeze(0), in_sr=tto_sr, embeddings=False, utter=True,
    ).squeeze(0)
    baseline_utter_vad = vad(
        baseline.squeeze(0), in_sr=baseline_sr, embeddings=False, utter=True,
    ).squeeze(0)
    tto_utter_vad = vad(
        tto.squeeze(0), in_sr=tto_sr, embeddings=False, utter=True,
    ).squeeze(0)

    base_mse = scalar_mse(baseline_target_vad, target_vad)
    tto_mse = scalar_mse(tto_target_vad, target_vad)
    improvement = (base_mse - tto_mse) / max(base_mse, 1e-12)

    base_np = baseline.detach().float().cpu().numpy()
    tto_np = tto.detach().float().cpu().numpy()
    return {
        "stem": stem,
        "emotion": row.get("emotion", ""),
        "target_word": row.get("target_word", ""),
        "target_span_start_sec": span[0],
        "target_span_end_sec": span[1],
        "target_alignment_score": row.get("target_alignment_score", ""),
        "target_vad_a": float(target_vad[0].cpu()),
        "target_vad_d": float(target_vad[1].cpu()),
        "target_vad_v": float(target_vad[2].cpu()),
        "baseline_target_mse": base_mse,
        "tto_target_mse": tto_mse,
        "target_mse_improvement": improvement,
        "baseline_vad_a": float(baseline_target_vad[0].cpu()),
        "baseline_vad_d": float(baseline_target_vad[1].cpu()),
        "baseline_vad_v": float(baseline_target_vad[2].cpu()),
        "tto_vad_a": float(tto_target_vad[0].cpu()),
        "tto_vad_d": float(tto_target_vad[1].cpu()),
        "tto_vad_v": float(tto_target_vad[2].cpu()),
        "baseline_global_vad_mse": scalar_mse(baseline_utter_vad, target_vad),
        "tto_global_vad_mse": scalar_mse(tto_utter_vad, target_vad),
        "baseline_duration_sec": baseline.shape[-1] / baseline_sr,
        "tto_duration_sec": tto.shape[-1] / tto_sr,
        "baseline_rms": float(np.sqrt(np.mean(base_np ** 2))),
        "tto_rms": float(np.sqrt(np.mean(tto_np ** 2))),
        "baseline_peak": float(np.max(np.abs(base_np))),
        "tto_peak": float(np.max(np.abs(tto_np))),
        "baseline_finite": bool(np.isfinite(base_np).all()),
        "tto_finite": bool(np.isfinite(tto_np).all()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summary_text(rows: list[dict], *, context_sec: float) -> str:
    base = mean(rows, "baseline_target_mse")
    tto = mean(rows, "tto_target_mse")
    rel = (base - tto) / max(base, 1e-12)
    improved = sum(float(row["tto_target_mse"]) < float(row["baseline_target_mse"]) for row in rows)
    finite = all(row["baseline_finite"] and row["tto_finite"] for row in rows)
    lines = [
        "Local target-span emotion pilot summary",
        "",
        f"n={len(rows)}",
        f"target_context_sec={context_sec}",
        "target span source=baseline audio + torchaudio MMS_FA forced alignment",
        "",
        f"mean baseline target-span VAD MSE={base:.6f}",
        f"mean TTO target-span VAD MSE={tto:.6f}",
        f"relative MSE improvement={rel:.2%}",
        f"samples improved={improved}/{len(rows)}",
        f"mean baseline global VAD MSE={mean(rows, 'baseline_global_vad_mse'):.6f}",
        f"mean TTO global VAD MSE={mean(rows, 'tto_global_vad_mse'):.6f}",
        f"mean baseline RMS={mean(rows, 'baseline_rms'):.6f}",
        f"mean TTO RMS={mean(rows, 'tto_rms'):.6f}",
        f"all outputs finite={finite}",
        "",
        "Interpretation: target-span VAD is an automatic proxy for the optimization",
        "objective. It is not a substitute for human judgment of local emotion",
        "strength, content preservation, or perceptual quality.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--tto-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-sec", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    device = torch.device(args.device)
    vad = GradVADExtractor(in_sr=24000, device=device)
    with torch.no_grad():
        metrics = [
            evaluate_row(
                row,
                baseline_dir=Path(args.baseline_dir),
                tto_dir=Path(args.tto_dir),
                vad=vad,
                device=device,
                context_sec=args.context_sec,
            )
            for row in rows
        ]

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "metrics.csv", metrics)
    (out_dir / "metrics.summary.txt").write_text(
        summary_text(metrics, context_sec=args.context_sec),
        encoding="utf-8",
    )
    print(summary_text(metrics, context_sec=args.context_sec), end="")


if __name__ == "__main__":
    main()
