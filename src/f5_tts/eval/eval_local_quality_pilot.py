"""Evaluate content and coarse audio quality for local emotion-control pilot."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import jiwer
import numpy as np
import torch
import torchaudio
from transformers import pipeline


WHISPER_MODEL = "/mnt/disk1/models/tts_eval/whisper-large-v3"
UTMOS_HUB_ROOT = Path("/mnt/disk1/models/tts_eval/utmos_hub/hub")
UTMOS_REPO_DIR = UTMOS_HUB_ROOT / "tarepan_SpeechMOS_v1.2.0"
TARGET_SR = 16000


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("stem", "gen_text", "prompt_wav"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing {key}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def to_16k_np(wav: torch.Tensor, sr: int) -> np.ndarray:
    wav = wav.detach().float().cpu()
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.numpy()


def text_metric(gen_text: str, hyp: str) -> tuple[str, float]:
    wer_tx = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_tx = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", gen_text))
    if has_cjk:
        return "cer", float(jiwer.cer(
            gen_text, hyp,
            reference_transform=cer_tx,
            hypothesis_transform=cer_tx,
        ))
    return "wer", float(jiwer.wer(
        gen_text, hyp,
        reference_transform=wer_tx,
        hypothesis_transform=wer_tx,
    ))


def load_whisper(device: str):
    dtype = torch.float16 if "cuda" in device else torch.float32
    device_id = 0 if device.startswith("cuda") else -1
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device_id,
        torch_dtype=dtype,
        chunk_length_s=30,
    )


def load_utmos(device: str):
    if not UTMOS_REPO_DIR.exists():
        raise FileNotFoundError(f"missing local UTMOS repo: {UTMOS_REPO_DIR}")
    torch.hub.set_dir(str(UTMOS_HUB_ROOT))
    predictor = torch.hub.load(
        str(UTMOS_REPO_DIR),
        "utmos22_strong",
        source="local",
        trust_repo=True,
        verbose=False,
    )
    return predictor.to(device).eval()


def transcribe(pipe, wav: torch.Tensor, sr: int, gen_text: str) -> str:
    arr = to_16k_np(wav, sr)
    lang = "chinese" if re.search(r"[\u4e00-\u9fff]", gen_text) else "english"
    result = pipe(
        {"array": arr, "sampling_rate": TARGET_SR},
        generate_kwargs={"language": lang, "task": "transcribe"},
    )
    return result["text"].strip()


def utmos_score(predictor, wav: torch.Tensor, sr: int, device: str) -> float:
    arr = to_16k_np(wav, sr)
    wav_16k = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        score = predictor(wav_16k, TARGET_SR)
    return float(score.view(-1)[0].detach().cpu())


def audio_stats(wav: torch.Tensor, sr: int) -> dict:
    arr = wav.detach().float().cpu().numpy()
    return {
        "duration_sec": wav.shape[-1] / sr,
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "peak": float(np.max(np.abs(arr))),
        "finite": bool(np.isfinite(arr).all()),
    }


def evaluate_variant(
    *,
    row: dict,
    variant: str,
    wav_path: Path,
    whisper,
    utmos,
    device: str,
) -> dict:
    wav, sr = load_mono(wav_path)
    hyp = transcribe(whisper, wav, sr, row["gen_text"])
    metric_name, metric_score = text_metric(row["gen_text"], hyp)
    stats = audio_stats(wav, sr)
    return {
        "stem": row["stem"],
        "emotion": row.get("emotion", ""),
        "target_word": row.get("target_word", ""),
        "variant": variant,
        "text_metric": metric_name,
        "text_error": metric_score,
        "hyp": hyp,
        "ref": row["gen_text"],
        "utmos": utmos_score(utmos, wav, sr, device),
        **stats,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> str:
    by_variant = {v: [r for r in rows if r["variant"] == v] for v in sorted({r["variant"] for r in rows})}
    lines = ["Local target-span quality/content summary", ""]
    for variant, subset in by_variant.items():
        lines.extend([
            f"variant={variant}",
            f"n={len(subset)}",
            f"mean text_error={np.mean([float(r['text_error']) for r in subset]):.6f}",
            f"mean utmos={np.mean([float(r['utmos']) for r in subset]):.6f}",
            f"mean rms={np.mean([float(r['rms']) for r in subset]):.6f}",
            f"mean peak={np.mean([float(r['peak']) for r in subset]):.6f}",
            f"all finite={all(r['finite'] for r in subset)}",
            "",
        ])
    base = by_variant.get("baseline", [])
    tto = by_variant.get("tto", [])
    if base and tto and len(base) == len(tto):
        base_by_stem = {r["stem"]: r for r in base}
        tto_by_stem = {r["stem"]: r for r in tto}
        stems = sorted(base_by_stem.keys() & tto_by_stem.keys())
        lines.extend([
            "paired deltas tto-baseline",
            f"mean text_error_delta={np.mean([float(tto_by_stem[s]['text_error']) - float(base_by_stem[s]['text_error']) for s in stems]):.6f}",
            f"mean utmos_delta={np.mean([float(tto_by_stem[s]['utmos']) - float(base_by_stem[s]['utmos']) for s in stems]):.6f}",
            f"samples text_error_worse={sum(float(tto_by_stem[s]['text_error']) > float(base_by_stem[s]['text_error']) for s in stems)}/{len(stems)}",
            f"samples utmos_worse={sum(float(tto_by_stem[s]['utmos']) < float(base_by_stem[s]['utmos']) for s in stems)}/{len(stems)}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--tto-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    whisper = load_whisper(args.device)
    utmos = load_utmos(args.device)
    out_rows = []
    for row in rows:
        out_rows.append(evaluate_variant(
            row=row,
            variant="baseline",
            wav_path=Path(args.baseline_dir) / f"{row['stem']}.wav",
            whisper=whisper,
            utmos=utmos,
            device=args.device,
        ))
        out_rows.append(evaluate_variant(
            row=row,
            variant="tto",
            wav_path=Path(args.tto_dir) / f"{row['stem']}.wav",
            whisper=whisper,
            utmos=utmos,
            device=args.device,
        ))

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "quality_metrics.csv", out_rows)
    summary = summarize(out_rows)
    (out_dir / "quality_metrics.summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
