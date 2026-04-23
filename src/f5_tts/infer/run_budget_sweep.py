"""Budget-sweep experiment for F5-TTS TTO.

For each sampled reference audio, synthesize with:
  - baseline (no TTO)
  - 3 opt_at configs x 3 per-point iter levels (20/50/80)
and log (VAD-MSE, WER, spk-SIM) to CSV incrementally.

Designed to tolerate mid-run interrupts: CSV is line-buffered + fsynced
after each sample, so a restart can skip already-done rows.

Run (from repo root, with f5-tts conda env):
  python src/f5_tts/infer/run_budget_sweep.py --n-samples 5 --ref-dir asset \
      --gen-text "I don't really care what you call me." \
      --ref-text "Some call me nature, others call me mother nature." \
      --out-dir experiments/budget_sweep
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import jiwer
import whisper
from speechbrain.inference.speaker import EncoderClassifier

from f5_tts.api import F5TTS
from f5_tts.infer.tto import (
    GradVADExtractor,
    _align_frames,
    precompute_reference_vad,
    sample_with_tto,
    sample_with_tto_budget_sweep,
)
from f5_tts.infer.utils_infer import (
    hop_length,
    target_rms,
    target_sample_rate,
)
from f5_tts.model.utils import convert_char_to_pinyin


# ------- fixed experiment design -------
OPT_AT_CONFIGS: dict[str, tuple[int, ...]] = {
    "early": (4, 8, 12, 16, 20),
    "mid":   (8, 12, 16, 20, 24),
    "late":  (12, 16, 20, 24, 28),
}
LEVELS: tuple[int, ...] = (20, 50, 80)

WHISPER_PATH = "/mnt/disk1/models/whisper/base.pt"
ECAPA_PATH = "/mnt/disk1/models/spkrec-ecapa-voxceleb"

CSV_HEADER = [
    "sample_id", "ref_name", "config", "level", "total_steps",
    "loss_mode", "seed",
    "VAD_mse", "WER", "spkSIM",
    "wav_path",
]

# jiwer text normalization: strip punct, lowercase, collapse whitespace
_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def _count_tto_iters_per_sample(
    n_configs: int = len(OPT_AT_CONFIGS),
    opt_at_len: int = 5,
    levels: tuple[int, ...] = LEVELS,
) -> int:
    """Total TTO inner iters per sample across baseline+configs.

    Per config: max(levels) (shared first point) + sum(levels) * (opt_at_len-1).
    Baseline contributes 0 TTO iters.
    """
    per_cfg = max(levels) + sum(levels) * (opt_at_len - 1)
    return n_configs * per_cfg


@dataclass
class Models:
    tts: F5TTS
    vad: GradVADExtractor
    asr: "whisper.Whisper"
    spk: EncoderClassifier
    device: torch.device  # normalized to torch.device in load_models


def load_models(device_str: str | None = None) -> Models:
    tts = F5TTS(model="F5TTS_v1_Base")
    device = torch.device(tts.device) if isinstance(tts.device, str) else tts.device
    vad = GradVADExtractor(in_sr=target_sample_rate, device=device)
    asr = whisper.load_model(WHISPER_PATH, device=device)
    spk_device = f"{device.type}:{device.index}" if device.index is not None else (
        f"{device.type}:0" if device.type == "cuda" else device.type
    )
    spk = EncoderClassifier.from_hparams(
        source=ECAPA_PATH,
        savedir=ECAPA_PATH,
        run_opts={"device": spk_device},
    )
    return Models(tts=tts, vad=vad, asr=asr, spk=spk, device=device)


def load_ref_wav(path: str, device: torch.device) -> torch.Tensor:
    """Load, mono-mix, RMS-normalize, resample to 24 kHz. Returns (1, T)."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(wav.square()))
    if rms < target_rms:
        wav = wav * target_rms / rms
    if sr != target_sample_rate:
        wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
    return wav.to(device)


def _to_16k_np(wav: torch.Tensor, in_sr: int) -> np.ndarray:
    """Return 1-D float32 numpy at 16 kHz, on CPU."""
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    if in_sr != 16000:
        wav = torchaudio.functional.resample(wav.float(), in_sr, 16000)
    return wav.detach().float().cpu().numpy()


@torch.no_grad()
def compute_metrics(
    gen_wav: torch.Tensor,
    ref_wav_16k_np: np.ndarray,
    ref_vad_val: torch.Tensor,
    gen_text: str,
    models: Models,
    *,
    window_size: float,
    hop_size: float,
) -> dict[str, float]:
    """Return {VAD_mse, WER, spkSIM} for a single generation."""
    # gen_wav: (1, T) or (T,) at 24 kHz
    gen_vad = models.vad(
        gen_wav.squeeze(0) if gen_wav.ndim == 2 else gen_wav,
        in_sr=target_sample_rate,
        window_size=window_size, hop_size=hop_size,
        embeddings=False,
    ).float().cpu()
    g, r = _align_frames(gen_vad, ref_vad_val)
    vad_mse = float(F.mse_loss(g, r))

    gen_16k_np = _to_16k_np(gen_wav, target_sample_rate)

    asr_result = models.asr.transcribe(
        gen_16k_np, language="en", fp16=(models.device.type == "cuda"),
    )
    hyp = asr_result["text"].strip()
    try:
        wer = float(jiwer.wer(
            gen_text, hyp,
            reference_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        ))
    except ValueError:
        wer = float("nan")

    ref_t = torch.from_numpy(ref_wav_16k_np).unsqueeze(0).to(models.device)
    gen_t = torch.from_numpy(gen_16k_np).unsqueeze(0).to(models.device)
    ref_emb = models.spk.encode_batch(ref_t).squeeze(1).squeeze(0)
    gen_emb = models.spk.encode_batch(gen_t).squeeze(1).squeeze(0)
    sim = float(F.cosine_similarity(
        ref_emb.unsqueeze(0), gen_emb.unsqueeze(0),
    ).item())

    return {"VAD_mse": vad_mse, "WER": wer, "spkSIM": sim}


def save_wav(wav: torch.Tensor, path: str) -> None:
    arr = wav.squeeze().detach().float().cpu().numpy()
    sf.write(path, arr, target_sample_rate, subtype="FLOAT")


def build_tts_inputs(
    ref_wav: torch.Tensor, ref_text: str, gen_text: str,
):
    """Return (final_text_list, duration) matching tto.py demo logic."""
    rt = ref_text
    if len(rt.encode("utf-8")) and len(rt[-1].encode("utf-8")) == 1:
        rt = rt + " "
    text_list = convert_char_to_pinyin([rt + gen_text])
    ref_audio_len = ref_wav.shape[-1] // hop_length
    rt_len = max(len(rt.encode("utf-8")), 1)
    gt_len = len(gen_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / rt_len * gt_len)
    return text_list, duration


def existing_rows_keys(csv_path: str) -> set[tuple[str, str, int]]:
    """Return set of (ref_name, config, level) already recorded."""
    if not os.path.exists(csv_path):
        return set()
    seen = set()
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                seen.add((row["ref_name"], row["config"], int(row["level"])))
            except (KeyError, ValueError):
                continue
    return seen


def main():
    p = argparse.ArgumentParser(description="F5-TTS TTO budget sweep")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--ref-dir", default="asset")
    p.add_argument("--ref-pattern", default="*.wav")
    p.add_argument(
        "--ref-text",
        default="Some call me nature, others call me mother nature.",
    )
    p.add_argument(
        "--gen-text",
        default="I don't really care what you call me. I've been a silent spectator.",
    )
    p.add_argument("--loss-mode", choices=("value", "embedding"), default="value")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--sway-coef", type=float, default=-1.0)
    p.add_argument("--opt-lr", type=float, default=1e-2)
    p.add_argument("--opt-cfg-strength", type=float, default=1.0)
    p.add_argument("--window-size", type=float, default=1.0)
    p.add_argument("--hop-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--levels", default=None,
        help="Comma-separated override for LEVELS (e.g. '2,5' for smoke test). "
             "Default: 20,50,80",
    )
    p.add_argument("--out-dir", default="experiments/budget_sweep")
    p.add_argument("--run-name", default=None,
                   help="Subdir name under --out-dir; default = timestamp.")
    p.add_argument("--save-wavs", action="store_true", default=True)
    p.add_argument("--no-save-wavs", dest="save_wavs", action="store_false")
    args = p.parse_args()

    global LEVELS
    if args.levels:
        LEVELS = tuple(int(x) for x in args.levels.split(",") if x.strip())
        print(f"overriding LEVELS = {LEVELS}")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    wavs_dir = os.path.join(run_dir, "wavs")
    os.makedirs(run_dir, exist_ok=True)
    if args.save_wavs:
        os.makedirs(wavs_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "results.csv")

    # sample refs (deterministic given seed)
    candidates = sorted(glob.glob(os.path.join(args.ref_dir, args.ref_pattern)))
    if not candidates:
        print(f"ERR: no files under {args.ref_dir}/{args.ref_pattern}", file=sys.stderr)
        sys.exit(1)
    n = min(args.n_samples, len(candidates))
    rng = random.Random(args.seed)
    ref_paths = rng.sample(candidates, n)

    # save run config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        import json
        cfg = vars(args).copy()
        cfg["OPT_AT_CONFIGS"] = {k: list(v) for k, v in OPT_AT_CONFIGS.items()}
        cfg["LEVELS"] = list(LEVELS)
        cfg["ref_paths"] = ref_paths
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # open CSV (append + header if new)
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_f = open(csv_path, "a", newline="", buffering=1)
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(CSV_HEADER)
        csv_f.flush()
        os.fsync(csv_f.fileno())
    done_keys = existing_rows_keys(csv_path)

    def write_row(row):
        writer.writerow(row)
        csv_f.flush()
        os.fsync(csv_f.fileno())

    print(f"Loading models on device (may take ~30s for Whisper+ECAPA)...")
    models = load_models()
    print(f"device = {models.device}")

    iters_per_sample = _count_tto_iters_per_sample(
        n_configs=len(OPT_AT_CONFIGS),
        opt_at_len=len(next(iter(OPT_AT_CONFIGS.values()))),
        levels=LEVELS,
    )
    print(f"per-sample TTO inner iters ~= {iters_per_sample}")

    outer = tqdm(total=n, desc="samples", position=0)
    for sample_idx, ref_path in enumerate(ref_paths):
        stem = pathlib.Path(ref_path).stem
        outer.set_description(f"samples [{stem[:40]}]")
        t0 = time.time()

        ref_wav = load_ref_wav(ref_path, models.device)
        ref_wav_16k_np = _to_16k_np(ref_wav, target_sample_rate)
        ref_vad_val = precompute_reference_vad(
            models.vad, ref_wav.squeeze(0),
            in_sr=target_sample_rate,
            window_size=args.window_size, hop_size=args.hop_size,
            embeddings=False,
        ).float().cpu()
        ref_vad_feat = precompute_reference_vad(
            models.vad, ref_wav.squeeze(0),
            in_sr=target_sample_rate,
            window_size=args.window_size, hop_size=args.hop_size,
            embeddings=(args.loss_mode == "embedding"),
        )

        text_list, duration = build_tts_inputs(ref_wav, args.ref_text, args.gen_text)

        inner = tqdm(total=iters_per_sample, desc="tto iters",
                     leave=False, position=1)

        def _on_opt_step(step_idx, iter_idx, loss):
            inner.update(1)
            inner.set_postfix(
                config=_on_opt_step.cfg_name,
                level=_on_opt_step.level,
                loss=f"{loss:.4f}",
            )
        _on_opt_step.cfg_name = "-"
        _on_opt_step.level = 0

        common = dict(
            cfm=models.tts.ema_model,
            vocoder=models.tts.vocoder,
            vad=models.vad,
            ref_vad_features=ref_vad_feat,
            cond=ref_wav,
            text=text_list,
            duration=duration,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_coef,
            seed=args.seed,
            loss_mode=args.loss_mode,
            window_size=args.window_size,
            hop_size=args.hop_size,
            vocoder_type=models.tts.mel_spec_type,
            sample_rate=target_sample_rate,
        )

        # --- baseline (no TTO) ---
        base_key = (stem, "baseline", 0)
        if base_key not in done_keys:
            inner.set_postfix(config="baseline", level=0)
            wav, _ = sample_with_tto(
                **common,
                opt_schedule=[],
                opt_steps=0,
                opt_lr=args.opt_lr,
                opt_cfg_strength=args.opt_cfg_strength,
                on_opt_step=None,
                on_opt_vad=None,
            )
            wav_path = os.path.join(wavs_dir, f"{stem}__baseline.wav") if args.save_wavs else ""
            if args.save_wavs:
                save_wav(wav, wav_path)
            m = compute_metrics(
                wav, ref_wav_16k_np, ref_vad_val, args.gen_text, models,
                window_size=args.window_size, hop_size=args.hop_size,
            )
            write_row([
                sample_idx, stem, "baseline", 0, 0,
                args.loss_mode, args.seed,
                f"{m['VAD_mse']:.6f}", f"{m['WER']:.6f}", f"{m['spkSIM']:.6f}",
                wav_path,
            ])
            done_keys.add(base_key)

        # --- 3 configs x 3 levels via budget sweep ---
        for cfg_name, opt_at in OPT_AT_CONFIGS.items():
            # skip entirely only if ALL levels already done
            missing = [L for L in LEVELS if (stem, cfg_name, L) not in done_keys]
            if not missing:
                continue
            _on_opt_step.cfg_name = cfg_name

            # sweep runs all LEVELS in one call (sharing first point); cheap
            # even if some are already done — keeps code simple.
            results = sample_with_tto_budget_sweep(
                **common,
                opt_at=opt_at,
                per_point_iters_levels=LEVELS,
                opt_lr=args.opt_lr,
                opt_cfg_strength=args.opt_cfg_strength,
                on_opt_step=_on_opt_step,
            )
            for L in sorted(results.keys()):
                if (stem, cfg_name, L) in done_keys:
                    continue
                _on_opt_step.level = L
                wav = results[L]
                wav_path = (
                    os.path.join(wavs_dir, f"{stem}__{cfg_name}_L{L}.wav")
                    if args.save_wavs else ""
                )
                if args.save_wavs:
                    save_wav(wav, wav_path)
                m = compute_metrics(
                    wav, ref_wav_16k_np, ref_vad_val, args.gen_text, models,
                    window_size=args.window_size, hop_size=args.hop_size,
                )
                total_steps = L * len(opt_at)
                write_row([
                    sample_idx, stem, cfg_name, L, total_steps,
                    args.loss_mode, args.seed,
                    f"{m['VAD_mse']:.6f}", f"{m['WER']:.6f}", f"{m['spkSIM']:.6f}",
                    wav_path,
                ])
                done_keys.add((stem, cfg_name, L))

        inner.close()
        dt = time.time() - t0
        outer.update(1)
        outer.write(f"[sample {sample_idx+1}/{n}] {stem} done in {dt/60:.1f} min")

    outer.close()
    csv_f.close()
    print(f"\nDone. CSV: {csv_path}")


if __name__ == "__main__":
    main()