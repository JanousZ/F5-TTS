"""Text-driven TTO: target VAD trajectory derived from gen_text via text_VAD.

Mirrors tto.py's __main__ workflow but replaces the ref-audio VAD path with
text-derived VAD. Use case: zero-shot emotional TTS where the ref audio
provides only speaker timbre (no emotion reference).

This file does NOT modify tto.py — it imports the heavy lifting:
  * ``GradVADExtractor`` — differentiable audio-side VAD (gradient flows back)
  * ``sample_with_tto`` — full TTO sampling loop
  * ``_save_vad_viz`` — visualization

…and only swaps the ground-truth VAD source from ref-audio to text via
``text_VAD.TextVADExtractor``.

Constraints:
  * ``--loss-mode value`` only (text VAD has no 1024-d embedding counterpart)
  * Currently only ``--text-vad-backend lexicon`` works (NRC-VAD lookup)

Single sample:
    python src/f5_tts/infer/text_tto.py \\
        --ref-audio asset/actor01_happy-strong_to_sad-strong.wav \\
        --gen-text "I am very happy today" \\
        --opt-at 16,24 --opt-steps 5 --opt-lr 5e-3 \\
        --output text_tto_demo.wav

Batch:
    python src/f5_tts/infer/text_tto.py \\
        --ref-dir asset --batch-size 8 \\
        --gen-text "I am very happy today" \\
        --output text_tto_outputs/happy_run \\
        --viz-path text_tto_viz/happy_run
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import pathlib
import random as _random
from importlib.resources import files

import soundfile as sf
import torch
import torchaudio

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    hop_length,
    target_rms,
    target_sample_rate,
)
from f5_tts.model.utils import convert_char_to_pinyin

# Reused unchanged from tto.py:
from f5_tts.infer.tto import (
    GradVADExtractor,
    sample_with_tto,
    _save_vad_viz,
)
from f5_tts.infer.text_VAD import TextVADExtractor, NRC_VAD_PATH, WEIGHT_MODES


def _run_one(
    args, tts, vad, text_vad, opt_schedule_list,
    ref_audio_path: str, out_path: str, viz_base: str | None,
):
    """Single-sample text-driven TTO. Mirrors tto.py:_run_one structure but
    builds ``ref_vad_features`` from gen_text rather than ref-audio VAD."""

    # ---- Ref audio loading: timbre + duration estimate ONLY (not emotion) ----
    ref_wav, sr = torchaudio.load(ref_audio_path)
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(ref_wav.square()))
    if rms < target_rms:
        ref_wav = ref_wav * target_rms / rms
        amp_scale = float(rms / target_rms)
    else:
        amp_scale = 1.0
    if sr != target_sample_rate:
        ref_wav = torchaudio.transforms.Resample(sr, target_sample_rate)(ref_wav)
    ref_wav = ref_wav.to(tts.device)

    # ---- Text + duration prep (same as tto.py) ----
    ref_text = args.ref_text
    if len(ref_text.encode("utf-8")) and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    final_text_list = convert_char_to_pinyin([ref_text + args.gen_text])

    ref_audio_len = ref_wav.shape[-1] // hop_length
    ref_text_len = max(len(ref_text.encode("utf-8")), 1)
    gen_text_len = len(args.gen_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

    # ---- KEY DIFFERENCE: text-derived target ----
    if args.vad_level == "utter":
        text_target = text_vad.get_utterance_vad(args.gen_text, device=tts.device)
        text_target_utter = None
    else:
        text_target = text_vad.get_per_token_trajectory(
            args.gen_text, device=tts.device,
        )
        if args.vad_level == "both":
            text_target_utter = text_vad.get_utterance_vad(
                args.gen_text, device=tts.device,
            )
        else:
            text_target_utter = None

    print(
        f"text VAD: primary={tuple(text_target.shape)}  "
        f"utter={None if text_target_utter is None else tuple(text_target_utter.shape)}  "
        f"backend={args.text_vad_backend}  scale={args.text_vad_scale}  "
        f"weight_mode={args.text_vad_weight_mode}"
    )
    # Print per-token VAD + weight for debugging — short text only
    tok_rows = text_vad.get_token_vad_with_weights(args.gen_text)
    total_w = sum(w for *_, w in tok_rows) or 1
    if len(tok_rows) <= 30:
        for tok, v, a, d, w in tok_rows:
            share = w / total_w * 100
            print(f"  {tok:18s}  V={v:.3f}  A={a:.3f}  D={d:.3f}  "
                  f"weight={w:>2d}  share={share:5.1f}%")

    # ---- Logging / viz callbacks ----
    def _log(step, it, loss):
        print(f"[TTO-text] step={step:02d} iter={it} loss={loss:.6f}")

    viz_records: list[tuple[int, int, torch.Tensor]] = []
    viz_records_utter: list[tuple[int, int, torch.Tensor]] = []
    if viz_base:
        def _record(step, it, gv):
            viz_records.append((step, it, gv.float().cpu()))
        def _record_utter(step, it, uv):
            viz_records_utter.append((step, it, uv.float().cpu()))
    else:
        _record = None
        _record_utter = None

    # ---- TTO sampling ----
    wav, _ = sample_with_tto(
        cfm=tts.ema_model,
        vocoder=tts.vocoder,
        vad=vad,
        ref_vad_features=text_target,        # ← text source
        cond=ref_wav,
        text=final_text_list,
        duration=duration,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_coef,
        seed=args.seed,
        opt_schedule=opt_schedule_list,
        opt_steps=args.opt_steps,
        opt_lr=args.opt_lr,
        opt_cfg_strength=args.opt_cfg_strength,
        loss_mode="value",                    # text VAD only supports value
        window_size=args.window_size,
        hop_size=args.hop_size,
        vad_level=args.vad_level,
        ref_vad_utter=text_target_utter,
        vocoder_type=tts.mel_spec_type,
        sample_rate=target_sample_rate,
        on_opt_step=_log,
        on_opt_vad=_record,
        on_opt_vad_utter=_record_utter,
        amp_scale=amp_scale,
    )

    wav_np = wav.squeeze().detach().float().cpu().numpy()
    sf.write(out_path, wav_np, target_sample_rate, subtype="FLOAT")
    print(f"saved: {out_path}  ({wav_np.shape[-1] / target_sample_rate:.2f}s)")

    # ---- Optional viz against text target ----
    if viz_base:
        wav_for_vad = wav.squeeze(0) if wav.ndim == 2 else wav
        with torch.no_grad():
            final_vad = vad(
                wav_for_vad, in_sr=target_sample_rate,
                window_size=args.window_size, hop_size=args.hop_size,
                embeddings=False,
            ).float().cpu()
            if args.vad_level in ("utter", "both"):
                final_utter = vad(
                    wav_for_vad, in_sr=target_sample_rate,
                    utter=True, embeddings=False,
                ).float().cpu()
            else:
                final_utter = None
        ref_utter_cpu = (text_target_utter.float().cpu()
                         if text_target_utter is not None else None)
        # NB: x-axis of the saved plot is in target-trajectory units
        # (token index for frame mode). To get audio-time x-axis, would need
        # to interp text_target to match final_vad's frame count first.
        _save_vad_viz(
            viz_base, viz_records,
            text_target.float().cpu(), final_vad=final_vad,
            ref_utter=ref_utter_cpu,
            records_utter=viz_records_utter,
            final_utter=final_utter,
        )
        print(f"viz saved: {viz_base}.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Text-driven TTO: target VAD from gen_text (no emotional ref)",
    )
    # ---- shared with tto.py ----
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--ckpt-file", default="")
    parser.add_argument("--vocab-file", default="")
    parser.add_argument("--vocoder-local-path", default="")
    parser.add_argument(
        "--ref-audio",
        default=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        help="ref audio (used for speaker timbre + duration estimate; NOT emotion)",
    )
    parser.add_argument(
        "--ref-text",
        default="Some call me nature, others call me mother nature.",
    )
    parser.add_argument(
        "--gen-text",
        default="I am very happy today.",
        help="文本驱动情感曲线就来自这个 gen_text",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-coef", type=float, default=-1.0)
    parser.add_argument("--opt-at", default="16,24",
                        help="逗号分隔的 ODE step 索引，TTO 在这些点优化")
    parser.add_argument("--opt-steps", type=int, default=3)
    parser.add_argument("--opt-lr", type=float, default=1e-2)
    parser.add_argument("--opt-cfg-strength", type=float, default=1.0)
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument("--hop-size", type=float, default=0.25)
    parser.add_argument(
        "--vad-level", choices=("frame", "utter", "both"), default="frame",
    )
    parser.add_argument(
        "--vad-slide-mode", choices=("audio", "hidden"), default="hidden",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="text_tto_demo.wav",
                        help="单条: 输出文件路径; 批量: 输出目录")
    parser.add_argument("--viz-path", default=None,
                        help="单条: 形如 'vis' 产 vis.png; 批量: 目录路径")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="启用批量模式，从 --ref-dir 随机采 N 条 ref")
    parser.add_argument("--ref-dir", default="asset")
    parser.add_argument("--ref-pattern", default="*.wav")

    # ---- text VAD specific ----
    parser.add_argument(
        "--text-vad-backend", choices=("lexicon", "model", "llm"),
        default="lexicon",
        help="text → VAD 后端 (当前只 'lexicon' 可用)",
    )
    parser.add_argument(
        "--text-vad-lexicon", default=NRC_VAD_PATH,
        help="NRC-VAD-Lexicon 路径",
    )
    parser.add_argument(
        "--text-vad-scale", default="0,1",
        help="把 NRC [0, 1] 仿射到 audeering 空间: 'lo,hi' (默认 '0,1' 即 "
             "identity, 只 reorder 不 rescale). 例 '0.2,0.9' 收窄到 ESD 实测范围.",
    )
    parser.add_argument(
        "--text-vad-weight-mode", choices=WEIGHT_MODES, default="chars",
        help="按 token 长度分配时间占比. 'uniform' = 每词等长; 'chars' (默认) "
             "用 len(token) ('happiness'=9, 'cake'=4, 时间比 9:4); 'phonemes' "
             "用 g2p_en 输出的音素数 (需 pip install g2p_en).",
    )
    args = parser.parse_args()

    # ---- Parse opt_at, scale ----
    opt_schedule_list = [int(s) for s in args.opt_at.split(",") if s.strip()]
    parts = args.text_vad_scale.split(",")
    if len(parts) != 2:
        raise SystemExit("--text-vad-scale must be 'lo,hi'")
    scale = (float(parts[0]), float(parts[1]))

    # ---- Build job list (single vs batch) ----
    if args.batch_size is not None:
        candidates = sorted(_glob.glob(
            os.path.join(args.ref_dir, args.ref_pattern)
        ))
        if not candidates:
            raise SystemExit(
                f"--ref-dir '{args.ref_dir}' has no '{args.ref_pattern}' files"
            )
        n = min(args.batch_size, len(candidates))
        if n < args.batch_size:
            print(f"提示: 目录仅 {len(candidates)} 个文件, 实际采 {n}")
        rng = _random.Random(args.seed)
        ref_paths = rng.sample(candidates, n)
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        viz_dir = args.viz_path
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
    else:
        ref_paths = [args.ref_audio]
        out_dir = None
        viz_dir = None

    # ---- Load models ONCE ----
    tts = F5TTS(
        model=args.model,
        ckpt_file=args.ckpt_file or "",
        vocab_file=args.vocab_file or "",
        vocoder_local_path=args.vocoder_local_path or None,
    )
    device = tts.device
    vad = GradVADExtractor(
        in_sr=target_sample_rate, device=device,
        slide_mode=args.vad_slide_mode,
    )
    text_vad = TextVADExtractor(
        backend=args.text_vad_backend,
        lexicon_path=args.text_vad_lexicon,
        scale=scale,
        weight_mode=args.text_vad_weight_mode,
    )
    # Force lexicon load now so any FileNotFoundError fires before TTO loop.
    if args.text_vad_backend == "lexicon":
        n_words = text_vad._backend.coverage  # triggers _ensure_loaded
        print(f"loaded NRC-VAD lexicon: {n_words} entries from "
              f"{args.text_vad_lexicon}")

    # ---- Run ----
    total = len(ref_paths)
    for idx, rp in enumerate(ref_paths, 1):
        stem = pathlib.Path(rp).stem
        if out_dir is not None:
            out_path = os.path.join(out_dir, f"{stem}.wav")
            viz_base = os.path.join(viz_dir, stem) if viz_dir else None
            print(f"\n=== [{idx}/{total}] ref={rp} ===")
        else:
            out_path = args.output
            viz_base = args.viz_path
        _run_one(args, tts, vad, text_vad, opt_schedule_list,
                 rp, out_path, viz_base)

    if total > 1:
        print(f"\n批量完成: {total} 条输出保存在 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
