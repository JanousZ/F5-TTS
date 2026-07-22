"""Test-time optimization (TTO) for F5-TTS latent code.

At selected ODE steps during sampling, optimize the current latent ``x_t`` so
the one-step ``x1`` estimate (decoded by the vocoder) matches reference
wav2vec2 (valence/arousal/dominance) features. Driven by
``opt_schedule`` / ``opt_steps`` / ``opt_lr``.

Only ``x_t`` is optimized — the CFM transformer, vocoder, and VAD encoder all
stay frozen. Integration is a manual Euler loop so TTO can be injected at any
grid index; this matches ``CFM.sample``'s default ``method='euler'``.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from f5_tts.infer.emo_loss import GradEmoExtractor, precompute_reference_emo
from f5_tts.infer.spk_loss import GradSpkExtractor, precompute_reference_spk_emb
from f5_tts.infer.vad_loss import GradVADExtractor, precompute_reference_vad
from f5_tts.model.utils import (
    convert_char_to_pinyin,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


# Re-export for back-compat with callers that imported from tto.
__all__ = [
    "GradVADExtractor", "precompute_reference_vad",
    "GradSpkExtractor", "precompute_reference_spk_emb",
    "GradEmoExtractor", "precompute_reference_emo",
    "sample_with_tto",
]


def _align_frames(
    gen: torch.Tensor, ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate along the frame axis so both match in length."""
    if gen.shape[0] == ref.shape[0]:
        return gen, ref
    target = max(gen.shape[0], ref.shape[0])

    def _interp(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == target:
            return x
        y = x.transpose(0, 1).unsqueeze(0)  # (1, D, N)
        y = F.interpolate(y, size=target, mode="linear", align_corners=False)
        return y.squeeze(0).transpose(0, 1)  # (target, D)

    return _interp(gen), _interp(ref)


def _tto_inference_kit(
    cfm, vocoder, vad, ref_vad_features,
    cond, text, duration,
    *,
    lens, steps, sway_sampling_coef, seed,
    max_duration, use_epss, no_ref_audio, edit_mask,
    vocoder_type, sample_rate,
    opt_cfg_strength, loss_mode,
    window_size, hop_size,
    on_opt_step,
    vad_level: str = "frame",
    ref_vad_utter: torch.Tensor | None = None,
    amp_scale: float = 1.0,
    spk: GradSpkExtractor | None = None,
    ref_spk_emb: torch.Tensor | None = None,
    spk_weight: float = 0.0,
    asr_loss: nn.Module | None = None,
    asr_weight: float = 0.0,
    emo: GradEmoExtractor | None = None,
    ref_emo_features: torch.Tensor | None = None,
    emo_weight: float = 0.0,
):
    """Shared setup + closures for ``sample_with_tto``.

    Returns a dict with:
      Tensors / state: ``t_grid``, ``y0``, ``cond_mask``, ``cond``,
        ``ref_audio_len``, ``vocoder_dtype``.
      Loss closure ``vad_loss(wav) -> (scalar, comps_dict)``.
      One-iter closure ``iter_vad(x_var, optimizer, t_cur) -> (loss_float, comps)``.
      ``tto_step(x_t, t_cur, n_iters, lr, step_idx)``: run an Adam loop on x_t.
    """
    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    if loss_mode not in ("value", "embedding"):
        raise ValueError("loss_mode must be 'value' or 'embedding'")
    if vad_level not in ("frame", "utter", "both"):
        raise ValueError("vad_level must be 'frame', 'utter', or 'both'")
    if vad_level == "both" and ref_vad_utter is None:
        raise ValueError("vad_level='both' requires ref_vad_utter")
    if vocoder_type not in ("vocos", "bigvgan"):
        raise ValueError(f"unknown vocoder_type: {vocoder_type}")
    spk_on = spk is not None and spk_weight != 0.0
    if spk_on and ref_spk_emb is None:
        raise ValueError("spk_weight!=0 requires ref_spk_emb")
    asr_on = asr_loss is not None and asr_weight != 0.0
    emo_on = emo is not None and emo_weight != 0.0
    if emo_on and ref_emo_features is None:
        raise ValueError("emo_weight!=0 requires ref_emo_features")

    cfm.eval()
    for p in cfm.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # cond / text / duration prep — mirrors CFM.sample
    # ------------------------------------------------------------------
    if cond.ndim == 2:
        cond = cfm.mel_spec(cond)
        cond = cond.permute(0, 2, 1)
        assert cond.shape[-1] == cfm.num_channels
    cond = cond.to(next(cfm.parameters()).dtype)

    batch, cond_seq_len, device = *cond.shape[:2], cond.device
    if not exists(lens):
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

    if isinstance(text, list):
        if exists(cfm.vocab_char_map):
            text = list_str_to_idx(text, cfm.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    cond_mask = lens_to_mask(lens)
    if edit_mask is not None:
        cond_mask = cond_mask & edit_mask

    if isinstance(duration, int):
        duration = torch.full((batch,), duration, device=device, dtype=torch.long)
    duration = torch.maximum(
        torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration,
    ).clamp(max=max_duration)
    max_dur = duration.amax()

    cond = F.pad(cond, (0, 0, 0, max_dur - cond_seq_len), value=0.0)
    if no_ref_audio:
        cond = torch.zeros_like(cond)
    cond_mask = F.pad(cond_mask, (0, max_dur - cond_mask.shape[-1]), value=False)
    cond_mask = cond_mask.unsqueeze(-1)
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

    valid_mask = lens_to_mask(duration)
    mask = valid_mask if batch > 1 else None
    ref_audio_len = int(lens[0].item())

    # ------------------------------------------------------------------
    # ODE grid + initial noise y0
    # ------------------------------------------------------------------
    dtype = step_cond.dtype
    if use_epss:
        t_grid = get_epss_timesteps(steps, device=device, dtype=dtype)
    else:
        t_grid = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    if sway_sampling_coef is not None:
        t_grid = t_grid + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_grid) - 1 + t_grid)

    y0 = []
    for dur in duration:
        if exists(seed):
            torch.manual_seed(seed)
        y0.append(torch.randn(dur, cfm.num_channels, device=device, dtype=dtype))
    y0 = pad_sequence(y0, padding_value=0, batch_first=True)

    # ------------------------------------------------------------------
    # flow / vocoder closures
    # ------------------------------------------------------------------
    def _predict_flow(x, t_scalar, cfg_s):
        if cfg_s < 1e-5:
            return cfm.transformer(
                x=x, cond=step_cond, text=text, time=t_scalar, mask=mask,
                drop_audio_cond=False, drop_text=False, cache=True,
            )
        pred_cfg = cfm.transformer(
            x=x, cond=step_cond, text=text, time=t_scalar, mask=mask,
            cfg_infer=True, cache=True,
        )
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        return pred + (pred - null_pred) * cfg_s

    try:
        _vocoder_dtype = next(vocoder.parameters()).dtype
    except (StopIteration, AttributeError):
        _vocoder_dtype = None

    def _x1_to_wav(x1: torch.Tensor) -> torch.Tensor:
        out = torch.where(cond_mask, cond, x1)
        gen = out[:, ref_audio_len:, :].permute(0, 2, 1)
        if _vocoder_dtype is not None and gen.dtype != _vocoder_dtype:
            gen = gen.to(_vocoder_dtype)
        if vocoder_type == "vocos":
            # bypass vocoder.decode's @torch.inference_mode so grads flow
            wav = vocoder.head(vocoder.backbone(gen))
        else:
            wav = vocoder(gen)
        if amp_scale != 1.0:
            wav = wav * amp_scale
        return wav

    # ------------------------------------------------------------------
    # loss + one-iter closure
    # ------------------------------------------------------------------
    embeddings_flag = (loss_mode == "embedding")

    def _mse_against(gen_vad: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if gen_vad.ndim == 3:
            if ref.ndim == 2:
                ref = ref.unsqueeze(0).expand(gen_vad.shape[0], -1, -1)
            losses = []
            for b in range(gen_vad.shape[0]):
                g, r = _align_frames(gen_vad[b], ref[b])
                losses.append(F.mse_loss(g, r))
            return torch.stack(losses).mean()
        g, r = _align_frames(gen_vad, ref)
        return F.mse_loss(g, r)

    def _vad_loss(wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        vad_in = wav.squeeze(0) if wav.shape[0] == 1 else wav
        total = None
        if vad_level in ("frame", "both"):
            gen_vad = vad(
                vad_in, in_sr=sample_rate,
                window_size=window_size, hop_size=hop_size,
                embeddings=embeddings_flag,
            )
            total = _mse_against(gen_vad, ref_vad_features)
        if vad_level in ("utter", "both"):
            utter_vad = vad(
                vad_in, in_sr=sample_rate, utter=True,
                embeddings=embeddings_flag,
            )
            ref_u = ref_vad_utter if vad_level == "both" else ref_vad_features
            u_loss = _mse_against(utter_vad, ref_u)
            total = u_loss if total is None else total + u_loss
        return total, {"vad": float(total.detach())}

    def _spk_loss(wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # wav: (B, T) at sample_rate. ref_spk_emb: (1, D) L2-normed.
        emb_gen = spk(wav, in_sr=sample_rate)               # (B, D)
        cos = (emb_gen * ref_spk_emb).sum(dim=-1)            # (B,)
        loss = (1.0 - cos).mean()
        return loss, {"spk": float(loss.detach())}

    def _emo_loss(wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # wav: (B, T) at sample_rate. ref_emo_features: (T_ref, 768) per-frame.
        # Linear-interp align (differentiable), then per-frame cos sim → 1-cos.
        # Mirrors EMO-sim_frame except for the align kernel (eval uses nearest;
        # we use linear so gradients flow through the alignment).
        emo_in = wav.squeeze(0) if wav.shape[0] == 1 else wav
        gen_emo = emo(emo_in, in_sr=sample_rate)
        if gen_emo.ndim == 3:
            ref = ref_emo_features
            if ref.ndim == 2:
                ref = ref.unsqueeze(0).expand(gen_emo.shape[0], -1, -1)
            losses = []
            for b in range(gen_emo.shape[0]):
                g, r = _align_frames(gen_emo[b], ref[b])
                cos = F.cosine_similarity(g, r, dim=-1)       # (N,)
                losses.append((1.0 - cos).mean())
            loss = torch.stack(losses).mean()
        else:
            g, r = _align_frames(gen_emo, ref_emo_features)
            cos = F.cosine_similarity(g, r, dim=-1)
            loss = (1.0 - cos).mean()
        return loss, {"emo": float(loss.detach())}

    def _iter_vad(x_var, optimizer, t_cur):
        optimizer.zero_grad(set_to_none=True)
        cfm.transformer.clear_cache()
        with torch.enable_grad():
            v = _predict_flow(x_var.to(dtype), t_cur, opt_cfg_strength)
            x1_hat = x_var + (1.0 - t_cur) * v.float()
            wav = _x1_to_wav(x1_hat)
            loss, comps = _vad_loss(wav)
            if spk_on:
                spk_l, spk_comps = _spk_loss(wav)
                loss = loss + spk_weight * spk_l
                comps.update(spk_comps)
            if asr_on:
                asr_l, asr_comps = asr_loss(wav)
                loss = loss + asr_weight * asr_l
                comps.update(asr_comps)
            if emo_on:
                emo_l, emo_comps = _emo_loss(wav)
                loss = loss + emo_weight * emo_l
                comps.update(emo_comps)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), comps

    def _tto_step(x_t: torch.Tensor, t_cur: torch.Tensor,
                  n_iters: int, lr: float, step_idx: int) -> torch.Tensor:
        if n_iters <= 0:
            return x_t
        x_var = x_t.detach().clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([x_var], lr=lr)
        for it in range(n_iters):
            loss_val, comps = _iter_vad(x_var, optimizer, t_cur)
            if on_opt_step is not None:
                on_opt_step(
                    step_idx, it, loss_val,
                    loss_vad=comps.get("vad"),
                    loss_spk=comps.get("spk"),
                    loss_asr=comps.get("asr"),
                    loss_emo=comps.get("emo"),
                )
        cfm.transformer.clear_cache()
        return x_var.detach().to(dtype)

    # ------------------------------------------------------------------
    # kit
    # ------------------------------------------------------------------
    return {
        "t_grid": t_grid,
        "y0": y0,
        "cond_mask": cond_mask,
        "cond": cond,
        "ref_audio_len": ref_audio_len,
        "vocoder_dtype": _vocoder_dtype,
        "predict_flow": _predict_flow,
        "x1_to_wav": _x1_to_wav,
        "vad_loss": _vad_loss,
        "spk_loss": _spk_loss if spk_on else None,
        "asr_loss": asr_loss if asr_on else None,
        "emo_loss": _emo_loss if emo_on else None,
        "iter_vad": _iter_vad,
        "tto_step": _tto_step,
    }


def sample_with_tto(
    cfm,
    vocoder: Callable,
    vad: GradVADExtractor,
    ref_vad_features: torch.Tensor,
    cond: torch.Tensor,
    text,
    duration,
    *,
    lens: torch.Tensor | None = None,
    steps: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float | None = None,
    seed: int | None = None,
    max_duration: int = 65536,
    use_epss: bool = True,
    no_ref_audio: bool = False,
    edit_mask: torch.Tensor | None = None,
    vocoder_type: str = "vocos",
    sample_rate: int = 24000,
    # --- TTO knobs ---
    opt_schedule: Iterable[int] | Mapping[int, int] = (),
    opt_steps: int = 3,
    opt_lr: float = 1e-2,
    opt_cfg_strength: float = 1.0,
    loss_mode: str = "value",
    window_size: float = 1.0,
    hop_size: float = 0.25,
    vad_level: str = "frame",
    ref_vad_utter: torch.Tensor | None = None,
    on_opt_step: Callable[[int, int, float], None] | None = None,
    amp_scale: float = 1.0,
    spk: GradSpkExtractor | None = None,
    ref_spk_emb: torch.Tensor | None = None,
    spk_weight: float = 0.0,
    asr_loss: nn.Module | None = None,
    asr_weight: float = 0.0,
    emo: GradEmoExtractor | None = None,
    ref_emo_features: torch.Tensor | None = None,
    emo_weight: float = 0.0,
):
    """Sample from ``cfm`` with VAD test-time optimization on intermediate latents.

    ``opt_schedule`` lists ODE grid indices ``i`` (the step *before* the Euler
    update ``t_grid[i] → t_grid[i+1]``); each entry triggers an Adam loop on
    ``vad_loss`` against ``ref_vad_features``. Schedule may be an iterable or a
    ``{step_idx: inner_iters}`` mapping (default iters from ``opt_steps``).

    ``ref_vad_features`` must match ``loss_mode`` / ``window_size`` /
    ``hop_size``. ``amp_scale`` re-applies the inverse of ref RMS scaling on
    the vocoder output so the VAD loss compares same-domain audio.
    """
    if isinstance(opt_schedule, Mapping):
        sched = {int(k): int(v) for k, v in opt_schedule.items()}
    else:
        sched = {int(k): int(opt_steps) for k in opt_schedule}

    kit = _tto_inference_kit(
        cfm, vocoder, vad, ref_vad_features, cond, text, duration,
        lens=lens, steps=steps,
        sway_sampling_coef=sway_sampling_coef, seed=seed,
        max_duration=max_duration, use_epss=use_epss,
        no_ref_audio=no_ref_audio, edit_mask=edit_mask,
        vocoder_type=vocoder_type, sample_rate=sample_rate,
        opt_cfg_strength=opt_cfg_strength, loss_mode=loss_mode,
        window_size=window_size, hop_size=hop_size,
        vad_level=vad_level, ref_vad_utter=ref_vad_utter,
        on_opt_step=on_opt_step,
        amp_scale=amp_scale,
        spk=spk, ref_spk_emb=ref_spk_emb, spk_weight=spk_weight,
        asr_loss=asr_loss, asr_weight=asr_weight,
        emo=emo, ref_emo_features=ref_emo_features, emo_weight=emo_weight,
    )
    t_grid = kit["t_grid"]

    x_t = kit["y0"]
    for i in range(steps):
        t_cur = t_grid[i]
        t_next = t_grid[i + 1]
        x_t = kit["tto_step"](x_t, t_cur, sched.get(i, 0), opt_lr, i)
        with torch.no_grad():
            v = kit["predict_flow"](x_t, t_cur, cfg_strength)
            x_t = x_t + (t_next - t_cur) * v

    cfm.transformer.clear_cache()

    sampled = torch.where(kit["cond_mask"], kit["cond"], x_t)
    with torch.no_grad():
        gen_mel = sampled[:, kit["ref_audio_len"]:, :].permute(0, 2, 1)
        if kit["vocoder_dtype"] is not None and gen_mel.dtype != kit["vocoder_dtype"]:
            gen_mel = gen_mel.to(kit["vocoder_dtype"])
        wav = vocoder.decode(gen_mel) if vocoder_type == "vocos" else vocoder(gen_mel)
        if amp_scale != 1.0:
            wav = wav * amp_scale
    return wav, sampled


if __name__ == "__main__":
    # End-to-end demo: load F5-TTS + vocoder, build reference VAD, then run
    # sample_with_tto. See README.md for full command examples.
    import argparse
    from importlib.resources import files

    import soundfile as sf
    import torchaudio

    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import (
        hop_length,
        target_rms,
        target_sample_rate,
    )
    from f5_tts.model.utils import convert_char_to_pinyin

    # ====================================================================
    # CLI
    # ====================================================================
    parser = argparse.ArgumentParser(description="F5-TTS TTO demo")

    # ---- model & vocoder ----------------------------------------------
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--ckpt-file",
                        default="/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/model_1250000.safetensors",
                        help="本地 CFM 权重 (.safetensors / .pt)；空则走 HF cache")
    parser.add_argument("--vocab-file",
                        default="/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/vocab.txt",
                        help="本地 vocab.txt；空则用包内默认")
    parser.add_argument("--vocoder-local-path",
                        default="/mnt/disk1/models/vocos-mel-24khz",
                        help="本地 vocoder 目录 (vocos: 含 config.yaml + pytorch_model.bin)；空则走 HF")

    # ---- reference & generation text ----------------------------------
    parser.add_argument("--ref-audio",
                        default=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")))
    parser.add_argument("--ref-text",
                        default="Some call me nature, others call me mother nature.")
    parser.add_argument("--gen-text",
                        default="I don't really care what you call me. I've been a silent spectator.")

    # ---- ODE sampling -------------------------------------------------
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)

    # ---- VAD loss / TTO knobs (主要 loss) -----------------------------
    parser.add_argument("--opt-at", default="2,4,6,8,10,12,14",
                        help="VAD TTO 触发的 ODE step 索引（逗号分隔）")
    parser.add_argument("--opt-steps", type=int, default=50,
                        help="VAD TTO 每个点 Adam 内迭代步数")
    parser.add_argument("--opt-lr", type=float, default=1e-2,
                        help="VAD TTO Adam 学习率")
    parser.add_argument("--opt-cfg-strength", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=("value", "embedding"), default="embedding")
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument("--hop-size", type=float, default=0.25)
    parser.add_argument("--vad-level", choices=("frame", "utter", "both"), default="frame",
                        help="frame=framewise MSE; utter=utterance-level MSE; both=两者求和")

    # ---- SIM-O speaker similarity loss (附加 loss, 加权和) -------------
    parser.add_argument("--spk-loss-weight", type=float, default=0.0,
                        help="SIM-O 说话人相似度 loss 的权重 λ; 0=关闭. "
                             "总 loss = L_vad + λ·L_spk, 其中 L_spk = 1 - cos(emb_gen, emb_ref). "
                             "参考端用与 VAD 相同的 ref_wav. 启用后 VRAM 会显著上涨.")

    # ---- Paraformer All-Tokens CE ASR loss (降低 WER 用) ---------------
    parser.add_argument("--asr-loss-weight", type=float, default=0.0,
                        help="Paraformer all-token CE loss 的权重 μ; 0=关闭. "
                             "总 loss = L_vad + λ·L_spk + μ·L_asr. "
                             "L_asr = CIF teacher-forced 到 len(gen_text) 后的 LS-CE. "
                             "需要 `pip install funasr modelscope`. 启用后 VRAM ↑ ~2GB.")
    parser.add_argument("--asr-model",
                        default="/mnt/disk1/models/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
                        help="FunASR Paraformer 模型 id 或本地路径。默认本地英文大模型；"
                             "改成 model id (iic/...) 走 ModelScope cache。")

    # ---- emotion2vec EMO-SIM frame loss (情感相似度) -------------------
    parser.add_argument("--emo-loss-weight", type=float, default=0.0,
                        help="EMO-SIM frame loss 的权重 ν; 0=关闭. "
                             "总 loss = L_vad + λ·L_spk + μ·L_asr + ν·L_emo. "
                             "L_emo = (1 - cos(emo2vec_frame(gen), emo2vec_frame(ref))).mean(), "
                             "对应评测里的 EMO-sim_frame（评测用最近邻对齐，训练侧用线性插值保留梯度）。"
                             "ref 端用与 VAD 相同的 ref_wav.")
    parser.add_argument("--emo-model",
                        default="/mnt/disk1/models/emotion2vec_base_finetuned",
                        help="emotion2vec_base_finetuned 本地路径。包含 config.yaml + "
                             "emotion2vec_base.pt + tokens.txt。")

    # ---- I/O & batch --------------------------------------------------
    parser.add_argument("--output", default="tto_demo.wav",
                        help="单条: 输出文件路径; 批量/manifest: 输出目录")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="启用批量模式，从 --ref-dir 随机采样 N 条 ref 循环处理 (与 --manifest 互斥)")
    parser.add_argument("--ref-dir", default="asset",
                        help="批量模式的 ref 音频目录")
    parser.add_argument("--ref-pattern", default="*.wav",
                        help="批量模式下的 ref 文件名 glob 模式")
    parser.add_argument("--manifest", default=None,
                        help="JSONL 模式: 每行带 stem/ref_wav/ref_text/gen_text；"
                             "ref_wav 相对路径以 manifest 父目录为基。"
                             "与 --batch-size 互斥；--output 用作输出目录。")
    args = parser.parse_args()

    if args.manifest is not None and args.batch_size is not None:
        raise SystemExit("--manifest 与 --batch-size 互斥；只能选其一")

    import glob as _glob
    import json as _json
    import pathlib
    import random as _random

    # ====================================================================
    # job list: 统一成 [(ref_wav_path, out_path, ref_text, gen_text), ...]
    # ====================================================================
    if args.manifest is not None:
        manifest_path = pathlib.Path(args.manifest).resolve()
        manifest_base = manifest_path.parent
        rows: list[dict] = []
        with open(manifest_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                for k in ("stem", "ref_wav", "ref_text", "gen_text"):
                    if k not in row:
                        raise SystemExit(f"manifest row {i} missing '{k}': {row}")
                ref_p = pathlib.Path(row["ref_wav"])
                if not ref_p.is_absolute():
                    ref_p = (manifest_base / ref_p).resolve()
                row["ref_wav"] = str(ref_p)
                rows.append(row)
        if not rows:
            raise SystemExit(f"manifest '{args.manifest}' 是空的")
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        jobs = [
            (row["ref_wav"], os.path.join(out_dir, f"{row['stem']}.wav"),
             row["ref_text"], row["gen_text"])
            for row in rows
        ]
    elif args.batch_size is not None:
        candidates = sorted(_glob.glob(os.path.join(args.ref_dir, args.ref_pattern)))
        if not candidates:
            raise SystemExit(f"--ref-dir '{args.ref_dir}' 下没有匹配 '{args.ref_pattern}' 的文件")
        n = min(args.batch_size, len(candidates))
        if n < args.batch_size:
            print(f"提示: 目录仅 {len(candidates)} 个文件 < batch-size {args.batch_size}, 实际采 {n} 个")
        ref_paths = _random.Random(args.seed).sample(candidates, n)
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        jobs = [
            (rp, os.path.join(out_dir, f"{pathlib.Path(rp).stem}.wav"),
             args.ref_text, args.gen_text)
            for rp in ref_paths
        ]
    else:
        jobs = [(args.ref_audio, args.output, args.ref_text, args.gen_text)]

    # ====================================================================
    # models (load ONCE) + schedule parsing
    # ====================================================================
    tts = F5TTS(
        model=args.model,
        ckpt_file=args.ckpt_file or "",
        vocab_file=args.vocab_file or "",
        vocoder_local_path=args.vocoder_local_path or None,
    )
    device = tts.device
    vad = GradVADExtractor(in_sr=target_sample_rate, device=device)
    spk = (
        GradSpkExtractor(in_sr=target_sample_rate, device=device)
        if args.spk_loss_weight != 0.0 else None
    )
    if spk is not None:
        print(f"[TTO] SIM-O loss enabled (λ={args.spk_loss_weight}); "
              "WavLM-Large + ECAPA-TDNN loaded")

    asr_loss_mod = None
    if args.asr_loss_weight != 0.0:
        from f5_tts.infer.asr_loss import ParaformerASRLoss
        # gen_text is per-job — we lazy-init the target via set_target inside _run_one.
        asr_loss_mod = ParaformerASRLoss(
            model_name=args.asr_model,
            gen_text="",
            device=device,
            in_sr=target_sample_rate,
        )
        print(f"[TTO] ASR loss enabled (μ={args.asr_loss_weight}); Paraformer={args.asr_model}")

    emo = (
        GradEmoExtractor(in_sr=target_sample_rate, device=device, model_path=args.emo_model)
        if args.emo_loss_weight != 0.0 else None
    )
    if emo is not None:
        print(f"[TTO] EMO-SIM frame loss enabled (ν={args.emo_loss_weight}); "
              f"emotion2vec={args.emo_model}")

    opt_schedule = [int(s) for s in args.opt_at.split(",") if s.strip()]

    # ====================================================================
    # local helpers
    # ====================================================================

    def _load_mono(path: str) -> tuple[torch.Tensor, int]:
        wav, wav_sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, wav_sr

    def _normalize_ref_text(ref_text: str) -> str:
        if len(ref_text.encode("utf-8")) and len(ref_text[-1].encode("utf-8")) == 1:
            return ref_text + " "
        return ref_text

    # ====================================================================
    # per-ref runner
    # ====================================================================
    def _prepare_classic_inputs(ref_audio_path: str, ref_text: str, gen_text: str):
        """Returns (cond, text, duration, amp_scale, ref_wav_for_vad, sr)."""
        ref_wav, sr = _load_mono(ref_audio_path)
        # Raw copy at native sr (no RMS scaling) for VAD extraction.
        ref_wav_for_vad = ref_wav.to(device)
        # CFM conditioning: RMS-normalize + resample to 24 kHz. When ref was scaled up to
        # target_rms, gen comes out in target_rms domain → pass amp_scale = rms/target_rms
        # back so VAD loss compares same-domain audio.
        rms = torch.sqrt(torch.mean(ref_wav.square()))
        if rms < target_rms:
            ref_wav = ref_wav * target_rms / rms
            amp_scale = float(rms / target_rms)
        else:
            amp_scale = 1.0
        if sr != target_sample_rate:
            ref_wav = torchaudio.transforms.Resample(sr, target_sample_rate)(ref_wav)
        ref_wav = ref_wav.to(device)

        ref_text = _normalize_ref_text(ref_text)
        text = convert_char_to_pinyin([ref_text + gen_text])

        ref_audio_len = ref_wav.shape[-1] // hop_length
        ref_text_len = max(len(ref_text.encode("utf-8")), 1)
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
        return ref_wav, text, duration, amp_scale, ref_wav_for_vad, sr

    def _run_one(ref_audio_path: str, out_path: str,
                 ref_text: str, gen_text: str) -> None:
        cond, text, duration, amp_scale, ref_wav_for_vad, sr = \
            _prepare_classic_inputs(ref_audio_path, ref_text, gen_text)

        # ---- precompute reference VAD ----
        emb_flag = (args.loss_mode == "embedding")
        ref_wav_1d = ref_wav_for_vad.squeeze(0)

        def _ref_vad(*, embeddings, utter):
            kw = {} if utter else dict(window_size=args.window_size, hop_size=args.hop_size)
            return precompute_reference_vad(
                vad, ref_wav_1d, in_sr=sr,
                embeddings=embeddings, utter=utter, **kw,
            )

        if args.vad_level == "utter":
            ref_vad, ref_vad_u = _ref_vad(embeddings=emb_flag, utter=True), None
        else:
            ref_vad = _ref_vad(embeddings=emb_flag, utter=False)
            ref_vad_u = _ref_vad(embeddings=emb_flag, utter=True) if args.vad_level == "both" else None
        print(
            f"ref VAD: primary={tuple(ref_vad.shape)} "
            f"utter={None if ref_vad_u is None else tuple(ref_vad_u.shape)} "
            f"(loss_mode={args.loss_mode} vad_level={args.vad_level})"
        )

        # ---- precompute reference SIM-O speaker embedding ----
        ref_spk_emb = (
            precompute_reference_spk_emb(spk, ref_wav_1d, in_sr=sr)
            if spk is not None else None
        )

        # ---- precompute reference EMO-SIM frame features ----
        ref_emo = (
            precompute_reference_emo(emo, ref_wav_1d, in_sr=sr)
            if emo is not None else None
        )
        if ref_emo is not None:
            print(f"ref EMO: frame={tuple(ref_emo.shape)} (cos-sim against gen)")

        # ---- per-job ASR target: re-encode gen_text for Paraformer ----
        if asr_loss_mod is not None:
            asr_loss_mod.set_target(gen_text)

        def _log(step_idx, it, loss, *, loss_vad=None, loss_spk=None, loss_asr=None,
                 loss_emo=None):
            parts = [f"loss={loss:.6f}"]
            if loss_vad is not None:
                parts.append(f"vad={loss_vad:.6f}")
            if loss_spk is not None:
                parts.append(f"spk={loss_spk:.6f}")
            if loss_asr is not None:
                parts.append(f"asr={loss_asr:.6f}")
            if loss_emo is not None:
                parts.append(f"emo={loss_emo:.6f}")
            print(f"[TTO][vad] step={step_idx:02d} iter={it} " + " ".join(parts))

        wav, _ = sample_with_tto(
            cfm=tts.ema_model,
            vocoder=tts.vocoder,
            vad=vad,
            ref_vad_features=ref_vad,
            cond=cond,
            text=text,
            duration=duration,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_coef,
            seed=args.seed,
            opt_schedule=opt_schedule,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            opt_cfg_strength=args.opt_cfg_strength,
            loss_mode=args.loss_mode,
            window_size=args.window_size,
            hop_size=args.hop_size,
            vad_level=args.vad_level,
            ref_vad_utter=ref_vad_u,
            vocoder_type=tts.mel_spec_type,
            sample_rate=target_sample_rate,
            on_opt_step=_log,
            amp_scale=amp_scale,
            spk=spk,
            ref_spk_emb=ref_spk_emb,
            spk_weight=args.spk_loss_weight,
            asr_loss=asr_loss_mod,
            asr_weight=args.asr_loss_weight,
            emo=emo,
            ref_emo_features=ref_emo,
            emo_weight=args.emo_loss_weight,
        )
        wav_np = wav.squeeze().detach().float().cpu().numpy()
        sf.write(out_path, wav_np, target_sample_rate, subtype="FLOAT")
        print(f"saved: {out_path}  ({wav_np.shape[-1] / target_sample_rate:.2f}s)")

    # ====================================================================
    # main loop
    # ====================================================================
    total = len(jobs)
    for idx, (rp, out_path, ref_text, gen_text) in enumerate(jobs, 1):
        if total > 1:
            print(f"\n=== [{idx}/{total}] ref={rp} ===")
        _run_one(rp, out_path, ref_text, gen_text)

    if total > 1:
        print(f"\n批量完成: {total} 条输出保存在 {os.path.dirname(jobs[0][1])}")
