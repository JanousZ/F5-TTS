"""Test-time optimization (TTO) for F5-TTS latent code.

At selected ODE steps during sampling, optimize the current latent ``x_t`` so
the one-step ``x1`` estimate (decoded by the vocoder) matches reference
acoustic features. Two loss families can be used jointly or separately:

* **VAD loss** — frame-level / utter-level MSE between the generated wav's
  wav2vec2 (valence/arousal/dominance) features and the reference's. Driven by
  ``opt_schedule`` / ``opt_steps`` / ``opt_lr``.
* **UTMOS loss** — naturalness regularizer ``w * (5 - mean MOS)`` from a
  frozen UTMOS22-strong predictor. Driven by ``utmos_opt_schedule`` /
  ``utmos_opt_steps`` / ``utmos_opt_lr``.

Per ODE step ``i``:
  * only-VAD step    → one Adam loop on ``vad_loss``.
  * only-UTMOS step  → one Adam loop on ``utmos_loss``.
  * both schedules hit ``i`` → first a merged phase of ``min(n_v, n_u)``
    iters, then a leftover phase of ``|n_v - n_u|`` iters running just the
    schedule with more budget. Adam state is shared across the two phases
    (lr swapped per phase, momentum / moment estimates carry over) so the
    leftover phase continues where the merged phase left off.

The merged phase is either ``vad_loss + utmos_loss`` in a single backward
(default) or, with ``grad_proj`` set, two per-task backwards followed by
projection of the UTMOS gradient w.r.t. the VAD gradient:
  * ``ortho``  — always subtract the component of ``g_utmos`` parallel to
    ``g_vad`` (UTMOS contributes only the part orthogonal to VAD).
  * ``pcgrad`` — only project when ``g_vad · g_utmos < 0`` (per-batch).

Only ``x_t`` is optimized — the CFM transformer, vocoder, VAD encoder, and
UTMOS predictor all stay frozen. Integration is a manual Euler loop so TTO
can be injected at any grid index; this matches ``CFM.sample``'s default
``method='euler'``.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from f5_tts.model.utils import (
    convert_char_to_pinyin,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


_VAD_MODEL_NAME = '/mnt/disk1/models/wav2vec2-large-robust-12-ft-emotion-msp-dim'
_VAD_SR = 16000
# wav2vec2-large 的 conv stack 总 stride = 320，输入 16 kHz → 输出 ~50 fps
# (20 ms / frame)。滑窗用这个把秒数换算成 frame 数。
_VAD_FRAME_HZ = 50

_UTMOS_HUB_DIR = '/mnt/disk1/models/tts_eval/utmos_hub/hub'


def load_utmos(device, dtype: torch.dtype | None = None) -> nn.Module:
    """加载 UTMOS22-strong（自然度 MOS, 越高越好），冻结。

    backbone 全部 eval；但所有 RNN 子模块强制 train()，绕过 cuDNN
    "RNN backward can only be called in training mode" 限制（LSTM 无 dropout，
    train/eval 行为一致）。
    """
    torch.hub.set_dir(_UTMOS_HUB_DIR)
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
        trust_repo=True, source="github", verbose=False,
    ).to(device).eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.train()
    return model


class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


class GradVADExtractor(nn.Module):
    """Differentiable VAD extractor over ``wav2vec2-large-robust-12-ft-emotion-msp-dim``.

    Frozen backbone; gradients flow back to input wav. One wav2vec2 forward
    over the full audio → ``(T_frame, 1024)`` hidden state → unfold over the
    time axis → mean per window → classifier per window. Frames see the full
    attention context, and only one backbone forward is needed.
    """

    def __init__(
        self,
        in_sr: int = 24000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_sr = in_sr
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(_VAD_MODEL_NAME)
        with open(os.path.join(_VAD_MODEL_NAME, 'config.json')) as f:
            cfg_dict = json.load(f)
        if cfg_dict.get('vocab_size') is None:
            cfg_dict['vocab_size'] = 32
        cfg = Wav2Vec2Config(**cfg_dict)
        model = _EmotionModel.from_pretrained(_VAD_MODEL_NAME, config=cfg)
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.do_normalize = getattr(self._processor, "do_normalize", True)
        self._default_resampler: nn.Module = (
            torchaudio.transforms.Resample(in_sr, _VAD_SR)
            if in_sr != _VAD_SR else nn.Identity()
        )
        if device is not None:
            self._default_resampler = self._default_resampler.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _to_16k(self, wav: torch.Tensor, in_sr: int) -> torch.Tensor:
        if in_sr == _VAD_SR:
            return wav
        if in_sr == self.in_sr and isinstance(self._default_resampler, torchaudio.transforms.Resample):
            return self._default_resampler(wav)
        return torchaudio.functional.resample(wav, in_sr, _VAD_SR)

    @staticmethod
    def _normalize(wav: torch.Tensor) -> torch.Tensor:
        # Matches Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm.
        mean = wav.mean(dim=-1, keepdim=True)
        var = wav.var(dim=-1, keepdim=True, unbiased=False)
        return (wav - mean) / torch.sqrt(var + 1e-7)

    def forward(
        self,
        wav: torch.Tensor,
        *,
        in_sr: int | None = None,
        window_size: float = 1.0,
        hop_size: float = 0.25,
        embeddings: bool = False,
        pad: bool = True,
        utter: bool = False,
    ):
        """Return frame-level features for ``wav``.

        ``wav`` may be ``(T,)`` or ``(B, T)``. Output is ``(N, D)`` /
        ``(B, N, D)`` where ``D = hidden_size`` when ``embeddings=True`` and 3
        otherwise (arousal/dominance/valence). When ``utter=True``, skips framing
        and runs the model on the whole signal (mirrors
        ``VAD_extractor.process_func``); output has ``N=1``.
        """
        squeeze_batch = wav.ndim == 1
        if squeeze_batch:
            wav = wav.unsqueeze(0)
        in_sr = in_sr if in_sr is not None else self.in_sr
        wav = self._to_16k(wav, in_sr)

        if utter:
            signal = self._normalize(wav) if self.do_normalize else wav
            signal = signal.to(next(self.model.parameters()).dtype)
            pooled, logits = self.model(signal)  # (B, D)
            feat = (pooled if embeddings else logits).unsqueeze(1)  # (B, 1, D)
            return feat.squeeze(0) if squeeze_batch else feat

        emb, val = self._frame_features(
            wav, window_size=window_size, hop_size=hop_size, pad=pad,
        )
        feat = emb if embeddings else val
        return feat.squeeze(0) if squeeze_batch else feat

    # ---- internal helpers ------------------------------------------------

    def _frame_features(
        self, wav: torch.Tensor, *, window_size: float, hop_size: float, pad: bool,
    ):
        """One backbone forward over full audio, then slide window on hidden."""
        signal = self._normalize(wav) if self.do_normalize else wav
        signal = signal.to(next(self.model.parameters()).dtype)
        # Skip the model's own mean+classifier — we'll do mean-per-window then
        # apply classifier ourselves. This keeps the gradient path short.
        hidden = self.model.wav2vec2(signal)[0]    # (B, T_frame, D)
        B, T_frame, D = hidden.shape

        win = max(1, int(round(window_size * _VAD_FRAME_HZ)))
        hop = max(1, int(round(hop_size * _VAD_FRAME_HZ)))

        if T_frame < win:
            if not pad:
                raise ValueError(f"hidden frames ({T_frame}) shorter than win ({win})")
            pooled = hidden.mean(dim=1, keepdim=True)        # (B, 1, D)
        else:
            if pad:
                rem = (T_frame - win) % hop
                if rem != 0:
                    pad_n = hop - rem
                    hidden = F.pad(hidden, (0, 0, 0, pad_n))  # pad time axis
            windows = hidden.unfold(1, win, hop)              # (B, n_win, D, win)
            pooled = windows.mean(dim=-1)                     # (B, n_win, D)

        n_win = pooled.shape[1]
        logits = self.model.classifier(
            pooled.reshape(B * n_win, D)
        ).view(B, n_win, -1)                                  # (B, n_win, 3)
        return pooled, logits


@torch.no_grad()
def precompute_reference_vad(
    vad: GradVADExtractor,
    ref_wav: torch.Tensor,
    *,
    in_sr: int,
    window_size: float = 1.0,
    hop_size: float = 0.25,
    embeddings: bool = False,
    utter: bool = False,
) -> torch.Tensor:
    """Extract reference frame-level (or utter-level) VAD once, no grad."""
    return vad(
        ref_wav,
        in_sr=in_sr,
        window_size=window_size,
        hop_size=hop_size,
        embeddings=embeddings,
        utter=utter,
    ).detach()


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


def build_segmented_attn_mask(
    ref_lens: Sequence[int] | torch.Tensor,
    gen_lens: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    batch_size: int | None = None,
    gen_visible_text_prefix_len: int | None = None,
) -> torch.Tensor:
    """Build a grouped-layout block attention mask for segmented inference.

    Sequence layout is ``[ref_1 ... ref_N, gen_1 ... gen_N]``. Reference
    segments can see all reference segments, plus their paired generation
    segment; all generation segments can see each other and their paired
    reference segment. Optionally, generation queries can also see a global
    prefix range ``[:gen_visible_text_prefix_len]``.
    """
    if torch.is_tensor(ref_lens):
        ref_lens_list = [int(x) for x in ref_lens.detach().cpu().tolist()]
    else:
        ref_lens_list = [int(x) for x in ref_lens]
    if torch.is_tensor(gen_lens):
        gen_lens_list = [int(x) for x in gen_lens.detach().cpu().tolist()]
    else:
        gen_lens_list = [int(x) for x in gen_lens]

    if len(ref_lens_list) != len(gen_lens_list):
        raise ValueError("ref_lens and gen_lens must have the same number of segments")
    if not ref_lens_list:
        raise ValueError("at least one segment is required")
    if any(x <= 0 for x in ref_lens_list + gen_lens_list):
        raise ValueError("all segment lengths must be positive")

    ref_total = sum(ref_lens_list)
    gen_total = sum(gen_lens_list)
    total = ref_total + gen_total
    mask = torch.zeros((total, total), device=device, dtype=torch.bool)

    gen_start = ref_total
    # Relaxed mode: allow full reference-reference interactions.
    mask[:gen_start, :gen_start] = True
    mask[gen_start:, gen_start:] = True

    r0 = 0
    g0 = gen_start
    for r_len, g_len in zip(ref_lens_list, gen_lens_list):
        r1 = r0 + r_len
        g1 = g0 + g_len

        mask[r0:r1, g0:g1] = True
        mask[g0:g1, r0:r1] = True

        r0 = r1
        g0 = g1

    if gen_visible_text_prefix_len is not None:
        p = max(0, min(int(gen_visible_text_prefix_len), total))
        if p > 0:
            # Directional relaxation: generation queries can attend to the
            # global text-valid prefix on the audio frame axis.
            mask[gen_start:, :p] = True

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(int(batch_size), -1, -1).clone()
    return mask


def build_segmented_text_tokens(
    cfm,
    ref_texts: Sequence[str],
    gen_texts: Sequence[str],
    ref_lens: Sequence[int] | torch.Tensor,
    gen_lens: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    convert_to_pinyin: bool = True,
) -> tuple[torch.Tensor, int]:
    """Tokenize one concatenated text sequence and pad once to total length."""
    ref_lens_list = [int(x) for x in (ref_lens.detach().cpu().tolist() if torch.is_tensor(ref_lens) else ref_lens)]
    gen_lens_list = [int(x) for x in (gen_lens.detach().cpu().tolist() if torch.is_tensor(gen_lens) else gen_lens)]
    if len(ref_texts) != len(gen_texts) or len(ref_texts) != len(ref_lens_list):
        raise ValueError("ref_texts, gen_texts, ref_lens and gen_lens must have the same segment count")
    total_len = sum(ref_lens_list) + sum(gen_lens_list)
    merged_text = "".join(list(ref_texts) + list(gen_texts))
    texts = [merged_text]
    if convert_to_pinyin:
        texts = convert_char_to_pinyin(texts)
    if exists(cfm.vocab_char_map):
        token = list_str_to_idx(texts, cfm.vocab_char_map)[0]
    else:
        token = list_str_to_tensor(texts)[0]

    valid = token[token != -1][:total_len]
    valid_len = int(valid.shape[0])
    if valid_len < total_len:
        valid = F.pad(valid, (0, total_len - valid_len), value=-1)
    return valid.unsqueeze(0).to(device), valid_len


def prepare_segmented_tto_inputs(
    cfm,
    ref_wavs: Sequence[torch.Tensor],
    ref_texts: Sequence[str],
    gen_texts: Sequence[str],
    *,
    gen_durations: Sequence[int] | torch.Tensor | None = None,
    speed: float = 1.0,
    device: torch.device | str | None = None,
    convert_to_pinyin: bool = True,
) -> dict[str, torch.Tensor | list[int]]:
    """Prepare ``cond``, ``text``, ``duration`` and block mask for segmented TTO.

    ``ref_wavs`` are expected to be mono waveforms already resampled/RMS-scaled
    for the CFM path. The returned sequence layout is grouped as
    ``[ref_1 ... ref_N, gen_1 ... gen_N]``.
    """
    if len(ref_wavs) != len(ref_texts) or len(ref_wavs) != len(gen_texts):
        raise ValueError("ref_wavs, ref_texts and gen_texts must have the same segment count")
    if speed <= 0:
        raise ValueError("speed must be positive")
    if device is None:
        device = next(cfm.parameters()).device

    ref_mels = []
    ref_lens: list[int] = []
    for wav in ref_wavs:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError("each ref wav must have shape (T,) or (1, T)")
        mel = cfm.mel_spec(wav.to(device)).permute(0, 2, 1)
        if mel.shape[0] != 1:
            raise ValueError("prepare_segmented_tto_inputs currently expects mono/single-item ref wavs")
        ref_mels.append(mel.squeeze(0))
        ref_lens.append(int(mel.shape[1]))

    if gen_durations is None:
        gen_lens = []
        for r_len, r_text, g_text in zip(ref_lens, ref_texts, gen_texts):
            ref_text_len = max(len(r_text.encode("utf-8")), 1)
            gen_text_len = max(len(g_text.encode("utf-8")), 1)
            gen_lens.append(max(1, int(r_len / ref_text_len * gen_text_len / speed)))
    elif torch.is_tensor(gen_durations):
        gen_lens = [int(x) for x in gen_durations.detach().cpu().tolist()]
    else:
        gen_lens = [int(x) for x in gen_durations]
    if len(gen_lens) != len(ref_lens):
        raise ValueError("gen_durations must match the number of segments")

    cond = torch.cat(ref_mels, dim=0).unsqueeze(0)
    text, text_valid_len = build_segmented_text_tokens(
        cfm, ref_texts, gen_texts, ref_lens, gen_lens,
        device=device, convert_to_pinyin=convert_to_pinyin,
    )
    attn_block_mask = build_segmented_attn_mask(
        ref_lens,
        gen_lens,
        device=device,
        batch_size=1,
        gen_visible_text_prefix_len=text_valid_len,
    )
    lens = torch.tensor([sum(ref_lens)], device=device, dtype=torch.long)
    duration = torch.tensor([sum(ref_lens) + sum(gen_lens)], device=device, dtype=torch.long)

    return {
        "cond": cond,
        "text": text,
        "duration": duration,
        "lens": lens,
        "attn_block_mask": attn_block_mask,
        "text_valid_len": text_valid_len,
        "ref_lens": ref_lens,
        "gen_lens": gen_lens,
    }


def _tto_inference_kit(
    cfm, vocoder, vad, ref_vad_features,
    cond, text, duration,
    *,
    lens, steps, cfg_strength, sway_sampling_coef, seed,
    max_duration, use_epss, no_ref_audio, edit_mask,
    vocoder_type, sample_rate,
    opt_cfg_strength, loss_mode,
    window_size, hop_size,
    on_opt_step,
    vad_level: str = "frame",
    ref_vad_utter: torch.Tensor | None = None,
    amp_scale: float = 1.0,
    attn_block_mask: torch.Tensor | None = None,
    utmos: nn.Module | None = None,
    utmos_weight: float = 0.0,
    grad_proj: str | None = None,
):
    """Shared setup + closures for ``sample_with_tto``.

    Returns a dict with:
      Tensors / state: ``t_grid``, ``y0``, ``cond_mask``, ``cond``,
        ``ref_audio_len``, ``vocoder_dtype``.
      Loss closures (``f(wav) -> (scalar, comps_dict)``):
        ``vad_loss``, ``utmos_loss`` (None when UTMOS off), ``combined_loss``.
      One-iter closures (``f(x_var, optimizer, t_cur) -> (loss_float, comps)``):
        ``iter_vad``, ``iter_utmos``, ``iter_combined``, ``iter_proj``. The
        last is None unless ``grad_proj`` is set and UTMOS is active; it runs
        two per-task backwards then projects ``g_utmos`` w.r.t. ``g_vad``.
      ``tto_step(x_t, t_cur, phases, step_idx)``: run a list of
        ``(n_iters, lr, iter_fn)`` phases sequentially with one shared Adam
        optimizer (lr swapped per phase, momentum kept).
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
    mask = valid_mask if batch > 1 or attn_block_mask is not None else None
    if attn_block_mask is not None:
        attn_block_mask = attn_block_mask.to(device=device, dtype=torch.bool)
        if attn_block_mask.ndim == 2:
            attn_block_mask = attn_block_mask.unsqueeze(0)
        if attn_block_mask.ndim != 3:
            raise ValueError("attn_block_mask must have shape (N, N) or (B, N, N)")
        if attn_block_mask.shape[0] == 1 and batch > 1:
            attn_block_mask = attn_block_mask.expand(batch, -1, -1).clone()
        if attn_block_mask.shape[0] != batch:
            raise ValueError("attn_block_mask batch size must match cond batch size")
        if attn_block_mask.shape[-2:] != (int(max_dur.item()), int(max_dur.item())):
            raise ValueError(
                "attn_block_mask shape must match final duration "
                f"({int(max_dur.item())}, {int(max_dur.item())}), got {tuple(attn_block_mask.shape[-2:])}"
            )
        attn_block_mask = attn_block_mask & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
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
                attn_block_mask=attn_block_mask,
            )
        pred_cfg = cfm.transformer(
            x=x, cond=step_cond, text=text, time=t_scalar, mask=mask,
            cfg_infer=True, cache=True, attn_block_mask=attn_block_mask,
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
    # loss closures
    # ------------------------------------------------------------------
    embeddings_flag = (loss_mode == "embedding")
    use_utmos = utmos is not None and utmos_weight > 0

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
        """仅 VAD loss；不含 UTMOS。"""
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

    def _utmos_loss(wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """仅 UTMOS loss = utmos_weight * (5 - mean MOS)；调用方需自行保证 use_utmos。"""
        wav_2d = wav if wav.ndim == 2 else wav.unsqueeze(0)
        score = utmos(wav_2d, sample_rate)            # (B,) MOS
        u = (5.0 - score.mean()) * utmos_weight
        return u, {"utmos": float(u.detach())}

    def _combined_loss(wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """VAD + UTMOS 求和；用于 VAD/UTMOS 重叠 step。"""
        total, comps = _vad_loss(wav)
        if use_utmos:
            u_l, u_c = _utmos_loss(wav)
            total = total + u_l
            comps = {**comps, **u_c}
        return total, comps

    # ------------------------------------------------------------------
    # one-iter closures: each runs a single Adam iteration on x_var and
    # returns (loss_float, comps). They are composed into multi-phase
    # optimization by ``_tto_step``.
    # ------------------------------------------------------------------
    if grad_proj not in (None, "ortho", "pcgrad"):
        raise ValueError(f"grad_proj must be None, 'ortho', or 'pcgrad'; got {grad_proj!r}")

    def _make_single_iter(loss_fn):
        def step(x_var, optimizer, t_cur):
            optimizer.zero_grad(set_to_none=True)
            cfm.transformer.clear_cache()
            with torch.enable_grad():
                v = _predict_flow(x_var.to(dtype), t_cur, opt_cfg_strength)
                x1_hat = x_var + (1.0 - t_cur) * v.float()
                wav = _x1_to_wav(x1_hat)
                loss, comps = loss_fn(wav)
            loss.backward()
            optimizer.step()
            return float(loss.detach()), comps
        return step

    def _make_proj_iter(primary_fn, secondary_fn, mode: str):
        """Per-task backward + projection of g_secondary w.r.t. g_primary.

        mode='ortho'  → always subtract the component of g_s parallel to g_p.
        mode='pcgrad' → only subtract when g_p · g_s < 0 (per batch element).
        """
        def step(x_var, optimizer, t_cur):
            optimizer.zero_grad(set_to_none=True)
            cfm.transformer.clear_cache()
            with torch.enable_grad():
                v = _predict_flow(x_var.to(dtype), t_cur, opt_cfg_strength)
                x1_hat = x_var + (1.0 - t_cur) * v.float()
                wav = _x1_to_wav(x1_hat)
                loss_p, comps_p = primary_fn(wav)
                loss_s, comps_s = secondary_fn(wav)

            # primary backward first; keep the graph for secondary backward.
            loss_p.backward(retain_graph=True)
            g_p = x_var.grad.detach().clone()
            x_var.grad.zero_()
            loss_s.backward()
            g_s = x_var.grad.detach().clone()

            # per-batch projection over the (T, D) axes.
            B = g_p.shape[0]
            g_p_flat = g_p.reshape(B, -1)
            g_s_flat = g_s.reshape(B, -1)
            denom = (g_p_flat * g_p_flat).sum(dim=1, keepdim=True).clamp_min(1e-12)
            dot = (g_s_flat * g_p_flat).sum(dim=1, keepdim=True)
            if mode == "ortho":
                g_s_flat = g_s_flat - (dot / denom) * g_p_flat
            else:  # 'pcgrad'
                mask = (dot < 0).to(dot.dtype)
                g_s_flat = g_s_flat - mask * (dot / denom) * g_p_flat
            g_s = g_s_flat.view_as(g_s)

            x_var.grad = g_p + g_s
            optimizer.step()

            comps = {**comps_p, **comps_s}
            total = float((loss_p + loss_s).detach())
            return total, comps
        return step

    iter_vad = _make_single_iter(_vad_loss)
    iter_utmos = _make_single_iter(_utmos_loss) if use_utmos else None
    iter_combined = _make_single_iter(_combined_loss) if use_utmos else None
    iter_proj = (
        _make_proj_iter(_vad_loss, _utmos_loss, grad_proj)
        if (use_utmos and grad_proj is not None) else None
    )

    def _tto_step(x_t: torch.Tensor, t_cur: torch.Tensor,
                  phases, step_idx: int) -> torch.Tensor:
        """Run ``phases`` sequentially with one shared Adam optimizer.

        ``phases``: iterable of ``(n_iters, lr, iter_fn)``. Phases with
        ``n_iters <= 0`` are skipped. Adam is created lazily on the first
        non-empty phase; subsequent phases overwrite ``lr`` on
        ``param_groups`` but momentum / moment estimates carry over so the
        next phase continues from where the previous one left off.
        """
        active = [(n, lr, fn) for n, lr, fn in phases if n > 0]
        if not active:
            return x_t
        x_var = x_t.detach().clone().float().requires_grad_(True)
        optimizer = None
        it_global = 0
        for n_iters, lr, iter_fn in active:
            if optimizer is None:
                optimizer = torch.optim.Adam([x_var], lr=lr)
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            for _ in range(n_iters):
                loss_val, comps = iter_fn(x_var, optimizer, t_cur)
                if on_opt_step is not None:
                    on_opt_step(step_idx, it_global, loss_val,
                                loss_vad=comps.get("vad"), loss_utmos=comps.get("utmos"))
                it_global += 1
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
        "utmos_loss": _utmos_loss if use_utmos else None,
        "combined_loss": _combined_loss,
        "iter_vad": iter_vad,
        "iter_utmos": iter_utmos,
        "iter_combined": iter_combined,
        "iter_proj": iter_proj,
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
    attn_block_mask: torch.Tensor | None = None,
    utmos: nn.Module | None = None,
    utmos_weight: float = 0.0,
    utmos_opt_schedule: Iterable[int] | Mapping[int, int] = (),
    utmos_opt_steps: int | None = None,
    utmos_opt_lr: float | None = None,
    grad_proj: str | None = None,
):
    """Sample from ``cfm`` with test-time optimization on intermediate latents.

    Two independent TTO schedules:

    * VAD: ``opt_schedule`` / ``opt_steps`` / ``opt_lr``
    * UTMOS: ``utmos_opt_schedule`` / ``utmos_opt_steps`` / ``utmos_opt_lr``
      (the two ``utmos_opt_*`` defaults fall back to their VAD counterparts.)

    Each schedule entry is an ODE grid index ``i`` (the step *before* the
    Euler update ``t_grid[i] → t_grid[i+1]``) and may be passed as an iterable
    or as a ``{step_idx: inner_iters}`` mapping.

    Per ``i``:
      * only-VAD     → Adam loop on ``vad_loss``.
      * only-UTMOS   → Adam loop on ``utmos_loss``.
      * both present → first ``min(n_v, n_u)`` iters on a merged loss, then
        ``|n_v - n_u|`` iters on whichever schedule had more (VAD uses
        ``opt_lr``; UTMOS uses ``utmos_opt_lr``). Adam state is shared
        across the two phases (lr swapped, momentum kept).

    Other key knobs:
      ``ref_vad_features``: precomputed reference VAD ``(N_ref, D)``; must
        match ``loss_mode`` / ``window_size`` / ``hop_size``.
      ``amp_scale``: gain applied to vocoder output to undo target_rms
        normalization (use ``rms_ref / target_rms``; ``1.0`` if not normalized).
      ``on_opt_step``: optional ``(step_idx, iter, loss, *, loss_vad, loss_utmos)``
        callback for logging.
      ``grad_proj``: ``None`` (default) merges the two losses in a single
        backward in the overlapping phase. ``"ortho"`` backprops each loss
        separately and subtracts the component of ``g_utmos`` parallel to
        ``g_vad`` before the Adam step (so UTMOS only moves orthogonal to
        VAD). ``"pcgrad"`` does the same projection only when
        ``g_vad · g_utmos < 0`` (per batch element).
    """
    def _expand_sched(sch, default_iters):
        if isinstance(sch, Mapping):
            return {int(k): int(v) for k, v in sch.items()}
        return {int(k): int(default_iters) for k in sch}

    sched = _expand_sched(opt_schedule, opt_steps)
    u_sched = _expand_sched(
        utmos_opt_schedule,
        opt_steps if utmos_opt_steps is None else utmos_opt_steps,
    )
    u_lr = opt_lr if utmos_opt_lr is None else utmos_opt_lr

    if grad_proj == "none":
        grad_proj = None

    kit = _tto_inference_kit(
        cfm, vocoder, vad, ref_vad_features, cond, text, duration,
        lens=lens, steps=steps, cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef, seed=seed,
        max_duration=max_duration, use_epss=use_epss,
        no_ref_audio=no_ref_audio, edit_mask=edit_mask,
        vocoder_type=vocoder_type, sample_rate=sample_rate,
        opt_cfg_strength=opt_cfg_strength, loss_mode=loss_mode,
        window_size=window_size, hop_size=hop_size,
        vad_level=vad_level, ref_vad_utter=ref_vad_utter,
        on_opt_step=on_opt_step,
        amp_scale=amp_scale, attn_block_mask=attn_block_mask,
        utmos=utmos, utmos_weight=utmos_weight,
        grad_proj=grad_proj,
    )
    t_grid = kit["t_grid"]

    if u_sched and kit["utmos_loss"] is None:
        raise ValueError("utmos_opt_schedule is non-empty but utmos/utmos_weight not set")

    # 重叠 step 的合并阶段：projection 模式分两次 backward 做正交化，
    # 否则单次 backward 跑 vad+utmos 求和。
    overlap_iter = kit["iter_proj"] if grad_proj is not None else kit["iter_combined"]

    x_t = kit["y0"]
    for i in range(steps):
        t_cur = t_grid[i]
        t_next = t_grid[i + 1]
        n_v, n_u = sched.get(i, 0), u_sched.get(i, 0)
        phases: list[tuple[int, float, Callable]] = []
        if n_v > 0 and n_u > 0:
            # 重叠 step: 先 min(n_v, n_u) 步 VAD+UTMOS 联合优化, 再用差额步数
            # 单独跑预算更多的那一边; Adam 状态在两阶段间复用 (lr 切换,
            # momentum/moment 保留)。
            n_shared = min(n_v, n_u)
            phases.append((n_shared, opt_lr, overlap_iter))
            if n_v > n_u:
                phases.append((n_v - n_u, opt_lr, kit["iter_vad"]))
            elif n_u > n_v:
                phases.append((n_u - n_v, u_lr, kit["iter_utmos"]))
        elif n_v > 0:
            phases.append((n_v, opt_lr, kit["iter_vad"]))
        elif n_u > 0:
            phases.append((n_u, u_lr, kit["iter_utmos"]))
        x_t = kit["tto_step"](x_t, t_cur, phases, i)
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

    # ---- segmented (block-attn) inputs --------------------------------
    parser.add_argument("--use-attn-mask", action="store_true",
                        help="启用分段 block attention mask；ref-audio/ref-text/gen-text 按 --segment-delimiter 切分。")
    parser.add_argument("--segment-delimiter", default="||",
                        help="--use-attn-mask 下用于切分多段 ref-audio/ref-text/gen-text 的分隔符")
    parser.add_argument("--gen-durations", default=None,
                        help="可选：--use-attn-mask 时每段生成 mel 帧数，按 --segment-delimiter 分隔；空则按文本长度比例估计")

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
    parser.add_argument("--vad-level", choices=("frame", "utter", "both"), default="both",
                        help="frame=framewise MSE; utter=utterance-level MSE; both=两者求和")

    # ---- UTMOS loss / TTO knobs (与 VAD 解耦) -------------------------
    parser.add_argument("--utmos-weight", type=float, default=0.0,
                        help=">0 时启用 UTMOS naturalness loss = w*(5 - MOS)")
    parser.add_argument("--utmos-opt-at", default="",
                        help="UTMOS TTO 触发的 ODE step 索引；空=不做 UTMOS opt（与 --opt-at 重叠时合并 loss 单次 Adam）")
    parser.add_argument("--utmos-opt-steps", type=int, default=None,
                        help="UTMOS TTO 每个点 Adam 内迭代步数；默认 fall back 到 --opt-steps")
    parser.add_argument("--utmos-opt-lr", type=float, default=None,
                        help="UTMOS TTO Adam 学习率；默认 fall back 到 --opt-lr")
    parser.add_argument("--grad-proj", choices=("none", "ortho", "pcgrad"), default="none",
                        help="VAD+UTMOS 重叠 step 合并阶段的梯度处理: "
                             "'none' 直接相加 (默认); "
                             "'ortho' 把 UTMOS 梯度投影到 VAD 梯度的正交补; "
                             "'pcgrad' 仅当 g_vad·g_utmos<0 时投影")

    # ---- I/O & batch --------------------------------------------------
    parser.add_argument("--output", default="tto_demo.wav",
                        help="单条: 输出文件路径; 批量: 输出目录")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="启用批量模式，从 --ref-dir 随机采样 N 条 ref 循环处理")
    parser.add_argument("--ref-dir", default="asset",
                        help="批量模式的 ref 音频目录")
    parser.add_argument("--ref-pattern", default="*.wav",
                        help="批量模式下的 ref 文件名 glob 模式")

    # ---- batch + attn_mask: 反查源 wav 的数据集根 ---------------------
    parser.add_argument("--ravdess-root", default="/mnt/disk1/datasets/RAVDESS",
                        help="批量 + --use-attn-mask 时反查 RAVDESS 源音频的根目录")
    parser.add_argument("--esd-root",
                        default="/mnt/disk1/datasets/ESD/Emotion Speech Dataset",
                        help="批量 + --use-attn-mask 时反查 ESD 源音频的根目录")
    parser.add_argument("--silence-db", type=float, default=-50.0,
                        help="批量 + --use-attn-mask 时对源 ref 静音裁剪阈值 (与 emotion_concat 一致)")
    args = parser.parse_args()

    import glob as _glob
    import pathlib
    import random as _random
    import re as _re

    # ====================================================================
    # job list: batch-sampled refs or a single ref
    # ====================================================================
    if args.batch_size is not None:
        candidates = sorted(_glob.glob(os.path.join(args.ref_dir, args.ref_pattern)))
        if not candidates:
            raise SystemExit(f"--ref-dir '{args.ref_dir}' 下没有匹配 '{args.ref_pattern}' 的文件")
        n = min(args.batch_size, len(candidates))
        if n < args.batch_size:
            print(f"提示: 目录仅 {len(candidates)} 个文件 < batch-size {args.batch_size}, 实际采 {n} 个")
        ref_paths = _random.Random(args.seed).sample(candidates, n)
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
    else:
        ref_paths = [args.ref_audio]
        out_dir = None

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
    utmos_model = load_utmos(device) if args.utmos_weight > 0 else None
    if utmos_model is not None:
        print(f"UTMOS loss enabled: weight={args.utmos_weight}")

    opt_schedule = [int(s) for s in args.opt_at.split(",") if s.strip()]
    utmos_opt_schedule = [int(s) for s in args.utmos_opt_at.split(",") if s.strip()]
    if utmos_opt_schedule and utmos_model is None:
        raise SystemExit("--utmos-opt-at 非空但 --utmos-weight=0；请同时设置两者")
    if args.utmos_weight > 0 and not utmos_opt_schedule:
        print("提示: --utmos-weight > 0 但 --utmos-opt-at 为空 → UTMOS loss 不会被优化（仅记录）")

    # ====================================================================
    # local helpers
    # ====================================================================

    def _split_segments(value: str) -> list[str]:
        return [x.strip() for x in value.split(args.segment_delimiter) if x.strip()]

    def _parse_gen_durations() -> list[int] | None:
        if args.gen_durations is None:
            return None
        values = _split_segments(args.gen_durations)
        if not values:
            return None
        return [int(x) for x in values]

    def _load_mono(path: str) -> tuple[torch.Tensor, int]:
        wav, wav_sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, wav_sr

    def _resample_to_target(wav: torch.Tensor, wav_sr: int) -> torch.Tensor:
        if wav_sr == target_sample_rate:
            return wav
        return torchaudio.transforms.Resample(wav_sr, target_sample_rate)(wav)

    def _normalize_ref_text(ref_text: str) -> str:
        if len(ref_text.encode("utf-8")) and len(ref_text[-1].encode("utf-8")) == 1:
            return ref_text + " "
        return ref_text

    def _split_asset_to_pair(asset_path: str) -> list[tuple[torch.Tensor, int]]:
        """根据 emotion_concat.py 输出的拼接 ref 文件名反查源 2 段，做同款静音裁剪后返回 [(wav, sr), (wav, sr)]。

        命名约定 (emotion_concat.py):
          RAVDESS: actor{AA}_{emo1}-{int1}_to_{emo2}-{int2}.wav
          ESD:     spk{SSSS}_{emo1}-{tag1}_to_{emo2}-{tag2}.wav
                   tag = "i{idx}" 或末 6 位数字 utt id
        """
        EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
        INT = {"01": "normal", "02": "strong"}
        INV_EMO = {v: k for k, v in EMO.items()}
        INV_INT = {v: k for k, v in INT.items()}

        def _find_ravdess(actor: str, emo_code: str, int_code: str) -> str:
            pat = os.path.join(args.ravdess_root, f"Actor_{actor}",
                               f"03-01-{emo_code}-{int_code}-*-*-{actor}.wav")
            return sorted(_glob.glob(pat))[0]

        def _find_esd(speaker: str, emo_name: str, tag: str) -> str:
            emo_dir = os.path.join(args.esd_root, speaker, emo_name)
            if tag.startswith("i"):
                return sorted(_glob.glob(os.path.join(emo_dir, "*.wav")))[int(tag[1:])]
            return os.path.join(emo_dir, f"{speaker}_{tag}.wav")

        def _trim_silence(wav: torch.Tensor, threshold_db: float) -> torch.Tensor:
            energy = 20 * torch.log10(wav.abs().clamp(min=1e-10))
            mask = (energy > threshold_db).squeeze(0)
            nz = torch.nonzero(mask)
            if len(nz) == 0:
                return wav
            return wav[:, nz[0].item(): nz[-1].item() + 1]

        stem = pathlib.Path(asset_path).stem
        m = _re.match(r"actor(\d+)_(\w+)-(\w+)_to_(\w+)-(\w+)$", stem)
        if m:
            actor, e1, i1, e2, i2 = m.groups()
            paths = [_find_ravdess(actor, INV_EMO[e1], INV_INT[i1]),
                     _find_ravdess(actor, INV_EMO[e2], INV_INT[i2])]
        else:
            m = _re.match(r"spk(\d{4})_([a-z]+)-(\w+)_to_([a-z]+)-(\w+)$", stem)
            spk, e1, t1, e2, t2 = m.groups()
            paths = [_find_esd(spk, e1.capitalize(), t1),
                     _find_esd(spk, e2.capitalize(), t2)]

        out: list[tuple[torch.Tensor, int]] = []
        for p in paths:
            wav, sr = torchaudio.load(p)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = _trim_silence(wav, args.silence_db)
            out.append((wav, sr))
        return out

    # ====================================================================
    # per-ref runner
    # ====================================================================
    def _prepare_attn_mask_inputs(ref_audio_path, ref_pair_loaded):
        """Returns (cond, text, duration, lens, attn_block_mask, amp_scale,
        ref_wav_for_vad, sr) for the segmented (block-attn) path."""
        ref_text_segments = [_normalize_ref_text(x) for x in _split_segments(args.ref_text)]
        gen_text_segments = [_normalize_ref_text(x) for x in _split_segments(args.gen_text)]
        if ref_pair_loaded is not None:
            loaded_refs = ref_pair_loaded
        else:
            loaded_refs = [_load_mono(p) for p in _split_segments(ref_audio_path)]
        gen_durations = _parse_gen_durations()
        n_ref = len(loaded_refs)
        if not (n_ref == len(ref_text_segments) == len(gen_text_segments)):
            raise ValueError(
                f"--use-attn-mask: segment count mismatch — refs={n_ref} "
                f"ref_text={len(ref_text_segments)} gen_text={len(gen_text_segments)}"
            )
        if gen_durations is not None and len(gen_durations) != n_ref:
            raise ValueError("--gen-durations segment count must match the number of ref segments")

        # VAD reads the cat'd raw audio at the first ref's native sr (no RMS scaling).
        sr = loaded_refs[0][1]
        raw_refs_for_vad = [
            torchaudio.functional.resample(wav, wav_sr, sr) if wav_sr != sr else wav
            for wav, wav_sr in loaded_refs
        ]
        ref_wav_for_vad = torch.cat(raw_refs_for_vad, dim=-1).to(device)

        # CFM conditioning: RMS-normalize the concatenated ref so all segments share one scale.
        ref_wav_concat = torch.cat(
            [_resample_to_target(wav, wav_sr) for wav, wav_sr in loaded_refs], dim=-1,
        )
        rms = torch.sqrt(torch.mean(ref_wav_concat.square()))
        if rms < target_rms:
            amp_scale, scale = float(rms / target_rms), target_rms / rms
        else:
            amp_scale, scale = 1.0, 1.0
        ref_wavs_for_cfm = [
            (_resample_to_target(wav, wav_sr) * scale).to(device)
            for wav, wav_sr in loaded_refs
        ]
        tto_inputs = prepare_segmented_tto_inputs(
            cfm=tts.ema_model,
            ref_wavs=ref_wavs_for_cfm,
            ref_texts=ref_text_segments,
            gen_texts=gen_text_segments,
            gen_durations=gen_durations,
            device=device,
        )
        print(
            f"segmented attn mask enabled: segments={n_ref} "
            f"ref_lens={tto_inputs['ref_lens']} gen_lens={tto_inputs['gen_lens']}"
        )
        return (tto_inputs["cond"], tto_inputs["text"], tto_inputs["duration"],
                tto_inputs["lens"], tto_inputs["attn_block_mask"],
                amp_scale, ref_wav_for_vad, sr)

    def _prepare_classic_inputs(ref_audio_path):
        """Returns (cond, text, duration, lens, attn_block_mask, amp_scale,
        ref_wav_for_vad, sr) for the single-ref path."""
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

        ref_text = _normalize_ref_text(args.ref_text)
        text = convert_char_to_pinyin([ref_text + args.gen_text])

        ref_audio_len = ref_wav.shape[-1] // hop_length
        ref_text_len = max(len(ref_text.encode("utf-8")), 1)
        gen_text_len = len(args.gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
        return (ref_wav, text, duration, None, None,
                amp_scale, ref_wav_for_vad, sr)

    def _run_one(ref_audio_path: str, out_path: str,
                 ref_pair_loaded: list[tuple[torch.Tensor, int]] | None = None) -> None:
        if args.use_attn_mask:
            cond, text, duration, lens, attn_block_mask, amp_scale, ref_wav_for_vad, sr = \
                _prepare_attn_mask_inputs(ref_audio_path, ref_pair_loaded)
        else:
            cond, text, duration, lens, attn_block_mask, amp_scale, ref_wav_for_vad, sr = \
                _prepare_classic_inputs(ref_audio_path)

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

        def _log(step_idx, it, loss, *, loss_vad=None, loss_utmos=None):
            if loss_vad is not None and loss_utmos is not None:
                tag = "vad+utmos"
            elif loss_vad is not None:
                tag = "vad"
            else:
                tag = "utmos"
            parts = [f"loss={loss:.6f}"]
            if loss_vad is not None:
                parts.append(f"vad={loss_vad:.6f}")
            if loss_utmos is not None:
                parts.append(f"utmos={loss_utmos:.6f}")
            print(f"[TTO][{tag}] step={step_idx:02d} iter={it} " + " ".join(parts))

        wav, _ = sample_with_tto(
            cfm=tts.ema_model,
            vocoder=tts.vocoder,
            vad=vad,
            ref_vad_features=ref_vad,
            cond=cond,
            text=text,
            duration=duration,
            lens=lens,
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
            attn_block_mask=attn_block_mask,
            utmos=utmos_model,
            utmos_weight=args.utmos_weight,
            utmos_opt_schedule=utmos_opt_schedule,
            utmos_opt_steps=args.utmos_opt_steps,
            utmos_opt_lr=args.utmos_opt_lr,
            grad_proj=args.grad_proj,
        )
        wav_np = wav.squeeze().detach().float().cpu().numpy()
        sf.write(out_path, wav_np, target_sample_rate, subtype="FLOAT")
        print(f"saved: {out_path}  ({wav_np.shape[-1] / target_sample_rate:.2f}s)")

    # ====================================================================
    # main loop (single or batch)
    # ====================================================================
    total = len(ref_paths)
    for idx, rp in enumerate(ref_paths, 1):
        if out_dir is not None:
            stem = pathlib.Path(rp).stem
            out_path = os.path.join(out_dir, f"{stem}.wav")
            print(f"\n=== [{idx}/{total}] ref={rp} ===")
        else:
            out_path = args.output
        if args.batch_size is not None and args.use_attn_mask:
            pair = _split_asset_to_pair(rp)
            print(f"  pair sources: {[p[0].shape[-1] for p in pair]} samples (post-trim) "
                  f"@ sr={[p[1] for p in pair]}")
            _run_one(rp, out_path, ref_pair_loaded=pair)
        else:
            _run_one(rp, out_path)

    if total > 1:
        print(f"\n批量完成: {total} 条输出保存在 {out_dir}")
