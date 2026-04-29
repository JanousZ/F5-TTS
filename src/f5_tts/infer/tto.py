"""Test-time optimization (TTO) for F5-TTS latent code.

At selected ODE steps during sampling, optimize the current latent ``x_t`` so
the one-step ``x1`` estimate, after vocoder decoding, matches the frame-level
VAD (valence/arousal/dominance) trajectory of a reference waveform. Supports
either the regression-head values (``loss_mode='value'``) or the pooled
wav2vec2 hidden states (``loss_mode='embedding'``) as the target.

Only ``x_t`` is optimized — the CFM transformer, vocoder, and VAD encoder stay
frozen. Integration uses a manual Euler loop so TTO can be injected at any
grid index; this matches ``CFM.sample``'s default ``method='euler'``.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable, Mapping

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
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


_VAD_MODEL_NAME = '/mnt/disk1/models/wav2vec2-large-robust-12-ft-emotion-msp-dim'
_VAD_SR = 16000
# wav2vec2-large 的 conv stack 总 stride = 320，输入 16 kHz → 输出 ~50 fps
# (20 ms / frame)。hidden-mode 滑窗用这个把秒数换算成 frame 数。
_VAD_FRAME_HZ = 50


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

    Frozen backbone; gradients flow back to input wav. Two implementations
    selectable via ``slide_mode``:

      ``audio`` (legacy): unfold raw 16 kHz audio into windows, run the FULL
        model (wav2vec2 + mean + classifier) on EACH window. Each frame's
        attention only sees its own ~1 s chunk → out-of-distribution input
        for the classifier (which was trained on whole utterances), and N×
        backbone forwards.

      ``hidden`` (default, recommended): one wav2vec2 forward over the full
        audio → ``(T_frame, 1024)`` hidden state → unfold over the time axis
        → mean per window → classifier per window. Frames see the full
        attention context (in-distribution), and only one backbone forward.
    """

    def __init__(
        self,
        in_sr: int = 24000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        slide_mode: str = "hidden",
    ):
        super().__init__()
        if slide_mode not in ("audio", "hidden"):
            raise ValueError(f"slide_mode must be 'audio' or 'hidden', got {slide_mode!r}")
        self.slide_mode = slide_mode
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
        return_both: bool = False,
        pad: bool = True,
        utter: bool = False,
    ):
        """Return frame-level features for ``wav``.

        ``wav`` may be ``(T,)`` or ``(B, T)``. Default output is ``(N, D)`` /
        ``(B, N, D)`` where ``D = hidden_size`` when ``embeddings=True`` and 3
        otherwise (arousal/dominance/valence). If ``return_both=True``, returns
        ``(emb, val)`` with shapes ``(..., hidden_size)`` and ``(..., 3)``
        sharing a single wav2vec2 forward. When ``utter=True``, skips framing
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
            if return_both:
                emb, val = pooled.unsqueeze(1), logits.unsqueeze(1)
                if squeeze_batch:
                    return emb.squeeze(0), val.squeeze(0)
                return emb, val
            feat = (pooled if embeddings else logits).unsqueeze(1)  # (B, 1, D)
            return feat.squeeze(0) if squeeze_batch else feat

        if self.slide_mode == "audio":
            emb, val = self._frame_features_audio_slide(
                wav, window_size=window_size, hop_size=hop_size, pad=pad,
            )
        else:  # "hidden"
            emb, val = self._frame_features_hidden_slide(
                wav, window_size=window_size, hop_size=hop_size, pad=pad,
            )
        if return_both:
            if squeeze_batch:
                emb, val = emb.squeeze(0), val.squeeze(0)
            return emb, val
        feat = emb if embeddings else val
        return feat.squeeze(0) if squeeze_batch else feat

    # ---- internal helpers ------------------------------------------------

    def _frame_features_audio_slide(
        self, wav: torch.Tensor, *, window_size: float, hop_size: float, pad: bool,
    ):
        """Legacy: unfold raw audio into windows, run full model on each."""
        win = int(round(window_size * _VAD_SR))
        hop = int(round(hop_size * _VAD_SR))
        if win <= 0 or hop <= 0:
            raise ValueError("window_size and hop_size must be positive")

        B, T = wav.shape
        if T < win:
            if not pad:
                raise ValueError(f"signal shorter ({T}) than window ({win})")
            wav = F.pad(wav, (0, win - T))
        elif pad:
            rem = (T - win) % hop
            if rem != 0:
                wav = F.pad(wav, (0, hop - rem))

        frames = wav.unfold(dimension=-1, size=win, step=hop)  # (B, N, win)
        num_frames = frames.shape[1]
        frames = frames.reshape(B * num_frames, win)

        if self.do_normalize:
            frames = self._normalize(frames)
        frames = frames.to(next(self.model.parameters()).dtype)

        pooled, logits = self.model(frames)
        emb = pooled.view(B, num_frames, -1)
        val = logits.view(B, num_frames, -1)
        return emb, val

    def _frame_features_hidden_slide(
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


def _save_vad_viz(
    base_path: str,
    records: list[tuple[int, int, torch.Tensor]],
    ref_vad: torch.Tensor,
    final_vad: torch.Tensor | None = None,
    *,
    ref_utter: torch.Tensor | None = None,
    records_utter: list[tuple[int, int, torch.Tensor]] | None = None,
    final_utter: torch.Tensor | None = None,
) -> None:
    """Save per-iter VAD trajectories as CSV + PNG.

    ``records`` is a list of ``(ode_step, opt_iter, gen_vad)`` where gen_vad
    is ``(N, 3)`` (arousal, dominance, valence). ``ref_vad`` is ``(N_ref, 3)``.
    ``final_vad`` is the VAD of the post-sampling final audio — shown as an
    extra solid line, distinct from the TTO trajectory (which stops at the
    last TTO iter, before the remaining pure-Euler steps).

    Utter-level counterparts (each ``(1, 3)``) — ``ref_utter`` / ``final_utter``
    / ``records_utter`` — are written to CSV only (no plot overlay).
    """
    import csv
    import matplotlib.pyplot as plt
    import numpy as np

    dim_names = ("arousal", "dominance", "valence")

    def _utter_row(x: torch.Tensor) -> list[float]:
        return x.view(-1).tolist()

    csv_path = base_path + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "ode_step", "opt_iter", "frame", *dim_names])
        for i, (a, d, v) in enumerate(ref_vad.tolist()):
            w.writerow(["ref", -1, -1, i, a, d, v])
        for step_idx, it, gv in records:
            for i, (a, d, v) in enumerate(gv.tolist()):
                w.writerow(["gen", step_idx, it, i, a, d, v])
        if final_vad is not None:
            for i, (a, d, v) in enumerate(final_vad.tolist()):
                w.writerow(["final", -1, -1, i, a, d, v])
        if ref_utter is not None:
            a, d, v = _utter_row(ref_utter)
            w.writerow(["ref_utter", -1, -1, 0, a, d, v])
        if records_utter:
            for step_idx, it, uv in records_utter:
                a, d, v = _utter_row(uv)
                w.writerow(["gen_utter", step_idx, it, 0, a, d, v])
        if final_utter is not None:
            a, d, v = _utter_row(final_utter)
            w.writerow(["final_utter", -1, -1, 0, a, d, v])

    # each record labelled by cumulative iter, colored dark->light
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    n_frames_ref = ref_vad.shape[0]
    xs_ref = np.arange(n_frames_ref)
    cmap = plt.get_cmap("viridis")
    n_total = len(records)

    for d_idx, ax in enumerate(axes):
        ax.plot(xs_ref, ref_vad[:, d_idx].numpy(),
                color="red", lw=2.5, ls="--", label="ref", zorder=10)
        for k, (step_idx, it, gv) in enumerate(records):
            xs = np.linspace(0, n_frames_ref - 1, gv.shape[0])
            color = cmap(k / max(n_total - 1, 1))
            label = f"step={step_idx} iter={it}" if (k == 0 or k == n_total - 1) else None
            ax.plot(xs, gv[:, d_idx].numpy(), color=color, lw=1.2,
                    alpha=0.75, label=label)
        if final_vad is not None:
            xs_f = np.linspace(0, n_frames_ref - 1, final_vad.shape[0])
            ax.plot(xs_f, final_vad[:, d_idx].numpy(),
                    color="black", lw=2.2, ls="-", label="final (post-Euler)",
                    zorder=11)
        ax.set_ylabel(dim_names[d_idx])
        ax.grid(True, alpha=0.3)
        if d_idx == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("frame index (aligned to ref grid)")
    fig.suptitle("VAD trajectories across TTO iterations (dark→light = early→late)")
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=120)
    plt.close(fig)


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
    lens, steps, cfg_strength, sway_sampling_coef, seed,
    max_duration, use_epss, no_ref_audio, edit_mask,
    vocoder_type, sample_rate,
    opt_lr, opt_cfg_strength, loss_mode,
    window_size, hop_size,
    on_opt_step, on_opt_vad,
    vad_level: str = "frame",
    ref_vad_utter: torch.Tensor | None = None,
    on_opt_vad_utter: Callable[[int, int, torch.Tensor], None] | None = None,
    amp_scale: float = 1.0,
):
    """Shared setup + closures for sample_with_tto and its budget-sweep variant.

    Returns a dict containing: ``t_grid``, ``y0``, ``cond_mask``, ``cond``,
    ``ref_audio_len``, ``vocoder_dtype``, ``predict_flow``, ``x1_to_wav``,
    ``vad_loss``, ``tto_step``, ``tto_step_with_snapshots``.
    """
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

    # --- cond / text / duration prep mirrors CFM.sample ---
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

    mask = lens_to_mask(duration) if batch > 1 else None
    ref_audio_len = int(lens[0].item())

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

    embeddings_flag = (loss_mode == "embedding")
    _cur = {"step": None, "iter": None}

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

    def _vad_loss(wav: torch.Tensor) -> torch.Tensor:
        vad_in = wav.squeeze(0) if wav.shape[0] == 1 else wav
        total = None
        if vad_level in ("frame", "both"):
            want_viz = on_opt_vad is not None and _cur["step"] is not None
            if embeddings_flag and want_viz:
                gen_vad, gen_val = vad(
                    vad_in, in_sr=sample_rate,
                    window_size=window_size, hop_size=hop_size,
                    return_both=True,
                )
                on_opt_vad(_cur["step"], _cur["iter"], gen_val.detach())
            else:
                gen_vad = vad(
                    vad_in, in_sr=sample_rate,
                    window_size=window_size, hop_size=hop_size,
                    embeddings=embeddings_flag,
                )
                if want_viz:
                    on_opt_vad(_cur["step"], _cur["iter"], gen_vad.detach())
            total = _mse_against(gen_vad, ref_vad_features)
        if vad_level in ("utter", "both"):
            want_viz_u = on_opt_vad_utter is not None and _cur["step"] is not None
            if embeddings_flag and want_viz_u:
                utter_vad, utter_val = vad(
                    vad_in, in_sr=sample_rate, utter=True, return_both=True,
                )
                on_opt_vad_utter(_cur["step"], _cur["iter"], utter_val.detach())
            else:
                utter_vad = vad(
                    vad_in, in_sr=sample_rate, utter=True,
                    embeddings=embeddings_flag,
                )
                if want_viz_u:
                    on_opt_vad_utter(_cur["step"], _cur["iter"], utter_vad.detach())
            ref_u = ref_vad_utter if vad_level == "both" else ref_vad_features
            u_loss = _mse_against(utter_vad, ref_u)
            total = u_loss if total is None else total + u_loss
        return total

    def _tto_step(x_t: torch.Tensor, t_cur: torch.Tensor, n_iters: int, step_idx: int) -> torch.Tensor:
        orig_dtype = x_t.dtype
        x_var = x_t.detach().clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([x_var], lr=opt_lr)
        for it in range(n_iters):
            optimizer.zero_grad(set_to_none=True)
            cfm.transformer.clear_cache()
            _cur["step"], _cur["iter"] = step_idx, it
            with torch.enable_grad():
                v = _predict_flow(x_var.to(orig_dtype), t_cur, opt_cfg_strength)
                x1_hat = x_var + (1.0 - t_cur) * v.float()
                wav = _x1_to_wav(x1_hat)
                loss = _vad_loss(wav)
            loss.backward()
            optimizer.step()
            if on_opt_step is not None:
                on_opt_step(step_idx, it, float(loss.detach()))
        cfm.transformer.clear_cache()
        return x_var.detach().to(orig_dtype)

    def _tto_step_with_snapshots(x_t, t_cur, max_iters, step_idx, snap_levels):
        """Run Adam for max_iters; capture x_var snapshots at requested levels.

        ``snap_levels`` are 1-indexed iter counts: level L == snapshot taken
        AFTER L optimizer steps. Returns ``{L: x_var.detach().clone()}``.
        """
        snap_set = set(int(x) for x in snap_levels)
        orig_dtype = x_t.dtype
        x_var = x_t.detach().clone().float().requires_grad_(True)
        optimizer = torch.optim.Adam([x_var], lr=opt_lr)
        snapshots: dict[int, torch.Tensor] = {}
        for it in range(1, int(max_iters) + 1):
            optimizer.zero_grad(set_to_none=True)
            cfm.transformer.clear_cache()
            _cur["step"], _cur["iter"] = step_idx, it - 1
            with torch.enable_grad():
                v = _predict_flow(x_var.to(orig_dtype), t_cur, opt_cfg_strength)
                x1_hat = x_var + (1.0 - t_cur) * v.float()
                wav = _x1_to_wav(x1_hat)
                loss = _vad_loss(wav)
            loss.backward()
            optimizer.step()
            if on_opt_step is not None:
                on_opt_step(step_idx, it - 1, float(loss.detach()))
            if it in snap_set:
                snapshots[it] = x_var.detach().clone().to(orig_dtype)
        cfm.transformer.clear_cache()
        return snapshots

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
        "tto_step": _tto_step,
        "tto_step_with_snapshots": _tto_step_with_snapshots,
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
    on_opt_vad: Callable[[int, int, torch.Tensor], None] | None = None,
    on_opt_vad_utter: Callable[[int, int, torch.Tensor], None] | None = None,
    amp_scale: float = 1.0,
):
    """Sample from ``cfm`` with test-time optimization on intermediate latents.

    ``opt_schedule``: iterable of ODE grid indices (each uses ``opt_steps``
    inner Adam iterations) or a mapping ``{step_idx: inner_iters}``. The index
    is the position *before* the Euler step (``t_grid[i] -> t_grid[i+1]``).

    ``ref_vad_features``: frame-level features from
    :func:`precompute_reference_vad` (match ``loss_mode``, ``window_size``,
    ``hop_size``). Shape ``(N_ref, D)`` or ``(B, N_ref, D)``.

    ``amp_scale``: multiplier applied to vocoder output (both in VAD loss and
    final return) to bring generated audio from target_rms back to the
    reference's original rms domain. Pass ``rms_ref / target_rms`` when the
    CFM condition was RMS-normalized; leave ``1.0`` otherwise.

    ``on_opt_step``: optional callback ``(step_idx, iter_idx, loss)`` invoked
    after each inner update — useful for logging.
    """
    if isinstance(opt_schedule, Mapping):
        sched = {int(k): int(v) for k, v in opt_schedule.items()}
    else:
        sched = {int(k): int(opt_steps) for k in opt_schedule}

    kit = _tto_inference_kit(
        cfm, vocoder, vad, ref_vad_features, cond, text, duration,
        lens=lens, steps=steps, cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef, seed=seed,
        max_duration=max_duration, use_epss=use_epss,
        no_ref_audio=no_ref_audio, edit_mask=edit_mask,
        vocoder_type=vocoder_type, sample_rate=sample_rate,
        opt_lr=opt_lr, opt_cfg_strength=opt_cfg_strength, loss_mode=loss_mode,
        window_size=window_size, hop_size=hop_size,
        vad_level=vad_level, ref_vad_utter=ref_vad_utter,
        on_opt_step=on_opt_step, on_opt_vad=on_opt_vad,
        on_opt_vad_utter=on_opt_vad_utter,
        amp_scale=amp_scale,
    )
    t_grid = kit["t_grid"]

    x_t = kit["y0"]
    for i in range(steps):
        t_cur = t_grid[i]
        t_next = t_grid[i + 1]
        if sched.get(i, 0) > 0:
            x_t = kit["tto_step"](x_t, t_cur, sched[i], i)
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


def sample_with_tto_budget_sweep(
    cfm,
    vocoder: Callable,
    vad: GradVADExtractor,
    ref_vad_features: torch.Tensor,
    cond: torch.Tensor,
    text,
    duration,
    *,
    opt_at: tuple[int, ...],
    per_point_iters_levels: Iterable[int],
    opt_lr: float = 1e-2,
    opt_cfg_strength: float = 1.0,
    loss_mode: str = "value",
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
    window_size: float = 1.0,
    hop_size: float = 0.25,
    vad_level: str = "frame",
    ref_vad_utter: torch.Tensor | None = None,
    on_opt_step: Callable[[int, int, float], None] | None = None,
    amp_scale: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Budget sweep: share TTO work at the first point across multiple per-point
    iter levels.

    Runs the ODE once; at ``opt_at[0]`` performs TTO for ``max(per_point_iters_levels)``
    Adam iters, snapshotting ``x_var`` at each level. For each snapshot level
    ``L``, the rest of the ODE (Euler + TTO at ``opt_at[1:]`` using ``L`` iters
    each) is completed independently and decoded.

    Returns ``{L: wav}`` keyed by per-point iter count. Multi-point schedules
    still re-run TTO at ``opt_at[1:]`` per level (those blocks cannot be shared),
    so savings are largest when ``len(opt_at) == 1``.
    """
    if not opt_at:
        raise ValueError("opt_at must be non-empty")
    levels = sorted(set(int(x) for x in per_point_iters_levels))
    if not levels or levels[0] < 1:
        raise ValueError("per_point_iters_levels must all be >= 1")

    kit = _tto_inference_kit(
        cfm, vocoder, vad, ref_vad_features, cond, text, duration,
        lens=lens, steps=steps, cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef, seed=seed,
        max_duration=max_duration, use_epss=use_epss,
        no_ref_audio=no_ref_audio, edit_mask=edit_mask,
        vocoder_type=vocoder_type, sample_rate=sample_rate,
        opt_lr=opt_lr, opt_cfg_strength=opt_cfg_strength, loss_mode=loss_mode,
        window_size=window_size, hop_size=hop_size,
        vad_level=vad_level, ref_vad_utter=ref_vad_utter,
        on_opt_step=on_opt_step, on_opt_vad=None,
        amp_scale=amp_scale,
    )
    t_grid = kit["t_grid"]
    first_pt = int(opt_at[0])
    max_iters = levels[-1]

    x_t = kit["y0"]
    for i in range(first_pt):
        t_cur = t_grid[i]
        t_next = t_grid[i + 1]
        with torch.no_grad():
            v = kit["predict_flow"](x_t, t_cur, cfg_strength)
            x_t = x_t + (t_next - t_cur) * v

    snapshots = kit["tto_step_with_snapshots"](
        x_t, t_grid[first_pt], max_iters, first_pt, levels,
    )

    remaining_pts = set(int(p) for p in opt_at[1:])
    results: dict[int, torch.Tensor] = {}
    for L in levels:
        x_t = snapshots[L]
        t_cur = t_grid[first_pt]
        t_next = t_grid[first_pt + 1]
        with torch.no_grad():
            v = kit["predict_flow"](x_t, t_cur, cfg_strength)
            x_t = x_t + (t_next - t_cur) * v
        for i in range(first_pt + 1, steps):
            t_cur = t_grid[i]
            t_next = t_grid[i + 1]
            if i in remaining_pts:
                x_t = kit["tto_step"](x_t, t_cur, L, i)
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
        results[L] = wav

    return results


if __name__ == "__main__":
    # End-to-end demo: load F5-TTS + vocoder, build reference VAD, then run
    # sample_with_tto with TTO at two intermediate steps.
    #
    # Run from repo root:
    #   python src/f5_tts/infer/tto.py --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
    #       --ref-text "Some call me nature, others call me mother nature." \
    #       --gen-text "I don't really care what you call me." \
    #       --loss-mode value --opt-at 16,24 --opt-steps 3 --opt-lr 1e-2 \
    #       --output tto_demo.wav
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

    parser = argparse.ArgumentParser(description="F5-TTS TTO demo")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument(
        "--ckpt-file", default="",
        help="本地 CFM 权重 (.safetensors / .pt)；空则走 HF cache",
    )
    parser.add_argument(
        "--vocab-file", default="",
        help="本地 vocab.txt；空则用包内默认",
    )
    parser.add_argument(
        "--vocoder-local-path", default="",
        help="本地 vocoder 目录 (vocos: 含 config.yaml + pytorch_model.bin)；空则走 HF",
    )
    parser.add_argument(
        "--ref-audio",
        default=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
    )
    parser.add_argument(
        "--ref-text",
        default="Some call me nature, others call me mother nature.",
    )
    parser.add_argument(
        "--gen-text",
        default="I don't really care what you call me. I've been a silent spectator.",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-coef", type=float, default=-1.0)
    parser.add_argument(
        "--opt-at", default="16,24",
        help="Comma-separated ODE step indices where TTO is performed.",
    )
    parser.add_argument("--opt-steps", type=int, default=3)
    parser.add_argument("--opt-lr", type=float, default=1e-2)
    parser.add_argument("--opt-cfg-strength", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode", choices=("value", "embedding"), default="value",
    )
    parser.add_argument("--window-size", type=float, default=1.0)
    parser.add_argument("--hop-size", type=float, default=0.25)
    parser.add_argument(
        "--vad-level", choices=("frame", "utter", "both"), default="frame",
        help="VAD loss level: 'frame' (framewise MSE), 'utter' "
             "(utterance-level MSE, mirrors VAD_extractor.process_func), or "
             "'both' (frame + utter).",
    )
    parser.add_argument(
        "--vad-slide-mode", choices=("audio", "hidden"), default="hidden",
        help="frame 模式下滑窗的实现: 'audio' 在原始音频上切窗(每窗独立 wav2vec2 "
             "forward, OOD 输入); 'hidden' 整段音频一次 forward 拿 hidden state, "
             "再在时间轴上滑窗(默认, in-distribution + 快几倍).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="tto_demo.wav",
                        help="单条: 输出文件路径; 批量: 输出目录")
    parser.add_argument(
        "--viz-path", default=None,
        help="单条: 形如 'vis' (产出 vis.csv/vis.png); 批量: 目录路径",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                        help="启用批量模式，从 --ref-dir 随机采样 N 条 ref 循环处理")
    parser.add_argument("--ref-dir", default="asset",
                        help="批量模式的 ref 音频目录")
    parser.add_argument("--ref-pattern", default="*.wav",
                        help="批量模式下的 ref 文件名 glob 模式")
    args = parser.parse_args()

    import glob as _glob
    import pathlib
    import random as _random

    # --- build job list: either batch-sampled refs or a single ref ---
    if args.batch_size is not None:
        candidates = sorted(_glob.glob(
            os.path.join(args.ref_dir, args.ref_pattern)
        ))
        if not candidates:
            print(f"错误: --ref-dir '{args.ref_dir}' 下没有匹配 '{args.ref_pattern}' 的文件")
            exit(1)
        n = min(args.batch_size, len(candidates))
        if n < args.batch_size:
            print(f"提示: 目录仅 {len(candidates)} 个文件 < batch-size {args.batch_size}, "
                  f"实际采 {n} 个")
        rng = _random.Random(args.seed)
        ref_paths = rng.sample(candidates, n)
        # batch mode: --output / --viz-path 视为目录
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        viz_dir = args.viz_path
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
    else:
        ref_paths = [args.ref_audio]
        out_dir = None
        viz_dir = None

    # --- load models ONCE ---
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
    opt_schedule = [int(s) for s in args.opt_at.split(",") if s.strip()]

    def _run_one(ref_audio_path: str, out_path: str, viz_base: str | None) -> None:
        # Load reference audio, mono-mix.
        ref_wav, sr = torchaudio.load(ref_audio_path)
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)

        # Raw copy at native sr (no RMS scaling) for VAD extraction — matches
        # VAD_extractor.process_func semantics on the original audio.
        ref_wav_for_vad = ref_wav.to(device)

        # CFM conditioning path: RMS-normalize + resample to 24 kHz. When the
        # ref was scaled up to target_rms, gen comes out in target_rms domain,
        # so we pass amp_scale = rms/target_rms back into sample_with_tto to
        # pull gen wav back to the ref's original amplitude (matches
        # utils_infer.py:514-515 and makes VAD loss same-domain vs ref VAD).
        rms = torch.sqrt(torch.mean(ref_wav.square()))
        if rms < target_rms:
            ref_wav = ref_wav * target_rms / rms
            amp_scale = float(rms / target_rms)
        else:
            amp_scale = 1.0
        if sr != target_sample_rate:
            ref_wav = torchaudio.transforms.Resample(sr, target_sample_rate)(ref_wav)
        ref_wav = ref_wav.to(device)

        ref_text = args.ref_text
        if len(ref_text.encode("utf-8")) and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        final_text_list = convert_char_to_pinyin([ref_text + args.gen_text])

        ref_audio_len = ref_wav.shape[-1] // hop_length
        ref_text_len = max(len(ref_text.encode("utf-8")), 1)
        gen_text_len = len(args.gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

        emb_flag = (args.loss_mode == "embedding")
        if args.vad_level == "utter":
            ref_vad = precompute_reference_vad(
                vad, ref_wav_for_vad.squeeze(0),
                in_sr=sr,
                embeddings=emb_flag, utter=True,
            )
            ref_vad_u = None
        else:
            ref_vad = precompute_reference_vad(
                vad, ref_wav_for_vad.squeeze(0),
                in_sr=sr,
                window_size=args.window_size, hop_size=args.hop_size,
                embeddings=emb_flag,
            )
            if args.vad_level == "both":
                ref_vad_u = precompute_reference_vad(
                    vad, ref_wav_for_vad.squeeze(0),
                    in_sr=sr,
                    embeddings=emb_flag, utter=True,
                )
            else:
                ref_vad_u = None
        print(
            f"ref VAD: primary={tuple(ref_vad.shape)} "
            f"utter={None if ref_vad_u is None else tuple(ref_vad_u.shape)}  "
            f"(loss_mode={args.loss_mode} vad_level={args.vad_level})\n"
            #f"ref_VAD: {ref_vad}"
        )

        if viz_base:
            ref_vad_val = precompute_reference_vad(
                vad, ref_wav_for_vad.squeeze(0),
                in_sr=sr,
                window_size=args.window_size, hop_size=args.hop_size,
                embeddings=False,
            )
            if args.vad_level in ("utter", "both"):
                ref_vad_val_utter = precompute_reference_vad(
                    vad, ref_wav_for_vad.squeeze(0),
                    in_sr=sr,
                    embeddings=False, utter=True,
                )
            else:
                ref_vad_val_utter = None
        else:
            ref_vad_val = None
            ref_vad_val_utter = None

        def _log(step_idx, it, loss):
            print(f"[TTO] step={step_idx:02d} iter={it} loss={loss:.6f}")

        viz_records: list[tuple[int, int, torch.Tensor]] = []
        viz_records_utter: list[tuple[int, int, torch.Tensor]] = []
        if viz_base:
            def _record(step_idx, it, gen_vad):
                viz_records.append((step_idx, it, gen_vad.float().cpu()))
            def _record_utter(step_idx, it, uv):
                viz_records_utter.append((step_idx, it, uv.float().cpu()))
        else:
            _record = None
            _record_utter = None

        wav, _ = sample_with_tto(
            cfm=tts.ema_model,
            vocoder=tts.vocoder,
            vad=vad,
            ref_vad_features=ref_vad,
            cond=ref_wav,
            text=final_text_list,
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
            on_opt_vad=_record,
            on_opt_vad_utter=_record_utter,
            amp_scale=amp_scale,
        )
        wav_np = wav.squeeze().detach().float().cpu().numpy()
        sf.write(out_path, wav_np, target_sample_rate, subtype="FLOAT")
        print(f"saved: {out_path}  ({wav_np.shape[-1] / target_sample_rate:.2f}s)")

        if viz_base:
            wav_for_vad = wav.squeeze(0) if wav.ndim == 2 else wav
            with torch.no_grad():
                final_vad = vad(
                    wav_for_vad,
                    in_sr=target_sample_rate,
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
            ref_utter_cpu = (ref_vad_val_utter.float().cpu()
                             if ref_vad_val_utter is not None else None)
            _save_vad_viz(
                viz_base, viz_records,
                ref_vad_val.float().cpu(), final_vad=final_vad,
                ref_utter=ref_utter_cpu,
                records_utter=viz_records_utter,
                final_utter=final_utter,
            )
            print(f"viz saved: {viz_base}.csv / {viz_base}.png")
            if ref_utter_cpu is not None or final_utter is not None:
                def _fmt(t):
                    a, d, v = t.view(-1).tolist()
                    return f"arousal={a:.4f} dominance={d:.4f} valence={v:.4f}"
                if ref_utter_cpu is not None:
                    print(f"utter VAD  ref  : {_fmt(ref_utter_cpu)}")
                if final_utter is not None:
                    print(f"utter VAD  final: {_fmt(final_utter)}")
                if ref_utter_cpu is not None and final_utter is not None:
                    d = (final_utter.view(-1) - ref_utter_cpu.view(-1)).abs()
                    print(f"utter VAD  |Δ| : arousal={d[0].item():.4f} "
                          f"dominance={d[1].item():.4f} valence={d[2].item():.4f}")

    # --- run (single or batch) ---
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
        _run_one(rp, out_path, viz_base)

    if total > 1:
        print(f"\n批量完成: {total} 条输出保存在 {out_dir}")