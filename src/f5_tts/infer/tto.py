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

from typing import Callable, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor
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


_VAD_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_VAD_SR = 16000


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
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


class GradVADExtractor(nn.Module):
    """Differentiable counterpart of ``VAD_extractor.process_func_framewise``.

    Shares weights with ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``
    (frozen). Accepts a waveform tensor at ``in_sr`` Hz, resamples to 16 kHz,
    applies zero-mean / unit-var normalization, and extracts frame-level
    features with a sliding window. Gradients flow through the pipeline back
    to the input waveform.
    """

    def __init__(
        self,
        in_sr: int = 24000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_sr = in_sr
        self._processor = Wav2Vec2Processor.from_pretrained(_VAD_MODEL_NAME)
        model = _EmotionModel.from_pretrained(_VAD_MODEL_NAME)
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.do_normalize = getattr(
            self._processor.feature_extractor, "do_normalize", True,
        )
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
    ) -> torch.Tensor:
        """Return frame-level features for ``wav``.

        ``wav`` may be ``(T,)`` or ``(B, T)``. Output is ``(N, D)`` or
        ``(B, N, D)`` respectively, where ``D`` is ``hidden_size`` when
        ``embeddings=True`` and 3 otherwise (arousal/dominance/valence).
        """
        squeeze_batch = wav.ndim == 1
        if squeeze_batch:
            wav = wav.unsqueeze(0)
        in_sr = in_sr if in_sr is not None else self.in_sr
        wav = self._to_16k(wav, in_sr)

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
        feat = pooled if embeddings else logits
        feat = feat.view(B, num_frames, -1)
        return feat.squeeze(0) if squeeze_batch else feat


@torch.no_grad()
def precompute_reference_vad(
    vad: GradVADExtractor,
    ref_wav: torch.Tensor,
    *,
    in_sr: int,
    window_size: float = 1.0,
    hop_size: float = 0.25,
    embeddings: bool = False,
) -> torch.Tensor:
    """Extract reference frame-level VAD once, without gradients."""
    return vad(
        ref_wav,
        in_sr=in_sr,
        window_size=window_size,
        hop_size=hop_size,
        embeddings=embeddings,
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
    on_opt_step: Callable[[int, int, float], None] | None = None,
):
    """Sample from ``cfm`` with test-time optimization on intermediate latents.

    ``opt_schedule``: iterable of ODE grid indices (each uses ``opt_steps``
    inner Adam iterations) or a mapping ``{step_idx: inner_iters}``. The index
    is the position *before* the Euler step (``t_grid[i] -> t_grid[i+1]``).

    ``ref_vad_features``: frame-level features from
    :func:`precompute_reference_vad` (match ``loss_mode``, ``window_size``,
    ``hop_size``). Shape ``(N_ref, D)`` or ``(B, N_ref, D)``.

    ``on_opt_step``: optional callback ``(step_idx, iter_idx, loss)`` invoked
    after each inner update — useful for logging.
    """
    if loss_mode not in ("value", "embedding"):
        raise ValueError("loss_mode must be 'value' or 'embedding'")
    if vocoder_type not in ("vocos", "bigvgan"):
        raise ValueError(f"unknown vocoder_type: {vocoder_type}")
    if isinstance(opt_schedule, Mapping):
        sched = {int(k): int(v) for k, v in opt_schedule.items()}
    else:
        sched = {int(k): int(opt_steps) for k in opt_schedule}

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
    # single-item assumption for ref slicing (same as utils_infer.py)
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

    def _x1_to_wav(x1: torch.Tensor) -> torch.Tensor:
        out = torch.where(cond_mask, cond, x1)
        gen = out[:, ref_audio_len:, :].permute(0, 2, 1)
        return vocoder.decode(gen) if vocoder_type == "vocos" else vocoder(gen)

    embeddings_flag = (loss_mode == "embedding")

    def _vad_loss(wav: torch.Tensor) -> torch.Tensor:
        vad_in = wav.squeeze(0) if wav.shape[0] == 1 else wav
        gen_vad = vad(
            vad_in, in_sr=sample_rate,
            window_size=window_size, hop_size=hop_size,
            embeddings=embeddings_flag,
        )
        if gen_vad.ndim == 3:
            ref = ref_vad_features
            if ref.ndim == 2:
                ref = ref.unsqueeze(0).expand(gen_vad.shape[0], -1, -1)
            losses = []
            for b in range(gen_vad.shape[0]):
                g, r = _align_frames(gen_vad[b], ref[b])
                losses.append(F.mse_loss(g, r))
            return torch.stack(losses).mean()
        g, r = _align_frames(gen_vad, ref_vad_features)
        return F.mse_loss(g, r)

    def _tto_step(x_t: torch.Tensor, t_cur: torch.Tensor, n_iters: int, step_idx: int) -> torch.Tensor:
        x_var = x_t.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_var], lr=opt_lr)
        for it in range(n_iters):
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                v = _predict_flow(x_var, t_cur, opt_cfg_strength)
                x1_hat = x_var + (1.0 - t_cur) * v
                wav = _x1_to_wav(x1_hat)
                loss = _vad_loss(wav)
            loss.backward()
            optimizer.step()
            if on_opt_step is not None:
                on_opt_step(step_idx, it, float(loss.detach()))
        cfm.transformer.clear_cache()
        return x_var.detach()

    # --- manual Euler ODE loop with optional TTO at each grid point ---
    x_t = y0
    for i in range(steps):
        t_cur = t_grid[i]
        t_next = t_grid[i + 1]
        if sched.get(i, 0) > 0:
            x_t = _tto_step(x_t, t_cur, sched[i], i)
        with torch.no_grad():
            v = _predict_flow(x_t, t_cur, cfg_strength)
            x_t = x_t + (t_next - t_cur) * v

    cfm.transformer.clear_cache()

    sampled = torch.where(cond_mask, cond, x_t)
    with torch.no_grad():
        gen_mel = sampled[:, ref_audio_len:, :].permute(0, 2, 1)
        wav = vocoder.decode(gen_mel) if vocoder_type == "vocos" else vocoder(gen_mel)
    return wav, sampled


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="tto_demo.wav")
    args = parser.parse_args()

    tts = F5TTS(model=args.model)
    device = tts.device

    # Load reference audio, mono-mix, RMS-normalize, resample to 24 kHz.
    ref_wav, sr = torchaudio.load(args.ref_audio)
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(ref_wav.square()))
    if rms < target_rms:
        ref_wav = ref_wav * target_rms / rms
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

    vad = GradVADExtractor(in_sr=target_sample_rate, device=device)
    ref_vad = precompute_reference_vad(
        vad, ref_wav.squeeze(0),
        in_sr=target_sample_rate,
        window_size=args.window_size, hop_size=args.hop_size,
        embeddings=(args.loss_mode == "embedding"),
    )
    print(f"ref VAD features: {tuple(ref_vad.shape)}  (loss_mode={args.loss_mode})")

    opt_schedule = [int(s) for s in args.opt_at.split(",") if s.strip()]

    def _log(step_idx, it, loss):
        print(f"[TTO] step={step_idx:02d} iter={it} loss={loss:.6f}")

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
        vocoder_type=tts.mel_spec_type,
        sample_rate=target_sample_rate,
        on_opt_step=_log,
    )
    wav_np = wav.squeeze().detach().float().cpu().numpy()
    sf.write(args.output, wav_np, target_sample_rate)
    print(f"saved: {args.output}  ({wav_np.shape[-1] / target_sample_rate:.2f}s)")