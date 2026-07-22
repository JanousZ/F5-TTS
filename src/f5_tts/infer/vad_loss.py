"""Differentiable VAD (valence/arousal/dominance) extractor for F5-TTS TTO.

Wraps ``wav2vec2-large-robust-12-ft-emotion-msp-dim`` so the forward
pass is differentiable end-to-end: input wav (tensor) → 16k resample →
zero-mean-unit-var norm → wav2vec2 backbone (one forward over full audio)
→ unfold along time → mean per window → classifier per window. Frames
see the full attention context and only one backbone forward is needed.

All parameters are frozen — only the gradient path is open. The
``precompute_reference_vad`` helper produces a frozen reference tensor
the TTO loop compares the generated wav's features against.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


VAD_MODEL_NAME = '/mnt/disk1/models/wav2vec2-large-robust-12-ft-emotion-msp-dim'
VAD_SR = 16000
# wav2vec2-large 的 conv stack 总 stride = 320，输入 16 kHz → 输出 ~50 fps
# (20 ms / frame)。滑窗用这个把秒数换算成 frame 数。
VAD_FRAME_HZ = 50


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
        model_path: str = VAD_MODEL_NAME,
    ):
        super().__init__()
        self.in_sr = in_sr
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        with open(os.path.join(model_path, 'config.json')) as f:
            cfg_dict = json.load(f)
        if cfg_dict.get('vocab_size') is None:
            cfg_dict['vocab_size'] = 32
        cfg = Wav2Vec2Config(**cfg_dict)
        model = _EmotionModel.from_pretrained(model_path, config=cfg)
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.do_normalize = getattr(self._processor, "do_normalize", True)
        self._default_resampler: nn.Module = (
            torchaudio.transforms.Resample(in_sr, VAD_SR)
            if in_sr != VAD_SR else nn.Identity()
        )
        if device is not None:
            self._default_resampler = self._default_resampler.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _to_16k(self, wav: torch.Tensor, in_sr: int) -> torch.Tensor:
        if in_sr == VAD_SR:
            return wav
        if in_sr == self.in_sr and isinstance(self._default_resampler, torchaudio.transforms.Resample):
            return self._default_resampler(wav)
        return torchaudio.functional.resample(wav, in_sr, VAD_SR)

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

        win = max(1, int(round(window_size * VAD_FRAME_HZ)))
        hop = max(1, int(round(hop_size * VAD_FRAME_HZ)))

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
