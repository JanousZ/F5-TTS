"""Differentiable emotion2vec_base_finetuned extractor for F5-TTS TTO.

Matches the model used by ``f5_tts.eval.eval_metric.compute_emotion_sim``
(emotion2vec_base_finetuned, modelscope). The eval path goes through a
FunASR pipeline whose forward is wrapped in ``torch.no_grad``; this
module loads the underlying ``Emotion2vec`` directly so gradients flow
back to the input wav.

Forward path: wav 16k mono → ``F.layer_norm`` (matches FunASR's
``inference``) → ``ConvFeatureExtractionModel`` (1D conv stack, ~50 fps)
→ 8 transformer blocks → ``(B, T_frame, 768)`` frame embedding.

All parameters are frozen — only the gradient PATH is open.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


EMO_MODEL_NAME = '/mnt/disk1/models/emotion2vec_base_finetuned'
EMO_SR = 16000


class GradEmoExtractor(nn.Module):
    """forward(wav, in_sr) → ``(B, T_frame, 768)`` frame embedding.

    Mirrors ``Emotion2vec.inference``:
      - resample to 16 kHz mono
      - optional ``F.layer_norm`` over the full waveform (when
        ``model.cfg.normalize=True``, which is the case for
        emotion2vec_base_finetuned)
      - ``model.extract_features(source, padding_mask=None)`` →
        ``feats['x']`` of shape ``(B, T_frame, embed_dim)``.

    Skips the 9-class classifier head — we want raw frame embeddings
    so cosine-similarity loss matches ``EMO-sim_frame`` from the eval.
    """

    def __init__(
        self,
        in_sr: int = 24000,
        device: torch.device | str | None = None,
        model_path: str = EMO_MODEL_NAME,
    ):
        super().__init__()
        from omegaconf import OmegaConf
        from funasr.models.emotion2vec.model import Emotion2vec

        self.in_sr = in_sr
        cfg = OmegaConf.load(os.path.join(model_path, 'config.yaml'))
        model_conf = OmegaConf.to_container(cfg.model_conf, resolve=True)

        # tokens.txt lines → vocab_size; lets the proj head's weights load
        # cleanly. We don't use the proj output, only the frame embeddings.
        with open(os.path.join(model_path, 'tokens.txt')) as f:
            vocab_size = sum(1 for line in f if line.strip())

        model = Emotion2vec(model_conf=model_conf, vocab_size=vocab_size)
        state = torch.load(
            os.path.join(model_path, 'emotion2vec_base.pt'),
            map_location='cpu',
            weights_only=False,
        )
        # strict=False: emotion2vec_base.pt is the full pretraining ckpt; the
        # finetune drops EMA/decoder tensors that aren't built when
        # `skip_ema=true` / training-only modules are absent. Missing-key
        # warnings here are expected and benign for inference.
        model.load_state_dict(state['model'], strict=False)

        # cfg.normalize controls the per-utterance F.layer_norm in inference.
        self._normalize = bool(model.cfg.get("normalize", True))

        for p in model.parameters():
            p.requires_grad_(False)
        if device is not None:
            model = model.to(device)
        self.model = model.eval()
        self._default_resampler: nn.Module = (
            torchaudio.transforms.Resample(in_sr, EMO_SR)
            if in_sr != EMO_SR else nn.Identity()
        )
        if device is not None:
            self._default_resampler = self._default_resampler.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _to_16k(self, wav: torch.Tensor, in_sr: int) -> torch.Tensor:
        if in_sr == EMO_SR:
            return wav
        if in_sr == self.in_sr and isinstance(self._default_resampler, torchaudio.transforms.Resample):
            return self._default_resampler(wav)
        return torchaudio.functional.resample(wav, in_sr, EMO_SR)

    def forward(self, wav: torch.Tensor, *, in_sr: int | None = None) -> torch.Tensor:
        """``wav`` may be ``(T,)`` or ``(B, T)``. Returns ``(T_frame, 768)`` or
        ``(B, T_frame, 768)``."""
        squeeze_batch = wav.ndim == 1
        if squeeze_batch:
            wav = wav.unsqueeze(0)
        in_sr = in_sr if in_sr is not None else self.in_sr
        wav_16k = self._to_16k(wav, in_sr)
        if self._normalize:
            # FunASR's inference does `F.layer_norm(source, source.shape)` on a
            # 1D tensor (full-shape norm); for batched input we normalize over
            # the time dimension per sample, which is the same operation.
            wav_16k = F.layer_norm(wav_16k, wav_16k.shape[1:])
        wav_16k = wav_16k.to(next(self.model.parameters()).dtype)
        feats = self.model.extract_features(wav_16k, padding_mask=None)
        x = feats["x"]  # (B, T_frame, 768)
        return x.squeeze(0) if squeeze_batch else x


@torch.no_grad()
def precompute_reference_emo(
    emo: GradEmoExtractor,
    ref_wav: torch.Tensor,
    *,
    in_sr: int,
) -> torch.Tensor:
    """Encode the prompt wav once → frozen ``(T_frame, 768)`` ref for EMO loss."""
    feat = emo(ref_wav, in_sr=in_sr)
    return feat.detach()
