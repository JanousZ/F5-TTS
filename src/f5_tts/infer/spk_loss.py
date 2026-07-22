"""Differentiable SIM-O speaker embedding extractor for F5-TTS TTO.

WavLM-Large + ECAPA-TDNN with the UniSpeech SV fine-tune checkpoint —
the same backbone + checkpoint that
``f5_tts.eval.eval_metric.compute_speaker_sim`` uses to report SIM-O.

Constructed with ``update_extract=True`` so the WavLM forward is not
wrapped in ``torch.no_grad`` (the eval-only default would cut the gradient
path back to the input wav). All parameters are then frozen — only the
gradient PATH is open, weights stay fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.eval.eval_metric import WAVLM_LARGE_BACKBONE, WAVLM_LARGE_SV_FT


SPK_SR = 16000


class GradSpkExtractor(nn.Module):
    """forward(wav, in_sr) → L2-normalized 192-d embedding ``(B, D)``."""

    def __init__(
        self,
        in_sr: int = 24000,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_sr = in_sr
        from s3prl.upstream.wavlm.hubconf import wavlm_local

        upstream = wavlm_local(ckpt=WAVLM_LARGE_BACKBONE)
        # ECAPA_TDNN_SMALL's constructor pulls the WavLM upstream via
        # torch.hub.load; monkey-patch it to return our local instance.
        _orig_hub_load = torch.hub.load
        torch.hub.load = lambda *a, **kw: upstream  # noqa: E731
        try:
            model = ECAPA_TDNN_SMALL(
                feat_dim=1024,
                feat_type="wavlm_large",
                config_path=None,
                update_extract=True,  # keep WavLM forward outside no_grad
            )
        finally:
            torch.hub.load = _orig_hub_load

        # weights_only=False: SV ckpt was saved by older fairseq with OmegaConf cfg.
        state = torch.load(WAVLM_LARGE_SV_FT, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        # WavLM's SV checkpoint defaults feature_grad_mult=0, which wraps the
        # conv frontend in `torch.no_grad()` (see s3prl wavlm/WavLM.py L360-366)
        # and severs the gradient path back to the input wav. Flip it to 1.0
        # so the CNN forward is differentiable; params are still frozen below.
        model.feature_extract.model.feature_grad_mult = 1.0
        for p in model.parameters():
            p.requires_grad_(False)
        if device is not None:
            model = model.to(device)
        self.model = model.eval()
        self._default_resampler: nn.Module = (
            torchaudio.transforms.Resample(in_sr, SPK_SR)
            if in_sr != SPK_SR else nn.Identity()
        )
        if device is not None:
            self._default_resampler = self._default_resampler.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _to_16k(self, wav: torch.Tensor, in_sr: int) -> torch.Tensor:
        if in_sr == SPK_SR:
            return wav
        if in_sr == self.in_sr and isinstance(self._default_resampler, torchaudio.transforms.Resample):
            return self._default_resampler(wav)
        return torchaudio.functional.resample(wav, in_sr, SPK_SR)

    def forward(self, wav: torch.Tensor, *, in_sr: int | None = None) -> torch.Tensor:
        """``wav`` may be ``(T,)`` or ``(B, T)``. Returns ``(B, D)`` L2-normed."""
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        in_sr = in_sr if in_sr is not None else self.in_sr
        wav_16k = self._to_16k(wav, in_sr)
        emb = self.model(wav_16k)  # (B, 192)
        return F.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def precompute_reference_spk_emb(
    spk: GradSpkExtractor,
    ref_wav: torch.Tensor,
    *,
    in_sr: int,
) -> torch.Tensor:
    """Encode the prompt wav once → frozen ``(1, D)`` ref embedding for SIM-O loss."""
    emb = spk(ref_wav, in_sr=in_sr)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return emb.detach()
