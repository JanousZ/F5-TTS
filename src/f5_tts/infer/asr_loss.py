"""Paraformer "All Tokens CE" loss for F5-TTS test-time optimization.

Differentiable ASR loss against ``gen_text`` using FunASR's Paraformer:

    wav_24k → resample 16k → WavFrontend (fbank + CMVN + LFR)
            → encoder → CIF predictor (teacher-forced to len(gen_text))
            → parallel decoder → LabelSmoothingLoss on every token position

Alignment is built into Paraformer's CIF — we pass
``target_label_length=text_lens`` and CIF forces ``N == len(gen_text)``.
The per-position loss is Paraformer's own ``criterion_att``
(LabelSmoothingLoss), matching its training objective; falls back to
plain ``F.cross_entropy`` if a legacy FunASR build doesn't expose it.

All Paraformer params are frozen; gradients flow only through wav.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ParaformerASRLoss(nn.Module):
    """All-token CE loss on a frozen Paraformer ASR.

    Single-text variant: target ``gen_text`` is fixed at construction
    (encoded once into token ids); call ``set_target`` to swap it without
    reloading the model. ``forward(wav)`` is invoked per TTO iteration.
    """

    def __init__(
        self,
        model_name: str = "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        gen_text: str = "",
        device: torch.device | str = "cuda",
        in_sr: int = 24000,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        from funasr import AutoModel

        # Load once. disable_update freezes the model + skips optimizer state.
        am = AutoModel(model=model_name, disable_update=True, device=str(device))
        self.paraformer = am.model.to(device)
        self.frontend = am.kwargs["frontend"]      # WavFrontend: fbank + CMVN + LFR
        self.tokenizer = am.kwargs["tokenizer"]

        # Freeze. We only ever backprop *through* this module to wav.
        self.paraformer.eval()
        for p in self.paraformer.parameters():
            p.requires_grad_(False)
        # Skip the CPU wer/cer error_calculator — we only want the loss tensor.
        self.paraformer.error_calculator = None
        # NOTE: we keep ``sampling_ratio`` at the model's training default
        # (typically 0.75). Paraformer's parallel decoder is *trained with*
        # scheduled sampling — without it, the decoder receives no GT
        # context and the LS-CE collapses to ~log(V). With it on, the
        # second-pass loss matches what Paraformer was optimized against.
        # The sampler injects randomness (torch.randperm), so the loss is
        # mildly stochastic — fine for TTO since inner iterations average it.

        self.device = torch.device(device)
        self.in_sr = in_sr
        self.target_sr = 16000  # Paraformer is 16 kHz only

        if dtype is not None:
            self.paraformer.to(dtype=dtype)

        self._resampler = (
            torchaudio.transforms.Resample(in_sr, self.target_sr).to(device)
            if in_sr != self.target_sr else None
        )

        self.set_target(gen_text)

    # ------------------------------------------------------------------
    def set_target(self, gen_text: str) -> None:
        """Encode ``gen_text`` into Paraformer token ids (raw, no SOS/EOS).

        ``_calc_att_loss`` internally adds SOS/EOS via ``add_sos_eos`` when
        ``predictor_bias==1``, so we pass the *raw* ids and their original
        length here — adding EOS ourselves would double-count.
        """
        ids = self.tokenizer.encode(gen_text)
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self._target_ids = ids.unsqueeze(0)                       # (1, L_raw)
        self._target_lens = torch.tensor(
            [ids.shape[0]], device=self.device, dtype=torch.long,
        )

    # ------------------------------------------------------------------
    def _features(self, wav_24k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav (B, T_24k) → (feats, feats_lens) via WavFrontend.

        WavFrontend does Kaldi fbank + CMVN + LFR — all torch ops in
        recent FunASR, so gradients flow back to wav. If a build does a
        ``.numpy()`` detour inside ``frontend.forward``, swap in a manual
        fbank here.
        """
        if wav_24k.ndim == 1:
            wav_24k = wav_24k.unsqueeze(0)
        wav_16k = self._resampler(wav_24k) if self._resampler is not None else wav_24k

        wav_lens = torch.full(
            (wav_16k.shape[0],), wav_16k.shape[-1],
            device=wav_16k.device, dtype=torch.long,
        )
        feats, feats_lens = self.frontend(wav_16k, wav_lens)
        return feats, feats_lens

    # ------------------------------------------------------------------
    def forward(self, wav_24k: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Run the all-token CE branch.

        Returns ``(loss_tensor, {'asr': float})``. We call Paraformer's
        own ``_calc_att_loss``, which internally:
          - adds SOS/EOS to the target (matches training convention),
          - teacher-forces the CIF predictor to ``len(target)+1``,
          - runs the parallel decoder,
          - applies LabelSmoothingLoss on every output position.
        Only the att branch is taken — no CTC, no CIF quantity loss —
        so the gradient signal aligns with token-level recognition error.
        """
        pf = self.paraformer
        feats, feats_lens = self._features(wav_24k)

        # Encoder — SANMEncoder returns (xs, olens, prev_state).
        enc_ret = pf.encoder(feats, feats_lens)
        enc_out, enc_lens = enc_ret[0], enc_ret[1]

        loss_att, _acc, _cer, _wer, _loss_pre, _pre_loss_att = pf._calc_att_loss(
            enc_out, enc_lens, self._target_ids, self._target_lens,
        )
        return loss_att, {"asr": float(loss_att.detach())}
