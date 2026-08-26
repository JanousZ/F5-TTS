"""Utilities for target-span emotion control.

The target span is defined on the generated waveform timeline.  Reference
audio supplies a pooled emotion value, while the span determines where that
value is applied in ``tts_text``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


Span = tuple[float, float]


def validate_span(span: Sequence[float]) -> Span:
    """Validate and normalize a ``(start_sec, end_sec)`` span."""
    if len(span) != 2:
        raise ValueError(f"target span must contain 2 values, got {span!r}")
    start, end = float(span[0]), float(span[1])
    if not (start >= 0.0 and end > start):
        raise ValueError(f"target span must satisfy 0 <= start < end, got {span!r}")
    return start, end


def span_to_sample_bounds(
    span: Sequence[float],
    *,
    sample_rate: int,
    num_samples: int,
    context_sec: float = 0.0,
) -> tuple[int, int]:
    """Convert a time span to clamped sample bounds.

    ``context_sec`` expands both sides of the span.  Clamping is intentional:
    a target near the beginning or end of an utterance still produces a valid
    crop without padding artifacts.
    """
    if context_sec < 0.0:
        raise ValueError(f"context_sec must be non-negative, got {context_sec}")
    start, end = validate_span(span)
    start_i = max(0, int(round((start - context_sec) * sample_rate)))
    end_i = min(num_samples, int(round((end + context_sec) * sample_rate)))
    if end_i <= start_i:
        raise ValueError(
            f"target span is outside the waveform: span={span!r}, "
            f"num_samples={num_samples}, sample_rate={sample_rate}"
        )
    return start_i, end_i


def crop_waveform(
    wav: torch.Tensor,
    span: Sequence[float],
    *,
    sample_rate: int,
    context_sec: float = 0.0,
) -> torch.Tensor:
    """Differentiably crop ``wav`` using a generated-audio time span."""
    if wav.ndim not in (1, 2):
        raise ValueError(f"wav must have shape (T,) or (B, T), got {tuple(wav.shape)}")
    start_i, end_i = span_to_sample_bounds(
        span,
        sample_rate=sample_rate,
        num_samples=wav.shape[-1],
        context_sec=context_sec,
    )
    return wav[..., start_i:end_i]


def blend_vad_targets(
    neutral_vad: torch.Tensor,
    emotion_vad: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Interpolate neutral and emotion VAD values for a target span."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1] for the pilot, got {alpha}")
    if neutral_vad.shape != emotion_vad.shape:
        raise ValueError(
            "neutral_vad and emotion_vad must have the same shape: "
            f"{tuple(neutral_vad.shape)} vs {tuple(emotion_vad.shape)}"
        )
    return neutral_vad + alpha * (emotion_vad - neutral_vad)


def target_latent_mask(
    *,
    span: Sequence[float],
    batch_size: int,
    latent_length: int,
    prompt_length: int,
    sample_rate: int,
    latent_hop_length: int,
    context_sec: float = 0.0,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a mask for local optimization on the generated latent frames.

    ``span`` is relative to generated audio, while the latent includes prompt
    frames at its beginning.  The returned mask has shape ``(B, T, 1)``.
    """
    if latent_length < prompt_length:
        raise ValueError(
            f"latent_length ({latent_length}) < prompt_length ({prompt_length})"
        )
    start, end = span_to_sample_bounds(
        span,
        sample_rate=sample_rate,
        num_samples=(latent_length - prompt_length) * latent_hop_length,
        context_sec=context_sec,
    )
    start_frame = prompt_length + start // latent_hop_length
    end_frame = prompt_length + (end + latent_hop_length - 1) // latent_hop_length
    end_frame = min(latent_length, end_frame)
    mask = torch.zeros(
        batch_size, latent_length, 1, dtype=torch.bool, device=device,
    )
    mask[:, start_frame:end_frame] = True
    return mask
