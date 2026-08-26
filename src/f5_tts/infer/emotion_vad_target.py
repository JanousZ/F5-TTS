"""Emotion-level VAD targets from the public NRC VAD lexicon.

NRC reports scores in V/A/D order on [0, 1]. The F5-TTS emotion model
returns A/D/V, so this module keeps the conversion explicit.
"""

from __future__ import annotations

import torch


# NRC anchor words: happy, sad, angry, surprise. Source scale is V/A/D.
NRC_VAD_ANCHORS = {
    "happy": (1.000, 0.735, 0.772),
    "sad": (0.225, 0.333, 0.149),
    "angry": (0.122, 0.830, 0.604),
    "surprise": (0.875, 0.875, 0.562),
}


def emotion_vad_target(
    emotion: str,
    *,
    alpha: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an A/D/V target interpolated from the neutral midpoint."""
    key = emotion.strip().lower()
    if key not in NRC_VAD_ANCHORS:
        raise ValueError(
            f"unsupported emotion {emotion!r}; "
            f"expected one of {sorted(NRC_VAD_ANCHORS)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    valence, arousal, dominance = NRC_VAD_ANCHORS[key]
    anchor_adv = torch.tensor(
        [arousal, dominance, valence], device=device, dtype=dtype,
    )
    neutral = torch.full_like(anchor_adv, 0.5)
    return neutral + alpha * (anchor_adv - neutral)
