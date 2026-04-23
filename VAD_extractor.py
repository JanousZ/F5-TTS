import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits



# load model from hub
device = 'cpu'
model_name = '/mnt/disk1/models/wav2vec2-large-robust-12-ft-emotion-msp-dim'

with open(os.path.join(model_name, 'config.json')) as f:
    _cfg_dict = json.load(f)
if _cfg_dict.get('vocab_size') is None:
    _cfg_dict['vocab_size'] = 32
config = Wav2Vec2Config(**_cfg_dict)

processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name, config=config).to(device)

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


def process_func_framewise(
    x: np.ndarray,
    sampling_rate: int,
    window_size: float = 1.0,
    hop_size: float = 0.25,
    embeddings: bool = False,
    pad: bool = True,
) -> np.ndarray:
    r"""Extract frame-level features via a sliding window.

    Returns an array of shape ``(num_frames, feature_dim)`` where each row is
    the output of :func:`process_func` applied to one window. ``feature_dim``
    is the embedding dim when ``embeddings=True`` and 3 (arousal, dominance,
    valence) otherwise.
    """

    sig = x[0] if x.ndim == 2 else x
    sig = sig.astype(np.float32, copy=False)

    win_samples = int(round(window_size * sampling_rate))
    hop_samples = int(round(hop_size * sampling_rate))
    if win_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_size and hop_size must be positive")

    total_len = sig.shape[0]
    if total_len < win_samples:
        if not pad:
            raise ValueError(
                f"signal shorter ({total_len}) than window ({win_samples})"
            )
        sig = np.pad(sig, (0, win_samples - total_len))
        total_len = sig.shape[0]
    elif pad:
        remainder = (total_len - win_samples) % hop_samples
        if remainder != 0:
            sig = np.pad(sig, (0, hop_samples - remainder))
            total_len = sig.shape[0]

    frames = []
    for start in range(0, total_len - win_samples + 1, hop_samples):
        frame = sig[start:start + win_samples]
        feat = process_func(
            frame[np.newaxis, :], sampling_rate, embeddings=embeddings,
        )
        frames.append(feat[0])

    return np.stack(frames, axis=0)


if __name__ == '__main__':
    import argparse, sys
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument('audio', nargs='?', help='audio file; omit to run dummy demo')
    ap.add_argument('--framewise', action='store_true')
    ap.add_argument('--embeddings', action='store_true')
    args = ap.parse_args()

    if args.audio is None:
        sig = signal
    else:
        sig, sr = sf.read(args.audio, dtype='float32', always_2d=False)
        if sig.ndim == 2:
            sig = sig.mean(axis=1)
        if sr != sampling_rate:
            import torchaudio
            sig = torchaudio.functional.resample(
                torch.from_numpy(sig).unsqueeze(0), sr, sampling_rate,
            ).squeeze(0).numpy()
        sig = sig[np.newaxis, :]

    if args.framewise:
        out = process_func_framewise(sig, sampling_rate, embeddings=args.embeddings)
    else:
        out = process_func(sig, sampling_rate, embeddings=args.embeddings)
    print(out.shape)
    print(out)

