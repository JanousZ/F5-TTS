"""Five independent evaluation metrics for TTO-generated audio.

Metrics (all take gen_wav / ref_wav / gen_text + sampling rates, return dict):
    compute_wer          — Whisper-large-v3 ASR vs gen_text (CER for Chinese,
                           WER otherwise). Lower is better.
    compute_utmos        — UTMOS22 naturalness / artifact score. Higher better.
    compute_speaker_sim  — WavLM-SV x-vector cosine vs ref wav. Higher better.
    compute_emotion_sim  — emotion2vec_base_finetuned embedding similarity.
                           Reads the pipeline's frame-level ``feats`` (~50 Hz
                           hidden vectors). Returns:
                             e2v_sim_utt   — cosine of mean-pooled utterance
                                             embeddings. ↑ better.
                             e2v_sim_frame — per-frame cosine after gen→ref
                                             length sync (nearest-neighbor
                                             interp). ↑ better.
                           Inputs MUST be wav file paths (modelscope pipeline
                           accepts paths directly, not raw tensors).
    compute_av_sim       — wav2vec2-large-robust-MSP-DIM regressor →
                           (arousal, valence) ∈ [-0.5, 0.5] (dominance dropped,
                           values centered). Returns:
                             av_sim_utt   — cosine over the utterance-level
                                            2-d vector. ↑ better.
                             av_sim_chunk — mean per-chunk cosine after gen→ref
                                            length sync. ↑ better.

Plus ``evaluate_all`` that runs all five and merges the result dicts.

Inputs: wav can be a file path (str), a 1D/2D torch.Tensor, or a numpy array
for WER / UTMOS / speaker sim / AV sim. ``compute_emotion_sim`` requires
path strings. When tensor/ndarray is passed, ``sr`` is required.

Extra pip deps (not part of F5-TTS base): ``jiwer``, ``modelscope``,
``scipy``, ``scikit-learn``.
Model weights live under ``/mnt/disk1/models/``:
    tts_eval/whisper-large-v3/                       — OpenAI Whisper (HF)
    tts_eval/wavlm-base-plus-sv/                     — Microsoft WavLM-SV
    tts_eval/utmos_hub/hub/                          — UTMOS22 torch.hub repo
    emotion2vec_base_finetuned/                      — modelscope emo2vec FT
    wav2vec2-large-robust-12-ft-emotion-msp-dim/     — MSP-DIM regressor

python src/f5_tts/eval/eval_metric.py \
  --ref ./compareasset/F_spk_02-angrycalm.wav \
  --gen ./compareasset/ttsctrlnet_demo.wav \
  --text "Dogs are sitting by the door. Dogs are sitting by the door."
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


_LOCAL_ROOT = "/mnt/disk1/models/tts_eval"
WHISPER_MODEL = f"{_LOCAL_ROOT}/whisper-large-v3"
WAVLM_SV_MODEL = f"{_LOCAL_ROOT}/wavlm-base-plus-sv"
UTMOS_HUB_DIR = f"{_LOCAL_ROOT}/utmos_hub/hub"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_NAME = "utmos22_strong"
TARGET_SR = 16000

# emotion2vec_base_finetuned: 9-class FT'd checkpoint hosted on modelscope.
# We use it for embedding similarity (NOT classification). granularity="frame"
# returns per-frame (~50 Hz) hidden features under pred["feats"] of shape
# (T, 768). Utterance similarity = cosine of mean-pool; frame similarity =
# mean per-row cosine after time-syncing gen → ref length.
E2V_MODEL = "/mnt/disk1/models/emotion2vec_base_finetuned"
E2V_REVISION = "v2.0.4"

# wav2vec2-large-robust-12-ft-emotion-msp-dim: regression head outputs
# (arousal, dominance, valence) ∈ [0, 1]. We keep arousal & valence (cols 0, 2)
# and center to [-0.5, 0.5] so cosine similarity is meaningful (centered
# vectors can take any direction; uncentered ones all live in the positive
# quadrant and yield artificially high cosine).
AV_MODEL = "/mnt/disk1/models/wav2vec2-large-robust-12-ft-emotion-msp-dim"
AV_CHUNK_WINDOW = 25  # hidden-state frames per chunk
AV_CHUNK_HOP = 12     # hidden-state frames between chunk starts

WavLike = Union[torch.Tensor, np.ndarray, str]
_MODELS: dict = {}


def _load_wav(x: WavLike, sr_hint: int | None) -> tuple[torch.Tensor, int]:
    if isinstance(x, str):
        wav, sr = torchaudio.load(x)
        return wav, sr
    if sr_hint is None:
        raise ValueError("sr must be provided when passing tensor/ndarray")
    if isinstance(x, np.ndarray):
        wav = torch.from_numpy(x)
    else:
        wav = x
    return wav, sr_hint


def _to_16k_mono(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Return a 1D float tensor at 16 kHz on CPU."""
    wav = wav.detach().float().cpu()
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _pick_dtype(device) -> torch.dtype:
    return torch.float16 if "cuda" in str(device) else torch.float32


def compute_wer(
    gen_wav: WavLike,
    gen_text: str,
    sr: int | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """Whisper ASR → WER (EN) or CER (ZH) vs gen_text.

    Returns ``{wer|cer, hyp, ref}``. Auto-picks CER if gen_text has CJK chars.
    """
    from transformers import pipeline
    import jiwer

    key = ("whisper", str(device))
    if key not in _MODELS:
        _MODELS[key] = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device,
            torch_dtype=_pick_dtype(device),
            chunk_length_s=30,
        )
    pipe = _MODELS[key]

    wav, in_sr = _load_wav(gen_wav, sr)
    wav_16k = _to_16k_mono(wav, in_sr).numpy()

    is_zh = _has_chinese(gen_text)
    lang = "chinese" if is_zh else "english"
    result = pipe(
        {"array": wav_16k, "sampling_rate": TARGET_SR},
        generate_kwargs={"language": lang, "task": "transcribe"},
    )
    hyp = result["text"].strip()

    # jiwer default doesn't normalize case/punctuation, so "Mother Nature."
    # vs "mother nature." counts as two errors. Apply a lowercase +
    # punctuation-strip transform to both sides before scoring.
    wer_tx = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_tx = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])
    if is_zh:
        score = float(jiwer.cer(
            gen_text, hyp, reference_transform=cer_tx, hypothesis_transform=cer_tx,
        ))
        return {"cer": score, "hyp": hyp, "ref": gen_text}
    score = float(jiwer.wer(
        gen_text, hyp, reference_transform=wer_tx, hypothesis_transform=wer_tx,
    ))
    return {"wer": score, "hyp": hyp, "ref": gen_text}


def compute_utmos(
    gen_wav: WavLike,
    sr: int | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """UTMOS22 naturalness score (roughly 1..5, higher is better)."""
    key = ("utmos", str(device))
    if key not in _MODELS:
        torch.hub.set_dir(UTMOS_HUB_DIR)
        predictor = torch.hub.load(
            UTMOS_REPO, UTMOS_NAME, trust_repo=True, source="github", verbose=False,
        )
        predictor = predictor.to(device).eval()
        _MODELS[key] = predictor
    predictor = _MODELS[key]

    wav, in_sr = _load_wav(gen_wav, sr)
    wav_16k = _to_16k_mono(wav, in_sr).unsqueeze(0).to(device)  # (1, T)
    with torch.no_grad():
        score = predictor(wav_16k, TARGET_SR)
    return {"utmos": float(score.view(-1)[0].cpu())}


def compute_speaker_sim(
    gen_wav: WavLike,
    ref_wav: WavLike,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """WavLM x-vector cosine similarity ∈ [-1, 1] (1 = same speaker)."""
    from transformers import AutoFeatureExtractor, WavLMForXVector

    key = ("wavlm_sv", str(device))
    if key not in _MODELS:
        fe = AutoFeatureExtractor.from_pretrained(WAVLM_SV_MODEL)
        model = WavLMForXVector.from_pretrained(WAVLM_SV_MODEL).to(device).eval()
        _MODELS[key] = (fe, model)
    fe, model = _MODELS[key]

    gw, gsr = _load_wav(gen_wav, sr_gen)
    rw, rsr = _load_wav(ref_wav, sr_ref)
    gw_16k = _to_16k_mono(gw, gsr).numpy()
    rw_16k = _to_16k_mono(rw, rsr).numpy()

    inputs = fe([gw_16k, rw_16k], sampling_rate=TARGET_SR,
                return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).embeddings
    emb = F.normalize(emb, dim=-1)
    sim = float((emb[0] * emb[1]).sum().cpu())
    return {"spk_sim": sim}


def _audio_emb_sync(z: np.ndarray, num_target_frames: int) -> np.ndarray:
    """Nearest-neighbor resample (T, D) → (num_target_frames, D).

    Used to match gen/ref frame counts before per-frame cosine similarity.
    """
    from scipy.interpolate import interp1d
    z = np.asarray(z)
    f = interp1d(np.linspace(0, 1, len(z)), z, axis=0, kind="nearest")
    return f(np.linspace(0, 1, num_target_frames)).astype("float32")


def _frame_cos_sim_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine similarity of two (T, D) embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    sims = []
    for i in range(a.shape[0]):
        sims.append(cosine_similarity(a[i:i + 1], b[i:i + 1])[0, 0])
    return float(np.mean(sims))


def _e2v_pipeline(device):
    """Lazy-load the emotion2vec_base_finetuned modelscope pipeline."""
    key = ("emo2vec_base_ft", str(device))
    if key not in _MODELS:
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
        _MODELS[key] = ms_pipeline(
            task=Tasks.emotion_recognition,
            model=E2V_MODEL,
            model_revision=E2V_REVISION,
            device=str(device),
        )
    return _MODELS[key]


def compute_emotion_sim(
    gen_wav: WavLike,
    ref_wav: WavLike,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
    granularity: str = "frame",
) -> dict:
    """emotion2vec_base_finetuned embedding similarity (utt + frame).

    Returns:
      e2v_sim_utt    cosine of mean-pooled utterance embeddings. ↑ better.
      e2v_sim_frame  mean per-frame cosine after nearest-neighbor sync of
                     gen frames to ref length. ↑ better.

    Requires file paths (modelscope pipeline reads wavs directly).
    """
    if not (isinstance(gen_wav, str) and isinstance(ref_wav, str)):
        raise TypeError("compute_emotion_sim 仅支持 wav 路径输入")

    pipe = _e2v_pipeline(device)
    with torch.no_grad():
        ref_pred = pipe([ref_wav], granularity=granularity)[0]
        gen_pred = pipe([gen_wav], granularity=granularity)[0]

    emb_ref = torch.tensor(ref_pred["feats"])
    emb_gen = torch.tensor(gen_pred["feats"])
    sim_utt = F.cosine_similarity(
        emb_ref.mean(dim=0, keepdim=True),
        emb_gen.mean(dim=0, keepdim=True),
    ).item()

    ref_np = emb_ref.cpu().numpy()
    gen_np = emb_gen.cpu().numpy()
    gen_np = _audio_emb_sync(gen_np, ref_np.shape[0])
    sim_frame = _frame_cos_sim_mean(ref_np, gen_np)

    return {
        "e2v_sim_utt": float(sim_utt),
        "e2v_sim_frame": float(sim_frame),
    }


# ---------------------------------------------------------------------------
# arousal / valence dimensional regression (wav2vec2-large-robust MSP-DIM)
# ---------------------------------------------------------------------------
class _AVRegressionHead(torch.nn.Module):
    """Regression head matching the MSP-DIM checkpoint architecture."""

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.1)
        self.out_proj = torch.nn.Linear(1024, 3)

    def forward(self, features):
        x = torch.mean(features, dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        hidden = self.dropout(x)
        return self.out_proj(hidden), hidden


def _av_build_model_class():
    """Defer-import transformers and build the EmotionModel class once."""
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model, Wav2Vec2PreTrainedModel,
    )

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _AVRegressionHead()
            # transformers' from_pretrained inspects this attr during weight tying
            self.all_tied_weights_keys = {}
            self.init_weights()

        @property
        def _tied_weights_keys(self):
            return []

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            all_hidden = outputs[0]                    # (1, T, 1024)
            logits, _ = self.classifier(all_hidden)    # (1, 3)
            return logits, all_hidden

    return EmotionModel


def _av_load(device):
    """Lazy-load the wav2vec2 emotion regressor + its processor."""
    key = ("av_msp_dim", str(device))
    if key not in _MODELS:
        from transformers import Wav2Vec2Processor
        EmotionModel = _av_build_model_class()
        processor = Wav2Vec2Processor.from_pretrained(AV_MODEL)
        model = EmotionModel.from_pretrained(AV_MODEL).to(device).eval()
        _MODELS[key] = (processor, model)
    return _MODELS[key]


def _av_extract(
    wav_np: np.ndarray, processor, model, device,
    chunk_window: int, chunk_hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full audio + chunked classifier; return ((1, 3), (n_chunks, 3))."""
    inp = processor(wav_np, sampling_rate=TARGET_SR)["input_values"][0]
    inp = torch.from_numpy(inp).reshape(1, -1).to(device)

    with torch.no_grad():
        outputs = model.wav2vec2(inp)
        all_hidden = outputs[0]                                  # (1, T, 1024)
        utt_logits, _ = model.classifier(all_hidden)             # (1, 3)

        T = all_hidden.size(1)
        if T < chunk_window:
            # Audio too short for any window — fall back to full-utt logits as
            # the single chunk so similarity stays well-defined.
            chunk_logits = utt_logits.clone()
        else:
            num_chunks = (T - chunk_window) // chunk_hop + 1
            chunks = [
                all_hidden[:, i * chunk_hop:i * chunk_hop + chunk_window, :]
                for i in range(num_chunks)
            ]
            stacked = torch.stack(chunks, dim=1).transpose(2, 1)  # (1, win, n, 1024)
            chunk_logits, _ = model.classifier(stacked)           # (1, n, 3)
            chunk_logits = chunk_logits.squeeze(0)

    return utt_logits.cpu().numpy(), chunk_logits.cpu().numpy()


def compute_av_sim(
    gen_wav: WavLike,
    ref_wav: WavLike,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
    chunk_window: int = AV_CHUNK_WINDOW,
    chunk_hop: int = AV_CHUNK_HOP,
) -> dict:
    """Arousal/Valence regressor cosine similarity (utt + per-chunk).

    Pipeline: wav → wav2vec2 → 3-d (arousal, dominance, valence) ∈ [0, 1].
    Drop dominance, center to [-0.5, 0.5], then:
      av_sim_utt    cosine over the 2-d utterance vector. ↑ better.
      av_sim_chunk  per-chunk cosine after gen→ref length sync
                    (nearest-neighbor interp), then mean. ↑ better.
    """
    processor, model = _av_load(device)

    # Use librosa for bit-exact parity with the reference arousal/valence
    # pipeline (different resamplers cause ~1e-3 drift on regression heads).
    import librosa
    if isinstance(gen_wav, str):
        g_np, _ = librosa.load(gen_wav, sr=TARGET_SR)
    else:
        g_wav, g_sr = _load_wav(gen_wav, sr_gen)
        g_np = _to_16k_mono(g_wav, g_sr).numpy()
    if isinstance(ref_wav, str):
        r_np, _ = librosa.load(ref_wav, sr=TARGET_SR)
    else:
        r_wav, r_sr = _load_wav(ref_wav, sr_ref)
        r_np = _to_16k_mono(r_wav, r_sr).numpy()

    g_utt, g_chunk = _av_extract(g_np, processor, model, device, chunk_window, chunk_hop)
    r_utt, r_chunk = _av_extract(r_np, processor, model, device, chunk_window, chunk_hop)

    # Keep arousal & valence (cols 0, 2), center.
    g_utt_av = g_utt[:, [0, 2]] - 0.5
    r_utt_av = r_utt[:, [0, 2]] - 0.5
    g_chunk_av = g_chunk[:, [0, 2]] - 0.5
    r_chunk_av = r_chunk[:, [0, 2]] - 0.5

    g_chunk_av = _audio_emb_sync(g_chunk_av, r_chunk_av.shape[0])

    sim_utt = _frame_cos_sim_mean(r_utt_av, g_utt_av)
    sim_chunk = _frame_cos_sim_mean(r_chunk_av, g_chunk_av)

    return {
        "av_sim_utt": float(sim_utt),
        "av_sim_chunk": float(sim_chunk),
    }


def evaluate_all(
    gen_wav: WavLike,
    ref_wav: WavLike,
    gen_text: str,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """Run all five metrics and merge results into a single dict.

    Keys: ``wer`` or ``cer``, ``hyp``, ``ref``, ``utmos``, ``spk_sim``,
    ``e2v_sim_utt``, ``e2v_sim_frame``, ``av_sim_utt``, ``av_sim_chunk``.
    """
    out: dict = {}
    out.update(compute_wer(gen_wav, gen_text, sr_gen, device=device))
    out.update(compute_utmos(gen_wav, sr_gen, device=device))
    out.update(compute_speaker_sim(gen_wav, ref_wav, sr_gen, sr_ref, device=device))
    out.update(compute_emotion_sim(gen_wav, ref_wav, sr_gen, sr_ref, device=device))
    out.update(compute_av_sim(gen_wav, ref_wav, sr_gen, sr_ref, device=device))
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="TTO output four-metric evaluation")
    p.add_argument("--gen", required=True, help="Generated wav path")
    p.add_argument("--ref", required=True, help="Reference wav path")
    p.add_argument("--text", required=True, help="Intended gen text (for WER/CER)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--short", action="store_true",
                   help="Skip long fields (hyp/ref text) in printed output")
    args = p.parse_args()

    result = evaluate_all(args.gen, args.ref, args.text, device=args.device)
    if args.short:
        for k in ("hyp", "ref"):
            result.pop(k, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
