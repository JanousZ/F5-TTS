"""Four independent evaluation metrics for TTO-generated audio.

Metrics (all take gen_wav / ref_wav / gen_text + sampling rates, return dict):
    compute_wer          — Whisper-large-v3 ASR vs gen_text (CER for Chinese,
                           WER otherwise). Lower is better.
    compute_utmos        — UTMOS22 naturalness / artifact score. Higher better.
    compute_speaker_sim  — WavLM-SV x-vector cosine vs ref wav. Higher better.
    compute_emotion_sim  — Independent IEMOCAP SER (superb/wav2vec2-base-superb-er),
                           trained on a *different* corpus than tto.py's VAD
                           model so it is a fair out-of-sample check. Returns
                           KL / JSD between gen and ref probability vectors,
                           top-1 label agreement, and raw probs.

Plus ``evaluate_all`` that runs all four and merges the result dicts.

Inputs: wav can be a file path (str), a 1D/2D torch.Tensor, or a numpy array.
When tensor/ndarray is passed, ``sr`` is required.

Extra pip deps (not part of F5-TTS base): ``jiwer``.
All four model weights live under ``/mnt/disk1/models/tts_eval/``:
    whisper-large-v3/     — OpenAI Whisper (HF snapshot)
    wavlm-base-plus-sv/   — Microsoft WavLM-SV x-vector (HF snapshot)
    wav2vec2-superb-er/   — superb IEMOCAP 4-class SER (HF snapshot)
    utmos_hub/hub/        — torch.hub repo + checkpoints for UTMOS22
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
SER_MODEL = f"{_LOCAL_ROOT}/wav2vec2-superb-er"
UTMOS_HUB_DIR = f"{_LOCAL_ROOT}/utmos_hub/hub"
UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_NAME = "utmos22_strong"
TARGET_SR = 16000

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


def _ser_bundle(device):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    key = ("ser", str(device))
    if key not in _MODELS:
        fe = AutoFeatureExtractor.from_pretrained(SER_MODEL)
        model = AutoModelForAudioClassification.from_pretrained(SER_MODEL)
        model = model.to(device).eval()
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        _MODELS[key] = (fe, model, labels)
    return _MODELS[key]


def _ser_probs(wav: WavLike, sr: int | None, device) -> tuple[torch.Tensor, list[str]]:
    fe, model, labels = _ser_bundle(device)
    w, s = _load_wav(wav, sr)
    w_16k = _to_16k_mono(w, s).numpy()
    inputs = fe(w_16k, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.softmax(dim=-1).squeeze(0).cpu(), labels


def compute_emotion_sim(
    gen_wav: WavLike,
    ref_wav: WavLike,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
) -> dict:
    """Independent SER on gen & ref; compare probability distributions.

    Uses an IEMOCAP-trained classifier, which is cross-corpus w.r.t. the
    MSP-Podcast-trained VAD model used in tto.py — makes this a fair out-of-
    sample emotion check.

    Returns KL(ref||gen), JSD, top-1 label match, and both probability vectors.
    """
    p_gen, labels = _ser_probs(gen_wav, sr_gen, device)
    p_ref, _ = _ser_probs(ref_wav, sr_ref, device)

    eps = 1e-8
    log_gen = p_gen.clamp(min=eps).log()
    log_ref = p_ref.clamp(min=eps).log()
    kl = float((p_ref * (log_ref - log_gen)).sum())
    m = 0.5 * (p_gen + p_ref)
    log_m = m.clamp(min=eps).log()
    jsd = 0.5 * float((p_gen * (log_gen - log_m)).sum()) + \
          0.5 * float((p_ref * (log_ref - log_m)).sum())
    top1_match = int(p_gen.argmax().item() == p_ref.argmax().item())

    return {
        "emo_kl": kl,
        "emo_jsd": jsd,
        "emo_top1_match": top1_match,
        "emo_gen_label": labels[int(p_gen.argmax())],
        "emo_ref_label": labels[int(p_ref.argmax())],
        "emo_labels": labels,
        "emo_gen_probs": [float(x) for x in p_gen.tolist()],
        "emo_ref_probs": [float(x) for x in p_ref.tolist()],
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
    """Run all four metrics and merge results into a single dict.

    Keys: ``wer`` or ``cer``, ``hyp``, ``ref``, ``utmos``, ``spk_sim``,
    ``emo_kl``, ``emo_jsd``, ``emo_top1_match``, ``emo_gen_label``,
    ``emo_ref_label``, ``emo_labels``, ``emo_gen_probs``, ``emo_ref_probs``.
    """
    out: dict = {}
    out.update(compute_wer(gen_wav, gen_text, sr_gen, device=device))
    out.update(compute_utmos(gen_wav, sr_gen, device=device))
    out.update(compute_speaker_sim(gen_wav, ref_wav, sr_gen, sr_ref, device=device))
    out.update(compute_emotion_sim(gen_wav, ref_wav, sr_gen, sr_ref, device=device))
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
                   help="Skip long fields (probs arrays, hyp) in printed output")
    args = p.parse_args()

    result = evaluate_all(args.gen, args.ref, args.text, device=args.device)
    if args.short:
        for k in ("emo_gen_probs", "emo_ref_probs", "emo_labels", "hyp", "ref"):
            result.pop(k, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
