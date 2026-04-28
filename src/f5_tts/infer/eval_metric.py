"""Four independent evaluation metrics for TTO-generated audio.

Metrics (all take gen_wav / ref_wav / gen_text + sampling rates, return dict):
    compute_wer          — Whisper-large-v3 ASR vs gen_text (CER for Chinese,
                           WER otherwise). Lower is better.
    compute_utmos        — UTMOS22 naturalness / artifact score. Higher better.
    compute_speaker_sim  — WavLM-SV x-vector cosine vs ref wav. Higher better.
    compute_emotion_sim  — emotion2vec_plus_large 9-class SLIDING-WINDOW
                           trajectory comparison. Pipeline:
                             full audio → backbone → (T, 1024) hidden once
                             → sliding window mean over T → (n_win, 1024)
                             → backbone.proj → (n_win, 9) raw softmax (no
                               funasr 'unuse' mask — all 9 classes exposed:
                               angry/disgusted/fearful/happy/neutral/other/
                               sad/surprised/unknown).
                           Compares gen and ref trajectories via DTW(JSD),
                           interp-aligned mean JSD, and label-sequence edit
                           distance. Captures emotion CHANGE, not just
                           utter-level distribution match.

Plus ``evaluate_all`` that runs all four and merges the result dicts.

Inputs: wav can be a file path (str), a 1D/2D torch.Tensor, or a numpy array.
When tensor/ndarray is passed, ``sr`` is required.

Extra pip deps (not part of F5-TTS base): ``jiwer``, ``funasr``.
Model weights live under ``/mnt/disk1/models/``:
    tts_eval/whisper-large-v3/   — OpenAI Whisper (HF snapshot)
    tts_eval/wavlm-base-plus-sv/ — Microsoft WavLM-SV x-vector
    tts_eval/utmos_hub/hub/      — torch.hub repo + checkpoints for UTMOS22
    emotion2vec_plus_large/      — funasr-style emotion2vec checkpoint
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

# emotion2vec_plus_large: 9-class emotion classifier on a data2vec backbone.
# Frame rate is ~50 Hz (20 ms hop) coming out of the wav2vec2-style encoder.
# tokens.txt has placeholder names ("unuse_0/1/2/3") for indices 1, 2, 5, 7;
# the trained proj head is genuinely 9-d and we read all 9 logits, NOT the
# 5-class mask that funasr's inference() applies.
E2V_MODEL = "/mnt/disk1/models/emotion2vec_plus_large"
E2V_FRAME_HZ = 50
E2V_CLASSES = (
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
)

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


def _e2v_load(device):
    """Lazy-load emotion2vec_plus_large (funasr backbone). Caches on device."""
    key = ("emotion2vec", str(device))
    if key not in _MODELS:
        from funasr import AutoModel
        m = AutoModel(model=E2V_MODEL, disable_update=True, device=str(device))
        backbone = m.model.to(device).eval()
        _MODELS[key] = backbone
    return _MODELS[key]


def _e2v_extract_hidden(
    wav: WavLike, sr: int | None, device,
) -> torch.Tensor:
    """Run the FULL audio through emotion2vec once, return last-layer hidden.

    Returns ``(T_frame, 1024)`` on the same device as the model. Frame rate is
    fixed by the data2vec audio encoder (~50 Hz, 20 ms / frame).
    """
    backbone = _e2v_load(device)
    w, s = _load_wav(wav, sr)
    w_16k = _to_16k_mono(w, s).to(device)
    src = F.layer_norm(w_16k, w_16k.shape).view(1, -1)
    with torch.no_grad():
        feats = backbone.extract_features(src, padding_mask=None)
    return feats["x"].squeeze(0)  # (T, 1024)


def _e2v_window_classify(
    hidden: torch.Tensor, window_s: float, hop_s: float, device,
) -> torch.Tensor:
    """Slide window over (T, D) hidden, mean-pool per window, apply proj.

    Sliding happens on the **hidden state**, not the raw audio — so the
    backbone only ran once. Returns ``(n_win, 9)`` raw softmax (no funasr
    'unuse' mask).
    """
    backbone = _e2v_load(device)
    win = max(1, int(round(window_s * E2V_FRAME_HZ)))
    hop = max(1, int(round(hop_s * E2V_FRAME_HZ)))
    T = hidden.shape[0]
    if T < win:
        pooled = hidden.mean(dim=0, keepdim=True)  # (1, D) fallback
    else:
        windows = hidden.unfold(0, win, hop)        # (n_win, D, win)
        pooled = windows.mean(dim=-1)               # (n_win, D)
    with torch.no_grad():
        logits = backbone.proj(pooled)              # (n_win, 9) — raw, all 9 classes
    return logits.softmax(dim=-1)                   # (n_win, 9)


def _interp_time(probs: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear interp a (N, K) prob trajectory to (target_len, K).

    Linear interpolation of probability rows preserves row sum=1 (linear
    combinations of distributions stay distributions).
    """
    if probs.shape[0] == target_len:
        return probs
    p = probs.transpose(0, 1).unsqueeze(0)  # (1, K, N)
    p = F.interpolate(p, size=target_len, mode="linear", align_corners=False)
    return p.squeeze(0).transpose(0, 1)     # (target_len, K)


def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _dtw_jsd(gen_probs: torch.Tensor, ref_probs: torch.Tensor) -> float:
    """DTW(gen ↔ ref) with per-cell cost = JSD(gen[i], ref[j]).

    Returns accumulated cost normalized by warping-path length, so values are
    comparable across different (n_g, n_r). Range roughly [0, ln 2 ≈ 0.693].
    """
    import librosa.sequence
    eps = 1e-8
    g = gen_probs.detach().cpu().numpy()
    r = ref_probs.detach().cpu().numpy()
    g_exp = g[:, None, :]                    # (Ng, 1, K)
    r_exp = r[None, :, :]                    # (1, Nr, K)
    m = 0.5 * (g_exp + r_exp)
    log_g = np.log(np.clip(g_exp, eps, None))
    log_r = np.log(np.clip(r_exp, eps, None))
    log_m = np.log(np.clip(m, eps, None))
    cost = (
        0.5 * (g_exp * (log_g - log_m)).sum(-1) +
        0.5 * (r_exp * (log_r - log_m)).sum(-1)
    ).astype(np.float32)
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return 0.0
    D, wp = librosa.sequence.dtw(C=cost, subseq=False)
    return float(D[-1, -1] / max(len(wp), 1))


def compute_emotion_sim(
    gen_wav: WavLike,
    ref_wav: WavLike,
    sr_gen: int | None = None,
    sr_ref: int | None = None,
    *,
    device: str | torch.device = "cuda",
    window_s: float = 1.0,
    hop_s: float = 0.25,
) -> dict:
    """emotion2vec_plus_large 9-class sliding-window trajectory comparison.

    Captures emotion CHANGE over time, not just utter-level distribution
    match. Window/hop are in seconds, applied to the hidden-state time axis
    after a single full-audio backbone pass.

    Returns:
      e2v_dtw_jsd          DTW distance (JSD per-cell), normalized by path
                           length. ↓ better.
      e2v_frame_jsd_mean   gen/ref interpolated to common length, then mean
                           per-window JSD. ↓ better.
      e2v_label_edit_norm  Levenshtein over per-window argmax label
                           sequences, normalized by max(len). ↓ better.
      e2v_top_label_match  1 iff argmax(mean_pool(gen_probs)) ==
                           argmax(mean_pool(ref_probs)).
      e2v_classes          tuple of 9 class names (fixed order).
      e2v_gen_label_seq    list[str], top-1 label per gen window.
      e2v_ref_label_seq    list[str], top-1 label per ref window.
      e2v_gen_probs_mean   list[9], gen window-averaged probability vector.
      e2v_ref_probs_mean   list[9], ref window-averaged probability vector.
    """
    gen_hidden = _e2v_extract_hidden(gen_wav, sr_gen, device)
    ref_hidden = _e2v_extract_hidden(ref_wav, sr_ref, device)

    gen_probs = _e2v_window_classify(gen_hidden, window_s, hop_s, device).cpu()
    ref_probs = _e2v_window_classify(ref_hidden, window_s, hop_s, device).cpu()

    gen_seq = [E2V_CLASSES[int(i)] for i in gen_probs.argmax(dim=-1).tolist()]
    ref_seq = [E2V_CLASSES[int(i)] for i in ref_probs.argmax(dim=-1).tolist()]

    target_len = max(gen_probs.shape[0], ref_probs.shape[0], 1)
    g_aligned = _interp_time(gen_probs, target_len)
    r_aligned = _interp_time(ref_probs, target_len)
    eps = 1e-8
    log_g = g_aligned.clamp(min=eps).log()
    log_r = r_aligned.clamp(min=eps).log()
    m = 0.5 * (g_aligned + r_aligned)
    log_m = m.clamp(min=eps).log()
    frame_jsd = (
        0.5 * (g_aligned * (log_g - log_m)).sum(dim=-1) +
        0.5 * (r_aligned * (log_r - log_m)).sum(dim=-1)
    )
    frame_jsd_mean = float(frame_jsd.mean())
    dtw = _dtw_jsd(gen_probs, ref_probs)
    edit_norm = _levenshtein(gen_seq, ref_seq) / max(len(gen_seq), len(ref_seq), 1)

    gen_overall = gen_probs.mean(dim=0)
    ref_overall = ref_probs.mean(dim=0)
    top_match = int(gen_overall.argmax().item() == ref_overall.argmax().item())

    return {
        "e2v_dtw_jsd": dtw,
        "e2v_frame_jsd_mean": frame_jsd_mean,
        "e2v_label_edit_norm": float(edit_norm),
        "e2v_top_label_match": top_match,
        "e2v_classes": list(E2V_CLASSES),
        "e2v_gen_label_seq": gen_seq,
        "e2v_ref_label_seq": ref_seq,
        "e2v_gen_probs_mean": [float(x) for x in gen_overall.tolist()],
        "e2v_ref_probs_mean": [float(x) for x in ref_overall.tolist()],
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
    ``e2v_dtw_jsd``, ``e2v_frame_jsd_mean``, ``e2v_label_edit_norm``,
    ``e2v_top_label_match``, ``e2v_classes``, ``e2v_gen_label_seq``,
    ``e2v_ref_label_seq``, ``e2v_gen_probs_mean``, ``e2v_ref_probs_mean``.
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
        for k in ("e2v_classes", "e2v_gen_label_seq", "e2v_ref_label_seq",
                  "e2v_gen_probs_mean", "e2v_ref_probs_mean", "hyp", "ref"):
            result.pop(k, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
