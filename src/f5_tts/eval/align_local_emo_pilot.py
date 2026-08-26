"""Force-align local emotion-control pilot targets from baseline audio.

This script consumes the manifest produced by ``build_local_emo_pilot.py`` and
baseline wav files named ``{stem}.wav``. It writes a second JSONL manifest with
``target_span_sec`` filled from CTC forced alignment against ``gen_text``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchaudio


_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class WordItem:
    text: str
    normalized: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class WordAlignment:
    word: WordItem
    start_sec: float
    end_sec: float
    score: float


def normalize_word(text: str) -> str:
    """Normalize an English word to the MMS_FA character inventory."""
    return "".join(ch for ch in text.lower() if ch.isalpha() or ch == "'")


def split_words(text: str) -> list[WordItem]:
    words: list[WordItem] = []
    for match in _WORD_RE.finditer(text):
        normalized = normalize_word(match.group(0))
        if normalized:
            words.append(WordItem(
                text=match.group(0),
                normalized=normalized,
                start_char=match.start(),
                end_char=match.end(),
            ))
    return words


def target_word_indices(row: dict, words: list[WordItem]) -> list[int]:
    """Find target word indices, preferring manifest character spans."""
    span = row.get("target_char_span")
    if span is not None:
        if len(span) != 2:
            raise ValueError(f"target_char_span must contain 2 values: {span!r}")
        start, end = int(span[0]), int(span[1])
        indices = [
            idx for idx, word in enumerate(words)
            if word.start_char < end and word.end_char > start
        ]
        if indices:
            return indices
        raise ValueError(
            f"target_char_span {span!r} does not overlap any aligned word in "
            f"{row.get('gen_text')!r}"
        )

    target = normalize_word(str(row.get("target_word", "")))
    if not target:
        raise ValueError("manifest row needs target_word or target_char_span")
    indices = [idx for idx, word in enumerate(words) if word.normalized == target]
    if len(indices) != 1:
        raise ValueError(
            f"target_word={row.get('target_word')!r} matched {len(indices)} words; "
            "provide target_char_span for disambiguation"
        )
    return indices


def _mms_checkpoint_name(bundle) -> str:
    parsed = urlparse(bundle._path)
    return Path(parsed.path).name or "model.pt"


def load_mms_fa(device: str, model_dir: Path, allow_download: bool):
    bundle = torchaudio.pipelines.MMS_FA
    model_dir.mkdir(parents=True, exist_ok=True)
    expected = model_dir / _mms_checkpoint_name(bundle)
    if not expected.exists() and not allow_download:
        raise FileNotFoundError(
            f"MMS_FA checkpoint is not cached at {expected}. "
            "Place the checkpoint there or rerun with --allow-download."
        )
    model = bundle.get_model(
        dl_kwargs={"model_dir": str(model_dir), "progress": True},
    )
    model.to(device).eval()
    return bundle, model, bundle.get_tokenizer(), bundle.get_aligner()


def load_audio_16k(path: Path, device: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def align_words(
    *,
    audio_path: Path,
    text: str,
    device: str,
    bundle,
    model,
    tokenizer,
    aligner,
) -> list[WordAlignment]:
    words = split_words(text)
    if not words:
        raise ValueError(f"no alignable English words in {text!r}")
    tokenized = tokenizer([word.normalized for word in words])
    wav = load_audio_16k(audio_path, device, bundle.sample_rate)
    with torch.inference_mode():
        emissions, _ = model(wav)
    emission = emissions[0].detach().cpu()
    spans = aligner(emission, tokenized)
    frame_to_sec = wav.shape[-1] / emissions.shape[1] / bundle.sample_rate

    aligned: list[WordAlignment] = []
    for word, token_spans in zip(words, spans, strict=True):
        if not token_spans:
            raise ValueError(f"empty token span for word {word.text!r}")
        start_frame = min(span.start for span in token_spans)
        end_frame = max(span.end for span in token_spans)
        score = sum(float(span.score) for span in token_spans) / len(token_spans)
        aligned.append(WordAlignment(
            word=word,
            start_sec=start_frame * frame_to_sec,
            end_sec=end_frame * frame_to_sec,
            score=score,
        ))
    return aligned


def align_row(
    row: dict,
    baseline_dir: Path,
    *,
    device: str,
    bundle,
    model,
    tokenizer,
    aligner,
) -> dict:
    stem = row["stem"]
    audio_path = Path(row.get("baseline_wav") or baseline_dir / f"{stem}.wav")
    if not audio_path.exists():
        raise FileNotFoundError(f"baseline audio not found for {stem}: {audio_path}")
    aligned = align_words(
        audio_path=audio_path,
        text=row["gen_text"],
        device=device,
        bundle=bundle,
        model=model,
        tokenizer=tokenizer,
        aligner=aligner,
    )
    indices = target_word_indices(row, [item.word for item in aligned])
    selected = [aligned[idx] for idx in indices]
    start_sec = min(item.start_sec for item in selected)
    end_sec = max(item.end_sec for item in selected)
    score = sum(item.score for item in selected) / len(selected)
    return {
        **row,
        "baseline_wav": str(audio_path.resolve()),
        "target_span_sec": [round(start_sec, 4), round(end_sec, 4)],
        "target_span_mode": "torchaudio_mms_fa_baseline",
        "target_aligned_text": " ".join(item.word.text for item in selected),
        "target_word_indices": indices,
        "target_alignment_score": round(score, 6),
    }


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("stem", "gen_text"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing {key}: {row}")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-dir",
        default="tto_eval/model_cache/torchaudio_mms_fa",
        help="Directory containing/downloading the MMS_FA checkpoint model.pt",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow torchaudio to download the MMS_FA checkpoint if missing.",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    bundle, model, tokenizer, aligner = load_mms_fa(
        args.device,
        Path(args.model_dir),
        allow_download=args.allow_download,
    )
    aligned_rows = [
        align_row(
            row,
            Path(args.baseline_dir),
            device=args.device,
            bundle=bundle,
            model=model,
            tokenizer=tokenizer,
            aligner=aligner,
        )
        for row in rows
    ]
    write_jsonl(Path(args.out), aligned_rows)
    print(f"wrote {len(aligned_rows)} aligned rows to {args.out}")
    for row in aligned_rows[:5]:
        print(
            f"{row['stem']}: {row['target_aligned_text']} "
            f"{row['target_span_sec']} score={row['target_alignment_score']}"
        )


if __name__ == "__main__":
    main()
