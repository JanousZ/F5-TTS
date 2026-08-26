"""Build a small local target-span emotion-control pilot manifest.

The pilot deliberately uses generation text that is unrelated to the ESD
prompt/target recordings.  Target spans are left pending because the intended
workflow is:

1. generate a no-TTO baseline;
2. force-align the baseline audio to ``gen_text``;
3. write the aligned target-word span back into a second manifest;
4. run local TTO with ``vad_level=target``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ESD_EMO_OFFSET = {
    "Neutral": 0,
    "Angry": 350,
    "Happy": 700,
    "Sad": 1050,
    "Surprise": 1400,
}

DEFAULT_SPEAKERS = ["0011"]
DEFAULT_SLOTS = [7, 15, 23, 30]
DEFAULT_TARGETS = [
    ("The silver train is leaving before sunrise.", "silver"),
    ("Please send the final report after lunch.", "final"),
    ("A quiet garden waited behind the station.", "quiet"),
    ("We should remember the small detail.", "remember"),
]
DEFAULT_EMOTIONS = ["Happy", "Sad", "Surprise", "Angry"]


def parse_transcript(path: Path) -> dict[int, str]:
    """Read ESD's speaker transcript keyed by the full utterance id."""
    rows: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        utt_id = int(parts[0].split("_")[1])
        rows[utt_id] = parts[1].strip()
    return rows


def find_word_span(text: str, word: str) -> tuple[int, int]:
    match = re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.I)
    if match is None:
        raise ValueError(f"target word {word!r} is not present in {text!r}")
    return match.start(), match.end()


def planned_generation_duration(
    prompt_wav: Path,
    prompt_text: str,
    gen_text: str,
    *,
    sample_rate: int = 16000,
) -> float:
    """Mirror the duration heuristic in ``infer.tto`` for English text."""
    import torchaudio

    wav, sr = torchaudio.load(str(prompt_wav))
    prompt_duration = wav.shape[-1] / sr
    prompt_chars = max(len(prompt_text.encode("utf-8")), 1)
    gen_chars = len(gen_text.encode("utf-8"))
    return prompt_duration * gen_chars / prompt_chars


def build(
    *,
    esd_root: Path,
    speakers: list[str],
    slots: list[int],
    emotions: list[str],
    out_path: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rows: list[dict] = []

    for speaker in speakers:
        speaker_root = esd_root / speaker
        transcript = parse_transcript(speaker_root / f"{speaker}.txt")
        for slot, (gen_text, target_word) in zip(slots, DEFAULT_TARGETS):
            speaker_wav = speaker_root / "Neutral" / f"{speaker}_{slot:06d}.wav"
            speaker_text = transcript[slot]
            start_char, end_char = find_word_span(gen_text, target_word)
            for emotion in emotions:
                utt_id = ESD_EMO_OFFSET[emotion] + slot
                ref_wav = speaker_root / emotion / f"{speaker}_{utt_id:06d}.wav"
                ref_text = transcript[utt_id]
                if not speaker_wav.exists() or not ref_wav.exists():
                    raise FileNotFoundError(f"missing pilot audio: {speaker_wav} / {ref_wav}")
                gen_duration = planned_generation_duration(
                    ref_wav, ref_text, gen_text,
                )
                rows.append({
                    "stem": f"{speaker}_slot{slot:03d}_{emotion.lower()}",
                    "speaker": speaker,
                    "source_slot": slot,
                    "emotion": emotion,
                    "prompt_wav": str(ref_wav.resolve()),
                    "prompt_text": ref_text,
                    "ref_wav": str(ref_wav.resolve()),
                    "ref_text": ref_text,
                    "speaker_wav": str(speaker_wav.resolve()),
                    "speaker_text": speaker_text,
                    "target_audio": str(ref_wav.resolve()),
                    "gen_text": gen_text,
                    "target_word": target_word,
                    "target_char_span": [start_char, end_char],
                    "target_span_sec": None,
                    "planned_gen_duration_sec": gen_duration,
                    "target_span_mode": "pending_forced_alignment",
                })

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")
    print(f"speakers={speakers} slots={slots} emotions={emotions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esd-root", default="/mnt/disk1/datasets/ESD")
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--slots", nargs="+", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--emotions", nargs="+", choices=DEFAULT_EMOTIONS,
                        default=DEFAULT_EMOTIONS)
    parser.add_argument(
        "--out",
        default="tto_eval/esd_local_target/pilot_manifest.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.slots) != len(DEFAULT_TARGETS):
        raise SystemExit(
            f"pilot currently needs exactly {len(DEFAULT_TARGETS)} slots"
        )
    build(
        esd_root=Path(args.esd_root),
        speakers=args.speakers,
        slots=args.slots,
        emotions=args.emotions,
        out_path=Path(args.out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
