"""Build an emo-change evaluation set from ESD English speakers.

ESD records the same 350 sentences in five emotions per speaker. For each
(speaker, emo-pair, sample) we concat ``emoA/{spk}_{offset+slot}.wav`` +
silence + ``emoB/{spk}_{offset+slot}.wav`` → ``refs/{stem}.wav`` and emit a
JSONL row with ref_wav / ref_text / gen_text / speaker / emo_pair.

``--gen-text-sources`` selects one or more gen_text policies; one manifest is
written per policy, all sharing the same ``refs/`` directory so the resulting
manifests form a controlled comparison (identical ref audio, different
gen_text).

Policies:

* ``disjoint`` — the 350 text slots are split into a ref pool and a gen pool
  (sizes via ``--ref-slot-count``); ref_text and gen_text are drawn from
  these disjoint pools so they never share a sentence.
* ``same-as-ref`` — gen_text is identical to ref_text (same slot IDs).

Run::

    # Default: produces manifest_disjoint.jsonl AND manifest_sameText.jsonl,
    # both pointing at the same refs/.
    python src/f5_tts/eval/build_emochange_eval.py \\
        --out-dir tto_eval/esd_emochange

    # Single policy only:
    python src/f5_tts/eval/build_emochange_eval.py \\
        --gen-text-sources disjoint --out-dir tto_eval/esd_emochange
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio


# ESD English transcripts: utterance id n (1..1750) is text slot ((n-1) % 350)+1
# rendered in emotion floor((n-1) // 350). Offsets are the (id - slot) base.
ESD_EMO_OFFSET = {
    "Neutral":  0,
    "Angry":    350,
    "Happy":    700,
    "Sad":      1050,
    "Surprise": 1400,
}

# 10 directed emo transitions covering energy-up, energy-down, and
# valence-flip patterns.
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("Neutral",  "Angry"),
    ("Neutral",  "Happy"),
    ("Neutral",  "Sad"),
    ("Neutral",  "Surprise"),
    ("Happy",    "Sad"),
    ("Happy",    "Angry"),
    ("Angry",    "Happy"),
    ("Angry",    "Sad"),
    ("Sad",      "Happy"),
    ("Surprise", "Sad"),
]

DEFAULT_SPEAKERS = ["0011", "0012"]


def parse_transcript(path: Path) -> dict[int, str]:
    """Parse an ESD ``{spk}.txt`` file → ``{slot: text}`` for slots 1..350.

    Files are CRLF. Each line is ``id<TAB>text<TAB>emotion``. All five
    emotions share the same 350 sentences, so we keep the first occurrence
    of each slot.
    """
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        utt_id = int(parts[0].split("_")[1])
        slot = ((utt_id - 1) % 350) + 1
        if slot not in out:
            out[slot] = parts[1].strip()
    return out


def join_texts(a: str, b: str) -> str:
    return f"{a.rstrip()} {b.lstrip()}"


def build(
    esd_root: Path,
    speakers: list[str],
    pairs: list[tuple[str, str]],
    samples_per_pair: int,
    ref_slot_count: int,
    silence_ms: int,
    seed: int,
    out_dir: Path,
    gen_text_sources: list[str],
) -> None:
    """Build the eval set and emit one manifest per gen_text source.

    All policies share the same ``refs/`` directory and per-stem ref slot
    draws, so a row with stem ``X`` references the same ref audio across all
    emitted manifests. Only ``gen_text`` / ``gen_slots`` differ.
    """
    unknown = [s for s in gen_text_sources if s not in {"disjoint", "same-as-ref"}]
    if unknown:
        raise ValueError(f"unknown gen_text_sources: {unknown}")
    if not gen_text_sources:
        raise ValueError("need at least one gen_text_sources entry")

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(exist_ok=True)

    # One row buffer per gen-text policy.
    manifests: dict[str, list[dict]] = {p: [] for p in gen_text_sources}

    for spk in speakers:
        spk_dir = esd_root / spk
        transcript = parse_transcript(spk_dir / f"{spk}.txt")
        if len(transcript) != 350:
            raise RuntimeError(
                f"{spk}: expected 350 text slots, got {len(transcript)}"
            )

        # ref_pool drives ref slot picks; if 'disjoint' is among policies we
        # need a gen_pool for that policy. Both pools are speaker-scoped and
        # shared across all policies that need them.
        slots = list(range(1, 351))
        rng.shuffle(slots)
        ref_pool = sorted(slots[:ref_slot_count])
        gen_pool_disjoint = sorted(slots[ref_slot_count:])
        if len(ref_pool) < 2 or len(gen_pool_disjoint) < 2:
            raise ValueError(
                "Need ≥2 slots in each pool; got "
                f"ref={len(ref_pool)}, gen={len(gen_pool_disjoint)}"
            )

        for emo_a, emo_b in pairs:
            for i in range(samples_per_pair):
                sa_r, sb_r = rng.sample(ref_pool, 2)
                # Always advance the disjoint gen pool draw so the rng state
                # is independent of which policies were requested — keeps
                # ref_pool draws reproducible regardless of policy set.
                sa_g_dj, sb_g_dj = rng.sample(gen_pool_disjoint, 2)

                utt_a = ESD_EMO_OFFSET[emo_a] + sa_r
                utt_b = ESD_EMO_OFFSET[emo_b] + sb_r
                wav_a_path = spk_dir / emo_a / f"{spk}_{utt_a:06d}.wav"
                wav_b_path = spk_dir / emo_b / f"{spk}_{utt_b:06d}.wav"

                wav_a, sr_a = torchaudio.load(str(wav_a_path))
                wav_b, sr_b = torchaudio.load(str(wav_b_path))
                if sr_a != sr_b:
                    raise RuntimeError(
                        f"sr mismatch: {wav_a_path}={sr_a}, {wav_b_path}={sr_b}"
                    )
                sil = torch.zeros(wav_a.shape[0], int(sr_a * silence_ms / 1000))
                ref_wav = torch.cat([wav_a, sil, wav_b], dim=1)

                stem = f"{spk}_{emo_a.lower()}2{emo_b.lower()}_{i:03d}"
                ref_path = refs_dir / f"{stem}.wav"
                torchaudio.save(str(ref_path), ref_wav, sr_a)

                ref_text = join_texts(transcript[sa_r], transcript[sb_r])
                base_row = {
                    "stem": stem,
                    "speaker": spk,
                    "emo_pair": [emo_a, emo_b],
                    "ref_wav": str(ref_path.resolve()),
                    "ref_text": ref_text,
                    "ref_src_wavs": [str(wav_a_path.resolve()),
                                     str(wav_b_path.resolve())],
                    "ref_slots": [sa_r, sb_r],
                    "ref_sr": sr_a,
                    "silence_ms": silence_ms,
                }

                for policy in gen_text_sources:
                    if policy == "disjoint":
                        sa_g, sb_g = sa_g_dj, sb_g_dj
                    else:  # same-as-ref
                        sa_g, sb_g = sa_r, sb_r
                    row = dict(base_row)
                    row["gen_text"] = join_texts(transcript[sa_g], transcript[sb_g])
                    row["gen_slots"] = [sa_g, sb_g]
                    row["gen_text_source"] = policy
                    manifests[policy].append(row)

    for policy, rows in manifests.items():
        suffix = "disjoint" if policy == "disjoint" else "sameText"
        manifest_path = out_dir / f"manifest_{suffix}.jsonl"
        with open(manifest_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_spk: dict[str, int] = {}
        by_pair: dict[str, int] = {}
        for r in rows:
            by_spk[r["speaker"]] = by_spk.get(r["speaker"], 0) + 1
            key = f'{r["emo_pair"][0]}->{r["emo_pair"][1]}'
            by_pair[key] = by_pair.get(key, 0) + 1
        print(f"[build:{policy}] {len(rows)} stems → {manifest_path}")
        print(f"[build:{policy}] per spk : {by_spk}")
        print(f"[build:{policy}] per pair: {by_pair}")
    print(f"[build] shared refs → {refs_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esd-root", default="/mnt/disk1/datasets/ESD")
    ap.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS,
                    help="ESD speaker IDs (e.g. 0011 0012)")
    ap.add_argument("--samples-per-pair", type=int, default=15,
                    help="Stems per (speaker, emo-pair); default 15")
    ap.add_argument("--ref-slot-count", type=int, default=200,
                    help="How many of the 350 text slots go to ref pool")
    ap.add_argument("--silence-ms", type=int, default=200,
                    help="Silence inserted between the two emo segments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen-text-sources", nargs="+",
                    choices=["disjoint", "same-as-ref"],
                    default=["disjoint", "same-as-ref"],
                    help="Which gen_text policies to emit manifests for. "
                         "Default emits both; both share the same refs/.")
    ap.add_argument("--out-dir", default="tto_eval/esd_emochange")
    args = ap.parse_args()

    build(
        esd_root=Path(args.esd_root),
        speakers=args.speakers,
        pairs=DEFAULT_PAIRS,
        samples_per_pair=args.samples_per_pair,
        ref_slot_count=args.ref_slot_count,
        silence_ms=args.silence_ms,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        gen_text_sources=args.gen_text_sources,
    )


if __name__ == "__main__":
    main()
