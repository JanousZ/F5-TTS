"""Text → frame-level VAD trajectory for TTO loss.

Pipeline:
  gen_text  → tokenize → look up VAD per token → uniform stretch over n_win
            → reorder + rescale to match audeering's (A, D, V) ∈ [-1, 1]
            → (n_win, 3) tensor

Three backends behind a single ``TextVADExtractor`` interface:
  ``lexicon`` (implemented): NRC-VAD word lookup. Deterministic, zero compute,
                              context-free (no negation handling).
  ``model``   (stub):        Future BERT-VAD style regressor.
  ``llm``     (stub):        Future LLM zero-shot rating with disk cache.

Output convention:
  - ``get_token_vad(text)`` returns NATIVE-space NRC values (V, A, D) ∈ [0, 1]
  - ``get_trajectory(text, n_win)`` returns audeering-space (A, D, V) ∈ [-1, 1]

The audeering wav2vec2 model used in tto.py outputs (arousal, dominance,
valence). NRC-VAD is in (valence, arousal, dominance) order — conversion
flips dimensions as well as scales [0, 1] → [-1, 1].

python src/f5_tts/infer/text_VAD.py \
    --text "I am very happy today" --n-win 20
"""

from __future__ import annotations

import re
from pathlib import Path

import torch


NRC_VAD_PATH = "/mnt/disk1/models/nrc_vad/NRC-VAD-Lexicon-v2.1.txt"
NRC_DOWNLOAD_URL = "https://saifmohammad.com/WebPages/nrc-vad.html"

# Audeering 输出 V/A/D 名义上 ∈ [0, 1] (实测 ESD 上偶有 ~1.15 outlier)，
# NRC-VAD 也在 [0, 1] —— 默认 scale (0, 1) = identity affine, 只 reorder 不
# rescale. 跑通后如果想做 distribution calibration (audeering ESD 分布偏
# high-arousal / low-valence), 可以传更精细的 (lo, hi) 或者每维独立 affine.
AUDEERING_SCALE_DEFAULT = (0.0, 1.0)

NEUTRAL_VAD = (0.5, 0.5, 0.5)  # OOV fallback in NRC space


# ---------- helpers ----------

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

WEIGHT_MODES = ("uniform", "chars", "phonemes")


def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation + split. Apostrophe-bearing tokens kept
    intact (e.g., ``don't`` stays as one token). Returns empty list if no
    alphabetic characters are present.
    """
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _token_weights(tokens: list[str], mode: str) -> list[int]:
    """Per-token weight, used to allocate time proportionally.

    ``uniform``  : each token = 1  (legacy behavior)
    ``chars``    : ``len(token)``  — character count, e.g., "happiness"=9, "cake"=4
    ``phonemes`` : phoneme count via ``g2p_en`` (optional dep)
    """
    if mode == "uniform":
        return [1] * len(tokens)
    if mode == "chars":
        return [max(1, len(t)) for t in tokens]
    if mode == "phonemes":
        try:
            from g2p_en import G2p
        except ImportError as e:
            raise ImportError(
                "phoneme weighting needs g2p_en: `pip install g2p_en` "
                "(downloads ~50 MB CMU dict + small NN for OOV)."
            ) from e
        if not hasattr(_token_weights, "_g2p"):
            _token_weights._g2p = G2p()  # cache instance
        out = []
        for tok in tokens:
            phones = [p for p in _token_weights._g2p(tok) if p[0].isalpha()]
            out.append(max(1, len(phones)))
        return out
    raise ValueError(f"unknown weight_mode {mode!r}; pick one of {WEIGHT_MODES}")


def _weighted_replicate(
    token_vads: list[tuple[float, float, float]], weights: list[int],
) -> torch.Tensor:
    """Repeat each token's VAD according to its weight.

    Result shape ``(sum(weights), 3)``. With ``chars`` mode this gives roughly
    one row per character; with ``phonemes`` mode roughly one row per phoneme.
    The downstream ``_align_frames`` linearly interpolates this to match the
    audio side's frame count, so absolute scale of weights doesn't matter —
    only their RATIOS shape the resulting time allocation.
    """
    if not token_vads:
        return torch.full((1, 3), NEUTRAL_VAD[0])
    expanded: list[tuple[float, float, float]] = []
    for vad, w in zip(token_vads, weights):
        expanded.extend([vad] * max(1, int(w)))
    return torch.tensor(expanded, dtype=torch.float32)  # (sum_w, 3)


def _weighted_stretch(
    token_vads: list[tuple[float, float, float]],
    weights: list[int],
    n_win: int,
) -> torch.Tensor:
    """Variable-share stretch. Token i occupies ``weights[i] / sum(weights)``
    of the n_win-long output. Same logical result as ``_uniform_stretch``
    applied to the replicated tensor, but skips the intermediate allocation.
    """
    if n_win <= 0:
        raise ValueError(f"n_win must be > 0, got {n_win}")
    if not token_vads:
        return torch.full((n_win, 3), NEUTRAL_VAD[0])

    arr = torch.tensor(token_vads, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32).clamp(min=1e-8)
    cum = torch.cumsum(w, dim=0) / w.sum()                 # ∈ [0, 1]
    centers = (torch.arange(n_win) + 0.5) / n_win          # window centers
    idx = torch.searchsorted(cum, centers).clamp(max=len(w) - 1)
    return arr[idx]


def _nrc_to_audeering(
    nrc: torch.Tensor, scale: tuple[float, float] = AUDEERING_SCALE_DEFAULT,
) -> torch.Tensor:
    """NRC (V, A, D) ∈ [0, 1]  →  audeering (A, D, V) ∈ ``scale``.

    Two operations:
      1. reorder dim from (V, A, D) to (A, D, V)
      2. linear map [0, 1] → [scale_low, scale_high]
    """
    lo, hi = scale
    v, a, d = nrc[..., 0], nrc[..., 1], nrc[..., 2]
    reordered = torch.stack([a, d, v], dim=-1)  # (..., 3) in (A, D, V)
    return reordered * (hi - lo) + lo


# ---------- backends ----------

class _LexiconBackend:
    """NRC-VAD word lookup.

    Loads the lexicon lazily on first call. OOV tokens fall back to neutral.
    """

    def __init__(self, lexicon_path: str | Path = NRC_VAD_PATH):
        self.lexicon_path = Path(lexicon_path)
        self._table: dict[str, tuple[float, float, float]] | None = None

    def _ensure_loaded(self) -> None:
        if self._table is not None:
            return
        if not self.lexicon_path.is_file():
            raise FileNotFoundError(
                f"NRC-VAD lexicon not found at {self.lexicon_path}.\n"
                f"Download from {NRC_DOWNLOAD_URL} (free, requires email "
                f"registration), unzip, and place ``NRC-VAD-Lexicon.txt`` at "
                f"the path above. Expected 4-column TSV: "
                f"word<TAB>valence<TAB>arousal<TAB>dominance."
            )
        table: dict[str, tuple[float, float, float]] = {}
        with open(self.lexicon_path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    continue  # skip malformed
                word, v_s, a_s, d_s = parts
                if ln == 1 and not _looks_like_float(v_s):
                    continue  # header row
                try:
                    table[word.lower()] = (float(v_s), float(a_s), float(d_s))
                except ValueError:
                    continue
        if not table:
            raise RuntimeError(
                f"loaded 0 entries from {self.lexicon_path}; file format "
                f"unexpected (need 4-column TSV with floats)."
            )
        self._table = table

    @property
    def coverage(self) -> int:
        self._ensure_loaded()
        assert self._table is not None
        return len(self._table)

    def get_token_vad(
        self, text: str,
    ) -> list[tuple[str, float, float, float]]:
        """Returns ``[(token, V, A, D), ...]`` in NRC space [0, 1]. OOV
        tokens get the neutral ``(0.5, 0.5, 0.5)``.
        """
        self._ensure_loaded()
        assert self._table is not None
        out: list[tuple[str, float, float, float]] = []
        for tok in _tokenize(text):
            v, a, d = self._table.get(tok, NEUTRAL_VAD)
            out.append((tok, v, a, d))
        return out


def _looks_like_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


class _ModelBackend:
    """Stub: BERT/RoBERTa-VAD regressor. To be implemented when a verified
    checkpoint is selected."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "model backend is not implemented yet. Use backend='lexicon' for "
            "now, or check back after a checkpoint has been verified "
            "(SungjoonPark/EmotionDetection candidate)."
        )

    def get_token_vad(self, text: str):
        raise NotImplementedError


class _LLMBackend:
    """Stub: LLM zero-shot rating with on-disk JSON cache. To be implemented
    when API integration is decided."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "llm backend is not implemented yet. Use backend='lexicon' for "
            "now."
        )

    def get_token_vad(self, text: str):
        raise NotImplementedError


# ---------- public class ----------

class TextVADExtractor:
    """Unified text → VAD-trajectory interface for TTO loss.

    Example
    -------
        ext = TextVADExtractor(backend="lexicon")
        # NRC-space per-token VAD (debug / inspection)
        ext.get_token_vad("I am very happy today")
        # → [('i', 0.5, 0.5, 0.5), ('am', ...), ..., ('today', ...)]

        # Audeering-space trajectory matching audio side
        traj = ext.get_trajectory("I am very happy today", n_win=17)
        # → tensor of shape (17, 3) in (A, D, V) order, ∈ [-1, 1]
    """

    BACKENDS = {"lexicon", "model", "llm"}

    def __init__(
        self,
        backend: str = "lexicon",
        *,
        lexicon_path: str | Path = NRC_VAD_PATH,
        scale: tuple[float, float] = AUDEERING_SCALE_DEFAULT,
        weight_mode: str = "chars",
        **backend_kwargs,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(
                f"backend must be one of {self.BACKENDS}, got {backend!r}"
            )
        if weight_mode not in WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}"
            )
        self.backend_name = backend
        self.scale = scale
        self.weight_mode = weight_mode
        if backend == "lexicon":
            self._backend = _LexiconBackend(lexicon_path=lexicon_path)
        elif backend == "model":
            self._backend = _ModelBackend(**backend_kwargs)
        else:
            self._backend = _LLMBackend(**backend_kwargs)

    # --- inspection / debugging ---

    def get_token_vad(
        self, text: str,
    ) -> list[tuple[str, float, float, float]]:
        """Per-token (V, A, D) in NRC-native [0, 1] space."""
        return self._backend.get_token_vad(text)

    def get_token_vad_with_weights(
        self, text: str,
    ) -> list[tuple[str, float, float, float, int]]:
        """[(token, V, A, D, weight), ...]. Weight is from ``self.weight_mode``."""
        rows = self._backend.get_token_vad(text)
        weights = _token_weights([t for (t, _, _, _) in rows], self.weight_mode)
        return [(t, v, a, d, w) for (t, v, a, d), w in zip(rows, weights)]

    # --- main API for TTO ---

    def get_per_token_trajectory(
        self,
        text: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        weight_mode: str | None = None,
    ) -> torch.Tensor:
        """Returns audeering-space (A, D, V) tensor with each token's VAD
        replicated according to its weight (chars / phonemes / uniform).

        Shape: ``(sum(weights), 3)``. With ``weight_mode='chars'`` (default),
        a sentence like "happiness cake" produces (9 + 4)=13 rows, where the
        first 9 carry happiness-VAD and the next 4 carry cake-VAD — so time
        share matches word length 9:4 once ``_align_frames`` linearly
        interpolates this to the audio's frame count.

        ``weight_mode=None`` uses ``self.weight_mode`` (set in constructor).
        """
        wm = weight_mode or self.weight_mode
        rows = self.get_token_vad(text)
        per_token = [(v, a, d) for (_, v, a, d) in rows]
        if not per_token:
            # Fallback for all-empty input: single neutral row.
            nrc = torch.tensor([NEUTRAL_VAD], dtype=torch.float32)
        else:
            tokens = [t for (t, _, _, _) in rows]
            weights = _token_weights(tokens, wm)
            nrc_replicated = _weighted_replicate(per_token, weights)
            nrc = nrc_replicated  # (sum_w, 3) in (V, A, D)
        traj = _nrc_to_audeering(nrc, scale=self.scale)
        return traj.to(device=device, dtype=dtype)

    def get_utterance_vad(
        self,
        text: str,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Returns ``(1, 3)`` audeering-space mean over all (in-vocab + OOV)
        tokens. Use this for ``vad_level='utter'`` mode where the TTO target
        is a single-vector summary."""
        per_token = self.get_per_token_trajectory(text, device=device, dtype=dtype)
        return per_token.mean(dim=0, keepdim=True)

    def get_trajectory(
        self,
        text: str,
        n_win: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        weight_mode: str | None = None,
    ) -> torch.Tensor:
        """Returns ``(n_win, 3)`` audeering-space (A, D, V) tensor where each
        token occupies a share of windows proportional to its weight (per
        ``weight_mode``, default = ``self.weight_mode``).

        Use this when you know n_win in advance. Otherwise prefer
        ``get_per_token_trajectory`` — that defers the stretch to the TTO
        loop's ``_align_frames`` and avoids a double interpolation step.
        """
        wm = weight_mode or self.weight_mode
        rows = self.get_token_vad(text)
        token_vads = [(v, a, d) for (_, v, a, d) in rows]
        tokens = [t for (t, _, _, _) in rows]
        weights = _token_weights(tokens, wm) if tokens else []
        nrc_traj = _weighted_stretch(token_vads, weights, n_win)  # (n_win, 3) (V,A,D)
        traj = _nrc_to_audeering(nrc_traj, scale=self.scale)      # (A, D, V)
        return traj.to(device=device, dtype=dtype)


# ---------- CLI for debugging ----------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect text → VAD trajectory.")
    p.add_argument("--text", required=True, help="input gen text")
    p.add_argument("--n-win", type=int, default=20,
                   help="number of audio windows to align to")
    p.add_argument("--backend", default="lexicon",
                   choices=sorted(TextVADExtractor.BACKENDS))
    p.add_argument("--lexicon-path", default=NRC_VAD_PATH)
    p.add_argument("--scale-low",  type=float, default=AUDEERING_SCALE_DEFAULT[0])
    p.add_argument("--scale-high", type=float, default=AUDEERING_SCALE_DEFAULT[1])
    p.add_argument("--weight-mode", choices=WEIGHT_MODES, default="chars",
                   help="time allocation per token: 'uniform' (each = 1), "
                        "'chars' (default, len(token)), 'phonemes' (g2p_en)")
    args = p.parse_args()

    ext = TextVADExtractor(
        backend=args.backend,
        lexicon_path=args.lexicon_path,
        scale=(args.scale_low, args.scale_high),
        weight_mode=args.weight_mode,
    )

    print(f"=== per-token VAD + weights (NRC space, weight_mode={args.weight_mode}) ===")
    rows = ext.get_token_vad_with_weights(args.text)
    total_w = sum(w for *_, w in rows) if rows else 0
    for tok, v, a, d, w in rows:
        oov = "  (OOV → neutral)" if (v, a, d) == NEUTRAL_VAD else ""
        share = (w / total_w * 100) if total_w else 0.0
        print(f"  {tok:18s}  V={v:.3f}  A={a:.3f}  D={d:.3f}  "
              f"weight={w:>2d}  share={share:5.1f}%{oov}")
    print(f"  total weight = {total_w}")

    traj = ext.get_trajectory(args.text, args.n_win)
    print(f"\n=== trajectory (audeering {args.scale_low}..{args.scale_high}, "
          f"A/D/V order, n_win={args.n_win}) ===")
    print(f"  shape: {tuple(traj.shape)}")
    print(f"  per-window:")
    for i, (a, d, v) in enumerate(traj.tolist()):
        print(f"    [{i:02d}]  A={a:+.3f}  D={d:+.3f}  V={v:+.3f}")
    print(f"\n  range: A ∈ [{traj[:,0].min():+.3f}, {traj[:,0].max():+.3f}]  "
          f"D ∈ [{traj[:,1].min():+.3f}, {traj[:,1].max():+.3f}]  "
          f"V ∈ [{traj[:,2].min():+.3f}, {traj[:,2].max():+.3f}]")

    pt = ext.get_per_token_trajectory(args.text)
    print(f"\n=== per-token trajectory (replicated by weight, audeering A/D/V) ===")
    print(f"  shape: {tuple(pt.shape)}  (= sum of weights, = {total_w})")
    print(f"  这就是 text_tto.py 实际喂给 ref_vad_features 的 target")
