#!/usr/bin/env bash
set -euo pipefail

# Defaults (mirror README example).
WINDOW_SIZE=1.0
HOP_SIZE=0.5
OPT_AT="2,4,6,8,10,12,14,16,18,20,24,28"
OPT_STEPS=50
OPT_LR=1e-2
LOSS_MODE=value          # value | embedding
VAD_LEVEL=frame          # frame | utter | both
VAD_SLIDE_MODE=hidden    # hidden (默认: 整段一次 forward + hidden 滑窗) | audio (旧)
BATCH_SIZE=8
REF_DIR=asset
REF_TEXT="Kids are talking by the door. Kids are talking by the door."
GEN_TEXT="Dogs are walking on the floor. Dogs are walking on the floor."
SKIP_EVAL=0

# 本地权重（留空则走 HF cache）。
CKPT_FILE=/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/model_1250000.safetensors
VOCAB_FILE=/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/vocab.txt
VOCODER_LOCAL_PATH=/mnt/disk1/models/vocos-mel-24khz

usage() {
  cat <<EOF
Usage: $0 [options]
  --window-size VAL   (default: ${WINDOW_SIZE})
  --hop-size    VAL   (default: ${HOP_SIZE})
  --opt-at      CSV   (default: ${OPT_AT})
  --opt-steps   INT   (default: ${OPT_STEPS})
  --opt-lr      VAL   (default: ${OPT_LR})
  --loss-mode   STR   value|embedding (default: ${LOSS_MODE})
  --vad-level   STR   frame|utter|both (default: ${VAD_LEVEL})
  --vad-slide-mode STR  hidden|audio (default: ${VAD_SLIDE_MODE})
                  hidden = single backbone forward + slide on hidden state
                  audio  = legacy: per-window wav2vec2 forward (slower, OOD)
  --batch-size  INT   (default: ${BATCH_SIZE})
  --ref-dir     DIR   (default: ${REF_DIR})
  --skip-eval         skip post-generation batch eval
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --hop-size)    HOP_SIZE="$2";    shift 2 ;;
    --opt-at)      OPT_AT="$2";      shift 2 ;;
    --opt-steps)   OPT_STEPS="$2";   shift 2 ;;
    --opt-lr)      OPT_LR="$2";      shift 2 ;;
    --loss-mode)      LOSS_MODE="$2";       shift 2 ;;
    --vad-level)      VAD_LEVEL="$2";       shift 2 ;;
    --vad-slide-mode) VAD_SLIDE_MODE="$2";  shift 2 ;;
    --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
    --ref-dir)     REF_DIR="$2";     shift 2 ;;
    --skip-eval)   SKIP_EVAL=1;      shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

OPT_AT_SLUG="${OPT_AT//,/-}"
# TAG prefixes loss-mode + vad-level so different loss configurations land in
# sibling directories instead of overwriting each other. _sm<mode> suffix
# distinguishes audio-slide vs hidden-slide implementations.
TAG="${LOSS_MODE}-${VAD_LEVEL}_w${WINDOW_SIZE}_h${HOP_SIZE}_at${OPT_AT_SLUG}_s${OPT_STEPS}_lr${OPT_LR}_sm${VAD_SLIDE_MODE}"

cd "$(dirname "$0")"

OUT_DIR="tto_outputs/${TAG}"
VIZ_DIR="tto_viz/${TAG}"

extra_tto_args=()
[[ -n "${CKPT_FILE}"          ]] && extra_tto_args+=(--ckpt-file          "${CKPT_FILE}")
[[ -n "${VOCAB_FILE}"         ]] && extra_tto_args+=(--vocab-file         "${VOCAB_FILE}")
[[ -n "${VOCODER_LOCAL_PATH}" ]] && extra_tto_args+=(--vocoder-local-path "${VOCODER_LOCAL_PATH}")

python src/f5_tts/infer/tto.py \
  "${extra_tto_args[@]}" \
  --ref-text "${REF_TEXT}" \
  --gen-text "${GEN_TEXT}" \
  --loss-mode "${LOSS_MODE}" \
  --opt-at "${OPT_AT}" --opt-steps "${OPT_STEPS}" --opt-lr "${OPT_LR}" \
  --vad-level "${VAD_LEVEL}" \
  --vad-slide-mode "${VAD_SLIDE_MODE}" \
  --window-size "${WINDOW_SIZE}" --hop-size "${HOP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ref-dir "${REF_DIR}" \
  --output "${OUT_DIR}" \
  --viz-path "${VIZ_DIR}"

if [[ "${SKIP_EVAL}" -eq 1 ]]; then
  echo "[run_tto] --skip-eval set, stopping after generation"
  exit 0
fi

echo
echo "=========================================================="
echo "[run_tto] batch eval: ${OUT_DIR}  vs  ${REF_DIR}"
echo "=========================================================="
python src/f5_tts/eval/batch_eval.py \
  --gen-dir "${OUT_DIR}" \
  --ref-dir "${REF_DIR}" \
  --gen-text "${GEN_TEXT}" \
  --out-csv "${OUT_DIR}/metrics.csv"
