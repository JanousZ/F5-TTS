#!/usr/bin/env bash
set -euo pipefail

# UTMOS-only sweep: VAD-related args stay at tto.py defaults; we sweep
# (utmos_opt_at × utmos_opt_steps) over the full asset/ (120 wavs).
#
# Fixed: --utmos-weight, --utmos-opt-lr; VAD defaults from tto.py.

UTMOS_WEIGHT=0.0001
UTMOS_OPT_LR=1e-2

UTMOS_OPT_AT_LIST=(
    "2,4,6,8,10,12,14"
    "10,12,14"
    "16,18,20"
)
UTMOS_OPT_STEPS_LIST=(25 50)

BATCH_SIZE=120
REF_DIR=asset
REF_TEXT="Kids are talking by the door. Kids are talking by the door."
GEN_TEXT="Dogs are walking on the floor. Dogs are walking on the floor."

# Local checkpoints (leave empty to fall back to HF cache).
CKPT_FILE=/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/model_1250000.safetensors
VOCAB_FILE=/mnt/disk1/models/F5-TTS_Emilia-ZH-EN/vocab.txt
VOCODER_LOCAL_PATH=/mnt/disk1/models/vocos-mel-24khz

GPUS="0"
LOG_DIR=""
SKIP_EVAL=0
DEFER_EVAL=0

usage() {
  cat <<EOF
Usage: $0 [--gpus LIST] [--log-dir DIR] [--skip-eval | --defer-eval]
  --gpus LIST    GPU ids, e.g. "0", "0,1,3", "0-3". One config per GPU at a time.
  --log-dir DIR  Per-run stdout logs dir (default: tto_outputs/_logs/utmos_<ts>/).
  --skip-eval    Generation only; never run batch_eval.py.
  --defer-eval   Run eval serially on the first GPU AFTER all generation finishes
                 (use when each TTO run's VRAM peak is close to capacity).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2";    shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --skip-eval)   SKIP_EVAL=1;  shift ;;
    --defer-eval)  DEFER_EVAL=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"

expand_gpus() {
  local input="$1" out=""
  IFS=',' read -ra parts <<< "$input"
  for p in "${parts[@]}"; do
    if [[ "$p" == *-* ]]; then
      local lo="${p%-*}" hi="${p#*-}"
      for i in $(seq "$lo" "$hi"); do out+="$i "; done
    else
      out+="$p "
    fi
  done
  echo "${out% }"
}
read -ra GPU_LIST <<< "$(expand_gpus "$GPUS")"
n_gpus=${#GPU_LIST[@]}
if [[ $n_gpus -eq 0 ]]; then
  echo "no valid GPU ids parsed from '${GPUS}'" >&2; exit 1
fi

[[ -z "$LOG_DIR" ]] && LOG_DIR="tto_outputs/_logs/utmos_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Build the cartesian product (utmos_opt_at × utmos_opt_steps).
configs=()
for at in "${UTMOS_OPT_AT_LIST[@]}"; do
  for s in "${UTMOS_OPT_STEPS_LIST[@]}"; do
    configs+=("${at}|${s}")
  done
done
total=${#configs[@]}

cfg_to_tag() {
  local cfg="$1" at s
  IFS='|' read -r at s <<< "$cfg"
  local slug="${at//,/-}"
  echo "utmos_w${UTMOS_WEIGHT}_at${slug}_s${s}_lr${UTMOS_OPT_LR}"
}

echo "============================================================"
echo "sweep_utmos: ${total} configs across ${n_gpus} GPU(s): ${GPU_LIST[*]}"
echo "logs dir   : ${LOG_DIR}"
echo "batch size : ${BATCH_SIZE} (full asset)"
echo "============================================================"

declare -A gpu_pid pid_info
for g in "${GPU_LIST[@]}"; do gpu_pid[$g]=""; done

trap 'echo; echo "[sweep] interrupted — killing children"; \
      for g in "${GPU_LIST[@]}"; do \
        p="${gpu_pid[$g]:-}"; [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; \
      done; wait; exit 130' INT TERM

launch() {
  local gpu="$1" idx="$2" cfg="$3"
  local at s
  IFS='|' read -r at s <<< "$cfg"
  local tag; tag=$(cfg_to_tag "$cfg")
  local log="${LOG_DIR}/${idx}_gpu${gpu}_${tag}.log"
  local out_dir="tto_outputs/${tag}"

  local extra=()
  [[ -n "${CKPT_FILE}"          ]] && extra+=(--ckpt-file          "${CKPT_FILE}")
  [[ -n "${VOCAB_FILE}"         ]] && extra+=(--vocab-file         "${VOCAB_FILE}")
  [[ -n "${VOCODER_LOCAL_PATH}" ]] && extra+=(--vocoder-local-path "${VOCODER_LOCAL_PATH}")

  CUDA_VISIBLE_DEVICES="$gpu" python src/f5_tts/infer/tto.py \
    "${extra[@]}" \
    --ref-text "${REF_TEXT}" --gen-text "${GEN_TEXT}" \
    --batch-size "${BATCH_SIZE}" --ref-dir "${REF_DIR}" \
    --output "${out_dir}" \
    --utmos-weight  "${UTMOS_WEIGHT}" \
    --utmos-opt-at  "${at}" \
    --utmos-opt-steps "${s}" \
    --utmos-opt-lr  "${UTMOS_OPT_LR}" \
    >"$log" 2>&1 &
  local pid=$!
  gpu_pid[$gpu]=$pid
  pid_info[$pid]="gpu=${gpu} idx=${idx}/${total} tag=${tag}"
  echo "[dispatch] $(date +%H:%M:%S)  gpu=${gpu} pid=${pid}  [$idx/$total]  at=${at}  s=${s}  → ${log}"
}

reap_free_gpu() {
  while true; do
    for g in "${GPU_LIST[@]}"; do
      local pid="${gpu_pid[$g]:-}"
      if [[ -z "$pid" ]]; then echo "$g"; return; fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        echo "[finish  ] $(date +%H:%M:%S)  ${pid_info[$pid]}  exit=$?" >&2
        unset "pid_info[$pid]"
        gpu_pid[$g]=""
        echo "$g"; return
      fi
    done
    sleep 2
  done
}

run_eval() {
  local tag="$1" gpu="$2"
  local out_dir="tto_outputs/${tag}"
  shopt -s nullglob
  local wavs=("$out_dir"/*.wav)
  shopt -u nullglob
  if [[ ${#wavs[@]} -eq 0 ]]; then
    echo "[eval] SKIP ${out_dir} (no wav)"
    return 1
  fi
  echo "[eval] >>> ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" python src/f5_tts/eval/batch_eval.py \
    --gen-dir "$out_dir" \
    --ref-dir "${REF_DIR}" \
    --gen-text "${GEN_TEXT}" \
    --out-csv "$out_dir/metrics.csv" \
    2>&1 | tee "${LOG_DIR}/eval_${tag}.log" | tail -15
}

t0=$SECONDS
for i in "${!configs[@]}"; do
  idx=$((i + 1))
  gpu=$(reap_free_gpu)
  launch "$gpu" "$idx" "${configs[$i]}"
done

# Drain remaining jobs.
for g in "${GPU_LIST[@]}"; do
  pid="${gpu_pid[$g]:-}"
  [[ -z "$pid" ]] && continue
  wait "$pid" || true
  echo "[finish  ] $(date +%H:%M:%S)  ${pid_info[$pid]}  exit=$?"
done

echo
echo "All ${total} runs finished in $((SECONDS - t0))s."
echo "Results      : tto_outputs/utmos_*/"
echo "Per-run logs : ${LOG_DIR}/"

if [[ $SKIP_EVAL -eq 1 ]]; then
  echo "[sweep_utmos] --skip-eval set, stopping after generation"
  exit 0
fi

if [[ $DEFER_EVAL -eq 1 ]]; then
  echo
  echo "============================================================"
  echo "deferred batch eval (serial on gpu=${GPU_LIST[0]})"
  echo "============================================================"
  eval_t0=$SECONDS
  for cfg in "${configs[@]}"; do
    tag=$(cfg_to_tag "$cfg")
    run_eval "$tag" "${GPU_LIST[0]}" || true
  done
  echo
  echo "deferred eval finished in $((SECONDS - eval_t0))s"
else
  # Inline eval: serialize on first GPU right after generation.
  echo
  echo "============================================================"
  echo "inline batch eval (serial on gpu=${GPU_LIST[0]})"
  echo "============================================================"
  for cfg in "${configs[@]}"; do
    tag=$(cfg_to_tag "$cfg")
    run_eval "$tag" "${GPU_LIST[0]}" || true
  done
fi

echo "Summaries    : tto_outputs/*/metrics.summary.txt"
