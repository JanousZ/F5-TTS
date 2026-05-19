#!/usr/bin/env bash
set -euo pipefail

# 4 个 README 实验配置: exp1 / exp2 / exp3 / exp4。共同部分在底下变量里写一次,
# 每行只列差异: tag | vad_level | extra_tto_args (UTMOS + grad-proj 或空)。
configs=(
    "exp11|frame|"
    "exp22|both|"
    "exp33|both|--utmos-opt-at 2,4,6,8,10,12,14 --utmos-opt-steps 25 --utmos-weight 0.0001 --grad-proj ortho"
    "exp44|frame|--utmos-opt-at 2,4,6,8,10,12,14 --utmos-opt-steps 25 --utmos-weight 0.0001 --grad-proj ortho"
)

# Common knobs (4 个 config 完全一致)。
REF_TEXT="Kids are talking by the door. Kids are talking by the door."
GEN_TEXT="Dogs are walking on the floor. Dogs are walking on the floor."
LOSS_MODE=embedding
OPT_AT="2,4,6,8,10,12,14"
OPT_STEPS=50
OPT_LR=1e-2
WINDOW_SIZE=1.0
HOP_SIZE=0.5
BATCH_SIZE=120
REF_DIR=asset
OUT_BASE=tto_outputs
NUM_RUNS=4
SEEDS=(12345 23456 34567 45678)

GPUS="0"
LOG_DIR=""
SKIP_EVAL=0
DEFER_EVAL=0

usage() {
  cat <<EOF
Usage: $0 [--gpus LIST] [--runs N] [--log-dir DIR] [--skip-eval | --defer-eval]
  --gpus LIST    GPU ids, e.g. "0", "0,1,3", "0-3". 每张卡同时跑一个 job,
                 跑完自动领下一条。jobs 数量 ≤ GPU 数时一次性全部并发。
  --runs N       每个 config 重复跑几次 (default: ${NUM_RUNS})。第 r 次用 seed=r,
                 输出落到 ${OUT_BASE}/<tag>/run<r>/。
  --log-dir DIR  Per-run stdout 日志目录 (default: ${OUT_BASE}/_logs/exps_<ts>/).
  --skip-eval    只生成, 不跑 batch_eval.py。
  --defer-eval   全部生成结束后再串行跑 eval (在第一张卡上)。
                 不传则采用 inline 模式: 生成全部跑完后立即在第一张卡串行 eval。
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2";    shift 2 ;;
    --runs)        NUM_RUNS="$2"; shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --skip-eval)   SKIP_EVAL=1;  shift ;;
    --defer-eval)  DEFER_EVAL=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ $NUM_RUNS -ge 1 ]] || { echo "--runs must be >= 1 (got ${NUM_RUNS})" >&2; exit 1; }
[[ $NUM_RUNS -le ${#SEEDS[@]} ]] || {
  echo "--runs (${NUM_RUNS}) exceeds SEEDS array length (${#SEEDS[@]}); add more entries to SEEDS." >&2
  exit 1
}

cd "$(dirname "$0")"

# 把 "0-3,5" 展开为 "0 1 2 3 5"。
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
[[ $n_gpus -eq 0 ]] && { echo "no valid GPU ids parsed from '${GPUS}'" >&2; exit 1; }

[[ -z "$LOG_DIR" ]] && LOG_DIR="${OUT_BASE}/_logs/exps_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 展开为 (cfg, run_idx) job 列表: 每个 config 重复 NUM_RUNS 次。
jobs=()
for cfg in "${configs[@]}"; do
  for ((r = 1; r <= NUM_RUNS; r++)); do
    jobs+=("${r}|${cfg}")
  done
done

total=${#jobs[@]}
echo "============================================================"
echo "run_exps: ${#configs[@]} configs × ${NUM_RUNS} runs = ${total} jobs"
echo "           across ${n_gpus} GPU(s): ${GPU_LIST[*]}"
echo "logs dir : ${LOG_DIR}"
echo "============================================================"

declare -A gpu_pid pid_info
for g in "${GPU_LIST[@]}"; do gpu_pid[$g]=""; done

trap 'echo; echo "[run_exps] interrupted — killing children"; \
      for g in "${GPU_LIST[@]}"; do \
        p="${gpu_pid[$g]:-}"; [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; \
      done; wait; exit 130' INT TERM

launch() {
  local gpu="$1" idx="$2" job="$3"
  local run_idx tag vl extra
  IFS='|' read -r run_idx tag vl extra <<< "$job"
  local out_dir="${OUT_BASE}/${tag}/run${run_idx}"
  local label="${tag}_r${run_idx}"
  local log="${LOG_DIR}/${idx}_gpu${gpu}_${label}.log"
  local seed="${SEEDS[run_idx - 1]}"

  local extra_array=()
  [[ -n "$extra" ]] && read -ra extra_array <<< "$extra"

  CUDA_VISIBLE_DEVICES="$gpu" python src/f5_tts/infer/tto.py \
    --ref-text "${REF_TEXT}" --gen-text "${GEN_TEXT}" \
    --loss-mode "${LOSS_MODE}" --vad-level "${vl}" \
    --opt-at "${OPT_AT}" --opt-steps "${OPT_STEPS}" --opt-lr "${OPT_LR}" \
    --window-size "${WINDOW_SIZE}" --hop-size "${HOP_SIZE}" \
    --batch-size "${BATCH_SIZE}" --ref-dir "${REF_DIR}" \
    --seed "${seed}" \
    --output "${out_dir}" \
    "${extra_array[@]}" \
    >"$log" 2>&1 &
  local pid=$!
  gpu_pid[$gpu]=$pid
  pid_info[$pid]="gpu=${gpu} idx=${idx}/${total} job=${label}"
  echo "[dispatch] $(date +%H:%M:%S)  gpu=${gpu} pid=${pid}  [$idx/$total] ${label} (vad=${vl} seed=${seed})  → ${log}"
}

# 阻塞轮询直到有 GPU 空闲, 返回它的 id。
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
  local tag="$1" run_idx="$2" gpu="$3"
  local out_dir="${OUT_BASE}/${tag}/run${run_idx}"
  local label="${tag}_r${run_idx}"
  shopt -s nullglob
  local wavs=("$out_dir"/*.wav)
  shopt -u nullglob
  if [[ ${#wavs[@]} -eq 0 ]]; then
    echo "[eval] SKIP ${out_dir} (no wav)"
    return 1
  fi
  echo "[eval] >>> ${label}"
  CUDA_VISIBLE_DEVICES="$gpu" python src/f5_tts/eval/batch_eval.py \
    --gen-dir "$out_dir" \
    --ref-dir "${REF_DIR}" \
    --gen-text "${GEN_TEXT}" \
    --out-csv "$out_dir/metrics.csv" \
    2>&1 | tee "${LOG_DIR}/eval_${label}.log" | tail -15
}

t0=$SECONDS
for i in "${!jobs[@]}"; do
  idx=$((i + 1))
  gpu=$(reap_free_gpu)
  launch "$gpu" "$idx" "${jobs[$i]}"
done

# Drain remaining jobs.
for g in "${GPU_LIST[@]}"; do
  pid="${gpu_pid[$g]:-}"
  [[ -z "$pid" ]] && continue
  wait "$pid" || true
  echo "[finish  ] $(date +%H:%M:%S)  ${pid_info[$pid]}  exit=$?"
done

echo
echo "All ${total} jobs finished in $((SECONDS - t0))s."
echo "Results     : ${OUT_BASE}/exp{1,2,3,4}/run{1..${NUM_RUNS}}/"
echo "Per-run logs: ${LOG_DIR}/"

if [[ $SKIP_EVAL -eq 1 ]]; then
  echo "[run_exps] --skip-eval set, stopping after generation"
  exit 0
fi

if [[ $DEFER_EVAL -eq 1 ]]; then
  echo
  echo "============================================================"
  echo "deferred batch eval (serial on gpu=${GPU_LIST[0]})"
  echo "============================================================"
  eval_t0=$SECONDS
  for job in "${jobs[@]}"; do
    IFS='|' read -r run_idx tag _ _ <<< "$job"
    run_eval "$tag" "$run_idx" "${GPU_LIST[0]}" || true
  done
  echo
  echo "deferred eval finished in $((SECONDS - eval_t0))s"
else
  echo
  echo "============================================================"
  echo "inline batch eval (serial on gpu=${GPU_LIST[0]})"
  echo "============================================================"
  for job in "${jobs[@]}"; do
    IFS='|' read -r run_idx tag _ _ <<< "$job"
    run_eval "$tag" "$run_idx" "${GPU_LIST[0]}" || true
  done
fi

echo "Summaries   : ${OUT_BASE}/*/run*/metrics.summary.txt"
