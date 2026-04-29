#!/usr/bin/env bash
set -euo pipefail

# Each row: window_size|hop_size|opt_at|opt_steps|opt_lr|loss_mode|vad_level|slide_mode
#   loss_mode  : value | embedding
#   vad_level  : frame | utter | both
#   slide_mode : hidden (default, 1× backbone forward + slide on hidden)
#                audio  (legacy: per-window wav2vec2, slower & OOD)
# Each row runs tto generation + batch eval into a TAG-named folder, so
# different combinations never clobber. The TAG suffix _sm<mode> distinguishes
# slide_mode runs explicitly.
configs=(
  # --- A/B comparison: hidden vs audio at a known-good config ---
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|250|1e-3|value|frame|hidden"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|250|1e-3|value|frame|audio"

  # --- Phase 1 (when ready): loss-mode × vad-level under hidden slide ---
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|frame|hidden"
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|utter|hidden"
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|both|hidden"
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|frame|hidden"
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|utter|hidden"
  # "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|both|hidden"
)

GPUS="0"
LOG_DIR=""
DEFER_EVAL=0

usage() {
  cat <<EOF
Usage: $0 [--gpus LIST] [--defer-eval]
  --gpus LIST    GPU ids to dispatch on. Comma list or range: "0,1,3" / "0-3"
                 (default: "${GPUS}"). Each GPU runs one config at a time;
                 jobs are queued and a GPU picks the next config as it frees up.
  --log-dir DIR  Where to store per-run stdout logs
                 (default: tto_outputs/_logs/<timestamp>/)
  --defer-eval   Skip per-run batch eval during generation (passes --skip-eval
                 to run_tto.sh) and run all evals serially AFTER the sweep
                 finishes, on the first GPU in --gpus. Use this when each
                 run's VRAM peak is close to capacity — avoids loading the
                 ~5 GB eval stack on top of the TTO sampler.
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2";    shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --defer-eval)  DEFER_EVAL=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"

# Expand "0-3,5" → "0 1 2 3 5" (space-separated id list)
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

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="tto_outputs/_logs/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

total=${#configs[@]}
echo "============================================================"
echo "sweep_tto: ${total} configs across ${n_gpus} GPU(s): ${GPU_LIST[*]}"
echo "logs dir : ${LOG_DIR}"
echo "============================================================"

# Track which pid is on which GPU so we can reclaim the slot when it exits.
declare -A gpu_pid    # gpu_id → pid (or empty)
declare -A pid_info   # pid → "gpu=X idx=Y tag=..."
for g in "${GPU_LIST[@]}"; do gpu_pid[$g]=""; done

# Kill all children if the sweep is interrupted (Ctrl-C, SIGTERM).
trap 'echo; echo "[sweep] interrupted — killing children"; \
      for g in "${GPU_LIST[@]}"; do \
        p="${gpu_pid[$g]:-}"; [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; \
      done; wait; exit 130' INT TERM

cfg_to_tag() {
  local cfg="$1" ws hs oa os lr lm vl sm
  IFS='|' read -r ws hs oa os lr lm vl sm <<< "$cfg"
  sm="${sm:-hidden}"   # backwards compat for older 7-field rows
  local oa_slug="${oa//,/-}"
  echo "${lm}-${vl}_w${ws}_h${hs}_at${oa_slug}_s${os}_lr${lr}_sm${sm}"
}

launch() {
  local gpu="$1" idx="$2" cfg="$3"
  local ws hs oa os lr lm vl sm
  IFS='|' read -r ws hs oa os lr lm vl sm <<< "$cfg"
  sm="${sm:-hidden}"
  local tag; tag=$(cfg_to_tag "$cfg")
  local log="${LOG_DIR}/${idx}_gpu${gpu}_${tag}.log"
  local extra_args=()
  [[ $DEFER_EVAL -eq 1 ]] && extra_args+=(--skip-eval)

  CUDA_VISIBLE_DEVICES="$gpu" ./run_tto.sh \
    "${extra_args[@]}" \
    --loss-mode "$lm" --vad-level "$vl" --vad-slide-mode "$sm" \
    --window-size "$ws" --hop-size "$hs" \
    --opt-at "$oa" --opt-steps "$os" --opt-lr "$lr" \
    >"$log" 2>&1 &
  local pid=$!
  gpu_pid[$gpu]=$pid
  pid_info[$pid]="gpu=${gpu} idx=${idx}/${total} tag=${tag}"
  echo "[dispatch] $(date +%H:%M:%S)  gpu=${gpu} pid=${pid}  [$idx/$total] ${lm}-${vl}-${sm} ws=${ws} hs=${hs}  → ${log}"
}

# Find a GPU whose previous job has exited; block (poll) until one is free.
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
echo "Results      : tto_outputs/<loss>-<vad>_w<ws>_h<hs>_.../"
echo "Per-run logs : ${LOG_DIR}/"

if [[ $DEFER_EVAL -eq 1 ]]; then
  # Pull gen_text / ref_dir out of run_tto.sh so deferred eval uses the exact
  # same values the generation step did. Brittle to literal edits but avoids
  # duplicating strings across scripts.
  GEN_TEXT=$(sed -n 's/^GEN_TEXT="\(.*\)"$/\1/p' run_tto.sh | head -1)
  REF_DIR=$(sed -n 's/^REF_DIR=\(.*\)$/\1/p' run_tto.sh | head -1 \
            | sed 's/^"//; s/"$//')
  eval_gpu="${GPU_LIST[0]}"

  echo
  echo "============================================================"
  echo "deferred batch eval (serial on gpu=${eval_gpu})"
  echo "  ref_dir : ${REF_DIR}"
  echo "  gen_text: ${GEN_TEXT}"
  echo "============================================================"
  eval_t0=$SECONDS
  eval_fail=0
  for cfg in "${configs[@]}"; do
    tag=$(cfg_to_tag "$cfg")
    out_dir="tto_outputs/${tag}"
    if [[ ! -d "$out_dir" ]]; then
      echo "[eval] SKIP missing ${out_dir}"; eval_fail=$((eval_fail+1)); continue
    fi
    # Skip if no wav was produced (tto.py must have failed).
    shopt -s nullglob
    wavs=("$out_dir"/*.wav)
    shopt -u nullglob
    if [[ ${#wavs[@]} -eq 0 ]]; then
      echo "[eval] SKIP ${out_dir} (no wav)"; eval_fail=$((eval_fail+1)); continue
    fi
    echo "[eval] >>> ${tag}"
    CUDA_VISIBLE_DEVICES="$eval_gpu" python src/f5_tts/infer/batch_eval.py \
      --gen-dir "$out_dir" \
      --ref-dir "$REF_DIR" \
      --gen-text "$GEN_TEXT" \
      --out-csv "$out_dir/metrics.csv" \
      2>&1 | tee "${LOG_DIR}/eval_${tag}.log" | tail -15
  done
  echo
  echo "deferred eval finished in $((SECONDS - eval_t0))s"
  [[ $eval_fail -gt 0 ]] && echo "  (skipped ${eval_fail} run(s) with no outputs)"
fi

echo "Summaries    : tto_outputs/*/metrics.summary.txt"
