#!/usr/bin/env bash
set -euo pipefail

# Each row: window_size|hop_size|opt_at|opt_steps|opt_lr|loss_mode|vad_level
#   loss_mode: value | embedding
#   vad_level: frame | utter | both
# Each row runs tto generation + batch eval into a TAG-named folder, so
# different (loss_mode, vad_level, hop/window) combinations never clobber.
configs=(
  # --- Phase 1: loss-mode × vad-level (single window/hop) ---
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|frame"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|utter"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|both"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|frame"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|utter"
  "1.0|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|embedding|both"

  # --- (optional) window/hop sweep at the Phase-1 winner ---
  # "1.5|1.0|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|frame"
  # "0.5|0.25|2,4,6,8,10,12,14,16,18,20,24,28|50|1e-2|value|frame"
)

cd "$(dirname "$0")"

total=${#configs[@]}
for i in "${!configs[@]}"; do
  IFS='|' read -r ws hs oa os lr lm vl <<< "${configs[$i]}"
  idx=$((i + 1))
  echo
  echo "============================================================"
  echo "[$idx/$total] loss=${lm} vad=${vl}  window=${ws} hop=${hs}"
  echo "            opt_at=${oa} steps=${os} lr=${lr}"
  echo "============================================================"
  t0=$SECONDS
  ./run_tto.sh \
    --loss-mode "$lm" --vad-level "$vl" \
    --window-size "$ws" --hop-size "$hs" \
    --opt-at "$oa" --opt-steps "$os" --opt-lr "$lr"
  echo "[$idx/$total] done in $((SECONDS - t0))s"
done

echo
echo "All $total runs finished."
echo "Results under: tto_outputs/<loss>-<vad>_w<ws>_h<hs>_..."
echo "Summaries    : each <run>/metrics.summary.txt"
