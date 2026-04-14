#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

TARGET_XINYE_VAL="${TARGET_XINYE_VAL:-0.7892}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/experiment/outputs/training/optimization/semantic_auc_push.log}"
CONDA_BIN="${CONDA_BIN:-/home/moonxkj/miniconda3/bin/conda}"

mkdir -p "$(dirname "$LOG_PATH")"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_PATH"
}

summary_path_for_dataset() {
  local dataset="$1"
  local run_name="$2"
  case "$dataset" in
    xinye_dgraph)
      printf '%s\n' "$PROJECT_ROOT/experiment/outputs/training/models/m5_temporal_graphsage/$run_name/summary.json"
      ;;
    *)
      printf '%s\n' "$PROJECT_ROOT/experiment/outputs/$dataset/training/models/m5_temporal_graphsage/$run_name/summary.json"
      ;;
  esac
}

extract_phase1_val_auc() {
  local summary_path="$1"
  python3 - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
value = data.get("phase1_val_auc")
if value is None:
    value = data.get("phase1_val_auc_mean")
if value is None:
    raise SystemExit(1)
print(f"{float(value):.6f}")
PY
}

run_recipe_train() {
  local dataset="$1"
  local recipe="$2"
  local run_name="$3"
  log "launch dataset=$dataset recipe=$recipe run_name=$run_name"
  "$CONDA_BIN" run -n Graph --no-capture-output python3 -u experiment/training/run_recipe.py train \
    --dataset "$dataset" \
    --recipe "$recipe" \
    --run-name "$run_name" 2>&1 | tee -a "$LOG_PATH"
}

maybe_backtest_public() {
  local recipe="$1"
  local xinye_prefix="$2"
  log "xinye winner confirmed recipe=$recipe; start public backtests"

  run_recipe_train elliptic_transactions "$recipe" "${xinye_prefix/_xinye_/_elliptic_}"
  local elliptic_summary
  elliptic_summary=$(summary_path_for_dataset elliptic_transactions "${xinye_prefix/_xinye_/_elliptic_}")
  if elliptic_val=$(extract_phase1_val_auc "$elliptic_summary" 2>/dev/null); then
    log "elliptic finished val_auc=$elliptic_val summary=$elliptic_summary"
  else
    log "elliptic summary missing or unreadable path=$elliptic_summary"
  fi

  run_recipe_train ellipticpp_transactions "$recipe" "${xinye_prefix/_xinye_/_ellipticpp_}"
  local ellipticpp_summary
  ellipticpp_summary=$(summary_path_for_dataset ellipticpp_transactions "${xinye_prefix/_xinye_/_ellipticpp_}")
  if ellipticpp_val=$(extract_phase1_val_auc "$ellipticpp_summary" 2>/dev/null); then
    log "ellipticpp finished val_auc=$ellipticpp_val summary=$ellipticpp_summary"
  else
    log "ellipticpp summary missing or unreadable path=$ellipticpp_summary"
  fi
}

declare -a RECIPE_QUEUE=(
  "prototype_bucketadv_dpdisc_ctxadaptive_semantic_drift_timeadapt15_softproto_curriculum_m5|plan59a_m5_semantic_drift_timeadapt15_softproto_xinye_v1"
  "prototype_bucketadv_dpdisc_ctxadaptive_semantic_drift_timeadapt20_softproto_curriculum_m5|plan59b_m5_semantic_drift_timeadapt20_softproto_xinye_v1"
  "prototype_bucketadv_dpdisc_ctxadaptive_semantic_drift_timeadapt15_softproto_lateproto_curriculum_m5|plan59c_m5_semantic_drift_timeadapt15_softproto_lateproto_xinye_v1"
)

log "semantic auc push start target_xinye_val=$TARGET_XINYE_VAL"

winner_recipe=""
winner_run_name=""

for entry in "${RECIPE_QUEUE[@]}"; do
  recipe="${entry%%|*}"
  run_name="${entry##*|}"
  run_recipe_train xinye_dgraph "$recipe" "$run_name"
  summary_path=$(summary_path_for_dataset xinye_dgraph "$run_name")
  if val_auc=$(extract_phase1_val_auc "$summary_path" 2>/dev/null); then
    log "xinye finished recipe=$recipe run_name=$run_name val_auc=$val_auc summary=$summary_path"
    if python3 - "$val_auc" "$TARGET_XINYE_VAL" <<'PY'
import sys
val = float(sys.argv[1])
target = float(sys.argv[2])
sys.exit(0 if val >= target else 1)
PY
    then
      winner_recipe="$recipe"
      winner_run_name="$run_name"
      log "xinye target reached recipe=$winner_recipe run_name=$winner_run_name"
      break
    fi
  else
    log "xinye summary missing or unreadable recipe=$recipe run_name=$run_name summary=$summary_path"
  fi
done

if [[ -n "$winner_recipe" ]]; then
  maybe_backtest_public "$winner_recipe" "$winner_run_name"
else
  log "no xinye run reached target after queue completion"
fi

log "semantic auc push end"
