#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

TARGET_VAL="${TARGET_VAL:-0.82}"
TARGET_EXT="${TARGET_EXT:-0.80}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/experiment/outputs/training/optimization/auto_continue_plan29_30.log}"
PLAN28_SUMMARY="$PROJECT_ROOT/experiment/outputs/training/blends/plan28_timeforward_proto_sparsemeta_v1/summary.json"
PLAN29_SUMMARY="$PROJECT_ROOT/experiment/outputs/training/models/m5_temporal_graphsage/plan29_m5_klabel_consisrecent_proto4_bganchor_warm4_stable_seed42_v1/summary.json"
PLAN30_SUMMARY="$PROJECT_ROOT/experiment/outputs/training/models/m5_temporal_graphsage/plan30_m5_klabel_consisrecent_ctxresid_proto4_bganchor_warm4_stable_seed42_v1/summary.json"

mkdir -p "$(dirname "$LOG_PATH")"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_PATH"
}

summary_meets_target() {
  local summary_path="$1"
  python3 - "$summary_path" "$TARGET_VAL" "$TARGET_EXT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_val = float(sys.argv[2])
target_ext = float(sys.argv[3])
if not path.exists():
    sys.exit(1)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
val = data.get("phase1_val_auc")
if val is None:
    val = data.get("phase1_val_auc_mean")
ext = data.get("phase2_external_auc")
if ext is None:
    ext = data.get("phase2_external_auc_mean")
if val is None or ext is None:
    sys.exit(1)
print(f"val={float(val):.6f} ext={float(ext):.6f}")
sys.exit(0 if float(val) >= target_val and float(ext) >= target_ext else 1)
PY
}

wait_for_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    log "waiting for pid=$pid to finish"
    sleep 60
  done
}

run_cmd() {
  log "launch: $*"
  "$@" 2>&1 | tee -a "$LOG_PATH"
}

run_plan29() {
  run_cmd conda run -n Graph --no-capture-output python3 -u experiment/training/run_training.py train \
    --model m5_temporal_graphsage \
    --run-name plan29_m5_klabel_consisrecent_proto4_bganchor_warm4_stable_seed42_v1 \
    --device cuda \
    --seeds 42 \
    --epochs 24 \
    --batch-size 512 \
    --hidden-dim 128 \
    --rel-dim 32 \
    --fanouts 15 10 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --dropout 0.2 \
    --feature-norm hybrid \
    --extra-groups temporal_snapshot temporal_recent temporal_relation_recent temporal_bucket_norm \
    --norm layer \
    --residual \
    --ffn \
    --jk sum \
    --edge-encoder gated \
    --subgraph-head meanmax \
    --grad-clip 1.0 \
    --scheduler plateau \
    --early-stop-patience 10 \
    --time-decay-strength 4.0 \
    --known-label-feature \
    --neighbor-sampler consistency_recent \
    --consistency-temperature 0.35 \
    --include-historical-background-negatives \
    --historical-background-negative-ratio 0.25 \
    --historical-background-negative-warmup-epochs 4 \
    --historical-background-aux-only \
    --prototype-multiclass-num-classes 4 \
    --prototype-loss-weight 0.05 \
    --prototype-temperature 0.15 \
    --prototype-momentum 0.9
}

run_plan30() {
  run_cmd conda run -n Graph --no-capture-output python3 -u experiment/training/run_training.py train \
    --model m5_temporal_graphsage \
    --run-name plan30_m5_klabel_consisrecent_ctxresid_proto4_bganchor_warm4_stable_seed42_v1 \
    --device cuda \
    --seeds 42 \
    --epochs 24 \
    --batch-size 512 \
    --hidden-dim 128 \
    --rel-dim 32 \
    --fanouts 15 10 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --dropout 0.2 \
    --feature-norm hybrid \
    --extra-groups temporal_snapshot temporal_recent temporal_relation_recent temporal_bucket_norm \
    --target-context-extra-groups graph_stats neighbor_similarity activation_early \
    --target-context-fusion logit_residual \
    --norm layer \
    --residual \
    --ffn \
    --jk sum \
    --edge-encoder gated \
    --subgraph-head meanmax \
    --grad-clip 1.0 \
    --scheduler plateau \
    --early-stop-patience 10 \
    --time-decay-strength 4.0 \
    --known-label-feature \
    --neighbor-sampler consistency_recent \
    --consistency-temperature 0.35 \
    --include-historical-background-negatives \
    --historical-background-negative-ratio 0.25 \
    --historical-background-negative-warmup-epochs 4 \
    --historical-background-aux-only \
    --prototype-multiclass-num-classes 4 \
    --prototype-loss-weight 0.05 \
    --prototype-temperature 0.15 \
    --prototype-momentum 0.9
}

WAIT_PID=""
if [[ "${1:-}" == "--wait-pid" ]]; then
  WAIT_PID="${2:-}"
fi

log "auto-continue watcher start"
wait_for_pid "$WAIT_PID"
log "plan28 wait completed"

if summary_meets_target "$PLAN28_SUMMARY" 2>&1 | tee -a "$LOG_PATH"; then
  log "plan28 already met target; stop"
  exit 0
fi

run_plan29
if summary_meets_target "$PLAN29_SUMMARY" 2>&1 | tee -a "$LOG_PATH"; then
  log "plan29 met target; stop"
  exit 0
fi

run_plan30
summary_meets_target "$PLAN30_SUMMARY" 2>&1 | tee -a "$LOG_PATH" || true
log "auto-continue watcher end"
