#!/usr/bin/env bash
# NavMorph R2R eval (your settings):
#   - val_unseen: ALL episodes -> metrics JSON
#   - GPU: physical card 1 (CUDA_VISIBLE_DEVICES=1, sim/torch use cuda:0)
#   - mp4: first 10 episodes only (separate phase)
#   - checkpoint: data/checkpoints/ckpt.pth
#
# Usage (from repo root):
#   bash run_r2r/run_eval_r2r.sh
#
# Watch progress in another terminal:
#   tail -f log/eval_r2r_latest.log
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p log
LOG_FILE="log/eval_r2r_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$LOG_FILE")" log/eval_r2r_latest.log

# Physical GPU 1 -> only visible device is cuda:0 inside the process
export CUDA_VISIBLE_DEVICES=1
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
# Line-buffered Python stdout/stderr for live tee
export PYTHONUNBUFFERED=1

source /home/lmy/miniconda3/etc/profile.d/conda.sh
conda activate morph

CKPT="data/checkpoints/ckpt.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT"
  exit 1
fi

run_phase() {
  local phase_name="$1"
  local exp_name="$2"
  local config="$3"
  echo ""
  echo "========== ${phase_name} =========="
  echo "exp_name=${exp_name} config=${config}"
  echo "log=${LOG_FILE}"
  stdbuf -oL -eL python run.py \
    --exp_name "${exp_name}" \
    --run-type eval \
    --exp-config "${config}"
}

{
  echo "NavMorph R2R eval started at $(date -Iseconds)"
  echo "Repo: ${REPO_ROOT}"
  echo "Checkpoint: ${CKPT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  run_phase \
    "Phase 1/2: full val_unseen metrics (no video)" \
    "r2r_full" \
    "run_r2r/eval_full_val_unseen.yaml"

  run_phase \
    "Phase 2/2: first 10 episodes + mp4" \
    "r2r_video10" \
    "run_r2r/eval_video_10.yaml"

  echo ""
  echo "Done at $(date -Iseconds)"
  echo "Metrics (aggregate): log/eval/r2r_full/stats_ckpt_0_val_unseen.json"
  echo "Per-episode metrics: log/eval/r2r_full/stats_ep_ckpt_0_val_unseen_r0_w1.json"
  echo "Videos (mp4): log/videos/r2r_video10/"
  echo "Training log mirror: data/logs/running_log/"
} 2>&1 | tee -a "$LOG_FILE"
