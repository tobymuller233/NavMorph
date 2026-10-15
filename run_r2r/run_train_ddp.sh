#!/usr/bin/env bash
# Multi-GPU R2R training with torchrun + DDP.
#
# Usage (from repo root):
#   NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash run_r2r/run_train_ddp.sh
#   NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash run_r2r/run_train_ddp.sh --exp_name my_ddp_run
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
EXP_NAME="${EXP_NAME:-release_r2r_ddp}"
CONFIG="${CONFIG:-run_r2r/iter_train_ddp.yaml}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${NPROC}" -gt "${#_GPUS[@]}" ]]; then
    echo "NPROC=${NPROC} exceeds visible GPUs (${#_GPUS[@]}) in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
  fi
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

mkdir -p data/logs/running_log
LOG_FILE="data/logs/running_log/train_ddp_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

source /home/lmy/miniconda3/etc/profile.d/conda.sh
conda activate morph

EXTRA_OPTS=()
if [[ $# -gt 0 ]]; then
  EXTRA_OPTS=("$@")
fi

GPU_LIST="["
for ((i=0; i<NPROC; i++)); do
  if [[ $i -gt 0 ]]; then GPU_LIST+=","; fi
  GPU_LIST+="$i"
done
GPU_LIST+="]"

echo "NPROC=${NPROC} MASTER_PORT=${MASTER_PORT} EXP_NAME=${EXP_NAME}"
echo "CONFIG=${CONFIG} GPU_LIST=${GPU_LIST} LOG=${LOG_FILE}"

stdbuf -oL -eL torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  run.py \
  --exp_name "${EXP_NAME}" \
  --run-type train \
  --exp-config "${CONFIG}" \
  GPU_NUMBERS "${NPROC}" \
  TORCH_GPU_IDS "${GPU_LIST}" \
  SIMULATOR_GPU_IDS "${GPU_LIST}" \
  "${EXTRA_OPTS[@]}" \
  2>&1 | tee "${LOG_FILE}"
