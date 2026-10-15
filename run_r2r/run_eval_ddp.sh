#!/usr/bin/env bash
# Multi-GPU val_unseen eval: episodes sharded by rank, metrics aggregated on rank 0.
#
# Usage (from repo root):
#   NPROC=2 CUDA_VISIBLE_DEVICES=0,1 CKPT=data/checkpoints/ckpt.pth bash run_r2r/run_eval_ddp.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29502}"
EXP_NAME="${EXP_NAME:-r2r_eval_ddp}"
CKPT="${CKPT:-data/checkpoints/ckpt.pth}"
CONFIG="${CONFIG:-run_r2r/eval_full_val_unseen.yaml}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export PYTHONUNBUFFERED=1

mkdir -p log data/logs/running_log
LOG_FILE="log/eval_ddp_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "${LOG_FILE}")" log/eval_ddp_latest.log

source /home/lmy/miniconda3/etc/profile.d/conda.sh
conda activate morph

# Build TORCH_GPU_IDS and SIMULATOR_GPU_IDS lists [0..NPROC-1]
GPU_LIST="["
for ((i=0; i<NPROC; i++)); do
  if [[ $i -gt 0 ]]; then GPU_LIST+=","; fi
  GPU_LIST+="$i"
done
GPU_LIST+="]"

echo "NPROC=${NPROC} CKPT=${CKPT} GPU_LIST=${GPU_LIST}"

stdbuf -oL -eL torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  run.py \
  --exp_name "${EXP_NAME}" \
  --run-type eval \
  --exp-config "${CONFIG}" \
  GPU_NUMBERS "${NPROC}" \
  TORCH_GPU_IDS "${GPU_LIST}" \
  SIMULATOR_GPU_IDS "${GPU_LIST}" \
  EVAL.CKPT_PATH_DIR "${CKPT}" \
  2>&1 | tee "${LOG_FILE}"
