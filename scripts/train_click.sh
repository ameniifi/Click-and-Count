#!/bin/bash

#SBATCH --job-name=clickcount_click_only_train
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=$HOME
#SBATCH --error=$HOME

set -euo pipefail

PROJECT_ROOT=$HOME
DATA_ROOT=$HOME
MODEL_ROOT=$HOME
VENV_PATH=$HOME
BACKBONE_CKPT=$HOME
SAM_MASK_DIR=${DATA_ROOT}/sam_masks_vit_b_512_pointbox

MODEL_NAME=lr_click_only_fsc

mkdir -p "${PROJECT_ROOT}/logs" "${MODEL_ROOT}"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=50190
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

cd "${PROJECT_ROOT}"

srun python -u scripts/train.py \
  --model_name "${MODEL_NAME}" \
  --dataset fsc147 \
  --data_path "${DATA_ROOT}" \
  --model_path "${MODEL_ROOT}" \
  --backbone resnet50 \
  --backbone_checkpoint "${BACKBONE_CKPT}" \
  --swav_backbone \
  --reduction 8 \
  --image_size 512 \
  --num_enc_layers 3 \
  --num_ope_iterative_steps 3 \
  --emb_dim 256 \
  --num_heads 8 \
  --kernel_dim 3 \
  --num_objects 3 \
  --epochs 200 \
  --lr 1e-4 \
  --backbone_lr 0 \
  --lr_drop 120 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --dropout 0.1 \
  --num_workers 8 \
  --max_grad_norm 0.1 \
  --aux_weight 0.3 \
  --tiling_p 0.5 \
  --prompt_type click_sam \
  --click_only \
  --sam_mask_dir "${SAM_MASK_DIR}"

