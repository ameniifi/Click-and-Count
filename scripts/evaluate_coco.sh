#!/bin/bash

#SBATCH --job-name=coco_eval
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.out
#SBATCH --error=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam
COUNTING_DIR=/home/<username>/projects/def-<account>/<username>/coco_counting
MODEL_ROOT=/home/<username>/projects/def-<account>/<username>/clickcount_models
VENV_PATH=/home/<username>/projects/def-<account>/<username>/venvs/clickcount-click-sam/bin/activate
BACKBONE_CKPT=/home/<username>/projects/def-<account>/<username>/weights/swav_800ep_pretrain.pth.tar
SAM_MASK_DIR=${COUNTING_DIR}/sam_masks_vit_b_512_pointbox
TEXT_CACHE_DIR=${COUNTING_DIR}/text_embeddings
TEXT_CACHE=${TEXT_CACHE_DIR}/clip_vitb16_coco.pt
MODEL_NAME=coco_catdisjoint

mkdir -p "${PROJECT_ROOT}/logs"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"
source "${PROJECT_ROOT}/coco_scratch.sh"
prepare_coco_scratch
DATA_ROOT="${COCO_DATA_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=50191
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

cd "${PROJECT_ROOT}"

srun python -u scripts/evaluate.py \
  --model_name "${MODEL_NAME}" \
  --dataset coco \
  --data_path "${DATA_ROOT}" \
  --coco_counting_dir "${COUNTING_DIR}" \
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
  --kernel_dim 1 \
  --num_objects 3 \
  --batch_size 1 \
  --dropout 0.1 \
  --num_workers 4 \
  --tiling_p 0.0 \
  --pre_norm \
  --prompt_type click_sam \
  --sam_mask_dir "${SAM_MASK_DIR}" \
  --text_embedding_path "${TEXT_CACHE}"

