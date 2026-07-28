#!/bin/bash

#SBATCH --job-name=masks
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/project/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.out
#SBATCH --error=/project/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/project/def-<account>/<username>/ClickCount-click-sam
DATA_ROOT=/project/def-<account>/<username>/MultimodalCounting-SAMVersion/MultimodalCounting-SAMVersion/data/FSC147
SAM_CKPT=/project/def-<account>/<username>/weights/sam_vit_b_01ec64.pth
VENV_PATH=/project/def-<account>/<username>/venvs/clickcount-click-sam/bin/activate
SAM_MASK_DIR=${DATA_ROOT}/sam_masks_vit_b_512_pointbox

mkdir -p "${PROJECT_ROOT}/logs" "${SAM_MASK_DIR}"

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"

pip install --quiet typing_extensions numpy scipy

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${PROJECT_ROOT}"

python -u utils/precompute_sam_masks.py \
  --data_path "${DATA_ROOT}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --sam_model_type vit_b \
  --image_size 512 \
  --num_objects 3 \
  --output_dir "${SAM_MASK_DIR}"