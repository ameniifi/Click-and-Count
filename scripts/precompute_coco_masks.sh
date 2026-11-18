#!/bin/bash

#SBATCH --job-name=coco_masks
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.out
#SBATCH --error=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/<username>/projects/def-<account>/<username>/ClickCount-click-sam
COUNTING_DIR=/home/<username>/projects/def-<account>/<username>/coco_counting_v2
VENV_PATH=/home/<username>/projects/def-<account>/<username>/venvs/clickcount-click-sam/bin/activate
SAM_CKPT=/home/<username>/projects/def-<account>/<username>/weights/sam_vit_b_01ec64.pth
SAM_MASK_DIR=${COUNTING_DIR}/sam_masks_vit_b_512_pointbox

mkdir -p "${PROJECT_ROOT}/logs" "${SAM_MASK_DIR}"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"
source "${PROJECT_ROOT}/coco_scratch.sh"
prepare_coco_scratch
DATA_ROOT="${COCO_DATA_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${PROJECT_ROOT}"

python -u utils/precompute_coco_sam_masks_v2.py \
  --data_path "${DATA_ROOT}" \
  --counting_dir "${COUNTING_DIR}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --sam_model_type vit_b \
  --image_size 512 \
  --num_objects 3 \
  --splits train val test \
  --output_dir "${SAM_MASK_DIR}" \
  --device cuda