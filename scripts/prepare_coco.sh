#!/bin/bash

#SBATCH --job-name=clickcount_coco_prepare_v2
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=$HOME
#SBATCH --error=$HOME

set -euo pipefail

PROJECT_ROOT=$HOME
COUNTING_DIR=$HOME
VENV_PATH=$HOME

mkdir -p "${PROJECT_ROOT}/logs" "${COUNTING_DIR}"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"
source "${PROJECT_ROOT}/coco_scratch.sh"
prepare_coco_scratch
DATA_ROOT="${COCO_DATA_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${PROJECT_ROOT}"

python -u utils/prepare_coco_counting_v2.py \
  --data_path "${DATA_ROOT}" \
  --output_dir "${COUNTING_DIR}" \
  --num_objects 3 \
  --min_area 64 \
  --min_box_size 4 \
  --split_seed 0 \
  --heldout_category_frac 0.2