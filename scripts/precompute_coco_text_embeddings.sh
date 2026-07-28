#!/bin/bash

#SBATCH --job-name=clickcount_coco_text_embeddings_v2
#SBATCH --account=def-<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=$HOME
#SBATCH --error=$HOME

set -euo pipefail

PROJECT_ROOT=$HOME
COUNTING_DIR=$HOME
TEXT_JSON=${COUNTING_DIR}/text_prompts_coco.json
TEXT_CACHE_DIR=${COUNTING_DIR}/text_embeddings
TEXT_CACHE=${TEXT_CACHE_DIR}/clip_vitb16_coco.pt
VENV_PATH=$HOME

mkdir -p "${PROJECT_ROOT}/logs" "${TEXT_CACHE_DIR}"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda cudnn opencv/4.10.0

source "$VENV_PATH"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${PROJECT_ROOT}"

python -u utils/precompute_text_embeddings.py \
  --input_json "${TEXT_JSON}" \
  --output_path "${TEXT_CACHE}" \
  --model_name ViT-B-16 \
  --pretrained openai \
  --batch_size 64