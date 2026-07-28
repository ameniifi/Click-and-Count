# ClickCount: Click-Prompted Object Counting with SAM Masks

This repository contains the training and evaluation code for **ClickCount**, a click-prompted object counting model that leverages SAM (Segment Anything Model) masks as spatial priors, with optional text-alignment supervision. The code is built on top of a low-shot object counting framework and extends it with click/SAM-based prompting and text-embedding alignment losses.

> **Note:** This is an anonymized version of the codebase prepared for double-blind review. All paths, usernames, and cluster-specific identifiers have been removed or replaced with placeholders.

## Overview

The model takes an image and a set of user clicks (converted into SAM masks) as exemplar prompts and predicts a density map whose sum estimates the object count. An optional text-alignment auxiliary loss encourages visual exemplar embeddings to align with CLIP text embeddings of the object category, improving generalization.

                # SLURM job logs (created at runtime)
```

## Dataset

Experiments use **FSC-147**, a standard low-shot counting benchmark. Preprocessing produces:

- **SAM masks**: precomputed point/box-prompted SAM masks per image, stored under `<data_root>/sam_masks_vit_b_512_pointbox/`
- **Text embeddings**: CLIP ViT-B/16 text embeddings for FSC-147 category names, cached at `<data_root>/text_embeddings/clip_vitb16_fsc147d.pt`

## Environment

- Python 3.10
- CUDA + cuDNN
- OpenCV 4.10
- PyTorch (with NCCL backend for multi-GPU training)

Dependencies are installed into a virtual environment; see `requirements.txt` for exact package versions.

## Usage

### Evaluation

```bash
python -u evaluate.py \
  --model_name pure_model_no_text \
  --data_path <path/to/FSC147> \
  --model_path <path/to/model_checkpoints> \
  --backbone resnet50 \
  --backbone_checkpoint <path/to/swav_800ep_pretrain.pth.tar> \
  --swav_backbone \
  --reduction 8 \
  --image_size 512 \
  --num_enc_layers 3 \
  --num_ope_iterative_steps 3 \
  --emb_dim 256 \
  --num_heads 8 \
  --kernel_dim 3 \
  --num_objects 3 \
  --batch_size 1 \
  --dropout 0.1 \
  --num_workers 8 \
  --tiling_p 0.0 \
  --prompt_type click_sam \
  --click_only \
  --sam_mask_dir <path/to/sam_masks_vit_b_512_pointbox> \
  --text_embedding_path <path/to/clip_vitb16_fsc147d.pt>
```

### Training

Multi-GPU training (2 GPUs, single node) with click-based SAM prompting and text-alignment supervision, initialized from a pretrained checkpoint:

```bash
python -u train.py \
  --model_name lr_click_text_fsc \
  --dataset fsc147 \
  --data_path <path/to/FSC147> \
  --model_path <path/to/model_checkpoints> \
  --backbone resnet50 \
  --backbone_checkpoint <path/to/swav_800ep_pretrain.pth.tar> \
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
  --sam_mask_dir <path/to/sam_masks_vit_b_512_pointbox> \
  --text_embedding_path <path/to/clip_vitb16_fsc147d.pt> \
  --text_alignment_weight 0.1 \
  --text_alignment_warmup_epochs 30 \
  --init_checkpoint <path/to/lr_drop_fsc.pt>
```

For multi-GPU / multi-node runs, launch with `torchrun` or your cluster's distributed launcher (e.g., `srun` with `ntasks-per-node` matching the number of GPUs), setting `MASTER_ADDR`/`MASTER_PORT` appropriately.

### Key arguments

| Argument | Description |
|---|---|
| `--prompt_type click_sam` | Use SAM masks derived from click prompts as exemplars |
| `--click_only` | Restrict prompting to click-derived exemplars (no box/scribble prompts) |
| `--sam_mask_dir` | Directory of precomputed SAM masks |
| `--text_embedding_path` | Path to cached CLIP text embeddings |
| `--text_alignment_weight` | Weight of the text-alignment auxiliary loss |
| `--text_alignment_warmup_epochs` | Number of epochs before the text-alignment loss is applied |
| `--tiling_p` | Probability of applying tiling-based data augmentation |
| `--aux_weight` | Weight of auxiliary supervision losses |
| `--init_checkpoint` | Checkpoint used to initialize training (e.g., after a warmup/pretraining stage) |

## Reproducibility

- All experiments were run with fixed random seeds (see `train.py` for seeding logic).
- Backbone weights were initialized from a publicly available SwAV 800-epoch pretrained checkpoint.
- Model and data paths are cluster-specific in the original SLURM scripts; this README replaces them with generic placeholders for anonymity.

## License

To be added upon de-anonymization.