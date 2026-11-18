# Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting

This repository contains the training, evaluation, and data-preparation code for **Click and Count (C&C)**, a class-agnostic object-counting method that combines click-guided visual exemplars with optional text guidance. A click prompts the Segment Anything Model (SAM) to produce an object mask. The model uses the mask to build shape and appearance prototypes, refines them with image and text features, and predicts a density map whose sum is the estimated count.

The paper was accepted for publication in the **2027 IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2027)**. The release includes code; datasets, model checkpoints, and cached masks or text embeddings must be obtained or generated separately.

## Paper

Amani Aifi, Wassim Bouachir, Ahmed Zgaren, and Nizar Bouguila, “Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting,” accepted for publication in the *Proceedings of the 2027 IEEE Symposium Series on Computational Intelligence*.

```bibtex
@inproceedings{aifi2027clickandcount,
  title     = {Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting},
  author    = {Aifi, Amani and Bouachir, Wassim and Zgaren, Ahmed and Bouguila, Nizar},
  booktitle = {Proceedings of the 2027 IEEE Symposium Series on Computational Intelligence},
  year      = {2027}
}
```

## Method at a glance

1. A ResNet backbone extracts image features.
2. One click per exemplar prompts SAM to produce an object mask. By default, the supplied example scripts use precomputed masks; online SAM inference is also supported.
3. The mask provides shape and appearance cues for visual prototypes. Optional CLIP text embeddings provide category-level guidance during prototype refinement.
4. Prototype matching and a density-map head produce the predicted count.

The FSC-147 experiments use three exemplars per image. `--click_only` withholds the annotated boxes from the counting model; if boxes are available, they can still be used as SAM disambiguation prompts when masks are precomputed.

## Repository layout

- `models/`: backbone, prototype adaptation, text conditioning, and density-map head.
- `utils/`: FSC-147 and COCO data loading, SAM-mask and text-embedding preparation, and cache utilities.
- `scripts/train.py`, `scripts/evaluate.py`: distributed training and evaluation entrypoints.
- `scripts/*.sh`: Compute Canada SLURM examples. Replace the account and filesystem placeholders before use. The COCO examples also expect a site-specific `coco_scratch.sh`, which is not included.

## Environment

The supplied pins target **Python 3.11**, PyTorch 2.2.1, and torchvision 0.17.1. Training and evaluation require a CUDA-capable PyTorch build with NCCL. Install a PyTorch/torchvision build matching your CUDA setup first; for CUDA 11.8, for example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Use the corresponding PyTorch wheel index for a different CUDA version or for CPU-only preprocessing. For online SAM inference or mask precomputation, install Meta's SAM package and OpenCV:

```bash
python -m pip install opencv-python
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

To create text embeddings, also install `open_clip_torch`. To enable Weights & Biases logging, install `wandb`.

## Data and preprocessing

The FSC-147 data directory passed to `--data_path` should contain the dataset's annotation and split files, images, and 512 by 512 density maps:

```text
<data_root>/
  Train_Test_Val_FSC_147.json
  annotation_FSC147_384.json
  images_384_VarV2/
  gt_density_map_adaptive_512_512_object_VarV2/
```

Precompute click-prompted SAM masks (requires a SAM checkpoint):

```bash
python -m utils.precompute_sam_masks \
  --data_path <data_root> \
  --sam_checkpoint <path/to/sam_vit_b_01ec64.pth> \
  --sam_model_type vit_b \
  --image_size 512 \
  --num_objects 3 \
  --output_dir <data_root>/sam_masks_vit_b_512_pointbox
```

Optional text embeddings can be generated from a JSON file containing image names and category descriptions:

```bash
python -m utils.precompute_text_embeddings \
  --input_json <path/to/text_descriptions.json> \
  --output_path <data_root>/text_embeddings/clip_vitb16_fsc147d.pt \
  --model_name ViT-B-16 \
  --pretrained openai
```

The pretrained SwAV backbone, SAM checkpoint, FSC-147 data, and generated caches are not included. Pass local checkpoint and cache paths in the commands below. Create the checkpoint output directory before training.

## Training

This example launches two local GPU workers. `torchrun` supplies the distributed environment required by the training entrypoint.

```bash
mkdir -p <path/to/model_checkpoints>
export PYTHONPATH=.

torchrun --standalone --nnodes=1 --nproc_per_node=2 -m scripts.train \
  --model_name clickcount_fsc147 \
  --dataset fsc147 \
  --data_path <data_root> \
  --model_path <path/to/model_checkpoints> \
  --backbone resnet50 \
  --backbone_checkpoint <path/to/swav_800ep_pretrain.pth.tar> \
  --swav_backbone \
  --reduction 8 \
  --image_size 512 \
  --num_objects 3 \
  --prompt_type click_sam \
  --click_only \
  --sam_mask_dir <data_root>/sam_masks_vit_b_512_pointbox \
  --text_embedding_path <data_root>/text_embeddings/clip_vitb16_fsc147d.pt \
  --text_alignment_weight 0.1 \
  --text_alignment_warmup_epochs 30
```

Remove the text arguments to train without text guidance. To run with one GPU, set `--nproc_per_node=1`. The command uses the default model dimensions and training settings from `utils/arg_parser.py`; override them there if your experiment uses different settings.

## Evaluation

The checkpoint is loaded from `<model_path>/<model_name>.pt`. Use the same model and prompt settings used during training; include the text cache when evaluating a text-conditioned model.

```bash
export PYTHONPATH=.

torchrun --standalone --nnodes=1 --nproc_per_node=1 -m scripts.evaluate \
  --model_name clickcount_fsc147 \
  --data_path <data_root> \
  --model_path <path/to/model_checkpoints> \
  --backbone resnet50 \
  --backbone_checkpoint <path/to/swav_800ep_pretrain.pth.tar> \
  --swav_backbone \
  --reduction 8 \
  --image_size 512 \
  --num_objects 3 \
  --batch_size 1 \
  --tiling_p 0.0 \
  --prompt_type click_sam \
  --click_only \
  --sam_mask_dir <data_root>/sam_masks_vit_b_512_pointbox \
  --text_embedding_path <data_root>/text_embeddings/clip_vitb16_fsc147d.pt
```

The evaluator reports MAE, RMSE, and NAE on the validation and test splits. Evaluation uses the distributed launcher as well, including for a single GPU.

## COCO-derived benchmark

COCO-derived category-disjoint sample preparation, density-map and mask caching, and evaluation are implemented in `utils/coco_data.py` and the related `utils/prepare_coco_counting_v2.py`, `utils/precompute_coco_sam_masks_v2.py`, and `utils/verify_coco_counting_v2.py` utilities. The SLURM examples require site-specific paths and scratch-storage setup.

## Attribution and license

This implementation builds on the LOCA low-shot object-counting framework. The upstream project is available at [djukicn/loca](https://github.com/djukicn/loca); its MIT attribution is preserved in [LICENSE](LICENSE). The Click and Count code is also distributed under the MIT License. SAM, pretrained weights, datasets, and other external resources are separate and remain subject to their own terms.

