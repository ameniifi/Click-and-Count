# Click and Count: Multimodal Prototype Adaptation for Class-Agnostic Object Counting

> **Anonymous submission — code and materials released for review purposes only.**
> Distribution, citation, or public sharing of this repository or the associated manuscript is strictly prohibited until the review process concludes. Author identities and affiliations have been withheld in accordance with the double-blind review policy.
---

## Installation

```bash
git clone <anonymous-repo-url>
cd click-and-count
conda create -n cnc python=3.10 -y
conda activate cnc
pip install -r requirements.txt
```



---

## Datasets

### FSC-147

```bash
python scripts/precompute_masks.py --dataset fsc147 --data-root /path/to/FSC147
```

### COCO-derived benchmark

```bash
python datasets/coco_derived.py --coco-root /path/to/coco --output-dir /path/to/output
```

---

## Usage

### Training

```bash
python scripts/train.py \
  --config configs/fsc147_point_text.yaml \
  --data-root /path/to/FSC147 \
  --batch-size 4 \
  --lr 1e-4 \
  --epochs 200
```

### Evaluation

```bash
python scripts/evaluate.py \
  --config configs/fsc147_point_text.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --split test
```

Supported prompt modes: `point`, `text`, `point+text` (set via the config or `--prompt-mode` flag).



---

## Citation

Citation details will be added upon acceptance and de-anonymization. Please **do not cite this repository or the manuscript** in its current anonymous form.

---

## License

To be determined upon publication. All rights reserved by the (currently anonymous) authors until then.