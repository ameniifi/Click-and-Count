import json
import os
import random
from collections import defaultdict

from PIL import Image
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

from utils.data import extract_exemplar_clicks, tiling_augmentation
from utils.text_embeddings import (
    annotate_text_embedding_bundle,
    load_text_embedding_cache,
    split_text_embedding_bundle,
)


def coco_counting_dir(data_path, counting_dir=''):
    return counting_dir or os.path.join(data_path, 'clickcount_coco_counting_v2')


def coco_image_path(data_path, coco_split, file_name):
    candidates = [
        os.path.join(data_path, 'images', f'{coco_split}2017', file_name),
        os.path.join(data_path, 'images', file_name),
        os.path.join(data_path, f'{coco_split}2017', file_name),
        os.path.join(data_path, file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _clip_xyxy_from_xywh(bbox, width, height):
    x, y, w, h = bbox
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), x1 + max(0.0, float(w)))
    y2 = min(float(height), y1 + max(0.0, float(h)))
    return [x1, y1, x2, y2]


def _scale_boxes(boxes, width, height, image_size):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scale = torch.tensor(
        [image_size / width, image_size / height, image_size / width, image_size / height],
        dtype=torch.float32,
    )
    return boxes * scale


def _make_density_map(points_xy, exemplar_boxes, image_size):
    dmap = torch.zeros(image_size, image_size, dtype=torch.float32)
    points = torch.tensor(points_xy, dtype=torch.float32)
    if points.numel() == 0:
        return dmap.unsqueeze(0)

    points = points.round().long()
    points[:, 0] = points[:, 0].clip(0, image_size - 1)
    points[:, 1] = points[:, 1].clip(0, image_size - 1)
    for point in points:
        dmap[point[1], point[0]] += 1

    wh = torch.stack([
        exemplar_boxes[:, 2] - exemplar_boxes[:, 0],
        exemplar_boxes[:, 3] - exemplar_boxes[:, 1],
    ], dim=1)
    sigma_yx = wh.mean(dim=0).clamp_min(1.0).numpy()[::-1] / 8.0
    original_sum = dmap.sum().clamp_min(1.0)
    dmap = torch.from_numpy(gaussian_filter(dmap.numpy(), sigma_yx)).float()
    dmap = dmap / dmap.sum().clamp_min(1e-6) * original_sum
    return dmap.unsqueeze(0)


# ---------------------------------------------------------------------------
# Packed-storage helpers (HDF5 per split instead of one file per sample)
#
# Density maps live at:
#   {counting_dir}/gt_density_map_adaptive_{image_size}_{image_size}_coco/{split}.h5
#   dataset key = sample_key, value = float32 array (image_size, image_size)
#
# SAM masks live at:
#   {sam_mask_dir}/{split}.h5
#   dataset key = sample_key, value = uint8 array (num_objects, image_size, image_size)
# ---------------------------------------------------------------------------

def _density_h5_dir(counting_dir, image_size):
    return os.path.join(
        counting_dir, f'gt_density_map_adaptive_{image_size}_{image_size}_coco'
    )


def _density_h5_path(counting_dir, image_size, split):
    return os.path.join(_density_h5_dir(counting_dir, image_size), f'{split}.h5')


def _sam_h5_path(sam_mask_dir, split):
    return os.path.join(sam_mask_dir, f'{split}.h5')


class COCOCountingDatasetV2(Dataset):

    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, prompt_type='box', sam_mask_dir='',
        text_embedding_path='', text_feature_dim=0, counting_dir='',
        text_dropout_p=0.3
    ):
        self.split = split
        self.data_path = data_path
        self.counting_dir = coco_counting_dir(data_path, counting_dir)
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.prompt_type = prompt_type
        self.sam_mask_dir = sam_mask_dir
        self.text_embedding_path = text_embedding_path
        self.text_feature_dim = text_feature_dim
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        # Raised from the original hardcoded 0.1: at 0.1 the model could
        # lean heavily on the text-conditioning pathway (category name ->
        # density pattern) since it saw a full, correct text embedding 90%
        # of the time. A higher dropout forces more reliance on the
        # exemplar/SAM mask pathway, which is the part that's supposed to
        # transfer to FSC-147's unseen categories.
        self.text_dropout_p = text_dropout_p
        self.text_embeddings = load_text_embedding_cache(text_embedding_path)

        sample_path = os.path.join(self.counting_dir, f'samples_{split}.json')
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(
                f"Missing COCO counting samples: {sample_path}. "
                "Run utils/prepare_coco_counting_v2.py first."
            )
        with open(sample_path, 'r', encoding='utf-8') as handle:
            self.samples = json.load(handle)

        if self.text_embeddings is not None and self.text_feature_dim > 0:
            sample_embedding = next(iter(self.text_embeddings.values()))
            sample_tokens, sample_features = split_text_embedding_bundle(sample_embedding)
            sample_dim = (
                sample_features.shape[-1]
                if sample_features is not None
                else sample_tokens.shape[-1]
                if sample_tokens is not None
                else 0
            )
            if sample_dim != self.text_feature_dim:
                raise ValueError(
                    f"Text cache dimension ({sample_dim}) does not match "
                    f"--text_feature_dim ({self.text_feature_dim})."
                )

        # HDF5 handles are opened lazily (see _get_density_h5 / _get_sam_h5)
        # rather than in __init__, because DataLoader worker processes are
        # forked/spawned after __init__ runs, and h5py file handles do not
        # survive fork safely. Each worker process opens its own handle on
        # first access and keeps it open for its lifetime.
        self._density_h5_file_path = _density_h5_path(
            self.counting_dir, self.img_size, self.split
        )
        self._density_h5 = None

        if self.prompt_type == 'click_sam':
            if not self.sam_mask_dir:
                raise ValueError("prompt_type='click_sam' requires --sam_mask_dir")
            self._sam_h5_file_path = _sam_h5_path(self.sam_mask_dir, self.split)
        else:
            self._sam_h5_file_path = None
        self._sam_h5 = None

    def _get_density_h5(self):
        if self._density_h5 is None:
            if not os.path.isfile(self._density_h5_file_path):
                raise FileNotFoundError(
                    f"Missing packed density map cache: {self._density_h5_file_path}. "
                    "Run utils/prepare_coco_counting_v2.py first."
                )
            self._density_h5 = h5py.File(self._density_h5_file_path, 'r')
        return self._density_h5

    def _get_sam_h5(self):
        if self._sam_h5 is None:
            if not os.path.isfile(self._sam_h5_file_path):
                raise FileNotFoundError(
                    f"Missing packed SAM mask cache: {self._sam_h5_file_path}. "
                    "Run utils/precompute_coco_sam_masks_v2.py first."
                )
            self._sam_h5 = h5py.File(self._sam_h5_file_path, 'r')
        return self._sam_h5

    def _load_density_map(self, sample_key):
        h5_file = self._get_density_h5()
        array = h5_file[sample_key][:]
        return torch.from_numpy(array).float().unsqueeze(0)

    def _load_sam_masks(self, sample_key):
        h5_file = self._get_sam_h5()
        array = h5_file[sample_key][:]  # (stored_objects, H, W) uint8
        masks = torch.from_numpy(array).float()

        if masks.shape[0] < self.num_objects:
            pad = self.num_objects - masks.shape[0]
            padding = torch.zeros(pad, *masks.shape[1:], dtype=masks.dtype)
            masks = torch.cat([masks, padding], dim=0)
        masks = masks[:self.num_objects]

        if masks.shape[-2:] != (self.img_size, self.img_size):
            masks = F.interpolate(
                masks.unsqueeze(1), size=(self.img_size, self.img_size), mode='nearest'
            ).squeeze(1)

        return masks.unsqueeze(1)  # (num_objects, 1, H, W)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = coco_image_path(
            self.data_path, sample['coco_split'], sample['file_name']
        )
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)

        bboxes = _scale_boxes(
            sample['exemplar_boxes_xyxy'][:self.num_objects],
            width,
            height,
            self.img_size,
        )
        click_points = extract_exemplar_clicks(bboxes)

        try:
            density_map = self._load_density_map(sample['sample_key'])
        except KeyError:
            # Fall back to generating on the fly if this particular sample
            # is missing from the packed cache (e.g. cache built with a
            # different --num_objects or --min_area than expected).
            point_scale = torch.tensor([self.img_size / width, self.img_size / height])
            points = (
                torch.tensor(sample['points_xy'], dtype=torch.float32) * point_scale
            ).tolist()
            density_map = _make_density_map(points, bboxes, self.img_size)

        sam_masks = (
            self._load_sam_masks(sample['sample_key'])
            if self.prompt_type == 'click_sam'
            else torch.empty(self.num_objects, 1, 0, 0, dtype=torch.float32)
        )

        if self.text_embeddings is None:
            text_embeddings = torch.empty(0, dtype=torch.float32)
        else:
            try:
                text_embeddings = annotate_text_embedding_bundle(
                    self.text_embeddings[sample['sample_key']],
                    present=True,
                )
            except KeyError as exc:
                raise KeyError(
                    f"Missing text embedding for {sample['sample_key']} in "
                    f"{self.text_embedding_path}"
                ) from exc

        if (
            self.split == 'train'
            and self.text_embeddings is not None
            and torch.rand(1).item() < self.text_dropout_p
        ):
            text_embeddings = annotate_text_embedding_bundle(text_embeddings, present=False)

        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, click_points, density_map, sam_masks = tiling_augmentation(
                img, bboxes, click_points, density_map, sam_masks, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.split == 'train' and not tiled and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]
            click_points[:, 0] = self.img_size - click_points[:, 0]
            if sam_masks.numel() > 0:
                sam_masks = torch.flip(sam_masks, dims=[-1])

        return img, bboxes, click_points, sam_masks, text_embeddings, density_map

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        # Close HDF5 handles cleanly if this worker opened any.
        for handle in (self._density_h5, self._sam_h5):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass


def build_coco_counting_samples_v2(
    data_path,
    output_dir='',
    num_objects=3,
    min_area=64.0,
    min_box_size=4.0,
    split_seed=0,
    category_split=True,
    heldout_category_frac=0.2,
):
    output_dir = coco_counting_dir(data_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_samples = {}
    all_text_prompts = {}
    all_category_ids = set()

    for coco_split in ('train', 'val'):
        annotation_path = os.path.join(
            data_path, 'annotations', f'instances_{coco_split}2017.json'
        )
        with open(annotation_path, 'r', encoding='utf-8') as handle:
            coco = json.load(handle)

        images = {image['id']: image for image in coco['images']}
        categories = {category['id']: category['name'] for category in coco['categories']}
        grouped = defaultdict(list)
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if ann.get('area', w * h) < min_area or w < min_box_size or h < min_box_size:
                continue
            grouped[(ann['image_id'], ann['category_id'])].append(ann)

        split_samples = []
        for (image_id, category_id), anns in grouped.items():
            if len(anns) < num_objects:
                continue
            image = images[image_id]
            width, height = image['width'], image['height']
            anns = sorted(anns, key=lambda item: item.get('area', item['bbox'][2] * item['bbox'][3]), reverse=True)
            boxes = [
                _clip_xyxy_from_xywh(ann['bbox'], width, height)
                for ann in anns
            ]
            boxes = [
                box for box in boxes
                if box[2] > box[0] and box[3] > box[1]
            ]
            if len(boxes) < num_objects:
                continue

            category_name = categories[category_id]
            all_category_ids.add(category_id)
            sample_key = f"{os.path.splitext(image['file_name'])[0]}_cat{category_id}"
            sample = {
                'sample_key': sample_key,
                'image_id': image_id,
                'file_name': image['file_name'],
                'coco_split': coco_split,
                'category_id': category_id,
                'category_name': category_name,
                'width': width,
                'height': height,
                'exemplar_boxes_xyxy': boxes[:num_objects],
                'points_xy': [
                    [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]
                    for box in boxes
                ],
                'count': len(boxes),
            }
            split_samples.append(sample)
            all_text_prompts[sample_key] = {
                'text': [
                    category_name,
                    f"a photo of {category_name}",
                    f"all {category_name} objects",
                    f"count every {category_name} in the image",
                ]
            }

        all_samples[coco_split] = sorted(split_samples, key=lambda item: item['sample_key'])

    category_split_info = None

    if category_split:
        # Zero-shot-style split: a fraction of COCO categories is held out
        # entirely for val/test and never appears in train. This mirrors
        # FSC-147's protocol (unseen categories at test time). The original
        # split shared all 80 categories between train and val/test, which
        # let the model partly memorize a closed-vocabulary "category name
        # -> density pattern" shortcut via the text-conditioning pathway --
        # a shortcut that doesn't exist for FSC-147's genuinely novel
        # categories, and doesn't transfer.
        sorted_category_ids = sorted(all_category_ids)
        rng = random.Random(split_seed)
        rng.shuffle(sorted_category_ids)
        num_heldout = max(1, int(round(len(sorted_category_ids) * heldout_category_frac)))
        heldout_categories = set(sorted_category_ids[:num_heldout])
        train_categories = set(sorted_category_ids[num_heldout:])

        train_samples = [
            sample for sample in all_samples['train']
            if sample['category_id'] in train_categories
        ]

        heldout_pool = (
            [s for s in all_samples['train'] if s['category_id'] in heldout_categories]
            + [s for s in all_samples['val'] if s['category_id'] in heldout_categories]
        )
        seen_keys = set()
        deduped_pool = []
        for sample in heldout_pool:
            if sample['sample_key'] in seen_keys:
                continue
            seen_keys.add(sample['sample_key'])
            deduped_pool.append(sample)

        heldout_image_ids = sorted({sample['image_id'] for sample in deduped_pool})
        rng.shuffle(heldout_image_ids)
        midpoint = len(heldout_image_ids) // 2
        val_ids = set(heldout_image_ids[:midpoint])

        val_samples = [s for s in deduped_pool if s['image_id'] in val_ids]
        test_samples = [s for s in deduped_pool if s['image_id'] not in val_ids]

        category_split_info = {
            'train_categories': sorted(train_categories),
            'heldout_categories': sorted(heldout_categories),
            'heldout_category_frac': heldout_category_frac,
        }
    else:
        train_samples = all_samples['train']
        val_image_ids = sorted({sample['image_id'] for sample in all_samples['val']})
        random.Random(split_seed).shuffle(val_image_ids)
        midpoint = len(val_image_ids) // 2
        val_ids = set(val_image_ids[:midpoint])
        val_samples = [
            sample for sample in all_samples['val'] if sample['image_id'] in val_ids
        ]
        test_samples = [
            sample for sample in all_samples['val'] if sample['image_id'] not in val_ids
        ]

    split_payloads = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }
    for split, samples in split_payloads.items():
        with open(os.path.join(output_dir, f'samples_{split}.json'), 'w', encoding='utf-8') as handle:
            json.dump(samples, handle)

    with open(os.path.join(output_dir, 'text_prompts_coco.json'), 'w', encoding='utf-8') as handle:
        json.dump(all_text_prompts, handle)

    # Density maps are packed one HDF5 file PER SPLIT instead of one .npy
    # per sample. This is the change that fixes the runaway file count:
    # previously this loop created one file per sample (hundreds of
    # thousands of files across train/val/test).
    density_dir = _density_h5_dir(output_dir, 512)
    os.makedirs(density_dir, exist_ok=True)
    for split, samples in split_payloads.items():
        h5_path = _density_h5_path(output_dir, 512, split)
        existing_keys = set()
        file_mode = 'w'
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as existing:
                existing_keys = set(existing.keys())
            file_mode = 'a'

        with h5py.File(h5_path, file_mode) as h5_out:
            for sample in samples:
                if sample['sample_key'] in existing_keys:
                    continue
                bboxes = _scale_boxes(
                    sample['exemplar_boxes_xyxy'],
                    sample['width'],
                    sample['height'],
                    512,
                )
                point_scale = torch.tensor([512 / sample['width'], 512 / sample['height']])
                points = (
                    torch.tensor(sample['points_xy'], dtype=torch.float32) * point_scale
                ).tolist()
                density_map = _make_density_map(points, bboxes[:num_objects], 512)
                h5_out.create_dataset(
                    sample['sample_key'],
                    data=density_map.squeeze(0).numpy(),
                    compression='gzip',
                    compression_opts=4,
                )

    summary = {
        'num_objects': num_objects,
        'min_area': min_area,
        'min_box_size': min_box_size,
        'category_split': category_split,
        'category_split_info': category_split_info,
        'splits': {split: len(samples) for split, samples in split_payloads.items()},
        'text_prompts': len(all_text_prompts),
        'density_maps': sum(len(samples) for samples in split_payloads.values()),
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    return summary