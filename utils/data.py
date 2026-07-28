import os
import json
import argparse

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

from tqdm import tqdm

from utils.text_embeddings import (
    annotate_text_embedding_bundle,
    load_text_embedding_cache,
    split_text_embedding_bundle,
)


def extract_exemplar_boxes(annotation, img_w, img_h, img_size, num_objects):
    bboxes = torch.tensor(
        annotation['box_examples_coordinates'],
        dtype=torch.float32
    )[:3, [0, 2], :].reshape(-1, 4)[:num_objects, ...]
    bboxes = bboxes / torch.tensor([img_w, img_h, img_w, img_h]) * img_size
    return bboxes


def extract_exemplar_clicks(bboxes):
    return 0.5 * (bboxes[:, :2] + bboxes[:, 2:])


def load_sam_masks(sam_mask_dir, image_name, num_objects, img_size):
    if not sam_mask_dir:
        return torch.empty(num_objects, 1, 0, 0, dtype=torch.float32)

    mask_path = os.path.join(
        sam_mask_dir,
        os.path.splitext(image_name)[0] + '.npy'
    )
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing SAM mask file for {image_name}: {mask_path}"
        )

    sam_masks = torch.from_numpy(np.load(mask_path)).float()
    if sam_masks.ndim == 3:
        sam_masks = sam_masks.unsqueeze(1)
    sam_masks = sam_masks[:num_objects, ...]

    if sam_masks.size(0) < num_objects:
        sam_masks = torch.cat([
            sam_masks,
            torch.zeros(
                num_objects - sam_masks.size(0),
                1,
                sam_masks.size(-2),
                sam_masks.size(-1)
            )
        ], dim=0)

    if sam_masks.size(-2) != img_size or sam_masks.size(-1) != img_size:
        sam_masks = F.interpolate(
            sam_masks, size=(img_size, img_size),
            mode='bilinear', align_corners=False
        )

    return sam_masks


def tiling_augmentation(
    img, bboxes, click_points, density_map, sam_masks, resize, jitter, tile_size, hflip_p
):

    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
        click_points[:, 0] = x_target - click_points[:, 0]
        if sam_masks.numel() > 0:
            sam_masks = torch.flip(sam_masks, dims=[-1])

    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    click_points = click_points / torch.tensor([x_tile, y_tile])

    if sam_masks.numel() > 0:
        crop_h = int((y_tile * y_target).item())
        crop_w = int((x_tile * x_target).item())
        pad_h = max(crop_h - sam_masks.size(-2), 0)
        pad_w = max(crop_w - sam_masks.size(-1), 0)
        sam_masks = F.pad(sam_masks, (0, pad_w, 0, pad_h))
        sam_masks = sam_masks[..., :crop_h, :crop_w]
        sam_masks = F.interpolate(
            sam_masks, size=(y_target, x_target),
            mode='bilinear', align_corners=False
        ).clamp_(0, 1)

    return img, bboxes, click_points, density_map, sam_masks


class FSC147Dataset(Dataset):

    def __init__(
        self, data_path, img_size, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, prompt_type='box', sam_mask_dir='',
        text_embedding_path='', text_feature_dim=0
    ):
        self.split = split
        self.data_path = data_path
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
        self.text_embeddings = load_text_embedding_cache(text_embedding_path)
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
        with open(
            os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
            os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        )).convert("RGB")
        w, h = img.size
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

        bboxes = extract_exemplar_boxes(
            self.annotations[self.image_names[idx]], w, h, self.img_size, self.num_objects
        )
        click_points = extract_exemplar_clicks(bboxes)
        sam_masks = (
            load_sam_masks(
                self.sam_mask_dir, self.image_names[idx], self.num_objects, self.img_size
            ) if self.prompt_type == 'click_sam'
            else torch.empty(self.num_objects, 1, 0, 0, dtype=torch.float32)
        )

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        if self.img_size != 512:
            original_sum = density_map.sum()
            density_map = self.resize(density_map)
            density_map = density_map / density_map.sum() * original_sum

        if self.text_embeddings is None:
            text_embeddings = torch.empty(0, dtype=torch.float32)
        else:
            try:
                text_embeddings = annotate_text_embedding_bundle(
                    self.text_embeddings[self.image_names[idx]],
                    present=True,
                )
            except KeyError as exc:
                raise KeyError(
                    f"Missing text embedding for {self.image_names[idx]} in "
                    f"{self.text_embedding_path}"
                ) from exc

        if (
            self.split == 'train'
            and self.text_embeddings is not None
            and torch.rand(1).item() < 0.1
        ):
            text_embeddings = annotate_text_embedding_bundle(text_embeddings, present=False)

        # data augmentation
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
        return len(self.image_names)


def collate_fn(batch):
    """
    Custom collate that handles text_embeddings being a mix of per-sample
    dicts/tensors (full embeddings) and per-sample zeroed bundles tagged with
    a `present` flag when text dropout is applied.

    All other fields have fixed shapes per sample and are stacked normally.
    """
    imgs, bboxes, click_points, sam_masks, text_embeddings, density_maps = zip(*batch)

    imgs         = torch.stack(imgs, 0)
    bboxes       = torch.stack(bboxes, 0)
    click_points = torch.stack(click_points, 0)
    sam_masks    = torch.stack(sam_masks, 0)
    density_maps = torch.stack(density_maps, 0)

    # Keep as a plain Python list — _rebatch_text_embeddings in train.py
    # will convert this to a single batched bundle (or None) after .to(device).
    text_embeddings = list(text_embeddings)

    return imgs, bboxes, click_points, sam_masks, text_embeddings, density_maps


def generate_density_maps(data_path, target_size=(512, 512)):

    density_map_path = os.path.join(
        data_path,
        f'gt_density_map_adaptive_{target_size[0]}_{target_size[1]}_object_VarV2'
    )
    if not os.path.isdir(density_map_path):
        os.makedirs(density_map_path)

    with open(
        os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i, (image_name, ann) in enumerate(tqdm(annotations.items())):
        _, h, w = T.ToTensor()(Image.open(os.path.join(
            data_path,
            'images_384_VarV2',
            image_name
        ))).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

        points = (
            torch.tensor(ann['points'], device=device) *
            torch.tensor([w_ratio, h_ratio], device=device)
        ).long()
        points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
        points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)
        bboxes = box_convert(torch.tensor(
            ann['box_examples_coordinates'],
            dtype=torch.float32,
            device=device
        )[:3, [0, 2], :].reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
        bboxes = bboxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=device)
        window_size = bboxes.mean(dim=0)[2:].cpu().numpy()[::-1]

        dmap = torch.zeros(*target_size)
        for p in range(points.size(0)):
            dmap[points[p, 1], points[p, 0]] += 1
        dmap = gaussian_filter(dmap.cpu().numpy(), window_size / 8)

        np.save(os.path.join(density_map_path, os.path.splitext(image_name)[0] + '.npy'), dmap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument(
        '--data_path',
        default='/path/to/data/fsc147/',
        type=str
    )
    parser.add_argument('--image_size', default=512, type=int)
    args = parser.parse_args()
    generate_density_maps(args.data_path, (args.image_size, args.image_size))
