import argparse
import json
import os

from PIL import Image
import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.coco_data import coco_counting_dir, coco_image_path, _scale_boxes
from utils.data import extract_exemplar_clicks
from utils.sam_utils import predict_exemplar_mask

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:  # pragma: no cover - depends on optional package
    SamPredictor = None
    sam_model_registry = None


def parse_args():
    parser = argparse.ArgumentParser("Precompute SAM masks for v2 COCO counting samples")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--counting_dir', default='', type=str)
    parser.add_argument('--sam_checkpoint', required=True, type=str)
    parser.add_argument('--sam_model_type', default='vit_b', type=str)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val', 'test'],
        choices=['train', 'val', 'test']
    )
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str,
    )
    return parser.parse_args()


def precompute_masks(args):
    if SamPredictor is None or sam_model_registry is None:
        raise ImportError(
            "segment_anything is required to precompute COCO click_sam masks."
        )

    counting_dir = coco_counting_dir(args.data_path, args.counting_dir)
    # NOTE: output_dir is a directory that will hold one .h5 file PER SPLIT,
    # e.g. sam_masks_vit_b_512_pointbox/train.h5, instead of one .npy per sample.
    output_dir = args.output_dir or os.path.join(
        counting_dir,
        f'sam_masks_{args.sam_model_type}_{args.image_size}_pointbox'
    )
    os.makedirs(output_dir, exist_ok=True)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    for split in args.splits:
        sample_path = os.path.join(counting_dir, f'samples_{split}.json')
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(
                f"Missing COCO counting samples: {sample_path}. "
                "Run utils/prepare_coco_counting_v2.py first."
            )
        with open(sample_path, 'r', encoding='utf-8') as handle:
            split_samples = json.load(handle)

        h5_path = os.path.join(output_dir, f'{split}.h5')

        # Resume support: if the file already has some keys and --overwrite
        # was not passed, skip samples already written instead of redoing
        # the whole split.
        existing_keys = set()
        file_mode = 'w'
        if os.path.exists(h5_path) and not args.overwrite:
            with h5py.File(h5_path, 'r') as existing:
                existing_keys = set(existing.keys())
            file_mode = 'a'

        active_image_key = None
        active_size = None

        with h5py.File(h5_path, file_mode) as h5_out:
            for sample in tqdm(split_samples, desc=f'SAM masks [{split}]'):
                key = sample['sample_key']
                if key in existing_keys:
                    continue

                image_key = (sample['coco_split'], sample['file_name'])
                if image_key != active_image_key:
                    image_path = coco_image_path(
                        args.data_path, sample['coco_split'], sample['file_name']
                    )
                    image = Image.open(image_path).convert("RGB")
                    width, height = image.size
                    resized_image = image.resize(
                        (args.image_size, args.image_size),
                        resample=Image.BILINEAR
                    )
                    predictor.set_image(np.asarray(resized_image))
                    active_image_key = image_key
                    active_size = (width, height)
                else:
                    width, height = active_size

                bboxes = _scale_boxes(
                    sample['exemplar_boxes_xyxy'][:args.num_objects],
                    width,
                    height,
                    args.image_size,
                )
                click_points = extract_exemplar_clicks(bboxes).numpy()
                bboxes_np = bboxes.numpy()

                image_masks = []
                for point, bbox in zip(click_points, bboxes_np):
                    best_mask = predict_exemplar_mask(predictor, point, bbox)
                    image_masks.append(best_mask.astype(np.uint8))

                stacked = np.stack(image_masks, axis=0)
                # gzip level 4 is a good speed/size tradeoff for binary masks
                h5_out.create_dataset(
                    key, data=stacked, compression='gzip', compression_opts=4
                )

        print(f"Wrote {len(split_samples)} samples to {h5_path}")


if __name__ == '__main__':
    precompute_masks(parse_args())