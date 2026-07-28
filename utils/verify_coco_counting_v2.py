import argparse
import json
import os

import h5py
import numpy as np

from utils.coco_data import (
    _density_h5_path,
    _sam_h5_path,
    coco_counting_dir,
    coco_image_path,
)
from utils.text_embeddings import load_text_embedding_keys


def parse_args():
    parser = argparse.ArgumentParser("Verify prepared v2 COCO counting caches")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--counting_dir', default='', type=str)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--sam_mask_dir', default='', type=str)
    parser.add_argument('--text_embedding_path', default='', type=str)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val', 'test'],
        choices=['train', 'val', 'test']
    )
    parser.add_argument('--max_density_count_error', default=1e-2, type=float)
    return parser.parse_args()


def check_split(args, counting_dir, text_cache, split, samples):
    errors_by_sample = {}

    density_h5_path = _density_h5_path(counting_dir, args.image_size, split)
    if not os.path.isfile(density_h5_path):
        errors_by_sample['__split__'] = [f"missing density cache: {density_h5_path}"]
        density_h5 = None
    else:
        density_h5 = h5py.File(density_h5_path, 'r')

    sam_h5 = None
    if args.sam_mask_dir:
        sam_h5_path = _sam_h5_path(args.sam_mask_dir, split)
        if not os.path.isfile(sam_h5_path):
            errors_by_sample.setdefault('__split__', []).append(
                f"missing SAM mask cache: {sam_h5_path}"
            )
        else:
            sam_h5 = h5py.File(sam_h5_path, 'r')

    for sample in samples:
        sample_key = sample['sample_key']
        errors = []

        image_path = coco_image_path(args.data_path, sample['coco_split'], sample['file_name'])
        if not os.path.isfile(image_path):
            errors.append(f"missing image: {image_path}")

        if density_h5 is not None:
            if sample_key not in density_h5:
                errors.append(f"missing density map key in {density_h5_path}: {sample_key}")
            else:
                density_sum = float(np.asarray(density_h5[sample_key]).sum())
                expected_count = float(sample['count'])
                if abs(density_sum - expected_count) > args.max_density_count_error:
                    errors.append(
                        f"density sum {density_sum:.4f} != count {expected_count:.4f}"
                    )

        if args.sam_mask_dir and sam_h5 is not None:
            if sample_key not in sam_h5:
                errors.append(f"missing SAM mask key in {sam_h5.filename}: {sample_key}")
            else:
                masks = np.asarray(sam_h5[sample_key])
                if masks.shape[0] < args.num_objects:
                    errors.append(
                        f"SAM mask has {masks.shape[0]} exemplars, expected {args.num_objects}"
                    )
                if masks.shape[-2:] != (args.image_size, args.image_size):
                    errors.append(
                        f"SAM mask size {masks.shape[-2:]} != "
                        f"({args.image_size}, {args.image_size})"
                    )

        if text_cache is not None and sample_key not in text_cache:
            errors.append(f"missing text embedding for {sample_key}")

        if errors:
            errors_by_sample[sample_key] = errors

    if density_h5 is not None:
        density_h5.close()
    if sam_h5 is not None:
        sam_h5.close()

    return errors_by_sample


def main():
    args = parse_args()
    counting_dir = coco_counting_dir(args.data_path, args.counting_dir)
    text_cache = (
        load_text_embedding_keys(args.text_embedding_path)
        if args.text_embedding_path
        else None
    )

    total = 0
    failures = []
    split_counts = {}

    for split in args.splits:
        sample_path = os.path.join(counting_dir, f'samples_{split}.json')
        if not os.path.isfile(sample_path):
            failures.append((split, 'split', [f"missing samples file: {sample_path}"]))
            continue

        with open(sample_path, 'r', encoding='utf-8') as handle:
            samples = json.load(handle)

        split_counts[split] = len(samples)
        total += len(samples)

        split_errors = check_split(args, counting_dir, text_cache, split, samples)
        for sample_key, errors in split_errors.items():
            failures.append((split, sample_key, errors))

    print("COCO v2 counting cache verification")
    print(f"  counting_dir: {counting_dir}")
    print(f"  splits: {split_counts}")
    print(f"  checked samples: {total}")
    print(f"  failures: {len(failures)}")

    if failures:
        for split, sample_key, errors in failures[:50]:
            print(f"[{split}] {sample_key}")
            for error in errors:
                print(f"  - {error}")
        if len(failures) > 50:
            print(f"... {len(failures) - 50} additional failing samples omitted")
        raise SystemExit(1)

    print("OK")


if __name__ == '__main__':
    main()