import argparse

from utils.coco_data import build_coco_counting_samples_v2


def parse_args():
    parser = argparse.ArgumentParser("Prepare category-disjoint COCO counting samples for ClickCount")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--min_area', default=64.0, type=float)
    parser.add_argument('--min_box_size', default=4.0, type=float)
    parser.add_argument('--split_seed', default=0, type=int)
    parser.add_argument(
        '--legacy_image_split', action='store_true',
        help=(
            'Use the old image-disjoint / same-category split instead of '
            'the category-disjoint (zero-shot-style) split.'
        )
    )
    parser.add_argument(
        '--heldout_category_frac', default=0.2, type=float,
        help='Fraction of COCO categories held out entirely for val/test.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_coco_counting_samples_v2(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_objects=args.num_objects,
        min_area=args.min_area,
        min_box_size=args.min_box_size,
        split_seed=args.split_seed,
        category_split=not args.legacy_image_split,
        heldout_category_frac=args.heldout_category_frac,
    )
    print("Prepared COCO counting samples:")
    for split, count in summary['splits'].items():
        print(f"  {split}: {count}")
    print(f"  text prompts: {summary['text_prompts']}")
    print(f"  density maps: {summary['density_maps']}")
    print(f"  category_split: {summary['category_split']}")
    if summary['category_split_info']:
        info = summary['category_split_info']
        print(f"  heldout categories ({len(info['heldout_categories'])}): {info['heldout_categories']}")


if __name__ == '__main__':
    main()