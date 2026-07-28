import os
import json
import argparse

from PIL import Image
import numpy as np

from tqdm import tqdm

from utils.data import extract_exemplar_boxes, extract_exemplar_clicks
from utils.sam_utils import predict_exemplar_mask

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:  
    SamPredictor = None
    sam_model_registry = None


def parse_args():
    parser = argparse.ArgumentParser("Precompute SAM exemplar masks")
    parser.add_argument('--data_path', required=True, type=str)
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
    return parser.parse_args()


def precompute_masks(args):
    if SamPredictor is None or sam_model_registry is None:
        raise ImportError(
            "segment_anything is required to precompute click_sam masks."
        )

    output_dir = args.output_dir or os.path.join(
        args.data_path,
        f'sam_masks_{args.sam_model_type}_{args.image_size}_pointbox'
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(args.data_path, 'Train_Test_Val_FSC_147.json'), 'rb') as file:
        splits = json.load(file)
    with open(os.path.join(args.data_path, 'annotation_FSC147_384.json'), 'rb') as file:
        annotations = json.load(file)

    image_names = list()
    for split in args.splits:
        image_names.extend(splits[split])

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    predictor = SamPredictor(sam)

    for image_name in tqdm(image_names):
        output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.npy')
        if os.path.exists(output_path) and not args.overwrite:
            continue

        image_path = os.path.join(args.data_path, 'images_384_VarV2', image_name)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        resized_image = image.resize((args.image_size, args.image_size), resample=Image.BILINEAR)

        bboxes = extract_exemplar_boxes(
            annotations[image_name], width, height, args.image_size, args.num_objects
        )
        click_points = extract_exemplar_clicks(bboxes).numpy()
        bboxes_np = bboxes.numpy()

        predictor.set_image(np.asarray(resized_image))

        image_masks = list()
        for point, bbox in zip(click_points, bboxes_np):
            best_mask = predict_exemplar_mask(predictor, point, bbox)
            image_masks.append(best_mask.astype(np.uint8))

        np.save(output_path, np.stack(image_masks, axis=0))


if __name__ == '__main__':
    precompute_masks(parse_args())
