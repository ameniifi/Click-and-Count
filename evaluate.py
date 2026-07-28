from models.clickcount import build_model
from utils.coco_data import COCOCountingDataset
from utils.data import FSC147Dataset, collate_fn
from utils.arg_parser import get_argparser
from utils.text_embeddings import (
    batch_text_embedding_bundles,
    infer_text_embedding_dim,
    infer_text_embedding_dim_from_state_dict,
    move_text_embedding_bundle,
)

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


def validate_args(args):
    if (
        args.prompt_type == 'click_sam' and
        not args.zero_shot and
        not args.sam_checkpoint and
        not args.sam_mask_dir
    ):
        raise ValueError(
            "click_sam prompting requires either --sam_checkpoint for online SAM "
            "or --sam_mask_dir for precomputed masks."
        )
    if args.click_only and args.prompt_type != 'click_sam':
        raise ValueError("--click_only requires --prompt_type click_sam.")
    if args.text_embedding_path:
        if args.text_feature_dim <= 0:
            args.text_feature_dim = infer_text_embedding_dim(args.text_embedding_path)
        if args.text_feature_dim <= 0:
            raise ValueError(
                "Could not infer a text embedding dimension from "
                f"{args.text_embedding_path}."
            )
    if args.text_embedding_path and not os.path.exists(args.text_embedding_path):
        raise FileNotFoundError(
            f"Missing text embedding cache: {args.text_embedding_path}"
        )


def _prepare_ddp_state_dict(state_dict):
    if all(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {
        key if key.startswith('module.') else f'module.{key}': value
        for key, value in state_dict.items()
    }


def build_dataset(args, split):
    dataset_kwargs = dict(
        data_path=args.data_path,
        img_size=args.image_size,
        split=split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        prompt_type=args.prompt_type,
        sam_mask_dir=args.sam_mask_dir,
        text_embedding_path=args.text_embedding_path,
        text_feature_dim=args.text_feature_dim,
    )
    if args.dataset == 'fsc147':
        return FSC147Dataset(**dataset_kwargs)
    if args.dataset == 'coco':
        return COCOCountingDataset(
            **dataset_kwargs,
            counting_dir=args.coco_counting_dir,
        )
    raise ValueError(
        f"Unsupported dataset: {args.dataset}. Note: this copy of evaluate.py "
        "does not implement 'coco_v2'."
    )


@torch.no_grad()
def evaluate(args):
    checkpoint_path = os.path.join(args.model_path, f'{args.model_name}.pt')
    if not args.text_embedding_path and args.text_feature_dim <= 0 and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        args.text_feature_dim = infer_text_embedding_dim_from_state_dict(state_dict)

    validate_args(args)

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(checkpoint_path)['model']
    incompatible = model.load_state_dict(
        _prepare_ddp_state_dict(state_dict),
        strict=False
    )
    if rank == 0 and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Checkpoint load:",
            f"missing={list(incompatible.missing_keys)[:20]}",
            f"unexpected={list(incompatible.unexpected_keys)[:20]}",
        )
    if rank == 0 and args.click_only:
        print("click_only=True: ground-truth boxes will be withheld from the "
              "model at evaluation time.")

    # Small epsilon to avoid division by zero when a ground-truth count is 0
    nae_eps = 1e-6

    for split in ['val', 'test']:
        test = build_dataset(args, split)
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        nae = torch.tensor(0.0).to(device)
        model.eval()
        for img, bboxes, click_points, sam_masks, text_embeddings, density_map in test_loader:
            img = img.to(device)
            bboxes = None if args.click_only else bboxes.to(device)
            click_points = click_points.to(device)
            sam_masks = sam_masks.to(device)
            text_embeddings = move_text_embedding_bundle(text_embeddings, device)
            text_embeddings = batch_text_embedding_bundles(text_embeddings)
            density_map = density_map.to(device)

            out, _, _ = model(
                img, bboxes=bboxes, click_points=click_points, sam_masks=sam_masks,
                text_embeddings=text_embeddings
            )

            gt_count = density_map.flatten(1).sum(dim=1)
            pred_count = out.flatten(1).sum(dim=1)
            abs_diff = torch.abs(gt_count - pred_count)

            ae += abs_diff.sum()
            se += (abs_diff ** 2).sum()
            nae += (abs_diff / gt_count.clamp(min=nae_eps)).sum()

        dist.all_reduce(ae)
        dist.all_reduce(se)
        dist.all_reduce(nae)

        if rank == 0:
            print(
                f"{split.capitalize()} set",
                f"MAE: {ae.item() / len(test):.2f}",
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
                f"NAE: {nae.item() / len(test):.4f}",
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ClickCount', parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)