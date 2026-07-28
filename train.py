from models.clickcount import build_model
from utils.coco_data import COCOCountingDataset
from utils.data import FSC147Dataset, collate_fn
from utils.arg_parser import get_argparser
from utils.losses import CountL1Loss, ObjectNormalizedL2Loss
from utils.text_embeddings import (
    batch_text_embedding_bundles,
    infer_text_embedding_dim,
    infer_text_embedding_dim_from_state_dict,
    move_text_embedding_bundle,
)

from time import perf_counter
import argparse
import importlib.util
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


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
    if args.text_alignment_weight > 0 and not args.text_embedding_path:
        raise ValueError(
            "--text_alignment_weight requires --text_embedding_path so the model can read text."
        )
    if args.init_checkpoint and args.resume_training:
        raise ValueError("Use either --resume_training or --init_checkpoint, not both.")
    if args.init_checkpoint and not os.path.isfile(args.init_checkpoint):
        raise FileNotFoundError(f"Missing init checkpoint: {args.init_checkpoint}")
    if args.text_embedding_path and not os.path.exists(args.text_embedding_path):
        raise FileNotFoundError(
            f"Missing text embedding cache: {args.text_embedding_path}"
        )
    if (
        args.wandb
        and args.wandb_mode != 'disabled'
        and importlib.util.find_spec('wandb') is None
    ):
        raise ImportError(
            "Weights & Biases logging was requested with --wandb, but wandb is "
            "not installed. Install it with `pip install wandb` or remove --wandb."
        )


def init_wandb(args, rank):
    if not args.wandb or args.wandb_mode == 'disabled' or rank != 0:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases logging was requested with --wandb, but wandb is "
            "not installed. Install it with `pip install wandb` or remove --wandb."
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.model_name,
        group=args.wandb_group or None,
        tags=tags or None,
        mode=args.wandb_mode,
        config=vars(args),
    )


def _prepare_ddp_state_dict(state_dict):
    if all(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {
        key if key.startswith('module.') else f'module.{key}': value
        for key, value in state_dict.items()
    }



# This helper drops any
# checkpoint tensor whose shape doesn't match the current model's parameter
# shape before calling load_state_dict, so the rest of the checkpoint
# (backbone, etc.) still warm-starts and only the resized layer falls back
# to its freshly-initialized weights.
def _filter_compatible_state_dict(model, state_dict, rank=0):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            current_shape = model_state[k].shape if k in model_state else 'N/A (key not in model)'
            skipped.append((k, tuple(v.shape), current_shape))
    if skipped and rank == 0:
        print(f"[checkpoint] Skipping {len(skipped)} keys due to shape mismatch or "
              f"absence in current model:")
        for k, ckpt_shape, model_shape in skipped:
            print(f"    {k}: checkpoint={ckpt_shape} vs model={model_shape}")
    return compatible



def build_dataset(args, split):
    dataset_kwargs = dict(
        data_path=args.data_path,
        img_size=args.image_size,
        split=split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
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
        f"Unsupported dataset: {args.dataset}. Note: this copy of train.py does "
        "not implement 'coco_v2' — merge --click_only into your v2 train.py"
    )


def train(args):
    validate_args(args)

    if (
        not args.text_embedding_path and
        args.text_feature_dim <= 0 and
        args.init_checkpoint
    ):
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        args.text_feature_dim = infer_text_embedding_dim_from_state_dict(state_dict)

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
    wandb_run = init_wandb(args, rank)

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True
    )

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        filtered_resume_state_dict = _filter_compatible_state_dict(
            model, _prepare_ddp_state_dict(checkpoint['model']), rank=rank
        )
        incompatible = model.load_state_dict(filtered_resume_state_dict, strict=False)
        if rank == 0:
            print(
                "Resume checkpoint load:",
                f"missing={list(incompatible.missing_keys)[:20]}",
                f"unexpected={list(incompatible.unexpected_keys)[:20]}",
            )
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        filtered_init_state_dict = _filter_compatible_state_dict(
            model, _prepare_ddp_state_dict(state_dict), rank=rank
        )
        incompatible = model.load_state_dict(filtered_init_state_dict, strict=False)
        if rank == 0:
            print(f"Loaded init checkpoint: {args.init_checkpoint}")
            print(f"Missing keys: {list(incompatible.missing_keys)[:20]}")
            print(f"Unexpected keys: {list(incompatible.unexpected_keys)[:20]}")

    criterion = ObjectNormalizedL2Loss()
    count_criterion = CountL1Loss()

    train = build_dataset(args, 'train')
    val = build_dataset(args, 'val')
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if rank == 0 and args.click_only:
        print("click_only=True: ground-truth boxes will be withheld from the "
              "model; OPEModule will derive boxes from SAM masks instead.")

    print(rank)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        aux_val_loss = torch.tensor(0.0).to(device)
        text_train_loss = torch.tensor(0.0).to(device)
        count_train_loss = torch.tensor(0.0).to(device)
        count_val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        text_weight = (
            args.text_alignment_weight
            if epoch >= args.text_alignment_warmup_epochs
            else 0.0
        )

        train_loader.sampler.set_epoch(epoch)
        model.train()
        for img, bboxes, click_points, sam_masks, text_embeddings, density_map in train_loader:
            img = img.to(device)
            # click_only: withhold ground-truth boxes from the model.
            bboxes = None if args.click_only else bboxes.to(device)
            click_points = click_points.to(device)
            sam_masks = sam_masks.to(device)
            text_embeddings = move_text_embedding_bundle(text_embeddings, device)
            text_embeddings = batch_text_embedding_bundles(text_embeddings)
            density_map = density_map.to(device)

            optimizer.zero_grad()
            out, aux_out, text_loss = model(
                img, bboxes=bboxes, click_points=click_points, sam_masks=sam_masks,
                text_embeddings=text_embeddings
            )

            with torch.no_grad():
                num_objects = density_map.sum()
                dist.all_reduce(num_objects, op=dist.ReduceOp.SUM)

            main_loss = criterion(out, density_map, num_objects)
            aux_loss = sum([
                args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
            ])
            count_loss = count_criterion(out, density_map)
            loss = main_loss + aux_loss + text_weight * text_loss
            if args.count_loss_weight > 0:
                loss = loss + args.count_loss_weight * count_loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            text_train_loss += text_loss * img.size(0)
            count_train_loss += count_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            for img, bboxes, click_points, sam_masks, text_embeddings, density_map in val_loader:
                img = img.to(device)
                bboxes = None if args.click_only else bboxes.to(device)
                click_points = click_points.to(device)
                sam_masks = sam_masks.to(device)
                text_embeddings = move_text_embedding_bundle(text_embeddings, device)
                text_embeddings = batch_text_embedding_bundles(text_embeddings)
                density_map = density_map.to(device)

                out, aux_out, text_loss = model(
                    img, bboxes=bboxes, click_points=click_points, sam_masks=sam_masks,
                    text_embeddings=text_embeddings
                )
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce(num_objects, op=dist.ReduceOp.SUM)

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                count_loss = count_criterion(out, density_map)

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                count_val_loss += count_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()

        dist.all_reduce(train_loss,      op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss,        op=dist.ReduceOp.SUM)
        dist.all_reduce(aux_train_loss,  op=dist.ReduceOp.SUM)
        dist.all_reduce(aux_val_loss,    op=dist.ReduceOp.SUM)
        dist.all_reduce(text_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_val_loss,   op=dist.ReduceOp.SUM)
        dist.all_reduce(train_ae,        op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ae,          op=dist.ReduceOp.SUM)

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val) < best:
                best = val_ae.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pt')
                )
                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Text train loss: {text_train_loss.item():.3f}",
                f"Count train loss: {count_train_loss.item():.3f}",
                f"Text loss weight: {text_weight:.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Count val loss: {count_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )
            if wandb_run is not None:
                metrics = {
                    'epoch': epoch,
                    'train/main_loss': train_loss.item() / len(train),
                    'train/aux_loss': aux_train_loss.item() / len(train),
                    'train/text_loss': text_train_loss.item() / len(train),
                    'train/count_loss': count_train_loss.item() / len(train),
                    'train/mae': train_ae.item() / len(train),
                    'val/main_loss': val_loss.item() / len(val),
                    'val/aux_loss': aux_val_loss.item() / len(val),
                    'val/count_loss': count_val_loss.item() / len(val),
                    'val/mae': val_ae.item() / len(val),
                    'lr/non_backbone': optimizer.param_groups[0]['lr'],
                    'lr/backbone': optimizer.param_groups[1]['lr'],
                    'text/alignment_weight': text_weight,
                    'click_only': float(args.click_only),
                    'time/epoch_seconds': end - start,
                    'best/val_mae': best,
                    'best/is_best_epoch': float(best_epoch),
                }
                wandb_run.log(metrics, step=epoch)
                wandb_run.summary['best_val_mae'] = best

    dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ClickCount', parents=[get_argparser()])
    args = parser.parse_args()
    train(args)