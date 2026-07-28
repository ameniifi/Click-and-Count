import argparse


def get_argparser():

    parser = argparse.ArgumentParser("ClickCount parser", add_help=False)

    parser.add_argument('--model_name', default='clickcount_3_shot', type=str)
    parser.add_argument('--dataset', default='fsc147', choices=['fsc147', 'coco', 'coco_v2'], type=str)
    parser.add_argument(
        '--data_path',
        default='/path/to/data/fsc147/',
        type=str
    )
    parser.add_argument('--coco_counting_dir', default='', type=str)
    parser.add_argument(
        '--model_path',
        default='/path/to/models/pretrained/',
        type=str
    )
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--backbone_checkpoint', default='', type=str)
    parser.add_argument('--swav_backbone', action='store_true')
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_ope_iterative_steps', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    # Optional second LR drop milestone. 0 disables it and training
    # falls back to the original single-drop StepLR schedule.
    parser.add_argument('--lr_drop2', default=0, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument(
        '--prompt_type', default='box', choices=['box', 'click_sam'], type=str
    )
    parser.add_argument('--sam_checkpoint', default='', type=str)
    parser.add_argument('--sam_model_type', default='vit_b', type=str)
    parser.add_argument('--sam_mask_dir', default='', type=str)
    parser.add_argument('--click_only', action='store_true',
        help='Withhold ground-truth exemplar boxes from the model at train/eval '
             'time. Only the click-derived SAM mask supplies shape/appearance '
             '(OPEModule derives a box from the mask via _masks_to_bboxes). '
             'Requires prompt_type=click_sam.')
    parser.add_argument('--text_embedding_path', default='', type=str)
    parser.add_argument('--text_feature_dim', default=0, type=int)
    parser.add_argument('--text_alignment_weight', default=0.0, type=float)
    parser.add_argument('--text_alignment_warmup_epochs', default=30, type=int)
    parser.add_argument('--count_loss_weight', default=0.0, type=float)
    # Probability of dropping the text embedding per train sample. Only
    # consumed by dataset='coco_v2' (COCOCountingDatasetV2); harmless no-op
    # for 'fsc147' and 'coco'.
    parser.add_argument('--text_dropout_p', default=0.3, type=float)
    # EMA decay for shadow-weight averaging. 0 disables EMA entirely
    # (ModelEMA is not constructed, and no EMA validation/checkpointing runs).
    parser.add_argument('--ema_decay', default=0.0, type=float)
    # Prefix path for per-sample validation outlier CSV logs, written
    # as f"{outlier_log_path}.rank{rank}.csv" on the final epoch. Empty
    # string disables outlier logging.
    parser.add_argument('--outlier_log_path', default='', type=str)
    parser.add_argument('--init_checkpoint', default='', type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='clickcount-click-sam', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_run_name', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)
    parser.add_argument('--wandb_tags', default='', type=str)
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], type=str)

    return parser