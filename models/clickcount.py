from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .sam import SAMMaskGenerator

import torch
from torch import nn
from torch.nn import functional as F


class ClickCount(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        backbone_checkpoint: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
        prompt_type: str,
        sam_checkpoint: str,
        sam_model_type: str,
        text_feature_dim: int = 0,
    ):

        super(ClickCount, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.prompt_type = prompt_type
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.text_feature_dim = text_feature_dim

        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone,
            checkpoint_path=backbone_checkpoint
        )
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm,
            zero_shot, text_feature_dim=text_feature_dim
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)
        self.sam_mask_generator = (
            SAMMaskGenerator(sam_checkpoint, sam_model_type)
            if not zero_shot and prompt_type == 'click_sam' and sam_checkpoint
            else None
        )

    def _get_click_sam_masks(self, x, click_points, bboxes, sam_masks):
        if sam_masks is not None and sam_masks.numel() > 0:
            return sam_masks
        if self.sam_mask_generator is None:
            raise ValueError(
                "click_sam prompting requires either precomputed --sam_mask_dir masks "
                "or an online --sam_checkpoint."
            )
        # bboxes is optional: when provided it is used as a SAM disambiguation
        # prompt (improves segmentation in crowded scenes); when None SAM uses
        # the click point alone.
        return self.sam_mask_generator(x, click_points, bboxes=bboxes)

    def forward(self, x, bboxes=None, click_points=None, sam_masks=None, text_embeddings=None):
        if self.zero_shot:
            num_objects = self.num_objects
        elif self.prompt_type == 'click_sam':
            if click_points is None:
                raise ValueError("click_points are required when prompt_type=click_sam.")
            # bboxes are now optional
            num_objects = click_points.size(1)
        else:
            if bboxes is None:
                raise ValueError("bboxes are required when prompt_type=box.")
            num_objects = bboxes.size(1)

        backbone_features = self.backbone(x)
        src = self.input_proj(backbone_features)
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)

        if self.num_encoder_layers > 0:
            image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            image_features = src

        f_e = image_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)

        if self.prompt_type == 'click_sam' and not self.zero_shot:
            sam_masks = self._get_click_sam_masks(x, click_points, bboxes, sam_masks)
            all_prototypes, text_loss = self.ope(
                f_e, pos_emb, bboxes=bboxes, click_points=click_points,
                exemplar_masks=sam_masks, text_embeddings=text_embeddings
            )
        else:
            all_prototypes, text_loss = self.ope(
                f_e, pos_emb, bboxes=bboxes, text_embeddings=text_embeddings
            )

        outputs = list()
        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]

            if i == all_prototypes.size(0) - 1:
                predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps = self.aux_heads[i](response_maps)
            outputs.append(predicted_dmaps)

        return outputs[-1], outputs[:-1], text_loss


def build_model(args):

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    return ClickCount(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
        prompt_type=args.prompt_type,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        text_feature_dim=args.text_feature_dim,
    )
