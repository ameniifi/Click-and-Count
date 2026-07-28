from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed
from .text_conditioning import (
    GatedTextCrossAttention,
    TextPrototypeInitializer,
    text_alignment_loss,
)

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.ops import roi_align


class OPEModule(nn.Module):

    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        reduction: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
        text_feature_dim: int = 0,
    ):

        super(OPEModule, self).__init__()

        self.num_iterative_steps = num_iterative_steps
        self.zero_shot = zero_shot
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction
        self.text_feature_dim = text_feature_dim

        self.text_conditioners = None
        self.text_prototype_initializer = None
        if text_feature_dim > 0:
            self.text_prototype_initializer = TextPrototypeInitializer(
                text_feature_dim, emb_dim, num_heads
            )
            self.text_conditioners = nn.ModuleList([
                GatedTextCrossAttention(text_feature_dim, emb_dim, num_heads)
                for _ in range(max(num_iterative_steps, 1))
            ])

        if num_iterative_steps > 0:
            self.iterative_adaptation = IterativeAdaptationModule(
                num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                zero_shot=zero_shot,
                text_conditioners=self.text_conditioners
            )

        if not self.zero_shot:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
            )
        else:
            self.shape_or_objectness = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_or_objectness)

        self.pos_emb = PositionalEncodingsFixed(emb_dim)
        self.text_alignment_projector = (
            self.text_conditioners[-1].projector
            if self.text_conditioners is not None
            else None
        )

    def _flatten_roi_boxes(self, boxes):
        bs, num_objects = boxes.size(0), boxes.size(1)
        return torch.cat([
            torch.arange(
                bs, requires_grad=False, device=boxes.device, dtype=boxes.dtype
            ).repeat_interleave(num_objects).reshape(-1, 1),
            boxes.flatten(0, 1),
        ], dim=1)

    @torch.no_grad()
    def _masks_to_bboxes(self, exemplar_masks):
        """
        Derive tight bounding boxes [x1, y1, x2, y2] in image pixel space from
        binary SAM masks  [B, n, 1, H, W].

        Used in click-only mode (bboxes=None) so the shape MLP and RoI-Align
        have a spatial extent to work with even when the user provided only a
        click point and did not annotate a bounding box.
        """
        bs, num_objects, _, H, W = exemplar_masks.size()
        bboxes = exemplar_masks.new_zeros(bs, num_objects, 4)
        foreground = (exemplar_masks[:, :, 0] > 0.5)   # [B, n, H, W]
        for b in range(bs):
            for i in range(num_objects):
                yx = foreground[b, i].nonzero(as_tuple=False)   # [k, 2]  rows=(y,x)
                if yx.numel() > 0:
                    y0, x0 = yx[:, 0].min(), yx[:, 1].min()
                    y1, x1 = yx[:, 0].max() + 1, yx[:, 1].max() + 1
                    bboxes[b, i] = bboxes.new_tensor(
                        [x0.float(), y0.float(), x1.float(), y1.float()]
                    )
                else:
                    # Degenerate empty mask =fall back to full image extent
                    bboxes[b, i, 2] = float(W)
                    bboxes[b, i, 3] = float(H)
        return bboxes

    def _extract_masked_appearance(self, f_e, bboxes, exemplar_masks):
        bs, num_objects, _, _, _ = exemplar_masks.size()
        _, _, h, w = f_e.size()

        masks_f = F.interpolate(
            exemplar_masks.flatten(0, 1), size=(h, w),
            mode='bilinear', align_corners=False
        ).view(bs, num_objects, 1, h, w).clamp_(0, 1)

        roi_boxes = self._flatten_roi_boxes(bboxes)

        masked_features = (f_e.unsqueeze(1) * masks_f).flatten(0, 1)
        pooled_features = roi_align(
            masked_features,
            boxes=roi_boxes,
            output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction,
            aligned=True
        )
        pooled_masks = roi_align(
            masks_f.flatten(0, 1),
            boxes=roi_boxes,
            output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction,
            aligned=True
        )
        pooled_box_features = roi_align(
            f_e,
            boxes=roi_boxes,
            output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction,
            aligned=True
        )
        pooled_features = torch.where(
            pooled_masks > 1e-3,
            pooled_features / pooled_masks.clamp_min(1e-3),
            pooled_box_features
        )

        return pooled_features.permute(0, 2, 3, 1).reshape(
            bs, num_objects * self.kernel_dim ** 2, -1
        ).transpose(0, 1)

    def forward(
        self, f_e, pos_emb, bboxes=None, click_points=None, exemplar_masks=None,
        text_embeddings=None
    ):
        bs, _, h, w = f_e.size()
        # extract the shape features or objectness
        if not self.zero_shot:
            if exemplar_masks is not None and exemplar_masks.numel() > 0 and bboxes is not None:
                # click_sam mode WITH bboxes: use annotated box for shape prior,
                # SAM mask for denoised appearance.
                num_objects = bboxes.size(1)
                box_hw = torch.zeros(bs, num_objects, 2, device=bboxes.device)
                box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
                box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]
                appearance = self._extract_masked_appearance(
                    f_e, bboxes, exemplar_masks
                )
            elif exemplar_masks is not None and exemplar_masks.numel() > 0 and bboxes is None:
                # click-only mode: no bounding boxes provided.  Derive a tight
                # bbox from the SAM mask foreground so the shape MLP and
                # RoI-Align still have a meaningful spatial extent.
                num_objects = exemplar_masks.size(1)
                derived_bboxes = self._masks_to_bboxes(exemplar_masks)
                box_hw = torch.zeros(bs, num_objects, 2, device=derived_bboxes.device)
                box_hw[:, :, 0] = derived_bboxes[:, :, 2] - derived_bboxes[:, :, 0]
                box_hw[:, :, 1] = derived_bboxes[:, :, 3] - derived_bboxes[:, :, 1]
                appearance = self._extract_masked_appearance(
                    f_e, derived_bboxes, exemplar_masks
                )
            elif bboxes is not None:
                # box mode: plain RoI-Align, no SAM mask.
                num_objects = bboxes.size(1)
                box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
                box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
                box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]
                appearance = roi_align(
                    f_e,
                    boxes=self._flatten_roi_boxes(bboxes),
                    output_size=self.kernel_dim,
                    spatial_scale=1.0 / self.reduction,
                    aligned=True
                ).permute(0, 2, 3, 1).reshape(
                    bs, num_objects * self.kernel_dim ** 2, -1
                ).transpose(0, 1)
            else:
                raise ValueError(
                    "Few-shot ClickCount requires at least one of: exemplar bboxes "
                    "or SAM masks (click_sam mode)."
                )

            shape_or_objectness = self.shape_or_objectness(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)
        else:
            num_objects = self.num_objects
            shape_or_objectness = self.shape_or_objectness.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1)
            appearance = None

        query_pos_emb = self.pos_emb(
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(num_objects, 1, 1)

        if self.text_prototype_initializer is not None:
            shape_or_objectness, appearance = self.text_prototype_initializer(
                shape_or_objectness, appearance, text_embeddings
            )

        if self.num_iterative_steps > 0:
            memory = f_e.flatten(2).permute(2, 0, 1)
            all_prototypes = self.iterative_adaptation(
                shape_or_objectness, appearance, memory, pos_emb, query_pos_emb,
                text_embeddings=text_embeddings
            )
        else:
            if shape_or_objectness is not None and appearance is not None:
                all_prototypes = shape_or_objectness + appearance
            else:
                all_prototypes = (
                    shape_or_objectness if shape_or_objectness is not None else appearance
                )

            if self.text_conditioners is not None:
                all_prototypes = self.text_conditioners[0](
                    all_prototypes, text_embeddings
                )

            all_prototypes = all_prototypes.unsqueeze(0)

        text_loss = (
            text_alignment_loss(
                self.text_alignment_projector,
                all_prototypes[-1],
                text_embeddings
            )
            if self.text_alignment_projector is not None
            else all_prototypes.new_zeros(())
        )

        return all_prototypes, text_loss


class IterativeAdaptationModule(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
        text_conditioners=None
    ):

        super(IterativeAdaptationModule, self).__init__()

        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            ) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()
        object.__setattr__(
            self,
            "_text_cond_refs",
            list(text_conditioners) if text_conditioners is not None else None
        )

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None, text_embeddings=None
    ):

        output = tgt
        outputs = list()
        for i, layer in enumerate(self.layers):
            text_conditioner = None
            if self._text_cond_refs is not None:
                text_conditioner = self._text_cond_refs[i]

            output = layer(
                output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                text_conditioner=text_conditioner,
                text_embeddings=text_embeddings
            )
            outputs.append(self.norm(output))

        return torch.stack(outputs)


class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()

        self.norm_first = norm_first
        self.zero_shot = zero_shot

        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask, text_conditioner=None,
        text_embeddings=None
    ):
        if self.norm_first:
            if not self.zero_shot:
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            if text_conditioner is not None:
                # Text cross-attention between the visual refinement and FFN.
                tgt = text_conditioner(tgt, text_embeddings)
            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=self.with_emb(memory, pos_emb),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))
            if text_conditioner is not None:
                # Text cross-attention is between the visual refinement and FFN.
                tgt = text_conditioner(tgt, text_embeddings)

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt
