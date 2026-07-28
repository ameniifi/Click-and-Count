import numpy as np

import torch
from torch import nn

from utils.sam_utils import predict_exemplar_mask

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:  # pragma: no cover - depends on optional package
    SamPredictor = None
    sam_model_registry = None


class SAMMaskGenerator(nn.Module):

    def __init__(self, checkpoint: str, model_type: str):

        super(SAMMaskGenerator, self).__init__()

        if SamPredictor is None or sam_model_registry is None:
            raise ImportError(
                "segment_anything is required for online click_sam prompting. "
                "Install the package or provide --sam_mask_dir with precomputed masks."
            )

        if not checkpoint:
            raise ValueError(
                "A SAM checkpoint is required for online click_sam prompting."
            )

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad_(False)
        self.predictor = SamPredictor(self.sam)

        self.register_buffer(
            'pixel_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False
        )

    @torch.no_grad()
    def forward(self, images, click_points, bboxes=None):
        if click_points is None:
            raise ValueError("click_points must be provided for SAM mask generation.")

        device = images.device
        images = (
            (images * self.pixel_std + self.pixel_mean)
            .clamp_(0, 1)
            .mul(255)
            .byte()
            .cpu()
        )
        click_points = click_points.cpu().numpy()
        bboxes = None if bboxes is None else bboxes.cpu().numpy()

        all_masks = list()
        for image_idx, (image, image_clicks) in enumerate(zip(images, click_points)):
            image_np = image.permute(1, 2, 0).numpy()
            self.predictor.set_image(image_np)
            image_bboxes = None if bboxes is None else bboxes[image_idx]

            image_masks = list()
            for object_idx, point in enumerate(image_clicks):
                exemplar_box = None if image_bboxes is None else image_bboxes[object_idx]
                best_mask = predict_exemplar_mask(self.predictor, point, exemplar_box)
                image_masks.append(torch.from_numpy(best_mask).unsqueeze(0))

            all_masks.append(torch.stack(image_masks, dim=0))

        return torch.stack(all_masks, dim=0).to(device=device)
