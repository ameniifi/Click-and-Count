import numpy as np


def mask_to_box(mask):
    foreground = mask > 0.5
    if not np.any(foreground):
        return None

    ys, xs = np.where(foreground)
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1.0, ys.max() + 1.0],
        dtype=np.float32
    )


def box_iou_xyxy(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def select_sam_mask(masks, scores, bbox=None):
    scores = np.asarray(scores, dtype=np.float32)
    if bbox is None:
        return int(np.argmax(scores))

    bbox = np.asarray(bbox, dtype=np.float32)
    best_idx = int(np.argmax(scores))
    best_key = (-1.0, float(scores[best_idx]))

    for idx, mask in enumerate(masks):
        mask_box = mask_to_box(mask)
        if mask_box is None:
            candidate_key = (-1.0, float(scores[idx]))
        else:
            candidate_key = (
                box_iou_xyxy(mask_box, bbox),
                float(scores[idx])
            )

        if candidate_key > best_key:
            best_key = candidate_key
            best_idx = idx

    return best_idx


def predict_exemplar_mask(predictor, point, bbox=None):
    point_coords = np.asarray([point], dtype=np.float32)
    point_labels = np.asarray([1], dtype=np.int32)

    if bbox is not None:
        # PSeCo improves crowded-object segmentation by augmenting point prompts
        # with a box. For FSC147 exemplars we can use the annotated exemplar box
        # directly as that disambiguating prompt.
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.asarray(bbox, dtype=np.float32),
            multimask_output=False,
        )
        box_masks = np.asarray(masks)
        best_mask = np.asarray(
            box_masks[0] if box_masks.ndim == 3 else box_masks,
            dtype=np.float32
        )
        if np.any(best_mask > 0.5):
            return best_mask

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = select_sam_mask(masks, scores, bbox)
    return np.asarray(masks[best_idx], dtype=np.float32)
