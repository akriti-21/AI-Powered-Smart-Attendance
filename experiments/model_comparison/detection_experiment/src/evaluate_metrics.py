import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchvision.ops import box_iou

ROOT = Path("detection_experiment")
CHECKPOINT = ROOT / "checkpoints" / "fasterrcnn_wider_smoke_best.pth"
ANNOTATIONS = ROOT / "results" / "wider_val_annotations.json"
OUTPUT = ROOT / "results" / "evaluation_metrics.json"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
IMAGE_SIZE = 512
RECALL_SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300


def build_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        box_detections_per_img=MAX_DETECTIONS,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def load_sample(record):
    with Image.open(record["image"]) as im:
        image = im.convert("RGB")
        width, height = image.size
        image_tensor = to_tensor(image)

    boxes = []
    for x1, y1, x2, y2 in record["boxes"]:
        x1 = max(0.0, min(float(x1), float(width)))
        x2 = max(0.0, min(float(x2), float(width)))
        y1 = max(0.0, min(float(y1), float(height)))
        y2 = max(0.0, min(float(y2), float(height)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return image_tensor, torch.tensor(
        boxes, dtype=torch.float32
    ).reshape(-1, 4)


def compute_ap50(predictions, targets):
    total_gt = sum(len(t) for t in targets)
    if total_gt == 0:
        return None

    detections = []
    for image_id, pred in enumerate(predictions):
        for box, score in zip(pred["boxes"], pred["scores"]):
            detections.append((float(score), image_id, box))

    detections.sort(key=lambda x: x[0], reverse=True)

    matched = [set() for _ in targets]
    tp = []
    fp = []

    for score, image_id, box in detections:
        gt = targets[image_id]
        if len(gt) == 0:
            tp.append(0)
            fp.append(1)
            continue

        overlaps = box_iou(box.reshape(1, 4), gt).squeeze(0)
        order = torch.argsort(overlaps, descending=True)
        match = next(
            (
                int(j) for j in order
                if float(overlaps[j]) >= IOU_THRESHOLD
                and int(j) not in matched[image_id]
            ),
            None,
        )

        if match is None:
            tp.append(0)
            fp.append(1)
        else:
            matched[image_id].add(match)
            tp.append(1)
            fp.append(0)

    if not tp:
        return 0.0

    tp = np.cumsum(tp, dtype=float)
    fp = np.cumsum(fp, dtype=float)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    # VOC-style all-points interpolated AP at IoU 0.50.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum(
            (mrec[changes + 1] - mrec[changes])
            * mpre[changes + 1]
        )
    )


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(ANNOTATIONS)

    records = json.loads(ANNOTATIONS.read_text())[:100]
    model = build_model()

    saved = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(saved["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []
    true_positives = 0
    total_gt = 0
    total_predictions_at_threshold = 0

    print("Device:", DEVICE)
    print("Validation images:", len(records))
    print("Evaluating saved checkpoint...")

    with torch.no_grad():
        for i, record in enumerate(records, 1):
            image, gt = load_sample(record)
            pred = model([image.to(DEVICE)])[0]

            boxes = pred["boxes"].detach().cpu()
            scores = pred["scores"].detach().cpu()

            predictions.append({"boxes": boxes, "scores": scores})
            targets.append(gt)
            total_gt += len(gt)

            keep = scores >= RECALL_SCORE_THRESHOLD
            kept_boxes = boxes[keep]
            total_predictions_at_threshold += len(kept_boxes)

            if len(gt) > 0 and len(kept_boxes) > 0:
                overlaps = box_iou(kept_boxes, gt)
                # Greedy matching for recall at the chosen threshold.
                pairs = []
                for pi in range(overlaps.shape[0]):
                    for gi in range(overlaps.shape[1]):
                        if float(overlaps[pi, gi]) >= IOU_THRESHOLD:
                            pairs.append(
                                (float(overlaps[pi, gi]), pi, gi)
                            )
                pairs.sort(reverse=True)

                used_p = set()
                used_g = set()
                for _, pi, gi in pairs:
                    if pi not in used_p and gi not in used_g:
                        used_p.add(pi)
                        used_g.add(gi)
                true_positives += len(used_g)

            if i % 20 == 0:
                print(f"  Evaluated {i}/{len(records)} images")

    ap50 = compute_ap50(predictions, targets)
    recall = true_positives / total_gt if total_gt else 0.0

    result = {
        "checkpoint": str(CHECKPOINT),
        "validation_images": len(records),
        "ground_truth_faces": total_gt,
        "ap50_voc_style_all_scores": ap50,
        "recall_at_iou_0_50_and_score_0_50": recall,
        "matched_ground_truth_faces_at_threshold": true_positives,
        "predictions_above_score_threshold": total_predictions_at_threshold,
        "iou_threshold": IOU_THRESHOLD,
        "recall_score_threshold": RECALL_SCORE_THRESHOLD,
        "max_detections_per_image": MAX_DETECTIONS,
        "note": (
            "Subset evaluation; AP50 uses VOC-style interpolation, "
            "not the official WIDER FACE benchmark protocol."
        ),
    }

    OUTPUT.write_text(json.dumps(result, indent=2))
    print("\nEvaluation results:")
    print(json.dumps(result, indent=2))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
