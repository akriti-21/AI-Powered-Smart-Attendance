import json
import time
from pathlib import Path

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
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

TRAIN_JSON = RESULTS / "wider_train_annotations.json"
VAL_JSON = RESULTS / "wider_val_annotations.json"

# Start small to test the complete training and evaluation pipeline.
TRAIN_LIMIT = 200
VAL_LIMIT = 100
EPOCHS = 1
IMAGE_SIZE = 512
SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_records(path, limit):
    records = json.loads(path.read_text())
    # Include negative images as well as images with usable face boxes.
    return records[:limit]


def load_sample(record):
    with Image.open(record["image"]) as im:
        image = im.convert("RGB")
        width, height = image.size
        image_tensor = to_tensor(image)

    boxes = []
    for x1, y1, x2, y2 in record["boxes"]:
        # Clip boxes to image boundaries.
        x1 = max(0.0, min(float(x1), float(width)))
        x2 = max(0.0, min(float(x2), float(width)))
        y1 = max(0.0, min(float(y1), float(height)))
        y2 = max(0.0, min(float(y2), float(height)))

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }
    return image_tensor, target


def build_model():
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        box_detections_per_img=300,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def calculate_ap50(all_predictions, all_targets):
    """Compute approximate AP50 using greedy IoU matching."""
    total_gt = sum(len(t) for t in all_targets)
    if total_gt == 0:
        return 0.0, 0.0, 0, 0

    detections = []
    for image_id, (pred, gt) in enumerate(
        zip(all_predictions, all_targets)
    ):
        scores = pred["scores"].cpu()
        boxes = pred["boxes"].cpu()
        keep = scores >= SCORE_THRESHOLD

        for box, score in zip(boxes[keep], scores[keep]):
            detections.append(
                (float(score), image_id, box)
            )

    detections.sort(key=lambda item: item[0], reverse=True)

    matched = [set() for _ in all_targets]
    tp = []
    fp = []

    for score, image_id, box in detections:
        gt = all_targets[image_id]
        if len(gt) == 0:
            tp.append(0)
            fp.append(1)
            continue

        overlaps = box_iou(box.reshape(1, 4), gt).squeeze(0)
        # Match only an as-yet-unmatched ground-truth box.
        order = torch.argsort(overlaps, descending=True)
        match = next(
            (
                int(j) for j in order
                if overlaps[j] >= IOU_THRESHOLD
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

    import numpy as np

    tp = np.cumsum(tp, dtype=float)
    fp = np.cumsum(fp, dtype=float)

    if len(tp) == 0:
        return 0.0, 0.0, 0, total_gt

    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    # All-points interpolated precision-recall AP approximation.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap50 = float(
        np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])
    )
    true_positives = int(tp[-1]) if len(tp) else 0
    recall_value = true_positives / total_gt

    return ap50, recall_value, true_positives, total_gt


def evaluate(model, records):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for index, record in enumerate(records, 1):
            image, target = load_sample(record)
            prediction = model([image.to(DEVICE)])[0]

            predictions.append({
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
            })
            targets.append(target["boxes"])

            if index % 20 == 0:
                print(f"  Validation images: {index}/{len(records)}")

    return calculate_ap50(predictions, targets)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    for path in (TRAIN_JSON, VAL_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")

    train_records = load_records(TRAIN_JSON, TRAIN_LIMIT)
    val_records = load_records(VAL_JSON, VAL_LIMIT)

    print("Device:", DEVICE)
    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Image size limit:", IMAGE_SIZE)
    print("Epochs:", EPOCHS)
    print("Loading detector...")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss_sum = 0.0
        successful_steps = 0
        start = time.time()

        for index, record in enumerate(train_records, 1):
            image, target = load_sample(record)
            target = {
                key: value.to(DEVICE)
                for key, value in target.items()
            }

            optimizer.zero_grad()
            losses = model([image.to(DEVICE)], [target])
            total_loss = sum(losses.values())

            if not torch.isfinite(total_loss):
                print(f"Skipping non-finite loss at image {index}")
                continue

            total_loss.backward()
            optimizer.step()

            total_loss_sum += float(total_loss.detach().cpu())
            successful_steps += 1

            if index % 20 == 0:
                avg = total_loss_sum / max(successful_steps, 1)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Image {index}/{len(train_records)} | "
                    f"Average loss: {avg:.4f}"
                )

        average_loss = total_loss_sum / max(successful_steps, 1)
        print(f"\nEpoch training loss: {average_loss:.4f}")
        print("Evaluating validation subset...")

        ap50, recall, tp, gt = evaluate(model, val_records)

        entry = {
            "epoch": epoch + 1,
            "train_average_loss": average_loss,
            "validation_subset_ap50_approx": ap50,
            "validation_subset_recall_at_50_iou": recall,
            "matched_ground_truth_faces": tp,
            "validation_ground_truth_faces": gt,
            "train_images": len(train_records),
            "validation_images": len(val_records),
            "score_threshold": SCORE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        history.append(entry)

        checkpoint_path = CHECKPOINTS / "fasterrcnn_wider_smoke_best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": entry,
            },
            checkpoint_path,
        )

        (RESULTS / "training_history.json").write_text(
            json.dumps(history, indent=2)
        )

        print("\nValidation results (small subset; not final benchmark):")
        print(f"Approximate AP50: {ap50:.4f}")
        print(f"Recall at IoU 0.50: {recall:.4f}")
        print(f"Matched faces: {tp}/{gt}")
        print("Checkpoint:", checkpoint_path)
        print("History: detection_experiment/results/training_history.json")
        print(f"Elapsed seconds: {entry['elapsed_seconds']:.1f}")

    print("\nSHORT TRAINING RUN COMPLETED.")


if __name__ == "__main__":
    main()
