
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor

ROOT = Path("detection_experiment")
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

TRAIN_JSON = RESULTS / "wider_train_annotations.json"
VAL_JSON = RESULTS / "wider_val_annotations.json"

TRAIN_LIMIT = 1000
VAL_LIMIT = 200
EPOCHS = 5
IMAGE_SIZE = 512
SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

CHECKPOINT_PATH = CHECKPOINTS / "fasterrcnn_wider_improved_best.pth"
HISTORY_PATH = RESULTS / "training_history_improved.json"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_records(path, limit):
    records = json.loads(path.read_text())
    return records[:limit]


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
    total_gt = sum(len(target) for target in all_targets)
    if total_gt == 0:
        return 0.0, 0.0, 0, 0

    detections = []
    for image_id, prediction in enumerate(all_predictions):
        keep = prediction["scores"] >= SCORE_THRESHOLD
        for box, score in zip(
            prediction["boxes"][keep],
            prediction["scores"][keep],
        ):
            detections.append((float(score), image_id, box))

    detections.sort(key=lambda item: item[0], reverse=True)
    matched = [set() for _ in all_targets]
    tp_list, fp_list = [], []

    for _, image_id, box in detections:
        gt = all_targets[image_id]
        if len(gt) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue

        overlaps = box_iou(box.reshape(1, 4), gt).squeeze(0)
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
            tp_list.append(0)
            fp_list.append(1)
        else:
            matched[image_id].add(match)
            tp_list.append(1)
            fp_list.append(0)

    if not tp_list:
        return 0.0, 0.0, 0, total_gt

    tp = np.cumsum(tp_list, dtype=float)
    fp = np.cumsum(fp_list, dtype=float)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap50 = float(
        np.sum(
            (mrec[changes + 1] - mrec[changes])
            * mpre[changes + 1]
        )
    )
    return ap50, float(tp[-1] / total_gt), int(tp[-1]), total_gt


def evaluate(model, records):
    model.eval()
    predictions, targets = [], []

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
                print(f"  Validation: {index}/{len(records)}")

    return calculate_ap50(predictions, targets)


def main():
    for path in (TRAIN_JSON, VAL_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    train_records = load_records(TRAIN_JSON, TRAIN_LIMIT)
    val_records = load_records(VAL_JSON, VAL_LIMIT)

    missing = [
        r["image"] for r in train_records + val_records
        if not Path(r["image"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} annotation image paths are missing. "
            f"Example: {missing[0]}"
        )

    print("Device:", DEVICE)
    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Epochs:", EPOCHS)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Loading pretrained detector...")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    history = []
    best_ap50 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        steps = 0
        start = time.time()

        for index, record in enumerate(train_records, 1):
            image, target = load_sample(record)
            target = {key: value.to(DEVICE) for key, value in target.items()}

            optimizer.zero_grad(set_to_none=True)
            losses = model([image.to(DEVICE)], [target])
            loss = sum(losses.values())

            if not torch.isfinite(loss):
                print(f"Non-finite loss at image {index}; stopping.")
                raise RuntimeError("Training stopped due to non-finite loss.")

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1

            if index % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Image {index}/{len(train_records)} | "
                    f"Average loss: {loss_sum / steps:.4f}"
                )

        avg_loss = loss_sum / max(steps, 1)
        print(f"\nEpoch {epoch + 1}: average loss = {avg_loss:.4f}")
        print("Evaluating validation subset...")
        ap50, recall, tp, gt = evaluate(model, val_records)

        entry = {
            "epoch": epoch + 1,
            "train_average_loss": avg_loss,
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
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

        print(f"Approximate AP50: {ap50:.4f}")
        print(f"Recall: {recall:.4f} ({tp}/{gt})")

        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": entry,
                },
                CHECKPOINT_PATH,
            )
            print("Saved new best checkpoint:", CHECKPOINT_PATH)
        else:
            print("AP50 did not improve; best checkpoint preserved.")

        scheduler.step()
        print(f"Epoch time: {entry['elapsed_seconds']:.1f} seconds")

    print("\nTRAINING COMPLETED.")
    print("Best validation AP50:", best_ap50)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()

