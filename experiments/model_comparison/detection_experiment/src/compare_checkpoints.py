
import json
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path("detection_experiment")
ANNOTATIONS = ROOT / "results" / "wider_val_annotations.json"
CHECKPOINTS = ROOT / "checkpoints"
OUTPUT = ROOT / "results" / "checkpoint_comparison.json"

IMAGE_SIZE = 512
SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
LIMIT = 200
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODELS = {
    "original_smoke": CHECKPOINTS / "fasterrcnn_wider_smoke_best.pth",
    "improved": CHECKPOINTS / "fasterrcnn_wider_improved_best.pth",
}


def build_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        box_detections_per_img=300,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def evaluate(model, records):
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for index, record in enumerate(records, 1):
            with Image.open(record["image"]) as im:
                image = to_tensor(im.convert("RGB"))
            pred = model([image.to(DEVICE)])[0]

            predictions.append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
            })

            boxes = []
            for x1, y1, x2, y2 in record["boxes"]:
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
            targets.append(
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            )

            if index % 20 == 0:
                print(f"  Evaluated {index}/{len(records)} images")

    total_gt = sum(len(t) for t in targets)
    detections = []

    for image_id, pred in enumerate(predictions):
        keep = pred["scores"] >= SCORE_THRESHOLD
        for box, score in zip(pred["boxes"][keep], pred["scores"][keep]):
            detections.append((float(score), image_id, box))

    detections.sort(key=lambda x: x[0], reverse=True)
    matched = [set() for _ in targets]
    tp, fp = [], []

    for _, image_id, box in detections:
        gt = targets[image_id]
        if len(gt) == 0:
            tp.append(0)
            fp.append(1)
            continue

        overlaps = box_iou(box.reshape(1, 4), gt).squeeze(0)
        order = torch.argsort(overlaps, descending=True)
        match = next(
            (int(j) for j in order
             if overlaps[j] >= IOU_THRESHOLD
             and int(j) not in matched[image_id]),
            None,
        )
        if match is None:
            tp.append(0)
            fp.append(1)
        else:
            matched[image_id].add(match)
            tp.append(1)
            fp.append(0)

    if total_gt == 0 or not tp:
        return {"approx_ap50": 0.0, "recall": 0.0,
                "matched_faces": 0, "ground_truth_faces": total_gt}

    import numpy as np
    tp_cum = np.cumsum(tp, dtype=float)
    fp_cum = np.cumsum(fp, dtype=float)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap50 = float(np.sum(
        (mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]
    ))

    return {
        "approx_ap50": ap50,
        "recall": float(tp_cum[-1] / total_gt),
        "matched_faces": int(tp_cum[-1]),
        "ground_truth_faces": total_gt,
    }


def main():
    records = json.loads(ANNOTATIONS.read_text())[:LIMIT]
    results = {}

    print("Device:", DEVICE)
    print("Validation images:", len(records))

    for name, path in MODELS.items():
        if not path.is_file():
            print(f"Skipping missing checkpoint: {path}")
            continue

        print(f"\nEvaluating {name}: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)

        results[name] = evaluate(model, records)
        results[name]["checkpoint_size_mb"] = round(
            path.stat().st_size / (1024 * 1024), 2
        )
        results[name]["parameter_count"] = sum(
            p.numel() for p in model.parameters()
        )
        print(results[name])

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    OUTPUT.write_text(json.dumps(results, indent=2))
    print("\nSaved comparison to:", OUTPUT)


if __name__ == "__main__":
    main()

