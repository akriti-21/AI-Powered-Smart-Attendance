import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

ROOT = Path("detection_experiment")
ANNOTATIONS = ROOT / "results" / "wider_train_annotations.json"

# Conservative settings for the first MacBook test.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGE_SIZE = 512

def main():
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(f"Missing annotations: {ANNOTATIONS}")

    records = json.loads(ANNOTATIONS.read_text())
    record = next((r for r in records if r["boxes"]), None)

    if record is None:
        raise RuntimeError("No training image with valid boxes was found.")

    image_path = Path(record["image"])
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        image_tensor = to_tensor(image)

    scale_x = image_tensor.shape[2] / original_width
    scale_y = image_tensor.shape[1] / original_height

    boxes = []
    for x1, y1, x2, y2 in record["boxes"]:
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        raise RuntimeError("The selected image has no valid boxes after resizing.")

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=DEVICE),
        "labels": torch.ones(len(boxes), dtype=torch.int64, device=DEVICE),
    }

    print("Device:", DEVICE)
    print("Image:", image_path)
    print("Resized image:", tuple(image_tensor.shape))
    print("Valid face boxes:", len(boxes))
    print("Loading pretrained detector...")

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )

    image_tensor = image_tensor.to(DEVICE)
    optimizer.zero_grad()

    losses = model([image_tensor], [target])
    total_loss = sum(losses.values())

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite loss: {total_loss.item()}")

    total_loss.backward()
    optimizer.step()

    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    print(f"Total loss: {total_loss.item():.4f}")
    print("SMOKE TEST PASSED — one training batch completed.")
    print("No checkpoint was saved.")

if __name__ == "__main__":
    main()
