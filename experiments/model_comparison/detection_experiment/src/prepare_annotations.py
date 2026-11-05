import json
from pathlib import Path

BASE = Path("detection_experiment/data")
OUTPUT = Path("detection_experiment/results")


def parse_split(split):
    image_root = BASE / f"WIDER_{split}" / "images"
    annotation_file = (
        BASE / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    )

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_root}")
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    lines = annotation_file.read_text(errors="replace").splitlines()
    records = []
    i = 0
    total_boxes = 0
    rejected_boxes = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if not ("/" in line and line.lower().endswith(".jpg")):
            i += 1
            continue

        image_name = line
        i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            raise ValueError(f"Missing face count for {image_name}")

        try:
            face_count = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid face count for {image_name}: {lines[i]!r}"
            ) from exc

        i += 1
        boxes = []

        for _ in range(face_count):
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                raise ValueError(f"Unexpected end of annotations: {image_name}")

            parts = lines[i].split()
            i += 1
            total_boxes += 1

            try:
                x, y, w, h = map(float, parts[:4])
            except (ValueError, IndexError):
                rejected_boxes += 1
                continue

            if len(parts) < 4 or w <= 0 or h <= 0:
                rejected_boxes += 1
                continue

            # Faster R-CNN expects [xmin, ymin, xmax, ymax].
            boxes.append([x, y, x + w, y + h])

        image_path = image_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image missing: {image_path}")

        records.append({
            "image": str(image_path),
            "image_name": image_name,
            "boxes": boxes,
        })

    return records, total_boxes, rejected_boxes


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        records, total_boxes, rejected = parse_split(split)
        output_file = OUTPUT / f"wider_{split}_annotations.json"
        output_file.write_text(json.dumps(records))

        empty_images = sum(not record["boxes"] for record in records)

        print(f"\n{split.upper()}")
        print(f"Images: {len(records)}")
        print(f"Boxes read: {total_boxes}")
        print(f"Invalid boxes skipped: {rejected}")
        print(f"Images with no usable boxes: {empty_images}")
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
