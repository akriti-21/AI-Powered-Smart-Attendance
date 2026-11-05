from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "experiments/model_comparison/detection_experiment/data"
OUTPUT = ROOT / "experiments/model_comparison/yolo_face_finetune/dataset"

SPLITS = {
    "train": {
        "images": SOURCE / "WIDER_train/images",
        "annotations": SOURCE / "wider_face_split/wider_face_train_bbx_gt.txt",
    },
    "val": {
        "images": SOURCE / "WIDER_val/images",
        "annotations": SOURCE / "wider_face_split/wider_face_val_bbx_gt.txt",
    },
}


def parse_annotations(path):
    lines = path.read_text(errors="replace").splitlines()
    i = 0

    while i < len(lines):
        image_name = lines[i].strip()
        i += 1

        if not image_name:
            continue
        if i >= len(lines):
            break

        try:
            count = int(lines[i].strip())
        except ValueError:
            print(f"Warning: invalid face count after {image_name}")
            continue
        i += 1

        boxes = []
        for _ in range(max(count, 0)):
            if i >= len(lines):
                break
            values = lines[i].split()
            i += 1

            if len(values) < 4:
                continue

            try:
                x, y, w, h = map(float, values[:4])
            except ValueError:
                continue

            if w > 0 and h > 0:
                boxes.append((x, y, w, h))

        yield image_name, boxes


def convert_split(split, config):
    image_root = config["images"]
    annotation_file = config["annotations"]
    image_out = OUTPUT / "images" / split
    label_out = OUTPUT / "labels" / split

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    missing_count = 0
    invalid_count = 0

    for relative_name, boxes in parse_annotations(annotation_file):
        source_image = image_root / relative_name

        if not source_image.is_file():
            missing_count += 1
            continue

        try:
            import cv2
            image = cv2.imread(str(source_image))
            if image is None:
                missing_count += 1
                continue
            height, width = image.shape[:2]
        except Exception as exc:
            raise RuntimeError(f"Could not read {source_image}: {exc}") from exc

        if width <= 0 or height <= 0:
            invalid_count += 1
            continue

        # Preserve category directories and image basenames.
        relative_path = Path(relative_name)
        target_relative = relative_path.with_suffix(".jpg")
        target_image = image_out / target_relative
        target_label = label_out / target_relative.with_suffix(".txt")

        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_image, target_image)

        lines = []
        for x, y, w, h in boxes:
            x1 = max(0.0, min(x, width))
            y1 = max(0.0, min(y, height))
            x2 = max(0.0, min(x + w, width))
            y2 = max(0.0, min(y + h, height))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                invalid_count += 1
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            nw = bw / width
            nh = bh / height

            lines.append(
                f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
            )

        target_label.write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        image_count += 1
        label_count += len(lines)

    print(
        f"{split}: images={image_count}, boxes={label_count}, "
        f"missing={missing_count}, invalid_boxes={invalid_count}"
    )


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {OUTPUT}\n"
            "Move or clear only this experiment dataset before rerunning."
        )

    for split, config in SPLITS.items():
        if not config["images"].is_dir():
            raise FileNotFoundError(config["images"])
        if not config["annotations"].is_file():
            raise FileNotFoundError(config["annotations"])
        convert_split(split, config)

    yaml_path = OUTPUT / "wider_face.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: face\n"
    )
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
