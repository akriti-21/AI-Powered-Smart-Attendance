
from pathlib import Path
from collections import defaultdict
import hashlib

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[3]
PHOTO_DIR = ROOT / "experiments/model_comparison/data/student_faces/enrollment"
YOLO_PATH = ROOT / "experiments/model_comparison/models/yolo_face/yolov11n-face.pt"

PHOTOS = [
    "STUDENT001.jpeg",
    "STUDENT001_extra2.jpeg",
    "STUDENT001_extra3.jpeg",
    "STUDENT001_extra4.jpeg",
    "STUDENT001_extra5.jpeg",
    "STUDENT002.jpeg",
    "STUDENT002_extra1.jpeg",
    "STUDENT002_extra2.jpeg",
    "STUDENT002_extra3.jpeg",
    "STUDENT002_extra4.jpeg",
    "STUDENT003.jpeg",
    "STUDENT003_extra2.jpeg",
]


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vector / norm


def main():
    print("YOLOv11 + ArcFace: per-detection diagnostic")
    print("Read-only: no production embeddings or attendance changes.")

    detector = YOLO(str(YOLO_PATH))
    arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    embeddings = defaultdict(list)
    labels = {}
    seen_hashes = {}
    valid_count = 0
    rejected_count = 0

    for filename in PHOTOS:
        path = PHOTO_DIR / filename
        if not path.is_file():
            print(f"\n{filename}: FILE MISSING")
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            print(f"\n{filename}: duplicate of {seen_hashes[digest]} — skipped")
            continue
        seen_hashes[digest] = filename

        image = cv2.imread(str(path))
        if image is None:
            print(f"\n{filename}: could not read image")
            continue

        label = path.stem.split("_")[0]
        result = detector.predict(
            source=image, imgsz=960, conf=0.25,
            device="cpu", verbose=False
        )[0]
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        print(f"\n{filename} (label={label}): YOLO detections={count}")

        if boxes is None:
            continue

        h, w = image.shape[:2]
        for i, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
            score = float(box.conf[0].item())
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                rejected_count += 1
                print(f"  Detection {i}: rejected invalid box")
                continue

            # Expand box to give the alignment detector surrounding context.
            px, py = int(bw * 0.25), int(bh * 0.25)
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(w, x2 + px), min(h, y2 + py)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                rejected_count += 1
                print(f"  Detection {i}: rejected empty crop")
                continue

            try:
                aligned = arc.get(crop)
            except Exception as exc:
                rejected_count += 1
                print(f"  Detection {i}: alignment error {type(exc).__name__}")
                continue

            if len(aligned) != 1:
                rejected_count += 1
                print(
                    f"  Detection {i}: rejected; "
                    f"Buffalo-L found {len(aligned)} faces (YOLO conf={score:.2f})"
                )
                continue

            vector = normalize(aligned[0].embedding)
            if vector is None or vector.shape != (512,):
                rejected_count += 1
                print(f"  Detection {i}: rejected invalid embedding")
                continue

            embeddings[label].append(vector)
            labels[label] = label
            valid_count += 1
            print(
                f"  Detection {i}: aligned, embedding=512-D, "
                f"YOLO conf={score:.2f}"
            )

    print("\n========== SUMMARY ==========")
    print(f"Valid aligned face detections: {valid_count}")
    print(f"Rejected detections: {rejected_count}")
    print("Valid detections grouped by source filename label:")
    for label in sorted(embeddings):
        print(f"  {label}: {len(embeddings[label])}")

    print(
        "\nImportant: this diagnostic groups detections by the image filename's "
        "label. That is valid only for photos containing the named student; "
        "background people must not be assigned that identity."
    )
    print("No production embeddings or attendance records were changed.")


if __name__ == "__main__":
    main()
