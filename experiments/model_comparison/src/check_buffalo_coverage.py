
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

DATA_DIR = Path("experiments/model_comparison/data")
TRAIN_DIR = DATA_DIR / "lfw_split_fullres_corrected" / "train"
TEST_DIR = DATA_DIR / "lfw_split_fullres_corrected" / "test"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))

for split_name, split_dir in [
    ("train", TRAIN_DIR),
    ("test", TEST_DIR),
]:
    files = sorted(split_dir.glob("*/*.jpg"))
    detected = 0
    unreadable = 0

    for path in files:
        image = cv2.imread(str(path))

        if image is None:
            unreadable += 1
            continue

        faces = app.get(image)
        if faces:
            detected += 1

    total = len(files)
    percentage = 100 * detected / total if total else 0

    print(f"\n{split_name.upper()}")
    print("Total images:", total)
    print("Images with faces detected:", detected)
    print("Images without detections:", total - detected - unreadable)
    print("Unreadable images:", unreadable)
    print(f"Detection coverage: {percentage:.2f}%")

