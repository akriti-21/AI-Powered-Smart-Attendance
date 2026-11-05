from pathlib import Path
from collections import Counter
import json

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "lfw_split"
SEED = 42

print("Loading the filtered LFW dataset...")

lfw = fetch_lfw_people(
    data_home=str(DATA_DIR / "sklearn_cache"),
    color=True,
    resize=0.5,
    min_faces_per_person=20,
)

rng = np.random.default_rng(SEED)
counts = Counter(lfw.target.tolist())

# Stop rather than accidentally mixing with an older split.
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise SystemExit(
        f"{OUTPUT_DIR} already contains files. "
        "Inspect or rename it before creating new splits."
    )

for split in ("train", "val", "test"):
    for class_id, name in enumerate(lfw.target_names):
        (OUTPUT_DIR / split / f"{class_id:03d}_{name}").mkdir(
            parents=True, exist_ok=True
        )

summary = {
    "dataset": "LFW filtered subset",
    "seed": SEED,
    "minimum_images_per_identity": 20,
    "identities": int(len(lfw.target_names)),
    "splits": {},
}

for class_id, name in enumerate(lfw.target_names):
    indices = np.flatnonzero(lfw.target == class_id)
    rng.shuffle(indices)

    n = len(indices)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))

    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    for split, selected in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        for image_number, idx in enumerate(selected):
            image = lfw.images[idx]

            # Support either [0, 1] or [0, 255] image values.
            if image.max() <= 1.0:
                image = image * 255.0

            image = np.clip(image, 0, 255).astype(np.uint8)
            filename = (
                OUTPUT_DIR / split /
                f"{class_id:03d}_{name}" /
                f"{image_number:04d}.jpg"
            )
            Image.fromarray(image).save(filename, quality=95)

        summary["splits"].setdefault(split, 0)
        summary["splits"][split] += len(selected)

summary["images"] = sum(summary["splits"].values())
summary["identity_image_counts"] = {
    str(name): int(counts[i])
    for i, name in enumerate(lfw.target_names)
}

summary_path = DATA_DIR / "split_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(json.dumps({
    "images": summary["images"],
    "identities": summary["identities"],
    "split_counts": summary["splits"],
    "summary_file": str(summary_path),
}, indent=2))
