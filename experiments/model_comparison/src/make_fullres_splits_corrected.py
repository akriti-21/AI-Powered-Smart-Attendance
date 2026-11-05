from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

BASE_DIR = Path("experiments/model_comparison")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "lfw_split_fullres_corrected"

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Not overwriting it."
    )

lfw = fetch_lfw_people(
    data_home=str(DATA_DIR / "sklearn_cache"),
    color=True,
    resize=None,
    min_faces_per_person=20,
)

rng = np.random.RandomState(42)

for split in ("train", "val", "test"):
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

for class_id, person_name in enumerate(lfw.target_names):
    indices = np.where(lfw.target == class_id)[0]
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    folder_name = f"{class_id:03d}_{person_name}"

    for split, selected_indices in split_indices.items():
        person_dir = OUTPUT_DIR / split / folder_name
        person_dir.mkdir(parents=True, exist_ok=True)

        for file_index, image_index in enumerate(selected_indices):
            image = np.clip(
                lfw.images[image_index] * 255.0,
                0,
                255,
            ).round().astype(np.uint8)

            Image.fromarray(image).save(
                person_dir / f"{file_index:04d}.jpg",
                quality=95,
            )

print("Corrected dataset created:", OUTPUT_DIR)
print("Identities:", len(lfw.target_names))
print("Total images:", len(lfw.images))
print("Original image dimensions:", lfw.images.shape[1:3])
