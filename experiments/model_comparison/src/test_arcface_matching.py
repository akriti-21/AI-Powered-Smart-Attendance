from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path("experiments/model_comparison")
DATA_DIR = ROOT / "data" / "lfw_split_fullres_corrected"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))

recognizer = app.models["recognition"]


def get_embedding(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Preserve the image's aspect ratio and pad it to 112 x 112.
    height, width = image.shape[:2]
    scale = min(112 / width, 112 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    x_offset = (112 - new_width) // 2
    y_offset = (112 - new_height) // 2

    crop[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    embedding = recognizer.get_feat(crop).flatten()
    norm = np.linalg.norm(embedding)

    if norm == 0 or not np.isfinite(norm):
        return None

    return embedding / norm


train_dir = DATA_DIR / "train"
test_dir = DATA_DIR / "test"

train_embeddings = []
train_labels = []

print("Extracting embeddings from a small training sample...")

for identity_dir in sorted(train_dir.iterdir()):
    if not identity_dir.is_dir():
        continue

    images = sorted(identity_dir.glob("*.jpg"))[:3]

    for image_path in images:
        embedding = get_embedding(image_path)

        if embedding is not None:
            train_embeddings.append(embedding)
            train_labels.append(identity_dir.name)

print("Training embeddings:", len(train_embeddings))

if not train_embeddings:
    raise RuntimeError("No training embeddings were generated.")

train_embeddings = np.asarray(train_embeddings)
train_labels = np.asarray(train_labels)

correct = 0
evaluated = 0

print("\nTesting the first 50 test images...")

test_images = sorted(test_dir.glob("*/*.jpg"))[:50]

for image_path in test_images:
    query_embedding = get_embedding(image_path)

    if query_embedding is None:
        continue

    similarities = train_embeddings @ query_embedding
    best_index = int(np.argmax(similarities))
    predicted = train_labels[best_index]
    actual = image_path.parent.name

    evaluated += 1
    correct += int(predicted == actual)

    print(
        f"{image_path.name}: "
        f"actual={actual}, predicted={predicted}, "
        f"similarity={similarities[best_index]:.4f}"
    )

print("\nSMALL-SAMPLE RESULTS")
print("Test images available:", len(test_images))
print("Test images evaluated:", evaluated)
print("Correct identity matches:", correct)

if evaluated:
    print(f"Top-1 identity accuracy: {100 * correct / evaluated:.2f}%")
else:
    print("Top-1 identity accuracy: unavailable")
