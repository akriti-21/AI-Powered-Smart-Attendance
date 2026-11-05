
from pathlib import Path
import json
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, classification_report


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lfw_split"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "test")


def load_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def extract_embeddings(app, split):
    split_dir = DATA_DIR / split
    embeddings = []
    labels = []
    failures = 0

    identity_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir()
    )

    for identity_dir in identity_dirs:
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failures += 1
                print(f"Unreadable image: {image_path}")
                continue

            faces = app.get(image)

            if not faces:
                failures += 1
                print(f"No face detected: {image_path}")
                continue

            # Use the largest detected face in the image.
            face = max(
                faces,
                key=lambda item: (
                    item.bbox[2] - item.bbox[0]
                ) * (
                    item.bbox[3] - item.bbox[1]
                ),
            )

            vector = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)

            if norm == 0:
                failures += 1
                continue

            embeddings.append(vector / norm)
            labels.append(identity_dir.name)

    if not embeddings:
        raise RuntimeError(f"No usable embeddings extracted from {split}")

    return (
        np.stack(embeddings),
        np.asarray(labels),
        failures,
    )


def main():
    start = time.time()

    print("Loading buffalo_l...")
    app = load_model()

    print("\nExtracting training embeddings...")
    train_x, train_y, train_failures = extract_embeddings(app, "train")

    print("\nExtracting test embeddings...")
    test_x, test_y, test_failures = extract_embeddings(app, "test")

    # Each test image is classified by the most similar training image.
    # Embeddings are L2-normalized, so the dot product is cosine similarity.
    similarities = test_x @ train_x.T
    nearest_indices = np.argmax(similarities, axis=1)
    predictions = train_y[nearest_indices]

    accuracy = accuracy_score(test_y, predictions)
    report = classification_report(
        test_y,
        predictions,
        labels=sorted(set(train_y)),
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "InsightFace buffalo_l",
        "dataset": "LFW 62-identity closed-set split",
        "method": "Nearest training-image embedding by cosine similarity",
        "train_images_used": int(len(train_y)),
        "test_images_evaluated": int(len(test_y)),
        "train_face_detection_failures": int(train_failures),
        "test_face_detection_failures": int(test_failures),
        "correct_predictions": int(np.sum(test_y == predictions)),
        "test_accuracy": float(accuracy),
        "test_accuracy_percent": float(accuracy * 100),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    output_path = RESULTS_DIR / "buffalo_l_results.json"
    output_path.write_text(json.dumps(results, indent=2))

    print("\n========== BUFFALO_L RESULTS ==========")
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
