from pathlib import Path
import json
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lfw_split_fullres_corrected"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding(recognizer, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]
    scale = min(112 / w, 112 / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))

    resized = cv2.resize(image, (nw, nh))
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    x, y = (112 - nw) // 2, (112 - nh) // 2
    crop[y:y + nh, x:x + nw] = resized

    vector = recognizer.get_feat(crop).flatten()
    norm = np.linalg.norm(vector)

    if norm == 0 or not np.isfinite(norm):
        return None

    return vector / norm


def main():
    start = time.time()

    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Dataset folders not found in {DATA_DIR}")

    print("Loading Buffalo_L recognition model...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    recognizer = app.models["recognition"]

    train_embeddings, train_labels = [], []
    failures = 0

    print("\nExtracting training embeddings...")
    for folder in sorted(train_dir.iterdir()):
        if not folder.is_dir():
            continue

        for path in sorted(folder.glob("*.jpg")):
            embedding = get_embedding(recognizer, path)
            if embedding is None:
                failures += 1
                continue
            train_embeddings.append(embedding)
            train_labels.append(folder.name)

    if not train_embeddings:
        raise RuntimeError("No training embeddings were generated.")

    train_embeddings = np.asarray(train_embeddings)
    train_labels = np.asarray(train_labels)

    test_embeddings, test_labels = [], []
    test_failures = 0

    print("\nExtracting test embeddings...")
    test_paths = sorted(test_dir.glob("*/*.jpg"))

    for i, path in enumerate(test_paths, 1):
        embedding = get_embedding(recognizer, path)
        if embedding is None:
            test_failures += 1
            continue

        test_embeddings.append(embedding)
        test_labels.append(path.parent.name)

        if i % 50 == 0:
            print(f"Processed {i}/{len(test_paths)} test images")

    if not test_embeddings:
        raise RuntimeError("No test embeddings were generated.")

    test_embeddings = np.asarray(test_embeddings)
    test_labels = np.asarray(test_labels)

    print("\nMatching test embeddings against training embeddings...")
    similarities = test_embeddings @ train_embeddings.T
    predictions = train_labels[np.argmax(similarities, axis=1)]

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        labels=sorted(set(train_labels)),
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "InsightFace Buffalo_L ArcFace recognition",
        "dataset": "LFW 62 identities, corrected full-resolution crops",
        "method": "Nearest training-image embedding by cosine similarity",
        "preprocessing": "Aspect-ratio-preserving resize and padding to 112x112",
        "training_embeddings": len(train_labels),
        "test_images_available": len(test_paths),
        "test_images_evaluated": len(test_labels),
        "training_embedding_failures": failures,
        "test_embedding_failures": test_failures,
        "correct_predictions": int(np.sum(test_labels == predictions)),
        "test_accuracy_percent": round(float(accuracy * 100), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "elapsed_seconds": round(time.time() - start, 2),
        "warning": (
            "Diagnostic embedding-crop evaluation; not the full detection pipeline. "
            "Split membership may differ from the original CNN split."
        ),
    }

    output = RESULTS_DIR / "buffalo_l_fullres_results.json"
    output.write_text(json.dumps(results, indent=2))

    print("\n========== BUFFALO_L FULL EVALUATION ==========")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
