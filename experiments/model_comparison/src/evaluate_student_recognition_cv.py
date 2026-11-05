from pathlib import Path
import sys
import cv2
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai.engine import FaceEngine
from insightface.app import FaceAnalysis

PHOTO_DIR = ROOT / "experiments/model_comparison/data/student_faces/enrollment"

PHOTOS = [
    "STUDENT001.jpeg",
    "STUDENT001_extra.jpeg",
    "STUDENT001_extra2.jpeg",
    "STUDENT001_extra4.jpeg",
    "STUDENT002.jpeg",
    "STUDENT002_extra1.jpeg",
    "STUDENT002_extra2.jpeg",
    "STUDENT002_extra3.jpeg",
    "STUDENT002_extra4.jpeg",
    "STUDENT003.jpeg",
    "STUDENT003_extra2.jpeg",
]

def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def predict(query, prototypes):
    scores = {
        label: float(np.dot(query, prototype))
        for label, prototype in prototypes.items()
    }
    winner = max(scores, key=scores.get)
    return winner, scores[winner], scores

def main():
    print("Loading current EfficientNetB0 engine...")
    engine = FaceEngine()

    print("Loading diagnostic ArcFace model...")
    arc = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    vectors = {"EfficientNetB0": {}, "ArcFace": {}}
    labels = {}
    skipped = []

    for filename in PHOTOS:
        path = PHOTO_DIR / filename
        image = cv2.imread(str(path))
        if image is None:
            skipped.append((filename, "image could not be read"))
            continue

        label = path.stem.split("_")[0]

        # EfficientNet features from the existing production pipeline.
        detections = engine.detect_faces(image)
        if len(detections) != 1:
            skipped.append((filename, f"EfficientNet detector found {len(detections)} faces"))
            continue

        eff = getattr(detections[0], "embedding", None)
        if eff is None:
            skipped.append((filename, "EfficientNet embedding missing"))
            continue

        # ArcFace uses its own face detector and aligned face crop.
        arc_faces = arc.get(image)
        if len(arc_faces) != 1:
            skipped.append((filename, f"ArcFace detector found {len(arc_faces)} faces"))
            continue

        arc_emb = getattr(arc_faces[0], "embedding", None)
        if arc_emb is None:
            skipped.append((filename, "ArcFace embedding missing"))
            continue

        vectors["EfficientNetB0"][filename] = normalize(eff)
        vectors["ArcFace"][filename] = normalize(arc_emb)
        labels[filename] = label
        print(f"Prepared {filename}: {label}")

    print("\nLeave-one-photo-out evaluation")
    print("Each test photo is excluded from its own identity prototype.")

    for model_name, model_vectors in vectors.items():
        correct = 0
        total = 0
        print(f"\n--- {model_name} ---")

        for test_name, query in model_vectors.items():
            true_label = labels[test_name]
            grouped = defaultdict(list)

            for train_name, vector in model_vectors.items():
                if train_name == test_name:
                    continue
                grouped[labels[train_name]].append(vector)

            prototypes = {
                label: normalize(np.mean(items, axis=0))
                for label, items in grouped.items()
            }

            prediction, score, scores = predict(query, prototypes)
            total += 1
            correct += int(prediction == true_label)

            detail = " | ".join(
                f"{label}={value:.3f}"
                for label, value in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )
            )
            print(
                f"{test_name}: actual={true_label}, predicted={prediction}, "
                f"{'CORRECT' if prediction == true_label else 'WRONG'}, "
                f"top_score={score:.3f} | {detail}"
            )

        if total:
            print(f"Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")
        else:
            print("No usable photos for evaluation.")

    if skipped:
        print("\nSkipped photos:")
        for filename, reason in skipped:
            print(f"  {filename}: {reason}")

    print("\nDiagnostic only: no production embeddings or checkpoints were changed.")
    print("Results depend on the filenames being correctly labeled.")

if __name__ == "__main__":
    main()
