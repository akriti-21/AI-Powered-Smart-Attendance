from pathlib import Path
import sys
import cv2
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai.engine import FaceEngine

DATA = ROOT / "experiments/model_comparison/data/student_faces"
ENROLL = DATA / "enrollment"

PHOTOS = {
    "STUDENT001": [
        "STUDENT001.jpeg",
        "STUDENT001_extra.jpeg",
        "STUDENT001_extra2.jpeg",
        "STUDENT001_extra4.jpeg",
    ],
    "STUDENT002": [
        "STUDENT002.jpeg",
        "STUDENT002_extra2.jpeg",
        "STUDENT002_extra3.jpeg",
        "STUDENT002_extra4.jpeg",
    ],
    "STUDENT003": [
        "STUDENT003.jpeg",
        "STUDENT003_extra2.jpeg",
    ],
}

def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else v

def cosine_scores(query, prototypes):
    query = unit(query)
    return {
        label: float(np.dot(query, unit(proto)))
        for label, proto in prototypes.items()
    }

def evaluate(vectors, title):
    correct = 0
    total = 0
    print(f"\n--- {title} ---")
    for test_name, (true_label, query) in vectors.items():
        grouped = defaultdict(list)
        for name, (label, vector) in vectors.items():
            if name == test_name:
                continue
            grouped[label].append(unit(vector))

        prototypes = {
            label: unit(np.mean(items, axis=0))
            for label, items in grouped.items()
        }
        scores = cosine_scores(query, prototypes)
        predicted = max(scores, key=scores.get)
        correct += int(predicted == true_label)
        total += 1
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        margin = ranking[0][1] - ranking[1][1]
        print(
            f"{test_name}: actual={true_label}, predicted={predicted}, "
            f"{'CORRECT' if predicted == true_label else 'WRONG'}, "
            f"top={ranking[0][1]:.3f}, margin={margin:.3f}"
        )

    print(f"Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")

def main():
    engine = FaceEngine()
    vectors = {}

    print("Extracting fresh EfficientNetB0 features; no saved embeddings are used.")
    for label, names in PHOTOS.items():
        for name in names:
            path = ENROLL / name
            image = cv2.imread(str(path))
            if image is None:
                print(f"SKIP {name}: unreadable")
                continue

            faces = engine.detect_faces(image)
            if len(faces) != 1:
                print(f"SKIP {name}: {len(faces)} faces")
                continue

            vectors[name] = (label, faces[0].embedding)
            print(f"Loaded {name} ({label})")

    # Leave-one-photo-out testing: every test image is excluded from its prototype.
    evaluate(vectors, "Multi-photo prototypes, leave-one-photo-out")

    # Also report the fixed held-out STUDENT002_extra3 result separately.
    test_name = "STUDENT002_extra3.jpeg"
    if test_name in vectors:
        true_label, query = vectors[test_name]
        grouped = defaultdict(list)
        for name, (label, vector) in vectors.items():
            if name == test_name:
                continue
            grouped[label].append(unit(vector))
        prototypes = {
            label: unit(np.mean(items, axis=0))
            for label, items in grouped.items()
        }
        scores = cosine_scores(query, prototypes)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("\nHeld-out STUDENT002_extra3:")
        print("Actual:", true_label)
        print("Ranking:", ", ".join(f"{k}={v:.3f}" for k, v in ranked))

    print("\nRead-only experiment complete. Production files were not changed.")

if __name__ == "__main__":
    main()
