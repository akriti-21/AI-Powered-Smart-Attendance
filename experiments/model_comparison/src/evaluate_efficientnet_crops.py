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

def crop_with_padding(image, bbox, padding):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - padding * bw))
    y1 = max(0, int(y1 - padding * bh))
    x2 = min(w, int(x2 + padding * bw))
    y2 = min(h, int(y2 + padding * bh))
    return image[y1:y2, x1:x2]

def extract_vector(engine, crop):
    # Reuse the exact EfficientNetB0 transform and weights from FaceEngine.
    return unit(engine._extract_features(crop))

def run_leave_one_out(vectors, title):
    correct, total = 0, 0
    print(f"\n--- {title} ---")
    for test_name, (actual, query) in vectors.items():
        grouped = defaultdict(list)
        for name, (label, vector) in vectors.items():
            if name != test_name:
                grouped[label].append(unit(vector))

        prototypes = {
            label: unit(np.mean(items, axis=0))
            for label, items in grouped.items()
        }
        scores = {
            label: float(np.dot(unit(query), proto))
            for label, proto in prototypes.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted = ranked[0][0]
        margin = ranked[0][1] - ranked[1][1]
        correct += int(predicted == actual)
        total += 1
        print(
            f"{test_name}: actual={actual}, predicted={predicted}, "
            f"{'CORRECT' if predicted == actual else 'WRONG'}, "
            f"top={ranked[0][1]:.3f}, margin={margin:.3f}"
        )
    print(f"Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")

def main():
    engine = FaceEngine()
    paddings = [0.00, 0.10, 0.20]
    results = {}

    print("Testing EfficientNetB0 crop padding without modifying production files.")
    for padding in paddings:
        vectors = {}
        print(f"\nPreparing crops with {padding:.0%} padding...")
        for label, filenames in PHOTOS.items():
            for filename in filenames:
                image = cv2.imread(str(ENROLL / filename))
                if image is None:
                    print("SKIP unreadable:", filename)
                    continue
                faces = engine.detect_faces(image)
                if len(faces) != 1:
                    print(f"SKIP {filename}: {len(faces)} faces")
                    continue
                crop = crop_with_padding(image, faces[0].bbox, padding)
                if crop.size == 0:
                    print("SKIP empty crop:", filename)
                    continue
                vectors[filename] = (label, extract_vector(engine, crop))
        results[padding] = vectors
        run_leave_one_out(vectors, f"Padding = {padding:.0%}")

    print("\nHeld-out photo comparison (STUDENT002_extra3.jpeg)")
    test_name = "STUDENT002_extra3.jpeg"
    for padding, vectors in results.items():
        if test_name not in vectors:
            continue
        actual, query = vectors[test_name]
        grouped = defaultdict(list)
        for name, (label, vector) in vectors.items():
            if name != test_name:
                grouped[label].append(unit(vector))
        prototypes = {
            label: unit(np.mean(items, axis=0))
            for label, items in grouped.items()
        }
        scores = {
            label: float(np.dot(unit(query), proto))
            for label, proto in prototypes.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(
            f"padding={padding:.0%}, actual={actual}, "
            + ", ".join(f"{label}={score:.3f}" for label, score in ranked)
        )

    print("\nRead-only experiment complete; no production files were modified.")

if __name__ == "__main__":
    main()
