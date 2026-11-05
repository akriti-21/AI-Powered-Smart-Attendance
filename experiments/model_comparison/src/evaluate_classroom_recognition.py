from pathlib import Path
import sys
import cv2
import numpy as np
from collections import defaultdict
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai.engine import FaceEngine

DATA = ROOT / "experiments/model_comparison/data/student_faces"
ENROLL = DATA / "enrollment"
CLASSROOM = DATA / "classroom_test"

# Use single-face photos only; do not use the held-out STUDENT002_extra3.
ENROLLMENT = {
    "STUDENT001": [
        "STUDENT001.jpeg",
        "STUDENT001_extra.jpeg",
        "STUDENT001_extra2.jpeg",
        "STUDENT001_extra4.jpeg",
    ],
    "STUDENT002": [
        "STUDENT002.jpeg",
        "STUDENT002_extra1.jpeg",
        "STUDENT002_extra2.jpeg",
        "STUDENT002_extra4.jpeg",
    ],
    "STUDENT003": [
        "STUDENT003.jpeg",
        "STUDENT003_extra2.jpeg",
    ],
}

TEST_IMAGES = ["classpic.jpg", "independent_test.png"]

def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else v

def make_prototypes(vectors):
    grouped = defaultdict(list)
    for student, vecs in vectors.items():
        grouped[student].extend(vecs)
    return {
        student: unit(np.mean(vecs, axis=0))
        for student, vecs in grouped.items() if vecs
    }

def match(embedding, prototypes):
    embedding = unit(embedding)
    scores = {
        student: float(np.dot(embedding, prototype))
        for student, prototype in prototypes.items()
    }
    if not scores:
        return "NO_ENROLLMENTS", 0.0, {}
    best = max(scores, key=scores.get)
    return best, scores[best], scores

def main():
    print("Loading EfficientNetB0 + YOLO production detector...")
    engine = FaceEngine()

    print("Loading diagnostic ArcFace detector/recognizer...")
    arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    efficient_vectors = defaultdict(list)
    arc_vectors = defaultdict(list)

    print("\nBuilding temporary prototypes from enrollment photos...")
    for student, filenames in ENROLLMENT.items():
        for filename in filenames:
            path = ENROLL / filename
            image = cv2.imread(str(path))
            if image is None:
                print(f"ENROLLMENT SKIP {filename}: unreadable")
                continue

            eff_faces = engine.detect_faces(image)
            arc_faces = arc.get(image)

            if len(eff_faces) != 1:
                print(f"ENROLLMENT SKIP {filename}: YOLO found {len(eff_faces)} faces")
                continue
            if len(arc_faces) != 1:
                print(f"ENROLLMENT SKIP {filename}: ArcFace found {len(arc_faces)} faces")
                continue

            efficient_vectors[student].append(unit(eff_faces[0].embedding))
            arc_vectors[student].append(unit(arc_faces[0].embedding))
            print(f"  {filename}: added for {student}")

    eff_proto = make_prototypes(efficient_vectors)
    arc_proto = make_prototypes(arc_vectors)

    print("\nEnrollment counts:")
    for student in sorted(ENROLLMENT):
        print(
            f"  {student}: EfficientNet={len(efficient_vectors[student])}, "
            f"ArcFace={len(arc_vectors[student])}"
        )

    print("\nClassroom image results")
    print("Predictions are candidates only; no attendance is recorded.")
    for filename in TEST_IMAGES:
        path = CLASSROOM / filename
        image = cv2.imread(str(path))
        if image is None:
            print(f"\n{filename}: ERROR reading image")
            continue

        eff_faces = engine.detect_faces(image)
        arc_faces = arc.get(image)

        print(f"\n=== {filename} | size={image.shape[1]}x{image.shape[0]} ===")
        print(f"YOLO/EfficientNet detected: {len(eff_faces)} faces")
        print(f"ArcFace pipeline detected: {len(arc_faces)} faces")

        print("\nEfficientNetB0 candidate matches (cosine similarity):")
        for i, face in enumerate(eff_faces, 1):
            label, score, scores = match(face.embedding, eff_proto)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            second = ranked[1][1] if len(ranked) > 1 else -1.0
            print(
                f"  face {i}: bbox={face.bbox.tolist()}, "
                f"candidate={label}, score={score:.3f}, "
                f"margin={score-second:.3f}"
            )

        print("\nArcFace candidate matches (cosine similarity):")
        for i, face in enumerate(arc_faces, 1):
            label, score, scores = match(face.embedding, arc_proto)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            second = ranked[1][1] if len(ranked) > 1 else -1.0
            bbox = np.asarray(face.bbox).astype(int).tolist()
            print(
                f"  face {i}: bbox={bbox}, candidate={label}, "
                f"score={score:.3f}, margin={score-second:.3f}"
            )

    print("\nDone. This script did not save embeddings, change checkpoints,")
    print("modify the database, or mark attendance.")
    print("Cosine scores are not calibrated probabilities. Inspect matches visually.")

if __name__ == "__main__":
    main()
