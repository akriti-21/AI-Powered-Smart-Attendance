
from pathlib import Path
import hashlib
import sys
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai.engine import FaceEngine

PHOTO_DIR = ROOT / "experiments/model_comparison/data/student_faces/enrollment"

# Exact duplicate identified by SHA-256:
# STUDENT001_extra.jpeg == STUDENT001.jpeg
EXCLUDED_DUPLICATES = {"STUDENT001_extra.jpeg"}

PHOTOS = [
    "STUDENT001.jpeg",
    "STUDENT001_extra2.jpeg",
    "STUDENT001_extra3.jpeg",
    "STUDENT001_extra4.jpeg",
    "STUDENT001_extra5.jpeg",
    "STUDENT002.jpeg",
    "STUDENT002_extra1.jpeg",
    "STUDENT002_extra2.jpeg",
    "STUDENT002_extra3.jpeg",
    "STUDENT002_extra4.jpeg",
    "STUDENT003.jpeg",
    "STUDENT003_extra2.jpeg",
]


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vector / norm


def label_for(filename):
    return Path(filename).stem.split("_")[0]


def main():
    print("Read-only student recognition comparison")
    print("No attendance endpoint will be called.")
    print("No registered embeddings will be saved or replaced.")

    engine = FaceEngine()

    arc = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    # Keep detection results separate. A failed detector must not
    # silently remove the image from both models.
    results = {
        "EfficientNetB0": {},
        "ArcFace": {},
    }
    failures = {
        "EfficientNetB0": {},
        "ArcFace": {},
    }
    labels = {}
    sha_seen = {}

    print("\nExtracting features from the same source photos...")

    for filename in PHOTOS:
        path = PHOTO_DIR / filename

        if not path.is_file():
            for model in results:
                failures[model][filename] = "file missing"
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in sha_seen:
            print(f"Duplicate excluded: {filename} == {sha_seen[digest]}")
            for model in results:
                failures[model][filename] = "duplicate image"
            continue
        sha_seen[digest] = filename

        image = cv2.imread(str(path))
        if image is None:
            for model in results:
                failures[model][filename] = "image could not be read"
            continue

        label = label_for(filename)
        labels[filename] = label

        # EfficientNet production detector + feature extractor.
        try:
            faces = engine.detect_faces(image)
            if len(faces) != 1:
                failures["EfficientNetB0"][filename] = (
                    f"detector found {len(faces)} faces"
                )
            else:
                vector = normalize(faces[0].embedding)
                if vector is None:
                    failures["EfficientNetB0"][filename] = "invalid embedding"
                else:
                    results["EfficientNetB0"][filename] = vector
        except Exception as exc:
            failures["EfficientNetB0"][filename] = (
                f"error: {type(exc).__name__}: {exc}"
            )

        # Buffalo-L detector + aligned ArcFace embedding.
        try:
            faces = arc.get(image)
            if len(faces) != 1:
                failures["ArcFace"][filename] = (
                    f"detector found {len(faces)} faces"
                )
            else:
                vector = normalize(faces[0].embedding)
                if vector is None:
                    failures["ArcFace"][filename] = "invalid embedding"
                else:
                    results["ArcFace"][filename] = vector
        except Exception as exc:
            failures["ArcFace"][filename] = (
                f"error: {type(exc).__name__}: {exc}"
            )

        print(
            f"{filename}: "
            f"EfficientNet={'OK' if filename in results['EfficientNetB0'] else 'FAILED'}, "
            f"ArcFace={'OK' if filename in results['ArcFace'] else 'FAILED'}"
        )

    # The primary comparison uses exactly the same images for both models.
    common = sorted(
        set(results["EfficientNetB0"]) & set(results["ArcFace"])
    )

    print("\n========== DETECTION SUMMARY ==========")
    print(f"Unique source photos requested: {len(PHOTOS)}")
    print(f"Common evaluable photos: {len(common)}")
    for model in results:
        print(f"\n{model}: usable={len(results[model])}")
        for filename, reason in sorted(failures[model].items()):
            print(f"  {filename}: {reason}")

    if not common:
        print("\nNo common photos could be evaluated.")
        return

    if len({labels[name] for name in common}) < 2:
        print("\nNot enough identities in common to compare models.")
        return

    print("\n========== COMMON-SET EVALUATION ==========")
    print("Leave-one-photo-out nearest identity prototype.")
    print("Treat results as preliminary; photos may be correlated.")

    for model, embeddings in results.items():
        correct = 0
        total = 0
        per_student = defaultdict(lambda: [0, 0])

        print(f"\n--- {model} ---")

        for test_name in common:
            query = embeddings[test_name]
            true_label = labels[test_name]

            grouped = defaultdict(list)
            for train_name in common:
                if train_name == test_name:
                    continue
                grouped[labels[train_name]].append(embeddings[train_name])

            prototypes = {}
            for student, vectors in grouped.items():
                prototype = normalize(np.mean(vectors, axis=0))
                if prototype is not None:
                    prototypes[student] = prototype

            if true_label not in prototypes or not prototypes:
                print(f"{test_name}: insufficient training examples")
                continue

            scores = {
                student: float(np.dot(query, prototype))
                for student, prototype in prototypes.items()
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            prediction, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else float("nan")
            margin = top_score - second_score

            is_correct = prediction == true_label
            correct += int(is_correct)
            total += 1
            per_student[true_label][1] += 1
            per_student[true_label][0] += int(is_correct)

            print(
                f"{test_name}: actual={true_label}, predicted={prediction}, "
                f"{'CORRECT' if is_correct else 'WRONG'}, "
                f"top={top_score:.3f}, margin={margin:.3f}"
            )

        accuracy = 100 * correct / total if total else 0
        print(f"Common-set accuracy: {correct}/{total} = {accuracy:.1f}%")
        print("Per-student accuracy:")
        for student in sorted(per_student):
            good, count = per_student[student]
            print(f"  {student}: {good}/{count}")

    print("\nDiagnostic complete. No production embeddings or attendance records changed.")


if __name__ == "__main__":
    main()
