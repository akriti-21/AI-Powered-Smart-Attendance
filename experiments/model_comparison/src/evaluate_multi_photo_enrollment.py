
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.engine import FaceEngine


DATA_DIR = (
    PROJECT_ROOT
    / "experiments/model_comparison/data/student_faces/enrollment"
)
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

STUDENTS = ["STUDENT001", "STUDENT002", "STUDENT003"]

# Temporary enrollment photos used only for this evaluation.
ENROLLMENT_PHOTOS = {
    "STUDENT001": [
        "STUDENT001_extra2.jpeg",
        "STUDENT001_extra3.jpeg",
        "STUDENT001_extra4.jpeg",
        "STUDENT001_extra5.jpeg",
    ],
    "STUDENT002": [
        "STUDENT002_extra2.jpeg",
        "STUDENT002_extra4.jpeg",
    ],
    "STUDENT003": [
        "STUDENT003_extra2.jpeg",
    ],
}

# Keep independent test images separate from enrollment.
TEST_PHOTOS = {
    "STUDENT002": [
        "STUDENT002_extra3.jpeg",
    ],
}

# Zero-based detection index.
# Face 1 was visually checked in both preview images.
VERIFIED_FACE_INDEX = {
    "STUDENT001_extra3.jpeg": 0,
    "STUDENT001_extra5.jpeg": 0,
}


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains invalid values")

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")

    return vector / norm


def cosine_similarity(a, b):
    return float(np.dot(normalize(a), normalize(b)))


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def main():
    engine = FaceEngine()

    # Load existing embeddings read-only.
    saved = {}

    for student in STUDENTS:
        path = EMBEDDING_DIR / f"{student}.npy"

        if not path.exists():
            raise FileNotFoundError(f"Missing embedding: {path}")

        saved[student] = normalize(
            np.load(path, allow_pickle=False)
        )

    # Build temporary prototypes in memory.
    prototypes = {
        student: [saved[student].copy()]
        for student in STUDENTS
    }

    print("\n=== Temporary enrollment ===")

    for student, filenames in ENROLLMENT_PHOTOS.items():
        for filename in filenames:
            path = DATA_DIR / filename

            if not path.exists():
                print(f"Skipping missing file: {filename}")
                continue

            image = load_image(path)
            faces = engine.detect_faces(image)

            if not faces:
                print(f"Skipping {filename}: no faces detected")
                continue

            if filename in VERIFIED_FACE_INDEX:
                face_index = VERIFIED_FACE_INDEX[filename]

                if face_index >= len(faces):
                    print(
                        f"Skipping {filename}: "
                        "verified face index not detected"
                    )
                    continue

                selected_face = faces[face_index]

            elif len(faces) == 1:
                selected_face = faces[0]

            else:
                print(
                    f"Skipping {filename}: "
                    f"{len(faces)} faces detected; "
                    "no verified face selection"
                )
                continue

            prototypes[student].append(
                normalize(selected_face.embedding)
            )

            print(f"Added temporary photo: {student} <- {filename}")

    # Average normalized vectors to create temporary prototypes.
    prototype_vectors = {}

    for student, vectors in prototypes.items():
        prototype_vectors[student] = normalize(
            np.mean(vectors, axis=0)
        )

    print("\n=== Independent-photo comparison ===")
    print("Scores are cosine similarities, not probabilities.")
    print("Existing embeddings and production files are unchanged.")

    tested_faces = 0
    correct_baseline = 0
    correct_prototype = 0

    for expected_student, filenames in TEST_PHOTOS.items():
        for filename in filenames:
            path = DATA_DIR / filename

            if not path.exists():
                print(f"\nMissing test image: {filename}")
                continue

            image = load_image(path)
            faces = engine.detect_faces(image)

            print(f"\n{filename}: {len(faces)} face(s)")

            if len(faces) != 1:
                print(
                    "Skipping test result: expected exactly one face "
                    "to avoid evaluating an ambiguous image."
                )
                continue

            query = normalize(faces[0].embedding)

            baseline_scores = {
                student: cosine_similarity(query, saved[student])
                for student in STUDENTS
            }

            prototype_scores = {
                student: cosine_similarity(
                    query, prototype_vectors[student]
                )
                for student in STUDENTS
            }

            baseline_best = max(
                baseline_scores, key=baseline_scores.get
            )

            prototype_best = max(
                prototype_scores, key=prototype_scores.get
            )

            tested_faces += 1
            correct_baseline += baseline_best == expected_student
            correct_prototype += prototype_best == expected_student

            print(f"Expected: {expected_student}")
            print(
                f"Single-photo prediction: {baseline_best} "
                f"({baseline_scores[baseline_best]:.4f})"
            )
            print(
                f"Multi-photo prediction:  {prototype_best} "
                f"({prototype_scores[prototype_best]:.4f})"
            )

            print("\nMulti-photo scores, highest first:")

            for student, score in sorted(
                prototype_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                marker = "  <-- expected" if student == expected_student else ""
                print(f"  {student}: {score:.4f}{marker}")

    print("\n=== Summary ===")

    if tested_faces:
        print(f"Independent test faces evaluated: {tested_faces}")
        print(
            "Single-photo correct predictions: "
            f"{correct_baseline}/{tested_faces}"
        )
        print(
            "Multi-photo correct predictions: "
            f"{correct_prototype}/{tested_faces}"
        )
    else:
        print("No valid independent test faces were evaluated.")

    print(
        "\nNote: This is a small diagnostic, not a reliable "
        "estimate of overall recognition accuracy."
    )


if __name__ == "__main__":
    main()
