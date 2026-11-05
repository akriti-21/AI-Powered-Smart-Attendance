
import cv2
import re
from pathlib import Path

from ai.arcface_engine import FaceEngine

# One shared instance; avoid loading models on every request.
_engine = FaceEngine()

RECOGNITION_THRESHOLD = 0.50


def recognize_classroom_image(
    image_path: str,
    threshold: float = RECOGNITION_THRESHOLD,
) -> dict:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    return _engine.recognize_faces(image, threshold)


def register_student_face(
    registration_number: str,
    image_path: str,
):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    result = _engine.register_student(registration_number, image)
    _engine.save_embedding(registration_number, result["embedding"])

    return result


def delete_student_embedding(registration_number: str) -> bool:
    if not isinstance(registration_number, str):
        raise ValueError("Registration number must be a string")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", registration_number):
        raise ValueError("Invalid registration number")

    file_path = _engine.embedding_dir / f"{registration_number}.npy"

    if file_path.exists():
        file_path.unlink()
        return True

    return False
