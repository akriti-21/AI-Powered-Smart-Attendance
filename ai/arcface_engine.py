from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments/model_comparison/models/yolo_face/yolov11n-face.pt"
)
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

FEATURE_DIM = 512
DETECTION_SIZE = 1600
DETECTION_CONFIDENCE = 0.15
CROP_EXPANSION = 0.25
MAX_DETECTIONS = 100
NMS_IOU_THRESHOLD = 0.7
RECOGNITION_THRESHOLD = 0.50


@dataclass
class DetectedFace:
    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float


class FaceEngine:
    def __init__(self):
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}"
            )

        self.detector = YOLO(str(YOLO_MODEL_PATH))

        self.arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.arcface.prepare(ctx_id=-1, det_size=(640, 640))

        self.embedding_dir = EMBEDDING_DIR
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_registration_number(registration_number):
        if not isinstance(registration_number, str):
            raise ValueError("Registration number must be a string")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", registration_number):
            raise ValueError("Invalid registration number")

    @staticmethod
    def _normalise_embedding(embedding):
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Embedding must have shape ({FEATURE_DIM},)"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains invalid values")

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Invalid zero embedding")

        return embedding / norm

    def detect_faces(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")

        height, width = image.shape[:2]

        prediction = self.detector.predict(
            source=image,
            imgsz=DETECTION_SIZE,
            conf=DETECTION_CONFIDENCE,
            device="cpu",
            max_det=MAX_DETECTIONS,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
        )[0]

        faces = []

        if prediction.boxes is None or len(prediction.boxes) == 0:
            return faces

        for box in prediction.boxes:
            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().astype(int)
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            ex = int(box_width * CROP_EXPANSION)
            ey = int(box_height * CROP_EXPANSION)

            cx1 = max(0, x1 - ex)
            cy1 = max(0, y1 - ey)
            cx2 = min(width, x2 + ex)
            cy2 = min(height, y2 + ey)

            crop = image[cy1:cy2, cx1:cx2]

            try:
                aligned_faces = self.arcface.get(crop)
            except Exception:
                continue

            # Use the largest face if the expanded crop contains
            # more than one face.
            if not aligned_faces:
                continue

            aligned_face = max(
                aligned_faces,
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1])
                ),
            )

            try:
                embedding = self._normalise_embedding(
                    aligned_face.embedding
                )
            except ValueError:
                continue

            faces.append(
                DetectedFace(
                    bbox=np.array([x1, y1, x2, y2]),
                    embedding=embedding,
                    det_score=float(box.conf[0].item()),
                )
            )

        return faces

    def get_embeddings(self, image):
        return [face.embedding for face in self.detect_faces(image)]

    def register_student(self, registration_number, image):
        self._validate_registration_number(registration_number)

        faces = self.detect_faces(image)

        if len(faces) == 0:
            raise ValueError(
                "No face detected. Use a clear, front-facing photo."
            )

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. Register one person at a time."
            )

        return {
            "registration_number": registration_number,
            "embedding": faces[0].embedding,
        }

    def save_embedding(self, registration_number, embedding):
        self._validate_registration_number(registration_number)
        embedding = self._normalise_embedding(embedding)

        file_path = self.embedding_dir / f"{registration_number}.npy"
        np.save(file_path, embedding)
        return file_path

    def load_embedding(self, registration_number):
        self._validate_registration_number(registration_number)

        file_path = self.embedding_dir / f"{registration_number}.npy"

        if not file_path.exists():
            raise FileNotFoundError(
                f"No embedding found for {registration_number}"
            )

        return self._normalise_embedding(
            np.load(file_path, allow_pickle=False)
        )

    def load_all_embeddings(self):
        embeddings = {}

        for file_path in self.embedding_dir.glob("*.npy"):
            registration_number = file_path.stem

            try:
                self._validate_registration_number(registration_number)
                embedding = np.load(
                    file_path, allow_pickle=False
                )
                embeddings[registration_number] = (
                    self._normalise_embedding(embedding)
                )
            except (OSError, ValueError):
                continue

        return embeddings

    def recognize_face(
        self, embedding, threshold=RECOGNITION_THRESHOLD
    ):
        embedding = self._normalise_embedding(embedding)
        registered = self.load_all_embeddings()

        if not registered:
            return {
                "registration_number": None,
                "confidence": 0.0,
                "status": "unknown",
                "best_match": None,
            }

        best_id = None
        best_similarity = -1.0

        for registration_number, stored in registered.items():
            similarity = float(np.dot(embedding, stored))

            if similarity > best_similarity:
                best_similarity = similarity
                best_id = registration_number

        if best_id is not None and best_similarity >= threshold:
            return {
                "registration_number": best_id,
                "confidence": best_similarity,
                "status": "recognized",
                "best_match": best_id,
            }

        return {
            "registration_number": None,
            "confidence": max(0.0, best_similarity),
            "status": "unknown",
            "best_match": best_id,
        }

    def recognize_faces(
        self, image, threshold=RECOGNITION_THRESHOLD
    ):
        faces = self.detect_faces(image)
        recognized_by_id = {}
        unknown_faces = []

        for face in faces:
            result = self.recognize_face(face.embedding, threshold)

            if result["status"] == "recognized":
                student_id = result["registration_number"]

                if (
                    student_id not in recognized_by_id
                    or result["confidence"]
                    > recognized_by_id[student_id]["confidence"]
                ):
                    recognized_by_id[student_id] = result
            else:
                unknown_faces.append(result)

        return {
            "face_count": len(faces),
            "recognized_students": list(recognized_by_id.values()),
            "unknown_faces": unknown_faces,
        }

    def draw_results(
        self, image, threshold=RECOGNITION_THRESHOLD
    ):
        if image is None:
            raise ValueError("Invalid image")

        output = image.copy()

        for face in self.detect_faces(image):
            result = self.recognize_face(face.embedding, threshold)
            x1, y1, x2, y2 = face.bbox.astype(int)

            if result["status"] == "recognized":
                label = (
                    f'{result["registration_number"]} '
                    f'{result["confidence"]:.2f}'
                )
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return output
