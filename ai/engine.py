from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments/model_comparison/models/yolo_face/yolov11n-face.pt"
)

EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments/model_comparison/finetune_results_90pct/"
    "checkpoints/efficientnetb0_90pct_best.pt"
)

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

FEATURE_DIM = 1280
DETECTION_SIZE = 960
DETECTION_CONFIDENCE = 0.25


@dataclass
class DetectedFace:
    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float


class FaceEngine:
    def __init__(self):
        for path in (YOLO_MODEL_PATH, EFFICIENTNET_CHECKPOINT):
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Face detection
        self.detector = YOLO(str(YOLO_MODEL_PATH))

        # Load the existing 62-class LFW checkpoint first.
        # Then remove its LFW classification head and use its features.
        state = torch.load(
            EFFICIENTNET_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 62
        )
        self.model.load_state_dict(state)

        # Expose the 1280-dimensional feature vector.
        self.model.classifier[1] = nn.Identity()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.embedding_dir = EMBEDDING_DIR
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_registration_number(registration_number):
        if not registration_number:
            raise ValueError("Registration number is required")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", registration_number):
            raise ValueError("Invalid registration number format")

    def _extract_features(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Invalid face crop")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model(tensor)

        embedding = features.squeeze(0).float().cpu().numpy()
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Expected ({FEATURE_DIM},) features, got {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Feature vector contains invalid values")

        return embedding

    def detect_faces(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image")

        if image.size == 0:
            raise ValueError("Empty image")

        result = self.detector.predict(
            source=image,
            imgsz=DETECTION_SIZE,
            conf=DETECTION_CONFIDENCE,
            device="mps" if self.device.type == "mps" else "cpu",
            verbose=False,
        )[0]

        faces = []
        height, width = image.shape[:2]

        if result.boxes is None:
            return faces

        for box in result.boxes:
            x1, y1, x2, y2 = (
                box.xyxy[0].detach().cpu().numpy().astype(int)
            )

            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            embedding = self._extract_features(crop)

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

        if image is None:
            raise ValueError("Invalid image")

        faces = self.detect_faces(image)

        if len(faces) == 0:
            raise ValueError("No face detected in image")

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected. Registration requires exactly one face"
            )

        return {
            "registration_number": registration_number,
            "embedding": faces[0].embedding,
        }

    def save_embedding(self, registration_number, embedding):
        self._validate_registration_number(registration_number)

        if embedding is None:
            raise ValueError("Embedding is required")

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Embedding must have shape ({FEATURE_DIM},)"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains invalid values")

        if np.linalg.norm(embedding) == 0:
            raise ValueError("Invalid zero embedding")

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

        embedding = np.load(file_path, allow_pickle=False)

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Incompatible embedding for {registration_number}. "
                "Re-register this student using the new model."
            )

        return embedding.astype(np.float32, copy=False)

    def load_all_embeddings(self):
        embeddings = {}

        for file_path in self.embedding_dir.glob("*.npy"):
            try:
                embedding = np.load(file_path, allow_pickle=False)
            except (OSError, ValueError):
                continue

            # Ignore old 512-dimensional InsightFace embeddings.
            if embedding.shape != (FEATURE_DIM,):
                continue

            if not np.isfinite(embedding).all():
                continue

            if np.linalg.norm(embedding) == 0:
                continue

            embeddings[file_path.stem] = embedding.astype(
                np.float32, copy=False
            )

        return embeddings

    def recognize_face(self, embedding, threshold=0.5):
        if embedding is None:
            raise ValueError("Embedding is required")

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Embedding must have shape ({FEATURE_DIM},)"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains invalid values")

        embedding_norm = np.linalg.norm(embedding)

        if embedding_norm == 0:
            raise ValueError("Invalid zero embedding")

        registered = self.load_all_embeddings()

        if not registered:
            return {
                "registration_number": None,
                "confidence": 0.0,
                "status": "unknown",
            }

        best_registration_number = None
        best_similarity = -1.0

        for registration_number, stored_embedding in registered.items():
            stored_norm = np.linalg.norm(stored_embedding)

            if stored_norm == 0:
                continue

            similarity = float(
                np.dot(embedding, stored_embedding)
                / (embedding_norm * stored_norm)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_registration_number = registration_number

        if (
            best_registration_number is not None
            and best_similarity >= threshold
        ):
            return {
                "registration_number": best_registration_number,
                "confidence": best_similarity,
                "status": "recognized",
            }

        return {
            "registration_number": None,
            "confidence": max(0.0, best_similarity),
            "status": "unknown",
        }

    def recognize_faces(self, image, threshold=0.5):
        if image is None:
            raise ValueError("Invalid image")

        faces = self.detect_faces(image)
        recognized_students = []
        unknown_faces = []
        seen_students = set()

        for face in faces:
            result = self.recognize_face(face.embedding, threshold)

            if result["status"] == "recognized":
                student_id = result["registration_number"]

                if student_id not in seen_students:
                    seen_students.add(student_id)
                    recognized_students.append(result)
                else:
                    # Do not count a duplicate identity as another unknown.
                    continue
            else:
                unknown_faces.append(result)

        return {
            "face_count": len(faces),
            "recognized_students": recognized_students,
            "unknown_faces": unknown_faces,
        }

    def draw_results(self, image, threshold=0.5):
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
