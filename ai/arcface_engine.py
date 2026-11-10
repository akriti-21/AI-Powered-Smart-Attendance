
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments/model_comparison/models/yolo_face/yolov11n-face.pt"
)

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

FEATURE_DIM = 512

DETECTION_SIZE = 1600
DETECTION_CONFIDENCE = 0.15
MAX_DETECTIONS = 100
NMS_IOU_THRESHOLD = 0.7

RECOGNITION_THRESHOLD = 0.50

# Minimum overlap required to associate YOLO and InsightFace boxes.
FACE_MATCH_IOU_THRESHOLD = 0.30


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class DetectedFace:
    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float


# ============================================================
# FACE ENGINE
# ============================================================

class FaceEngine:

    def __init__(self):

        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}"
            )

        print("Loading YOLO face detector...")

        self.detector = YOLO(str(YOLO_MODEL_PATH))

        print("Loading InsightFace and ArcFace...")

        self.arcface = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self.arcface.prepare(
            ctx_id=-1,
            det_size=(640, 640),
        )

        self.embedding_dir = EMBEDDING_DIR
        self.embedding_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("FaceEngine initialized successfully.")

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_registration_number(registration_number):

        if not isinstance(registration_number, str):
            raise ValueError(
                "Registration number must be a string"
            )

        if not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            registration_number,
        ):
            raise ValueError(
                "Invalid registration number"
            )

    @staticmethod
    def _normalise_embedding(embedding):

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Embedding must have shape ({FEATURE_DIM},)"
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Embedding contains invalid values"
            )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError(
                "Invalid zero embedding"
            )

        return embedding / norm

    # ========================================================
    # IOU CALCULATION
    # ========================================================

    @staticmethod
    def calculate_iou(box_a, box_b):

        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])

        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

        area_a = (
            (box_a[2] - box_a[0])
            * (box_a[3] - box_a[1])
        )

        area_b = (
            (box_b[2] - box_b[0])
            * (box_b[3] - box_b[1])
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces(self, image):

        if image is None or not isinstance(
            image,
            np.ndarray,
        ):
            raise ValueError("Invalid image")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Image must be a 3-channel BGR image"
            )

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # STAGE 1: YOLO DETECTION
        # ----------------------------------------------------

        prediction = self.detector.predict(
            source=image,
            imgsz=DETECTION_SIZE,
            conf=DETECTION_CONFIDENCE,
            device="cpu",
            max_det=MAX_DETECTIONS,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
        )[0]

        if (
            prediction.boxes is None
            or len(prediction.boxes) == 0
        ):
            return []

        yolo_boxes = []

        for box in prediction.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = (
                coordinates.astype(int)
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            yolo_boxes.append({
                "bbox": np.array(
                    [x1, y1, x2, y2],
                    dtype=np.int32,
                ),
                "confidence": float(
                    box.conf[0].item()
                ),
            })

        if not yolo_boxes:
            return []

        # ----------------------------------------------------
        # STAGE 2: INSIGHTFACE ON THE ORIGINAL IMAGE
        # ----------------------------------------------------

        # Important:
        # Do not run InsightFace on tiny YOLO crops.
        # It performs its own face detection and alignment
        # on the complete original image.

        insight_faces = self.arcface.get(image)

        if not insight_faces:
            return []

        # ----------------------------------------------------
        # STAGE 3: MATCH YOLO AND INSIGHTFACE DETECTIONS
        # ----------------------------------------------------

        matches = []

        for yolo_index, yolo_face in enumerate(yolo_boxes):

            for insight_index, insight_face in enumerate(
                insight_faces
            ):

                insight_box = (
                    insight_face.bbox.astype(int)
                )

                overlap = self.calculate_iou(
                    yolo_face["bbox"],
                    insight_box,
                )

                if overlap >= FACE_MATCH_IOU_THRESHOLD:

                    matches.append((
                        overlap,
                        yolo_index,
                        insight_index,
                    ))

        # Highest IoU matches are considered first.
        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # ----------------------------------------------------
        # STAGE 4: ONE-TO-ONE MATCHING
        # ----------------------------------------------------

        used_yolo = set()
        used_insight = set()

        detected_faces = []

        for overlap, yolo_index, insight_index in matches:

            if yolo_index in used_yolo:
                continue

            if insight_index in used_insight:
                continue

            yolo_face = yolo_boxes[yolo_index]

            insight_face = insight_faces[insight_index]

            try:
                embedding = self._normalise_embedding(
                    insight_face.embedding
                )

            except ValueError:
                continue

            detected_faces.append(
                DetectedFace(
                    bbox=yolo_face["bbox"],
                    embedding=embedding,
                    det_score=yolo_face["confidence"],
                )
            )

            used_yolo.add(yolo_index)
            used_insight.add(insight_index)

        return detected_faces

    # ========================================================
    # EMBEDDING EXTRACTION
    # ========================================================

    def get_embeddings(self, image):

        faces = self.detect_faces(image)

        return [
            face.embedding
            for face in faces
        ]

    # ========================================================
    # SAVE STUDENT EMBEDDING
    # ========================================================

    def save_embedding(
        self,
        registration_number,
        embedding,
    ):

        self._validate_registration_number(
            registration_number
        )

        embedding = self._normalise_embedding(
            embedding
        )

        file_path = (
            self.embedding_dir
            / f"{registration_number}.npy"
        )

        temporary_path = (
            self.embedding_dir
            / f"{registration_number}.tmp.npy"
        )

        try:
            np.save(
                temporary_path,
                embedding,
                allow_pickle=False,
            )

            temporary_path.replace(file_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        return file_path

    # ========================================================
    # LOAD ONE STUDENT EMBEDDING
    # ========================================================

    def load_embedding(self, registration_number):

        self._validate_registration_number(
            registration_number
        )

        file_path = (
            self.embedding_dir
            / f"{registration_number}.npy"
        )

        if not file_path.exists():
            return None

        embedding = np.load(
            file_path,
            allow_pickle=False,
        )

        return self._normalise_embedding(
            embedding
        )

    # ========================================================
    # LOAD ALL STUDENT EMBEDDINGS
    # ========================================================

    def load_all_embeddings(self):

        embeddings = {}

        for file_path in sorted(
            self.embedding_dir.glob("*.npy")
        ):

            registration_number = file_path.stem

            try:
                self._validate_registration_number(
                    registration_number
                )

                embedding = np.load(
                    file_path,
                    allow_pickle=False,
                )

                embeddings[registration_number] = (
                    self._normalise_embedding(
                        embedding
                    )
                )

            except (OSError, ValueError):
                continue

        return embeddings

    # ========================================================
    # STUDENT REGISTRATION
    # ========================================================

    def register_student(
        self,
        registration_number,
        images,
    ):

        self._validate_registration_number(
            registration_number
        )

        # Support one image or multiple images.
        if isinstance(images, np.ndarray):
            images = [images]

        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("Provide at least one enrollment image.")

        embeddings = []

        for index, image in enumerate(images):

            if image is None:
                raise ValueError(
                    f"Invalid image at position {index + 1}."
                )

            faces = self.detect_faces(image)

            if len(faces) == 0:
                raise ValueError(
                    f"No face detected in image {index + 1}."
                )

            if len(faces) > 1:
                raise ValueError(
                    f"Multiple faces detected in image {index + 1}."
                )

            embeddings.append(faces[0].embedding)

        # Average the normalized embeddings.
        mean_embedding = np.mean(
            np.stack(embeddings),
            axis=0,
        )

        final_embedding = self._normalise_embedding(
            mean_embedding
        )

        # Preserve existing .npy storage.
        self.save_embedding(
            registration_number,
            final_embedding,
        )

        print(
            f"Registered {registration_number} "
            f"using {len(embeddings)} image(s)."
        )

        return final_embedding

    # RECOGNIZE ONE FACE
    # ========================================================

    def recognize_face(
        self,
        embedding,
        threshold=RECOGNITION_THRESHOLD,
        registered=None,
    ):

        embedding = self._normalise_embedding(
            embedding
        )

        if registered is None:
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

            similarity = float(
                np.dot(embedding, stored)
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_id = registration_number

        if (
            best_id is not None
            and best_similarity >= threshold
        ):

            return {
                "registration_number": best_id,
                "confidence": best_similarity,
                "status": "recognized",
                "best_match": best_id,
            }

        return {
            "registration_number": None,
            "confidence": max(
                0.0,
                best_similarity,
            ),
            "status": "unknown",
            "best_match": best_id,
        }

    # ========================================================
    # RECOGNIZE MULTIPLE FACES
    # ========================================================

    
    def recognize_faces(
        self,
        image,
        threshold=RECOGNITION_THRESHOLD,
        registered=None,
    ):
        faces = self.detect_faces(image)

        # Use all embeddings only when no restricted list is supplied.
        # Attendance will pass a class-specific dictionary.
        if registered is None:
            registered = self.load_all_embeddings()

        recognized_by_id = {}
        unknown_faces = []

        for face in faces:
            result = self.recognize_face(
                face.embedding,
                threshold,
                registered=registered,
            )

            result["bbox"] = face.bbox.tolist()
            result["detection_confidence"] = face.det_score

            if result["status"] == "recognized":
                student_id = result["registration_number"]

                # Keep the highest-confidence detection for each student.
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


    # ========================================================
    # DRAW RECOGNITION RESULTS
    # ========================================================

    def draw_results(
        self,
        image,
        threshold=RECOGNITION_THRESHOLD,
    ):

        if image is None:
            raise ValueError("Invalid image")

        output = image.copy()

        faces = self.detect_faces(image)

        registered = self.load_all_embeddings()

        for face in faces:

            result = self.recognize_face(
                face.embedding,
                threshold,
                registered=registered,
            )

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            if result["status"] == "recognized":

                label = (
                    f'{result["registration_number"]} '
                    f'{result["confidence"]:.2f}'
                )

                color = (0, 255, 0)

            else:

                label = (
                    f'Unknown {result["confidence"]:.2f}'
                )

                color = (0, 0, 255)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return output

