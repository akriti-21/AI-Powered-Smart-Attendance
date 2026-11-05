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
OUT = ROOT / "experiments/model_comparison/classroom_comparison"
OUT.mkdir(parents=True, exist_ok=True)

ENROLLMENT = {
    "STUDENT001": [
        "STUDENT001.jpeg", "STUDENT001_extra.jpeg",
        "STUDENT001_extra2.jpeg", "STUDENT001_extra4.jpeg",
    ],
    "STUDENT002": [
        "STUDENT002.jpeg", "STUDENT002_extra2.jpeg",
        "STUDENT002_extra4.jpeg",
    ],
    "STUDENT003": ["STUDENT003.jpeg", "STUDENT003_extra2.jpeg"],
}

def unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else v

def prototypes(vectors):
    grouped = defaultdict(list)
    for label, vecs in vectors.items():
        grouped[label].extend(vecs)
    return {
        label: unit(np.mean(items, axis=0))
        for label, items in grouped.items() if items
    }

def identify(embedding, proto, threshold, margin_threshold):
    query = unit(embedding)
    scores = sorted(
        [(label, float(np.dot(query, vector)))
         for label, vector in proto.items()],
        key=lambda item: item[1], reverse=True
    )
    if not scores:
        return "UNKNOWN", 0.0
    label, score = scores[0]
    margin = score - (scores[1][1] if len(scores) > 1 else -1)
    if score < threshold or margin < margin_threshold:
        return "UNKNOWN", score
    return label, score

def draw(image, faces, identify_fn, title):
    canvas = image.copy()
    for face in faces:
        label, score = identify_fn(face)
        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 170, 0) if label != "UNKNOWN" else (0, 165, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            canvas, text, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )
    cv2.putText(
        canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 0, 0), 2, cv2.LINE_AA
    )
    return canvas

def main():
    engine = FaceEngine()
    arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    arc.prepare(ctx_id=-1, det_size=(640, 640))

    eff_vectors = defaultdict(list)
    arc_vectors = defaultdict(list)

    for student, names in ENROLLMENT.items():
        for name in names:
            image = cv2.imread(str(ENROLL / name))
            if image is None:
                continue
            eff_faces = engine.detect_faces(image)
            arc_faces = arc.get(image)
            if len(eff_faces) == 1:
                eff_vectors[student].append(unit(eff_faces[0].embedding))
            if len(arc_faces) == 1:
                arc_vectors[student].append(unit(arc_faces[0].embedding))

    eff_proto = prototypes(eff_vectors)
    arc_proto = prototypes(arc_vectors)

    # These are conservative inspection settings, not validated production thresholds.
    for name in ["classpic.jpg", "independent_test.png"]:
        image = cv2.imread(str(CLASSROOM / name))
        if image is None:
            print("Could not read:", name)
            continue

        eff_faces = engine.detect_faces(image)
        arc_faces = arc.get(image)

        eff_img = draw(
            image, eff_faces,
            lambda f: identify(f.embedding, eff_proto, 0.50, 0.08),
            "EfficientNetB0 - tentative"
        )
        arc_img = draw(
            image, arc_faces,
            lambda f: identify(f.embedding, arc_proto, 0.50, 0.08),
            "ArcFace - tentative"
        )

        cv2.imwrite(str(OUT / f"{Path(name).stem}_efficientnet.jpg"), eff_img)
        cv2.imwrite(str(OUT / f"{Path(name).stem}_arcface.jpg"), arc_img)
        print(f"Saved annotated comparison for {name}")

    print("\nOutput folder:", OUT)
    print("Green = candidate match; orange = UNKNOWN.")
    print("Thresholds are for visual inspection only, not attendance decisions.")

if __name__ == "__main__":
    main()
