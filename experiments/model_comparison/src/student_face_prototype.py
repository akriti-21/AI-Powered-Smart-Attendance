
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENROLLMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_comparison"
    / "data"
    / "student_faces"
    / "enrollment"
)

def main():
    print("=== Student Face Prototype: Image Check ===")
    print("Enrollment folder:", ENROLLMENT_DIR)

    if not ENROLLMENT_DIR.exists():
        raise FileNotFoundError(
            f"Enrollment folder not found: {ENROLLMENT_DIR}"
        )

    image_paths = sorted(
        path for path in ENROLLMENT_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        print("No enrollment images found.")
        return

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"FAILED: {path.name}")
            continue

        height, width = image.shape[:2]
        print(
            f"OK: {path.name} | "
            f"dimensions={width}x{height} | "
            f"channels={image.shape[2]}"
        )

    print("Image check complete.")
    print(
        "Note: this script checks images only; "
        "it does not recognize student identities yet."
    )

if __name__ == "__main__":
    main()
