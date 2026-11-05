
from pathlib import Path
from collections import Counter
import json

from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading/loading LFW. This may take some time...")

lfw = fetch_lfw_people(
    data_home=str(DATA_DIR / "sklearn_cache"),
    color=True,
    resize=None,
    min_faces_per_person=20,
)

counts = Counter(lfw.target.tolist())

summary = {
    "dataset": "Labeled Faces in the Wild (LFW)",
    "images": int(len(lfw.images)),
    "identities": int(len(lfw.target_names)),
    "minimum_images_per_identity": 20,
    "image_shape": list(lfw.images.shape[1:]),
    "identity_counts": {
        str(lfw.target_names[i]): int(counts[i])
        for i in range(len(lfw.target_names))
    },
}

output = DATA_DIR / "lfw_summary.json"
output.write_text(json.dumps(summary, indent=2))

print(json.dumps({
    "dataset": summary["dataset"],
    "images": summary["images"],
    "identities": summary["identities"],
    "image_shape": summary["image_shape"],
    "summary_file": str(output),
}, indent=2))
