import time
import torch
from pathlib import Path
from torch import nn
from torchvision import models, datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

checkpoint = Path(
    "experiments/model_comparison/finetune_results_90pct/checkpoints/efficientnetb0_90pct_best.pt"
)
test_dir = "experiments/model_comparison/data/lfw_split/test"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
num_classes = state["classifier.1.weight"].shape[0]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(state)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

dataset = datasets.ImageFolder(test_dir, transform=transform)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=False, num_workers=0
)

y_true, y_pred = [], []
total_inference_seconds = 0.0
total_images = 0

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(device)

        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()

        outputs = model(images)

        if device.type == "mps":
            torch.mps.synchronize()
        total_inference_seconds += time.perf_counter() - start

        predictions = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.tolist())
        total_images += len(labels)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n===== EfficientNetB0 Test Evaluation =====")
print("Device:", device)
print("Test images:", len(dataset))
print("Classes:", len(dataset.classes))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro precision: {precision * 100:.2f}%")
print(f"Macro recall: {recall * 100:.2f}%")
print(f"Macro F1-score: {f1 * 100:.2f}%")
print(f"Total inference time: {total_inference_seconds:.3f} seconds")
print(f"Average inference time: {total_inference_seconds / total_images * 1000:.2f} ms/image")
print("==========================================")
