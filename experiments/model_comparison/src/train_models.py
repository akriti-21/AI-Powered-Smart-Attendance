
from pathlib import Path
import copy
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lfw_split"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

RESULTS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 25
BATCH_SIZE = 32
NUM_WORKERS = 0


torch.manual_seed(SEED)

if not torch.backends.mps.is_available():
    print("MPS unavailable; using CPU.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("mps")

print("Device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_ds = datasets.ImageFolder(DATA / "train", transform=transform)
val_ds = datasets.ImageFolder(DATA / "val", transform=transform)
test_ds = datasets.ImageFolder(DATA / "test", transform=transform)

if not (
    train_ds.class_to_idx == val_ds.class_to_idx
    == test_ds.class_to_idx
):
    raise RuntimeError("Class mappings differ between dataset splits.")

num_classes = len(train_ds.classes)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS,
)

def build_model(name):
    if name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head = model.classifier[1]

    elif name == "AlexNet":
        model = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT
        )
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        head = model.classifier[6]

    elif name == "ResNet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc

    else:
        raise ValueError(name)

    # Train only the new identity-classification layer.
    for p in head.parameters():
        p.requires_grad = True

    return model

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    predictions = []
    targets = []

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    accuracy = accuracy_score(targets, predictions)
    return total_loss / len(loader.dataset), accuracy, predictions, targets

def train_one(name):
    print(f"\n{'=' * 55}\nTraining {name}\n{'=' * 55}")
    model = build_model(name).to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion
        )

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"seconds={elapsed:.1f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint = CHECKPOINTS / f"{name.lower()}_best.pt"
    torch.save(model.state_dict(), checkpoint)

    # Evaluate the test set only after selecting the best validation checkpoint.
    test_loss, test_acc, predictions, targets = run_epoch(
        model, test_loader, criterion
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    result = {
        "model": name,
        "num_classes": num_classes,
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_images": len(test_ds),
        "epochs": EPOCHS,
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": test_acc,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1,
        "test_loss": test_loss,
        "checkpoint": str(checkpoint),
        "history": history,
    }

    result_file = RESULTS / f"{name.lower()}_results.json"
    result_file.write_text(json.dumps(result, indent=2))

    print(
        f"\n{name} TEST accuracy: {test_acc * 100:.2f}%\n"
        f"Macro precision: {precision:.4f}\n"
        f"Macro recall: {recall:.4f}\n"
        f"Macro F1: {f1:.4f}"
    )

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return result

def main():
    print(
        f"Classes: {num_classes} | "
        f"Train: {len(train_ds)} | "
        f"Validation: {len(val_ds)} | "
        f"Test: {len(test_ds)}"
    )

    results = []
    for name in ("EfficientNetB0", "AlexNet", "ResNet50"):
        try:
            results.append(train_one(name))
        except Exception as exc:
            print(f"\nERROR in {name}: {type(exc).__name__}: {exc}")
            print("Stopping so the error can be diagnosed.")
            break

    if results:
        with (RESULTS / "comparison.csv").open(
            "w", newline=""
        ) as f:
            fields = [
                "model", "test_accuracy", "test_precision_macro",
                "test_recall_macro", "test_f1_macro",
                "best_validation_accuracy", "test_images",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    key: r[key] for key in fields
                })

        print("\nCompleted model results:")
        for r in results:
            print(
                f"{r['model']}: "
                f"{r['test_accuracy'] * 100:.2f}% accuracy"
            )

        print("Comparison CSV:", RESULTS / "comparison.csv")

if __name__ == "__main__":
    main()

