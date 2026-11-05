"""Train AlexNet, EfficientNet-B0, and ResNet-50 toward 90% validation accuracy.

Uses the existing experiments/model_comparison/data/lfw_split/{train,val,test}
ImageFolder layout. Does not overwrite earlier checkpoints/results.
"""
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lfw_split"
OUT = ROOT / "finetune_results_90pct"
CKPT = OUT / "checkpoints"
TARGET_ACCURACY = 0.90
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32
SEED = 42
MODEL_NAMES = ["AlexNet", "EfficientNetB0", "ResNet50"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(DATA / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA / "val", transform=eval_tf)
test_ds = datasets.ImageFolder(DATA / "test", transform=eval_tf)
if not (train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx):
    raise ValueError("Train/validation/test class mappings do not match.")
NUM_CLASSES = len(train_ds.classes)


def make_model(name):
    if name == "AlexNet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        # Fine-tune the last convolutional block and the classifier.
        for p in model.features[10:].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        for idx in [5, 6, 7]:
            for p in model.features[idx].parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        for p in model.layer3.parameters():
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def evaluate(model, loader, criterion):
    model.eval()
    losses, targets, predictions = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses.append(criterion(logits, y).item())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def run(name):
    print(f"\n===== {name}: target validation accuracy {TARGET_ACCURACY:.0%} =====", flush=True)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = make_model(name).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("classifier.") or n.startswith("fc."))]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("classifier.") or n.startswith("fc."))]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-4})
    if head_params:
        groups.append({"params": head_params, "lr": 1e-4, "weight_decay": 1e-4})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    checkpoint_path = CKPT / f"{name.lower()}_90pct_best.pt"
    result_path = OUT / f"{name.lower()}_90pct_results.json"
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    best_state = None
    stale = 0
    first_target_epoch = None
    history = []
    started = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val = evaluate(model, val_loader, criterion)
        scheduler.step(val["loss"])
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        print(f"{name} epoch {epoch}/{MAX_EPOCHS} | train_loss={row['train_loss']:.4f} | val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.2%} | macro_f1={val['macro_f1']:.4f}", flush=True)

        improved = (val["accuracy"] > best_acc) or (val["accuracy"] == best_acc and val["loss"] < best_loss)
        if improved:
            best_acc, best_loss, best_epoch = val["accuracy"], val["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            stale = 0
        else:
            stale += 1

        if first_target_epoch is None and val["accuracy"] >= TARGET_ACCURACY:
            first_target_epoch = epoch
            print(f"*** {name} reached {TARGET_ACCURACY:.0%} validation accuracy at epoch {epoch}. Continuing to improve/save the best checkpoint. ***", flush=True)

        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}: no improvement for {PATIENCE} epochs.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was saved for {name}.")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, criterion)
    result = {
        "model": name,
        "target_validation_accuracy": TARGET_ACCURACY,
        "reached_target": best_acc >= TARGET_ACCURACY,
        "first_target_epoch": first_target_epoch,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_acc,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_loss": test["loss"],
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "training_seconds": round(time.time() - started, 2),
        "epochs_completed": len(history),
        "device": str(DEVICE),
        "checkpoint": str(checkpoint_path),
    }
    result_path.write_text(json.dumps({"result": result, "history": history}, indent=2))
    print("RESULT:\n" + json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    print("Device:", DEVICE)
    print("Dataset:", DATA)
    print("Classes:", NUM_CLASSES)
    print("Train/validation/test sizes:", len(train_ds), len(val_ds), len(test_ds))
    results = []
    for model_name in MODEL_NAMES:
        results.append(run(model_name))
    summary_path = OUT / "three_model_90pct_comparison.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nCombined comparison:", summary_path)
    print("Completed. Check 'reached_target' before comparing model sizes.")
