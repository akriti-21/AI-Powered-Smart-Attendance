
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


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lfw_split"
OUT = ROOT / "finetune_results"
CKPT = OUT / "checkpoints"

EPOCHS = 50
PATIENCE = 9
BATCH_SIZE = 32
SEED = 42

MODEL_NAME = "EfficientNetB0_v2"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("Dataset:", DATA)

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.1,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASETS
# ============================================================

train_ds = datasets.ImageFolder(DATA / "train", train_tf)
val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
test_ds = datasets.ImageFolder(DATA / "test", eval_tf)

assert train_ds.class_to_idx == val_ds.class_to_idx, (
    "Training and validation class mappings do not match."
)
assert train_ds.class_to_idx == test_ds.class_to_idx, (
    "Training and test class mappings do not match."
)

NUM_CLASSES = len(train_ds.classes)


# ============================================================
# MODEL
# ============================================================

def make_model(name):
    if name != "EfficientNetB0_v2":
        raise ValueError(f"Unsupported model: {name}")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze the complete pretrained model first.
    for p in model.parameters():
        p.requires_grad = False

    # Replace the classification head for our 62 classes.
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES,
    )

    # Fine-tune later feature blocks.
    # Earlier pretrained feature blocks remain frozen.
    for layer_idx in [5, 6, 7]:
        for p in model.features[layer_idx].parameters():
            p.requires_grad = True

    # Ensure the new classification head is trainable.
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            losses.append(loss.item())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# TRAINING
# ============================================================

def run(name):
    print(f"\n===== {name} =====")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = make_model(name).to(DEVICE)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)
    print("Frozen parameters:", total - trainable)

    criterion = nn.CrossEntropyLoss()

    # Use different learning rates for the backbone and head.
    backbone_params = [
        p
        for param_name, p in model.named_parameters()
        if p.requires_grad
        and not param_name.startswith("classifier.")
    ]

    head_params = [
        p
        for param_name, p in model.named_parameters()
        if p.requires_grad
        and param_name.startswith("classifier.")
    ]

    optimizer = torch.optim.AdamW([
        {
            "params": backbone_params,
            "lr": 1e-5,
            "weight_decay": 1e-4,
        },
        {
            "params": head_params,
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_acc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    started = time.time()

    checkpoint_path = CKPT / "efficientnetb0_v2_finetuned_best.pt"
    results_path = OUT / "efficientnetb0_v2_finetuned.json"

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val = evaluate(model, val_loader, criterion)

        scheduler.step(val["accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }

        history.append(row)

        print(
            f"{name} epoch {epoch}/{EPOCHS} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4%} | "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

        # Save the best checkpoint using validation accuracy.
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            torch.save(best_state, checkpoint_path)

            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            print(
                "Early stopping: validation accuracy "
                "did not improve."
            )
            break

    # --------------------------------------------------------
    # Final evaluation on the test set
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    test = evaluate(model, test_loader, criterion)

    result = {
        "model": name,
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
    }

    # Save experiment-specific results.
    results_path.write_text(
        json.dumps(
            {
                "result": result,
                "history": history,
            },
            indent=2,
        )
    )

    # Save a separate one-model CSV.
    csv_path = OUT / "efficientnetb0_v2_comparison.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )
        writer.writeheader()
        writer.writerow(result)

    print("\nRESULT:")
    print(json.dumps(result, indent=2))
    print("\nCheckpoint:", checkpoint_path)
    print("Results:", results_path)

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Classes:", NUM_CLASSES)
    print(
        "Train/validation/test:",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    run(MODEL_NAME)

    print("\nEfficientNetB0_v2 experiment completed.")
