"""
Smart Attendance: Model Comparison Dashboard
Run from the project root:
    ./detection-venv/bin/python experiments/model_comparison/src/display_model_comparison.py
Outputs a PNG dashboard and a text summary in finetune_results_90pct/.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune_results_90pct"
CSV_PATH = OUT / "three_model_90pct_comparison.csv"

# Fallback values from the completed experiment, used if the CSV is unavailable
FALLBACK = [
    {"model":"AlexNet","best_validation_accuracy":0.7361419069,"test_accuracy":0.6984478936,"test_macro_f1":0.5825899716,"total_parameters":57257854,"trainable_parameters":55378238,"training_seconds":722.99},
    {"model":"EfficientNetB0","best_validation_accuracy":0.8691796009,"test_accuracy":0.8026607539,"test_macro_f1":0.7100151836,"total_parameters":4086970,"trainable_parameters":3366150,"training_seconds":2666.32},
    {"model":"ResNet-50","best_validation_accuracy":0.8314855876,"test_accuracy":0.8159645233,"test_macro_f1":0.7182767454,"total_parameters":23635070,"trainable_parameters":22190142,"training_seconds":4841.60},
]

def load_results():
    if not CSV_PATH.exists():
        print(f"WARNING: CSV not found; using recorded values: {CSV_PATH}")
        return FALLBACK
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    results = []
    for row in rows:
        name = " ".join(str(v) for k, v in row.items() if k.lower() in {"model", "model_name", "name", "checkpoint"}).lower()
        if "alex" in name:
            label = "AlexNet"
        elif "efficient" in name:
            label = "EfficientNetB0"
        elif "resnet" in name:
            label = "ResNet-50"
        else:
            continue
        base = next(x for x in FALLBACK if x["model"] == label)
        def num(keys, default):
            for key in keys:
                if row.get(key) not in (None, ""):
                    try:
                        return float(row[key])
                    except ValueError:
                        pass
            return default
        results.append({
            "model": label,
            "best_validation_accuracy": num(["best_validation_accuracy","validation_accuracy","best_val_accuracy"], base["best_validation_accuracy"]),
            "test_accuracy": num(["test_accuracy","accuracy"], base["test_accuracy"]),
            "test_macro_f1": num(["test_macro_f1","macro_f1","test_f1"], base["test_macro_f1"]),
            "total_parameters": int(num(["total_parameters","total_params"], base["total_parameters"])),
            "trainable_parameters": int(num(["trainable_parameters","trainable_params"], base["trainable_parameters"])),
            "training_seconds": num(["training_seconds","training_time_seconds"], base["training_seconds"]),
        })
    by_name = {r["model"]: r for r in results}
    return [by_name.get(r["model"], r) for r in FALLBACK]

def percent(v):
    return v * 100 if v <= 1 else v

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    names = [r["model"] for r in results]
    val = [percent(r["best_validation_accuracy"]) for r in results]
    test = [percent(r["test_accuracy"]) for r in results]
    f1 = [percent(r["test_macro_f1"]) for r in results]
    params = [r["total_parameters"] / 1e6 for r in results]
    trainable = [r["trainable_parameters"] / 1e6 for r in results]
    mins = [r["training_seconds"] / 60 for r in results]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Smart Attendance — CNN Model Comparison", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names)); width = 0.35
    a = ax.bar(x-width/2, val, width, label="Best validation accuracy")
    b = ax.bar(x+width/2, test, width, label="Test accuracy")
    ax.set_title("Accuracy comparison"); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, names); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    ax.bar_label(a, fmt="%.1f%%", padding=2, fontsize=8); ax.bar_label(b, fmt="%.1f%%", padding=2, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(names, params)
    ax.set_title("Total model parameters"); ax.set_ylabel("Parameters (millions)"); ax.grid(axis="y", alpha=.25)
    ax.bar_label(bars, labels=[f"{p:.2f}M" for p in params], padding=3)

    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(names, f1)
    ax.set_title("Test macro F1-score"); ax.set_ylabel("Macro F1 (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.25)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in f1], padding=3)

    ax = fig.add_subplot(gs[1, 1])
    for r, p, acc in zip(results, params, test):
        ax.scatter(p, acc, s=120)
        ax.annotate(r["model"], (p, acc), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Test accuracy vs model size"); ax.set_xlabel("Total parameters (millions)")
    ax.set_ylabel("Test accuracy (%)"); ax.grid(alpha=.25)
    ax.set_ylim(min(test)-5, max(test)+5)

    fig.text(.5, .025, "Training time (minutes): " + " | ".join(f"{r['model']}: {m:.1f}" for r,m in zip(results,mins)), ha="center", fontsize=9)
    fig.text(.5, .005, "LFW classification results; not classroom face-detection benchmarks.", ha="center", fontsize=9, style="italic")
    image_path = OUT / "model_comparison_dashboard.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    eff = next(r for r in results if r["model"] == "EfficientNetB0")
    res = next(r for r in results if r["model"] == "ResNet-50")
    alex = next(r for r in results if r["model"] == "AlexNet")
    summary = [
        "SMART ATTENDANCE — MODEL SELECTION SUMMARY", "="*46, "",
        *[f"{r['model']}: val={percent(r['best_validation_accuracy']):.2f}%, test={percent(r['test_accuracy']):.2f}%, macro-F1={percent(r['test_macro_f1']):.2f}%, total params={r['total_parameters']:,}, trainable params={r['trainable_parameters']:,}, training={r['training_seconds']/60:.1f} min" for r in results],
        "", "WHY EFFICIENTNETB0 IS A REASONABLE PROVISIONAL CHOICE", "-"*46,
        f"- {100*(1-eff['total_parameters']/res['total_parameters']):.1f}% fewer total parameters than ResNet-50.",
        f"- {100*(1-eff['total_parameters']/alex['total_parameters']):.1f}% fewer total parameters than AlexNet.",
        f"- Highest validation accuracy: {percent(eff['best_validation_accuracy']):.2f}%.",
        f"- Test accuracy is {percent(res['test_accuracy'])-percent(eff['test_accuracy']):.2f} percentage points below ResNet-50; ResNet-50 has the highest test accuracy.",
        "- Smaller parameter count may help memory/storage, but inference latency must be measured on the target device.",
        "- None of the models reached the 90% validation target.",
        "- LFW classification only: EfficientNetB0 alone does not detect face locations and its classes are not the enrolled students.",
        "", f"Dashboard: {image_path}",
    ]
    summary_path = OUT / "model_comparison_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("Dashboard saved:", image_path)
    print("Summary saved:", summary_path)
    print("\n" + "\n".join(summary))

if __name__ == "__main__":
    main()
