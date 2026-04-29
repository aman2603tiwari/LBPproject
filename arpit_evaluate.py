"""
Evaluation — CodeBERT fine-tuned model
Includes: classification report, confusion matrix plot,
per-class F1 bar chart, binary safe/not-safe evaluation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("model_output_arpit")
PLOTS_DIR  = OUTPUT_DIR / "plots"
MAX_LEN    = 256
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── DATASET ──────────────────────────────────────────────────────────────────

class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─── INFERENCE ────────────────────────────────────────────────────────────────

def run_inference(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            out   = model(input_ids=batch["input_ids"].to(DEVICE),
                          attention_mask=batch["attention_mask"].to(DEVICE))
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, title, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Actual", fontsize=11)
    plt.xlabel("Predicted", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / filename}")


def plot_f1_bars(report_dict, title, filename):
    classes = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    f1s     = [report_dict[k]["f1-score"]  for k in classes]
    precs   = [report_dict[k]["precision"] for k in classes]
    recalls = [report_dict[k]["recall"]    for k in classes]
    x, w    = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*1.5), 5))
    ax.bar(x - w, precs,   w, label="Precision", color="#4C72B0")
    ax.bar(x,     f1s,     w, label="F1-Score",  color="#55A868")
    ax.bar(x + w, recalls, w, label="Recall",    color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / filename}")


def plot_class_distribution(labels, class_names, title, filename):
    counts = pd.Series(labels).value_counts().sort_index()
    names  = [class_names[i] for i in counts.index]
    plt.figure(figsize=(max(6, len(class_names)*1.2), 4))
    plt.bar(names, counts.values, color="#4C72B0", edgecolor="white")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / filename}")


def plot_binary_comparison(multi_acc, binary_acc, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Multi-class\n(vuln type)", "Binary\n(safe/unsafe)"],
                  [multi_acc, binary_acc], color=["#4C72B0", "#55A868"], width=0.4)
    for bar, val in zip(bars, [multi_acc, binary_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-class vs Binary Classification", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / filename}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    with open(OUTPUT_DIR / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    classes = [label_map[i] for i in range(len(label_map))]

    test_df = pd.read_csv(OUTPUT_DIR / "test_split.csv")
    print(f"Test set: {len(test_df)} samples")
    print(test_df["label"].value_counts().to_string())

    le = LabelEncoder()
    le.classes_ = np.array(classes)
    test_df["label_id"] = le.transform(test_df["label"])

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR / "best_model")
    model     = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR / "best_model").to(DEVICE)

    loader = DataLoader(
        ContractDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )

    # ── MULTI-CLASS ───────────────────────────────────────────────────────────
    print("\n=== Multi-class Evaluation ===")
    true_ids, pred_ids, probs = run_inference(model, loader)

    present_ids   = sorted(set(true_ids))
    present_names = [classes[i] for i in present_ids]

    report_str  = classification_report(true_ids, pred_ids, labels=present_ids,
                                        target_names=present_names, zero_division=0)
    report_dict = classification_report(true_ids, pred_ids, labels=present_ids,
                                        target_names=present_names, zero_division=0,
                                        output_dict=True)
    multi_acc = sum(p == l for p, l in zip(pred_ids, true_ids)) / len(true_ids)
    print(report_str)
    print(f"Overall Accuracy: {multi_acc:.4f}")

    # ── BINARY ────────────────────────────────────────────────────────────────
    print("\n=== Binary Evaluation (safe / not-safe) ===")
    binary_acc = 0.0
    auc        = None
    safe_id    = classes.index("safe") if "safe" in classes else -1

    if safe_id == -1:
        print("  'safe' class not in label map — skipping binary eval")
    else:
        bin_true  = (true_ids == safe_id).astype(int)
        bin_pred  = (pred_ids == safe_id).astype(int)
        bin_probs = probs[:, safe_id]
        bin_report = classification_report(bin_true, bin_pred,
                                           target_names=["vulnerable", "safe"],
                                           zero_division=0)
        print(bin_report)
        binary_acc = sum(bt == bp for bt, bp in zip(bin_true, bin_pred)) / len(bin_true)
        print(f"Binary Accuracy: {binary_acc:.4f}")
        try:
            auc = roc_auc_score(bin_true, bin_probs)
            print(f"AUC-ROC (safe vs rest): {auc:.4f}")
        except Exception:
            pass

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    print("\n=== Saving plots ===")
    plot_confusion_matrix(true_ids, pred_ids, present_names,
                          "Confusion Matrix — Multi-class",
                          "confusion_matrix_multiclass.png")
    plot_f1_bars(report_dict,
                 "Precision / F1 / Recall per Vulnerability Class",
                 "f1_bars_multiclass.png")
    plot_class_distribution(true_ids, classes,
                            "Test Set Class Distribution",
                            "class_distribution.png")
    if safe_id != -1:
        plot_confusion_matrix(bin_true, bin_pred, ["vulnerable", "safe"],
                              "Confusion Matrix — Binary (safe vs vulnerable)",
                              "confusion_matrix_binary.png")
        plot_binary_comparison(multi_acc, binary_acc, "accuracy_comparison.png")

    # ── PREDICTIONS CSV — only use columns that actually exist ────────────────
    save_cols = {"label": test_df["label"].values,
                 "predicted": [classes[p] for p in pred_ids],
                 "correct":   [l == classes[p] for l, p in zip(test_df["label"], pred_ids)],
                 "confidence": [probs[i].max() for i in range(len(probs))]}

    if safe_id != -1:
        save_cols["binary_true"]      = ["safe" if t == safe_id else "vulnerable" for t in true_ids]
        save_cols["binary_predicted"] = ["safe" if p == safe_id else "vulnerable" for p in pred_ids]

    # Add filename column only if it exists in test_df
    if "filename" in test_df.columns:
        save_cols["filename"] = test_df["filename"].values

    pd.DataFrame(save_cols).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/test_predictions.csv")

    # ── SUMMARY JSON ──────────────────────────────────────────────────────────
    summary = {
        "multiclass_accuracy": round(multi_acc, 4),
        "binary_accuracy":     round(binary_acc, 4),
        "auc_roc":             round(auc, 4) if auc else None,
        "classes":             classes,
        "classes_in_test":     present_names,
        "classification_report": report_str,
    }
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/eval_results.json")

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"  Multi-class accuracy : {multi_acc:.2%}")
    print(f"  Binary accuracy      : {binary_acc:.2%}")
    if auc:
        print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  Plots saved to       : {PLOTS_DIR}/")
    print("="*50)


if __name__ == "__main__":
    main()