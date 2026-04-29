"""
Fine-tuning CodeBERT — FAST version
Freezes bottom 10 transformer layers, only trains top 2 + classifier head.
~4x faster on CPU. Still proper fine-tuning, not training from scratch.
"""

import json, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INDEX_CSV        = "contracts_index.csv"
MODEL_NAME       = "microsoft/codebert-base"
OUTPUT_DIR       = Path("model_output_arpit")
MAX_LEN          = 256    # reduced from 512 — 2x speed boost, minimal accuracy loss
BATCH_SIZE       = 16     # larger batch since less memory needed with shorter seqs
EPOCHS           = 5
LR               = 3e-4   # higher LR since most layers frozen
SEED             = 42
MIN_SAMPLES      = 5
TARGET_PER_CLASS = 50
FREEZE_LAYERS    = 10     # freeze bottom 10 of 12 layers — only train top 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR.mkdir(exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_dataset(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["filename"])
    texts, labels = [], []
    for _, row in df.iterrows():
        path = Path(str(row["filename"]))
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:3000]  # shorter = faster
            texts.append(text)
            labels.append(str(row["vuln_type"]).strip().lower())
        except Exception:
            continue
    data = pd.DataFrame({"text": texts, "label": labels})
    counts = data["label"].value_counts()
    rare   = counts[counts < MIN_SAMPLES].index.tolist()
    if rare:
        print(f"Merging rare classes into 'other': {rare}")
        data["label"] = data["label"].apply(lambda x: "other" if x in rare else x)
    print(f"\nLoaded {len(data)} contracts")
    print(data["label"].value_counts().to_string())
    return data

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────

def augment_text(text, strategy):
    if strategy == 0:
        headers = ["// security review\n", "// audit target\n", "// vulnerability analysis\n",
                   "// smart contract\n", "// solidity code\n"]
        return headers[hash(text) % len(headers)] + text
    elif strategy == 1:
        return re.sub(r'//[^\n]*', '', text)
    elif strategy == 2:
        text = re.sub(r'\bowner\b', 'admin', text)
        text = re.sub(r'\bbalance\b', 'amount', text)
        return text
    elif strategy == 3:
        lines = text.split('\n')
        return '\n'.join(lines[:len(lines)//2]) if len(lines) > 20 else text
    return text

def augment(data):
    augmented = [data]
    for label, count in data["label"].value_counts().items():
        if count >= TARGET_PER_CLASS:
            continue
        subset = data[data["label"] == label].reset_index(drop=True)
        extras = []
        for i in range(TARGET_PER_CLASS - count):
            row = subset.iloc[i % len(subset)].copy()
            row["text"] = augment_text(row["text"], i % 4)
            extras.append(row)
        augmented.append(pd.DataFrame(extras))
    result = pd.concat(augmented, ignore_index=True).sample(frac=1, random_state=SEED)
    print(f"\nAfter augmentation: {len(result)} samples")
    print(result["label"].value_counts().to_string())
    return result

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

# ─── FREEZE LAYERS — key function ─────────────────────────────────────────────

def freeze_base_layers(model, freeze_up_to=10):
    """
    CodeBERT has 12 transformer layers (0–11).
    We freeze 0–9, only train 10–11 + classifier.
    This makes it 4x faster and prevents catastrophic forgetting.
    """
    # Freeze embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    # Freeze bottom N transformer layers
    for i in range(freeze_up_to):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    # Count trainable vs frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"Frozen layers: 0–{freeze_up_to-1} | Training layers: {freeze_up_to}–11 + classifier")

# ─── TRAIN / EVAL ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        lbls   = batch["label"].to(DEVICE)
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss   = loss_fn(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        correct    += (logits.argmax(-1) == lbls).sum().item()
        total      += lbls.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, le):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE))
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    present_ids   = sorted(set(all_labels))
    present_names = [le.classes_[i] for i in present_ids]
    report = classification_report(all_labels, all_preds, labels=present_ids,
                                   target_names=present_names, zero_division=0)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, report

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    raw  = load_dataset(INDEX_CSV)
    data = augment(raw)

    le = LabelEncoder()
    data["label_id"] = le.fit_transform(data["label"])
    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump({int(i): c for i, c in enumerate(le.classes_)}, f, indent=2)

    train_df, temp_df = train_test_split(data, test_size=0.25, random_state=SEED)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR  / "val_split.csv",  index=False)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_), ignore_mismatched_sizes=True
    ).to(DEVICE)

    # ── FREEZE BOTTOM LAYERS ──
    freeze_base_layers(model, freeze_up_to=FREEZE_LAYERS)

    counts       = data["label_id"].value_counts().sort_index().values
    weights      = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights      = weights / weights.sum() * len(le.classes_)
    loss_fn      = torch.nn.CrossEntropyLoss(weight=weights)

    sampler      = WeightedRandomSampler(
        [1.0 / Counter(train_df["label_id"].tolist())[l] for l in train_df["label_id"].tolist()],
        num_samples=len(train_df), replacement=True
    )
    train_loader = DataLoader(ContractDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(ContractDataset(val_df["text"].tolist(),   val_df["label_id"].tolist(),   tokenizer), batch_size=BATCH_SIZE)

    # Only pass trainable params to optimizer
    optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_val_acc = 0.0
    print("\n=== Fine-tuning (frozen layers 0-9, training layers 10-11 + head) ===")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc  = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_acc, val_rep = evaluate(model, val_loader, le)
        print(f"\nEpoch {epoch}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")
        print(val_rep)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(OUTPUT_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUT_DIR / "best_model")
            print(f"  [Saved best model — val_acc={val_acc:.4f}]")

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")
    print("Now run: python evaluate.py")

if __name__ == "__main__":
    main()