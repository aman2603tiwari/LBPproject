"""
train_gnn.py v4 — fixed for actual graph JSON format:
  - edges are dicts: {"src": 0, "dst": 1, "type": "AST_CHILD"}
  - label field is integer (0/1) — vuln_type field is the string we want
  - graph_features is already a 10-dim list of floats
"""

import json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GRAPHS_DIR   = Path("graphs_augmented")   # use augmented dataset
MODEL_DIR    = Path("models")
RESULTS_DIR  = Path("results")

NODE_FEATURES  = 16
GRAPH_FEATURES = 10
HIDDEN_DIM     = 128
HEADS          = 4
NUM_CLASSES    = 6

BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 20
DROPOUT_GAT = 0.3
DROPOUT_CLS = 0.4

TRAIN_AUG_NOISE_SIGMA = 0.04
TRAIN_AUG_EDGE_DROP   = 0.10
TRAIN_AUG_PROB        = 0.5

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# vuln_type string → class index
VULN_TO_IDX = {
    "safe":                0,
    "reentrancy":          1,
    "access_control":      2,
    "integer_overflow":    3,
    "logic_error":         4,
    "flash_loan":          5,
    "oracle_manipulation": 5,
    "bridge_hack":         5,
    "unknown":             5,
    "other_vuln":          5,
}

# augment.py canonical label string → class index (same mapping)
CANONICAL_TO_IDX = VULN_TO_IDX.copy()

LABEL_NAMES = ["safe", "reentrancy", "access_control",
               "integer_overflow", "logic_error", "other_vuln"]


# ── Model (identical architecture to v3) ─────────────────────────────────────
class GATClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(NODE_FEATURES, HIDDEN_DIM,
                            heads=HEADS, dropout=DROPOUT_GAT)
        self.gat2 = GATConv(HIDDEN_DIM * HEADS, HIDDEN_DIM,
                            heads=1, concat=False, dropout=DROPOUT_GAT)
        self.graph_proj = nn.Sequential(
            nn.Linear(GRAPH_FEATURES, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 32, 128), nn.ReLU(),
            nn.Dropout(DROPOUT_CLS), nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x, edge_index, batch, graph_feat):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        gf = self.graph_proj(graph_feat)
        return self.classifier(torch.cat([x, gf], dim=1))


# ── Data loading ──────────────────────────────────────────────────────────────
def parse_label(g: dict) -> int:
    """
    Resolve label from graph JSON.
    Priority:
      1. 'label' key if it's a string matching VULN_TO_IDX
      2. 'vuln_type' key (string)
      3. 'label' key if it's an int 0/1 — but we need the vuln_type string
         because binary label 1 just means 'vulnerable', not which class.
    """
    # augment.py writes canonical string labels like "reentrancy" into 'label'
    label_val = g.get("label", "")
    if isinstance(label_val, str) and label_val.lower() in VULN_TO_IDX:
        return VULN_TO_IDX[label_val.lower()]

    # vuln_type is always the string we want
    vuln = str(g.get("vuln_type", "")).strip().lower()
    if vuln in VULN_TO_IDX:
        return VULN_TO_IDX[vuln]

    # last resort
    return 5  # other_vuln


def load_graphs(graphs_dir: Path) -> list:
    if not graphs_dir.exists():
        print(f"[WARN] {graphs_dir} not found, falling back to 'graphs'")
        graphs_dir = Path("graphs")

    files = sorted(graphs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in {graphs_dir}")

    dataset = []
    skipped = 0

    for fp in files:
        try:
            with open(fp) as f:
                g = json.load(f)

            # ── Node features
            nodes = g.get("nodes", [])
            if not nodes:
                skipped += 1
                continue

            x = torch.tensor(
                [n["features"] for n in nodes], dtype=torch.float
            )
            # Pad/truncate to NODE_FEATURES
            if x.shape[1] < NODE_FEATURES:
                x = F.pad(x, (0, NODE_FEATURES - x.shape[1]))
            elif x.shape[1] > NODE_FEATURES:
                x = x[:, :NODE_FEATURES]

            # ── Edges  ← key fix: edges are dicts {"src":, "dst":, "type":}
            raw_edges = g.get("edges", [])
            if raw_edges:
                if isinstance(raw_edges[0], dict):
                    srcs = [e["src"] for e in raw_edges]
                    dsts = [e["dst"] for e in raw_edges]
                else:
                    # list format [src, dst] or [src, dst, ...]
                    srcs = [e[0] for e in raw_edges]
                    dsts = [e[1] for e in raw_edges]
                ei = torch.tensor([srcs, dsts], dtype=torch.long)
            else:
                ei = torch.zeros((2, 1), dtype=torch.long)

            # ── Graph-level features
            gf_raw = g.get("graph_features", [0.0] * GRAPH_FEATURES)
            if len(gf_raw) < GRAPH_FEATURES:
                gf_raw = gf_raw + [0.0] * (GRAPH_FEATURES - len(gf_raw))
            gf = torch.tensor(gf_raw[:GRAPH_FEATURES], dtype=torch.float)

            # ── Label
            y_int = parse_label(g)
            y = torch.tensor([y_int], dtype=torch.long)

            data = Data(x=x, edge_index=ei, y=y, graph_feat=gf)
            data.file_name = fp.name
            dataset.append(data)

        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping {fp.name}: {e}")

    print(f"Loaded {len(dataset)} graphs ({skipped} skipped) from {graphs_dir}")
    return dataset


# ── In-training augmentation ──────────────────────────────────────────────────
CONTINUOUS_IDX = [0, 6, 14, 15]

def augment_data(data: Data) -> Data:
    if random.random() > TRAIN_AUG_PROB:
        return data
    data = data.clone()
    # Node feature noise
    noise = torch.zeros_like(data.x)
    for i in CONTINUOUS_IDX:
        if i < data.x.shape[1]:
            noise[:, i] = torch.randn(data.x.shape[0]) * TRAIN_AUG_NOISE_SIGMA
    data.x = (data.x + noise).clamp(0.0, 1.0)
    # Edge dropout
    if data.edge_index.shape[1] > 2:
        keep = torch.rand(data.edge_index.shape[1]) > TRAIN_AUG_EDGE_DROP
        if keep.sum() > 0:
            data.edge_index = data.edge_index[:, keep]
    # Graph feature noise
    data.graph_feat = (data.graph_feat +
                       torch.randn_like(data.graph_feat) * 0.06).clamp(min=0.0)
    return data


# ── Class weights ─────────────────────────────────────────────────────────────
def compute_class_weights(dataset):
    counts = torch.zeros(NUM_CLASSES)
    for d in dataset:
        counts[d.y.item()] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (NUM_CLASSES * counts)
    print("Class weights:", {LABEL_NAMES[i]: f"{weights[i]:.2f}"
                              for i in range(NUM_CLASSES)})
    return weights


# ── Train / eval loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        aug_list = [augment_data(d) for d in batch.to_data_list()]
        batch = Batch.from_data_list(aug_list).to(device)
        optimizer.zero_grad()
        graph_feat = batch.graph_feat.view(batch.num_graphs, -1)
        out = model(batch.x, batch.edge_index, batch.batch, graph_feat)
        loss = criterion(out, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        graph_feat = batch.graph_feat.view(batch.num_graphs, -1)
        out = model(batch.x, batch.edge_index, batch.batch, graph_feat)
        all_preds.extend(out.argmax(dim=1).cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())
    return all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    dataset = load_graphs(GRAPHS_DIR)

    labels = [d.y.item() for d in dataset]
    print(f"\nClass distribution:")
    for cls_id, cnt in sorted(Counter(labels).items()):
        print(f"  {LABEL_NAMES[cls_id]:<22} {cnt:>4}")

    if len(dataset) == 0:
        print("[ERROR] No graphs loaded. Check graphs_augmented/ folder.")
        return

    # Stratified split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15,
        stratify=[labels[i] for i in train_idx], random_state=SEED
    )

    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in test_idx]
    print(f"\nSplit — train:{len(train_set)}  val:{len(val_set)}  test:{len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

    model     = GATClassifier().to(device)
    weights   = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0
    patience_ctr = 0
    best_epoch   = 0

    print(f"\nTraining up to {EPOCHS} epochs (patience={PATIENCE})...\n")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro",
                          zero_division=0)
        scheduler.step()

        if epoch % 10 == 0 or val_f1 > best_val_f1:
            print(f"Ep {epoch:03d} | loss={loss:.4f} | val_F1={val_f1:.4f}"
                  + (" ★ best" if val_f1 > best_val_f1 else ""))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_ctr = 0
            torch.save(model.state_dict(),
                       MODEL_DIR / "gnn_multiclass_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch} (best={best_epoch})")
                break

    # ── Test set evaluation
    print(f"\nLoading best model (epoch {best_epoch}, val_F1={best_val_f1:.4f})")
    model.load_state_dict(torch.load(MODEL_DIR / "gnn_multiclass_best.pt",
                                     map_location=device))
    test_preds, test_labels = evaluate(model, test_loader, device)

    report  = classification_report(test_labels, test_preds,
                                    target_names=LABEL_NAMES, zero_division=0)
    macro_f1 = f1_score(test_labels, test_preds, average="macro",
                        zero_division=0)
    acc = sum(p == l for p, l in zip(test_preds, test_labels)) / len(test_labels)

    print(f"\n{'═'*55}")
    print(f"MULTICLASS TEST RESULTS")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"\n{report}")

    bin_true  = [0 if l == 0 else 1 for l in test_labels]
    bin_preds = [0 if p == 0 else 1 for p in test_preds]
    bin_f1  = f1_score(bin_true, bin_preds, average="macro", zero_division=0)
    bin_acc = sum(p == l for p, l in zip(bin_preds, bin_true)) / len(bin_true)
    print(f"BINARY  Acc={bin_acc:.4f}  F1={bin_f1:.4f}")

    results = {
        "multiclass_accuracy": acc,
        "multiclass_f1_macro": macro_f1,
        "binary_accuracy": bin_acc,
        "binary_f1_macro": bin_f1,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "n_train": len(train_set),
        "n_val":   len(val_set),
        "n_test":  len(test_set),
        "classification_report": report,
    }
    with open(RESULTS_DIR / "metrics_multiclass.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/metrics_multiclass.json")
    print(f"Saved → {MODEL_DIR}/gnn_multiclass_best.pt")


if __name__ == "__main__":
    main()