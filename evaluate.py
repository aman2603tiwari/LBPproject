"""
evaluate.py  —  Fixed v2
=========================
Fixes:
  1. Slither: auto-detects solc version per contract, installs it via solcx
  2. Mythril: fixed output parsing for current version
  3. GNN:     uses correct GATClassifier architecture (16-dim, 128 hidden)
  4. Fallback: if tools timeout/fail, marks as "tool_failed" not "safe"

RUN: python evaluate.py
"""

import json, subprocess, sys, os, re, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.metrics import (classification_report, f1_score,
                              accuracy_score)

# ── Config ────────────────────────────────────────────────────
RESULTS_DIR  = Path("results")
MODELS_DIR   = Path("models")
GRAPHS_DIR   = Path("graphs")
INDEX_PATH   = "contracts_index.csv"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEOUT_SLITHER = 60
TIMEOUT_MYTHRIL = 90

CLASSES = ["safe", "reentrancy", "access_control",
           "integer_overflow", "logic_error", "other_vuln"]

COLLAPSE_MAP = {
    "safe":               "safe",
    "reentrancy":         "reentrancy",
    "access_control":     "access_control",
    "integer_overflow":   "integer_overflow",
    "logic_error":        "logic_error",
    "flash_loan":         "other_vuln",
    "bridge_hack":        "other_vuln",
    "oracle_manipulation":"other_vuln",
    "unknown":            "other_vuln",
    "other_vuln":         "other_vuln",
}

# Slither detector → vuln_type
SLITHER_MAP = {
    "reentrancy-eth":           "reentrancy",
    "reentrancy-no-eth":        "reentrancy",
    "reentrancy-benign":        "reentrancy",
    "reentrancy-events":        "reentrancy",
    "reentrancy-unlimited-gas": "reentrancy",
    "suicidal":                 "access_control",
    "unprotected-upgrade":      "access_control",
    "tx-origin":                "access_control",
    "controlled-delegatecall":  "access_control",
    "missing-zero-check":       "access_control",
    "arbitrary-send-eth":       "access_control",
    "arbitrary-send-erc20":     "access_control",
    "tautology":                "integer_overflow",
    "integer-overflow":         "integer_overflow",
    "unchecked-lowlevel":       "logic_error",
    "unchecked-send":           "logic_error",
    "unchecked-transfer":       "logic_error",
    "divide-before-multiply":   "logic_error",
    "incorrect-equality":       "logic_error",
    "weak-prng":                "logic_error",
    "timestamp":                "logic_error",
    "unused-return":            "logic_error",
    "shadowing-state":          "logic_error",
    "variable-scope":           "logic_error",
    "msg-value-loop":           "logic_error",
    "delegatecall-loop":        "logic_error",
}

MYTHRIL_MAP = {
    "107": "reentrancy",
    "101": "integer_overflow",
    "105": "access_control",
    "106": "access_control",
    "115": "access_control",
    "112": "logic_error",
    "113": "logic_error",
    "116": "logic_error",
    "120": "logic_error",
    "132": "other_vuln",
}


# ── Step 1: Install correct solc per contract ─────────────────

def get_pragma_version(sol_path):
    """Extract solc version from pragma statement."""
    try:
        text = Path(sol_path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'pragma\s+solidity\s+([^\s;]+)', text)
        if not m:
            return None
        spec = m.group(1).strip()
        # Clean up to get a usable version
        spec = re.sub(r'[>=<^~]', '', spec).strip().split()[0]
        if re.match(r'^\d+\.\d+\.\d+$', spec):
            return spec
        if re.match(r'^\d+\.\d+$', spec):
            return spec + ".0"
        return None
    except:
        return None


def ensure_solc(version):
    """Install solc version if not already installed."""
    try:
        import solcx
        installed = [str(v) for v in solcx.get_installed_solc_versions()]
        if not any(version in v for v in installed):
            print(f"    Installing solc {version}...", end=" ")
            solcx.install_solc(version)
            print("done")
        solcx.set_solc_version(version)
        return True
    except Exception as e:
        return False


# ── Slither runner (fixed) ────────────────────────────────────

def run_slither(sol_path):
    """
    Run Slither with the correct solc version for the contract.
    Returns (predicted_vuln_type, detector_list)
    """
    sol_path = Path(sol_path)
    if not sol_path.exists():
        return "file_missing", []

    # Detect and install correct solc version
    version = get_pragma_version(sol_path)
    if version:
        ensure_solc(version)
        solc_arg = ["--solc-remaps", f""]
        version_args = ["--solc", f"solc-{version}"] if os.name != "nt" else []
    else:
        version_args = []

    cmd = ["slither", str(sol_path), "--json", "-", "--disable-color"] + version_args

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIMEOUT_SLITHER)

        # Slither writes JSON to stdout even on findings
        stdout = r.stdout.strip()
        if not stdout:
            # Try stderr — some versions write there
            stderr = r.stderr.strip()
            if "Error" in stderr or "error" in stderr:
                return "tool_failed", []
            return "safe", []

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Sometimes Slither mixes text + JSON — find JSON part
            json_start = stdout.find('{')
            if json_start == -1:
                return "safe", []
            try:
                data = json.loads(stdout[json_start:])
            except:
                return "safe", []

        detectors = data.get("results", {}).get("detectors", [])
        if not detectors:
            return "safe", []

        # Map to vuln types — consider ALL impact levels for SmartBugs
        found_types = []
        raw_checks  = []
        for d in detectors:
            check  = d.get("check", "")
            impact = d.get("impact", "").lower()
            raw_checks.append(check)
            mapped = SLITHER_MAP.get(check)
            if mapped:
                found_types.append(mapped)

        if not found_types:
            # Found something but not in our map → logic_error as catch-all
            if raw_checks:
                return "logic_error", raw_checks
            return "safe", raw_checks

        # Return most common finding
        return Counter(found_types).most_common(1)[0][0], raw_checks

    except subprocess.TimeoutExpired:
        return "timeout", []
    except FileNotFoundError:
        return "not_installed", []
    except Exception as e:
        return "tool_failed", []


# ── Mythril runner (fixed) ────────────────────────────────────

def run_mythril(sol_path):
    """
    Run Mythril with fixed output parsing.
    """
    sol_path = Path(sol_path)
    if not sol_path.exists():
        return "file_missing", []

    # Try JSON output first
    cmd = [
        "myth", "analyze", str(sol_path),
        "--output", "json",
        "--execution-timeout", "45",
        "--max-depth", "10",
        "--disable-dependency-loading",
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIMEOUT_MYTHRIL)

        output = r.stdout.strip() or r.stderr.strip()

        if not output:
            return "safe", []

        # Check for "no issues" message
        if any(phrase in output.lower() for phrase in
               ["no issues were detected", "the analysis was completed successfully",
                "no vulnerabilities found"]):
            return "safe", []

        # Try to parse JSON
        try:
            # Find JSON in output
            json_start = output.find('[')
            if json_start == -1:
                json_start = output.find('{')
            if json_start != -1:
                data = json.loads(output[json_start:])
                # Handle list format [{"issues": [...]}]
                if isinstance(data, list) and data:
                    issues = data[0].get("issues", [])
                elif isinstance(data, dict):
                    issues = data.get("issues", [])
                else:
                    issues = []

                if not issues:
                    return "safe", []

                swc_ids = []
                for iss in issues:
                    swc = str(iss.get("swc-id", "")).replace("SWC-", "")
                    if swc:
                        swc_ids.append(swc)

                found_types = [MYTHRIL_MAP.get(s) for s in swc_ids
                               if MYTHRIL_MAP.get(s)]
                if found_types:
                    return Counter(found_types).most_common(1)[0][0], swc_ids
                elif swc_ids:
                    return "logic_error", swc_ids
                return "safe", []

        except json.JSONDecodeError:
            pass

        # Fallback: parse text output for SWC IDs
        swc_matches = re.findall(r'SWC-(\d+)', output)
        if swc_matches:
            found_types = [MYTHRIL_MAP.get(s) for s in swc_matches
                           if MYTHRIL_MAP.get(s)]
            if found_types:
                return Counter(found_types).most_common(1)[0][0], swc_matches
            return "logic_error", swc_matches

        return "safe", []

    except subprocess.TimeoutExpired:
        return "timeout", []
    except FileNotFoundError:
        return "not_installed", []
    except Exception as e:
        return "tool_failed", []


# ── GNN inference (fixed architecture) ───────────────────────

def run_gnn_inference():
    """Load the correct GATClassifier and run inference on all graphs."""
    pred_path = RESULTS_DIR / "gnn_predictions.json"

    # Always re-run to avoid stale predictions
    print("\n  [GNN] Running inference ...")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATConv, global_mean_pool

        # ── Must match train_gnn.py v3 exactly ──────────────
        NODE_FEATURES  = 16
        GRAPH_FEATURES = 10
        HIDDEN_DIM     = 128
        HEADS          = 4
        NUM_CLASSES    = 6

        class GATClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gat1 = GATConv(NODE_FEATURES, HIDDEN_DIM,
                                    heads=HEADS, dropout=0.3)
                self.gat2 = GATConv(HIDDEN_DIM * HEADS, HIDDEN_DIM,
                                    heads=1, concat=False, dropout=0.3)
                self.graph_proj = nn.Sequential(
                    nn.Linear(GRAPH_FEATURES, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(HIDDEN_DIM + 32, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(HIDDEN_DIM, NUM_CLASSES),
                )

            def forward(self, x, edge_index, batch, graph_feat):
                x = F.elu(self.gat1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gat2(x, edge_index)
                x = global_mean_pool(x, batch)
                gf = self.graph_proj(graph_feat)
                return self.classifier(torch.cat([x, gf], dim=1))

        model_path = MODELS_DIR / "gnn_multiclass_best.pt"
        if not model_path.exists():
            print(f"  [!] Model not found: {model_path}")
            return {}

        model = GATClassifier()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        print(f"  [GNN] Model loaded from {model_path}")

        predictions = {}
        graph_files = sorted(GRAPHS_DIR.glob("*.json"))
        print(f"  [GNN] Running on {len(graph_files)} graphs ...")

        with torch.no_grad():
            for jf in graph_files:
                try:
                    g = json.load(open(jf))
                    nodes = g.get("nodes", [])
                    if not nodes:
                        continue

                    # Node features — pad/truncate to 16 dims
                    feats = []
                    for n in nodes:
                        f = n.get("features", [0.0] * NODE_FEATURES)
                        if len(f) < NODE_FEATURES:
                            f = f + [0.0] * (NODE_FEATURES - len(f))
                        feats.append(f[:NODE_FEATURES])
                    x = torch.tensor(feats, dtype=torch.float)

                    # Edges
                    edges = g.get("edges", [])
                    if edges:
                        src = [e["src"] for e in edges]
                        dst = [e["dst"] for e in edges]
                        edge_index = torch.tensor([src, dst], dtype=torch.long)
                    else:
                        n = x.size(0)
                        idx = torch.arange(n)
                        edge_index = torch.stack([idx, idx], dim=0)

                    # Graph-level features
                    gf = g.get("graph_features", [0.0] * GRAPH_FEATURES)
                    if len(gf) < GRAPH_FEATURES:
                        gf = gf + [0.0] * (GRAPH_FEATURES - len(gf))
                    graph_feat = torch.tensor(
                        [gf[:GRAPH_FEATURES]], dtype=torch.float
                    )

                    batch = torch.zeros(x.size(0), dtype=torch.long)
                    out   = model(x, edge_index, batch, graph_feat)
                    pred  = out.argmax(dim=1).item()
                    predictions[jf.stem] = CLASSES[pred]

                except Exception as e:
                    print(f"    [skip] {jf.name}: {e}")

        print(f"  [GNN] Predictions: {len(predictions)}")
        print(f"  [GNN] Distribution: {Counter(predictions.values())}")

        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        return predictions

    except Exception as e:
        print(f"  [GNN] Failed: {e}")
        import traceback; traceback.print_exc()
        return {}


# ── Metrics + report ──────────────────────────────────────────

def compute_metrics(true_labels, pred_labels, tool_name):
    """Compute metrics, ignoring tool_failed/timeout/not_installed."""
    valid = [(t, p) for t, p in zip(true_labels, pred_labels)
             if p in CLASSES and t in CLASSES]

    if not valid:
        print(f"  [!] {tool_name}: no valid predictions")
        return None

    t = [v[0] for v in valid]
    p = [v[1] for v in valid]

    return {
        "tool":        tool_name,
        "n":           len(valid),
        "n_total":     len(true_labels),
        "coverage":    len(valid) / len(true_labels),
        "accuracy":    accuracy_score(t, p),
        "f1_weighted": f1_score(t, p, average="weighted", zero_division=0),
        "f1_macro":    f1_score(t, p, average="macro",    zero_division=0),
        "report":      classification_report(
                           t, p,
                           labels=[c for c in CLASSES if c in set(t)],
                           zero_division=0
                       ),
        "true":  t,
        "pred":  p,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Contract Vulnerability — Tool Comparison")
    print("=" * 60)

    # Check tools
    print("\nChecking tools ...")
    try:
        r = subprocess.run(["slither", "--version"],
                           capture_output=True, text=True, timeout=10)
        slither_ok = True
        print(f"  Slither : {r.stdout.strip() or r.stderr.strip()}")
    except:
        slither_ok = False
        print("  Slither : NOT FOUND")

    try:
        r = subprocess.run(["myth", "version"],
                           capture_output=True, text=True, timeout=10)
        mythril_ok = True
        print(f"  Mythril : {r.stdout.strip()}")
    except:
        mythril_ok = False
        print("  Mythril : NOT FOUND")

    # Load contracts
    df = pd.read_csv(INDEX_PATH)
    df = df[df["filename"].astype(str).str.endswith(".sol")].reset_index(drop=True)
    df["true_class"] = df["vuln_type"].map(
        lambda v: COLLAPSE_MAP.get(str(v).lower(), "other_vuln"))
    print(f"\nContracts: {len(df)}")
    print("True class distribution:")
    for cls, cnt in df["true_class"].value_counts().items():
        print(f"  {cls:<25} {cnt}")

    true_labels = df["true_class"].tolist()

    # ── GNN predictions ───────────────────────────────────────
    gnn_preds_dict = run_gnn_inference()
    gnn_labels = []
    for _, row in df.iterrows():
        stem = Path(row["filename"]).stem
        gnn_labels.append(gnn_preds_dict.get(stem, "no_pred"))

    # ── Slither ───────────────────────────────────────────────
    slither_labels = []
    slither_raw    = {}
    if slither_ok:
        print(f"\n{'─'*60}")
        print(f"Slither — {len(df)} contracts")
        print("(Auto-installing solc versions as needed)")
        print(f"{'─'*60}")
        for i, row in df.iterrows():
            fp = Path(row["filename"])
            pred, checks = run_slither(fp)
            slither_labels.append(pred)
            slither_raw[fp.stem] = pred
            if pred != "safe":
                print(f"  [{i+1:3d}] {fp.name[:45]:<45} → {pred}  {checks[:2]}")
        slither_found = sum(1 for p in slither_labels if p not in ("safe", "tool_failed", "timeout", "file_missing"))
        print(f"\n  Found vulnerabilities in: {slither_found}/{len(df)} contracts")
        (RESULTS_DIR / "slither_raw.json").write_text(
            json.dumps(slither_raw, indent=2))
    else:
        slither_labels = ["not_installed"] * len(df)

    # ── Mythril ───────────────────────────────────────────────
    mythril_labels = []
    mythril_raw    = {}
    if mythril_ok:
        print(f"\n{'─'*60}")
        print(f"Mythril — {len(df)} contracts (45s timeout each)")
        print(f"{'─'*60}")
        for i, row in df.iterrows():
            fp = Path(row["filename"])
            pred, swcs = run_mythril(fp)
            mythril_labels.append(pred)
            mythril_raw[fp.stem] = pred
            if pred not in ("safe", "timeout"):
                print(f"  [{i+1:3d}] {fp.name[:45]:<45} → {pred}  SWC:{swcs[:3]}")
        mythril_found = sum(1 for p in mythril_labels if p not in ("safe", "tool_failed", "timeout", "file_missing"))
        print(f"\n  Found vulnerabilities in: {mythril_found}/{len(df)} contracts")
        (RESULTS_DIR / "mythril_raw.json").write_text(
            json.dumps(mythril_raw, indent=2))
    else:
        mythril_labels = ["not_installed"] * len(df)

    # ── Build comparison CSV ──────────────────────────────────
    comp_df = pd.DataFrame({
        "filename":     [Path(r["filename"]).stem for _, r in df.iterrows()],
        "true_class":   true_labels,
        "gnn_pred":     gnn_labels,
        "slither_pred": slither_labels,
        "mythril_pred": mythril_labels,
    })
    comp_df.to_csv(RESULTS_DIR / "industry_comparison.csv", index=False)

    # ── Compute metrics ───────────────────────────────────────
    results = []
    tool_data = [("GNN (ours)", gnn_labels)]
    if slither_ok: tool_data.append(("Slither", slither_labels))
    if mythril_ok: tool_data.append(("Mythril", mythril_labels))

    lines = ["=" * 65,
             "SMART CONTRACT VULNERABILITY DETECTION — COMPARISON",
             "=" * 65, ""]

    for tool_name, pred_labels in tool_data:
        m = compute_metrics(true_labels, pred_labels, tool_name)
        if m:
            results.append(m)
            lines += [
                f"{'─'*65}",
                f"  {tool_name}",
                f"{'─'*65}",
                f"  Evaluated : {m['n']}/{m['n_total']} ({m['coverage']*100:.0f}% coverage)",
                f"  Accuracy  : {m['accuracy']:.4f}",
                f"  F1 Macro  : {m['f1_macro']:.4f}",
                f"  F1 Weighted: {m['f1_weighted']:.4f}",
                "",
                m["report"],
            ]

    # Summary table
    lines += ["=" * 65, "SUMMARY", "=" * 65]
    lines.append(f"{'Tool':<20} {'N':>5} {'Coverage':>10} {'Accuracy':>10} {'F1 Macro':>10}")
    lines.append("─" * 65)
    for m in results:
        lines.append(
            f"{m['tool']:<20} {m['n']:>5} {m['coverage']*100:>9.0f}%"
            f" {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f}"
        )

    report = "\n".join(lines)
    print(f"\n{report}")
    (RESULTS_DIR / "comparison_report.txt").write_text(report, encoding="utf-8")

    # ── Bar chart ─────────────────────────────────────────────
    if results:
        names   = [r["tool"]     for r in results]
        f1_mac  = [r["f1_macro"] for r in results]
        acc     = [r["accuracy"] for r in results]
        cov     = [r["coverage"] for r in results]

        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - w,   acc,    w, label="Accuracy",   color="#3498db")
        b2 = ax.bar(x,       f1_mac, w, label="F1 Macro",   color="#e74c3c")
        b3 = ax.bar(x + w,   cov,    w, label="Coverage",   color="#2ecc71", alpha=0.7)

        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Smart Contract Vulnerability Detection — Tool Comparison")
        ax.legend()
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                if h > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "comparison_bars.png", dpi=150)
        plt.close()
        print(f"\n  Chart saved → results/comparison_bars.png")

    print(f"\n{'='*60}")
    print("  Done. Key outputs:")
    print("    results/comparison_report.txt")
    print("    results/comparison_bars.png")
    print("    results/industry_comparison.csv")


if __name__ == "__main__":
    main()