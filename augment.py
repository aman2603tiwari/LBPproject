"""
augment_graphs.py — Expand graph dataset from ~175 to ~650+ samples
Fix: matches graph filenames (e.g. 0x0eee3e38_reentrancy.json) against
     CSV filename column (e.g. contracts\\vulnerable\\0x0eee3e38_reentrancy.sol)
     by comparing stems only.
"""

import json, os, random, copy, csv
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GRAPHS_DIR    = Path("graphs")
OUTPUT_DIR    = Path("graphs_augmented")
CONTRACTS_CSV = Path("contracts_index.csv")

NOISE_SIGMA       = 0.05
EDGE_DROP_LO      = 0.05
EDGE_DROP_HI      = 0.15
GRAPH_FEAT_NOISE  = 0.08
MASK_PROB         = 0.10

# Additional copies per class (original always kept → total = 1 + multiplier)
CLASS_MULTIPLIER = {
    "reentrancy":       1,   # x2 total
    "integer_overflow": 1,   # x2
    "other_vuln":       2,   # x3
    "logic_error":      5,   # x6
    "access_control":   5,   # x6
    "safe":             5,   # x6
}

BINARY_FEATURE_INDICES     = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13}
CONTINUOUS_FEATURE_INDICES = {0, 6, 14, 15}

LABEL_MAP = {
    "safe":                "safe",
    "reentrancy":          "reentrancy",
    "access_control":      "access_control",
    "integer_overflow":    "integer_overflow",
    "logic_error":         "logic_error",
    "flash_loan":          "other_vuln",
    "oracle_manipulation": "other_vuln",
    "bridge_hack":         "other_vuln",
    "unknown":             "other_vuln",
    "other_vuln":          "other_vuln",
}

random.seed(42)


# ── Label loading ─────────────────────────────────────────────────────────────
def load_label_map(csv_path: Path) -> dict:
    """
    Returns {stem -> canonical_label}.
    stem = filename without directory or extension.
    e.g. 'contracts\\vulnerable\\0x0eee3e38_reentrancy.sol' -> '0x0eee3e38_reentrancy'
    """
    result = {}
    if not csv_path.exists():
        print(f"[WARN] {csv_path} not found")
        return result

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get("filename", "").strip()
            if not raw_path:
                continue
            # Normalise Windows backslashes, then take stem
            stem = Path(raw_path.replace("\\", "/")).stem
            raw_label = row.get("vuln_type", row.get("label", "other_vuln")).strip().lower()
            result[stem] = LABEL_MAP.get(raw_label, "other_vuln")

    print(f"Loaded {len(result)} labels from {csv_path}")
    return result


# ── Augmentation helpers ──────────────────────────────────────────────────────
def add_gaussian_noise(features):
    result = features[:]
    for i in CONTINUOUS_FEATURE_INDICES:
        if i < len(result):
            result[i] = float(max(0.0, min(1.0,
                result[i] + random.gauss(0, NOISE_SIGMA))))
    return result

def flip_binary(features, flip_p=0.04):
    result = features[:]
    for i in BINARY_FEATURE_INDICES:
        if i < len(result) and random.random() < flip_p:
            result[i] = 1.0 - result[i]
    return result

def mask_features(features):
    result = features[:]
    for i in CONTINUOUS_FEATURE_INDICES:
        if i < len(result) and random.random() < MASK_PROB:
            result[i] = 0.0
    return result

def drop_edges(edges, drop_rate=None):
    if not edges:
        return edges

    if drop_rate is None:
        drop_rate = random.uniform(EDGE_DROP_LO, EDGE_DROP_HI)

    pair_set = set()

    for e in edges:
        if isinstance(e, dict):
            src, dst = e["src"], e["dst"]
        else:
            src, dst = e[:2]
        pair_set.add((min(src, dst), max(src, dst)))

    pairs = list(pair_set)
    n_drop = max(0, int(len(pairs) * drop_rate))
    to_drop = set(random.sample(pairs, min(n_drop, len(pairs))))

    kept = []
    for e in edges:
        if isinstance(e, dict):
            src, dst = e["src"], e["dst"]
            key = (min(src, dst), max(src, dst))
            if key not in to_drop:
                kept.append(e)
        else:
            src, dst = e[:2]
            key = (min(src, dst), max(src, dst))
            if key not in to_drop:
                kept.append(e)

    return kept if kept else edges[:1]
def perturb_graph_features(gf):
    return [max(0.0, v + random.gauss(0, GRAPH_FEAT_NOISE * max(abs(v), 0.1)))
            for v in gf]

def augment_graph(graph, aug_id):
    g = copy.deepcopy(graph)
    new_nodes = []
    for node in g.get("nodes", []):
        feats = node.get("features", [])
        feats = add_gaussian_noise(feats)
        if random.random() < 0.5:
            feats = flip_binary(feats)
        if random.random() < 0.4:
            feats = mask_features(feats)
        node["features"] = feats
        new_nodes.append(node)
    g["nodes"] = new_nodes

    if g.get("edges") and random.random() < 0.7:
        g["edges"] = drop_edges(g["edges"])

    if "graph_features" in g:
        g["graph_features"] = perturb_graph_features(g["graph_features"])

    g["augmented"] = True
    g["aug_id"] = aug_id
    return g


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete stale files in output dir from previous broken run
    stale = list(OUTPUT_DIR.glob("*.json"))
    if stale:
        print(f"Clearing {len(stale)} stale files from {OUTPUT_DIR}...")
        for f in stale:
            f.unlink()

    label_map = load_label_map(CONTRACTS_CSV)

    graph_files = sorted(GRAPHS_DIR.glob("*.json"))
    if not graph_files:
        print(f"[ERROR] No .json files in {GRAPHS_DIR}")
        return

    print(f"Found {len(graph_files)} original graphs\n")

    class_counts = {}
    total_written = 0
    label_misses  = 0

    for gf in graph_files:
        with open(gf) as f:
            graph = json.load(f)

        stem = gf.stem  # e.g. '0x0eee3e38_reentrancy'

        # ── Priority 1: CSV label map (most reliable)
        if stem in label_map:
            label = label_map[stem]

        # ── Priority 2: label embedded in graph JSON
        else:
            raw = str(graph.get("label", graph.get("vuln_type", ""))).strip().lower()
            label = LABEL_MAP.get(raw, None)

            # ── Priority 3: parse from filename itself
            # e.g. '0x0eee3e38_reentrancy' → last underscore segment
            if label is None:
                parts = stem.split("_")
                label = None
                for part in reversed(parts):
                    if part in LABEL_MAP:
                        label = LABEL_MAP[part]
                        break
                if label is None:
                    label = "other_vuln"
                    label_misses += 1

        graph["label"] = label

        # Write original
        with open(OUTPUT_DIR / gf.name, "w") as f:
            json.dump(graph, f)
        total_written += 1
        class_counts[label] = class_counts.get(label, 0) + 1

        # Write augmented copies
        n_copies = CLASS_MULTIPLIER.get(label, 1)
        for i in range(n_copies):
            aug = augment_graph(graph, i)
            aug_name = f"{stem}__aug{i}.json"
            with open(OUTPUT_DIR / aug_name, "w") as f:
                json.dump(aug, f)
            total_written += 1
            class_counts[label] = class_counts.get(label, 0) + 1

    print(f"{'─'*52}")
    print(f"Total graphs written to {OUTPUT_DIR}: {total_written}")
    if label_misses:
        print(f"[WARN] {label_misses} graphs used filename-parsed fallback label")
    print(f"\nClass distribution after augmentation:")
    for cls, cnt in sorted(class_counts.items()):
        bar = "█" * (cnt // 5)
        print(f"  {cls:<22} {cnt:>4}  {bar}")
    print(f"{'─'*52}")
    print(f"\nNext step: python train_gnn.py")


if __name__ == "__main__":
    main()