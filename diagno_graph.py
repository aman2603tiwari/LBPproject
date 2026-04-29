"""Run this from your LBP_PROJECT directory to diagnose graph quality."""
import json, numpy as np, sys
from pathlib import Path
from collections import defaultdict, Counter

COLLAPSE_MAP = {
    "safe":"safe","reentrancy":"reentrancy","access_control":"access_control",
    "integer_overflow":"integer_overflow","logic_error":"logic_error",
    "flash_loan":"other_vuln","bridge_hack":"other_vuln",
    "oracle_manipulation":"other_vuln","unknown":"other_vuln",
}

graphs_dir = Path("graphs")
if not graphs_dir.exists():
    print("ERROR: graphs/ directory not found. Run from LBP_PROJECT directory.")
    sys.exit(1)

files = list(graphs_dir.glob("*.json"))
print(f"Found {len(files)} graph JSON files\n")

# Peek at first file structure
sample = json.load(open(files[0]))
print(f"Sample graph keys: {list(sample.keys())}")
print(f"Sample method: {sample.get('method','?')}")
print(f"Sample vuln_type: {sample.get('vuln_type','?')}")
print(f"Sample n_nodes: {len(sample['nodes'])}")
print(f"Sample node[0]: {sample['nodes'][0] if sample['nodes'] else 'EMPTY'}\n")

stats = defaultdict(list)
empty_graphs = 0
regex_count  = 0
solc_count   = 0

for jf in files:
    g = json.load(open(jf))
    vtype  = COLLAPSE_MAP.get(g.get("vuln_type","unknown"), "other_vuln")
    nodes  = g["nodes"]
    edges  = g["edges"]
    method = g.get("method", "?")

    if method == "regex_fallback": regex_count += 1
    elif method == "solc_ast":     solc_count  += 1

    if not nodes:
        empty_graphs += 1
        continue

    feats = np.array([n["features"] for n in nodes], dtype=float)
    stats[vtype].append({
        "n_nodes":     len(nodes),
        "n_edges":     len(edges),
        "nonzero_pct": float((feats != 0).mean()),
        "feat_mean":   feats.mean(axis=0).tolist(),
        "has_call":    float(feats[:, 1].max()),
        "method":      method,
    })

print(f"Build method breakdown:  solc_ast={solc_count}  regex_fallback={regex_count}")
print(f"Empty graphs: {empty_graphs}\n")

print(f"{'CLASS':<22} {'N':>4} {'AVG_NODES':>10} {'AVG_EDGES':>10} {'NONZERO%':>9} {'CALL%':>7}")
print("-"*65)
for cls in ["safe","reentrancy","access_control","integer_overflow","logic_error","other_vuln"]:
    s = stats.get(cls, [])
    if not s: continue
    print(f"{cls:<22} {len(s):>4} "
          f"{np.mean([x['n_nodes'] for x in s]):>10.1f} "
          f"{np.mean([x['n_edges'] for x in s]):>10.1f} "
          f"{np.mean([x['nonzero_pct'] for x in s])*100:>8.1f}% "
          f"{np.mean([x['has_call'] for x in s])*100:>6.1f}%")

print("\nFeature vector means per class:")
print(f"{'CLASS':<22}  [type  call   send   delg   stat   pub    dep    chld]")
print("-"*80)
for cls in ["safe","reentrancy","access_control","integer_overflow","logic_error","other_vuln"]:
    s = stats.get(cls, [])
    if not s: continue
    fm = np.mean([x["feat_mean"] for x in s], axis=0)
    print(f"{cls:<22}  {[round(v,3) for v in fm]}")

# KEY DIAGNOSTIC: are feature vectors distinguishable between classes?
print("\n--- SEPARABILITY CHECK ---")
print("If all classes have nearly identical feature means → model CANNOT distinguish them")
all_means = {}
for cls, s in stats.items():
    if s:
        all_means[cls] = np.mean([x["feat_mean"] for x in s], axis=0)

classes = list(all_means.keys())
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        diff = np.linalg.norm(
            np.array(all_means[classes[i]]) - np.array(all_means[classes[j]])
        )
        print(f"  {classes[i]:20} vs {classes[j]:20}  L2 dist = {diff:.4f}")