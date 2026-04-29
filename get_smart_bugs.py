"""
SmartBugs Dataset Downloader
==============================
SmartBugs is the standard academic benchmark for smart contract
vulnerability detection. Used in 50+ papers.

Two sources pulled here:
  1. SmartBugs-Curated — 143 hand-labelled contracts, 7 vuln categories
     https://github.com/smartbugs/smartbugs-curated
  
  2. SWC Registry samples — Smart Contract Weakness Classification
     registry contracts (the official EIP for vuln classification)
     https://github.com/SmartContractSecurity/SWC-registry

Both map cleanly to your existing vuln_type labels.
Output merges with your existing contracts_index.csv.
"""

import re, requests, base64, time, pandas as pd
from pathlib import Path

# ── Output dirs (same structure as collect_contracts.py) ──
VULN_DIR = Path("contracts/vulnerable")
SAFE_DIR  = Path("contracts/safe")
VULN_DIR.mkdir(parents=True, exist_ok=True)
SAFE_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_API  = "https://api.github.com"
DELAY       = 0.1   # GitHub allows 60 unauthenticated req/min

# ── Vuln type mapping: SmartBugs folder name → your label ──
SMARTBUGS_LABEL_MAP = {
    "reentrancy":            "reentrancy",
    "access_control":        "access_control",
    "arithmetic":            "integer_overflow",   # SmartBugs calls it arithmetic
    "unchecked_low_level_calls": "logic_error",
    "denial_of_service":     "logic_error",
    "front_running":         "logic_error",
    "time_manipulation":     "logic_error",
    "bad_randomness":        "logic_error",
    "short_addresses":       "logic_error",
    "other":                 "unknown",
}

new_rows = []


# ─────────────────────────────────────────────────────
# SOURCE 1: SmartBugs-Curated
# ─────────────────────────────────────────────────────

def fetch_github_dir(owner, repo, path=""):
    """List contents of a GitHub directory via API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, timeout=10)
    if r.status_code == 403:
        print(f"  [!] GitHub rate limit hit. Wait 60s and re-run.")
        return []
    if r.status_code != 200:
        return []
    return r.json()


def fetch_github_file(download_url):
    """Download raw file content."""
    r = requests.get(download_url, timeout=10)
    if r.status_code != 200:
        return None
    return r.text


def pull_smartbugs_curated():
    print("\n[SOURCE 1] SmartBugs-Curated dataset ...")
    print("  Repo: github.com/smartbugs/smartbugs-curated")

    owner = "smartbugs"
    repo  = "smartbugs-curated"

    # Top-level has dataset/ folder with one subfolder per vuln type
    top = fetch_github_dir(owner, repo, "dataset")
    if not top:
        print("  [!] Could not reach SmartBugs-Curated. Check internet.")
        return

    for vuln_folder in top:
        if vuln_folder.get("type") != "dir":
            continue

        folder_name = vuln_folder["name"]
        vuln_type   = SMARTBUGS_LABEL_MAP.get(folder_name, "unknown")

        print(f"\n  Category: {folder_name} → {vuln_type}")
        files = fetch_github_dir(owner, repo, f"dataset/{folder_name}")
        time.sleep(DELAY)

        for f in files:
            if not f.get("name","").endswith(".sol"):
                continue

            fname    = f["name"]
            dl_url   = f.get("download_url")
            if not dl_url:
                continue

            out_name = f"sb_{folder_name[:8]}_{fname}"
            out_path = VULN_DIR / out_name

            if out_path.exists():
                print(f"    SKIP  {out_name}")
                new_rows.append({"address": "", "label": 1, "vuln_type": vuln_type,
                                  "filename": str(out_path), "status": "exists",
                                  "source": "smartbugs_curated"})
                continue

            source = fetch_github_file(dl_url)
            time.sleep(DELAY)

            if source:
                out_path.write_text(source, encoding="utf-8")
                print(f"    ✓  {out_name}  ({len(source):,} chars)")
                new_rows.append({"address": "", "label": 1, "vuln_type": vuln_type,
                                  "filename": str(out_path), "status": "downloaded",
                                  "source": "smartbugs_curated"})
            else:
                print(f"    ✗  {fname}")


# ─────────────────────────────────────────────────────
# SOURCE 2: SWC Registry
# ─────────────────────────────────────────────────────

# SWC IDs mapped to your vuln_type labels
SWC_MAP = {
    "SWC-107": "reentrancy",          # Reentrancy
    "SWC-101": "integer_overflow",    # Integer Overflow
    "SWC-105": "access_control",      # Unprotected Ether Withdrawal
    "SWC-106": "access_control",      # Unprotected SELFDESTRUCT
    "SWC-112": "logic_error",         # Delegatecall to Untrusted Callee
    "SWC-113": "logic_error",         # DoS with Failed Call
    "SWC-115": "access_control",      # Authorization via tx.origin
    "SWC-116": "logic_error",         # Block values as time proxy
    "SWC-120": "logic_error",         # Weak Sources of Randomness
    "SWC-132": "flash_loan",          # Unexpected Ether balance
}


def pull_swc_registry():
    print("\n[SOURCE 2] SWC Registry sample contracts ...")
    print("  Repo: github.com/SmartContractSecurity/SWC-registry")

    owner = "SmartContractSecurity"
    repo  = "SWC-registry"

    top = fetch_github_dir(owner, repo, "dataset")
    if not top:
        print("  [!] Could not reach SWC-registry.")
        return

    for swc_folder in top:
        if swc_folder.get("type") != "dir":
            continue

        swc_id    = swc_folder["name"]          # e.g. "SWC-107"
        vuln_type = SWC_MAP.get(swc_id, None)
        if not vuln_type:
            continue   # skip unmapped SWC IDs

        print(f"\n  {swc_id} → {vuln_type}")
        files = fetch_github_dir(owner, repo, f"dataset/{swc_id}")
        time.sleep(DELAY)

        for f in files:
            name = f.get("name","")
            if not name.endswith(".sol"):
                continue

            dl_url = f.get("download_url")
            if not dl_url:
                continue

            out_name = f"swc_{swc_id.lower().replace('-','_')}_{name}"
            out_path = VULN_DIR / out_name

            if out_path.exists():
                print(f"    SKIP  {out_name}")
                new_rows.append({"address": "", "label": 1, "vuln_type": vuln_type,
                                  "filename": str(out_path), "status": "exists",
                                  "source": "swc_registry"})
                continue

            source = fetch_github_file(dl_url)
            time.sleep(DELAY)

            if source:
                out_path.write_text(source, encoding="utf-8")
                print(f"    ✓  {out_name}  ({len(source):,} chars)")
                new_rows.append({"address": "", "label": 1, "vuln_type": vuln_type,
                                  "filename": str(out_path), "status": "downloaded",
                                  "source": "swc_registry"})
            else:
                print(f"    ✗  {name}")


# ─────────────────────────────────────────────────────
# MERGE WITH EXISTING contracts_index.csv
# ─────────────────────────────────────────────────────

def merge_index():
    existing_path = Path("contracts_index.csv")

    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        print(f"\n  Existing index: {len(existing)} rows")
    else:
        existing = pd.DataFrame(columns=["address","label","vuln_type","filename","status"])
        print(f"\n  No existing index found — creating fresh")

    # Add source column to existing if missing
    if "source" not in existing.columns:
        existing["source"] = "etherscan"

    new_df  = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicate by filename
    combined = combined.drop_duplicates(subset=["filename"], keep="first")
    combined = combined[combined["filename"].astype(str) != ""]
    combined.to_csv("contracts_index.csv", index=False)

    n_vuln = len(combined[combined["label"] == 1])
    n_safe = len(combined[combined["label"] == 0])

    print(f"\n{'='*55}")
    print(f"  contracts_index.csv updated")
    print(f"  Total    : {len(combined)}")
    print(f"  Vuln = 1 : {n_vuln}")
    print(f"  Safe = 0 : {n_safe}")
    print(f"{'='*55}")
    print("\nFull vuln_type breakdown:")
    print(combined[combined["label"]==1]["vuln_type"].value_counts().to_string())


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    pull_smartbugs_curated()
    pull_swc_registry()
    merge_index()
    print("\nDone. Run your GNN/Transformer pipeline on contracts_index.csv")