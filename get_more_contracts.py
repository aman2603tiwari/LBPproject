"""
get_more_contracts.py
======================
Collects additional labelled Solidity contracts to expand
the dataset beyond 175 samples.

Target: 100+ samples per class minimum

Sources:
  1. SmartBugs Wild   — 47k real contracts labelled by Slither
  2. SWC Registry     — official vulnerability examples
  3. DeFiVulnLabs    — real DeFi PoC vulnerability contracts
  4. Etherscan API    — more real-world contracts
  5. Safe examples    — OpenZeppelin, Solidity docs

RUN: python get_more_contracts.py
"""

import requests
import os
import re
import json
import time
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
ETHERSCAN_API_KEY = "8WR8V4YZ8NACCYUHKWFAI8V9MXNGEAFUSH"
OUTPUT_DIR        = "contracts"
INDEX_CSV         = "contracts_index.csv"
DELAY             = 0.3

UA = {"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"}

os.makedirs(f"{OUTPUT_DIR}/vulnerable", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/safe",       exist_ok=True)

# ── Label map ─────────────────────────────────────────────────
VULN_LABEL_MAP = {
    "reentrancy":          1,
    "access_control":      2,
    "integer_overflow":    3,
    "logic_error":         4,
    "other_vuln":          5,
    "safe":                0,
}

# ── Existing contracts (to avoid duplicates) ──────────────────
def load_existing():
    if not os.path.exists(INDEX_CSV):
        return set(), []
    df = pd.read_csv(INDEX_CSV)
    existing_files = set(df["filename"].tolist())
    return existing_files, df.to_dict("records")

existing_files, existing_records = load_existing()
new_records = []

def already_exists(filename):
    return filename in existing_files

def save_and_index(source_code, filename, folder,
                   vuln_type, label, source_name, amount="N/A"):
    path = os.path.join(OUTPUT_DIR, folder, filename)
    if already_exists(f"{folder}/{filename}"):
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(source_code)
    new_records.append({
        "address":       "N/A",
        "label":         label,
        "vuln_type":     vuln_type,
        "scam_type":     "none",
        "contract_name": Path(filename).stem,
        "filename":      f"{folder}/{filename}",
        "source_lines":  source_code.count("\n"),
        "amount_usd":    amount,
        "source":        source_name,
        "fetched_at":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    existing_files.add(f"{folder}/{filename}")
    return True


# ══════════════════════════════════════════════════════════════
#  SOURCE 1 — SWC Registry GitHub Examples
#  Each SWC entry has Solidity example files showing the vuln
# ══════════════════════════════════════════════════════════════

SWC_VULN_MAP = {
    "SWC-107": "reentrancy",
    "SWC-101": "integer_overflow",
    "SWC-105": "access_control",
    "SWC-106": "access_control",
    "SWC-115": "access_control",
    "SWC-112": "logic_error",
    "SWC-113": "logic_error",
    "SWC-116": "logic_error",
    "SWC-120": "logic_error",
    "SWC-104": "logic_error",
    "SWC-110": "logic_error",
    "SWC-114": "logic_error",
    "SWC-128": "logic_error",
    "SWC-132": "other_vuln",
    "SWC-100": "access_control",
    "SWC-108": "access_control",
}


def fetch_swc_examples():
    print("\n" + "="*55)
    print("  SOURCE 1: SWC Registry Examples")
    print("="*55)

    count = 0
    base  = "https://api.github.com/repos/SmartContractSecurity/SWC-registry/contents/entries"

    r = requests.get(base, headers=UA, timeout=15)
    if r.status_code != 200:
        print(f"  [!] GitHub API returned {r.status_code}")
        return 0

    entries = r.json()
    print(f"  Found {len(entries)} SWC entries")

    for entry in entries:
        swc_id   = entry["name"].upper()          # e.g. SWC-107
        vuln_type = SWC_VULN_MAP.get(swc_id, "logic_error")

        # Get files inside this SWC folder
        folder_url = entry["url"]
        r2 = requests.get(folder_url, headers=UA, timeout=15)
        if r2.status_code != 200:
            time.sleep(0.5)
            continue

        files = r2.json()
        for f in files:
            if isinstance(f, str):
                continue
            fname = f.get("name", "")
            if not fname.endswith(".sol"):
                continue

            # Skip fixed/safe versions (filename contains "fixed" or "safe")
            is_safe = any(x in fname.lower() for x in ["fixed", "_safe", "correct"])
            folder  = "safe" if is_safe else "vulnerable"
            label   = 0 if is_safe else VULN_LABEL_MAP.get(vuln_type, 5)
            actual_vuln = "safe" if is_safe else vuln_type

            # Fetch raw content
            raw_url = f["download_url"]
            if not raw_url:
                continue

            r3 = requests.get(raw_url, headers=UA, timeout=15)
            if r3.status_code != 200:
                time.sleep(0.3)
                continue

            out_name = f"swc_{swc_id.lower()}_{fname}"
            if save_and_index(r3.text, out_name, folder,
                              actual_vuln, label, "SWC Registry"):
                count += 1
                print(f"  [{swc_id}] {fname} → {actual_vuln}")

            time.sleep(DELAY)

    print(f"  Total from SWC Registry: {count}")
    return count


# ══════════════════════════════════════════════════════════════
#  SOURCE 2 — DeFiVulnLabs PoC Contracts
#  Real DeFi vulnerability proof-of-concept contracts
# ══════════════════════════════════════════════════════════════

DEFI_VULN_KEYWORDS = {
    "reentrancy":       ["reentrancy", "reentrant", "reentr"],
    "flash_loan":       ["flashloan", "flash_loan", "flash-loan"],
    "access_control":   ["access", "ownable", "privilege", "auth"],
    "integer_overflow": ["overflow", "underflow", "arithmetic"],
    "oracle_manipulation": ["oracle", "price", "manipulation"],
    "logic_error":      ["logic", "incorrect", "wrong", "bug"],
}


def detect_vuln_from_name(filename):
    name = filename.lower()
    for vuln, keywords in DEFI_VULN_KEYWORDS.items():
        if any(k in name for k in keywords):
            return vuln
    return "logic_error"


def fetch_defivulnlabs():
    print("\n" + "="*55)
    print("  SOURCE 2: DeFiVulnLabs")
    print("="*55)

    count = 0
    # DeFiVulnLabs stores test contracts in src/test/
    api_url = "https://api.github.com/repos/SunWeb3Sec/DeFiVulnLabs/contents/src/test"
    r = requests.get(api_url, headers=UA, timeout=15)
    if r.status_code != 200:
        print(f"  [!] GitHub API {r.status_code}")
        return 0

    files = r.json()
    sol_files = [f for f in files if f["name"].endswith(".sol")]
    print(f"  Found {len(sol_files)} .sol files")

    for f in sol_files:
        fname     = f["name"]
        raw_url   = f["download_url"]
        vuln_type = detect_vuln_from_name(fname)
        label     = VULN_LABEL_MAP.get(vuln_type, 5)

        r2 = requests.get(raw_url, headers=UA, timeout=15)
        if r2.status_code != 200:
            time.sleep(0.3)
            continue

        out_name = f"defivuln_{fname}"
        if save_and_index(r2.text, out_name, "vulnerable",
                          vuln_type, label, "DeFiVulnLabs"):
            count += 1
            print(f"  {fname} → {vuln_type}")

        time.sleep(DELAY)

    print(f"  Total from DeFiVulnLabs: {count}")
    return count


# ══════════════════════════════════════════════════════════════
#  SOURCE 3 — More Safe Contracts via Etherscan
#  Expand the safe class (currently only 15 samples)
# ══════════════════════════════════════════════════════════════

MORE_SAFE_CONTRACTS = [
    # Uniswap V3 core contracts
    {"address": "0x1F98431c8aD98523631AE4a59f267346ea31F984", "name": "UniV3_Factory"},
    {"address": "0xE592427A0AEce92De3Edee1F18E0157C05861564", "name": "UniV3_Router"},
    {"address": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88", "name": "UniV3_Positions"},
    # Compound V3
    {"address": "0xc3d688B66703497DAA19211EEdff47f25384cdc3", "name": "Compound_V3_USDC"},
    {"address": "0x316f9708bB98af7dA9c68C1C3b5e79039cD336E3", "name": "Compound_Bulker"},
    # Curve Finance
    {"address": "0xD51a44d3FaE010294C616388b506AcdA1bfAAE46", "name": "Curve_TriCrypto"},
    {"address": "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7", "name": "Curve_3Pool"},
    # Lido
    {"address": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84", "name": "Lido_stETH"},
    {"address": "0x47EbaB13B806773ec2A2d16873e2dF770D130b50", "name": "Lido_NodeOperators"},
    # MakerDAO
    {"address": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "name": "MakerDAO_MKR"},
    {"address": "0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B", "name": "MakerDAO_VAT"},
    # Balancer
    {"address": "0xBA12222222228d8Ba445958a75a0704d566BF2C8", "name": "Balancer_Vault"},
    # OpenZeppelin standard implementations
    {"address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "name": "OZ_USDT"},
    {"address": "0x514910771AF9Ca656af840dff83E8264EcF986CA", "name": "Chainlink_LINK_token"},
    # Arbitrum bridge (safe)
    {"address": "0x72Ce9c846789fdB6fC1f34aC4AD25Dd9ef7031ef", "name": "Arbitrum_GatewayRouter"},
    # 1inch
    {"address": "0x1111111254EEB25477B68fb85Ed929f73A960582", "name": "1inch_Router_V5"},
    # ENS Resolver
    {"address": "0x4976fb03C32e5B8cfe2b6cCB31c09Ba78EBaBa41", "name": "ENS_PublicResolver"},
    # Opensea Seaport
    {"address": "0x00000000000000ADc04C56Bf30aC9d3c0aAF14dC", "name": "Seaport_1_5"},
    # USDC
    {"address": "0x43506849D7C04F9138D1A2050bbF3A0c054402dd", "name": "USDC_Proxy"},
    # Aave V3
    {"address": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2", "name": "Aave_V3_Pool"},
]


def fetch_more_safe():
    print("\n" + "="*55)
    print("  SOURCE 3: More Safe Contracts (Etherscan)")
    print("="*55)

    count = 0
    base  = "https://api.etherscan.io/v2/api"

    for target in MORE_SAFE_CONTRACTS:
        addr = target["address"]
        name = target["name"]
        print(f"  Fetching {name} ({addr[:10]}...)", end=" ")

        params = {
            "chainid": 1,
            "module":  "contract",
            "action":  "getsourcecode",
            "address": addr,
            "apikey":  ETHERSCAN_API_KEY,
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            data = r.json()
            if data.get("status") != "1":
                print("no source")
                time.sleep(DELAY)
                continue

            result = data["result"][0]
            source = result.get("SourceCode", "")
            if not source or source.strip() == "":
                print("empty")
                time.sleep(DELAY)
                continue

            # Handle JSON bundle
            if source.startswith("{{"):
                try:
                    bundle = json.loads(source[1:-1])
                    parts  = bundle.get("sources", {})
                    source = "\n".join(
                        f"// FILE: {k}\n{v.get('content','')}"
                        for k, v in parts.items()
                    )
                except:
                    pass

            filename = f"{addr[:10]}_{name[:20]}.sol"
            if save_and_index(source, filename, "safe",
                              "safe", 0, "Etherscan"):
                count += 1
                print("✓")
            else:
                print("already exists")

        except Exception as e:
            print(f"error: {e}")

        time.sleep(DELAY)

    print(f"  Total new safe contracts: {count}")
    return count


# ══════════════════════════════════════════════════════════════
#  SOURCE 4 — SmartBugs Wild (subset)
#  Pull categorised vulnerable contracts from the wild dataset
#  Only pulling the labelled subset, not all 47k
# ══════════════════════════════════════════════════════════════

SMARTBUGS_WILD_DIRS = {
    "reentrancy":       "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/reentrancy",
    "access_control":   "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/access_control",
    "arithmetic":       "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/arithmetic",
    "unchecked_calls":  "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/unchecked_calls",
    "denial_of_service":"https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/denial_of_service",
    "time_manipulation":"https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/time_manipulation",
    "front_running":    "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/front_running",
    "bad_randomness":   "https://api.github.com/repos/smartbugs/smartbugs-wild/contents/contracts/bad_randomness",
}

WILD_LABEL_MAP = {
    "reentrancy":        "reentrancy",
    "access_control":    "access_control",
    "arithmetic":        "integer_overflow",
    "unchecked_calls":   "logic_error",
    "denial_of_service": "logic_error",
    "time_manipulation": "logic_error",
    "front_running":     "logic_error",
    "bad_randomness":    "logic_error",
}

MAX_PER_CATEGORY = 80   # cap per category to avoid imbalance


def fetch_smartbugs_wild():
    print("\n" + "="*55)
    print("  SOURCE 4: SmartBugs Wild (labelled subset)")
    print("="*55)

    total = 0

    for category, api_url in SMARTBUGS_WILD_DIRS.items():
        vuln_type = WILD_LABEL_MAP[category]
        label     = VULN_LABEL_MAP[vuln_type]

        print(f"\n  [{category}] → {vuln_type}")

        r = requests.get(api_url, headers=UA, timeout=15)
        if r.status_code == 404:
            print(f"    Directory not found — skipping")
            continue
        if r.status_code != 200:
            print(f"    API error {r.status_code}")
            time.sleep(1)
            continue

        files     = r.json()
        sol_files = [f for f in files if f.get("name", "").endswith(".sol")]
        # Cap to avoid flooding one class
        sol_files = sol_files[:MAX_PER_CATEGORY]
        print(f"    Fetching {len(sol_files)} files ...")

        count = 0
        for f in sol_files:
            raw_url = f.get("download_url")
            if not raw_url:
                continue

            r2 = requests.get(raw_url, headers=UA, timeout=15)
            if r2.status_code != 200:
                time.sleep(0.3)
                continue

            fname    = f["name"]
            out_name = f"wild_{category[:6]}_{fname}"
            if save_and_index(r2.text, out_name, "vulnerable",
                              vuln_type, label, "SmartBugs-Wild"):
                count += 1

            time.sleep(DELAY)

        print(f"    Added {count} contracts")
        total += count

    print(f"\n  Total from SmartBugs Wild: {total}")
    return total


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  Getting More Contracts")
    print("="*55)

    counts = {}
    counts["SWC Registry"]     = fetch_swc_examples()
    counts["DeFiVulnLabs"]     = fetch_defivulnlabs()
    counts["More Safe"]        = fetch_more_safe()
    counts["SmartBugs Wild"]   = fetch_smartbugs_wild()

    # Merge new records with existing
    all_records = existing_records + new_records
    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["filename"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(INDEX_CSV, index=False, encoding="utf-8")

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"\n  New contracts added:")
    for src, cnt in counts.items():
        print(f"    {src:<25} {cnt}")

    total_new = sum(counts.values())
    print(f"\n  Total new              : {total_new}")
    print(f"  Previous total         : {len(existing_records)}")
    print(f"  New total in index     : {len(df)}")

    print("\n  Class distribution:")
    for vt, cnt in df["vuln_type"].value_counts().items():
        bar = "█" * min(cnt // 5, 30)
        print(f"    {vt:<25} {bar} {cnt}")

    print(f"\n  Next steps:")
    print(f"  1. Delete graphs/ folder")
    print(f"  2. python build_graphs.py")
    print(f"  3. python train_gnn.py")
    print(f"  4. python evaluate.py")


if __name__ == "__main__":
    main()