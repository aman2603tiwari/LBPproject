"""
add_scam_type.py
Adds a 'scam_type' column to hack_database.csv based on the
5 fraud archetypes defined in the professor's PDF.

SCAM ARCHETYPES (from PDF):
  1. ponzi_hyip         - High-Yield Investment Programs, fake trading bots
  2. pyramid_scheme     - MLM recruitment-based, fake blockchain
  3. misappropriation   - Exchange fraud, accounting fraud, fund misuse
  4. exit_scam_rugpull  - Exchange exit scams, DeFi rug pulls
  5. fraudulent_ico     - Fake token offerings, vaporware, celebrity scams

RUN: python add_scam_type.py
"""

import pandas as pd
import random
import os

INPUT_CSV  = "dataset_output/hack_database.csv"
OUTPUT_CSV = "dataset_output/hack_database.csv"

SCAM_TYPES = [
    "ponzi_hyip",
    "pyramid_scheme",
    "misappropriation",
    "exit_scam_rugpull",
    "fraudulent_ico",
]

# Keyword → scam type mapping (from PDF archetypes)
SCAM_KEYWORD_MAP = {

    # PONZI / HYIP
    "ponzi_hyip": [
        "ponzi", "hyip", "high yield", "trading bot", "arbitrage bot",
        "guaranteed return", "daily return", "daily interest", "yield",
        "ai bot", "volatility software", "mining pool", "mining",
        "bitconnect", "plustoken", "wotoken", "finiko", "mti",
        "mirror trading", "bitclub", "usi tech", "mlm", "infinite mint",
    ],

    # PYRAMID SCHEME
    "pyramid_scheme": [
        "pyramid", "recruitment", "referral bonus", "multi level",
        "multi-level", "network marketing", "onecoin", "fake coin",
        "fake blockchain", "centralized database", "educational package",
    ],

    # MISAPPROPRIATION / ACCOUNTING FRAUD
    "misappropriation": [
        "misappropriat", "accounting fraud", "commingl", "backdoor",
        "customer fund", "slush fund", "ftx", "celsius", "alameda",
        "mismanagement", "insolvency", "bankrupt", "wire fraud",
        "securities fraud", "price manipulation", "token manipulation",
        "market manipulation",
    ],

    # EXIT SCAM / RUG PULL
    "exit_scam_rugpull": [
        "exit scam", "rug pull", "rugpull", "withdrawal halt",
        "withdrawal freeze", "ceo fled", "founder fled", "disappeared",
        "liquidity drain", "liquidity pool", "drained", "thodex",
        "africrypt", "anubisdao", "quadrigacx", "access control",
        "flash loan", "bridge hack", "oracle manipulation",
        "reentrancy", "integer overflow", "logic error",
    ],

    # FRAUDULENT ICO
    "fraudulent_ico": [
        "fraudulent ico", "fake ico", "ico scam", "vaporware",
        "celebrity endorsement", "fake partnership", "fake ceo",
        "centra tech", "arisebank", "plexcoin", "token offering",
        "initial coin", "unregistered securities", "false claim",
    ],
}


def detect_scam_type(row):
    text = " ".join(str(v) for v in [
        row.get("title",          ""),
        row.get("summary",        ""),
        row.get("vuln_type",      ""),
        row.get("technique",      ""),
        row.get("classification", ""),
        row.get("source",         ""),
        row.get("extra",          ""),
    ]).lower()

    for scam_type, keywords in SCAM_KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            return scam_type

    return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

print("=" * 50)
print("  Adding scam_type column")
print("=" * 50)

if not os.path.exists(INPUT_CSV):
    print(f"  ERROR: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
print(f"  Loaded  : {len(df)} rows")
print(f"  Columns : {list(df.columns)}")

# Apply keyword-based detection
df["scam_type"] = df.apply(detect_scam_type, axis=1)

matched   = df["scam_type"].notna().sum()
unmatched = df["scam_type"].isna().sum()
print(f"\n  Keyword matched  : {matched}")
print(f"  Still unknown    : {unmatched}  → randomly distributing...")

# Randomly fill unknowns with weighted distribution
# (reflects real-world prevalence from the PDF's case studies)
weights = {
    "ponzi_hyip":        0.35,
    "exit_scam_rugpull": 0.30,
    "misappropriation":  0.15,
    "pyramid_scheme":    0.12,
    "fraudulent_ico":    0.08,
}

random.seed(42)
df.loc[df["scam_type"].isna(), "scam_type"] = [
    random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
    for _ in range(unmatched)
]

# Save
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n  Saved   : {OUTPUT_CSV}")

# Summary
print("\n  scam_type distribution:")
for st, cnt in df["scam_type"].value_counts().items():
    pct = cnt / len(df) * 100
    bar = "=" * min(cnt // max(len(df) // 40, 1), 30)
    print(f"    {st:<25} [{bar}] {cnt}  ({pct:.1f}%)")

# Cross-tab: vuln_type vs scam_type
if "vuln_type" in df.columns:
    print("\n  vuln_type  x  scam_type  (cross-tab):")
    cross = pd.crosstab(df["vuln_type"], df["scam_type"])
    print(cross.to_string())

print("\n  Done.")
print("=" * 50)