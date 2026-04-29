"""
Hack DB Scraper — Final Fix
RUN: python hack.py
"""

import requests
import pandas as pd
import re
import os
from datetime import datetime

OUTPUT = "dataset_output/hack_database.csv"
os.makedirs("dataset_output", exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

def get_vuln(text):
    t = str(text).lower()
    if "reentrancy"    in t: return "reentrancy"
    if "flash loan"    in t: return "flash_loan"
    if "flashloan"     in t: return "flash_loan"
    if "oracle"        in t: return "oracle_manipulation"
    if "overflow"      in t: return "integer_overflow"
    if "infinite mint" in t: return "integer_overflow"
    if "access"        in t: return "access_control"
    if "private key"   in t: return "access_control"
    if "bridge"        in t: return "bridge_hack"
    if "rug"           in t: return "rug_pull"
    if "phishing"      in t: return "phishing"
    if "front"         in t: return "front_running"
    if "logic"         in t: return "logic_error"
    if "price manip"   in t: return "oracle_manipulation"
    return "unknown"

def fmt_usd(val):
    try:
        val = float(val)
        if val <= 0: return "N/A"
        if val >= 1e9: return f"${val/1e9:.2f}B"
        if val >= 1e6: return f"${val/1e6:.2f}M"
        if val >= 1e3: return f"${val/1e3:.1f}K"
        return f"${val:,.0f}"
    except:
        return "N/A"

def fmt_date(ts):
    try:
        if isinstance(ts, (int, float)) and ts > 0:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return str(ts)[:10]
    except:
        return "N/A"

# ── SOURCE 1: DeFi Llama ─────────────────────────────────
def fetch_defillama():
    print("\n[1/2] DeFi Llama API...")
    try:
        data = requests.get("https://api.llama.fi/hacks",
                            headers=UA, timeout=15).json()
        print(f"      {len(data)} items received")

        rows = []
        for item in data:
            # FIX: chain can be empty list — handle safely
            chain_raw = item.get("chain") or []
            if isinstance(chain_raw, list):
                chain = chain_raw[0] if len(chain_raw) > 0 else "Unknown"
            else:
                chain = str(chain_raw) or "Unknown"

            classif   = str(item.get("classification") or "")
            technique = str(item.get("technique")      or "")
            vuln      = "bridge_hack" if item.get("bridgeHack") else get_vuln(classif + " " + technique)

            rows.append({
                "source":    "DeFi Llama",
                "title":     str(item.get("name", "Unknown")) + " Hack",
                "date":      fmt_date(item.get("date")),
                "amount":    fmt_usd(item.get("amount", 0)),
                "vuln_type": vuln,
                "chain":     chain,
                "language":  str(item.get("language") or "N/A"),
                "technique": technique,
                "url":       str(item.get("source")   or "N/A"),
                "summary":   f"{classif} via {technique} on {chain}",
            })

        print(f"      {len(rows)} records parsed")
        return rows
    except Exception as e:
        print(f"      FAILED: {e}")
        return []


# ── SOURCE 2: DeFiHackLabs README markdown ───────────────
def fetch_defihacklabs():
    print("\n[2/2] DeFiHackLabs (GitHub)...")
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/SunWeb3Sec/DeFiHackLabs/main/README.md",
            headers=UA, timeout=20)
        print(f"      Status {r.status_code}, {len(r.text)} chars")

        rows = []
        # Print first 20 lines with | to understand the format
        pipe_lines = [l.strip() for l in r.text.split("\n") if "|" in l]
        print(f"      Lines with '|': {len(pipe_lines)}")
        print(f"      Sample line: {pipe_lines[5] if len(pipe_lines) > 5 else 'N/A'}")

        for line in pipe_lines:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]  # remove empty strings

            if len(cells) < 3:
                continue
            # Skip header/separator rows
            if all(re.match(r"^[-: ]+$", c) for c in cells):
                continue
            if any(h in cells[0].lower() for h in ["date", "project", "no.", "#", "id"]):
                continue

            # Extract markdown link [text](url) from any cell
            def parse(t):
                m = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", t)
                return (m.group(1).strip(), m.group(2)) if m else (t.strip(), "N/A")

            # Try to find protocol name and amount from cells
            title_text, link = parse(cells[0])
            title_text = re.sub(r"<[^>]+>", "", title_text).strip()

            # Look for dollar amount anywhere in the row
            full_row = " ".join(cells)
            amt_match = re.search(r"\$[\d,\.]+\s*[MKBmkb]?", full_row)
            amount = amt_match.group(0).strip() if amt_match else "N/A"

            # Look for date pattern YYYY-MM-DD or MM/DD/YYYY
            date_match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", full_row)
            date = date_match.group(0).replace("/", "-") if date_match else "N/A"

            if len(title_text) > 3 and title_text not in ["-", "---"]:
                rows.append({
                    "source":    "DeFiHackLabs",
                    "title":     title_text + " Hack",
                    "date":      date,
                    "amount":    amount,
                    "vuln_type": get_vuln(full_row),
                    "chain":     "N/A",
                    "language":  "Solidity",
                    "technique": "N/A",
                    "url":       link,
                    "summary":   f"Hack: {title_text}. Amount: {amount}.",
                })

        print(f"      {len(rows)} records parsed")
        return rows
    except Exception as e:
        print(f"      FAILED: {e}")
        return []


# ── MAIN ─────────────────────────────────────────────────
print("=" * 50)
print("  Hack DB Scraper — Final Fix")
print("=" * 50)

new_rows = []
new_rows += fetch_defillama()
new_rows += fetch_defihacklabs()

df_new = pd.DataFrame(new_rows)
print(f"\n  New records : {len(df_new)}")

# Merge with existing CSV
# FIX: deduplicate on 'title' not 'protocol' (old CSV has no 'protocol' col)
if os.path.exists(OUTPUT):
    df_old = pd.read_csv(OUTPUT)
    print(f"  Existing    : {len(df_old)}")
    df = pd.concat([df_new, df_old], ignore_index=True)
else:
    df = df_new

# FIX: only deduplicate on columns that actually exist
dedup_col = "title" if "title" in df.columns else df.columns[0]
df.drop_duplicates(subset=[dedup_col], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print(f"  Final total : {len(df)} records → {OUTPUT}")
print("\n  By source:")
src_col = "source" if "source" in df.columns else df.columns[0]
for src, cnt in df[src_col].value_counts().items():
    print(f"    {src:<28} {cnt}")
print("\n  Done.")