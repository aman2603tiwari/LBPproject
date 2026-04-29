"""
diagnostic.py
Run this FIRST to see exactly what each API/source returns.
This tells us which sources are working and what their real data structure looks like.

RUN: python diagnostic.py
"""

import requests
import json
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
}

def check(label, url, is_json=True):
    print(f"\n{'─'*60}")
    print(f"  CHECKING: {label}")
    print(f"  URL: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        print(f"  STATUS: {resp.status_code}")
        if resp.status_code != 200:
            print(f"  ✗ FAILED")
            return

        if is_json:
            data = resp.json()
            if isinstance(data, list):
                print(f"  TYPE: list  |  LENGTH: {len(data)}")
                if data:
                    print(f"  FIRST ITEM KEYS: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
                    print(f"  FIRST ITEM:\n{json.dumps(data[0], indent=4)[:600]}")
            elif isinstance(data, dict):
                print(f"  TYPE: dict  |  TOP KEYS: {list(data.keys())}")
                # Check if any key holds a list
                for k, v in data.items():
                    if isinstance(v, list):
                        print(f"  LIST KEY '{k}': {len(v)} items")
                        if v:
                            print(f"  SAMPLE ITEM:\n{json.dumps(v[0], indent=4)[:600]}")
                        break
        else:
            print(f"  CONTENT (first 500 chars):\n{resp.text[:500]}")

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
    time.sleep(1)


print("=" * 60)
print("  HACK DATABASE SOURCE DIAGNOSTIC")
print("=" * 60)

# ── DeFi Llama ──────────────────────────────────────────────
check("DeFi Llama Hacks API (v1)",
      "https://api.llama.fi/hacks")

check("DeFi Llama Hacks API (v2)",
      "https://api.llama.fi/v2/hacks")

check("DeFi Llama Raises/Hacks",
      "https://api.llama.fi/raises")

# ── DeFiYield ───────────────────────────────────────────────
check("DeFiYield REKT (page 1)",
      "https://api.de.fi/v1/rekt?page=1&pageSize=10")

check("DeFiYield REKT (alternate)",
      "https://de.fi/api/v1/rekt?page=1&pageSize=10")

# ── Rekt News ───────────────────────────────────────────────
check("Rekt News RSS",
      "https://rekt.news/rss.xml", is_json=False)

# ── SlowMist ────────────────────────────────────────────────
check("SlowMist GitHub JSON",
      "https://raw.githubusercontent.com/slowmist/Blockchain-dark-forest-selfguard-handbook/main/README.md",
      is_json=False)

# ── GitHub Datasets ─────────────────────────────────────────
check("SmartBugs Dataset (GitHub)",
      "https://api.github.com/repos/smartbugs/smartbugs/contents/dataset",
      is_json=True)

check("SWC Registry JSON",
      "https://raw.githubusercontent.com/SmartContractSecurity/SWC-registry/master/SWC-registry.json")

print("\n" + "=" * 60)
print("  DIAGNOSTIC COMPLETE")
print("  Use the output above to see which sources are alive")
print("  and what their real field names are.")
print("=" * 60)
