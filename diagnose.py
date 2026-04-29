import requests, pandas as pd, json, sys

API_KEY = sys.argv[1] if len(sys.argv) > 1 else "YOUR_KEY"

# ── Check CSV ──────────────────────────────────────
print("="*60)
print("CSV DIAGNOSTIC")
print("="*60)

try:
    df = pd.read_csv("dataset_output/hack_database.csv")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    if len(df) > 0:
        print(f"\nFirst row (all columns):")
        for col in df.columns:
            print(f"  {col:20s} : {str(df[col].iloc[0])[:120]}")

    if "url" in df.columns:
        print(f"\nSample 'url' values:")
        for v in df["url"].dropna().head(5):
            print(f"  {str(v)[:120]}")

    if "technique" in df.columns:
        print(f"\nSample 'technique' values:")
        for v in df["technique"].dropna().head(10):
            print(f"  {v}")

except Exception as e:
    print(f"CSV error: {e}")


# ── Check Etherscan API (V2) ───────────────────────
print("\n" + "="*60)
print("ETHERSCAN API DIAGNOSTIC (V2)")
print("="*60)

if API_KEY == "YOUR_KEY":
    print("Pass your API key: python diagnose.py YOUR_KEY")
else:
    test_addr = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

    url = "https://api.etherscan.io/v2/api"
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": test_addr,
        "apikey": API_KEY,
        "chainid": "1"   # 🔴 REQUIRED for V2
    }

    try:
        r = requests.get(url, params=params, timeout=10)

        print(f"HTTP status : {r.status_code}")
        print(f"Raw response: {r.text[:300]}")

        data = r.json()

        print(f"API status  : {data.get('status')}")
        print(f"API message : {data.get('message')}")

        result = data.get("result", [])

        if isinstance(result, list) and len(result) > 0:
            src = result[0].get("SourceCode", "")
            print(f"SourceCode length : {len(src)}")
            print(f"SourceCode preview:\n{src[:200]}")

        elif isinstance(result, str):
            print(f"Result string: {result}")

        else:
            print("Unexpected result format:", result)

    except Exception as e:
        print(f"API error: {e}")