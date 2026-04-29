"""Run this to see the raw Etherscan response right now."""
import requests

API_KEY = "8WR8V4YZ8NACCYUHKWFAI8V9MXNGEAFUSH"

r = requests.get("https://api.etherscan.io/v2/api", params={
    "chainid": 1,
    "module": "contract", "action": "getsourcecode",
    "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    "apikey": API_KEY,
}, timeout=10)

print(f"HTTP: {r.status_code}")
data = r.json()
print(f"status : {data.get('status')}")
print(f"message: {data.get('message')}")
result = data.get('result')
if isinstance(result, list) and result:
    src = result[0].get('SourceCode','')
    print(f"SourceCode length: {len(src)}")
    print(f"Preview: {src[:100]}")
else:
    print(f"result (raw): {result}")