"""
Phase 2 — Contract Collection (v4)
====================================
Fix: Etherscan migrated to v2 API — now requires chainid=1 for Ethereum mainnet
     URL changed: api.etherscan.io/api  →  api.etherscan.io/v2/api
"""

import re, time, json, requests, pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────
ETHERSCAN_API_KEY = "8WR8V4YZ8NACCYUHKWFAI8V9MXNGEAFUSH"
ETHERSCAN_BASE    = "https://api.etherscan.io/v2/api"   # <-- v2
CHAIN_ID          = 1                                    # <-- Ethereum mainnet
REQUEST_DELAY     = 0.25

ETH_ADDR_RE = re.compile(r'\b(0x[a-fA-F0-9]{40})\b')

# ─────────────────────────────────────────────────────
SAFE_CONTRACTS = [
    {"address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "name": "uniswap_v2_router"},
    {"address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f", "name": "uniswap_v2_factory"},
    {"address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9", "name": "aave_lending_pool_v2"},
    {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "name": "usdc_token"},
    {"address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "name": "dai_token"},
    {"address": "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5", "name": "compound_ceth"},
    {"address": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419", "name": "chainlink_eth_usd"},
    {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "name": "weth_token"},
    {"address": "0xE592427A0AEce92De3Edee1F18E0157C05861564", "name": "uniswap_v3_router"},
    {"address": "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7", "name": "curve_3pool"},
    {"address": "0x1F98431c8aD98523631AE4a59f267346ea31F984", "name": "uniswap_v3_factory"},
    {"address": "0xBA12222222228d8Ba445958a75a0704d566BF2C8", "name": "balancer_vault"},
    {"address": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F", "name": "sushiswap_router"},
    {"address": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84", "name": "lido_steth"},
    {"address": "0x00000000219ab540356cBB839Cbe05303d7705Fa", "name": "eth2_deposit"},
]

# ─────────────────────────────────────────────────────
# Vulnerable contracts — these are the VICTIM protocol
# contracts (not attacker contracts), so they ARE
# verified on Etherscan
# ─────────────────────────────────────────────────────
KNOWN_VULNERABLE = [
    # ── Reentrancy ────────────────────────────────────
    # The DAO — original reentrancy hack 2016
    {"address": "0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413", "vuln_type": "reentrancy",           "name": "the_dao"},
    # Uniswap V1 — had reentrancy via ERC777 tokens
    {"address": "0xc0a47dFe034B400B47bDaD5FecDa2621de6c4d95", "vuln_type": "reentrancy",           "name": "uniswap_v1"},
    # Lendf.me (dForce) reentrancy 2020
    {"address": "0x0eEe3E3828A45f7601D5F54bF49bB01d1A9dF5ea", "vuln_type": "reentrancy",           "name": "lendf_dforce"},
    # Cream Finance reentrancy
    {"address": "0x780F75ad0B02afeb6039672E6a6CEDe7447a8b45", "vuln_type": "reentrancy",           "name": "cream_finance"},
    # Rari Capital reentrancy 2022
    {"address": "0xd4Bc1AFAb4b7E2B8b3c98F0d61B7EBa3e1Dfd5C7", "vuln_type": "reentrancy",           "name": "rari_capital"},

    # ── Flash Loan ────────────────────────────────────
    # bZx protocol (flash loan + oracle)
    {"address": "0x9441D7556e7820B5ca42082cfa99487D56AcA958", "vuln_type": "flash_loan",           "name": "bzx_fulcrum"},
    # Harvest Finance flash loan 2020
    {"address": "0x3FDA67f7583380E67ef93072294a7fAc882FD7E7", "vuln_type": "flash_loan",           "name": "harvest_finance"},
    # PancakeBunny flash loan 2021
    {"address": "0xb4B1d6a89564564E39074c17CBCbD9FDDec5cA8F", "vuln_type": "flash_loan",           "name": "pancakebunny"},
    # Euler Finance flash loan 2023
    {"address": "0x27182842E098f60e3D576794A5bFFb0777E025d3", "vuln_type": "flash_loan",           "name": "euler_finance"},

    # ── Access Control ────────────────────────────────
    # Parity Multisig Wallet 2017 (the self-destruct one)
    {"address": "0x863DF6BFa4469f3ead0bE8f9F2AAE51c91A907b4", "vuln_type": "access_control",      "name": "parity_multisig"},
    # Poly Network 2021 — largest DeFi hack
    {"address": "0x250e76987d838a75310c34bf422ea9f1AC4Cc906", "vuln_type": "access_control",      "name": "poly_network"},
    # Ronin Bridge (Axie Infinity) 2022
    {"address": "0x1A2a1c938CE3eC39b6D47113c7955bAa9DD454F2", "vuln_type": "access_control",      "name": "ronin_bridge"},

    # ── Oracle Manipulation ───────────────────────────
    # Compound COMP oracle manipulation 2020
    {"address": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B", "vuln_type": "oracle_manipulation", "name": "compound_comptroller"},
    # Mango Markets oracle manipulation 2022
    {"address": "0x7A66b5c2e29E7592504AdaBDa4bB55A64A656Df2", "vuln_type": "oracle_manipulation", "name": "mango_markets"},
    # Cheese Bank oracle attack 2020
    {"address": "0xb30dE43B8BFE1A6966bfaE0c58a1D3e552ceB1E8", "vuln_type": "oracle_manipulation", "name": "cheese_bank"},

    # ── Integer Overflow ──────────────────────────────
    # BeautyChain BEC token overflow 2018
    {"address": "0xC5d105E63711398aF9bbFF092d4B6769C82F793D", "vuln_type": "integer_overflow",    "name": "beautychain_bec"},
    # SMT token overflow 2018
    {"address": "0x55f93985431Fc9304077687a35A1BA103dC1e081", "vuln_type": "integer_overflow",    "name": "smt_token"},
    # UET token overflow
    {"address": "0x27f706edde3aD952EF647Dd67E24e38CD0803DD6", "vuln_type": "integer_overflow",    "name": "uet_token"},

    # ── Logic Error ───────────────────────────────────
    # Akutars NFT — locked ETH logic error 2022
    {"address": "0xF42c318dbfBaab0EEE040279C6a2588Fa01a961d", "vuln_type": "logic_error",         "name": "akutars_nft"},
    # Optimism - mint logic error 2022
    {"address": "0x4200000000000000000000000000000000000042", "vuln_type": "logic_error",         "name": "optimism_token"},

    # ── Bridge Hack ───────────────────────────────────
    # Wormhole bridge 2022
    {"address": "0x98f3c9e6E3fAce36bAAd05FE09d375Ef1464288B", "vuln_type": "bridge_hack",        "name": "wormhole_bridge"},
    # Nomad bridge 2022
    {"address": "0x5D94309E5a0090b165FA4181519701637B6DAEBA", "vuln_type": "bridge_hack",        "name": "nomad_bridge"},
]


# ─────────────────────────────────────────────────────
# ETHERSCAN v2
# ─────────────────────────────────────────────────────

def fetch_source_code(address):
    """v2 API — chainid is now required."""
    try:
        r = requests.get(ETHERSCAN_BASE, params={
            "chainid": CHAIN_ID,          # required in v2
            "module":  "contract",
            "action":  "getsourcecode",
            "address": address,
            "apikey":  ETHERSCAN_API_KEY,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"      [NET ERROR] {e}")
        return None

    if data.get("status") != "1":
        # Uncomment next line to debug specific errors:
        # print(f"      API msg: {data.get('message')}  result: {data.get('result')}")
        return None

    result = data.get("result", [])
    if not result or not isinstance(result, list):
        return None

    source = result[0].get("SourceCode", "").strip()
    if not source:
        return None

    # Multi-file JSON unwrapping (Etherscan wraps with {{ }})
    if source.startswith("{"):
        try:
            inner  = source[1:-1] if source.startswith("{{") else source
            parsed = json.loads(inner)
            srcs   = parsed.get("sources", parsed)
            parts  = [v.get("content","") for v in srcs.values()
                      if isinstance(v, dict) and "content" in v]
            if parts:
                source = "\n\n// ===== NEXT FILE =====\n\n".join(parts)
        except Exception:
            pass

    return source


def save_contract(source, folder, filename):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(source, encoding="utf-8")
    return path


def download_batch(entries, out_dir, label, label_name):
    rows  = []
    dl    = 0
    skip  = 0
    total = len(entries)

    for i, entry in enumerate(entries):
        addr     = entry["address"]
        vuln     = entry.get("vuln_type", "safe")
        short    = addr[:10].lower()
        safe_v   = re.sub(r'[^a-z0-9_]', '_', vuln.lower())
        filename = f"{short}_{safe_v}.sol"
        out_path = out_dir / filename

        if out_path.exists():
            print(f"  [{i+1}/{total}] SKIP  {filename}")
            rows.append({"address": addr, "label": label, "vuln_type": vuln,
                         "filename": str(out_path), "status": "exists"})
            skip += 1
            continue

        print(f"  [{i+1}/{total}] {addr}  ({vuln})")
        source = fetch_source_code(addr)
        time.sleep(REQUEST_DELAY)

        if source:
            save_contract(source, out_dir, filename)
            print(f"      ✓  {filename}  ({len(source):,} chars)")
            rows.append({"address": addr, "label": label, "vuln_type": vuln,
                         "filename": str(out_path), "status": "downloaded"})
            dl += 1
        else:
            print(f"      ✗  Not verified on Etherscan")
            rows.append({"address": addr, "label": label, "vuln_type": vuln,
                         "filename": "", "status": "not_verified"})
            skip += 1

    print(f"\n  {label_name}: {dl} downloaded, {skip} skipped")
    return rows


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def collect_contracts():
    vuln_dir = Path("contracts/vulnerable")
    safe_dir  = Path("contracts/safe")
    vuln_dir.mkdir(parents=True, exist_ok=True)
    safe_dir.mkdir(parents=True, exist_ok=True)

    print("\n[VULNERABLE]")
    vuln_rows = download_batch(KNOWN_VULNERABLE, vuln_dir, label=1, label_name="Vulnerable")

    print("\n[SAFE]")
    safe_rows = download_batch(SAFE_CONTRACTS, safe_dir, label=0, label_name="Safe")

    all_rows = vuln_rows + safe_rows

    if not all_rows:
        print("\n[!] Nothing downloaded. Check API key / network.")
        return

    df       = pd.DataFrame(all_rows)
    verified = df[df["filename"].astype(str) != ""].copy()
    verified.to_csv("contracts_index.csv", index=False)

    n_vuln = len(verified[verified["label"] == 1])
    n_safe = len(verified[verified["label"] == 0])

    print(f"\n{'='*55}")
    print(f"  contracts_index.csv written")
    print(f"  Total  : {len(verified)}")
    print(f"  Vuln=1 : {n_vuln}")
    print(f"  Safe=0 : {n_safe}")
    print(f"{'='*55}")

    if n_vuln > 0:
        print("\nVuln type breakdown:")
        print(verified[verified["label"]==1]["vuln_type"].value_counts().to_string())


if __name__ == "__main__":
    collect_contracts()