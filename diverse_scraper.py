"""
=============================================================
  Smart Contract Vulnerability — DIVERSE Dataset Scraper
  Project: Vulnerability Detection in Smart Contracts using GNN
  Version: 4.0  (Multi-Source)
=============================================================

  SOURCE CATEGORIES:
  ┌─────────────────────────────────────────────────────┐
  │  1. NEWS RSS      — CoinDesk, CoinTelegraph,        │
  │                     Decrypt, BeInCrypto,            │
  │                     Bitcoin Magazine, Google News   │
  │                                                     │
  │  2. HACK DATABASE — Rekt.news (dedicated hack DB)   │
  │                     DeFiYield REKT API              │
  │                     DeFi Llama Hacks API            │
  │                                                     │
  │  3. SECURITY RESEARCH — ArXiv (academic papers)     │
  │                          Trail of Bits Blog RSS     │
  │                          OpenZeppelin Blog RSS      │
  │                          Immunefi Blog RSS          │
  │                                                     │
  │  4. COMMUNITY     — Reddit r/ethdev RSS             │
  │                     Reddit r/ethereum RSS           │
  │                     Reddit r/defi RSS               │
  │                                                     │
  │  5. VULNERABILITY REGISTRY — SWC Registry (GitHub) │
  │                               DeFiYield REKT list   │
  └─────────────────────────────────────────────────────┘

  WHY DIVERSITY MATTERS FOR GNN TRAINING:
    • News         → real exploit events + labels
    • Hack DBs     → structured data (amount stolen, vuln type, date)
    • Research     → technical descriptions of vulnerability patterns
    • Community    → informal reports, early warnings, post-mortems
    • Registries   → standardized vulnerability taxonomy

  OUTPUTS:
    all_articles.csv          — all collected data (combined)
    news_articles.csv         — news only
    hack_database.csv         — structured hack records
    research_papers.csv       — academic/security research
    community_posts.csv       — reddit discussions
    dataset_report.txt        — full summary report

  SETUP:
    pip install requests beautifulsoup4 pandas lxml

  RUN:
    python diverse_scraper.py
=============================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import json
import random
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 Version/16.1 Safari/605.1.15",
]

KEYWORDS = [
    "smart contract", "reentrancy", "flash loan", "defi hack",
    "solidity", "exploit", "vulnerability", "contract hack",
    "blockchain hack", "ethereum hack", "defi exploit",
    "rug pull", "protocol hack", "bridge hack", "oracle manipulation",
    "integer overflow", "access control", "front running",
    "price manipulation", "logic error", "audit",
]

VULN_TYPES = {
    "reentrancy":           ["reentrancy", "re-entrancy", "recursive call"],
    "flash_loan":           ["flash loan", "flashloan", "flash-loan"],
    "integer_overflow":     ["overflow", "underflow", "integer", "arithmetic"],
    "access_control":       ["access control", "onlyowner", "privilege", "authorization"],
    "oracle_manipulation":  ["oracle", "price manipulation", "price oracle"],
    "front_running":        ["front.?run", "mev", "sandwich attack"],
    "bridge_hack":          ["bridge", "cross.?chain"],
    "rug_pull":             ["rug pull", "rugpull", "exit scam"],
    "logic_error":          ["logic", "business logic", "logic flaw"],
}

BASE_DELAY   = 1.5
MAX_RETRIES  = 3
RETRY_DELAYS = [5, 10, 20]

OUTPUT_DIR = "dataset_output"


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def get_headers(accept="html"):
    accept_map = {
        "html": "text/html,application/xhtml+xml,*/*;q=0.8",
        "xml":  "application/rss+xml, application/xml, text/xml, */*",
        "json": "application/json, */*",
    }
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          accept_map.get(accept, "*/*"),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection":      "keep-alive",
    }


def fetch_html(url, timeout=15):
    """Fetch URL, return BeautifulSoup (lxml parser for HTML)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=get_headers("html"), timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_DELAYS[attempt] * 3
                print(f"    [429] Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code in [403, 404]:
                print(f"    [{resp.status_code}] {url}")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.Timeout:
            time.sleep(RETRY_DELAYS[min(attempt, 2)])
        except Exception as e:
            print(f"    [ERROR] {url} → {e}")
            return None
    return None


def fetch_xml(url, timeout=15):
    """Fetch RSS/XML feed, return BeautifulSoup (lxml-xml parser)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=get_headers("xml"), timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_DELAYS[attempt] * 3
                print(f"    [429] Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code in [403, 404]:
                print(f"    [{resp.status_code}] Feed unavailable: {url}")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "lxml-xml")
        except requests.exceptions.Timeout:
            time.sleep(RETRY_DELAYS[min(attempt, 2)])
        except Exception as e:
            print(f"    [ERROR] {url} → {e}")
            return None
    return None


def fetch_json(url, timeout=15):
    """Fetch a JSON API endpoint, return parsed dict/list."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=get_headers("json"), timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_DELAYS[attempt] * 3
                time.sleep(wait)
                continue
            if resp.status_code in [403, 404]:
                print(f"    [{resp.status_code}] API unavailable: {url}")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            time.sleep(RETRY_DELAYS[min(attempt, 2)])
        except Exception as e:
            print(f"    [ERROR] {url} → {e}")
            return None
    return None


def is_relevant(text):
    return any(kw.lower() in text.lower() for kw in KEYWORDS)


def detect_vuln_type(text):
    """Detect vulnerability type from text using pattern matching."""
    text_lower = text.lower()
    for vuln_type, patterns in VULN_TYPES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return vuln_type
    return "unknown"


def get_keyword(text):
    text_lower = text.lower()
    return next((k for k in KEYWORDS if k.lower() in text_lower), "general")


def clean(text):
    if not text:
        return "N/A"
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # HTML entities
    return re.sub(r"\s+", " ", text).strip()


def parse_date(raw):
    if not raw:
        return "N/A"
    raw = clean(raw)
    formats = [
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d", "%d %b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw[:30], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw[:10]


def make_record(category, source, title, url, date="N/A",
                summary="N/A", amount_usd=None, vuln_type=None, extra=None):
    """Unified record builder for all source types."""
    combined = title + " " + summary
    return {
        "category":        category,
        "source":          source,
        "title":           clean(title),
        "url":             url,
        "date":            date,
        "summary":         clean(summary)[:400] + "..." if len(clean(summary)) > 400 else clean(summary),
        "amount_usd":      amount_usd or "N/A",
        "vuln_type":       vuln_type or detect_vuln_type(combined),
        "keyword_matched": get_keyword(combined),
        "extra":           json.dumps(extra) if extra else "{}",
        "scraped_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_category(records, filename):
    """Save a list of records to a CSV file in the output directory."""
    if not records:
        print(f"    [!] No records to save for {filename}")
        return 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df   = pd.DataFrame(records)
    df.drop_duplicates(subset=["url"],   inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    [✓] Saved {len(df)} records → {path}")
    return len(df)


def parse_rss(source, url, category="news"):
    """Generic RSS parser — handles RSS 2.0 and Atom formats."""
    soup = fetch_xml(url)
    if not soup:
        return []
    items = soup.find_all("item") or soup.find_all("entry")
    records = []
    for item in items:
        title_tag = item.find("title")
        title     = clean(title_tag.get_text()) if title_tag else ""
        if not title or len(title) < 10:
            continue

        link_tag = item.find("link")
        url_val  = ""
        if link_tag:
            url_val = link_tag.get("href") or clean(link_tag.get_text())
        if not url_val or not url_val.startswith("http"):
            guid    = item.find("guid")
            url_val = clean(guid.get_text()) if guid else "N/A"

        date_tag = (item.find("pubDate") or item.find("published") or
                    item.find("updated") or item.find("dc:date"))
        date = parse_date(date_tag.get_text() if date_tag else "")

        desc_tag = (item.find("description") or item.find("summary") or
                    item.find("content"))
        summary  = clean(desc_tag.get_text()) if desc_tag else "N/A"

        if is_relevant(title + " " + summary):
            records.append(make_record(category, source, title, url_val, date, summary))

    return records


# ═══════════════════════════════════════════════════════
#  CATEGORY 1 — NEWS RSS FEEDS
# ═══════════════════════════════════════════════════════

NEWS_RSS_FEEDS = [
    ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph",  "https://cointelegraph.com/rss"),
    ("CoinTelegraph",  "https://cointelegraph.com/tags/hacks/rss"),
    ("CoinTelegraph",  "https://cointelegraph.com/tags/smart-contracts/rss"),
    ("Decrypt.co",     "https://decrypt.co/feed"),
    ("BeInCrypto",     "https://beincrypto.com/feed/"),
    ("Bitcoin Mag",    "https://bitcoinmagazine.com/feed"),
    ("CryptoSlate",    "https://cryptoslate.com/feed/"),
    ("CryptoBriefing", "https://cryptobriefing.com/feed/"),
    # Google News RSS — most powerful: searches 1000s of outlets at once
    ("Google News",    "https://news.google.com/rss/search?q=smart+contract+hack&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=smart+contract+exploit+vulnerability&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=DeFi+hack+stolen&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=reentrancy+attack+ethereum&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=flash+loan+attack+blockchain&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=solidity+vulnerability+exploit&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=bridge+hack+crypto+millions&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=oracle+manipulation+defi&hl=en-US&gl=US&ceid=US:en"),
]


def scrape_news_rss():
    print("\n" + "═" * 55)
    print("  CATEGORY 1 — NEWS RSS FEEDS")
    print("═" * 55)
    print(f"  {len(NEWS_RSS_FEEDS)} feeds across {len(set(s for s,_ in NEWS_RSS_FEEDS))} sources\n")

    all_records = []
    source_groups = {}
    for source, url in NEWS_RSS_FEEDS:
        source_groups.setdefault(source, []).append(url)

    for source, urls in source_groups.items():
        source_records = []
        for url in urls:
            print(f"  [{source}] {url[-55:]}")
            try:
                batch = parse_rss(source, url, category="news")
                source_records += batch
                print(f"           → {len(batch)} relevant articles")
            except Exception as e:
                print(f"           → ERROR: {e}")
            time.sleep(BASE_DELAY + random.uniform(0, 0.5))

        all_records += source_records

    save_category(all_records, "news_articles.csv")
    return all_records


# ═══════════════════════════════════════════════════════
#  CATEGORY 2 — HACK DATABASES
#  These are the most valuable for labelled training data
#  because they have structured fields: amount, date, type
# ═══════════════════════════════════════════════════════

def scrape_rekt_news():
    """
    Rekt.news — the most comprehensive smart contract hack database.
    Dedicated site that covers every major DeFi exploit in detail.
    Has RSS feed + leaderboard page.
    """
    print("\n  [Rekt.news]")
    records = []

    # RSS feed first
    rss_url = "https://rekt.news/rss.xml"
    print(f"  RSS: {rss_url}")
    rss_records = parse_rss("Rekt.news", rss_url, category="hack_database")
    # All rekt.news articles are about hacks — force category
    for r in rss_records:
        r["category"] = "hack_database"
    records += rss_records
    print(f"  → {len(rss_records)} articles from RSS")

    # Also scrape the leaderboard page — has structured data (amount, rank, protocol)
    leaderboard_url = "https://rekt.news/leaderboard/"
    print(f"  Leaderboard: {leaderboard_url}")
    soup = fetch_html(leaderboard_url)
    if soup:
        # Each row on leaderboard = one hack event
        rows = soup.find_all("tr") or soup.find_all("div", class_=re.compile(r"row|item|entry", re.I))
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            text = " ".join(c.get_text() for c in cells)
            a_tag = row.find("a", href=True)
            url_val = a_tag["href"] if a_tag else "N/A"
            if url_val.startswith("/"):
                url_val = "https://rekt.news" + url_val

            # Try to extract dollar amount from cell text
            amount_match = re.search(r"\$[\d,\.]+[MKB]?", text)
            amount = amount_match.group(0) if amount_match else "N/A"

            title = clean(cells[0].get_text()) if cells else "N/A"
            if len(title) > 5 and title.lower() != "protocol":
                records.append(make_record(
                    "hack_database", "Rekt.news", title, url_val,
                    amount_usd=amount,
                    extra={"rank": clean(cells[0].get_text()) if len(cells) > 0 else "N/A"}
                ))

    time.sleep(BASE_DELAY)
    return records


def scrape_defillama_hacks():
    """
    DeFi Llama Hacks API — structured JSON database of all DeFi hacks.
    Returns: protocol name, date, amount stolen, vulnerability category.
    This is the most structured data source — perfect for labelling.
    API: https://api.llama.fi/hacks
    """
    print("\n  [DeFi Llama Hacks API]")
    url  = "https://api.llama.fi/hacks"
    data = fetch_json(url)
    if not data:
        print("  → API unavailable")
        return []

    records = []
    hack_list = data if isinstance(data, list) else data.get("hacks", [])

    for hack in hack_list:
        # DeFiLlama gives us very clean structured fields
        name       = hack.get("name", "Unknown Protocol")
        date_raw   = hack.get("date", "N/A")
        amount     = hack.get("amount", 0)
        vuln_type  = hack.get("category", "unknown")
        chain      = hack.get("chain", "N/A")
        technique  = hack.get("technique", "N/A")
        link       = hack.get("links", ["N/A"])[0] if hack.get("links") else "N/A"

        # Format date from timestamp if needed
        if isinstance(date_raw, int):
            date_str = datetime.fromtimestamp(date_raw).strftime("%Y-%m-%d")
        else:
            date_str = str(date_raw)[:10]

        # Format amount
        amount_str = f"${amount:,.0f}" if isinstance(amount, (int, float)) and amount > 0 else "N/A"

        title   = f"{name} Hack — {amount_str} stolen"
        summary = f"Chain: {chain}. Technique: {technique}. Vulnerability: {vuln_type}."

        records.append(make_record(
            "hack_database", "DeFi Llama", title, link,
            date=date_str, summary=summary,
            amount_usd=amount_str, vuln_type=vuln_type.lower().replace(" ", "_"),
            extra={"protocol": name, "chain": chain, "technique": technique}
        ))

    print(f"  → {len(records)} hack records from DeFi Llama API")
    time.sleep(BASE_DELAY)
    return records


def scrape_defiyield_rekt():
    """
    DeFiYield REKT Database API — another structured hack database.
    Has additional fields like contract address, audit status.
    API: https://api.de.fi/v1/rekt
    """
    print("\n  [DeFiYield REKT API]")
    url  = "https://api.de.fi/v1/rekt?page=1&pageSize=100"
    data = fetch_json(url)
    if not data:
        print("  → API unavailable")
        return []

    records = []
    items = data if isinstance(data, list) else data.get("data", data.get("items", []))

    for item in items:
        name      = item.get("projectName", item.get("name", "Unknown"))
        date_raw  = item.get("date", item.get("createdAt", "N/A"))
        amount    = item.get("fundsLost", item.get("amount", 0))
        vuln      = item.get("category", item.get("type", "unknown"))
        chain     = item.get("chain", "N/A")
        desc      = item.get("description", item.get("shortDescription", "N/A"))
        link      = item.get("link", item.get("url", "N/A"))
        audited   = item.get("auditStatus", item.get("audited", "unknown"))
        contract  = item.get("contractAddress", "N/A")

        # Normalize date
        if isinstance(date_raw, int) and date_raw > 1000000000:
            date_str = datetime.fromtimestamp(date_raw).strftime("%Y-%m-%d")
        else:
            date_str = str(date_raw)[:10]

        amount_str = f"${amount:,.0f}" if isinstance(amount, (int, float)) and amount > 0 else "N/A"
        title      = f"{name} — {vuln} exploit ({amount_str})"

        records.append(make_record(
            "hack_database", "DeFiYield REKT", title, link,
            date=date_str, summary=clean(desc),
            amount_usd=amount_str,
            vuln_type=vuln.lower().replace(" ", "_") if vuln else "unknown",
            extra={
                "protocol": name, "chain": chain,
                "audited": str(audited), "contract_address": contract
            }
        ))

    print(f"  → {len(records)} records from DeFiYield REKT")
    time.sleep(BASE_DELAY)
    return records


def scrape_slowmist():
    """
    SlowMist is a blockchain security firm that maintains
    a public hacked list on GitHub as JSON.
    GitHub raw: blockchain-threat-intelligence / hacked-list
    """
    print("\n  [SlowMist Hacked List — GitHub]")
    url  = "https://raw.githubusercontent.com/slowmist/Blockchain-dark-forest-selfguard-handbook/main/README.md"
    # Also try their actual hacked list
    urls = [
        "https://raw.githubusercontent.com/slowmist/Knowledge-Base/master/en-us/hacked-list.json",
        "https://hacked.slowmist.io/",
    ]

    records = []
    for url in urls:
        if url.endswith(".json"):
            data = fetch_json(url)
            if not data:
                continue
            items = data if isinstance(data, list) else []
            for item in items:
                name    = item.get("project", item.get("name", "Unknown"))
                date    = item.get("date", "N/A")
                amount  = item.get("amount", "N/A")
                vuln    = item.get("type", item.get("vulnerability", "unknown"))
                link    = item.get("link", item.get("url", "N/A"))
                desc    = item.get("description", item.get("detail", "N/A"))
                records.append(make_record(
                    "hack_database", "SlowMist", f"{name} Hack",
                    link, date=str(date)[:10], summary=clean(str(desc)),
                    amount_usd=str(amount), vuln_type=str(vuln).lower()
                ))
        else:
            soup = fetch_html(url)
            if soup:
                rows = soup.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue
                    title_text = clean(cells[0].get_text())
                    a_tag = row.find("a", href=True)
                    link  = a_tag["href"] if a_tag else "N/A"
                    if len(title_text) > 5:
                        records.append(make_record(
                            "hack_database", "SlowMist", title_text, link
                        ))
        time.sleep(BASE_DELAY)

    print(f"  → {len(records)} records from SlowMist")
    return records


def scrape_hack_databases():
    print("\n" + "═" * 55)
    print("  CATEGORY 2 — HACK DATABASES")
    print("═" * 55)

    all_records = []
    for scraper_fn in [scrape_rekt_news, scrape_defillama_hacks,
                        scrape_defiyield_rekt, scrape_slowmist]:
        try:
            all_records += scraper_fn()
        except Exception as e:
            print(f"  [!] Scraper failed: {e} — continuing...")
        time.sleep(BASE_DELAY)

    save_category(all_records, "hack_database.csv")
    return all_records


# ═══════════════════════════════════════════════════════
#  CATEGORY 3 — SECURITY RESEARCH
#  Academic papers + security firm blogs give technical depth
#  about vulnerability patterns — crucial for GNN feature design
# ═══════════════════════════════════════════════════════

RESEARCH_RSS = [
    # ArXiv — academic papers on smart contract security
    ("ArXiv CS.CR",    "https://arxiv.org/rss/cs.CR"),     # Crypto & Security
    # Trail of Bits — top smart contract auditing firm blog
    ("Trail of Bits",  "https://blog.trailofbits.com/feed/"),
    # OpenZeppelin — security blog (they write the standard contracts)
    ("OpenZeppelin",   "https://blog.openzeppelin.com/feed.xml"),
    # Immunefi — bug bounty platform blog
    ("Immunefi",       "https://medium.com/feed/immunefi"),
    # Halborn — security firm specializing in blockchain
    ("Halborn",        "https://halborn.com/blog/feed/"),
    # PeckShield — security firm that often first reports hacks
    ("PeckShield",     "https://medium.com/feed/@peckshield"),
    # Certik — audit firm
    ("CertiK",         "https://certik.com/blog/feed"),
    # Google Scholar search via Semantic Scholar API
]


def scrape_arxiv_papers():
    """
    ArXiv has a search API — we can query for smart contract
    security papers specifically.
    API: http://export.arxiv.org/api/query
    Returns Atom XML with paper title, abstract, authors, date.
    """
    print("\n  [ArXiv API — Smart Contract Security Papers]")
    records = []
    queries = [
        "smart contract vulnerability detection",
        "smart contract security analysis",
        "solidity vulnerability GNN",
        "DeFi exploit analysis blockchain",
        "reentrancy smart contract formal verification",
    ]

    for query in queries:
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query=all:{query.replace(' ', '+')}"
            f"&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
        )
        soup = fetch_xml(url)
        if not soup:
            time.sleep(BASE_DELAY)
            continue

        entries = soup.find_all("entry")
        for entry in entries:
            title_tag   = entry.find("title")
            summary_tag = entry.find("summary")
            id_tag      = entry.find("id")
            date_tag    = entry.find("published")
            authors     = [a.find("name").get_text() for a in entry.find_all("author") if a.find("name")]

            title   = clean(title_tag.get_text())   if title_tag   else "N/A"
            summary = clean(summary_tag.get_text()) if summary_tag else "N/A"
            url_val = clean(id_tag.get_text())      if id_tag      else "N/A"
            date    = parse_date(date_tag.get_text() if date_tag else "")

            if is_relevant(title + " " + summary):
                records.append(make_record(
                    "research", "ArXiv", title, url_val, date, summary,
                    extra={"authors": ", ".join(authors[:3]), "query": query}
                ))

        print(f"  ArXiv query '{query[:40]}': {len(entries)} papers, {len(records)} relevant total")
        time.sleep(BASE_DELAY + random.uniform(0, 1))

    return records


def scrape_security_blogs():
    """Scrape security firm RSS blogs for technical vulnerability writeups."""
    print("\n  [Security Research Blogs]")
    all_records = []

    for source, url in RESEARCH_RSS:
        print(f"  [{source}] {url[-50:]}")
        try:
            records = parse_rss(source, url, category="research")
            all_records += records
            print(f"           → {len(records)} relevant articles")
        except Exception as e:
            print(f"           → ERROR: {e}")
        time.sleep(BASE_DELAY + random.uniform(0, 0.5))

    return all_records


def scrape_research():
    print("\n" + "═" * 55)
    print("  CATEGORY 3 — SECURITY RESEARCH")
    print("═" * 55)

    all_records = []
    try:
        all_records += scrape_arxiv_papers()
    except Exception as e:
        print(f"  [!] ArXiv failed: {e}")

    try:
        all_records += scrape_security_blogs()
    except Exception as e:
        print(f"  [!] Security blogs failed: {e}")

    save_category(all_records, "research_papers.csv")
    return all_records


# ═══════════════════════════════════════════════════════
#  CATEGORY 4 — COMMUNITY DISCUSSIONS
#  Reddit posts often contain early warnings, post-mortems,
#  and informal technical discussions not found in news
# ═══════════════════════════════════════════════════════

REDDIT_FEEDS = [
    ("Reddit r/ethdev",       "https://www.reddit.com/r/ethdev/search.rss?q=vulnerability+hack+exploit&sort=new"),
    ("Reddit r/ethereum",     "https://www.reddit.com/r/ethereum/search.rss?q=smart+contract+hack&sort=new"),
    ("Reddit r/defi",         "https://www.reddit.com/r/defi/search.rss?q=exploit+hack&sort=new"),
    ("Reddit r/solidity",     "https://www.reddit.com/r/solidity/search.rss?q=vulnerability+security&sort=new"),
    ("Reddit r/netsec",       "https://www.reddit.com/r/netsec/search.rss?q=smart+contract+vulnerability&sort=new"),
    ("Reddit r/crypto",       "https://www.reddit.com/r/CryptoCurrency/search.rss?q=smart+contract+hack+exploit&sort=new"),
]


def scrape_community():
    print("\n" + "═" * 55)
    print("  CATEGORY 4 — COMMUNITY DISCUSSIONS (Reddit)")
    print("═" * 55)

    all_records = []
    for source, url in REDDIT_FEEDS:
        print(f"  [{source}]")
        try:
            # Reddit RSS needs a slightly different User-Agent
            records = parse_rss(source, url, category="community")
            all_records += records
            print(f"  → {len(records)} relevant posts")
        except Exception as e:
            print(f"  → ERROR: {e}")
        time.sleep(BASE_DELAY + random.uniform(0.5, 1.5))  # Reddit is strict about rate limits

    save_category(all_records, "community_posts.csv")
    return all_records


# ═══════════════════════════════════════════════════════
#  CATEGORY 5 — VULNERABILITY REGISTRY
#  SWC = Smart Contract Weakness Classification Registry
#  Official taxonomy of all known vulnerability types
# ═══════════════════════════════════════════════════════

SWC_ENTRIES = [
    ("SWC-100", "Function Default Visibility",            "access_control",    "https://swcregistry.io/docs/SWC-100"),
    ("SWC-101", "Integer Overflow and Underflow",         "integer_overflow",  "https://swcregistry.io/docs/SWC-101"),
    ("SWC-102", "Outdated Compiler Version",              "configuration",     "https://swcregistry.io/docs/SWC-102"),
    ("SWC-103", "Floating Pragma",                        "configuration",     "https://swcregistry.io/docs/SWC-103"),
    ("SWC-104", "Unchecked Call Return Value",            "logic_error",       "https://swcregistry.io/docs/SWC-104"),
    ("SWC-105", "Unprotected Ether Withdrawal",           "access_control",    "https://swcregistry.io/docs/SWC-105"),
    ("SWC-106", "Unprotected SELFDESTRUCT Instruction",   "access_control",    "https://swcregistry.io/docs/SWC-106"),
    ("SWC-107", "Reentrancy",                             "reentrancy",        "https://swcregistry.io/docs/SWC-107"),
    ("SWC-108", "State Variable Default Visibility",      "access_control",    "https://swcregistry.io/docs/SWC-108"),
    ("SWC-109", "Uninitialized Storage Pointer",          "logic_error",       "https://swcregistry.io/docs/SWC-109"),
    ("SWC-110", "Assert Violation",                       "logic_error",       "https://swcregistry.io/docs/SWC-110"),
    ("SWC-111", "Use of Deprecated Solidity Functions",   "configuration",     "https://swcregistry.io/docs/SWC-111"),
    ("SWC-112", "Delegatecall to Untrusted Callee",       "access_control",    "https://swcregistry.io/docs/SWC-112"),
    ("SWC-113", "DoS with Failed Call",                   "denial_of_service", "https://swcregistry.io/docs/SWC-113"),
    ("SWC-114", "Transaction Order Dependence",           "front_running",     "https://swcregistry.io/docs/SWC-114"),
    ("SWC-115", "Authorization through tx.origin",        "access_control",    "https://swcregistry.io/docs/SWC-115"),
    ("SWC-116", "Block values as a proxy for time",       "logic_error",       "https://swcregistry.io/docs/SWC-116"),
    ("SWC-120", "Weak Sources of Randomness",             "logic_error",       "https://swcregistry.io/docs/SWC-120"),
    ("SWC-123", "Requirement Violation",                  "logic_error",       "https://swcregistry.io/docs/SWC-123"),
    ("SWC-124", "Write to Arbitrary Storage Location",    "access_control",    "https://swcregistry.io/docs/SWC-124"),
    ("SWC-125", "Incorrect Inheritance Order",            "logic_error",       "https://swcregistry.io/docs/SWC-125"),
    ("SWC-127", "Arbitrary Jump with Function Type Var",  "logic_error",       "https://swcregistry.io/docs/SWC-127"),
    ("SWC-128", "DoS With Block Gas Limit",               "denial_of_service", "https://swcregistry.io/docs/SWC-128"),
    ("SWC-129", "Typographical Error",                    "logic_error",       "https://swcregistry.io/docs/SWC-129"),
    ("SWC-131", "Presence of unused variables",           "configuration",     "https://swcregistry.io/docs/SWC-131"),
    ("SWC-132", "Unexpected Ether balance",               "logic_error",       "https://swcregistry.io/docs/SWC-132"),
    ("SWC-135", "Code With No Effects",                   "logic_error",       "https://swcregistry.io/docs/SWC-135"),
    ("SWC-136", "Unencrypted Private Data On-Chain",      "privacy",           "https://swcregistry.io/docs/SWC-136"),
]


def build_swc_registry():
    """
    Build records from the SWC (Smart Contract Weakness Classification) Registry.
    This is hardcoded because the registry is stable and authoritative.
    We also attempt to fetch descriptions from the live site.
    Provides the official vulnerability TAXONOMY for our GNN labels.
    """
    print("\n" + "═" * 55)
    print("  CATEGORY 5 — SWC VULNERABILITY REGISTRY")
    print("═" * 55)

    records = []
    for swc_id, name, vuln_type, url in SWC_ENTRIES:
        # Try fetching description from live site
        summary = f"Official SWC Registry entry for {name}. Vulnerability type: {vuln_type}."
        soup = fetch_html(url)
        if soup:
            desc_tag = soup.find("div", class_=re.compile(r"description|content|markdown", re.I))
            if desc_tag:
                p_tags = desc_tag.find_all("p")
                if p_tags:
                    summary = clean(p_tags[0].get_text())[:300]

        records.append(make_record(
            "vulnerability_registry", "SWC Registry",
            f"{swc_id}: {name}", url,
            date="2023-01-01",
            summary=summary,
            vuln_type=vuln_type,
            extra={"swc_id": swc_id, "standard": "SWC"}
        ))
        print(f"  {swc_id}: {name} → {vuln_type}")
        time.sleep(0.5)

    save_category(records, "swc_registry.csv")
    return records


# ═══════════════════════════════════════════════════════
#  FINAL MERGE & REPORT
# ═══════════════════════════════════════════════════════

def merge_and_report(all_records):
    print("\n" + "═" * 55)
    print("  MERGING ALL SOURCES → all_articles.csv")
    print("═" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(all_records)

    if df.empty:
        print("  [!] No data collected at all.")
        return df

    df.drop_duplicates(subset=["url"],   inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)

    # Sort by category then date
    if "date" in df.columns and "category" in df.columns:
        df.sort_values(by=["category", "date"], ascending=[True, False], inplace=True)

    df.reset_index(drop=True, inplace=True)
    merged_path = os.path.join(OUTPUT_DIR, "all_articles.csv")
    df.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"  [✓] Merged CSV → {merged_path}  ({len(df)} total records)")

    # ── Report ──────────────────────────────────────
    sep  = "=" * 65
    thin = "-" * 65
    lines = [
        sep,
        "  SMART CONTRACT VULNERABILITY DATASET REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        f"\n  TOTAL RECORDS    : {len(df)}",
        f"  UNIQUE SOURCES   : {df['source'].nunique()}",
        f"  CATEGORIES       : {df['category'].nunique()}",
        "\n" + thin,
        "  BREAKDOWN BY CATEGORY",
        thin,
    ]

    for cat, grp in df.groupby("category"):
        lines.append(f"\n  [{cat.upper()}]  {len(grp)} records")
        for _, row in grp.head(3).iterrows():
            t = row["title"][:60] + "..." if len(str(row["title"])) > 60 else row["title"]
            lines.append(f"    • {t}")

    lines += ["\n" + thin, "  VULNERABILITY TYPES DISTRIBUTION", thin]
    for vt, cnt in df["vuln_type"].value_counts().items():
        bar = "█" * min(int(cnt / max(df["vuln_type"].value_counts()) * 20), 20)
        lines.append(f"  {str(vt):<30}  {bar} {cnt}")

    lines += ["\n" + thin, "  TOP SOURCES BY VOLUME", thin]
    for src, cnt in df["source"].value_counts().head(10).items():
        lines.append(f"  {str(src):<30}  {cnt} records")

    lines += [
        "\n" + sep,
        "  DATASET QUALITY NOTES FOR GNN TRAINING",
        sep,
        f"""
  CATEGORY PURPOSES:
  ┌───────────────────────┬────────────────────────────────────┐
  │ news_articles.csv     │ Real exploit events → positive     │
  │                       │ training labels (confirmed hacks)  │
  ├───────────────────────┼────────────────────────────────────┤
  │ hack_database.csv     │ Structured data with amount, date, │
  │                       │ vuln type → best labelled data     │
  ├───────────────────────┼────────────────────────────────────┤
  │ research_papers.csv   │ Technical patterns → GNN feature   │
  │                       │ engineering guidance               │
  ├───────────────────────┼────────────────────────────────────┤
  │ community_posts.csv   │ Early warnings, informal reports,  │
  │                       │ post-mortems → supplementary data  │
  ├───────────────────────┼────────────────────────────────────┤
  │ swc_registry.csv      │ Official taxonomy → defines the    │
  │                       │ classification labels for GNN      │
  └───────────────────────┴────────────────────────────────────┘

  NEXT STEP — Phase 2:
    Extract Ethereum contract addresses from hack_database.csv
    and fetch verified Solidity source code from Etherscan API.
    These source files become the actual GNN training samples.
""",
        sep,
    ]

    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  [✓] Report saved → {report_path}")
    return df


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    start = datetime.now()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Smart Contract DIVERSE Dataset Scraper  v4.0")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print("""
  SOURCE PLAN:
    Category 1 — News RSS      (18 feeds, 8 sources)
    Category 2 — Hack DBs      (Rekt, DeFiLlama, DeFiYield, SlowMist)
    Category 3 — Research      (ArXiv API + 7 security blogs)
    Category 4 — Community     (6 Reddit RSS feeds)
    Category 5 — SWC Registry  (28 official vulnerability entries)
    """)

    all_records = []

    # ── Run all 5 categories ──────────────────────────
    # Each saves its own CSV immediately on completion.
    # Even if later categories fail, earlier ones are safe.

    categories = [
        ("News RSS",           scrape_news_rss),
        ("Hack Databases",     scrape_hack_databases),
        ("Security Research",  scrape_research),
        ("Community",          scrape_community),
        ("SWC Registry",       build_swc_registry),
    ]

    for cat_name, fn in categories:
        try:
            records = fn()
            all_records += records
            print(f"\n  ✓ {cat_name}: {len(records)} records collected")
        except Exception as e:
            print(f"\n  ✗ {cat_name} FAILED: {e} — continuing with next category...")
        time.sleep(BASE_DELAY)

    # ── Merge & report ───────────────────────────────
    df = merge_and_report(all_records)

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'=' * 55}")
    print(f"  COMPLETE in {elapsed}s")
    print(f"  {len(df)} total unique records collected")
    print(f"  Output folder: ./{OUTPUT_DIR}/")
    print(f"    all_articles.csv")
    print(f"    news_articles.csv")
    print(f"    hack_database.csv")
    print(f"    research_papers.csv")
    print(f"    community_posts.csv")
    print(f"    swc_registry.csv")
    print(f"    dataset_report.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()
