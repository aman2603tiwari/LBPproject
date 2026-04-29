"""
=============================================================
  Smart Contract Hack News Scraper  v3.0  (RSS-based)
  Project: Vulnerability Detection in Smart Contracts using GNN
=============================================================
  WHY v3?
    v2 used HTML scraping — these sites are React/JS apps,
    so requests gets empty HTML and 403s.

    v3 uses RSS FEEDS instead:
      ✓ Plain XML — no JavaScript needed
      ✓ Never returns 403 / Cloudflare blocks
      ✓ Structured data (title, date, summary built-in)
      ✓ Much faster — one request per feed

  Sources  : CoinDesk, CoinTelegraph, Decrypt.co,
             Bitcoin Magazine, BeInCrypto, Google News RSS
  Focus    : Smart contract hacks & exploits
  Outputs  : scraped_articles.csv  +  scrape_report.txt

  SETUP (run once):
      pip install requests beautifulsoup4 pandas lxml

  RUN:
      python scraper.py
=============================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
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
]

# Keywords to filter relevant articles
KEYWORDS = [
    "smart contract", "reentrancy", "flash loan", "defi hack",
    "solidity", "exploit", "vulnerability", "contract hack",
    "blockchain hack", "ethereum hack", "defi exploit",
    "rug pull", "protocol hack", "bridge hack", "oracle manipulation",
]

BASE_DELAY    = 1
MAX_RETRIES   = 3
RETRY_DELAYS  = [5, 10, 15]
OUTPUT_CSV    = "scraped_articles.csv"
OUTPUT_REPORT = "scrape_report.txt"

# ─────────────────────────────────────────────
#  RSS FEED SOURCES
# ─────────────────────────────────────────────
# Each entry: (source_name, rss_url)
RSS_FEEDS = [
    # CoinDesk — full RSS feed
    ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),

    # CoinTelegraph — full RSS + hacks tag feed
    ("CoinTelegraph",  "https://cointelegraph.com/rss"),
    ("CoinTelegraph",  "https://cointelegraph.com/tags/hacks/rss"),

    # Decrypt
    ("Decrypt.co",     "https://decrypt.co/feed"),

    # Bitcoin Magazine
    ("Bitcoin Mag",    "https://bitcoinmagazine.com/feed"),

    # BeInCrypto — good DeFi hack coverage
    ("BeInCrypto",     "https://beincrypto.com/feed/"),

    # Google News RSS — searches news across ALL outlets
    # This is the most powerful: aggregates NYT, Reuters, Bloomberg, etc.
    ("Google News",    "https://news.google.com/rss/search?q=smart+contract+hack&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=smart+contract+exploit&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=DeFi+hack+smart+contract&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=reentrancy+attack+ethereum&hl=en-US&gl=US&ceid=US:en"),
    ("Google News",    "https://news.google.com/rss/search?q=flash+loan+attack+blockchain&hl=en-US&gl=US&ceid=US:en"),
]


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def get_headers():
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection":      "keep-alive",
    }


def fetch(url, timeout=15):
    """Fetch URL with retry + rate-limit handling. Returns BeautifulSoup or None."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=get_headers(), timeout=timeout)

            if resp.status_code == 429:
                wait = RETRY_DELAYS[attempt] * 3
                print(f"    [429] Rate limited — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                print(f"    [403] Blocked: {url}")
                return None

            if resp.status_code == 404:
                print(f"    [404] Feed not found: {url}")
                return None

            resp.raise_for_status()
            # Use lxml-xml parser for RSS/Atom feeds
            return BeautifulSoup(resp.content, "lxml-xml")

        except requests.exceptions.Timeout:
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS)-1)]
            print(f"    [Timeout] Attempt {attempt+1}/{MAX_RETRIES} — retrying in {wait}s...")
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS)-1)]
            print(f"    [Connection Error] Attempt {attempt+1}/{MAX_RETRIES} — retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"    [ERROR] {url} → {e}")
            return None

    print(f"    [FAILED] Gave up on {url} after {MAX_RETRIES} attempts.")
    return None


def is_relevant(text):
    """Return True if any keyword appears in the text."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in KEYWORDS)


def get_keyword(text):
    """Return the first matching keyword found in text."""
    text_lower = text.lower()
    return next((k for k in KEYWORDS if k.lower() in text_lower), "general")


def clean(text):
    """Strip tags and extra whitespace from text."""
    if not text:
        return "N/A"
    # Remove HTML tags if any leaked through
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(raw_date):
    """Try to parse and standardise an RSS date string."""
    if not raw_date:
        return "N/A"
    raw_date = clean(raw_date)
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw_date, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Return raw if all parsers fail
    return raw_date[:20]


def save_progress(new_articles, source_name):
    """Append new_articles to CSV immediately. Safe against mid-run crashes."""
    if not new_articles:
        print(f"    [!] No relevant articles found for {source_name}")
        return 0

    df_new = pd.DataFrame(new_articles)

    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["url"],   inplace=True)
        df_combined.drop_duplicates(subset=["title"], inplace=True)
        df_combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        saved = len(df_combined)
    else:
        df_new.drop_duplicates(subset=["url"],   inplace=True)
        df_new.drop_duplicates(subset=["title"], inplace=True)
        df_new.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        saved = len(df_new)

    print(f"    [✓] Saved — CSV now has {saved} total articles")
    return saved


# ─────────────────────────────────────────────
#  RSS PARSER  (core engine)
# ─────────────────────────────────────────────

def parse_rss_feed(source_name, rss_url):
    """
    Parse a single RSS feed URL.
    Handles both RSS 2.0 (<item>) and Atom (<entry>) formats.
    Returns list of article dicts.
    """
    soup = fetch(rss_url)
    if not soup:
        return []

    articles = []

    # RSS 2.0 uses <item>, Atom uses <entry>
    items = soup.find_all("item") or soup.find_all("entry")

    for item in items:
        # ── Title ──────────────────────────────────
        title_tag = item.find("title")
        title = clean(title_tag.get_text()) if title_tag else ""
        if not title or len(title) < 10:
            continue

        # ── URL ────────────────────────────────────
        link_tag = item.find("link")
        if link_tag:
            # Atom feeds store URL as text content OR in href attribute
            url = link_tag.get("href") or clean(link_tag.get_text())
        else:
            url = ""
        if not url or not url.startswith("http"):
            guid = item.find("guid")
            url  = clean(guid.get_text()) if guid else "N/A"

        # ── Date ───────────────────────────────────
        date_tag = (
            item.find("pubDate") or
            item.find("published") or
            item.find("updated") or
            item.find("dc:date")
        )
        date = parse_date(date_tag.get_text() if date_tag else "")

        # ── Summary ────────────────────────────────
        desc_tag = (
            item.find("description") or
            item.find("summary") or
            item.find("content")
        )
        summary = clean(desc_tag.get_text()) if desc_tag else "Visit URL for full article"
        summary = summary[:300] + "..." if len(summary) > 300 else summary

        # ── Relevance filter ───────────────────────
        combined_text = title + " " + summary
        if not is_relevant(combined_text):
            continue

        articles.append({
            "source":          source_name,
            "title":           title,
            "url":             url,
            "date":            date,
            "summary":         summary,
            "keyword_matched": get_keyword(combined_text),
            "scraped_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return articles


# ─────────────────────────────────────────────
#  MAIN SCRAPE LOOP
# ─────────────────────────────────────────────

def run_all_feeds():
    """Iterate over all RSS feeds, parse, filter, and save progressively."""

    # Group feeds by source for clean progress display
    source_feeds = {}
    for source, url in RSS_FEEDS:
        source_feeds.setdefault(source, []).append(url)

    all_articles = []
    total_sources = len(source_feeds)

    for idx, (source, urls) in enumerate(source_feeds.items(), 1):
        print(f"\n{'─' * 55}")
        print(f"  [{idx}/{total_sources}] {source}  ({len(urls)} feed(s))")
        print(f"{'─' * 55}")

        source_articles = []

        for url in urls:
            print(f"    Fetching: {url[-60:]}")
            try:
                batch = parse_rss_feed(source, url)
                source_articles += batch
                print(f"    → {len(batch)} relevant articles from this feed")
            except Exception as e:
                print(f"    [!] Feed failed: {e} — skipping")

            time.sleep(BASE_DELAY + random.uniform(0, 0.5))

        # Deduplicate within this source before saving
        seen_titles = set()
        unique = []
        for a in source_articles:
            if a["title"] not in seen_titles:
                seen_titles.add(a["title"])
                unique.append(a)

        save_progress(unique, source)
        all_articles += unique

    return all_articles


# ─────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────

def generate_report(df):
    sep  = "=" * 65
    thin = "-" * 65
    lines = [
        sep,
        "  SMART CONTRACT HACK NEWS — SCRAPE REPORT  (v3 RSS)",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        f"\n  Total Articles : {len(df)}",
        f"  Unique Sources : {df['source'].nunique()}  ({', '.join(df['source'].unique())})",
        f"  Date Range     : {df['date'].min()}  →  {df['date'].max()}",
        "\n" + thin,
        "  ARTICLES BY SOURCE",
        thin,
    ]

    for source, group in df.groupby("source"):
        lines.append(f"\n  {source}  ({len(group)} articles)")
        for _, row in group.head(5).iterrows():
            t = row["title"][:65] + "..." if len(row["title"]) > 65 else row["title"]
            lines += [
                f"    • {t}",
                f"      {row['url']}",
                f"      Date: {row['date']}  |  Keyword: {row['keyword_matched']}",
            ]
        if len(group) > 5:
            lines.append(f"      ... and {len(group)-5} more (see CSV)")

    lines += [
        "\n" + thin,
        "  KEYWORDS FREQUENCY",
        thin,
    ]
    for kw, cnt in df["keyword_matched"].value_counts().items():
        bar = "█" * min(cnt, 30)
        lines.append(f"  {kw:<30}  {bar} {cnt}")

    lines += [
        "\n" + sep,
        "  PROJECT PIPELINE — WHERE THIS DATA FITS",
        sep,
        """
  PHASE 1  ✅ DONE — Threat Intelligence Dataset
    → scraped_articles.csv  (this file)
    → identifies real exploits + vulnerability types

  PHASE 2  NEXT — Extract Contract Addresses
    → scan article text for '0x...' Ethereum addresses
    → run: python extract_addresses.py

  PHASE 3  — Fetch Solidity Source Code
    → use Etherscan API with extracted addresses
    → run: python etherscan_fetch.py

  PHASE 4  — Build Graphs (CFG / AST / PDG)
    → parse Solidity → nodes & edges
    → run: python build_graphs.py

  PHASE 5  — Train GNN Model
    → label graphs: Vulnerable / Safe
    → train GCN or GAT classifier
    → run: python train_gnn.py

  VULNERABILITY TYPES TARGETED:
    • Reentrancy          (e.g. DAO Hack 2016 — $60M)
    • Flash Loan Attack   (e.g. bZx, Euler Finance)
    • Integer Overflow    (arithmetic bugs in tokens)
    • Access Control      (missing onlyOwner checks)
    • Oracle Manipulation (price feed attacks)
""",
        sep,
    ]

    text = "\n".join(lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    start = datetime.now()

    print("=" * 55)
    print("  Smart Contract Hack Scraper  v3.0  (RSS Mode)")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Feeds   : {len(RSS_FEEDS)} RSS feeds across {len(set(s for s,_ in RSS_FEEDS))} sources")
    print("=" * 55)

    # Clear old CSV for a fresh run
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"  [i] Cleared old {OUTPUT_CSV} — starting fresh")

    # Run all feeds
    all_articles = run_all_feeds()

    # Final clean read from CSV (all sources combined + deduped)
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
    else:
        if not all_articles:
            print("\n  [!] No articles collected at all.")
            print("  Check your internet connection and try again.")
            return
        df = pd.DataFrame(all_articles)

    # Final sort (safe — checks columns exist first)
    sort_cols = [c for c in ["source", "date"] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, ascending=[True, False], inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    elapsed = (datetime.now() - start).seconds

    print(f"\n{'=' * 55}")
    print(f"  [✓] Done in {elapsed}s")
    print(f"  [✓] {len(df)} unique articles → {OUTPUT_CSV}")

    if len(df) == 0:
        print("\n  [!] 0 articles collected.")
        print("  All feeds may be down or returning no results.")
        print("  Try opening one of the RSS URLs in your browser:")
        for _, url in RSS_FEEDS[:3]:
            print(f"    {url}")
        return

    # Print and save report
    report = generate_report(df)
    print(report)
    print(f"\n  [✓] Report saved → {OUTPUT_REPORT}")
    print(f"  Open '{OUTPUT_CSV}' in Excel to review all articles.")
    print("=" * 55)


if __name__ == "__main__":
    main()