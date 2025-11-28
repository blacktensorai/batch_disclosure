"""
sec_ingest.py — SEC Filing Ingestion Layer
----------------------------------------------------
- Fetching ticker → CIK mappings
- Market cap filtering
- Fetching 10-Q filings
- Downloading filings
- Producing metadata objects for extractors
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import yfinance as yf

from pathlib import Path
from datetime import datetime, timedelta

# Paths
DATA_DIR = Path("data/sec/")
LOG_DIR = Path("logs/")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "sec_ingest.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

HEADERS = {
    "User-Agent": "CatalystScan (contact: your_email@example.com)"
}

MARKET_CAP_MIN = 100_000_000
MARKET_CAP_MAX = 800_000_000
TARGET_FORMS = ["10-Q"]


# Utility: Resolve date range
def get_date_range(filter_option="6m"):
    today = datetime.today().date()

    if filter_option == "today":
        return today, today
    if filter_option == "6m":
        return today - timedelta(days=180), today
    if filter_option == "1y":
        return today - timedelta(days=365), today

    if isinstance(filter_option, tuple):
        return filter_option

    return today - timedelta(days=180), today


# Step 1 — Fetch SEC Ticker → CIK
def fetch_cik_mapping():
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    raw = resp.json()

    # Convert dict → list
    header = list(raw.values())[0]
    data = list(raw.values())[1]

    all_records = [header]
    all_records.extend(data)

    df = pd.DataFrame.from_records(all_records)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    df.to_csv("data/sec/cik_map.csv", index=False)
    return df


# Step 2 — Market Cap Filtering
def apply_market_cap_filter(df):
    caps = []
    for ticker in df["ticker"]:
        try:
            info = yf.Ticker(ticker).info
            caps.append(info.get("marketCap", None))
        except Exception:
            caps.append(None)
        time.sleep(0.2)

    df["market_cap"] = caps

    filtered = df[
        (df["market_cap"].fillna(0) >= MARKET_CAP_MIN) &
        (df["market_cap"].fillna(0) <= MARKET_CAP_MAX)
    ]
    filtered.to_csv("data/sec/sec_filtered_tickers.csv", index=False)
    return filtered


# Step 3 — Fetch filings inside date range
def get_recent_filings(cik, start_date, end_date):
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    data = resp.json()
    rec = data["filings"]["recent"]

    results = []

    for form, accession, rep_date, doc in zip(
        rec["form"],
        rec["accessionNumber"],
        rec["filingDate"],
        rec["primaryDocument"]
    ):

        # Filter out everything except 10-Q
        if form != "10-Q":
            continue

        try:
            f_date = datetime.strptime(rep_date, "%Y-%m-%d").date()
        except:
            continue

        if not (start_date <= f_date <= end_date):
            continue

        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"

        results.append({
            "cik": cik,
            "form": form,
            "report_date": rep_date,
            "url": url
        })

        break

    return results


# Step 4 — Download filings
def download_filings(filings):
    downloaded = []

    for f in filings:
        fname = f"{f['cik']}_{f['form']}_{f['report_date']}.html"
        path = DATA_DIR / fname

        if not path.exists():
            try:
                resp = requests.get(f["url"], headers=HEADERS, timeout=15)
                resp.raise_for_status()
                with open(path, "wb") as fp:
                    fp.write(resp.content)
                time.sleep(0.3)
            except Exception:
                continue

        downloaded.append({
            "local_path": str(path),
            "exchange": "SEC",
            "filing_type": f["form"],
            "cik": f["cik"],
            "report_date": f["report_date"]
        })

    return downloaded


# High-Level Ingestion API
def ingest_sec_filings(date_filter="6m"):
    """
    Returns list of documents ready to pass to extractors.
    """

    start_date, end_date = get_date_range(date_filter)

    df = fetch_cik_mapping()
    df = apply_market_cap_filter(df)

    all_filings = []

    for _, row in df.iterrows():
        cik = row["cik"]
        filings = get_recent_filings(cik, start_date, end_date)
        all_filings.extend(filings)

    downloaded = download_filings(all_filings)

    return downloaded


if __name__ == "__main__":
    docs = ingest_sec_filings("6m")
    print(f"Ready for extraction → {len(docs)} filings")
