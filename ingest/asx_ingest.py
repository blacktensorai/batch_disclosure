"""
ASX Ingestion Layer for CatalystScan
------------------------------------
- Pure ingestion (no NLP, no extraction)
- Standard output format for extractors
- Detects announcement type (Quarterly / Annual / Presentation)
- Downloads PDFs reliably from ASX agreement pages
"""

import os
import re
import time
import logging
import sqlite3
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# Configuration
ASX_ENDPOINT = "https://www.asx.com.au/asx/v2/statistics/announcements.do"
ASX_BASE_URL = "https://www.asx.com.au"

DATA_DIR = Path("data/asx/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("logs/")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("asx_ingest")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_DIR / "asx_ingest.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
USER_AGENT = "CatalystScan ASX Fetcher — contact@example.com"


# Helpers
def classify_announcement(title: str) -> str:
    t = title.lower()

    if "quarter" in t:
        return "QUARTERLY"
    if "investor presentation" in t or "presentation" in t:
        return "PRESENTATION"
    if "annual" in t and "report" in t:
        return "ANNUAL"

    return "OTHER"


def safe_request(session, url, params=None, retries=3):
    for _ in range(retries):
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp
        except:
            time.sleep(1.5)
    return None


def extract_pdf_url(html: str):
    """
    Parse ASX 'Agree and Proceed' page to get the real PDF.
    """
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("input", {"name": "pdfURL"})

    if tag and tag.get("value", "").endswith(".pdf"):
        return tag["value"]

    m = re.search(r"https://announcements\.asx\.com\.au/asxpdf/.+?\.pdf", html)
    return m.group(0) if m else None


def parse_date(d: str):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date()
        except:
            continue
    return datetime.today().date()


# Scrape announcements for a single ASX code
def fetch_announcements_for_code(asx_code: str, period="M", year="2024", session=None):
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    params = {"by": "asxCode", "asxCode": asx_code, "timeframe": period, "year": year}
    resp = safe_request(session, ASX_ENDPOINT, params=params)
    if not resp:
        return []

    results = []
    soup = BeautifulSoup(resp.text, "html.parser")

    for tr in soup.select("tbody > tr"):
        tds = [td.get_text(strip=True) for td in tr.select("td")]
        if len(tds) < 3:
            continue

        date_txt, code, title, *_ = tds
        link = tr.find("a", href=True)
        if not link:
            continue

        pdf_page_url = ASX_BASE_URL + link["href"]

        filing_type = classify_announcement(title)
        if filing_type == "OTHER":
            continue

        results.append({
            "date": parse_date(date_txt).strftime("%Y-%m-%d"),
            "asx_code": asx_code,
            "title": title,
            "filing_type": filing_type,
            "pdf_page_url": pdf_page_url
        })

    return results


# Download PDF for each announcement
def download_pdfs(announcements, session=None):
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    output = []

    for ann in announcements:
        try:
            logger.info(f"Downloading PDF for {ann['asx_code']} → {ann['title']}")

            # Load agreement page
            resp = safe_request(session, ann["pdf_page_url"])
            if not resp:
                continue

            # Is it already a PDF?
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type.lower():
                pdf_url = resp.url
            else:
                pdf_url = extract_pdf_url(resp.text)
            if not pdf_url:
                logger.warning(f"Could not extract pdfURL: {ann['pdf_page_url']}")
                continue

            # Download the actual PDF
            pdf_resp = safe_request(session, pdf_url)
            if not pdf_resp:
                continue

            fname = f"{ann['asx_code']}_{ann['date']}_{ann['filing_type']}.pdf"
            fpath = DATA_DIR / fname

            with open(fpath, "wb") as f:
                f.write(pdf_resp.content)

            output.append({
                "local_path": str(fpath),
                "final_url": pdf_url,
                "exchange": "ASX",
                "filing_type": ann["filing_type"],
                "asx_code": ann["asx_code"],
                "title": ann["title"],
                "date": ann["date"],
            })

            time.sleep(1)

        except Exception as e:
            logger.error(f"Error downloading PDF for {ann}: {e}")

    return output


# Iterate over tickers.db
def ingest_asx_from_db(db_path="tickers.db", limit=None, period="M", year="2024"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT asx_code FROM companies WHERE exchange='ASX';", conn)
    conn.close()

    if limit:
        df = df.head(limit)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_anns = []

    for idx, row in df.iterrows():
        code = row["asx_code"]
        if not isinstance(code, str):
            continue

        anns = fetch_announcements_for_code(code, period=period, year=year, session=session)
        all_anns.extend(anns)

    downloaded = download_pdfs(all_anns, session=session)
    return downloaded


# CLI
if __name__ == "__main__":
    results = ingest_asx_from_db(limit=10, period="M")
    print(f"ASX ingestion complete → {len(results)} files")
    for r in results:
        print(r)
